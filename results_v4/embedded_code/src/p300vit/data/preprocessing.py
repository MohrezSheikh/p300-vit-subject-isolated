"""Shared EEG preprocessing -- the single source of truth.

Pipeline (identical for every model):
  1. optional resample to ``sfreq_target`` (ASZED files arrive at 256/200/100 Hz)
  2. 50 Hz notch, 0.5-30 Hz zero-phase FIR bandpass
  3. Common Average Reference
  4. Epoch -200..800 ms around target stimuli, baseline-corrected (-200..0 ms)
  5. Artifact rejection, two modes:
       * ``absolute`` (legacy, used for the local .cnt dataset): drop epochs
         with any-channel peak-to-peak > ``reject_uv`` uV -- identical to the
         original protocol.
       * ``robust`` (used for ASZED, whose amplitudes are an order of
         magnitude larger): drop epochs whose max-channel ptp exceeds
         ``robust_k`` x the median ptp of the same recording -- scale-free,
         so one threshold works across devices/units.
  6. Truncate to exactly ``n_times`` samples (200 @ 200 Hz)
  7. Optional per-trial z-score (uses only the trial itself -> leak-free)
  8. Flat-channel guard: channels with std < ``flat_uv`` are reported in the
     result so dataset adapters can drop or skip affected subjects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import mne
import numpy as np

from ..config import PreprocessConfig

mne.set_log_level("WARNING")


@dataclass
class SubjectEpochs:
    """All valid epochs for one subject, in canonical channel order."""
    subject_id: str
    label: int                       # 0 = control, 1 = schizophrenia
    site: str                        # acquisition site / dataset key
    data: np.ndarray                 # [n_trials, n_channels, n_times] float32
    ch_names: List[str]
    times: np.ndarray                # [n_times] in seconds
    sfreq: float
    n_rejected: int = 0
    flat_channels: List[str] = field(default_factory=list)
    files_used: int = 0


def _align_length(data: np.ndarray, times: np.ndarray, n_times: int):
    if data.shape[-1] >= n_times:
        return data[..., :n_times], times[:n_times]
    pad = n_times - data.shape[-1]
    return (
        np.pad(data, ((0, 0), (0, 0), (0, pad))),
        np.pad(times, (0, pad)),
    )


def preprocess_raw(
    raw: mne.io.BaseRaw,
    cfg: PreprocessConfig,
    picks: Sequence[str],
    event_id: int | str,
    event_desc: Optional[dict] = None,
) -> SubjectEpochs:
    """Filter, re-reference, epoch, and reject artifacts for one recording.

    ``raw`` must be freshly read (filtering/re-referencing modify it).
    Events are resolved AFTER resampling: either from EDF annotations
    (``event_desc`` maps annotation descriptions to codes, e.g.
    ``{'3000Hz': 1, '1000Hz': 0}``) or from stim-channel triggers.
    ``event_id`` selects which code to epoch.
    Returns epochs in the order of ``picks`` restricted to surviving
    (non-flat) channels.
    """
    raw = raw.pick(list(picks))

    # flat-channel guard (dead electrodes corrupt CAR and produce junk epochs)
    stds = raw.get_data().std(axis=1)
    flat = [picks[i] for i in np.where(stds < cfg.flat_uv * 1e-6)[0]]
    if flat:
        raw.drop_channels(flat)

    if not np.isclose(raw.info["sfreq"], cfg.sfreq_target):
        raw.resample(cfg.sfreq_target, npad="auto", verbose=False)

    raw.notch_filter(cfg.notch_freqs, fir_design="firwin")
    raw.filter(cfg.l_freq, cfg.h_freq, fir_design="firwin")
    raw.set_eeg_reference("average", projection=False)

    if event_desc is not None:
        events, _ = mne.events_from_annotations(raw, event_id=event_desc, verbose=False)
    else:
        events = mne.find_events(raw, shortest_event=1, verbose=False)
    if len(events) == 0:
        raise ValueError("No event triggers found in recording.")

    if cfg.artifact_rejection == "absolute":
        reject = dict(eeg=cfg.reject_uv * 1e-6)
    else:
        reject = None
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=cfg.tmin, tmax=cfg.tmax, baseline=cfg.baseline,
        reject=reject, preload=True, verbose=False,
    )
    n_events = len(epochs)

    if cfg.artifact_rejection == "robust" and len(epochs) > 1:
        data_all = epochs.get_data(copy=True)                    # [N, C, T]
        ptp = np.ptp(data_all, axis=-1).max(axis=1)              # [N]
        median_ptp = np.median(ptp)
        keep = ptp <= cfg.robust_k * median_ptp
        epochs = epochs[keep]
    data = epochs.get_data(copy=True)
    times = epochs.times.copy()
    n_rejected = n_events - len(epochs)

    data, times = _align_length(data, times, cfg.n_times)

    if cfg.trial_zscore:
        mu = data.mean(axis=-1, keepdims=True)
        sd = data.std(axis=-1, keepdims=True) + 1e-8
        data = (data - mu) / sd

    return SubjectEpochs(
        subject_id="",
        label=-1,
        site="",
        data=data.astype(np.float32),
        ch_names=[ch for ch in picks if ch in epochs.ch_names],
        times=times.astype(np.float32),
        sfreq=float(cfg.sfreq_target),
        n_rejected=int(n_rejected),
        flat_channels=flat,
    )
