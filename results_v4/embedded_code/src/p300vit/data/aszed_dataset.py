"""Adapter for the ASZED-153 public dataset (Zenodo 14178398, CC-BY-4.0).

Layout (Kaggle: /kaggle/input/datasets/mmdation/aszedeeg):
    ASZED/version_1.1/node_<N>/<subset>_<k>/subject_<id>/<session>/Phase X.edf
    ASZED_SpreadSheet.csv          -> subject labels (Patient/Control), age, node
    keymaps.kmp                    -> protocol definitions (oddball phase number)
    nodes.txt                      -> node -> site name

Verified facts (see INSPECTION_REPORT.md):
* Auditory oddball: standard = 1000 Hz tone, deviant/target = 3000 Hz tone,
  present as EDF annotations -> stimulus-locked, xDAWN applies.
* Oddball phase file: 'Phase 4.edf' in protocol 1.1 (subset_1),
  'Phase 5.edf' in protocol 1.2 (subsets 2/3).
* Mixed native sampling rates (256 / 200 / 100 Hz) -> resampled to 200 Hz.
* Amplitudes are ~10x typical scalp EEG -> scale-free ("robust") artifact
  rejection instead of the legacy absolute 120 uV rule.
* 19 EEG channels shared with the local dataset (FPZ is not recorded);
  a handful of recordings have a dead CZ -> subjects whose healthy files do
  not cover the full common montage are skipped and logged.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

from ..config import DatasetConfig, PreprocessConfig
from .cnt_dataset import SCALP_CHANNELS
from .preprocessing import SubjectEpochs, preprocess_raw

# EEG channels present in ASZED and in the local dataset (FPZ excluded).
COMMON_CHANNELS = [ch for ch in SCALP_CHANNELS if ch != "FPZ"]

_MIN_ODDBALL_TRIALS = 4   # subjects with fewer surviving deviant epochs are skipped


def _find_spreadsheet(root: Path) -> Path:
    hits = list(root.rglob("ASZED_SpreadSheet.csv"))
    if not hits:
        raise FileNotFoundError("ASZED_SpreadSheet.csv not found under " + str(root))
    return hits[0]


def _parse_subset_protocols(root: Path) -> Dict[str, str]:
    """Parse subsets.txt -> {'subset_1': '1.1', ...} (fallback: known layout)."""
    txt_files = list(root.rglob("subsets.txt"))
    mapping = {}
    if txt_files:
        current = None
        for line in txt_files[0].read_text().splitlines():
            s = line.strip()
            m = re.match(r"subset_(\d+):\s*$", s)
            if m:
                current = f"subset_{m.group(1)}"
                continue
            m = re.match(r"protocol:\s*([\d.]+)", s)
            if m and current:
                mapping[current] = m.group(1)
    # verified published layout as fallback
    mapping.setdefault("subset_1", "1.1")
    mapping.setdefault("subset_2", "1.2")
    mapping.setdefault("subset_3", "1.2")
    return mapping


# protocol -> oddball phase file (keymap.kmp PROTOCOL definitions)
_PROTOCOL_ODDBALL_PHASE = {"1.1": "Phase 4.edf", "1.2": "Phase 5.edf"}


def _clean_channel_name(name: str) -> str:
    """'Fp1[1]' -> 'FP1'; 'EEG Fp1-LE' -> 'FP1'; 'Fz[17]' -> 'FZ'."""
    name = re.sub(r"\[\d+\]", "", name)          # drop [n] indices
    name = re.sub(r"^EEG\s+", "", name)          # drop 'EEG ' prefix
    name = re.sub(r"-\w+$", "", name)            # drop '-LE' / '-A2' reference suffix
    return name.upper()


def _load_labels(root: Path) -> Dict[str, Dict]:
    """sn -> {label, site, age, gender}."""
    labels = {}
    with open(_find_spreadsheet(root), newline="") as f:
        for row in csv.DictReader(f):
            labels[row["sn"]] = {
                "label": 1 if row["category"].strip().lower() == "patient" else 0,
                "site": row.get("node", "ASZED").strip(),
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
            }
    return labels


def _oddball_files(subject_dir: Path, protocols: Dict[str, str]) -> List[Path]:
    """All oddball recordings of one subject, per its subset's protocol.

    Protocol 1.1 (subset_1) puts the oddball in Phase 4; protocol 1.2
    (subsets 2/3) in Phase 5. Sessions are distinct recordings (verified:
    not byte-identical), so all are used.
    """
    subset = subject_dir.parent.name
    phase = _PROTOCOL_ODDBALL_PHASE.get(protocols.get(subset, ""), None)
    if phase is None:
        return []
    return sorted(subject_dir.glob(f"*/{phase}"))


def _prep_oddball_file(
    path: Path, prep_cfg: PreprocessConfig
) -> Tuple[np.ndarray, List[str]]:
    """One EDF -> (epochs [n, C, T] float32, surviving channel names)."""
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    raw.rename_channels(_clean_channel_name)
    keep = [ch for ch in COMMON_CHANNELS if ch in raw.ch_names]
    if len(keep) != len(COMMON_CHANNELS):
        raise ValueError(f"missing channels {sorted(set(COMMON_CHANNELS) - set(keep))}")

    subj = preprocess_raw(
        raw, prep_cfg, picks=keep, event_id=1,
        event_desc={"3000Hz": 1, "1000Hz": 0},
    )
    return subj.data, subj.ch_names


def load_dataset(
    dataset_cfg: DatasetConfig,
    prep_cfg: PreprocessConfig,
    sfreq_target: float = 200.0,
    min_trials: int = _MIN_ODDBALL_TRIALS,
    verbose: bool = True,
    cache_dir: Optional[str] = None,
) -> List[SubjectEpochs]:
    """Load every ASZED subject's oddball epochs (19 common channels, 200 Hz).

    ``cache_dir``: if given, preprocessed per-subject epochs are cached there
    keyed by a hash of the preprocessing config (preprocessing is by far the
    slowest stage; the Kaggle runner always caches).
    """
    prep_cfg.artifact_rejection = "robust"
    prep_cfg.sfreq_target = sfreq_target
    root = dataset_cfg.resolve()
    labels = _load_labels(root)
    protocols = _parse_subset_protocols(root)

    cache_key = None
    if cache_dir is not None:
        import hashlib, json
        payload = json.dumps(
            {**prep_cfg.__dict__,
             "sfreq_target": sfreq_target, "channels": COMMON_CHANNELS},
            sort_keys=True, default=str,
        )
        cache_key = Path(cache_dir) / ("aszed_" + hashlib.md5(payload.encode()).hexdigest()[:12])
        cache_key.mkdir(parents=True, exist_ok=True)

    # group oddball files by subject id across subsets
    subject_files: Dict[str, List[Path]] = {}
    for subj_dir in sorted(root.rglob("subject_*")):
        sid = subj_dir.name
        if sid in labels:
            subject_files.setdefault(sid, []).extend(_oddball_files(subj_dir, protocols))

    subjects: List[SubjectEpochs] = []
    for sid, files in sorted(subject_files.items()):
        meta = labels[sid]

        # cache fast path
        cache_file = cache_key / f"{sid}.npz" if cache_key is not None else None
        if cache_file is not None and cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as z:
                subjects.append(SubjectEpochs(
                    subject_id=sid, label=int(z["label"]), site=str(z["site"]),
                    data=z["data"], ch_names=list(COMMON_CHANNELS),
                    times=np.linspace(prep_cfg.tmin, prep_cfg.tmax, prep_cfg.n_times, dtype=np.float32),
                    sfreq=sfreq_target, files_used=int(z["files_used"]),
                ))
            continue

        chunks, ch_union = [], None
        n_used = 0
        for f in files:
            try:
                data, chs = _prep_oddball_file(f, prep_cfg)
            except Exception as e:      # unreadable/corrupt file: skip, but say so
                if verbose:
                    print(f"[ASZED] {sid}: skipping {f.parent.name}/{f.name}: {e}")
                continue
            if len(data) == 0:
                continue
            chunks.append(data)
            alive = set(chs)
            ch_union = alive if ch_union is None else (ch_union & alive)
            n_used += 1
        if not chunks:
            continue
        if ch_union != set(COMMON_CHANNELS):
            dead = sorted(set(COMMON_CHANNELS) - (ch_union or set()))
            if verbose:
                print(f"[ASZED] {sid}: dead channels {dead} across all files — subject skipped")
            continue
        data = np.concatenate(chunks, axis=0)
        if len(data) < min_trials:
            if verbose:
                print(f"[ASZED] {sid}: only {len(data)} valid deviant epochs (< {min_trials}) — skipped")
            continue
        subj = SubjectEpochs(
            subject_id=sid,
            label=meta["label"],
            site=meta["site"],
            data=data,
            ch_names=list(COMMON_CHANNELS),
            times=np.linspace(prep_cfg.tmin, prep_cfg.tmax, prep_cfg.n_times, dtype=np.float32),
            sfreq=sfreq_target,
            files_used=n_used,
        )
        subjects.append(subj)
        if cache_file is not None:
            np.savez_compressed(
                cache_file, data=subj.data.astype(np.float32),
                label=subj.label, site=subj.site, files_used=subj.files_used,
            )
    if verbose:
        n_pos = sum(s.label for s in subjects)
        counts = [len(s.data) for s in subjects]
        print(f"[ASZED] loaded {len(subjects)} subjects ({n_pos} patients, "
              f"{len(subjects)-n_pos} controls); trials/subject: "
              f"min={min(counts)} median={int(np.median(counts))} max={max(counts)}")
    return subjects
