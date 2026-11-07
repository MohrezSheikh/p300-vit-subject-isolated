"""Adapter for the Sabeti P300 / P300-IEEE dataset (Neuroscan .cnt).

Two supported layouts (auto-detected):

Layout A -- original local archive:
    <root>/CT/cH<CODE>.cnt   -> healthy control recordings (c-prefixed files;
                                the h*02/h*20 variants are alternate
                                acquisitions of the same subjects and are
                                NOT used, per the dataset owner's instruction)
    <root>/PB/cH<CODE>.cnt   -> schizophrenia patient recordings
    Corrupt files (e.g. CT/cHGTRPB1.cnt, a broken duplicate of HGTRPB) are
    detected at load time and skipped with a logged reason.

Layout B -- Kaggle export (p300-ieee):
    <root>/ch<CODE>20.cnt    -> control     (re-export of h<CODE>20 recordings)
    <root>/ph<CODE>20.cnt    -> patient
    <root>/p.cnt             -> provenance-unknown stray (excluded; almost
                                certainly a damaged export of CT/cHGTRPB1)

Both layouts: 20 scalp channels, 200 Hz, auditory oddball with ~80 target
('1') and ~720 standard ('2') stimuli per recording, delivered either as
stim-channel triggers (Kaggle export) or as CNT annotations (originals).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import mne
import numpy as np

from ..config import DatasetConfig, PreprocessConfig
from .preprocessing import SubjectEpochs, preprocess_raw

SCALP_CHANNELS = [
    "FPZ", "FZ", "CZ", "PZ", "C3", "T3", "C4", "T4",
    "FP1", "FP2", "F3", "F4", "F7", "F8", "P3", "P4",
    "T5", "T6", "O1", "O2",
]


def _read_cnt_robust(path: Path) -> mne.io.BaseRaw:
    """Read a Neuroscan .cnt, working around two mne parsing bugs.

    1. Dangling BAD span: mne's annotation parser raises IndexError when a
       recording ends on an open BAD segment (no closing good marker) --
       several original recordings (e.g. CT/cHDBZPB.cnt) have one. Fallback:
       ``_read_annotations_cnt`` is replaced by a replica that closes the
       dangling span with a sentinel good marker.
    2. Corrupt header sample count: some files carry a garbage n_samples
       (e.g. 2**30-ish), so mne tries to allocate hundreds of GiB. Fallback:
       read with data_format='int16' + recompute_n_samples=True.
    """
    import mne.io.cnt.cnt as C

    try:
        return mne.io.read_raw_cnt(str(path), preload=True, verbose=False)
    except IndexError:
        pass
    except (RuntimeError, MemoryError):
        try:
            return mne.io.read_raw_cnt(
                str(path), preload=True, verbose=False,
                data_format="int16", recompute_n_samples=True,
            )
        except (RuntimeError, MemoryError) as e2:
            raise RuntimeError(f"{path.name}: unreadable CNT header ({e2})") from e2

    # --- fallback path: dangling BAD span ------------------------------- #
    def _accept_reject_function(keypad_accept):
        accept_list = []
        for code in keypad_accept:
            if "xd0" in str(code):
                accept_list.append("good")
            elif "xc0" in str(code):
                accept_list.append("bad")
            else:
                accept_list.append("NA")
        return np.array(accept_list)

    def _translating_function(offset, n_channels, event_type, *, n_bytes):
        if event_type == C.CNTEventType3:
            offset *= n_bytes * n_channels
        event_time = offset - C._DATA_OFFSET - (C._CH_SIZE * n_channels)
        event_time //= n_channels * n_bytes
        event_time = event_time - 1
        np.clip(event_time, 0, None, out=event_time)
        return event_time

    def _read_annotations_fixed(fname, *, data_format, verbose=None):
        with open(fname, "rb") as fid:
            fid.seek(C._RATE_OFFSET)
            sfreq = float(np.frombuffer(fid.read(2), dtype="<u2").item())
            n_channels, _, n_bytes, event_table_pos = C._compute_robust_sizes(
                fid=fid, data_format=data_format, recompute_n_samples=False
            )
            teeg = C._read_teeg(fid, teeg_offset=event_table_pos)
            event_parser = C._get_event_parser(event_type=teeg.event_type)
            fid.seek(event_table_pos + 9)
            buffer = fid.read(teeg.total_length)

        my_events = list(event_parser(buffer))
        if not my_events:
            return mne.Annotations([], [], [], None)

        onset = _translating_function(
            np.array([e.Offset for e in my_events], dtype=float),
            n_channels=n_channels, event_type=type(my_events[0]), n_bytes=n_bytes,
        )
        duration = np.zeros(len(my_events), dtype=float)
        accept_reject = _accept_reject_function(
            np.array([e.KeyPad_Accept for e in my_events])
        )
        description = []
        for event in my_events:
            keypad = event.KeyPad_Accept[0] & 0x0F
            if str(keypad) != "0":
                description.append(f"KeyPad Response {keypad}")
            elif event.KeyBoard != 0:
                description.append(f"Keyboard Response {event.KeyBoard}")
            else:
                description.append(str(event.StimType))
        description = np.array(description)

        # robust replacement of mne's _update_bad_span_onset: each BAD marker
        # is closed by the NEXT bad/good marker (mne assumes strict
        # alternation, which these files violate -- e.g. consecutive BADs)
        ar = [str(v) for v in accept_reject]
        positions = [i for i, v in enumerate(ar) if v in ("bad", "good")]
        onset_l = np.asarray(onset, dtype=float)
        duration = np.asarray(duration, dtype=float)
        description = np.array([str(d) for d in description], dtype=object)
        for pos_idx, i in enumerate(positions):
            if ar[i] == "bad":
                nxt = positions[pos_idx + 1] if pos_idx + 1 < len(positions) else None
                duration[i] = (onset_l[nxt] - onset_l[i]) if nxt is not None else 1.0
                description[i] = "BAD_" + description[i]
        keep = [i for i in range(len(ar)) if ar[i] != "good"]
        return mne.Annotations(
            onset=onset_l[keep].tolist(),
            duration=duration[keep].tolist(),
            description=description[keep].tolist(),
            orig_time=None,
        )

    orig_reader = C._read_annotations_cnt
    C._read_annotations_cnt = _read_annotations_fixed
    try:
        return mne.io.read_raw_cnt(str(path), preload=True, verbose=False)
    finally:
        C._read_annotations_cnt = orig_reader


def _detect_layout(root: Path) -> str:
    return "local" if (root / "CT").exists() and (root / "PB").exists() else "kaggle"


def _pick_target_code(codes: np.ndarray, counts: np.ndarray) -> Optional[int]:
    """Oddball target = a stimulus code occurring at a deviant-like rate.

    Requires a code whose frequency lies in [5%, 40%] of all stimulus events
    with at least 20 occurrences (targets: ~80 of ~800 in this dataset).
    Block/rare markers and near-empty event tables return None so the caller
    can fall back to another recording of the same subject.
    """
    total = counts.sum()
    freq = counts / total
    candidates = [
        (int(c), int(n)) for c, f, n in zip(codes, freq, counts)
        if 0.05 <= f <= 0.40 and n >= 20
    ]
    if not candidates:
        return None
    return min(c for c, _ in candidates)


def _load_recording(
    path: Path, label: int, site: str, prep_cfg: PreprocessConfig
) -> SubjectEpochs:
    raw = _read_cnt_robust(path)
    available = [ch for ch in SCALP_CHANNELS if ch in raw.ch_names]
    if len(available) != len(SCALP_CHANNELS):
        missing = set(SCALP_CHANNELS) - set(available)
        raise ValueError(f"{path.name}: missing canonical channels {sorted(missing)}")

    # resolve the target stimulus code from triggers or annotations
    target_code: Optional[int] = None
    events = None
    try:
        events = mne.find_events(raw, shortest_event=1, verbose=False)
    except ValueError:
        events = None
    if events is not None and len(events) > 0:
        codes, counts = np.unique(events[:, 2], return_counts=True)
        target_code = _pick_target_code(codes, counts)
        if target_code is None:
            raise ValueError(f"{path.name}: no plausible target code in triggers")
        subj = preprocess_raw(raw, prep_cfg, picks=available, event_id=target_code)
    else:
        ann = raw.annotations
        stimulus_desc = [d for d in np.unique(ann.description) if str(d).isdigit()]
        if not stimulus_desc:
            raise ValueError(f"{path.name}: no triggers or numeric annotations found")
        codes = np.array([int(d) for d in stimulus_desc])
        counts = np.array([int((ann.description == d).sum()) for d in stimulus_desc])
        target_code = _pick_target_code(codes, counts)
        if target_code is None:
            raise ValueError(f"{path.name}: no plausible target code in annotations")
        # epoch the target code ONLY (exclude standards and block markers)
        subj = preprocess_raw(
            raw, prep_cfg, picks=available, event_id=1,
            event_desc={str(target_code): 1},
        )
    subj.subject_id = path.stem
    subj.label = label
    subj.site = site
    return subj


def discover_subjects(root: Path) -> List[Dict]:
    """Enumerate subjects with explicit labels and candidate files in order.

    Local layout: the primary recording is the c-prefixed file; some
    primaries have lost their stimulus event table at acquisition (verified
    for cHDBZPB/cHIZNPB: only 2-3 markers instead of ~800). For those, the
    subject's h*20 variant -- the same acquisition stream that the Kaggle
    export used -- serves as fallback.
    """
    layout = _detect_layout(root)
    records: List[Dict] = []
    if layout == "local":
        for sub, label in (("CT", 0), ("PB", 1)):
            per_subject: Dict[str, List[Path]] = {}
            primaries: Dict[str, Path] = {}
            for f in sorted((root / sub).glob("*.cnt")):
                stem = f.stem
                if stem.startswith("c"):               # primary c-recording
                    # 'cHDBZPB' -> 'HDBZPB'; 'cHGTRPB1' -> 'HGTRPB'
                    # (all subject codes end in 'PB'; trailing digits mark
                    #  a secondary recording of the same subject)
                    code = re.sub(r"(?<=PB)\d+$", "", stem[1:])
                    per_subject.setdefault(code, []).insert(0, f)
                    primaries[code] = f
                elif stem.startswith("h"):             # h-variant fallbacks
                    # 'hDBZPB20' -> 'DBZPB' -> canonical code 'HDBZPB'
                    code = "H" + re.sub(r"\d+$", "", stem[1:])
                    # prefer the complete '20' variant over '02'
                    lst = per_subject.setdefault(code, [])
                    lst.insert(min(1 if "20" in stem else 2, len(lst)), f)
            for code, files in sorted(per_subject.items()):
                if code not in primaries:
                    # no primary c-recording exists for this code: identity
                    # cannot be verified against the 20/20 cohort -> excluded
                    print(f"[p300-ieee] excluding {code}: no primary c-recording "
                          f"(h-variants only)")
                    continue
                records.append({"subject_id": code, "label": label, "files": files})
    else:
        for f in sorted(root.glob("*.cnt")):
            name = f.stem
            if name.startswith("ch"):
                label, note = 0, ""
            elif name.startswith("ph"):
                label, note = 1, ""
            elif name == "p":
                label, note = None, (
                    "stray 'p.cnt': excluded from the 20/20 cohort "
                    "(damaged export; see INSPECTION_REPORT.md)"
                )
            else:
                label, note = None, "unrecognised filename pattern"
            records.append({
                "subject_id": name, "label": label, "files": [f], "note": note,
            })
    return records


def load_dataset(
    dataset_cfg: DatasetConfig,
    prep_cfg: PreprocessConfig,
    include_ambiguous: bool = False,
    verbose: bool = True,
    cache_dir: Optional[str] = None,
) -> List[SubjectEpochs]:
    """Load the full 40-subject cohort (c-recordings for the local layout)."""
    import hashlib, json

    root = dataset_cfg.resolve()
    records = discover_subjects(root)
    cache_key = None
    if cache_dir is not None:
        payload = json.dumps(prep_cfg.__dict__, sort_keys=True, default=str)
        cache_key = Path(cache_dir) / ("p300ieee_" + hashlib.md5(payload.encode()).hexdigest()[:12])
        cache_key.mkdir(parents=True, exist_ok=True)

    subjects: List[SubjectEpochs] = []
    for rec in records:
        if rec["label"] is None:
            if verbose and rec.get("note"):
                print(f"[p300-ieee] excluding {rec['subject_id']}: {rec['note']}")
            if not include_ambiguous:
                continue
            raise ValueError(f"Cannot include {rec['subject_id']}: label unknown.")
        cache_file = cache_key / f"{rec['subject_id']}.npz" if cache_key else None
        if cache_file and cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as z:
                subjects.append(SubjectEpochs(
                    subject_id=rec["subject_id"], label=int(z["label"]),
                    site=dataset_cfg.name, data=z["data"],
                    ch_names=list(SCALP_CHANNELS),
                    times=np.linspace(prep_cfg.tmin, prep_cfg.tmax, prep_cfg.n_times, dtype=np.float32),
                    sfreq=prep_cfg.sfreq_target,
                ))
            continue
        subj = None
        for f in rec["files"]:                       # primary c-file first
            try:
                candidate = _load_recording(f, rec["label"], dataset_cfg.name, prep_cfg)
                # canonical subject identity: the discovery code (no filename
                # prefix/suffix) -- keeps IDs identical between the fresh-load
                # and cache paths, otherwise paired statistics misalign
                candidate.subject_id = rec["subject_id"]
            except Exception as e:
                if verbose:
                    print(f"[p300-ieee] FAILED to load {f.name}: {type(e).__name__}: {e}")
                continue
            if set(candidate.ch_names) != set(SCALP_CHANNELS):
                if verbose:
                    dropped = sorted(set(SCALP_CHANNELS) - set(candidate.ch_names))
                    print(f"[p300-ieee] {f.name}: unusable montage (dead {dropped}) "
                          f"— trying next recording of {rec['subject_id']}")
                continue
            if len(candidate.data) > 0:
                subj = candidate
                break
            if verbose:
                print(f"[p300-ieee] {f.name}: 0 valid target epochs "
                      f"(empty event table) — trying next recording of {rec['subject_id']}")
        if subj is None:
            if verbose:
                print(f"[p300-ieee] excluding {rec['subject_id']}: no usable recording")
            continue
        subjects.append(subj)
        if cache_file:
            np.savez_compressed(cache_file, data=subj.data.astype(np.float32), label=subj.label)

    if verbose:
        n_pos = sum(s.label for s in subjects)
        counts = [len(s.data) for s in subjects]
        print(f"[p300-ieee] loaded {len(subjects)} subjects ({n_pos} patients, "
              f"{len(subjects)-n_pos} controls); trials/subject: "
              f"min={min(counts)} median={int(np.median(counts))} max={max(counts)}")
    return subjects
