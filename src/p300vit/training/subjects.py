"""Torch-free subject-level helpers shared by all CV harnesses."""
from __future__ import annotations

import numpy as np


def stack_subjects(subjects):
    """Concatenate SubjectEpochs into trial-level arrays with subject index."""
    X = np.concatenate([s.data for s in subjects], axis=0)
    y = np.concatenate([np.full(len(s.data), s.label, dtype=np.int64) for s in subjects])
    sid = np.concatenate([
        np.full(len(s.data), i, dtype=np.int64) for i, s in enumerate(subjects)
    ])
    site = np.array([s.site for s in subjects])
    subj_ids = np.array([s.subject_id for s in subjects])
    return {"X": X, "y": y, "sid": sid, "site": site, "subject_ids": subj_ids}


def _stratify_key(data: dict, subjects, mode: str) -> np.ndarray:
    """Per-trial stratification key: label, site, or site x label."""
    if mode == "label":
        return data["y"]
    site_of_subject = {i: s.site for i, s in enumerate(subjects)}
    site_arr = np.array([site_of_subject[i] for i in data["sid"]])
    if mode == "site":
        return site_arr
    return np.array([f"{s}_{l}" for s, l in zip(site_arr, data["y"])])


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity (Youden J)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])
