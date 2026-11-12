"""Subject-wise cross-validation harness with in-fold xDAWN.

Protocol
--------
* Folds are constructed over SUBJECTS (never trials), stratified by label
  (or site x label for pooled runs).
* Per fold: xDAWN is fitted on training subjects ONLY, then applied to all
  trials; a stratified slice of the training subjects forms an inner
  validation set for model selection and threshold tuning; the test subjects
  are scored exactly once at the end.
* Every artifact (OOF predictions, metrics, model weights, config) is saved
  so that all statistics and figures can be regenerated from files alone.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..config import ExperimentConfig
from ..spatial.xdawn import apply_xdawn, fit_xdawn
from .engine import predict_trials, subject_probs, train_model
from .subjects import _stratify_key, stack_subjects


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_cv(
    cfg: ExperimentConfig,
    subjects,
    model_factory: Callable[[], object],
    experiment_dir: Path,
    xdawn_mode: str = "xdawn",          # xdawn | none
) -> Dict:
    """Run the full CV protocol and write every artifact to experiment_dir."""
    from ..config import set_seed

    set_seed(cfg.cv.seed)
    data = stack_subjects(subjects)
    strat = _stratify_key(data, subjects, cfg.cv.stratify_by)
    # stratify per SUBJECT (unique per subject via first occurrence)
    subject_index = np.unique(data["sid"])
    subj_key = np.array([strat[data["sid"] == i][0] for i in subject_index])

    skf = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=True, random_state=cfg.cv.seed)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(experiment_dir / "config.yaml")

    n_channels = data["X"].shape[1]
    oof_subject_ids, oof_probs, oof_labels = [], [], []
    fold_rows = []

    for fold, (tr_sub_pos, te_sub_pos) in enumerate(
        skf.split(subject_index, subj_key)
    ):
        t0 = time.time()
        tr_subj = subject_index[tr_sub_pos]
        te_subj = subject_index[te_sub_pos]
        tr_mask = np.isin(data["sid"], tr_subj)
        te_mask = np.isin(data["sid"], te_subj)
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        # ---- in-fold xDAWN (train subjects only) ---------------------- #
        X_all = data["X"]
        if xdawn_mode == "xdawn":
            train_epochs = [data["X"][data["sid"] == i] for i in tr_subj]
            W = fit_xdawn(train_epochs, cfg.xdawn)
            np.save(fold_dir / "xdawn_W.npy", W)
            X_tr, X_te = apply_xdawn(X_all[tr_mask], W), apply_xdawn(X_all[te_mask], W)
        else:
            X_tr, X_te = X_all[tr_mask], X_all[te_mask]

        y_tr, sid_tr = data["y"][tr_mask], data["sid"][tr_mask]
        y_te, sid_te = data["y"][te_mask], data["sid"][te_mask]

        # ---- inner validation split (training subjects, stratified) ---- #
        unique_tr = np.unique(sid_tr)
        tr_key = np.array([y_tr[sid_tr == i][0] for i in unique_tr])
        n_inner = min(5, int(np.bincount(tr_key.astype(int)).min()))
        inner = StratifiedKFold(n_splits=max(2, n_inner), shuffle=True,
                                random_state=cfg.cv.seed + fold)
        inner_tr_pos, va_pos = next(inner.split(unique_tr, tr_key))
        it_subj, va_subj = unique_tr[inner_tr_pos], unique_tr[va_pos]
        it_mask = np.isin(sid_tr, it_subj)
        va_mask = np.isin(sid_tr, va_subj)

        model = model_factory()
        params = count_parameters(model)
        hist = train_model(
            model, cfg,
            X_tr[it_mask], y_tr[it_mask], sid_tr[it_mask],
            X_tr[va_mask], y_tr[va_mask], sid_tr[va_mask],
            fold_dir=fold_dir, fold_seed=fold,
        )

        # ---- final test prediction ------------------------------------- #
        from .engine import EEGTrials, plain_loader
        loader = plain_loader(X_te, y_te, sid_te, cfg.train.batch_size)
        tp, ty, tsid = predict_trials(model, loader, cfg.device, cfg.train.amp)

        subj_p = subject_probs(tp, tsid)
        te_subj_ids = np.unique(tsid)
        for s in te_subj_ids:
            oof_subject_ids.append(str(data["subject_ids"][s]))
            oof_probs.append(subj_p[str(s)])
            oof_labels.append(int(y_te[tsid == s][0]))

        # per-fold test metrics at the val-chosen threshold and at 0.5
        from sklearn.metrics import accuracy_score, roc_auc_score
        val_p, val_y, _ = predict_trials(
            model, plain_loader(X_tr[va_mask], y_tr[va_mask], sid_tr[va_mask], cfg.train.batch_size),
            cfg.device, cfg.train.amp,
        )
        vp_s = subject_probs(val_p, sid_tr[va_mask])
        val_subj_ids = np.unique(sid_tr[va_mask])
        vs = np.array([vp_s[str(s)] for s in val_subj_ids])
        vy = np.array([int(y_tr[va_mask][sid_tr[va_mask] == s][0]) for s in val_subj_ids])
        thr = hist["threshold"]

        ts_probs = np.array([subj_p[str(s)] for s in te_subj_ids])
        ts_labels = np.array([int(y_te[tsid == s][0]) for s in te_subj_ids])
        # persist inner-validation subject probabilities so post-hoc per-fold
        # calibration (Platt / threshold anchoring) can be fitted LEAK-FREE,
        # i.e. without ever touching the test fold (see evaluation.stats).
        np.savez_compressed(
            fold_dir / "val_preds.npz",
            probs=vs, labels=vy, threshold=float(thr),
            subject_ids=np.array([str(data["subject_ids"][s]) for s in val_subj_ids],
                                 dtype=np.str_),
        )
        row = {
            "fold": fold,
            "n_train_subjects": int(len(tr_subj)),
            "n_test_subjects": int(len(te_subj)),
            "n_params": int(params),
            "best_val_auc": hist["best_val_auc"],
            "best_epoch": hist["best_epoch"],
            "threshold": thr,
            "test_auc": float(roc_auc_score(ts_labels, ts_probs)),
            "test_acc@thr": float(accuracy_score(ts_labels, (ts_probs >= thr).astype(int))),
            "test_acc@0.5": float(accuracy_score(ts_labels, (ts_probs >= 0.5).astype(int))),
            "seconds": round(time.time() - t0, 1),
        }
        fold_rows.append(row)
        print(f"  fold {fold}: test AUC {row['test_auc']:.3f} | "
              f"acc@thr {row['test_acc@thr']:.3f} | params {params:,} "
              f"| {row['seconds']}s", flush=True)

        # persist OOF trial probabilities for this fold
        np.savez_compressed(
            fold_dir / "oof_trials.npz",
            probs=tp, labels=ty, sid=tsid,
            subject_ids=np.array([str(data["subject_ids"][s]) for s in tsid], dtype=np.str_),
        )

    # ---- pooled OOF at subject level ---------------------------------- #
    oof_probs_arr = np.array(oof_probs)
    oof_labels_arr = np.array(oof_labels)
    from sklearn.metrics import accuracy_score, roc_auc_score
    summary = {
        "experiment": cfg.experiment_name,
        "n_subjects": int(len(oof_probs_arr)),
        "metrics": {
            "auc": float(roc_auc_score(oof_labels_arr, oof_probs_arr)),
            "acc@0.5": float(accuracy_score(oof_labels_arr, (oof_probs_arr >= 0.5).astype(int))),
        },
        "folds": fold_rows,
    }
    np.savez_compressed(
        experiment_dir / "oof_subjects.npz",
        subject_ids=np.array(oof_subject_ids), probs=oof_probs_arr,
        labels=oof_labels_arr,
    )
    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  pooled OOF AUC {summary['metrics']['auc']:.4f} | "
          f"acc@0.5 {summary['metrics']['acc@0.5']:.4f} "
          f"(N={summary['n_subjects']})", flush=True)
    return summary
