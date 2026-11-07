"""Trial-randomized (event-shuffled) CV -- the literature's leaky protocol.

Vendored verbatim (module renamed) from the benchmark's wall-1 machinery so
the evaluation-level contrast uses the exact same leaky-protocol definition:
folds stratify TRIALS by label; every fold's training set contains trials of
every subject (including future-test subjects). xDAWN is still fitted inside
the fold on train trials only -- the leak comes from the split, not the
filter. Subject-level decisions are the soft vote over held-out trials.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..config import ExperimentConfig, set_seed
from ..evaluation.stats import full_metrics
from ..spatial.xdawn import apply_xdawn, fit_xdawn
from ..training.engine import plain_loader, predict_trials, train_model
from ..training.subjects import stack_subjects


def _save_experiment(results_dir: Path, name: str, protocol: str,
                     ids: List[str], probs: np.ndarray, labels: np.ndarray,
                     extra: Dict | None = None) -> dict:
    out_dir = Path(results_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    np.savez_compressed(out_dir / "oof_subjects.npz",
                        subject_ids=np.array([str(s) for s in ids], dtype=np.str_),
                        probs=probs, labels=labels)
    m = full_metrics(labels, probs)
    summary = {"experiment": name, "protocol": protocol,
               "n_subjects": int(len(labels)), "metrics": m}
    if extra:
        summary.update(extra)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{name}] AUC {m['auc']['point']:.4f} "
          f"[{m['auc']['ci_low']:.4f},{m['auc']['ci_high']:.4f}] "
          f"acc {m['accuracy']['point']:.3f} sens {m['sensitivity']['point']:.3f} "
          f"spec {m['specificity']['point']:.3f}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
# 1-2. Trial-level (leaky) CV
# --------------------------------------------------------------------------- #
def trialwise_cv(cfg: ExperimentConfig, subjects,
                 model_factory: Callable[[], object],
                 experiment_dir: Path, xdawn_mode: str = "xdawn",
                 inner_val_frac: float = 0.1) -> dict:
    """Trial-level (event-shuffled) CV — the literature's leaky protocol.

    Folds stratify TRIALS by label; in every fold a slice of every subject's
    trials appears in the training set (including future-test subjects).
    xDAWN is still fitted inside the fold on train trials only — the leak
    comes from the split, not the filter. Subject-level decisions are the
    soft vote over a subject's held-out trials, exactly how the inflated
    protocol reports subject accuracy.

    Returns a summary with BOTH the trial-level OOF AUC/accuracy (the
    ~98% number the literature reports) and the subject-level battery.
    """
    from ..config import set_seed
    from ..spatial.xdawn import apply_xdawn, fit_xdawn
    from ..training.engine import train_model

    set_seed(cfg.cv.seed)
    data = stack_subjects(subjects)
    X, y, sid, subj_ids = data["X"], data["y"], data["sid"], data["subject_ids"]
    experiment_dir = Path(experiment_dir)
    results_dir = experiment_dir.parent
    name = experiment_dir.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(experiment_dir / "config.yaml")

    n_trials = len(y)
    skf = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=True,
                          random_state=cfg.cv.seed)
    oof_trial_probs = np.full(n_trials, np.nan)
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(n_trials), y)):
        t0 = time.time()
        fold_dir = experiment_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        sid_tr, sid_te = sid[tr_idx], sid[te_idx]

        if xdawn_mode == "xdawn":
            W = fit_xdawn([X_tr[sid_tr == u] for u in np.unique(sid_tr)],
                          cfg.xdawn)
            np.save(fold_dir / "xdawn_W.npy", W)
            X_tr, X_te = apply_xdawn(X_tr, W), apply_xdawn(X_te, W)

        # inner validation slice (fold-train trials) for early stopping
        rng = np.random.default_rng(cfg.cv.seed + 100 + fold)
        perm = rng.permutation(len(y_tr))
        n_val = max(64, int(inner_val_frac * len(y_tr)))
        va_pos, it_pos = perm[:n_val], perm[n_val:]
        model = model_factory()
        hist = train_model(model, cfg,
                           X_tr[it_pos], y_tr[it_pos], sid_tr[it_pos],
                           X_tr[va_pos], y_tr[va_pos], sid_tr[va_pos],
                           fold_dir=fold_dir, fold_seed=fold)

        loader = plain_loader(X_te, y_te, sid_te, cfg.train.batch_size)
        tp, _, _ = predict_trials(model, loader, cfg.device, cfg.train.amp)
        oof_trial_probs[te_idx] = tp
        row = {
            "fold": fold,
            "n_train_trials": int(len(y_tr)),
            "n_test_trials": int(len(y_te)),
            "best_val_auc": hist["best_val_auc"],
            "best_epoch": hist["best_epoch"],
            "seconds": round(time.time() - t0, 1),
        }
        fold_rows.append(row)
        print(f"  fold {fold}: val_auc {row['best_val_auc']:.3f} "
              f"({row['n_test_trials']} test trials, {row['seconds']}s)",
              flush=True)
        np.savez_compressed(fold_dir / "oof_trials.npz",
                            probs=tp, labels=y_te, sid=sid_te,
                            subject_ids=np.array(
                                [str(subj_ids[s]) for s in sid_te],
                                dtype=np.str_))

    assert not np.isnan(oof_trial_probs).any(), "unassigned OOF trials"
    labels_arr = y  # every trial appears exactly once as a test trial
    trial_auc = float(roc_auc_score(labels_arr, oof_trial_probs))
    trial_acc = float(accuracy_score(labels_arr,
                                     (oof_trial_probs >= 0.5).astype(int)))
    np.savez_compressed(experiment_dir / "oof_trials_all.npz",
                        probs=oof_trial_probs, labels=labels_arr, sid=sid)

    # subject-level soft vote over each subject's held-out trials
    ids, probs, subj_labels = [], [], []
    for u in np.unique(sid):
        m = sid == u
        ids.append(str(subj_ids[u]))
        probs.append(float(oof_trial_probs[m].mean()))
        subj_labels.append(int(labels_arr[m][0]))

    return _save_experiment(
        results_dir, name,
        "trial-level CV (event-shuffled; leaky by construction)",
        ids, np.array(probs), np.array(subj_labels),
        extra={"trial_auc": trial_auc, "trial_accuracy": trial_acc,
               "n_trials": int(n_trials), "folds": fold_rows,
               "paradigm": "baseline-only" if cfg.preprocess.tmax == 0.0
               else "full-epoch"})


# --------------------------------------------------------------------------- #
