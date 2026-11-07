#!/usr/bin/env python
"""P300-ViT leak-free evaluation driver (standalone companion-paper repo).

Runs the manuscript's pipeline — xDAWN spatial filtering -> CWT scalograms
-> Vision Transformer — end to end and produces every number the paper
reports, from one frozen protocol (docs/PROTOCOL.md):

  participant : strict subject-wise 5-fold CV on the clinical cohort,
                ASZED, and the pooled cohort (site x label stratification).
  trialwise   : the literature's leaky trial-randomized protocol for the
                SAME variant — the window-level row of the contrast.
  selective   : reject-option / coverage-accuracy analysis from the saved
                participant-level OOF subject probabilities (post-hoc).
  fam         : input-gradient attribution through the frozen wavelet
                front-end + stimulus-locking check.

Artifacts: <out>/<run>/summary.json, oof_subjects.npz, fold_*/oof_trials.npz,
selective_classification.json, fam_summary.json, and a consolidated
<out>/xdawn_vit_summary.json.

This module executes inside the p300vit-standalone Kaggle kernel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p300vit.config import DatasetConfig, ExperimentConfig, set_seed  # noqa: E402
from p300vit.datasets import DATASET_REGISTRY  # noqa: E402

STAGES = ("participant", "trialwise", "selective", "fam")
RUN_NAMES = {"local": "xdawnvit_local", "aszed": "xdawnvit_aszed",
             "pooled": "xdawnvit_pooled"}


# --------------------------------------------------------------------------- #
# Model factory (the manuscript's pipeline)
# --------------------------------------------------------------------------- #
def make_model(cfg: ExperimentConfig):
    """xDAWN-projected epochs -> CWT scalograms -> ViT (K = xdawn components)."""
    from p300vit.models.cwt_vit import CWTViT

    return (
        lambda: CWTViT(
            n_channels=cfg.xdawn.n_components,
            n_times=cfg.preprocess.n_times,
            sfreq=cfg.preprocess.sfreq_target,
        )
    ), "xdawn"


# --------------------------------------------------------------------------- #
# Environment discovery (mount-layout agnostic)
# --------------------------------------------------------------------------- #
def find_input_dir(name: str) -> str | None:
    for base in ("/kaggle/input",):
        root = Path(base)
        if not root.exists():
            continue
        hits = [p for p in root.rglob(name) if p.is_dir()]
        if hits:
            return str(min(hits, key=lambda p: len(p.parts)))
    return None


def resolve_roots(args) -> tuple[str, str]:
    local_root = args.local_root or find_input_dir("p300-ieee") or find_input_dir("sabeti")
    aszed_root = args.aszed_root or find_input_dir("aszedeeg") or find_input_dir("aszed")
    if not local_root or not aszed_root:
        raise SystemExit(
            f"dataset roots not found (local={local_root}, aszed={aszed_root})")
    print(f"local_root={local_root}\naszed_root={aszed_root}", flush=True)
    return local_root, aszed_root


def build_cfg(dataset: str, local_root: str, aszed_root: str, out: Path,
              cache_dir: str, max_epochs: int | None) -> ExperimentConfig:
    ds_local = DatasetConfig(name="sabeti-p300", root=local_root)
    ds_aszed = DatasetConfig(name="aszedeeg", root=aszed_root)
    datasets = {"local": [ds_local], "aszed": [ds_aszed],
                "pooled": [ds_local, ds_aszed]}[dataset]
    cfg = ExperimentConfig(experiment_name=RUN_NAMES[dataset],
                           datasets=datasets, cache_dir=cache_dir)
    cfg.n_channels = 20 if dataset == "local" else 19
    cfg.output_dir = str(out)
    if dataset == "pooled":
        cfg.cv.stratify_by = "site_label"
    if max_epochs is not None:
        cfg.train.max_epochs = max_epochs
        cfg.train.patience = min(cfg.train.patience, max_epochs)
    return cfg


# --------------------------------------------------------------------------- #
# Stage A — participant-level (strict subject-wise 5-fold, leak-free)
# --------------------------------------------------------------------------- #
def stage_participant(args, local_root: str, aszed_root: str, out: Path) -> None:
    from p300vit.training.cv import run_cv

    for dataset in ("local", "aszed", "pooled"):
        cfg = build_cfg(dataset, local_root, aszed_root, out,
                        args.cache_dir, args.max_epochs)
        experiment_dir = Path(cfg.output_dir) / cfg.experiment_name
        if (experiment_dir / "summary.json").exists():
            print(f"[participant] [skip] {cfg.experiment_name} exists", flush=True)
            continue
        set_seed(cfg.cv.seed)
        subjects = DATASET_REGISTRY[dataset](cfg)
        print(f"[participant] {cfg.experiment_name}: {len(subjects)} subjects "
              f"({sum(s.label for s in subjects)} patients)", flush=True)
        model_factory, xdawn_mode = make_model(cfg)
        assert xdawn_mode == "xdawn"
        run_cv(cfg, subjects, model_factory, experiment_dir, xdawn_mode=xdawn_mode)


# --------------------------------------------------------------------------- #
# Stage B — trial-randomized (leaky) contrast, same variant
# --------------------------------------------------------------------------- #
def stage_trialwise(args, local_root: str, aszed_root: str, out: Path) -> None:
    from p300vit.training.trialwise import trialwise_cv

    for dataset in ("local", "aszed"):
        cfg = build_cfg(dataset, local_root, aszed_root, out,
                        args.cache_dir, args.max_epochs)
        experiment_dir = Path(cfg.output_dir) / f"trialwise_{cfg.experiment_name}"
        if (experiment_dir / "summary.json").exists():
            print(f"[trialwise] [skip] {experiment_dir.name} exists", flush=True)
            continue
        set_seed(cfg.cv.seed)
        subjects = DATASET_REGISTRY[dataset](cfg)
        print(f"[trialwise] {experiment_dir.name}: {len(subjects)} subjects (leaky)",
              flush=True)
        model_factory, xdawn_mode = make_model(cfg)
        trialwise_cv(cfg, subjects, model_factory, experiment_dir, xdawn_mode=xdawn_mode)


# --------------------------------------------------------------------------- #
# Stage C — selective classification (reject option), pure post-processing
# --------------------------------------------------------------------------- #
def selective_curve(probs: np.ndarray, labels: np.ndarray,
                    thetas: np.ndarray) -> list[dict]:
    """Coverage-accuracy trade-off: abstain when max(p, 1-p) < theta."""
    rows = []
    for theta in thetas:
        conf = np.maximum(probs, 1.0 - probs)
        covered = conf >= theta
        n_cov = int(covered.sum())
        if n_cov == 0:
            rows.append({"theta": float(theta), "coverage": 0.0, "n_covered": 0,
                         "accuracy": None, "sensitivity": None, "specificity": None})
            continue
        p_c, y_c = probs[covered], labels[covered]
        pred = (p_c >= 0.5).astype(int)
        pos, neg = y_c == 1, y_c == 0
        rows.append({
            "theta": round(float(theta), 2),
            "coverage": round(n_cov / len(probs), 4),
            "n_covered": n_cov,
            "accuracy": round(float((pred == y_c).mean()), 4),
            "sensitivity": None if not pos.any() else round(float(pred[pos].mean()), 4),
            "specificity": None if not neg.any() else round(float((pred[neg] == 0).mean()), 4),
        })
    return rows


def stage_selective(args, out: Path) -> None:
    thetas = np.arange(0.50, 0.96, 0.05)
    for run in RUN_NAMES.values():
        npz = out / run / "oof_subjects.npz"
        if not npz.exists():
            print(f"[selective] [skip] {npz} missing", flush=True)
            continue
        z = np.load(npz)
        probs, labels = z["probs"].astype(float), z["labels"].astype(int)
        rows = selective_curve(probs, labels, thetas)
        with open(out / run / "selective_classification.json", "w") as f:
            json.dump({"run": run, "n_subjects": int(len(probs)),
                       "rule": "abstain when max(p, 1-p) < theta",
                       "curve": rows}, f, indent=2)
        print(f"[selective] {run}: N={len(probs)}", flush=True)
        for r in rows:
            if r["n_covered"] and r["theta"] >= 0.60:
                print(f"    theta={r['theta']:.2f} coverage={r['coverage']:.2f} "
                      f"acc={r['accuracy']}", flush=True)


# --------------------------------------------------------------------------- #
# Stage D — FAM: input-gradient attribution + stimulus-locking check
# --------------------------------------------------------------------------- #
def stage_fam(args, local_root: str, aszed_root: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from p300vit.spatial.xdawn import apply_xdawn
    from p300vit.training.engine import plain_loader
    from p300vit.training.subjects import _stratify_key, stack_subjects

    cfg = build_cfg("local", local_root, aszed_root, out, args.cache_dir, None)
    experiment_dir = Path(cfg.output_dir) / cfg.experiment_name
    fold_dir = experiment_dir / f"fold_{cfg.cv.n_folds - 1}"
    ckpt, Wp = fold_dir / "model.pt", fold_dir / "xdawn_W.npy"
    if not ckpt.exists() or not Wp.exists():
        print(f"[fam] [skip] {ckpt} or {Wp} missing", flush=True)
        return

    set_seed(cfg.cv.seed)
    subjects = DATASET_REGISTRY["local"](cfg)
    data = stack_subjects(subjects)
    # reproduce the last fold's subject split exactly (same seed / scheme)
    from sklearn.model_selection import StratifiedKFold
    strat = _stratify_key(data, subjects, cfg.cv.stratify_by)
    subject_index = np.unique(data["sid"])
    subj_key = np.array([strat[data["sid"] == i][0] for i in subject_index])
    skf = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=True,
                          random_state=cfg.cv.seed)
    _, te_sub_pos = list(skf.split(subject_index, subj_key))[-1]
    te_subj = subject_index[te_sub_pos]
    te_mask = np.isin(data["sid"], te_subj)

    W = np.load(Wp)
    X_te = apply_xdawn(data["X"][te_mask], W)
    y_te = data["y"][te_mask]

    model_factory, _ = make_model(cfg)
    model = model_factory()
    model.load_state_dict(torch.load(ckpt, map_location=cfg.device))
    model.eval().to(cfg.device)

    # input-gradient attribution of the true-class score wrt the projected
    # epochs, through the frozen wavelet front-end
    loader = plain_loader(X_te, y_te, data["sid"][te_mask], cfg.train.batch_size)
    sfreq = cfg.preprocess.sfreq_target
    onset = int(round((0.0 - cfg.preprocess.tmin) * sfreq))
    T = X_te.shape[-1]
    grad_sum = np.zeros(T, dtype=np.float64)
    n_seen = 0
    for x, y, _ in loader:
        x = x.to(cfg.device).float().requires_grad_(True)
        logits, _ = model(x)
        score = logits.gather(1, y.to(cfg.device).view(-1, 1)).sum()
        (grad,) = torch.autograd.grad(score, x)
        grad_sum += grad.abs().sum(dim=0).sum(dim=0).detach().cpu().numpy()
        n_seen += x.shape[0]
    profile = grad_sum / max(n_seen, 1)
    profile /= profile.sum() + 1e-12
    pre, post = float(profile[:onset].sum()), float(profile[onset:].sum())
    peak = int(np.argmax(profile))
    summary = {
        "model": "xdawn_cwt_vit",
        "fold": cfg.cv.n_folds - 1,
        "n_trials": int(n_seen),
        "onset_sample": onset,
        "attribution_pre_stimulus_frac": round(pre, 4),
        "attribution_post_stimulus_frac": round(post, 4),
        "peak_sample": peak,
        "peak_latency_ms": round((peak - onset) / sfreq * 1000.0, 1),
    }
    with open(experiment_dir / "fam_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    t_ms = (np.arange(T) - onset) / sfreq * 1000.0
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(t_ms, profile, lw=1.2, color="#1f4e9c")
    ax.axvline(0.0, color="k", ls=":", lw=1)
    ax.axvspan(300, 450, color="0.85", zorder=0)
    ax.set_xlabel("Time from stimulus (ms)")
    ax.set_ylabel("mean |dS/dx| (norm.)")
    ax.set_title("xDAWN-ViT input-gradient attribution (held-out fold)")
    fig.tight_layout()
    fig.savefig(experiment_dir / "fam_profile.png", dpi=200)
    plt.close(fig)
    print(f"[fam] post-stimulus mass {post:.3f}, peak {summary['peak_latency_ms']} ms",
          flush=True)


# --------------------------------------------------------------------------- #
# Consolidated summary for the manuscript refresh
# --------------------------------------------------------------------------- #
def write_summary(out: Path) -> None:
    def _load(rel):
        p = out / rel
        return json.load(open(p)) if p.exists() else None

    summary = {}
    for ds, run in RUN_NAMES.items():
        s = _load(f"{run}/summary.json")
        if s:
            summary[f"participant_{ds}"] = {
                "n_subjects": s["n_subjects"],
                "pooled_oof_auc": s["metrics"]["auc"],
                "acc@0.5": s["metrics"]["acc@0.5"],
                "fold_mean_auc": round(float(np.mean([f["test_auc"] for f in s["folds"]])), 4),
                "fold_std_auc": round(float(np.std([f["test_auc"] for f in s["folds"]])), 4),
            }
        t = _load(f"trialwise_{run}/summary.json")
        if t:
            summary[f"trialwise_{ds}"] = {
                "trial_auc": t.get("trial_auc"),
                "trial_accuracy": t.get("trial_accuracy"),
                "subject_metrics": t.get("metrics"),
                "n_trials": t.get("n_trials"),
            }
        sel = _load(f"{run}/selective_classification.json")
        if sel:
            summary[f"selective_{ds}"] = sel["curve"]
    fam = _load("xdawnvit_local/fam_summary.json")
    if fam:
        summary["fam_local"] = fam
    with open(out / "xdawn_vit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] -> {out / 'xdawn_vit_summary.json'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["all", *STAGES], default="all")
    p.add_argument("--out", default="results/xdawn_vit")
    p.add_argument("--cache-dir", default="/tmp/p300_cache")
    p.add_argument("--local-root", default=None)
    p.add_argument("--aszed-root", default=None)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="override training epochs (smoke runs only)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    local_root, aszed_root = resolve_roots(args)

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n===== stage: {stage} =====", flush=True)
        if stage == "participant":
            stage_participant(args, local_root, aszed_root, out)
        elif stage == "trialwise":
            stage_trialwise(args, local_root, aszed_root, out)
        elif stage == "selective":
            stage_selective(args, out)
        elif stage == "fam":
            stage_fam(args, local_root, aszed_root, out)
    write_summary(out)


if __name__ == "__main__":
    main()
