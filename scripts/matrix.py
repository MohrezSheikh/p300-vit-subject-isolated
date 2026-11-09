#!/usr/bin/env python
"""Full experimental matrix: cohort x montage x model size x augmentation.

Every cell runs the SAME frozen protocol (subject-wise stratified 5-fold CV,
seed 42, leak-free per-fold calibration) so the cells are directly comparable:

  cohorts   : clinical (S1 clinical oddball), ASZED
  montages  : full (all channels) and AEP (a-priori auditory-evoked sites)
              - clinical AEP = Pz, P3, P4  (P3b maximum, verified empirically)
              - ASZED   AEP = Fz, Cz, Pz   (no P3b in this passive MMDA paradigm)
  models    : base  (256-d, 6 blocks, 8 heads)  and lite (128-d, 4 blocks, 4 heads)
  augment   : off | on  (training batches only: amplitude scale, +/-10 ms shift,
              SNR-controlled Gaussian noise, same-class mixup, channel dropout,
              plus SpecAugment time/frequency masking of the log-scalogram)

Leakage contract: augmentation is applied inside the fold to training batches
only; validation and test inputs are never augmented. xDAWN is fitted on
training-fold subjects only. Calibration uses inner-validation subjects only.

Usage
-----
    python scripts/matrix.py --out results/matrix \
        --local-root /kaggle/input/... --aszed-root /kaggle/input/...

    # smoke test (2 epochs, one cell)
    python scripts/matrix.py --out /tmp/m --epochs 2 --only clinical:full:base:off
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p300vit.config import load_config  # noqa: E402
from p300vit.data.augment import EpochAugment, augment_noop  # noqa: E402
from p300vit.data.cnt_dataset import SCALP_CHANNELS  # noqa: E402
from p300vit.data.preprocessing import SubjectEpochs  # noqa: E402
from p300vit.models.cwt_vit import CWTViT, count_parameters  # noqa: E402
from p300vit.training.cv import run_cv  # noqa: E402

# ---------------------------------------------------------------- config ----
COHORTS = {
    "clinical": dict(
        cfg_name="xdawnvit_local",
        root_attr="local_root",
        montage_aep=["PZ", "P3", "P4"],
    ),
    "aszed": dict(
        cfg_name="xdawnvit_aszed",
        root_attr="aszed_root",
        montage_aep=["FZ", "CZ", "PZ"],
    ),
}
MODELS = {
    "base": dict(embed_dim=256, depth=6, heads=8),
    "lite": dict(embed_dim=128, depth=4, heads=4),
}

# Augmentation presets. "strong" is the aggressive recipe that collapsed the
# lite model (best epoch 1-2, near-constant validation probabilities);
# "weak" is the fair retest: less noise, +/-1 sample shift, no mixup, short
# masks and light component dropout.
AUG_PRESETS = {
    "strong": dict(p_noise=0.5, snr_db=15.0, p_scale=0.5, scale=(0.90, 1.10),
                   p_shift=0.5, max_shift=2, p_mixup=0.3, p_chan_drop=0.2,
                   mask_time_max=30, mask_freq_max=2),
    "weak":   dict(p_noise=0.4, snr_db=25.0, p_scale=0.4, scale=(0.95, 1.05),
                   p_shift=0.4, max_shift=1, p_mixup=0.0, p_chan_drop=0.1,
                   mask_time_max=10, mask_freq_max=1),
}


# ------------------------------------------------------------- subjects ----
def load_subjects(cfg, dataset_key: str, roots: dict):
    """Load one cohort through the released adapters (with cache + roots)."""
    from p300vit.datasets import DATASET_REGISTRY

    root = roots.get(COHORTS[dataset_key]["root_attr"])
    if root:
        # override the dataset root(s) for this cohort
        for ds in cfg.datasets:
            if dataset_key == "clinical" and ds.name == "sabeti-p300":
                ds.root = root
            if dataset_key == "aszed" and ds.name == "aszedeeg":
                ds.root = root
    key = "local" if dataset_key == "clinical" else "aszed"
    return DATASET_REGISTRY[key](cfg)


def subset(subjects, chans):
    idx = [subjects[0].ch_names.index(c) for c in chans]
    out = []
    for s in subjects:
        out.append(
            SubjectEpochs(
                subject_id=s.subject_id, label=s.label, site=s.site,
                data=s.data[:, idx, :], ch_names=list(chans), times=s.times,
                sfreq=s.sfreq, n_rejected=s.n_rejected,
                flat_channels=list(s.flat_channels), files_used=s.files_used,
            )
        )
    return out


# ---------------------------------------------------------------- cells ----
def run_cell(cohort, montage, model_name, aug, cfg, subjects, out_root, epochs,
             aug_strength="strong"):
    n_ch = len(subjects[0].ch_names)
    K = min(4, n_ch)
    cfg.xdawn.n_components = K
    if epochs:
        cfg.train.max_epochs = epochs

    m = MODELS[model_name]
    preset = AUG_PRESETS[aug_strength]
    augment_fn = (
        EpochAugment(seed=cfg.cv.seed,
                     p_noise=preset["p_noise"], snr_db=preset["snr_db"],
                     p_scale=preset["p_scale"], scale=preset["scale"],
                     p_shift=preset["p_shift"], max_shift=preset["max_shift"],
                     p_mixup=preset["p_mixup"], p_chan_drop=preset["p_chan_drop"])
        if aug else augment_noop
    )

    def factory():
        return CWTViT(
            n_channels=K,
            n_times=cfg.preprocess.n_times,
            sfreq=cfg.preprocess.sfreq_target,
            embed_dim=m["embed_dim"], depth=m["depth"], heads=m["heads"],
            mask_scalogram=bool(aug),
            mask_time_max=preset["mask_time_max"],
            mask_freq_max=preset["mask_freq_max"],
        )

    tag = f"aug_{aug_strength}" if aug else "noaug"
    exp_dir = Path(out_root) / f"{cohort}_{montage}_{model_name}_{tag}"
    t0 = time.time()
    res = run_cv(cfg, subjects, factory, exp_dir, xdawn_mode="xdawn",
                 augment=(augment_fn if aug else None))
    folds = json.loads((exp_dir / "summary.json").read_text())["folds"]
    best_eps = [f["best_epoch"] for f in folds]
    val_aucs = [f["best_val_auc"] for f in folds]
    return {
        "cohort": cohort, "montage": montage, "model": model_name,
        "augment": bool(aug), "aug_strength": (aug_strength if aug else None),
        "n_channels": n_ch, "K": K,
        "mean_best_epoch": float(np.mean(best_eps)),
        "n_folds_stalled": int(sum(e <= 2 for e in best_eps)),
        "mean_val_auc": float(np.mean(val_aucs)),
        "did_not_train": bool(sum(e <= 2 for e in best_eps) >= 3),
        "n_params": int(count_parameters(factory())),
        "n_subjects": int(res["n_subjects"]),
        "pooled_auc_raw": float(res["metrics"]["auc"]),
        "acc_at_0.5_raw": float(res["metrics"]["acc@0.5"]),
        "seconds": round(time.time() - t0, 1),
        "experiment_dir": str(exp_dir),
    }


def add_calibrated(rows, out_root):
    """Attach leak-free calibrated AUC / accuracy from the saved artifacts."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    from p300vit.evaluation.stats import load_oof_harmonized

    for r in rows:
        try:
            d = load_oof_harmonized(Path(out_root), Path(r["experiment_dir"]).name,
                                    method="threshold")
            y, ph = d["labels"], d["probs_harm"]
            r["cal_auc"] = float(roc_auc_score(y, ph))
            r["cal_acc"] = float(accuracy_score(y, (ph >= 0.5).astype(int)))
            r["fold_mean_auc"] = float(d["fold_mean_auc"])
            r["fold_sd_auc"] = float(d["fold_sd_auc"])
            r["harmonization"] = d["harmonization"]
        except Exception as e:  # pragma: no cover
            r["cal_auc"] = None
            r["cal_acc"] = None
            r["error"] = str(e)[:200]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/matrix")
    ap.add_argument("--local-root", default=None)
    ap.add_argument("--aszed-root", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="smoke-test override")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--aug-strength", default="strong", choices=list(AUG_PRESETS))
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to cells 'cohort:montage:model:aug'")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_paths = {}
    repo = Path(__file__).resolve().parents[1]
    for cohort, spec in COHORTS.items():
        # 1) the explicit, committed copy (works inside the Kaggle code blob,
        #    whose packaging excludes any path containing a "results" component)
        # 2) the frozen v4 run config, if present
        for cand in (repo / "configs" / f"{cohort}.yaml",
                     repo / "results_v4" / "results" / "xdawn_vit" / spec["cfg_name"] / "config.yaml"):
            if Path(cand).exists():
                cfg_paths[cohort] = str(cand)
                break

    roots = {"local_root": args.local_root, "aszed_root": args.aszed_root}

    # cells
    cells = []
    for cohort, spec in COHORTS.items():
        for montage in ("full", "aep"):
            for model in ("base", "lite"):
                for aug in (False, True):
                    cells.append((cohort, montage, model, aug))
    if args.only:
        wanted = set(tuple(s.split(":")) for s in args.only)
        cells = [c for c in cells if (c[0], c[1], c[2], "aug" if c[3] else "noaug") in wanted]

    if not cells:
        raise SystemExit("[matrix] --only matched no cells; check the "
                         "'cohort:montage:model:aug' spelling (space-separated)")
    rows = []
    cohort_cache = {}
    for cohort, montage, model, aug in cells:
        if cohort not in cfg_paths:
            print(f"[matrix] no cached config for {cohort}; run the participant stage first",
                  flush=True)
            continue
        cfg = load_config(cfg_paths[cohort])
        if args.device:
            cfg.device = args.device
            if args.device == 'cpu':
                cfg.train.amp = False
            cfg.cache_dir = str(Path(args.out) / 'cache')
        cfg.experiment_name = f"{cohort}_{montage}_{model}_{'aug' if aug else 'noaug'}"

        # subjects: load once per cohort, reuse across the 8 cells
        if cohort not in cohort_cache:
            cohort_cache[cohort] = load_subjects(cfg, cohort, roots)
        subs_full = cohort_cache[cohort]
        if montage == "full":
            subs = subs_full
        else:
            subs = subset(subs_full, COHORTS[cohort]["montage_aep"])

        print(f"\n===== {cohort} | {montage} | {model} | aug={aug} "
              f"(ch={len(subs[0].ch_names)}, K={min(4, len(subs[0].ch_names))}) =====",
              flush=True)
        row = run_cell(cohort, montage, model, aug, cfg, subs, out_root, args.epochs,
                       aug_strength=args.aug_strength)
        rows.append(row)
        print(f"[matrix] {cfg.experiment_name}: best_epoch~{row['mean_best_epoch']:.1f} "
              f"(stalled {row['n_folds_stalled']}/5) raw AUC {row['pooled_auc_raw']:.4f} "
              f"| acc {row['acc_at_0.5_raw']:.4f} | {row['n_params']:,} params "
              f"| {row['seconds']/60:.1f} min", flush=True)
        (out_root / "matrix_rows.json").write_text(json.dumps(rows, indent=2))

    add_calibrated(rows, out_root)
    (out_root / "matrix_summary.json").write_text(json.dumps(rows, indent=2))

    # plain-text table
    lines = [f"{'cohort':<9}{'montage':<9}{'model':<6}{'aug':<10}{'ep~':>5}{'stall':>6}"
             f"{'params':>10}{'rawAUC':>8}{'calAUC':>8}{'calAcc':>8}{'min':>7}"]
    for r in rows:
        lines.append(
            f"{r['cohort']:<9}{r['montage']:<9}{r['model']:<6}"
            f"{(r.get('aug_strength') or 'off'):<10}"
            f"{r.get('mean_best_epoch', 0):>5.1f}{r.get('n_folds_stalled', 0):>4}/5"
            f"{r['n_params']:>10,}"
            f"{(r['pooled_auc_raw'] or 0):>8.4f}{(r.get('cal_auc') or 0):>8.4f}"
            f"{(r.get('cal_acc') or 0):>8.4f}{r['seconds']/60:>7.1f}")
    table = "\n".join(lines)
    (out_root / "matrix_table.txt").write_text(table + "\n")
    print("\n" + table, flush=True)


if __name__ == "__main__":
    main()
