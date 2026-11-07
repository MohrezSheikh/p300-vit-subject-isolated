#!/usr/bin/env python
"""Render the manuscript's example figures from the frozen pipeline itself.

Produces the two qualitative figures of the companion manuscript from the
EXACT representations the model consumed — no legacy images:

  fig7_xdawn.png      one representative participant's single target trial:
                      (a) 20 raw channels, (b) the K=4 xDAWN components
                      (fold-4 filters, fitted on training-fold subjects).
  fig8_scalograms.png the exact single-trial ViT inputs: scalograms of the
                      four xDAWN components for one control (top row) and
                      one schizophrenia participant (bottom row), through
                      the frozen 11-scale Morlet CWT (1-30 Hz).

Choices are deterministic (first retained participant per class, trial 0)
and recorded in figures/example_provenance.json alongside the frozen
parameters, so the figures are reproducible and the caption can state
exactly what is shown. Runs inside the p300vit-figures Kaggle kernel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p300vit.config import DatasetConfig, ExperimentConfig, set_seed  # noqa: E402
from p300vit.datasets import DATASET_REGISTRY  # noqa: E402
from p300vit.spatial.xdawn import apply_xdawn  # noqa: E402


def find_mounted(name: str) -> Path | None:
    for root, dirs, files in __import__("os").walk("/kaggle/input"):
        if name in dirs or name in files:
            return Path(root) / name
    return None


FREQS = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30)  # model CWT front-end scales


def build_cfg(local_root: str, cache_dir: str) -> ExperimentConfig:
    cfg = ExperimentConfig(
        experiment_name="figures",
        datasets=[DatasetConfig(name="sabeti-p300", root=local_root)],
        cache_dir=cache_dir,
    )
    cfg.n_channels = 20
    cfg.output_dir = "results"
    return cfg


def pick_subjects(subjects):
    control = next(s for s in subjects if s.label == 0)
    patient = next(s for s in subjects if s.label == 1)
    return control, patient


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/kaggle/working/figures")
    p.add_argument("--cache-dir", default="/kaggle/working/cache")
    p.add_argument("--local-root", default=None)
    p.add_argument("--fold", type=int, default=4, help="fold whose xDAWN filters are shown")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def find_dir(fragment: str) -> str | None:
        import glob, os
        for cand in (glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")
                     + glob.glob("/kaggle/input/*/*/*")):
            if os.path.isdir(cand) and fragment in os.path.basename(cand):
                return cand
        return None

    local_root = args.local_root or find_dir("p300-ieee") or find_dir("sabeti")
    assert local_root, "clinical dataset not mounted"
    print(f"local_root={local_root}", flush=True)

    # fold-4 xDAWN filters from the standalone kernel's saved artifacts
    suffix = f"/xdawnvit_local/fold_{args.fold}"
    for root, dirs, files in __import__("os").walk("/kaggle/input"):
        if root.replace("\\", "/").endswith(suffix) and "xdawn_W.npy" in files:
            w_path = Path(root) / "xdawn_W.npy"
            break
    if not w_path:
        import glob
        tree = sorted(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")
                      + glob.glob("/kaggle/input/*/*/*"))
        raise SystemExit(f"filters under {suffix} not found; mounted inputs:\n  "
                         + "\n  ".join(tree[:40]))
    W = np.load(w_path)
    assert W.shape[0] == 20, f"expected 20-channel clinical filters, got {W.shape}"
    print(f"filters: {w_path} -> W {W.shape}", flush=True)

    cfg = build_cfg(local_root, args.cache_dir)
    set_seed(cfg.cv.seed)
    subjects = DATASET_REGISTRY["local"](cfg)
    control, patient = pick_subjects(subjects)
    print(f"control: {control.subject_id} ({len(control.data)} trials) | "
          f"patient: {patient.subject_id} ({len(patient.data)} trials)", flush=True)

    sfreq = cfg.preprocess.sfreq_target
    t_ms = np.arange(cfg.preprocess.n_times) / sfreq * 1000.0 + cfg.preprocess.tmin * 1000.0
    onset_ms = 0.0

    # ---------------- Figure 7: before / after xDAWN, one trial ---------- #
    trial = patient.data[0]                    # [20, T] raw single trial
    projected = apply_xdawn(trial[None, ...], W)[0]   # [4, T]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
    offset = 8.0
    for c in range(trial.shape[0]):
        axes[0].plot(t_ms, trial[c] + offset * (trial.shape[0] - 1 - c),
                     lw=0.7, color="0.35")
    axes[0].set_yticks([])
    axes[0].set_ylabel("raw channels (stacked)")
    axes[0].set_title(f"(a) 20 raw channels — participant {patient.subject_id}, trial 0")
    for c in range(projected.shape[0]):
        axes[1].plot(t_ms, projected[c] + offset * (projected.shape[0] - 1 - c),
                     lw=0.9, color="#1f4e9c")
    axes[1].set_yticks([])
    axes[1].set_ylabel(f"xDAWN components (stacked), $K={{4}}$")
    axes[1].set_xlabel("Time from stimulus (ms)")
    axes[1].set_title("(b) after xDAWN spatial enhancement")
    for ax in axes:
        ax.axvline(onset_ms, color="k", ls=":", lw=0.9)
        ax.axvspan(0, 800, color="0.94", zorder=0)
    fig.tight_layout()
    fig.savefig(out / "fig7_xdawn.png", dpi=220)
    plt.close(fig)

    # ---------------- Figure 8: the exact ViT inputs --------------------- #
    import torch
    from p300vit.models.cwt_vit import CWTScalogram

    cwt = CWTScalogram(sfreq=sfreq)
    rows = [("healthy control", control), ("schizophrenia", patient)]
    scal = {}
    for name, subj in rows:
        proj = apply_xdawn(subj.data[0][None, ...], W)[0]      # [4, T]
        with torch.no_grad():
            s = cwt(torch.from_numpy(proj.astype(np.float32))[None, ...])  # [1, C*F, T]
        s = np.log1p(s[0].numpy())                             # [C*F, T], model's transform
        g = s.reshape(cwt.n_freqs, proj.shape[0], -1).transpose(1, 0, 2)  # [C, F, T]
        # per-frequency standardization FOR DISPLAY ONLY: the 1/f dynamic
        # range otherwise crushes everything above ~5 Hz into one color
        mu = g.mean(axis=2, keepdims=True)
        sd = g.std(axis=2, keepdims=True) + 1e-9
        scal[name] = (g - mu) / sd

    fig, axes = plt.subplots(2, scal["healthy control"].shape[0],
                             figsize=(11.5, 6.2), sharex=True, sharey=True)
    for r, (name, _) in enumerate(rows):
        for c in range(scal[name].shape[0]):
            ax = axes[r, c]
            im = ax.imshow(scal[name][c], origin="lower", aspect="auto",
                           extent=[t_ms[0], t_ms[-1], 1, 30], cmap="RdBu_r",
                           vmin=-3, vmax=3)
            if r == 0:
                ax.set_title(f"component {c + 1}", fontsize=9)
            if c == 0:
                ax.set_yticks(range(len(FREQS)))
                ax.set_yticklabels([str(f) for f in FREQS], fontsize=7)
                ax.set_ylabel(f"{name}\nMorlet scale (Hz)", fontsize=8)
            if r == 1:
                ax.set_xlabel("Time from stimulus (ms)")
    fig.suptitle("Single-trial scalograms — the exact ViT inputs (log-magnitude; per-frequency standardized for display)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "fig8_scalograms.png", dpi=220)
    plt.close(fig)

    provenance = {
        "figure_7": {"participant": str(patient.subject_id), "label": "schizophrenia",
                     "trial_index": 0, "filters": f"fold_{args.fold} xdawn_W.npy",
                     "n_channels_raw": int(trial.shape[0]),
                     "n_components": int(projected.shape[0])},
        "figure_8": {"control": str(control.subject_id),
                     "schizophrenia": str(patient.subject_id),
                     "trial_index": 0, "scales": list(FREQS), "note": "scales are irregularly spaced; at the lowest scales the 7-cycle wavelet spans the whole 1-s epoch, so convolution edge behavior is part of the representation",
                     "display": "log1p magnitude; per-frequency standardized for display"},
        "pipeline": "frozen preprocessing + fold-4 xDAWN + model CWT front-end",
    }
    with open(out / "example_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[figures] -> {out}/fig7_xdawn.png, fig8_scalograms.png", flush=True)

    # ---------------- Figure 9: conventional ERP grand averages --------- #
    groups = {"healthy control": [], "schizophrenia": []}
    for subj in subjects:
        evoked = subj.data.mean(axis=0)                     # [C, T] subject average
        groups["healthy control" if subj.label == 0 else "schizophrenia"].append(evoked)
    ch_names = list(subjects[0].ch_names)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), sharey=True)
    erp_stats = {}
    for ax, ch in zip(axes, ("CZ", "PZ")):
        ci = ch_names.index(ch)
        for name, evokeds in groups.items():
            arr = np.stack([e[ci] for e in evokeds])        # [n_subj, T]
            mean, sem = arr.mean(axis=0), arr.std(axis=0) / np.sqrt(len(arr))
            color = "#3f6bb5" if name == "healthy control" else "#c2504d"
            ax.plot(t_ms, mean, lw=1.3, color=color, label=name)
            ax.fill_between(t_ms, mean - sem, mean + sem, color=color, alpha=0.18)
            win = (t_ms >= 250) & (t_ms <= 500)
            pk = int(np.argmax(mean[win]))
            erp_stats.setdefault(ch, {})[name] = {
                "n_subjects": int(len(evokeds)),
                "peak_latency_ms": round(float(t_ms[win][pk]), 1),
                "peak_amplitude_z": round(float(mean[win][pk]), 3),
            }
    for ax in axes:
        ax.axvline(0, color="k", ls=":", lw=0.9)
        ax.axvspan(250, 500, color="0.92", zorder=0)
        ax.set_xlabel("Time from stimulus (ms)")
    axes[0].set_ylabel("Amplitude (z-scored a.u.)")
    axes[0].set_title("Cz"); axes[1].set_title("Pz")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Grand-average target ERPs per group (shaded: SEM across participants)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig9_erp.png", dpi=220)
    plt.close(fig)
    provenance["figure_9"] = {"channels": ["CZ", "PZ"], "groups": {k: len(v) for k, v in groups.items()},
                              "peaks": erp_stats}
    with open(out / "example_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[figures] ERP grand averages rendered -> {out}/fig9_erp.png", flush=True)


if __name__ == "__main__":
    main()
