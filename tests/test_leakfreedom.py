"""Synthetic wiring and leak-freedom checks for the xDAWN->CWT->ViT pipeline.

Runs on CPU with random arrays only — no datasets, no training. Verifies:

1. the model factory declares ``xdawn_mode == "xdawn"``;
2. the composed model builds at K = cfg.xdawn.n_components and its forward
   pass produces [B, 2] logits on the projected epoch shape;
3. xDAWN filters are fold-dependent (the leakage contract): fitting on
   different training-subject sets yields different projections;
4. ``apply_xdawn`` output feeds the model with the expected shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # repo root (scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # package

from p300vit.config import ExperimentConfig  # noqa: E402
from p300vit.spatial.xdawn import apply_xdawn, fit_xdawn  # noqa: E402
from scripts.run_all import make_model  # noqa: E402


def _synthetic_epochs(n_subjects: int = 12, n_trials: int = 15,
                      n_channels: int = 19, n_times: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    epochs = []
    for i in range(n_subjects):
        x = rng.normal(0.0, 1.0, size=(n_trials, n_channels, n_times)).astype(np.float32)
        x += (i % 2) * 0.3          # class-dependent tilt so xDAWN has signal
        epochs.append(x)
    return epochs


def test_factory_and_forward():
    cfg = ExperimentConfig(cache_dir="/tmp/p300vit_test")
    factory, xdawn_mode = make_model(cfg)
    assert xdawn_mode == "xdawn"
    model = factory()
    x = torch.randn(3, cfg.xdawn.n_components, cfg.preprocess.n_times)
    logits, feats = model(x)
    assert logits.shape == (3, 2), logits.shape


def test_xdawn_fold_dependence():
    cfg = ExperimentConfig(cache_dir="/tmp/p300vit_test")
    epochs = _synthetic_epochs()
    w_a = fit_xdawn(epochs[:8], cfg.xdawn)     # fold A training subjects
    w_b = fit_xdawn(epochs[4:], cfg.xdawn)     # fold B training subjects
    assert w_a.shape == (19, cfg.xdawn.n_components)
    assert not np.allclose(w_a, w_b), "xDAWN filters identical across folds -> leakage"


def test_projection_feeds_model():
    cfg = ExperimentConfig(cache_dir="/tmp/p300vit_test")
    factory, _ = make_model(cfg)
    model = factory()
    epochs = _synthetic_epochs(n_subjects=6)
    w = fit_xdawn(epochs[:4], cfg.xdawn)
    projected = apply_xdawn(epochs[4], w)
    assert projected.shape == (epochs[4].shape[0], cfg.xdawn.n_components,
                               cfg.preprocess.n_times)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(projected[:5]))
    assert logits.shape == (5, 2)


if __name__ == "__main__":
    test_factory_and_forward()
    test_xdawn_fold_dependence()
    test_projection_feeds_model()
    print("all xDAWN->CWT->ViT synthetic checks passed")
