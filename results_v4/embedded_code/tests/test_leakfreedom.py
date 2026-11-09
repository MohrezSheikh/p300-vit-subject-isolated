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


def test_frontend_responds_to_input():
    """The scalogram front-end must be a function of the input.

    Regression test for the [2F, 1, L] buffer-shape bug that produced a
    constant scalogram (sqrt(zf^2 + zi^2 + eps) with zf = zi = 0) — a model
    that sees a constant input yields constant predictions, zero input
    gradients, and exactly-chance AUC under every protocol.
    """
    import torch
    from p300vit.models.cwt_vit import CWTScalogram

    cwt = CWTScalogram(sfreq=200.0)
    assert tuple(cwt.kernels.shape) == (11, 1, 2, 1400), cwt.kernels.shape
    x = torch.randn(2, 4, 200, requires_grad=True)
    scal = cwt(x)
    assert float(scal.std()) > 1e-3, "scalogram is constant — front-end broken"
    (g,) = torch.autograd.grad(scal.sum(), x)
    assert float(g.abs().sum()) > 0, "no gradient reaches the input"


def test_every_band_responds_to_its_own_frequency():
    """Every declared band must respond to a sine at its own centre frequency.

    Regression test for the second front-end bug: kernels were zero-padded to
    the 1 Hz kernel length (1400) and convolved at that padded length with
    ``padding = 1400 // 2``. The wavelet support then sat outside the padded
    signal window for every band whose kernel is shorter than 1400 samples, so
    9 of the 11 bands (4-30 Hz) returned a constant and the model only ever saw
    1-2 Hz envelopes. The previous test passed with a single live band, which is
    why the defect survived.
    """
    import numpy as np
    import torch
    from p300vit.models.cwt_vit import CWTScalogram

    freqs = (1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30)
    cwt = CWTScalogram(sfreq=200.0, freqs=freqs)
    assert [int(v) for v in cwt.lengths] == [int(7.0 * 200 / f) for f in freqs]

    t = np.arange(200) / 200.0
    for i, f in enumerate(freqs):
        xs = torch.tensor(np.sin(2 * np.pi * f * t), dtype=torch.float32).view(1, 1, 200)
        with torch.no_grad():
            row = cwt(xs)[0, i]          # C == 1, so row i is band i
        assert float(row.std()) > 1e-3, f"{f} Hz band is blind to a {f} Hz sine"


def test_scalogram_patch_coverage():
    """The patch grid must consume every band row and every time sample.

    Regression test for the floor-sized grid: 44 rows // 16 = 2 patch rows and
    200 samples // 16 = 12 patch columns silently discarded the 20/24/30 Hz rows
    and the last 8 samples (760-800 ms).
    """
    import torch
    from p300vit.models.cwt_vit import CWTViT

    model = CWTViT(n_channels=4, n_times=200, sfreq=200.0)
    rows, cols = 4 * 11, 200
    expected = (-(-rows // 16)) * (-(-cols // 16))
    assert model._n_tokens == expected == 39, model._n_tokens
    assert model._padded_shape == (48, 208), model._padded_shape
    assert model._n_tokens * 16 * 16 >= rows * cols, "grid does not cover the image"

    with torch.no_grad():
        logits, _ = model(torch.randn(2, 4, 200))
    assert logits.shape == (2, 2), logits.shape


if __name__ == "__main__":
    test_factory_and_forward()
    test_xdawn_fold_dependence()
    test_projection_feeds_model()
    test_frontend_responds_to_input()
    test_every_band_responds_to_its_own_frequency()
    test_scalogram_patch_coverage()
    print("all xDAWN->CWT->ViT synthetic checks passed")
