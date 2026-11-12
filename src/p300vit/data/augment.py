"""Training-only augmentation for xDAWN-projected ERP epochs.

LEAKAGE CONTRACT: ``EpochAugment`` is applied to *training batches only*,
inside the fold, after the per-trial z-scoring and after the training-fold
xDAWN projection. Validation and test arrays are never passed through it, and
no statistic is estimated from them. ``mixup`` mixes trials of the *same class*
only, so the label stays valid.

Rationale for the choice of transforms: single-trial ERPs are short (1 s) and
the discriminative structure is a low-frequency, stimulus-locked waveform, so
label-preserving transforms must not destroy latency or polarity. We therefore
use (a) amplitude scaling, (b) small time shifts (+/-10 ms), (c) additive
band-limited Gaussian noise at a target SNR, (d) same-class mixup, and
(e) channel dropout. SpecAugment-style time/frequency masking of the scalogram
is implemented in the model (``CWTViT(mask_scalogram=True)``) because the
scalogram is produced inside the forward pass.

Generative (GAN/diffusion) synthesis is deliberately *not* used: a generator
trained on the full cohort is a leakage channel (it encodes held-out subjects),
and synthetic trials cannot add information about unseen participants. See the
manuscript's discussion of augmentation.
"""
from __future__ import annotations

import numpy as np
import torch


class EpochAugment:
    """Per-batch, label-preserving augmentation of projected ERP epochs.

    Parameters
    ----------
    p_noise : probability of adding Gaussian noise to a trial.
    snr_db : target signal-to-noise ratio for the noise injection (per trial).
    p_scale : probability of applying a global amplitude scale.
    scale : (low, high) uniform range for the amplitude scale.
    p_shift : probability of a circular time shift.
    max_shift : maximum shift in samples (2 samples = 10 ms at 200 Hz).
    p_mixup : probability of mixing a trial with another trial of its class.
    mixup_alpha : Beta(a, a) shape for the mixup coefficient.
    p_chan_drop : probability of dropping one input channel/component per trial.
    seed : RNG seed (fold-dependent in the harness).
    """

    def __init__(
        self,
        p_noise: float = 0.5,
        snr_db: float = 15.0,
        p_scale: float = 0.5,
        scale: tuple[float, float] = (0.9, 1.1),
        p_shift: float = 0.5,
        max_shift: int = 2,
        p_mixup: float = 0.3,
        mixup_alpha: float = 0.2,
        p_chan_drop: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.p_noise = p_noise
        self.snr_db = snr_db
        self.p_scale = p_scale
        self.scale = scale
        self.p_shift = p_shift
        self.max_shift = max_shift
        self.p_mixup = p_mixup
        self.mixup_alpha = mixup_alpha
        self.p_chan_drop = p_chan_drop
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def __call__(self, X: torch.Tensor, y: torch.Tensor):
        """X [B, C, T] float tensor, y [B] long tensor -> (X', y')."""
        X = X.clone()
        y = y.clone()
        B, C, T = X.shape
        device = X.device

        # (a) amplitude scaling
        for i in range(B):
            if self.rng.random() < self.p_scale:
                s = self.rng.uniform(*self.scale)
                X[i] = X[i] * s

        # (b) circular time shift (keeps the epoch length and the P300 intact
        # for small shifts; 2 samples = 10 ms)
        if self.max_shift > 0:
            for i in range(B):
                if self.rng.random() < self.p_shift:
                    k = int(self.rng.integers(-self.max_shift, self.max_shift + 1))
                    if k:
                        X[i] = torch.roll(X[i], shifts=k, dims=-1)

        # (c) additive band-limited Gaussian noise at a target SNR
        for i in range(B):
            if self.rng.random() < self.p_noise:
                p_sig = float((X[i] ** 2).mean())
                if p_sig > 0:
                    p_noise = p_sig / (10.0 ** (self.snr_db / 10.0))
                    noise = torch.randn(C, T, device=device) * np.sqrt(p_noise)
                    X[i] = X[i] + noise

        # (d) same-class mixup: only trials with the same label are mixed
        if self.p_mixup > 0 and B > 1:
            lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
            y_np = y.detach().cpu().numpy()
            for cls in np.unique(y_np):
                idx = np.where(y_np == cls)[0]
                if len(idx) < 2:
                    continue
                partner = self.rng.permutation(idx)
                for src, dst in zip(idx, partner):
                    if src != dst and self.rng.random() < self.p_mixup:
                        X[src] = lam * X[src] + (1.0 - lam) * X[dst]

        # (e) channel / component dropout
        if self.p_chan_drop > 0:
            for i in range(B):
                if C > 1 and self.rng.random() < self.p_chan_drop:
                    c = int(self.rng.integers(0, C))
                    X[i, c] = 0.0

        return X, y


def augment_noop(X, y):
    """Identity augmentation (used for the no-augmentation arm)."""
    return X, y
