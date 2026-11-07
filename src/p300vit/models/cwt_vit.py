"""Real CWT 2D-ViT baseline: continuous wavelet scalograms -> Vision
Transformer (the "EEG-as-an-image" paradigm the paper argues against).

The Morlet analysis is implemented as grouped 1-D convolutions with fixed
complex wavelet kernels (fast, differentiable, dependency-free), producing a
per-trial scalogram tensor [B, C*F, T] which is patchified 2-D and encoded
by a compact ViT. The model is sized for a single Kaggle T4 (16 GB).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def morlet_kernel(freq: float, sfreq: float, n_cycles: float = 7.0) -> torch.Tensor:
    """Complex Morlet kernel sampled at ``sfreq`` (length ~ n_cycles/freq)."""
    n = int(n_cycles * sfreq / freq)
    t = (np.arange(n) - (n - 1) / 2) / sfreq
    sigma = n_cycles / (2 * np.pi * freq)
    wavelet = np.exp(2j * np.pi * freq * t) * np.exp(-(t**2) / (2 * sigma**2))
    wavelet /= np.sqrt(sigma * np.sqrt(np.pi))
    return torch.tensor(np.stack([wavelet.real, wavelet.imag]), dtype=torch.float32)


class CWTScalogram(nn.Module):
    """Fixed-weight Morlet CWT front-end: [B, C, T] -> [B, C*F, T']."""

    def __init__(self, sfreq: float = 200.0, freqs=(1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30)):
        super().__init__()
        kernels = [morlet_kernel(f, sfreq) for f in freqs]
        max_len = max(k.shape[-1] for k in kernels)
        padded = [torch.nn.functional.pad(k, (0, max_len - k.shape[-1])) for k in kernels]
        # weight tensor [C*F, 2, L]; applied per EEG channel via groups
        self.register_buffer(
            "kernels", torch.cat(padded, dim=0).unsqueeze(1)
        )  # [F, 1, 2, L] placeholder shape
        self.n_freqs = len(freqs)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, C, T] -> |CWT| scalogram [B, C*F, T] (same length, 'same' conv)."""
        B, C, T = x.shape
        pad = self.max_len // 2
        out = []
        for f_idx in range(self.n_freqs):
            k = self.kernels[f_idx]                     # [1, 2, L]
            L = k.shape[-1]
            k_even, k_odd = k[0, 0], k[0, 1]
            # complex convolution via two real convs, grouped over channels
            zf = torch.nn.functional.conv1d(
                x, k_even.view(1, 1, -1).repeat(C, 1, 1), padding=L // 2, groups=C
            )
            zi = torch.nn.functional.conv1d(
                x, k_odd.view(1, 1, -1).repeat(C, 1, 1), padding=L // 2, groups=C
            )
            # 'same' length correction when L is even
            if zf.shape[-1] > T:
                zf, zi = zf[..., :T], zi[..., :T]
            out.append(torch.sqrt(zf**2 + zi**2 + 1e-12))
        return torch.cat(out, dim=1)                    # [B, C*F, T]


class PatchEmbed2D(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.patch = patch

    def forward(self, x):
        tokens = self.proj(x)                           # [B, D, H', W']
        return tokens.flatten(2).transpose(1, 2)        # [B, H'*W', D]


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class CWTViT(nn.Module):
    """The old-paradigm baseline. Input: raw epochs [B, C, T] (no xDAWN)."""

    def __init__(
        self,
        n_channels: int = 20,
        n_times: int = 200,
        sfreq: float = 200.0,
        freqs=(1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 30),
        patch: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cwt = CWTScalogram(sfreq=sfreq, freqs=freqs)
        in_ch = n_channels * len(freqs)
        # patchify the scalogram image: height = C*F rows, width = T samples
        self.patch_embed = PatchEmbed2D(1, embed_dim, patch=patch)
        W = max(1, n_times // patch)
        n_tokens = (in_ch // patch) * W
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, heads, drop=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 2)
        self._n_tokens = n_tokens

    def forward(self, x):
        scal = self.cwt(x)                                  # [B, C*F, T]
        scal = torch.log1p(scal)                            # log-compression
        tokens = self.patch_embed(scal.unsqueeze(1))        # [B, n_tokens, D]
        tokens = tokens[:, : self._n_tokens]
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(self.norm(tokens[:, 0])), self.norm(tokens[:, 0])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
