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
        lengths = [k.shape[-1] for k in kernels]
        max_len = max(lengths)
        padded = [torch.nn.functional.pad(k, (0, max_len - k.shape[-1])) for k in kernels]
        # per-frequency complex kernel: buffer [F, 1, 2, L]; forward slices
        # kernels[f_idx] -> [1, 2, L], k[0, 0] / k[0, 1] -> the real / imag
        # vectors of length L (applied per EEG channel via groups).
        #
        # ``lengths`` records each band's *own* support length: the forward pass
        # must convolve a band at its true length, never at the padded max_len.
        # Convolving with the padded buffer shifts the wavelet support outside
        # the padded signal window and returns a constant for every band whose
        # kernel is shorter than the 1 Hz kernel -- i.e. 9 of the 11 bands
        # (4-30 Hz) carried no information. See test_frontend_responds_to_input.
        self.register_buffer(
            "kernels", torch.cat([k.unsqueeze(0) for k in padded], dim=0).unsqueeze(1)
        )  # [F, 1, 2, L]
        self.register_buffer("lengths", torch.tensor(lengths, dtype=torch.long))
        self.n_freqs = len(freqs)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, C, T] -> |CWT| scalogram [B, C*F, T] (centred 'same' conv)."""
        B, C, T = x.shape
        out = []
        for f_idx in range(self.n_freqs):
            L = int(self.lengths[f_idx])                # true kernel length
            k = self.kernels[f_idx, 0, :, :L]           # [2, L] real / imag taps
            k_even, k_odd = k[0], k[1]
            # centred 'same' padding (L//2, L-1-L//2): output length stays T and
            # output[t] is the wavelet centred on input sample t
            left, right = L // 2, L - 1 - L // 2
            xp = torch.nn.functional.pad(x, (left, right))
            # complex convolution via two real convs, grouped over channels
            zf = torch.nn.functional.conv1d(
                xp, k_even.view(1, 1, -1).repeat(C, 1, 1), groups=C
            )
            zi = torch.nn.functional.conv1d(
                xp, k_odd.view(1, 1, -1).repeat(C, 1, 1), groups=C
            )
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
        # Patchify the scalogram image: height = C*F rows, width = T samples.
        # The grid is computed with ceil and the image is replicate-padded to the
        # next multiple of ``patch`` in forward(), so that *every* band and *every*
        # time sample reaches the transformer. Sizing the grid with floor
        # (44 // 16 = 2 rows, 200 // 16 = 12 cols) silently discarded the last
        # 12 band rows (20/24/30 Hz) and the last 8 samples (760-800 ms).
        self.patch_embed = PatchEmbed2D(1, embed_dim, patch=patch)
        self.patch = patch
        grid_h = -(-in_ch // patch)                  # ceil -> 3 rows for 44
        grid_w = -(-n_times // patch)                # ceil -> 13 cols for 200
        n_tokens = grid_h * grid_w                   # 39 tokens (+ CLS = 40)
        self._grid = (grid_h, grid_w)
        self._padded_shape = (grid_h * patch, grid_w * patch)  # (48, 208)
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
        # replicate-pad to the next patch multiple (44x200 -> 48x208) so the
        # patchify step consumes all rows and all samples
        H, W = scal.shape[-2:]
        pad_h, pad_w = (-H) % self.patch, (-W) % self.patch
        if pad_h or pad_w:
            scal = torch.nn.functional.pad(scal, (0, pad_w, 0, pad_h), mode="replicate")
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
