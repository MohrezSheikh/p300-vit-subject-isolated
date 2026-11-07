"""Dataset loaders: clinical (Sabeti .cnt), ASZED-153 (public), pooled.

Vendored from the benchmark's registry so this repository is fully
self-contained; identical loading and channel-unification semantics.
"""
from __future__ import annotations

from .config import DatasetConfig, ExperimentConfig


def _select_dataset(cfg: ExperimentConfig, name: str) -> DatasetConfig:
    """Pick a dataset config by name (fall back to the first entry)."""
    for ds in cfg.datasets:
        if ds.name == name:
            return ds
    return cfg.datasets[0]


def load_local(cfg: ExperimentConfig):
    from .data.cnt_dataset import load_dataset

    ds = _select_dataset(cfg, "sabeti-p300")
    return load_dataset(ds, cfg.preprocess, cache_dir=cfg.cache_dir)


def load_aszed(cfg: ExperimentConfig):
    from .data.aszed_dataset import load_dataset

    ds = _select_dataset(cfg, "aszedeeg")
    return load_dataset(ds, cfg.preprocess, cache_dir=cfg.cache_dir)


def load_pooled(cfg: ExperimentConfig):
    """Local + ASZED restricted to the 19 common channels, site-labelled."""
    from .data.aszed_dataset import COMMON_CHANNELS
    from .data.cnt_dataset import SCALP_CHANNELS
    from .data.cnt_dataset import load_dataset as load_cnt
    from .data.aszed_dataset import load_dataset as load_aszed_raw

    local = load_cnt(cfg.datasets[0], cfg.preprocess, cache_dir=cfg.cache_dir)
    aszed = load_aszed_raw(cfg.datasets[1], cfg.preprocess, cache_dir=cfg.cache_dir)
    idx = [SCALP_CHANNELS.index(ch) for ch in COMMON_CHANNELS]
    for s in local:
        s.data = s.data[:, idx, :]
        s.ch_names = list(COMMON_CHANNELS)
        s.site = "sabeti"
    for s in aszed:
        s.site = "aszed"
    return local + aszed


DATASET_REGISTRY: dict = {
    "local": load_local,
    "aszed": load_aszed,
    "pooled": load_pooled,
}
