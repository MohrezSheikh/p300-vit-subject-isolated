"""Central experiment configuration and deterministic seeding.

Every hyperparameter, path, and protocol constant lives here or in a YAML
config file -- never inline in model/training code. This is the single source
of truth for reproducibility.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

try:                      # torch is only needed for the deep models; the
    import torch          # classic baselines run in CPU-only environments
    _TORCH_OK = True
except ImportError:       # pragma: no cover
    torch = None
    _TORCH_OK = False


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that influences the pipeline and force determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _TORCH_OK:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_dataloader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Dataset configs
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    """Paths and layout of one EEG acquisition dataset.

    ``root`` may point at a Kaggle input mount or a local directory; the
    adapters resolve subject files relative to it.
    """

    name: str
    root: str
    # 10-20 channel order used across the whole project. Datasets expose a
    # subset of these; the adapter aligns and reorders to this canonical list.
    channels: List[str] = field(default_factory=lambda: [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
        "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2", "FPZ",
    ])
    sfreq: float = 200.0
    label_map: dict = field(default_factory=lambda: {"control": 0, "schizophrenia": 1})

    def resolve(self) -> Path:
        """Return the first existing candidate root (Kaggle mount or local)."""
        candidates = [Path(self.root)]
        # Kaggle sometimes mounts as /kaggle/input/<slug> without "datasets/"
        stripped = str(self.root).replace("/datasets/", "/")
        if stripped != self.root:
            candidates.append(Path(stripped))
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"Dataset '{self.name}' root not found. Tried: "
            + ", ".join(str(c) for c in candidates)
        )


@dataclass
class PreprocessConfig:
    """Shared EEG preprocessing (identical for every model and dataset)."""
    l_freq: float = 0.5
    h_freq: float = 30.0
    notch_freqs: List[float] = field(default_factory=lambda: [50.0])
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple = (-0.2, 0.0)
    reject_uv: float = 120.0
    # "absolute" = legacy peak-to-peak threshold (reject_uv) used for the
    # local .cnt dataset; "robust" = scale-free rejection (robust_k x median
    # epoch ptp of the same recording) required for ASZED whose amplitudes
    # are ~10x larger.
    artifact_rejection: str = "absolute"
    robust_k: float = 5.0
    # channels with std below this (uV) are flagged as flat/dead
    flat_uv: float = 5.0
    n_times: int = 200          # T = 200 samples at 200 Hz (1 s epoch)
    # target sampling rate -- ASZED files arrive at 256/200/100 Hz and are
    # resampled to this before filtering
    sfreq_target: float = 200.0
    # per-trial z-score applied to every epoch (no fold statistics involved)
    trial_zscore: bool = True


@dataclass
class XDAWNConfig:
    n_components: int = 4
    shrinkage: float = 1e-4


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 8            # early stopping on validation subject-AUC
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    # fraction of *training subjects* held out inside each fold for model
    # selection + threshold tuning (subject-level, stratified)
    val_frac_subjects: float = 0.2
    amp: bool = True
    num_workers: int = 2


@dataclass
class CVConfig:
    n_folds: int = 5
    seed: int = 42
    # optional site stratification key for pooled runs ("site" column in manifest)
    stratify_by: str = "label"   # label | site | site_label


@dataclass
class ExperimentConfig:
    """Top-level experiment description: which datasets, models, and outputs."""
    experiment_name: str = "moe_p300vit_main"
    datasets: List[DatasetConfig] = field(default_factory=list)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    xdawn: XDAWNConfig = field(default_factory=XDAWNConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    output_dir: str = "results"
    figures_dir: str = "figures"
    cache_dir: str = "/tmp/p300_cache"      # preprocessed-epoch cache
    device: str = ("cuda" if (_TORCH_OK and torch.cuda.is_available()) else "cpu")
    bootstrap_iters: int = 10000
    non_inferiority_margin: float = 0.05   # AUC margin vs foundation model
    # set by the runner: input channels after dataset/channel reconciliation
    # (20 for local-only runs, 19 for pooled/ASZED runs)
    n_channels: int = 20

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from YAML, overriding dataclass defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    datasets = [
        DatasetConfig(**ds) for ds in raw.pop("datasets", [])
    ]
    preprocess = PreprocessConfig(**raw.pop("preprocess", {}))
    preprocess.baseline = tuple(preprocess.baseline)
    xdawn = XDAWNConfig(**raw.pop("xdawn", {}))
    train = TrainConfig(**raw.pop("train", {}))
    cv = CVConfig(**raw.pop("cv", {}))

    cfg = ExperimentConfig(
        datasets=datasets,
        preprocess=preprocess,
        xdawn=xdawn,
        train=train,
        cv=cv,
        **raw,
    )
    return cfg


# --------------------------------------------------------------------------- #
# Default dataset roots (Kaggle mounts provided by the user; overridable in YAML)
# --------------------------------------------------------------------------- #
LOCAL_P300_IEEE = "/kaggle/input/datasets/mmdation/p300-ieee"
ASZED_ROOT = "/kaggle/input/datasets/mmdation/aszedeeg"


def default_local_dataset() -> DatasetConfig:
    return DatasetConfig(name="p300-ieee", root=LOCAL_P300_IEEE)


def default_aszed_dataset(subset: str = "oddball") -> DatasetConfig:
    return DatasetConfig(name="aszedeeg", root=ASZED_ROOT)
