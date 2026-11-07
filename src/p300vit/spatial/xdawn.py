"""Leak-free xDAWN spatial filtering.

Closed-form supervised spatial filter maximising the Signal-to-Signal-plus-
Noise Ratio of the evoked response (Rivet et al., 2009), fitted via
generalized eigenvalue decomposition of the evoked covariance against the
regularised total covariance.

LEAKAGE CONTRACT: ``fit`` must only ever receive epochs from TRAINING
subjects. The CV harness enforces this; ``check_leak_freedom`` in
tests verifies filters differ between folds.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

from ..config import XDAWNConfig


def fit_xdawn(
    train_epochs: list[np.ndarray],
    cfg: XDAWNConfig,
) -> np.ndarray:
    """Fit xDAWN on training-fold epochs only.

    Parameters
    ----------
    train_epochs : list of [n_trials, n_channels, n_times] arrays, one per
        training subject.
    cfg : XDAWNConfig

    Returns
    -------
    W : [n_channels, n_components] projection matrix (columns ordered by
        decreasing eigenvalue).
    """
    X = np.concatenate(train_epochs, axis=0).astype(np.float64)  # [N, C, T]
    # evoked covariance: template ERP against itself (stimulus-locked SSNR)
    template = X.mean(axis=0)                    # [C, T]
    Sigma_S = template @ template.T              # [C, C]

    # total covariance, regularised by trace-shrinkage for numerical rank
    Sigma_X = np.einsum("nct,ndt->cd", X, X) / X.shape[0]
    Sigma_X += cfg.shrinkage * np.trace(Sigma_X) * np.eye(Sigma_X.shape[0])

    evals, evecs = scipy.linalg.eigh(Sigma_S, Sigma_X)
    order = np.argsort(evals)[::-1][: cfg.n_components]
    W = evecs[:, order].astype(np.float32)
    return W


def apply_xdawn(epochs: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project epochs [n_trials, C, T] onto spatial filters -> [n_trials, K, T]."""
    return np.einsum("nct,ck->nkt", epochs, W).astype(np.float32)
