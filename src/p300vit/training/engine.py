"""Training engine: subject-balanced training, early stopping, checkpointing.

Protocol notes
--------------
* Trials inherit their subject's label (trial-level pseudo-replication);
  subject-balanced sampling prevents high-trial-count subjects from
  dominating each batch.
* Model selection uses an inner validation split at the SUBJECT level
  (held-out training subjects), never the test subjects.
* The decision threshold is chosen on the inner validation set (Youden J)
  and reported alongside the raw probabilities so any threshold can be
  re-applied post hoc.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..config import ExperimentConfig, make_dataloader_generator, seed_worker


class EEGTrials(Dataset):
    """Trial-level dataset: X [N, C, T] float32, y [N] int64."""

    def __init__(self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subject_ids[idx]


def subject_balanced_loader(
    X, y, subject_ids, batch_size: int, seed: int, shuffle: bool = True
) -> DataLoader:
    """Loader in which every subject contributes equally per epoch."""
    _, counts = np.unique(subject_ids, return_counts=True)
    w = np.array([1.0 / counts[list(np.unique(subject_ids)).index(s)] for s in subject_ids])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(w), num_samples=len(y), replacement=True,
        generator=make_dataloader_generator(seed),
    )
    return DataLoader(
        EEGTrials(X, y, subject_ids),
        batch_size=batch_size,
        sampler=sampler if shuffle else None,
        shuffle=False if shuffle else True,
        num_workers=0,
        worker_init_fn=seed_worker,
    )


def plain_loader(X, y, subject_ids, batch_size: int) -> DataLoader:
    return DataLoader(
        EEGTrials(X, y, subject_ids), batch_size=batch_size, shuffle=False, num_workers=0
    )


@torch.no_grad()
def predict_trials(
    model: nn.Module, loader: DataLoader, device: str, amp: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (trial_probs, trial_labels, trial_subject_ids)."""
    model.eval()
    probs, labels, sids = [], [], []
    for X, y, s in loader:
        X = X.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], enabled=amp and "cuda" in device):
            logits, _ = model(X)
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
        sids.append(np.asarray(s))
    return (
        np.concatenate(probs),
        np.concatenate(labels),
        np.concatenate(sids),
    )


def subject_probs(trial_probs: np.ndarray, subject_ids: np.ndarray) -> Dict[str, float]:
    """Soft voting: subject probability = mean of its trial probabilities."""
    out = {}
    for s in np.unique(subject_ids):
        out[str(s)] = float(trial_probs[subject_ids == s].mean())
    return out


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity on a validation set."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, probs)
    return float(thr[np.argmax(tpr - fpr)])


def train_model(
    model: nn.Module,
    cfg: ExperimentConfig,
    X_train, y_train, sid_train,
    X_val, y_val, sid_val,
    fold_dir: Optional[Path] = None,
    fold_seed: int = 0,
) -> dict:
    """Train with early stopping on validation subject-level AUC.

    Returns a dict with the best state dict (on CPU), history, and the
    Youden threshold derived from the validation subjects.
    """
    from sklearn.metrics import roc_auc_score

    device = cfg.device
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.max_epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and "cuda" in device)

    train_loader = subject_balanced_loader(
        X_train, y_train, sid_train, cfg.train.batch_size,
        seed=cfg.cv.seed + fold_seed, shuffle=True,
    )
    val_loader = plain_loader(X_val, y_val, sid_val, cfg.train.batch_size)

    best_auc, best_state, best_epoch, patience_left = -1.0, None, -1, cfg.train.patience
    history = []

    for epoch in range(cfg.train.max_epochs):
        model.train()
        running_loss, n_batches = 0.0, 0
        for X, y, _ in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.split(":")[0],
                enabled=cfg.train.amp and "cuda" in device,
            ):
                logits, _ = model(X)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            running_loss += float(loss.item())
            n_batches += 1
        sched.step()

        # subject-level validation AUC
        vp, vy, _ = predict_trials(model, val_loader, device, cfg.train.amp)
        val_sids = np.unique(sid_val)
        sid_to_prob = subject_probs(vp, sid_val)
        val_subj_probs = np.array([sid_to_prob[str(s)] for s in val_sids])
        val_subj_labels = np.array([y_val[sid_val == s][0] for s in val_sids])
        val_auc = roc_auc_score(val_subj_labels, val_subj_probs)

        history.append(
            {"epoch": epoch, "loss": running_loss / max(n_batches, 1), "val_auc": val_auc}
        )
        if val_auc > best_auc:
            best_auc, best_epoch, patience_left = val_auc, epoch, cfg.train.patience
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Youden threshold on validation subjects with the best model
    vp, vy, _ = predict_trials(model, val_loader, device, cfg.train.amp)
    sid_to_prob = subject_probs(vp, sid_val)
    val_sids = np.unique(sid_val)
    val_subj_probs = np.array([sid_to_prob[str(s)] for s in val_sids])
    val_subj_labels = np.array([y_val[sid_val == s][0] for s in val_sids])
    threshold = youden_threshold(val_subj_labels, val_subj_probs)

    result = {
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "threshold": threshold,
        "history": history,
    }
    if fold_dir is not None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / "model.pt")
        with open(fold_dir / "train_history.json", "w") as f:
            json.dump(result, f, indent=2)
    return result
