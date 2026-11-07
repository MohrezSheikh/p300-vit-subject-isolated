"""Statistics computed from saved OOF prediction artifacts.

All hypothesis tests and intervals here take observed predictions/probabilities
as inputs -- nothing is hand-typed. Outputs feed Table 1 and the figures.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact binomial CI for a proportion."""
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def _auc_from_ranks(probs_mat: np.ndarray, y_mat: np.ndarray) -> np.ndarray:
    """Vectorized AUC per row via the Mann-Whitney rank formula.

    AUC = (sum of ranks of positives - n_pos(n_pos+1)/2) / (n_pos * n_neg)

    Uses average ranks within tied values so the result matches
    ``sklearn.roc_auc_score`` exactly, including on the (rare) resamples
    that contain ties.
    """
    from scipy.stats import rankdata
    B, n = probs_mat.shape
    # ordinal ranks via double argsort (stable); then average within ties
    order = np.argsort(probs_mat, axis=1, kind="stable")
    ordinal = np.empty_like(order)
    np.put_along_axis(ordinal, order, np.broadcast_to(
        np.arange(n), (B, n)).copy(), axis=1)
    sorted_vals = np.take_along_axis(probs_mat, order, axis=1)
    # rows with ties on ANY feature need average ranks (rare): loop only those
    tie_rows = np.zeros(B, dtype=bool)
    if n > 1:
        tie_rows = (sorted_vals[:, 1:] == sorted_vals[:, :-1]).any(axis=1)
    ranks = (ordinal + 1).astype(np.float64)
    if tie_rows.any():
        for b in np.nonzero(tie_rows)[0]:
            ranks[b] = rankdata(probs_mat[b], method="average")
    pos = y_mat == 1
    n_pos = pos.sum(axis=1).astype(np.float64)
    n_neg = n - n_pos
    sum_ranks = (ranks * pos).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sum_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def bootstrap_auc_ci(y: np.ndarray, probs: np.ndarray, iters: int = 10000,
                     alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    """Point AUC + percentile-bootstrap CI (resamples subjects).

    Vectorized: all resamples are drawn and scored in one pass, which is
    ~100x faster than the per-resample sklearn loop and gives identical
    CIs for tie-free probabilities.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=np.float64)
    n = len(y)
    point = roc_auc_score(y, probs)
    idx = rng.integers(0, n, size=(iters, n))
    aucs = _auc_from_ranks(probs[idx], y[idx])
    aucs = aucs[~np.isnan(aucs)]
    return float(point), float(np.quantile(aucs, alpha / 2)), float(
        np.quantile(aucs, 1 - alpha / 2))


def paired_bootstrap_auc_diff(
    y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray,
    iters: int = 10000, alpha: float = 0.05, seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI + two-sided permutation p for AUC(a) - AUC(b), paired."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diff_point = roc_auc_score(y, probs_a) - roc_auc_score(y, probs_b)

    boots = np.empty(iters)
    for b in range(iters):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            boots[b] = np.nan
            continue
        boots[b] = roc_auc_score(y[idx], probs_a[idx]) - roc_auc_score(y[idx], probs_b[idx])
    boots = boots[~np.isnan(boots)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])

    # permutation test: swap paired predictions within resamples
    perm = np.empty(iters)
    signs = rng.integers(0, 2, size=(iters, n))
    for b in range(iters):
        swap = signs[b].astype(bool)
        pa, pb = probs_a.copy(), probs_b.copy()
        pa[swap], pb[swap] = probs_b[swap], probs_a[swap]
        if len(np.unique(y)) < 2:
            perm[b] = np.nan
            continue
        perm[b] = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    perm = perm[~np.isnan(perm)]
    p_val = float((np.sum(np.abs(perm) >= abs(diff_point)) + 1) / (len(perm) + 1))
    return {"auc_diff": float(diff_point), "ci_low": float(lo), "ci_high": float(hi),
            "perm_p": p_val}


def non_inferiority(y, probs_new, probs_ref, margin: float = 0.05,
                    iters: int = 10000, seed: int = 42) -> Dict[str, float]:
    """One-sided test: AUC(new) - AUC(ref) > -margin (bootstrap lower bound)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(iters)
    for b in range(iters):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            boots[b] = np.nan
            continue
        boots[b] = roc_auc_score(y[idx], probs_new[idx]) - roc_auc_score(y[idx], probs_ref[idx])
    boots = boots[~np.isnan(boots)]
    lower = float(np.quantile(boots, 0.05))
    return {
        "auc_diff": float(roc_auc_score(y, probs_new) - roc_auc_score(y, probs_ref)),
        "ci95_lower": lower,
        "margin": margin,
        "non_inferior": bool(lower > -margin),
    }


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #
def mcnemar_exact(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired subject-level predictions."""
    ca, cb = pred_a == y, pred_b == y
    n01 = int(np.sum(~ca & cb))   # a wrong, b right
    n10 = int(np.sum(ca & ~cb))   # a right, b wrong
    discordant = n01 + n10
    if discordant == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p": 1.0}
    k = min(n01, n10)
    p = float(stats.binomtest(k, discordant, 0.5).pvalue * 2)
    return {"n01": n01, "n10": n10, "statistic": float(max(n01, n10)), "p": min(1.0, p)}


def holm_bonferroni(pvals: list) -> list:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running = max(running, adj)
        adjusted[idx] = min(1.0, running)
    return [float(a) for a in adjusted]


# --------------------------------------------------------------------------- #
# Calibration & decision analysis
# --------------------------------------------------------------------------- #
def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    bins = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    ece, n = 0.0, len(y)
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        conf, acc = probs[mask].mean(), y[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def calibration_summary(y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, probs)),
        "log_loss": float(log_loss(y, np.clip(probs, 1e-7, 1 - 1e-7))),
        "ece_15bin": expected_calibration_error(y, probs),
        "auc_pr": float(average_precision_score(y, probs)),
    }


def decision_curve_analysis(
    y: np.ndarray, probs: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """Net benefit of the model vs treat-all / treat-none across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y)
    nb_model, nb_all, nb_none = [], [], []
    for t in thresholds:
        pred_pos = probs >= t
        tp = int(np.sum(pred_pos & (y == 1)))
        fp = int(np.sum(pred_pos & (y == 0)))
        nb_model.append(tp / n - fp / n * (t / (1 - t)))
        nb_all.append((y == 1).sum() / n - (y == 0).sum() / n * (t / (1 - t)))
        nb_none.append(0.0)
    return {
        "thresholds": thresholds,
        "net_benefit_model": np.array(nb_model),
        "net_benefit_all": np.array(nb_all),
        "net_benefit_none": np.array(nb_none),
    }


def trial_budget_curve(
    trial_probs_by_subject: Dict[str, np.ndarray],
    labels_by_subject: Dict[str, int],
    budgets,
    iters: int = 200,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Metrics vs number of trials used in soft voting (protocol shortening).

    For each budget k, k trials per subject are sampled (with replacement
    across draws, without within a draw), soft-voted, and metrics computed.
    """
    rng = np.random.default_rng(seed)
    sids = sorted(trial_probs_by_subject)
    out = {"acc": [], "auc": [], "sens": [], "spec": []}
    from sklearn.metrics import roc_auc_score

    for k in budgets:
        accs, aucs, senss, specs = [], [], [], []
        for _ in range(iters):
            probs, ys = [], []
            for s in sids:
                p = trial_probs_by_subject[s]
                idx = rng.choice(len(p), size=min(k, len(p)), replace=False)
                probs.append(p[idx].mean())
                ys.append(labels_by_subject[s])
            probs, ys = np.array(probs), np.array(ys)
            preds = (probs >= 0.5).astype(int)
            accs.append((preds == ys).mean())
            if len(np.unique(ys)) == 2:
                aucs.append(roc_auc_score(ys, probs))
            senss.append(preds[ys == 1].mean())
            specs.append((preds[ys == 0] == 0).mean())
        out["acc"].append(float(np.mean(accs)))
        out["auc"].append(float(np.mean(aucs)))
        out["sens"].append(float(np.mean(senss)))
        out["spec"].append(float(np.mean(specs)))
    return {k2: np.array(v) for k2, v in out.items()}


def full_metrics(y: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """Standard metric battery + exact CIs for a fixed threshold."""
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    sens = tp / n_pos if n_pos else float("nan")
    spec = tn / n_neg if n_neg else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else float("nan")
    auc, auc_lo, auc_hi = bootstrap_auc_ci(y, probs)

    def ci(k, n):
        lo, hi = clopper_pearson(k, n)
        return {"point": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi}

    return {
        "n_subjects": len(y),
        "accuracy": ci(tp + tn, len(y)),
        "sensitivity": ci(tp, n_pos),
        "specificity": ci(tn, n_neg),
        "precision": {"point": prec, "ci_low": None, "ci_high": None},
        "f1": {"point": f1, "ci_low": None, "ci_high": None},
        "auc": {"point": auc, "ci_low": auc_lo, "ci_high": auc_hi},
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "calibration": calibration_summary(y, probs),
    }


# --------------------------------------------------------------------------- #
# Fold-harmonized pooling (leak-free)
# --------------------------------------------------------------------------- #
def rank_normalize(x: np.ndarray) -> np.ndarray:
    """Average-rank transform to [0, 1].

    NOTE: applying this to a TEST fold uses the test fold's own predictions
    to build the transform -> test-fold leakage. Kept only for the explicit
    leaky-vs-leak-free audit comparison; new code must use calibrate_fold().
    """
    from scipy.stats import rankdata
    r = rankdata(x, method="average")
    return (r - 0.5) / len(x)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def calibrate_fold(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    method: str = "threshold",
    threshold: Optional[float] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Leak-free per-fold alignment of test probabilities.

    Uses ONLY inner-validation subjects (never the test fold) to estimate the
    transform. All transforms are strictly monotone within a fold, so fold AUC
    is unchanged; only cross-fold comparability (pooled AUC, pooled metrics,
    threshold transfer) is affected.

    Methods
    -------
    threshold : shift logits so the validation Youden threshold maps to 0.5
                (monotone, most stable on tiny validation sets; leak-free).
    zval      : z-anchor validation logits (mean/std) and map test logits
                through the same affine transform (monotone; leak-free).
    platt     : Platt scaling fitted on validation logits (LR, monotone).
    temp      : temperature scaling on validation logits (NLL, monotone).
    rank      : LEAKY baseline: rank-normalize the TEST fold itself (kept only
                to quantify the effect of the leakage the audit removes).

    Returns (calibrated_test_probs, info).
    """
    tp = np.asarray(test_probs, dtype=np.float64)
    if method == "rank":
        return rank_normalize(tp), {"method": "rank", "leaky": True}

    vp = np.asarray(val_probs, dtype=np.float64)
    vy = np.asarray(val_labels)
    info: Dict = {"method": method, "val_n": int(len(vp))}
    two_class = len(np.unique(vy)) == 2

    if method == "threshold":
        t = float(threshold) if threshold is not None else None
        if t is None:
            if not two_class:
                return tp, {**info, "fallback": "identity (degenerate validation)"}
            from ..training.engine import youden_threshold  # lazy: torch import
            t = youden_threshold(vy, vp)
        shift = float(_logit(np.array([t]))[0])
        info["threshold"] = t
        return _sigmoid(_logit(tp) - shift), info

    if method == "zval":
        if not two_class or float(np.std(_logit(vp))) < 1e-6:
            return tp, {**info, "fallback": "identity (degenerate validation)"}
        lv = _logit(vp)
        mu, sd = float(lv.mean()), max(float(lv.std()), 1e-3)
        info.update({"mu_logit": mu, "sd_logit": sd})
        return _sigmoid((_logit(tp) - mu) / sd), info

    if method == "platt":
        if not two_class or float(np.std(_logit(vp))) < 1e-6:
            return tp, {**info, "fallback": "identity (degenerate validation)"}
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, random_state=seed)
        lr.fit(_logit(vp).reshape(-1, 1), vy)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        info.update({"platt_a": a, "platt_b": b})
        return _sigmoid(a * _logit(tp) + b), info

    if method == "temp":
        if not two_class:
            return tp, {**info, "fallback": "identity (degenerate validation)"}
        from scipy.optimize import minimize_scalar

        lv = _logit(vp)

        def nll(log_t: float) -> float:
            p = np.clip(_sigmoid(lv / np.exp(log_t)), 1e-7, 1 - 1e-7)
            return float(-np.mean(vy * np.log(p) + (1 - vy) * np.log(1 - p)))

        res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        t = float(np.exp(res.x))
        info["temperature"] = t
        return _sigmoid(_logit(tp) / t), info

    raise ValueError(f"unknown calibration method '{method}'")


def load_oof_harmonized(
    results_dir, experiment: str, method: str = "threshold",
    allow_leaky_fallback: bool = True,
) -> dict:
    """Pooled out-of-fold SUBJECT probabilities with LEAK-FREE per-fold alignment.

    For every fold, the transform is estimated from that fold's inner-validation
    subjects (saved by the training harness as ``fold_*/val_preds.npz``); the
    test-fold predictions only get transformed, never used for fitting.

    If ``val_preds.npz`` is missing (old artifacts) the legacy leaky
    rank-normalization within the test fold is used only when
    ``allow_leaky_fallback=True``; the returned ``harmonization`` field records
    which route was taken so audits can detect it.

    Returns {'ids', 'probs_raw', 'probs_harm', 'labels', 'fold_aucs',
             'fold_mean_auc', 'fold_sd_auc', 'harmonization',
             'harmonization_per_fold'}.
    """
    from pathlib import Path
    from sklearn.metrics import roc_auc_score

    results_dir = Path(results_dir)
    ids, raw, harm, labels, fold_aucs = [], [], [], [], []
    infos = []
    methods_used = set()
    for fold_k, fold_dir in enumerate(sorted(results_dir.glob(experiment + "/fold_*"))):
        # allow_pickle=True: these are our own artifacts (self-produced)
        with np.load(fold_dir / "oof_trials.npz", allow_pickle=True) as z:
            probs, sids, y = z["probs"], z["sid"], z["labels"]
            snames = z["subject_ids"]
        subj = np.unique(sids)
        if len(subj) == 0:
            continue
        sp = np.array([probs[sids == s].mean() for s in subj])
        sy = np.array([int(y[sids == s][0]) for s in subj])
        fold_aucs.append(float(roc_auc_score(sy, sp)))
        ids.extend([f"fold{fold_k}:{snames[sids == s][0]}" for s in subj])
        raw.extend(sp.tolist())

        val_path = fold_dir / "val_preds.npz"
        if val_path.exists() and method != "rank":
            with np.load(val_path, allow_pickle=True) as vz:
                vs, vy = vz["probs"], vz["labels"]
                thr = float(vz["threshold"]) if "threshold" in vz.files else None
            sp_h, info = calibrate_fold(vs, vy, sp, method=method, threshold=thr)
            methods_used.add(f"leakfree:{info.get('method')}")
        elif allow_leaky_fallback:
            sp_h, info = calibrate_fold(None, None, sp, method="rank")
            methods_used.add("rank_leaky_fallback")
        else:
            raise FileNotFoundError(
                f"{val_path} missing and leaky fallback disabled for '{experiment}'"
            )
        info = {**info, "fold": fold_k}
        infos.append(info)
        harm.extend(np.asarray(sp_h).tolist())
        labels.extend(sy.tolist())
    if not fold_aucs:
        raise ValueError(f"no out-of-fold trials found for '{experiment}' — experiment missing or failed")
    # deterministic global order independent of fold enumeration
    order = np.argsort(np.array(ids, dtype=object), kind="stable")
    used = sorted(methods_used)
    return {
        "ids": np.array(ids, dtype=object)[order],
        "probs_raw": np.array(raw)[order],
        "probs_harm": np.array(harm)[order],
        "labels": np.array(labels)[order],
        "fold_aucs": np.array(fold_aucs),
        "fold_mean_auc": float(np.mean(fold_aucs)),
        "fold_sd_auc": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "harmonization": ",".join(used) if used else "none",
        "harmonization_per_fold": infos,
    }
