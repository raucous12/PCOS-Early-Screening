"""Evaluation metrics: discrimination, calibration, thresholds and bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Ideal: a = 0, b = 1."""
    from sklearn.linear_model import LogisticRegression

    x = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-frequency bins."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    n = len(p)
    return float(sum(len(c) / n * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def sens_spec(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = p >= threshold
    pos, neg = y == 1, y == 0
    sens = float(pred[pos].mean()) if pos.any() else float("nan")
    spec = float((~pred[neg]).mean()) if neg.any() else float("nan")
    return sens, spec


def summarize(y, p) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    if not np.isfinite(p).all():
        raise ValueError("Predictions contain NaN/inf")
    out = {"n": int(len(y)), "prevalence": float(y.mean())}
    if len(np.unique(y)) == 2:
        out.update(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)))
        slope, intercept = calibration_slope_intercept(y, p)
        out.update(calibration_slope=slope, calibration_intercept=intercept)
    out.update(brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
               ece=expected_calibration_error(y, p), mean_predicted=float(p.mean()))
    return out


def bootstrap_ci(y, p, metric=roc_auc_score, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI (stratified, so every resample has both classes)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_auc_diff(y, p_a, p_b, n: int = 1000, seed: int = 0) -> dict[str, float]:
    """AUC(a) - AUC(b) with a 95% CI and a two-sided bootstrap p-value."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "p_value": float(min(1.0, p_value))}


def choose_thresholds(y, p, min_sensitivity: float = 0.9, min_specificity: float = 0.8) -> dict[str, float]:
    """Risk-band cut-offs from out-of-fold probabilities.

    * ``low``: the highest threshold that still keeps sensitivity >= ``min_sensitivity``,
      so a "low" result misses at most 10% of PCOS cases.
    * ``high``: the lowest threshold with specificity >= ``min_specificity``.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    cands = np.unique(np.round(p, 4))
    low = max((t for t in cands if sens_spec(y, p, t)[0] >= min_sensitivity), default=float(cands.min()))
    high = min((t for t in cands if sens_spec(y, p, t)[1] >= min_specificity), default=float(cands.max()))
    if high <= low:  # weak model: collapse the moderate band but keep the order valid
        high = float(min(1.0, low + 1e-3))
    s_lo, c_lo = sens_spec(y, p, low)
    s_hi, c_hi = sens_spec(y, p, high)
    return {"low": float(low), "high": float(high),
            "low_sensitivity": s_lo, "low_specificity": c_lo,
            "high_sensitivity": s_hi, "high_specificity": c_hi}


def reliability_curve(y, p, bins: int = 10) -> list[dict[str, float]]:
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            out.append({"mean_predicted": float(np.mean(np.asarray(p)[chunk])),
                        "observed": float(np.mean(np.asarray(y)[chunk])), "n": int(len(chunk))})
    return out
