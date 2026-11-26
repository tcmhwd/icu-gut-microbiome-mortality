"""metrics.py — discrimination, probabilistic, calibration, threshold metrics,
and patient-level bootstrap confidence intervals.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)

from . import config

EPS = 1e-6


def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


def calibration_intercept_slope(y_true, y_prob):
    """Standard calibration-in-the-large / calibration slope via logistic
    regression of the observed outcome on the logit of the predicted
    probability (Cox calibration model). No recalibration is applied to the
    model itself -- this only *measures* calibration.
    """
    p = _clip(np.asarray(y_prob, dtype=float))
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p, y)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0][0])
    return intercept, slope


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    out = {}
    if len(np.unique(y_true)) < 2:
        out.update({"AUROC": np.nan, "AUPRC": np.nan})
    else:
        out["AUROC"] = roc_auc_score(y_true, y_prob)
        out["AUPRC"] = average_precision_score(y_true, y_prob)

    out["Brier"] = brier_score_loss(y_true, y_prob)
    out["logloss"] = log_loss(y_true, _clip(y_prob), labels=[0, 1])

    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    out["calibration_intercept"] = intercept
    out["calibration_slope"] = slope

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    out["PPV"] = tp / (tp + fp) if (tp + fp) else np.nan
    out["NPV"] = tn / (tn + fn) if (tn + fn) else np.nan
    out["F1"] = f1_score(y_true, y_pred, zero_division=0)
    out["accuracy"] = (tp + tn) / len(y_true)
    out["threshold_used"] = threshold
    out["N"] = len(y_true)
    out["N_events"] = int(y_true.sum())
    return out


def bootstrap_ci(y_true, y_prob, metric_fns: dict, n_boot: int = config.N_BOOTSTRAP,
                  seed: int = config.RANDOM_SEED) -> dict:
    """metric_fns: {'AUROC': roc_auc_score, ...} each fn(y_true, y_prob) -> float.
    Row-level (patient-level, since each row is one admission) resampling
    with replacement. Returns {'AUROC': (point, ci_low, ci_high), ...}.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    boot_values = {name: [] for name in metric_fns}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        for name, fn in metric_fns.items():
            try:
                boot_values[name].append(fn(yt, yp))
            except Exception:
                continue

    results = {}
    for name, fn in metric_fns.items():
        point = fn(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        vals = np.array(boot_values[name])
        if len(vals) >= 10:
            lo, hi = np.percentile(vals, [2.5, 97.5])
        else:
            lo, hi = np.nan, np.nan
        results[name] = (point, lo, hi)
    return results


DEFAULT_BOOTSTRAP_METRICS = {
    "AUROC": roc_auc_score,
    "AUPRC": average_precision_score,
    "Brier": brier_score_loss,
}
