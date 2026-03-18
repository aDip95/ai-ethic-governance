"""Bias remediation — reweighing and threshold optimization.

Implements pre-processing (reweighing) and post-processing (threshold
optimization) to reduce fairness violations.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from loguru import logger


def compute_reweighing_weights(
    y_true: np.ndarray, sensitive: np.ndarray, priv_val: Any,
) -> np.ndarray:
    """Compute sample weights to achieve demographic parity.

    Reweighing (Kamiran & Calders, 2012): assigns weights inversely
    proportional to P(Y|S) to equalize positive rates across groups.

    Args:
        y_true: Ground truth labels.
        sensitive: Sensitive feature values.
        priv_val: Privileged group value.

    Returns:
        Sample weights array.
    """
    n = len(y_true)
    priv = sensitive == priv_val
    unpriv = ~priv

    # Joint probabilities
    p_priv = priv.sum() / n
    p_unpriv = unpriv.sum() / n
    p_pos = y_true.mean()
    p_neg = 1 - p_pos

    weights = np.ones(n, dtype=np.float64)

    # P(S=s, Y=y) / (P(S=s) * P(Y=y))
    for s_mask, p_s in [(priv, p_priv), (unpriv, p_unpriv)]:
        for y_val, p_y in [(1, p_pos), (0, p_neg)]:
            mask = s_mask & (y_true == y_val)
            p_joint = mask.sum() / n
            expected = p_s * p_y
            if p_joint > 0 and expected > 0:
                weights[mask] = expected / p_joint

    logger.info("Reweighing: weight range [{:.3f}, {:.3f}]", weights.min(), weights.max())
    return weights


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive: np.ndarray,
    priv_val: Any,
    metric: str = "demographic_parity",
    n_thresholds: int = 100,
) -> dict[str, float]:
    """Find group-specific thresholds to minimize fairness violation.

    For each group (priv/unpriv), finds the threshold that minimizes
    the specified fairness metric while maintaining reasonable accuracy.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        sensitive: Sensitive feature.
        priv_val: Privileged group value.
        metric: Target fairness metric.
        n_thresholds: Number of threshold candidates.

    Returns:
        Dict with optimal thresholds and resulting metrics.
    """
    priv_mask = sensitive == priv_val
    unpriv_mask = ~priv_mask
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    best_combo = None
    best_fairness = float("inf")

    for t_priv in thresholds[::5]:  # coarser grid for speed
        for t_unpriv in thresholds[::5]:
            preds = np.zeros_like(y_true)
            preds[priv_mask] = (y_prob[priv_mask] >= t_priv).astype(int)
            preds[unpriv_mask] = (y_prob[unpriv_mask] >= t_unpriv).astype(int)

            priv_rate = preds[priv_mask].mean()
            unpriv_rate = preds[unpriv_mask].mean()
            dp_diff = abs(priv_rate - unpriv_rate)

            acc = (preds == y_true).mean()
            if acc < 0.6:  # don't sacrifice too much accuracy
                continue

            if dp_diff < best_fairness:
                best_fairness = dp_diff
                best_combo = {
                    "threshold_privileged": round(float(t_priv), 3),
                    "threshold_unprivileged": round(float(t_unpriv), 3),
                    "demographic_parity_diff": round(dp_diff, 4),
                    "accuracy": round(float(acc), 4),
                }

    if best_combo is None:
        best_combo = {"threshold_privileged": 0.5, "threshold_unprivileged": 0.5,
                      "demographic_parity_diff": 1.0, "accuracy": 0.0}

    logger.info("Optimal thresholds: priv={}, unpriv={}, DP_diff={:.4f}",
                best_combo["threshold_privileged"],
                best_combo["threshold_unprivileged"],
                best_combo["demographic_parity_diff"])
    return best_combo
