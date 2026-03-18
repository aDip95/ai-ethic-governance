"""SHAP explainability — waterfall, beeswarm, and force plot generation.

Wraps SHAP TreeExplainer/KernelExplainer for tabular models,
producing structured explanation outputs for the dashboard.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from loguru import logger


def compute_shap_values(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str] | None = None,
    max_background: int = 100,
) -> dict[str, Any]:
    """Compute SHAP values using TreeExplainer or KernelExplainer.

    Args:
        model: Trained sklearn-compatible model.
        X_background: Background data for SHAP (subsample for speed).
        X_explain: Instances to explain.
        feature_names: Optional feature names.
        max_background: Max background samples.

    Returns:
        Dict with shap_values, base_value, feature_names, feature_importances.
    """
    import shap

    bg = X_background[:max_background]
    try:
        explainer = shap.TreeExplainer(model, bg)
        logger.info("Using TreeExplainer")
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        logger.info("Using KernelExplainer")

    sv = explainer.shap_values(X_explain)

    # Handle multi-class: take positive class
    if isinstance(sv, list):
        shap_vals = sv[1] if len(sv) > 1 else sv[0]
    else:
        shap_vals = sv

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_explain.shape[1])]

    # Global feature importance
    importance = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(-importance)

    return {
        "shap_values": shap_vals.tolist(),
        "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
        "feature_names": feature_names,
        "feature_importances": [
            {"feature": feature_names[i], "importance": round(float(importance[i]), 4)}
            for i in sorted_idx
        ],
        "n_explained": len(X_explain),
    }


def explain_single_instance(
    model: Any,
    X_background: np.ndarray,
    instance: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Explain a single prediction with per-feature contributions."""
    result = compute_shap_values(model, X_background, instance.reshape(1, -1), feature_names)
    sv = result["shap_values"][0]
    names = result["feature_names"]

    contributions = sorted(
        [{"feature": n, "shap_value": round(float(v), 4)} for n, v in zip(names, sv)],
        key=lambda x: -abs(x["shap_value"]),
    )

    return {
        "base_value": result["base_value"],
        "contributions": contributions,
        "prediction_explanation": _generate_text_explanation(contributions, result["base_value"]),
    }


def _generate_text_explanation(contributions: list[dict], base_value: float) -> str:
    """Generate human-readable explanation from SHAP values."""
    top_pos = [c for c in contributions if c["shap_value"] > 0][:3]
    top_neg = [c for c in contributions if c["shap_value"] < 0][:3]

    parts = [f"Starting from base probability {base_value:.2f}:"]
    if top_pos:
        factors = ", ".join(f"{c['feature']} (+{c['shap_value']:.3f})" for c in top_pos)
        parts.append(f"Factors increasing risk: {factors}")
    if top_neg:
        factors = ", ".join(f"{c['feature']} ({c['shap_value']:.3f})" for c in top_neg)
        parts.append(f"Factors decreasing risk: {factors}")
    return ". ".join(parts)
