"""LIME explainability — local interpretable explanations for tabular data."""
from __future__ import annotations
from typing import Any
import numpy as np
from loguru import logger


def explain_with_lime(
    model: Any,
    X_train: np.ndarray,
    instance: np.ndarray,
    feature_names: list[str] | None = None,
    num_features: int = 10,
    num_samples: int = 500,
) -> dict[str, Any]:
    """Generate LIME explanation for a single instance.

    Args:
        model: Trained model with predict_proba.
        X_train: Training data for LIME background.
        instance: Single instance to explain.
        feature_names: Optional feature names.
        num_features: Number of top features to show.
        num_samples: Number of perturbed samples for LIME.

    Returns:
        Dict with feature contributions, prediction, and intercept.
    """
    from lime.lime_tabular import LimeTabularExplainer

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    explainer = LimeTabularExplainer(
        X_train, feature_names=feature_names, mode="classification",
        discretize_continuous=True, random_state=42,
    )

    exp = explainer.explain_instance(
        instance.ravel(), model.predict_proba,
        num_features=num_features, num_samples=num_samples,
    )

    pred_proba = model.predict_proba(instance.reshape(1, -1))[0]
    contributions = [
        {"feature": feat, "weight": round(float(weight), 4)}
        for feat, weight in exp.as_list()
    ]

    logger.info("LIME: explained instance with {} features", len(contributions))
    return {
        "contributions": contributions,
        "predicted_class": int(np.argmax(pred_proba)),
        "predicted_proba": [round(float(p), 4) for p in pred_proba],
        "intercept": round(float(list(exp.intercept.values())[1]) if len(exp.intercept) > 1 else round(float(list(exp.intercept.values())[0]), 4), 4),
        "score": round(float(exp.score), 4),
    }
