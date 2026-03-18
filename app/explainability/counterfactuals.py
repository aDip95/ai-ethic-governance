"""Counterfactual explanations — minimal changes to flip a prediction.

Implements a simple gradient-free counterfactual search for tabular data,
showing what would need to change for a different outcome.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from loguru import logger


def find_counterfactuals(
    model: Any,
    instance: np.ndarray,
    feature_names: list[str] | None = None,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    categorical_features: list[int] | None = None,
    target_class: int = 0,
    n_counterfactuals: int = 3,
    max_iterations: int = 1000,
    step_size: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Find counterfactual explanations via random perturbation search.

    Args:
        model: Trained model with predict/predict_proba.
        instance: Original instance (1D array).
        feature_names: Feature names.
        feature_ranges: Min/max per feature for valid perturbations.
        categorical_features: Indices of categorical features (perturbed discretely).
        target_class: Desired class for counterfactual.
        n_counterfactuals: Number of counterfactuals to find.
        max_iterations: Maximum search iterations.
        step_size: Perturbation magnitude (fraction of range).
        seed: Random seed.

    Returns:
        Dict with counterfactuals and changes required.
    """
    rng = np.random.default_rng(seed)
    inst = instance.ravel().copy()
    n_features = len(inst)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    categorical_features = categorical_features or []

    # Determine ranges
    ranges = {}
    for i in range(n_features):
        if feature_ranges and feature_names[i] in feature_ranges:
            ranges[i] = feature_ranges[feature_names[i]]
        else:
            ranges[i] = (inst[i] - 3 * abs(inst[i] + 1e-6), inst[i] + 3 * abs(inst[i] + 1e-6))

    original_pred = int(model.predict(inst.reshape(1, -1))[0])
    counterfactuals = []

    for _ in range(max_iterations):
        if len(counterfactuals) >= n_counterfactuals:
            break

        # Perturb random subset of features
        n_perturb = rng.integers(1, max(2, n_features // 2))
        features_to_perturb = rng.choice(n_features, size=n_perturb, replace=False)

        candidate = inst.copy()
        for fi in features_to_perturb:
            lo, hi = ranges[fi]
            if fi in categorical_features:
                candidate[fi] = rng.choice([lo, hi])
            else:
                delta = (hi - lo) * step_size * rng.standard_normal()
                candidate[fi] = np.clip(candidate[fi] + delta, lo, hi)

        pred = int(model.predict(candidate.reshape(1, -1))[0])
        if pred == target_class:
            changes = []
            for i in range(n_features):
                if abs(candidate[i] - inst[i]) > 1e-6:
                    changes.append({
                        "feature": feature_names[i],
                        "original": round(float(inst[i]), 4),
                        "counterfactual": round(float(candidate[i]), 4),
                        "change": round(float(candidate[i] - inst[i]), 4),
                    })
            if changes:
                counterfactuals.append({
                    "changes": sorted(changes, key=lambda x: -abs(x["change"])),
                    "n_features_changed": len(changes),
                    "new_prediction": pred,
                })

    # Deduplicate and sort by sparsity
    counterfactuals.sort(key=lambda x: x["n_features_changed"])
    counterfactuals = counterfactuals[:n_counterfactuals]

    logger.info("Found {} counterfactuals (target_class={})", len(counterfactuals), target_class)
    return {
        "original_prediction": original_pred,
        "target_class": target_class,
        "n_found": len(counterfactuals),
        "counterfactuals": counterfactuals,
    }
