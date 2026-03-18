"""Model Card Generator — auto-generate model documentation from metadata.

Based on Mitchell et al. (2019) "Model Cards for Model Reporting".
Generates structured model cards with bias/fairness results integrated.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from loguru import logger


@dataclass
class ModelCard:
    """Structured model card following Mitchell et al. (2019)."""
    # Model Details
    model_name: str = ""
    model_version: str = "1.0"
    model_type: str = ""
    framework: str = ""
    description: str = ""
    developers: list[str] = field(default_factory=list)
    license: str = "Proprietary"

    # Intended Use
    primary_use: str = ""
    primary_users: str = ""
    out_of_scope: str = ""

    # Training Data
    training_data_description: str = ""
    training_data_size: int = 0
    training_data_preprocessing: str = ""

    # Evaluation
    metrics: dict[str, float] = field(default_factory=dict)
    evaluation_data_description: str = ""

    # Fairness
    sensitive_features: list[str] = field(default_factory=list)
    bias_metrics: dict[str, Any] = field(default_factory=dict)
    bias_mitigations: list[str] = field(default_factory=list)

    # Limitations & Risks
    limitations: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)

    # EU AI Act
    eu_risk_level: str = ""
    eu_obligations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_details": {
                "name": self.model_name, "version": self.model_version,
                "type": self.model_type, "framework": self.framework,
                "description": self.description, "developers": self.developers,
                "license": self.license,
            },
            "intended_use": {
                "primary_use": self.primary_use, "primary_users": self.primary_users,
                "out_of_scope": self.out_of_scope,
            },
            "training_data": {
                "description": self.training_data_description,
                "size": self.training_data_size,
                "preprocessing": self.training_data_preprocessing,
            },
            "evaluation": {
                "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
                "data_description": self.evaluation_data_description,
            },
            "fairness": {
                "sensitive_features": self.sensitive_features,
                "bias_metrics": self.bias_metrics,
                "mitigations_applied": self.bias_mitigations,
            },
            "limitations_and_risks": {
                "limitations": self.limitations,
                "risks": self.risks,
                "ethical_considerations": self.ethical_considerations,
            },
            "regulatory": {
                "eu_ai_act_risk_level": self.eu_risk_level,
                "eu_obligations": self.eu_obligations,
            },
        }

    def to_markdown(self) -> str:
        """Generate markdown model card."""
        lines = [
            f"# Model Card: {self.model_name}",
            f"\n**Version:** {self.model_version} | **Type:** {self.model_type} | "
            f"**Framework:** {self.framework}",
            f"\n## Description\n{self.description}",
            f"\n**Developers:** {', '.join(self.developers) if self.developers else 'N/A'}",
            "\n## Intended Use",
            f"**Primary use:** {self.primary_use}",
            f"**Primary users:** {self.primary_users}",
            f"**Out of scope:** {self.out_of_scope}",
            f"\n## Training Data\n{self.training_data_description}",
            f"**Size:** {self.training_data_size:,} samples" if self.training_data_size else "",
            "\n## Evaluation Metrics",
        ]
        for k, v in self.metrics.items():
            lines.append(f"- **{k}:** {v:.4f}")

        if self.sensitive_features:
            lines.append("\n## Fairness Analysis")
            lines.append(f"**Sensitive features tested:** {', '.join(self.sensitive_features)}")
            for feat, vals in self.bias_metrics.items():
                lines.append(f"\n### {feat}")
                if isinstance(vals, dict):
                    for mk, mv in vals.items():
                        lines.append(f"- {mk}: {mv}")
            if self.bias_mitigations:
                lines.append("\n**Mitigations applied:**")
                for m in self.bias_mitigations:
                    lines.append(f"- {m}")

        if self.limitations:
            lines.append("\n## Limitations")
            for lim in self.limitations:
                lines.append(f"- {lim}")

        if self.eu_risk_level:
            lines.append(f"\n## EU AI Act Classification\n**Risk level:** {self.eu_risk_level}")
            if self.eu_obligations:
                lines.append("**Obligations:**")
                for o in self.eu_obligations:
                    lines.append(f"- {o}")

        return "\n".join(lines)


def generate_model_card(
    model_name: str,
    model_type: str = "Binary classifier",
    description: str = "",
    metrics: dict[str, float] | None = None,
    bias_report: dict[str, Any] | None = None,
    compliance_result: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ModelCard:
    """Generate a model card from metadata and analysis results.

    Args:
        model_name: Name of the model.
        model_type: Type of model.
        description: Model description.
        metrics: Performance metrics dict.
        bias_report: Output from run_bias_audit().to_dict().
        compliance_result: Output from classify_risk().to_dict().
        **kwargs: Additional ModelCard fields.

    Returns:
        Populated ModelCard.
    """
    logger.info("Generating model card for '{}'", model_name)

    card = ModelCard(
        model_name=model_name,
        model_type=model_type,
        description=description,
        metrics=metrics or {},
        **{k: v for k, v in kwargs.items() if hasattr(ModelCard, k)},
    )

    if bias_report:
        card.sensitive_features = [bias_report.get("sensitive_feature", "")]
        card.bias_metrics = {bias_report.get("sensitive_feature", "unknown"): bias_report.get("metrics", {})}
        if bias_report.get("has_significant_bias"):
            card.risks.append("Significant bias detected — see fairness analysis")

    if compliance_result:
        card.eu_risk_level = compliance_result.get("risk_level", "")
        card.eu_obligations = compliance_result.get("obligations", [])

    return card
