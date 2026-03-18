"""EU AI Act risk classifier — maps AI system descriptions to risk levels.

Implements the 4-tier risk classification (Unacceptable, High, Limited,
Minimal) with obligation checklists per Annex III of the EU AI Act.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from loguru import logger


class RiskLevel:
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


# Annex III high-risk categories
HIGH_RISK_DOMAINS = {
    "biometric": "Real-time and post remote biometric identification",
    "critical_infrastructure": "Safety components of critical infrastructure",
    "education": "Access to educational institutions, assessment of students",
    "employment": "Recruitment, hiring, performance evaluation, task allocation",
    "essential_services": "Access to essential services (credit, insurance, social benefits)",
    "law_enforcement": "Risk assessment, polygraphs, evidence analysis",
    "migration": "Border control, visa applications, asylum",
    "justice": "Assistance to judicial authorities in fact-finding",
}

HIGH_RISK_KEYWORDS = {
    "biometric": ["facial recognition", "biometric", "face detection", "identification"],
    "critical_infrastructure": ["energy", "water", "transport", "road traffic"],
    "education": ["admission", "exam", "grading", "student assessment", "education"],
    "employment": ["hiring", "recruitment", "cv screening", "employee evaluation", "promotion", "firing"],
    "essential_services": ["credit scoring", "loan", "insurance", "mortgage", "social benefit", "welfare"],
    "law_enforcement": ["crime prediction", "recidivism", "surveillance", "profiling"],
    "migration": ["asylum", "visa", "border", "immigration"],
    "justice": ["sentencing", "parole", "judicial", "court"],
}

UNACCEPTABLE_KEYWORDS = [
    "social scoring", "mass surveillance", "subliminal manipulation",
    "exploit vulnerability", "cognitive behavioral manipulation",
    "real-time biometric public", "emotion recognition workplace",
]

LIMITED_RISK_KEYWORDS = [
    "chatbot", "deepfake", "emotion recognition", "content generation",
    "ai-generated", "synthetic media",
]

HIGH_RISK_OBLIGATIONS = [
    "Risk management system (Art. 9)",
    "Data governance and quality (Art. 10)",
    "Technical documentation (Art. 11)",
    "Record-keeping / logging (Art. 12)",
    "Transparency and information to users (Art. 13)",
    "Human oversight measures (Art. 14)",
    "Accuracy, robustness, cybersecurity (Art. 15)",
    "Quality management system (Art. 17)",
    "EU conformity assessment (Art. 43)",
    "Registration in EU database (Art. 49)",
    "Post-market monitoring (Art. 61)",
    "Incident reporting to authorities (Art. 62)",
    "Bias testing and mitigation (Art. 10.2f)",
    "Explainability of outputs (Art. 13.3d)",
    "Fundamental rights impact assessment (Art. 29a)",
]


@dataclass
class ComplianceResult:
    """EU AI Act compliance assessment result."""
    system_description: str
    risk_level: str
    matched_domain: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_description": self.system_description[:200],
            "risk_level": self.risk_level,
            "matched_domain": self.matched_domain,
            "matched_keywords": self.matched_keywords,
            "n_obligations": len(self.obligations),
            "obligations": self.obligations,
            "recommendations": self.recommendations,
        }


def classify_risk(system_description: str) -> ComplianceResult:
    """Classify an AI system under the EU AI Act risk framework.

    Args:
        system_description: Free-text description of the AI system.

    Returns:
        ComplianceResult with risk level, obligations, and recommendations.
    """
    desc_lower = system_description.lower()
    logger.info("Classifying AI system risk: '{}'", system_description[:80])

    # Check unacceptable
    for kw in UNACCEPTABLE_KEYWORDS:
        if kw in desc_lower:
            return ComplianceResult(
                system_description=system_description,
                risk_level=RiskLevel.UNACCEPTABLE,
                matched_keywords=[kw],
                obligations=["System PROHIBITED under EU AI Act (Art. 5)"],
                recommendations=["Immediately cease development/deployment",
                                 "Consult legal team for alternatives"],
            )

    # Check high-risk
    for domain, keywords in HIGH_RISK_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in desc_lower]
        if matched:
            return ComplianceResult(
                system_description=system_description,
                risk_level=RiskLevel.HIGH,
                matched_domain=HIGH_RISK_DOMAINS[domain],
                matched_keywords=matched,
                obligations=HIGH_RISK_OBLIGATIONS,
                recommendations=[
                    "Conduct fundamental rights impact assessment",
                    "Implement comprehensive risk management system",
                    "Ensure data quality and bias testing",
                    "Set up human oversight mechanisms",
                    "Register in EU AI database before deployment",
                    "Prepare technical documentation package",
                ],
            )

    # Check limited risk
    for kw in LIMITED_RISK_KEYWORDS:
        if kw in desc_lower:
            return ComplianceResult(
                system_description=system_description,
                risk_level=RiskLevel.LIMITED,
                matched_keywords=[kw],
                obligations=[
                    "Transparency: inform users they are interacting with AI (Art. 52)",
                    "If generating synthetic content: label as AI-generated",
                    "If emotion recognition: inform subjects of system operation",
                ],
                recommendations=["Implement clear AI disclosure in user interface"],
            )

    # Minimal risk
    return ComplianceResult(
        system_description=system_description,
        risk_level=RiskLevel.MINIMAL,
        obligations=["No mandatory obligations (voluntary codes of conduct encouraged)"],
        recommendations=["Consider adopting voluntary AI ethics guidelines"],
    )
