# ⚖️ AI Ethics & Governance Dashboard

[![CI](https://github.com/yourusername/ai-ethics-governance/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/ai-ethics-governance/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Responsible AI platform with EU AI Act compliance, 8 fairness metrics, SHAP/LIME explainability, counterfactual explanations, and immutable audit trail.

## Problem

ML models in production often exhibit bias, lack transparency, and don't comply with regulations like the EU AI Act. Teams need a unified workflow to detect bias, explain decisions, check compliance, and maintain audit trails.

## Case Study

**Loan Approval Model** → detect race/gender bias → apply reweighing → show improvement across all metrics → EU AI Act: High Risk → 15-obligation checklist → auto-generated model card.

## Architecture

```mermaid
graph TD
    A[Synthetic Loan Data<br>5k samples, biased] --> B[Train GBM Classifier]
    B --> C[Bias Detector<br>8 fairness metrics]
    C --> D{Bias Found?}
    D -->|Yes| E[Remediation<br>Reweighing + Threshold Opt]
    E --> C
    D -->|No| F[Explainability]
    F --> G[SHAP Waterfall]
    F --> H[LIME Local]
    F --> I[Counterfactuals]
    B --> J[EU AI Act Classifier]
    J --> K[Risk Level + Obligations]
    B --> L[Model Card Generator]
    C --> M[Audit Trail<br>SQLite immutable log]
    J --> M
```

## Fairness Metrics (8)

| Metric | Ideal | Flag |
|--------|-------|------|
| Demographic Parity Diff | 0 | > 0.1 |
| Disparate Impact | 1.0 | < 0.8 (4/5 rule) |
| Equalized Odds Diff | 0 | > 0.1 |
| Equal Opportunity Diff | 0 | > 0.1 |
| Predictive Parity Diff | 0 | > 0.1 |
| Treatment Equality | 0 | > 0.2 |
| Calibration Diff | 0 | > 0.1 |
| Individual Fairness | 1.0 | < 0.9 |

## Quickstart

```bash
git clone https://github.com/yourusername/ai-ethics-governance.git
cd ai-ethics-governance
pip install -r requirements.txt
pytest tests/ --cov=app -v
streamlit run app/main.py
```

## Docker

```bash
cd docker && docker-compose up --build
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Bias Detection | Custom (8 metrics) |
| Explainability | SHAP, LIME, Counterfactuals |
| Compliance | EU AI Act classifier |
| Audit | SQLite immutable trail |
| Viz | Plotly |
| Dashboard | Streamlit |
