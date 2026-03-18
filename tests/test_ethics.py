"""Tests for AI Ethics & Governance Dashboard."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from app.bias.detector import (
    demographic_parity_difference, disparate_impact, equalized_odds_difference,
    run_bias_audit,
)
from app.bias.remediation import compute_reweighing_weights, find_optimal_threshold
from app.compliance.eu_ai_act_classifier import classify_risk, RiskLevel
from app.compliance.model_card_generator import generate_model_card
from app.audit.trail import AuditTrail
from app.explainability.counterfactuals import find_counterfactuals
from app.viz.charts import bias_radar_chart, compliance_gauge
from data.generate_loan_data import generate_loan_dataset


# --- Fixtures ---

@pytest.fixture
def loan_data() -> pd.DataFrame:
    return generate_loan_dataset(n_samples=1000, seed=99)


@pytest.fixture
def trained_model(loan_data):
    """Train a simple model for testing."""
    feature_cols = ["income", "credit_score", "debt_to_income", "employment_years", "loan_amount"]
    X = loan_data[feature_cols].values
    y = loan_data["approved"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, feature_cols


# --- Bias Detection ---

class TestBiasDetector:
    def test_demographic_parity(self) -> None:
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
        sensitive = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        dp = demographic_parity_difference(y_pred, sensitive, 1)
        assert 0 <= dp <= 1

    def test_disparate_impact(self) -> None:
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        sensitive = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        di = disparate_impact(y_pred, sensitive, 1)
        assert di >= 0

    def test_equalized_odds(self) -> None:
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 0])
        sensitive = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        eo = equalized_odds_difference(y_true, y_pred, sensitive, 1)
        assert 0 <= eo <= 1

    def test_run_bias_audit(self, loan_data, trained_model) -> None:
        model, _, X_test, _, y_test, _ = trained_model
        y_pred = model.predict(X_test)
        sensitive = (loan_data.iloc[-len(y_test):]["race"] == "white").values.astype(int)
        report = run_bias_audit(y_test, y_pred, sensitive, 1, "race")
        assert "demographic_parity_diff" in report.metrics
        assert "disparate_impact" in report.metrics
        assert isinstance(report.has_significant_bias, bool)

    def test_audit_to_dict(self) -> None:
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        sensitive = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        report = run_bias_audit(y_true, y_pred, sensitive, 1, "test_feature")
        d = report.to_dict()
        assert "sensitive_feature" in d
        assert "metrics" in d


# --- Remediation ---

class TestRemediation:
    def test_reweighing(self) -> None:
        y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        s = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        weights = compute_reweighing_weights(y, s, 1)
        assert len(weights) == 8
        assert all(w > 0 for w in weights)

    def test_optimal_threshold(self, trained_model) -> None:
        model, _, X_test, _, y_test, _ = trained_model
        y_prob = model.predict_proba(X_test)[:, 1]
        sensitive = np.random.default_rng(42).choice([0, 1], len(y_test))
        result = find_optimal_threshold(y_test, y_prob, sensitive, 1)
        assert "threshold_privileged" in result
        assert "accuracy" in result


# --- EU AI Act ---

class TestCompliance:
    def test_high_risk(self) -> None:
        result = classify_risk("AI system for credit scoring and loan approval decisions")
        assert result.risk_level == RiskLevel.HIGH
        assert len(result.obligations) > 5

    def test_unacceptable(self) -> None:
        result = classify_risk("Social scoring system for citizen behavior")
        assert result.risk_level == RiskLevel.UNACCEPTABLE

    def test_limited(self) -> None:
        result = classify_risk("Customer service chatbot for FAQs")
        assert result.risk_level == RiskLevel.LIMITED

    def test_minimal(self) -> None:
        result = classify_risk("Spam email filter for internal use")
        assert result.risk_level == RiskLevel.MINIMAL

    def test_to_dict(self) -> None:
        result = classify_risk("Hiring tool for CV screening")
        d = result.to_dict()
        assert "risk_level" in d
        assert "obligations" in d


# --- Model Card ---

class TestModelCard:
    def test_generate(self) -> None:
        card = generate_model_card(
            "LoanApproval_v1", model_type="GBM", description="Loan approval classifier",
            metrics={"accuracy": 0.85, "auc_roc": 0.91},
        )
        assert card.model_name == "LoanApproval_v1"
        assert "accuracy" in card.metrics

    def test_to_markdown(self) -> None:
        card = generate_model_card("Test", metrics={"f1": 0.88})
        md = card.to_markdown()
        assert "# Model Card:" in md
        assert "f1" in md

    def test_with_bias_and_compliance(self) -> None:
        bias = {"sensitive_feature": "race", "metrics": {"dp_diff": 0.15}, "has_significant_bias": True}
        compliance = {"risk_level": "high", "obligations": ["Art. 9", "Art. 10"]}
        card = generate_model_card("M", bias_report=bias, compliance_result=compliance)
        assert card.eu_risk_level == "high"
        assert len(card.risks) > 0


# --- Audit Trail ---

class TestAuditTrail:
    def test_log_and_retrieve(self, tmp_path) -> None:
        trail = AuditTrail(str(tmp_path / "test.db"))
        eid = trail.log_event("bias_audit", {"model": "test", "dp_diff": 0.15}, "test_model", "tester")
        events = trail.get_events(event_type="bias_audit")
        assert len(events) == 1
        assert events[0]["id"] == eid

    def test_integrity_check(self, tmp_path) -> None:
        trail = AuditTrail(str(tmp_path / "test.db"))
        trail.log_event("test", {"x": 1})
        trail.log_event("test", {"x": 2})
        result = trail.verify_integrity()
        assert result["integrity_ok"] is True
        assert result["total_records"] == 2

    def test_count_events(self, tmp_path) -> None:
        trail = AuditTrail(str(tmp_path / "test.db"))
        trail.log_event("bias_audit", {})
        trail.log_event("bias_audit", {})
        trail.log_event("compliance", {})
        counts = trail.count_events()
        assert counts["bias_audit"] == 2
        assert counts["compliance"] == 1


# --- Counterfactuals ---

class TestCounterfactuals:
    def test_find(self, trained_model) -> None:
        model, X_train, X_test, _, _, feature_names = trained_model
        # Find an instance predicted as 0
        preds = model.predict(X_test)
        denied_idx = np.where(preds == 0)[0]
        if len(denied_idx) > 0:
            instance = X_test[denied_idx[0]]
            result = find_counterfactuals(model, instance, feature_names, target_class=1)
            assert result["original_prediction"] == 0
            assert result["target_class"] == 1


# --- Viz ---

class TestViz:
    def test_radar(self) -> None:
        fig = bias_radar_chart({"dp_diff": 0.1, "di": 0.85, "eo_diff": 0.05})
        assert fig is not None

    def test_gauge(self) -> None:
        fig = compliance_gauge("high")
        assert fig is not None


# --- Data ---

class TestDataGeneration:
    def test_generate_loan(self) -> None:
        df = generate_loan_dataset(n_samples=500)
        assert len(df) == 500
        assert "approved" in df.columns
        assert set(df["gender"].unique()) == {"male", "female"}

    def test_bias_exists(self) -> None:
        df = generate_loan_dataset(n_samples=5000)
        white_rate = df[df["race"] == "white"]["approved"].mean()
        black_rate = df[df["race"] == "black"]["approved"].mean()
        # Verify bias is injected (white approval > black approval)
        assert white_rate > black_rate


# --- Additional coverage tests ---

class TestShapExplainer:
    def test_compute_shap(self, trained_model) -> None:
        model, X_tr, X_te, _, _, feat_cols = trained_model
        from app.explainability.shap_explainer import compute_shap_values
        result = compute_shap_values(model, X_tr[:50], X_te[:5], feat_cols)
        assert "shap_values" in result
        assert len(result["feature_importances"]) == len(feat_cols)

    def test_single_instance(self, trained_model) -> None:
        model, X_tr, X_te, _, _, feat_cols = trained_model
        from app.explainability.shap_explainer import explain_single_instance
        result = explain_single_instance(model, X_tr[:50], X_te[0], feat_cols)
        assert "contributions" in result
        assert "prediction_explanation" in result


class TestLimeExplainer:
    def test_explain(self, trained_model) -> None:
        model, X_tr, X_te, _, _, feat_cols = trained_model
        from app.explainability.lime_explainer import explain_with_lime
        result = explain_with_lime(model, X_tr[:100], X_te[0], feat_cols, num_features=5, num_samples=100)
        assert "contributions" in result
        assert "predicted_class" in result


class TestVizExtra:
    def test_comparison_bar(self) -> None:
        from app.viz.charts import bias_comparison_bar
        fig = bias_comparison_bar({"a": 0.3, "b": 0.5}, {"a": 0.1, "b": 0.2})
        assert fig is not None

    def test_shap_waterfall(self) -> None:
        from app.viz.charts import shap_waterfall_plotly
        fig = shap_waterfall_plotly([{"feature": "x", "shap_value": 0.3}], 0.5)
        assert fig is not None
