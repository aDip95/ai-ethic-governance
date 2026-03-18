"""Visualization utilities — Plotly charts for bias and compliance dashboards."""
from __future__ import annotations
from typing import Any
import plotly.graph_objects as go


def bias_radar_chart(metrics: dict[str, float], title: str = "Fairness Metrics") -> go.Figure:
    """Create radar chart for fairness metrics."""
    names = list(metrics.keys())
    values = list(metrics.values())
    fig = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=names + [names[0]],
                                     fill="toself", name="Current"))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(range=[0, 1])), height=450)
    return fig


def bias_comparison_bar(before: dict[str, float], after: dict[str, float],
                        title: str = "Bias Before vs After Remediation") -> go.Figure:
    """Side-by-side bar chart: before/after remediation."""
    metrics = list(before.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=metrics, y=[before[m] for m in metrics],
                         marker_color="indianred"))
    fig.add_trace(go.Bar(name="After", x=metrics, y=[after.get(m, 0) for m in metrics],
                         marker_color="seagreen"))
    fig.update_layout(title=title, barmode="group", yaxis_title="Value", height=400)
    return fig


def compliance_gauge(risk_level: str) -> go.Figure:
    """Gauge chart for EU AI Act risk level."""
    level_map = {"minimal": 1, "limited": 2, "high": 3, "unacceptable": 4}
    color_map = {"minimal": "green", "limited": "gold", "high": "orange", "unacceptable": "red"}
    val = level_map.get(risk_level, 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=val,
        title={"text": f"EU AI Act Risk: {risk_level.upper()}"},
        gauge={"axis": {"range": [0, 4], "tickvals": [1, 2, 3, 4],
                        "ticktext": ["Minimal", "Limited", "High", "Unacceptable"]},
               "bar": {"color": color_map.get(risk_level, "gray")},
               "steps": [{"range": [0, 1], "color": "lightgreen"},
                         {"range": [1, 2], "color": "lightyellow"},
                         {"range": [2, 3], "color": "navajowhite"},
                         {"range": [3, 4], "color": "lightcoral"}]}))
    fig.update_layout(height=350)
    return fig


def shap_waterfall_plotly(contributions: list[dict[str, Any]], base_value: float,
                          title: str = "Feature Contributions") -> go.Figure:
    """Plotly waterfall chart mimicking SHAP waterfall plot."""
    contribs = contributions[:12]  # top 12
    features = [c["feature"] for c in contribs]
    values = [c.get("shap_value", c.get("weight", 0)) for c in contribs]
    colors = ["red" if v > 0 else "blue" for v in values]
    fig = go.Figure(go.Bar(x=values, y=features, orientation="h", marker_color=colors))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"{title} (base={base_value:.3f})", xaxis_title="SHAP value",
                      height=max(350, len(contribs) * 30))
    return fig
