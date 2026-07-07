"""Streamlit model-monitoring dashboard (Module 7).

Reads the prediction log and the training reference, then shows:
- overall health + retraining recommendation,
- realized-performance KPIs vs the validation baseline,
- the retraining-trigger panel (Module 5 handoff section 10),
- input & prediction drift (PSI) with reference-vs-production distributions,
- rolling error over time.

Run (from Module6_7_Deployment/)::

    streamlit run monitoring/dashboard.py

If the log is empty, generate demo traffic first::

    python -m monitoring.simulate_production --scenario both --n 300 --reset
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from monitoring.common import (  # noqa: E402
    ACTUAL_COL,
    CATEGORICAL_MONITOR_FEATURES,
    NUMERIC_MONITOR_FEATURES,
    PREDICTION_COL,
    TIMESTAMP_COL,
    default_log_path,
    load_validation_baseline,
)
from monitoring.drift import (  # noqa: E402
    evaluate_triggers,
    feature_drift_report,
    load_predictions,
    overall_status,
    performance_metrics,
    recommend_retraining,
    rolling_performance,
)

st.set_page_config(page_title="Model Monitoring — House Prices", page_icon="📈", layout="wide")

SEVERITY_EMOJI = {"ok": "🟢", "warning": "🟠", "critical": "🔴"}


@st.cache_data(show_spinner=False)
def _load_reference() -> pd.DataFrame:
    from monitoring.reference import load_reference

    return load_reference()


def _load_logs(log_path: str) -> pd.DataFrame:
    return load_predictions(log_path)


def _distribution_figure(ref: pd.Series, cur: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ref, name="reference (training)", opacity=0.6,
                               histnorm="probability density", marker_color="#94a3b8"))
    fig.add_trace(go.Histogram(x=cur, name="production", opacity=0.6,
                               histnorm="probability density", marker_color="#2563eb"))
    fig.update_layout(barmode="overlay", title=title, height=320,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


# ----------------------------------------------------------------------------
st.title("📈 Model Monitoring — House Price Valuation")
st.caption("Data drift, prediction drift, and realized error for model `lasso_v1`. Module 7.")

with st.sidebar:
    st.header("Data")
    log_path = st.text_input("Prediction log path", value=str(default_log_path()))
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
    st.markdown("---")
    st.caption("Generate demo traffic from a terminal:")
    st.code("python -m monitoring.simulate_production \\\n  --scenario both --n 300 --reset", language="bash")

baseline = load_validation_baseline()

try:
    reference = _load_reference()
except FileNotFoundError:
    st.error("Reference dataset not found. Run `python -m monitoring.reference` first.")
    st.stop()

logs = _load_logs(log_path)
if logs.empty:
    st.warning("No predictions logged yet. Generate demo traffic (see the sidebar) or send requests to the API.")
    st.stop()

# Optional time filter.
if TIMESTAMP_COL in logs.columns and logs[TIMESTAMP_COL].notna().any():
    tmin = pd.to_datetime(logs[TIMESTAMP_COL].min()).date()
    tmax = pd.to_datetime(logs[TIMESTAMP_COL].max()).date()
    with st.sidebar:
        st.header("Window")
        date_range = st.date_input("Analysis window", value=(tmin, tmax), min_value=tmin, max_value=tmax)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        lo, hi = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
        logs = logs[(logs[TIMESTAMP_COL] >= lo) & (logs[TIMESTAMP_COL] < hi)]

if logs.empty:
    st.warning("No predictions in the selected window.")
    st.stop()

# --- compute everything ---
drift_report = feature_drift_report(reference, logs)
perf = performance_metrics(logs)
triggers = evaluate_triggers(drift_report, perf, baseline)
status = overall_status(triggers)
retrain = recommend_retraining(triggers)

# --- header banner ---
banner = {
    "ok": ("🟢 Healthy", st.success),
    "warning": ("🟠 Warning — investigate", st.warning),
    "critical": ("🔴 Critical — action needed", st.error),
}[status]
banner[1](
    f"**{banner[0]}** · {len(logs):,} predictions "
    f"({perf.get('n_with_actual', 0):,} with known sale price) · "
    f"**Retraining recommended: {'YES' if retrain else 'No'}**"
)

# --- KPI row ---
st.subheader("Realized performance vs validation baseline")
if perf.get("n_with_actual", 0) > 0:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAPE", f"{perf['mape']:.2%}", f"{(perf['mape'] - baseline['valid_mape']):+.2%}",
              delta_color="inverse")
    c2.metric("Within ±10%", f"{perf['within_10_pct']:.1%}",
              f"{(perf['within_10_pct'] - baseline['valid_within_10_pct']):+.1%}")
    c3.metric("RMSE", f"${perf['rmse']:,.0f}", f"{(perf['rmse'] - baseline['valid_rmse']):+,.0f}",
              delta_color="inverse")
    c4.metric("MAE", f"${perf['mae']:,.0f}", f"{(perf['mae'] - baseline['valid_mae']):+,.0f}",
              delta_color="inverse")
    c5.metric("Median residual", f"${perf['median_residual']:,.0f}",
              help="actual − predicted. Positive = model underpredicts; negative = overpredicts.")
else:
    st.info("No ground-truth sale prices logged yet — showing drift only. "
            "Back-fill `actual_sale_price` in the log to see realized error.")

# --- triggers ---
st.subheader("Retraining triggers")
st.caption("Thresholds from the Module 5 handoff (§10). Retrain if any trigger is critical, or ≥2 are warnings.")
trig_df = pd.DataFrame(triggers)
trig_df["  "] = trig_df["severity"].map(SEVERITY_EMOJI)
st.dataframe(
    trig_df[["  ", "trigger", "severity", "detail"]].rename(
        columns={"trigger": "Trigger", "severity": "Severity", "detail": "Detail"}
    ),
    hide_index=True,
    width="stretch",
)

# --- drift ---
st.subheader("Feature & prediction drift (PSI)")
left, right = st.columns([3, 2])
with left:
    show = drift_report.copy()
    show["PSI"] = show["psi"]
    fig = px.bar(
        show, x="PSI", y="feature", orientation="h", color="status",
        color_discrete_map={"stable": "#22c55e", "moderate": "#f59e0b", "significant": "#ef4444"},
        category_orders={"feature": show.sort_values("psi")["feature"].tolist()},
    )
    fig.add_vline(x=0.10, line_dash="dot", line_color="#f59e0b")
    fig.add_vline(x=0.25, line_dash="dot", line_color="#ef4444")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="PSI (dotted: 0.10 moderate, 0.25 significant)")
    st.plotly_chart(fig, width="stretch")
with right:
    st.dataframe(
        drift_report[["feature", "psi", "status"]].rename(columns={"psi": "PSI", "status": "Status"}),
        hide_index=True, width="stretch", height=380,
    )

# --- distributions ---
st.subheader("Distribution: training reference vs production")
feat_options = [PREDICTION_COL] + [c for c in NUMERIC_MONITOR_FEATURES if c in logs.columns]
sel = st.selectbox("Feature", feat_options, index=0)
d1, d2 = st.columns(2)
with d1:
    if sel in reference.columns and sel in logs.columns:
        st.plotly_chart(_distribution_figure(reference[sel], logs[sel], sel), width="stretch")
with d2:
    cat = CATEGORICAL_MONITOR_FEATURES[0]
    if cat in reference.columns and cat in logs.columns:
        ref_top = reference[cat].value_counts(normalize=True).head(12).rename("reference")
        cur_top = logs[cat].value_counts(normalize=True).rename("production")
        comp = pd.concat([ref_top, cur_top], axis=1).fillna(0).reset_index(names=cat)
        comp = comp.melt(id_vars=cat, var_name="dataset", value_name="share")
        figc = px.bar(comp, x=cat, y="share", color="dataset", barmode="group",
                      color_discrete_map={"reference": "#94a3b8", "production": "#2563eb"},
                      title=f"{cat} mix (top categories)")
        figc.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(figc, width="stretch")

# --- rolling performance ---
if perf.get("n_with_actual", 0) > 0:
    st.subheader("Rolling performance over time")
    freq = st.radio("Bucket", ["D", "W"], index=1, horizontal=True,
                    format_func=lambda f: {"D": "Daily", "W": "Weekly"}[f])
    roll = rolling_performance(logs, freq=freq)
    if not roll.empty:
        f1, f2 = st.columns(2)
        with f1:
            fig1 = px.line(roll, x="period", y="mape", markers=True, title="MAPE over time")
            fig1.add_hline(y=baseline["valid_mape"], line_dash="dot", line_color="#22c55e",
                           annotation_text="baseline")
            fig1.add_hline(y=0.12, line_dash="dot", line_color="#ef4444", annotation_text="critical")
            fig1.update_layout(height=320, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig1, width="stretch")
        with f2:
            fig2 = px.line(roll, x="period", y="within_10_pct", markers=True, title="% within ±10% over time")
            fig2.add_hline(y=baseline["valid_within_10_pct"], line_dash="dot", line_color="#22c55e",
                           annotation_text="baseline")
            fig2.update_layout(height=320, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, width="stretch")

with st.expander("About drift metrics & thresholds"):
    st.markdown(
        "- **PSI** (Population Stability Index): < 0.10 stable · 0.10–0.25 moderate · > 0.25 significant drift.\n"
        "- **Input drift** is a *leading* indicator — it often fires before realized error degrades.\n"
        "- **Residual** = actual − predicted (positive ⇒ underprediction).\n"
        "- Baselines are the `lasso_v1` validation metrics from `Module4_5_Modeling/outputs/metrics_summary.csv`.\n"
        "- For the full statistical report, generate the Evidently HTML: "
        "`python -m monitoring.evidently_report`."
    )
