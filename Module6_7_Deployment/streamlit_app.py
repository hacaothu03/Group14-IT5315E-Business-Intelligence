"""Streamlit property-valuation demo (Module 6).

A simple tool an agent or buyer could use to get an instant price estimate with
a confidence range. Runs in two modes:

- **In-process** (default): imports the model service directly. Ideal for
  Streamlit Community Cloud — no separate API process needed.
- **API client**: set ``API_URL`` (env var or the sidebar) to call a running
  FastAPI service instead.

Run::

    streamlit run streamlit_app.py     # from Module6_7_Deployment/
"""

from __future__ import annotations

import os
from datetime import date

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Property Valuation Tool", page_icon="🏠", layout="wide")

# Ames neighborhoods (fallback list; refined from training data when available).
NEIGHBORHOODS = [
    "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert", "NridgHt",
    "Sawyer", "NWAmes", "SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge",
    "Timber", "IDOTRR", "ClearCr", "StoneBr", "SWISU", "MeadowV", "Blmngtn",
    "BrDale", "Veenker", "NPkVill", "Blueste",
]
KITCHEN_QUAL = ["Ex", "Gd", "TA", "Fa", "Po"]
SALE_CONDITION = ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"]


@st.cache_resource(show_spinner="Loading model...")
def _load_service():
    """In-process model service (cached across reruns)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from app.model_service import get_service

    svc = get_service()
    svc.warm_up()
    return svc


@st.cache_data(show_spinner=False)
def _neighborhood_options() -> list[str]:
    """Prefer the actual neighborhoods in the training data; fall back otherwise."""
    try:
        import sys
        from pathlib import Path

        import pandas as pd

        sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / "Module4_5_Modeling" / "src"))
        from house_price import config

        df = pd.read_csv(config.CLEANDATA_PROCESSED_DIR / "train_cleaned.csv", usecols=lambda c: c.lower() == "neighborhood")
        col = df.columns[0]
        vals = sorted(str(v) for v in df[col].dropna().unique())
        return vals or NEIGHBORHOODS
    except Exception:
        return sorted(NEIGHBORHOODS)


def _predict_in_process(payload: dict, confidence_level: float) -> dict:
    svc = _load_service()
    return svc.predict(payload, confidence_level=confidence_level)


def _predict_via_api(payload: dict, confidence_level: float, api_url: str) -> dict:
    import requests

    resp = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json=payload,
        params={"confidence_level": confidence_level},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _interval_figure(point: float, lower: float, upper: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=point,
            number={"prefix": "$", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [lower * 0.9, upper * 1.1], "tickprefix": "$", "tickformat": ",.0f"},
                "bar": {"color": "#2563eb"},
                "steps": [{"range": [lower, upper], "color": "#dbeafe"}],
                "threshold": {"line": {"color": "#1e3a8a", "width": 3}, "value": point},
            },
            title={"text": "Estimated sale price"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ----------------------------------------------------------------------------
st.title("🏠 Property Valuation Tool")
st.caption(
    "Instant market-price estimate with a confidence range. Model `lasso_v1` "
    "(Module 4-5). This is a decision-support estimate, **not a legal appraisal**."
)

with st.sidebar:
    st.header("Settings")
    default_api = os.environ.get("API_URL", "")
    use_api = st.toggle("Use API endpoint", value=bool(default_api))
    api_url = ""
    if use_api:
        api_url = st.text_input("API URL", value=default_api or "http://localhost:8000")
    confidence = st.slider("Confidence level", 0.50, 0.99, 0.80, 0.05)
    st.markdown("---")
    st.markdown(
        "**Modes**\n\n"
        "- *In-process*: model loaded inside this app.\n"
        "- *API endpoint*: calls a running FastAPI service."
    )

with st.form("valuation"):
    st.subheader("Property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        neighborhood = st.selectbox("Neighborhood", _neighborhood_options(), index=0)
        overall_qual = st.slider("Overall quality", 1, 10, 7)
        overall_cond = st.slider("Overall condition", 1, 10, 5)
        gr_liv_area = st.number_input("Above-grade living area (sq ft)", 300, 10000, 1700, step=50)
        total_bsmt = st.number_input("Basement area (sq ft)", 0, 6000, 900, step=50)
    with c2:
        garage_cars = st.slider("Garage (cars)", 0, 5, 2)
        garage_area = st.number_input("Garage area (sq ft)", 0, 2000, 480, step=20)
        full_bath = st.slider("Full bathrooms", 0, 5, 2)
        half_bath = st.slider("Half bathrooms", 0, 5, 1)
        bedrooms = st.slider("Bedrooms (above grade)", 0, 10, 3)
    with c3:
        year_built = st.number_input("Year built", 1870, 2026, 2003, step=1)
        year_remod = st.number_input("Year remodeled", 1870, 2026, 2005, step=1)
        kitchen_qual = st.selectbox("Kitchen quality", KITCHEN_QUAL, index=1)
        sale_condition = st.selectbox("Sale condition", SALE_CONDITION, index=0)
        valuation_dt = st.date_input("Valuation date", value=date(2010, 6, 1))

    st.subheader("Market context (synthetic / macro)")
    d1, d2, d3 = st.columns(3)
    with d1:
        mortgage_rate = st.number_input("Mortgage rate (%)", 0.0, 20.0, 4.7, step=0.1)
    with d2:
        days_on_market = st.number_input("Days on market (as of valuation)", 0, 1000, 45, step=5)
    with d3:
        distance = st.number_input("Distance to center", 0.0, 50.0, 5.0, step=0.5)

    submitted = st.form_submit_button("💰 Estimate price", width="stretch")

if submitted:
    payload = {
        "Neighborhood": neighborhood,
        "OverallQual": int(overall_qual),
        "OverallCond": int(overall_cond),
        "GrLivArea": float(gr_liv_area),
        "TotalBsmtSF": float(total_bsmt),
        "GarageCars": int(garage_cars),
        "GarageArea": float(garage_area),
        "FullBath": int(full_bath),
        "HalfBath": int(half_bath),
        "BedroomAbvGr": int(bedrooms),
        "KitchenQual": kitchen_qual,
        "YearBuilt": int(year_built),
        "YearRemodAdd": int(year_remod),
        "SaleCondition": sale_condition,
        "MortgageRate": float(mortgage_rate),
        "DaysOnMarket": float(days_on_market),
        "DistanceToCenter": float(distance),
        "valuation_date": valuation_dt.isoformat(),
    }
    try:
        if use_api and api_url:
            result = _predict_via_api(payload, confidence, api_url)
        else:
            result = _predict_in_process(payload, confidence)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        st.stop()

    point = result["predicted_sale_price"]
    lower, upper = result["price_lower"], result["price_upper"]

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        st.metric("Estimated market price", f"${point:,.0f}")
        st.metric(
            f"{int(result['confidence_level'] * 100)}% confidence range",
            f"${lower:,.0f}  –  ${upper:,.0f}",
        )
        half_pct = result.get("interval_half_width_pct")
        if half_pct is not None:
            st.caption(f"± {half_pct * 100:.1f}% around the point estimate.")
        st.caption(
            f"Model `{result['model_version']}` · "
            f"validation MAPE ≈ {(result.get('approx_validation_mape') or 0) * 100:.1f}%"
        )
    with right:
        st.plotly_chart(_interval_figure(point, lower, upper), width="stretch")

    with st.expander("Raw response"):
        st.json(result)
