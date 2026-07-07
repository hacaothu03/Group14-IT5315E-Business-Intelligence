"""API tests for the valuation service.

Run from Module6_7_Deployment/::

    python -m pytest tests/ -q          # if pytest is installed
    python tests/test_api.py            # plain-python fallback (no pytest needed)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Log to a temp file so tests never touch the real prediction log.
os.environ.setdefault("PREDICTION_LOG_PATH", str(Path(tempfile.gettempdir()) / "test_predictions.jsonl"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)

SAMPLE = {
    "Neighborhood": "NridgHt",
    "OverallQual": 8,
    "OverallCond": 5,
    "GrLivArea": 1800,
    "TotalBsmtSF": 900,
    "GarageCars": 2,
    "GarageArea": 480,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "KitchenQual": "Gd",
    "YearBuilt": 2005,
    "YearRemodAdd": 2008,
    "SaleCondition": "Normal",
    "search_year": 2010,
    "search_month": 6,
    "MortgageRate": 4.7,
    "DaysOnMarket": 45,
    "DistanceToCenter": 1.2,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "lasso_v1"


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["model_version"] == "lasso_v1"


def test_predict_happy_path():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_sale_price"] > 0
    # Interval brackets the point estimate.
    assert body["price_lower"] < body["predicted_sale_price"] < body["price_upper"]
    assert body["model_version"] == "lasso_v1"


def test_predict_missing_date_context_is_422():
    bad = {k: v for k, v in SAMPLE.items() if k not in ("search_year", "search_month")}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_out_of_range_quality_is_422():
    bad = dict(SAMPLE, OverallQual=99)
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_confidence_level_widens_interval():
    r80 = client.post("/predict", json=SAMPLE, params={"confidence_level": 0.80}).json()
    r95 = client.post("/predict", json=SAMPLE, params={"confidence_level": 0.95}).json()
    width80 = r80["price_upper"] - r80["price_lower"]
    width95 = r95["price_upper"] - r95["price_lower"]
    assert width95 > width80


def test_age_sensitivity_regression():
    """Regression test for the age default-fill bug: a new home must be valued
    higher than an otherwise-identical old home."""
    new_home = client.post("/predict", json=dict(SAMPLE, YearBuilt=2009, YearRemodAdd=2009)).json()
    old_home = client.post("/predict", json=dict(SAMPLE, YearBuilt=1950, YearRemodAdd=1950)).json()
    assert new_home["predicted_sale_price"] > old_home["predicted_sale_price"], (
        "YearBuilt is being ignored — the age default-fill bug has regressed."
    )


def test_batch():
    r = client.post("/predict/batch", json={"items": [SAMPLE, dict(SAMPLE, OverallQual=5)]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    # Lower quality should not predict higher than higher quality, all else equal.
    assert body["predictions"][1]["predicted_sale_price"] <= body["predictions"][0]["predicted_sale_price"]


def _run_all():
    """Plain-python runner so the suite works without pytest installed."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
