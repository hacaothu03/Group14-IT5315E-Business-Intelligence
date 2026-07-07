# Module 6-7 — Deployment & Monitoring

House-price valuation service built on the Module 4-5 model (`lasso_v1`). This
folder contains:

- **Module 6 — Deployment**: a FastAPI prediction API and a Streamlit demo tool,
  both returning a price estimate **with a confidence range**, packaged for a
  cloud free tier.
- **Module 7 — Monitoring**: data-drift and prediction-error tracking on logged
  requests using [Evidently](https://github.com/evidentlyai/evidently), with
  defined retraining triggers. *(See the "Module 7" section below.)*

The service **does not reimplement feature engineering**. It loads the saved
full sklearn pipeline and calls the official inference path in
`Module4_5_Modeling/src/predict.py`, exactly as the Module 5 handoff requires.

```text
Module6_7_Deployment/
├── app/
│   ├── model_service.py     # loads pipeline, adds confidence interval + logging
│   ├── schemas.py           # Pydantic request/response models
│   └── main.py              # FastAPI app
├── streamlit_app.py         # demo valuation UI (in-process or API-client)
├── monitoring/              # Module 7 (drift + performance) — see below
├── tests/test_api.py        # API test suite (runs with or without pytest)
├── logs/                    # prediction logs (JSONL) written at runtime
├── requirements.txt
├── Dockerfile               # build from repo root
└── render.yaml              # Render blueprint (reference)
```

## Quick start (local)

From this folder (`Module6_7_Deployment/`):

```bash
pip install -r requirements.txt

# 1) API
uvicorn app.main:app --reload --port 8000
#    -> http://localhost:8000/docs   (interactive Swagger UI)

# 2) Demo UI (separate terminal). Runs the model in-process by default.
streamlit run streamlit_app.py
```

Run the tests:

```bash
python tests/test_api.py         # plain python, no pytest needed
# or, if pytest is installed:
python -m pytest tests -q
```

## API reference

| Method | Path             | Purpose                                        |
| ------ | ---------------- | ---------------------------------------------- |
| GET    | `/health`        | Liveness/readiness + model version             |
| GET    | `/model-info`    | Model version, transforms, required fields     |
| POST   | `/predict`       | Single valuation with confidence interval      |
| POST   | `/predict/batch` | Up to 1000 valuations in one call              |
| GET    | `/docs`          | Swagger UI                                      |

### Request

Required: `Neighborhood`, `OverallQual`, `GrLivArea`, `YearBuilt`, `YearRemodAdd`,
**and a date context** — either `valuation_date` (`YYYY-MM-DD`) or both
`search_year` and `search_month`. All other Ames fields are optional and improve
accuracy; unknown fields are accepted and filled from training defaults.

```bash
curl -X POST "http://localhost:8000/predict?confidence_level=0.90" \
  -H "Content-Type: application/json" \
  -d '{
    "Neighborhood": "NridgHt", "OverallQual": 8, "OverallCond": 5,
    "GrLivArea": 1800, "TotalBsmtSF": 900, "GarageCars": 2, "GarageArea": 480,
    "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3, "KitchenQual": "Gd",
    "YearBuilt": 2005, "YearRemodAdd": 2008, "SaleCondition": "Normal",
    "valuation_date": "2010-06-01",
    "MortgageRate": 4.7, "DaysOnMarket": 45, "DistanceToCenter": 1.2
  }'
```

### Response

```json
{
  "request_id": "544fd9ff...",
  "timestamp": "2010-06-01T00:00:00+00:00",
  "predicted_sale_price": 181894.91,
  "currency": "USD",
  "confidence_level": 0.90,
  "price_lower": 151710.62,
  "price_upper": 218084.63,
  "interval_half_width_pct": 0.1825,
  "model_version": "lasso_v1",
  "target_transform": "log1p",
  "inverse_transform": "expm1",
  "approx_validation_mape": 0.0785,
  "valuation_year": 2010,
  "valuation_month": 6,
  "notes": "Estimate for the provided valuation/search date context. Not a legal appraisal."
}
```

### How the confidence range is computed

Residuals are approximately homoscedastic on the `log1p` scale, so we form a
symmetric interval in log space using the **validation log-RMSE** as σ
(`valid_log_rmse` ≈ 0.1103 for `lasso_v1`), then invert with `expm1`:

```text
lower = expm1(log1p(point) − z·σ)      upper = expm1(log1p(point) + z·σ)
```

`z` is the two-sided normal quantile for the requested `confidence_level`
(default **0.80**). The interval is (correctly) asymmetric in dollars.

## Prediction logging

Every prediction is appended as one JSON line to `PREDICTION_LOG_PATH`
(default `logs/predictions.jsonl`), including the request id, timestamp, model
version, predicted price + interval, valuation year/month, the key monitoring
features (raw and engineered, e.g. `total_sf`, `age_at_sale`), and an
`actual_sale_price` slot to be back-filled when ground truth arrives. This log
is the input to Module 7.

## Deployment

The Docker image bundles the API with the Module 4-5 pipeline and the training
schema. **Build from the repository root** (the build context must include
`Module4_5_Modeling/` and `CleanData/`):

```bash
docker build -f Module6_7_Deployment/Dockerfile -t valuation-api .
docker run -p 8000:8000 valuation-api
```

**Render (Docker, free tier)** — copy `render.yaml` to the repo root (Render
auto-detects blueprints at root), or create a Web Service manually: runtime
Docker, Dockerfile `Module6_7_Deployment/Dockerfile`, context `.`, health check
path `/health`. Render provides `$PORT`; the `CMD` honors it.

**Hugging Face Spaces** — Docker SDK Space pointing at this Dockerfile, or a
Streamlit SDK Space that runs `streamlit_app.py` with this `requirements.txt`.

**Streamlit Community Cloud** — point it at
`Module6_7_Deployment/streamlit_app.py` with this `requirements.txt`. It runs the
model **in-process** (no separate API needed) because the whole repo is checked
out, so `Module4_5_Modeling/` and `CleanData/` are available.

To have the Streamlit UI call a deployed API instead of running in-process, set
`API_URL` (env var) or toggle **Use API endpoint** in the sidebar.

## Module 7 — Monitoring

Tracks **data drift**, **prediction drift**, and **realized error** by comparing
logged production requests to the training distribution, and evaluates the
Module 5 retraining triggers.

```text
monitoring/
├── common.py               # shared feature schema, baselines, thresholds
├── reference.py            # builds reference_data.csv from training data
├── drift.py                # PSI + performance metrics + trigger logic (no heavy deps)
├── evidently_report.py     # rich Evidently HTML report (optional)
├── simulate_production.py  # generate demo traffic (stable + drift scenarios)
├── dashboard.py            # Streamlit monitoring dashboard
└── reference_data.csv      # generated reference snapshot
```

### One-time setup + demo

```bash
# 1) Build the training reference (distribution the model was trained on)
python -m monitoring.reference

# 2) Generate demo traffic: a stable period, then a drifted period
python -m monitoring.simulate_production --scenario both --n 300 --reset

# 3) Launch the dashboard
streamlit run monitoring/dashboard.py

# 4) (optional) Rich Evidently HTML report
python -m monitoring.evidently_report      # writes monitoring/reports/evidently_report.html
```

In production you skip step 2 — the dashboard reads the real
`logs/predictions.jsonl` written by the API.

### What it monitors

- **Input drift** — PSI per feature (`OverallQual`, `GrLivArea`, `TotalSF`,
  `AgeAtSale`, `YearsSinceRemodel`, `Neighborhood`, `GarageCars`, `TotalBath`,
  `MortgageRate`, `DaysOnMarket`, `DistanceToCenter`). PSI bands: < 0.10 stable,
  0.10–0.25 moderate, > 0.25 significant.
- **Prediction drift** — PSI of the predicted-price distribution vs training.
- **Realized performance** (when `actual_sale_price` is known) — RMSE, MAE,
  MAPE, % within ±10/20%, and residual analysis, all vs the `lasso_v1`
  validation baseline.
- **Rolling error over time** — MAPE and % within ±10% bucketed by day/week.

### Retraining triggers (Module 5 handoff §10)

| Trigger | Warning | Critical |
| --- | --- | --- |
| MAPE | > 10% | > 12% |
| % within ±10% | < 65% | < 55% |
| Prediction drift (PSI) | ≥ 0.10 | ≥ 0.25 |
| Input feature drift | ≥ 1 feature PSI ≥ 0.25 | ≥ 3 features PSI ≥ 0.25 |

**Retrain** when any trigger is critical, or when ≥ 2 are warnings. The dashboard
shows a single overall status (🟢/🟠/🔴) and a YES/No retraining recommendation.
Note that **input drift is a leading indicator** — in the demo it fires *before*
realized error crosses its threshold.

### Back-filling ground truth

Predictions are logged with `actual_sale_price: null`. When a property's true
sale price is later known, set that field on the matching `request_id` line (the
simulator does this automatically). Performance metrics use only rows where the
actual is present.

## Note: inference bug fixed during Module 6

While wiring up the API we found and fixed a latent inference bug in the shared
path (`Module4_5_Modeling/src/predict.py`): `age_at_sale` / `years_since_remodel`
were pre-filled with the **training median** for any request that didn't pass
them explicitly, which suppressed the pipeline's per-record derivation. Effect:
a 1900s home and a brand-new home received the **same** age term and thus nearly
identical prices (±$8-9k systematic error). The fix lets the fitted
`FeatureEngineer` derive them from `YearBuilt`/`YearRemodAdd` + the date context
(guaranteed present). `tests/test_api.py::test_age_sensitivity_regression` guards
against regressions. This does not change reported training/validation metrics
(age was present and correct in the training data).
