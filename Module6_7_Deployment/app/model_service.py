"""Model service for Module 6 deployment.

This is a thin wrapper around the Module 4-5 inference utility
(``Module4_5_Modeling/src/predict.py``). Per the Module 5 handoff, it does
**not** reimplement any feature engineering — it loads the saved full sklearn
pipeline and calls the official inference path. On top of that it adds three
deployment concerns:

1. a **prediction (confidence) interval**, derived from the validation log-scale
   RMSE, so the API can answer "how confident is that estimate?";
2. **prediction request logging** to a JSONL file, which feeds the Module 7
   monitoring dashboard;
3. an **engineered-feature snapshot** (e.g. ``total_sf``, ``age_at_sale``) taken
   from the fitted ``feature_engineering`` step, so drift can be tracked on the
   same features the model actually sees.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

# --- Locate the repo and expose the Module 4-5 inference code -------------------
THIS_FILE = Path(__file__).resolve()
DEPLOY_DIR = THIS_FILE.parents[1]              # Module6_7_Deployment/
REPO_ROOT = DEPLOY_DIR.parent                  # repository root
MODELING_SRC = REPO_ROOT / "Module4_5_Modeling" / "src"
if str(MODELING_SRC) not in sys.path:
    sys.path.insert(0, str(MODELING_SRC))

# The saved pipeline was pickled with a newer scikit-learn than the runtime.
# The load is functionally compatible (verified by the smoke test), so we
# silence the version-mismatch warning to keep service logs clean.
try:  # pragma: no cover - defensive import
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:  # pragma: no cover
    pass

from house_price import config  # noqa: E402  (import after sys.path setup)
from predict import (  # noqa: E402
    MODEL_VERSION,
    REQUIRED_CORE_FIELDS,
    load_artifacts,
    predict_price,
)

try:  # internal helper of predict.py; used for the engineered snapshot only
    from predict import _prepare_input as _predict_prepare_input  # noqa: E402
except Exception:  # pragma: no cover
    _predict_prepare_input = None


DEFAULT_CONFIDENCE_LEVEL = 0.80

# Flat monitoring features logged with every prediction (Module 5 handoff
# section 8/9). Raw fields come straight from the request; engineered fields are
# read from the fitted feature_engineering transformer.
RAW_MONITOR_FIELDS = [
    "overall_qual",
    "gr_liv_area",
    "neighborhood",
    "garage_cars",
    "total_bsmt_sf",
    "mortgage_rate",
    "days_on_market",
    "distance_to_center",
]
ENGINEERED_MONITOR_FIELDS = [
    "total_sf",
    "age_at_sale",
    "years_since_remodel",
    "total_bath",
]

# Fallback if metrics_summary.csv is unavailable: lasso validation log-RMSE.
FALLBACK_SIGMA_LOG = 0.11031568920740205


def _default_log_path() -> Path:
    env = os.environ.get("PREDICTION_LOG_PATH")
    if env:
        return Path(env)
    return DEPLOY_DIR / "logs" / "predictions.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert numpy / pandas scalars to plain JSON-serialisable values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        fv = float(value)
        return fv if np.isfinite(fv) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):  # pandas NaT / NA
        return None
    return str(value)


class ModelService:
    """Loads the saved pipeline once and serves predictions with intervals."""

    def __init__(
        self,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        log_path: Path | str | None = None,
        enable_logging: bool = True,
    ) -> None:
        self.default_confidence_level = confidence_level
        self.log_path = Path(log_path) if log_path else _default_log_path()
        self.enable_logging = enable_logging
        self._artifacts: dict[str, Any] | None = None
        self._sigma_log: float | None = None
        self._load_lock = Lock()
        self._log_lock = Lock()

    # -- loading ---------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._artifacts is not None:
            return
        with self._load_lock:
            if self._artifacts is not None:
                return
            artifacts = load_artifacts()
            self._sigma_log = self._read_sigma_log()
            self._artifacts = artifacts

    def warm_up(self) -> None:
        """Eagerly load the model (call at API startup)."""
        self._ensure_loaded()

    @property
    def artifacts(self) -> dict[str, Any]:
        self._ensure_loaded()
        assert self._artifacts is not None
        return self._artifacts

    @property
    def sigma_log(self) -> float:
        self._ensure_loaded()
        return self._sigma_log if self._sigma_log else FALLBACK_SIGMA_LOG

    def _read_sigma_log(self) -> float:
        """Validation RMSE on the log1p scale — the interval's sigma."""
        path = config.OUTPUT_DIR / "metrics_summary.csv"
        try:
            metrics = pd.read_csv(path)
            row = metrics.loc[metrics["model"].eq("lasso_regression")]
            if row.empty:
                row = metrics.head(1)
            sigma = float(row.iloc[0]["valid_log_rmse"])
            if np.isfinite(sigma) and sigma > 0:
                return sigma
        except Exception:
            pass
        return FALLBACK_SIGMA_LOG

    def model_info(self) -> dict[str, Any]:
        a = self.artifacts
        return {
            "model_version": a["model_version"],
            "model_type": "Lasso Regression (full sklearn Pipeline)",
            "target_transform": a["target_transform"],
            "inverse_transform": a["inverse_transform"],
            "approx_validation_mape": a.get("approx_validation_mape"),
            "interval_sigma_log": self.sigma_log,
            "default_confidence_level": self.default_confidence_level,
            "required_core_fields": list(REQUIRED_CORE_FIELDS),
            "date_context_options": ["valuation_date", "search_year + search_month"],
            "recommended_fields": [
                "Neighborhood", "OverallQual", "OverallCond", "GrLivArea",
                "TotalBsmtSF", "GarageCars", "GarageArea", "FullBath", "HalfBath",
                "BedroomAbvGr", "KitchenQual", "YearBuilt", "YearRemodAdd",
                "SaleCondition", "MortgageRate", "DaysOnMarket", "DistanceToCenter",
            ],
        }

    # -- prediction ------------------------------------------------------------
    def _interval(self, point_price: float, confidence_level: float) -> tuple[float, float]:
        """Log-normal prediction interval around the point estimate.

        Residuals are approximately homoscedastic on the log1p scale, so the
        interval is symmetric in log space and (correctly) asymmetric in USD.
        """
        level = min(max(confidence_level, 0.50), 0.999)
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        pred_log = np.log1p(max(point_price, 0.0))
        lower = float(np.expm1(pred_log - z * self.sigma_log))
        upper = float(np.expm1(pred_log + z * self.sigma_log))
        return max(lower, 0.0), max(upper, 0.0)

    def _engineered_snapshot(self, prepared: pd.DataFrame | None) -> dict[str, Any]:
        if prepared is None:
            return {}
        try:
            fe = self.artifacts["pipeline"].named_steps.get("feature_engineering")
            if fe is None:
                return {}
            engineered = fe.transform(prepared)
            if not isinstance(engineered, pd.DataFrame):
                return {}
            row = engineered.iloc[0]
            return {
                f: _json_safe(row[f]) for f in ENGINEERED_MONITOR_FIELDS if f in engineered.columns
            }
        except Exception:
            return {}

    def predict(
        self,
        raw_input: dict[str, Any],
        confidence_level: float | None = None,
        log: bool = True,
        request_id: str | None = None,
        timestamp: str | None = None,
        actual_sale_price: float | None = None,
    ) -> dict[str, Any]:
        """Predict sale price (USD) with a confidence interval for one record.

        ``timestamp`` and ``actual_sale_price`` are optional: they let callers
        replay historical valuations or back-fill ground truth into the log
        (used by the Module 7 production simulator and by monitoring backfill).
        """
        self._ensure_loaded()
        level = self.default_confidence_level if confidence_level is None else confidence_level

        # Authoritative point prediction via the official inference path.
        base = predict_price(dict(raw_input), artifacts=self._artifacts, return_details=True)
        point = float(base["predicted_sale_price"])

        # Best-effort prepared frame for the engineered monitoring snapshot.
        prepared = None
        if _predict_prepare_input is not None:
            try:
                prepared, _ = _predict_prepare_input(dict(raw_input), self._artifacts)
            except Exception:
                prepared = None

        lower, upper = self._interval(point, level)
        val_year, val_month = self._valuation_context(prepared, raw_input)

        result = {
            "request_id": request_id or uuid.uuid4().hex,
            "timestamp": timestamp or _utc_now_iso(),
            "predicted_sale_price": round(point, 2),
            "currency": "USD",
            "confidence_level": round(level, 4),
            "price_lower": round(lower, 2),
            "price_upper": round(upper, 2),
            "interval_half_width_pct": round((upper - lower) / 2.0 / point, 4) if point > 0 else None,
            "model_version": base["model_version"],
            "target_transform": base["target_transform"],
            "inverse_transform": base["inverse_transform"],
            "approx_validation_mape": base.get("approx_validation_mape"),
            "valuation_year": val_year,
            "valuation_month": val_month,
            "notes": (
                "Estimate for the provided valuation/search date context. "
                "Not a legal appraisal."
            ),
        }

        if log and self.enable_logging:
            self._write_log(result, raw_input, prepared, actual_sale_price=actual_sale_price)
        return result

    def predict_batch(
        self,
        rows: list[dict[str, Any]],
        confidence_level: float | None = None,
        log: bool = True,
    ) -> list[dict[str, Any]]:
        return [self.predict(r, confidence_level=confidence_level, log=log) for r in rows]

    @staticmethod
    def _valuation_context(prepared: pd.DataFrame | None, raw_input: dict[str, Any]) -> tuple[Any, Any]:
        if prepared is not None:
            try:
                return (
                    _json_safe(prepared.iloc[0].get("yr_sold")),
                    _json_safe(prepared.iloc[0].get("mo_sold")),
                )
            except Exception:
                pass
        # Fall back to whatever the caller supplied.
        yr = raw_input.get("search_year") or raw_input.get("YrSold") or raw_input.get("yr_sold")
        mo = raw_input.get("search_month") or raw_input.get("MoSold") or raw_input.get("mo_sold")
        return _json_safe(yr), _json_safe(mo)

    # -- logging ---------------------------------------------------------------
    def _write_log(
        self,
        result: dict[str, Any],
        raw_input: dict[str, Any],
        prepared: pd.DataFrame | None,
        actual_sale_price: float | None = None,
    ) -> None:
        try:
            record: dict[str, Any] = {
                "request_id": result["request_id"],
                "timestamp": result["timestamp"],
                "model_version": result["model_version"],
                "predicted_sale_price": result["predicted_sale_price"],
                "price_lower": result["price_lower"],
                "price_upper": result["price_upper"],
                "confidence_level": result["confidence_level"],
                "valuation_year": result["valuation_year"],
                "valuation_month": result["valuation_month"],
                # None until ground truth arrives; the simulator/backfill sets it.
                "actual_sale_price": _json_safe(actual_sale_price),
            }
            # Flat raw monitoring features (normalised names) from the prepared row.
            if prepared is not None:
                prow = prepared.iloc[0]
                for f in RAW_MONITOR_FIELDS:
                    if f in prepared.columns:
                        record[f] = _json_safe(prow[f])
            # Engineered monitoring features.
            record.update(self._engineered_snapshot(prepared))
            # Full raw request as provided, for traceability.
            record["raw_input"] = {k: _json_safe(v) for k, v in raw_input.items()}

            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self._log_lock:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception:
            # Logging must never break a prediction response.
            pass


# Module-level singleton for import-and-use in the API and Streamlit app.
_service_singleton: ModelService | None = None


def get_service() -> ModelService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = ModelService()
    return _service_singleton
