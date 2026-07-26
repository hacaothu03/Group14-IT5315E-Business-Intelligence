"""Simulate production traffic so the Module 7 dashboard is demonstrable.

Generates realistic valuation requests from the held-out test data, scores them
through the model service (which logs each prediction), spreads them over a time
window, and back-fills ground-truth sale prices so realized-error metrics can be
computed.

Two scenarios:

- ``stable``: a representative random sample -> production ≈ training, little
  drift, error near the validation baseline.
- ``drift``: a covariate + macro shift (older homes, larger/pricier homes, and a
  mortgage-rate regime ~+2.5 points above anything seen in training). This
  triggers feature/prediction drift and degrades realized error — exactly the
  condition the retraining triggers are meant to catch.

Usage (from Module6_7_Deployment/)::

    python -m monitoring.simulate_production --scenario stable --n 300 --reset
    python -m monitoring.simulate_production --scenario drift  --n 200
    python -m monitoring.simulate_production --scenario both --reset   # stable then drift
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from app.model_service import ModelService  # noqa: E402
from monitoring.common import default_log_path  # noqa: E402


def _load_test_pool() -> pd.DataFrame:
    from house_price import config
    from house_price.features import normalize_column_names

    df = normalize_column_names(pd.read_csv(config.CLEANDATA_PROCESSED_DIR / "test_cleaned.csv"))
    df = df.dropna(subset=["sale_price"]).reset_index(drop=True)
    return df


def _leakage_cols() -> set[str]:
    from house_price import config

    return set(config.LEAKAGE_COLUMNS)


def _row_to_request(row: pd.Series, leakage: set[str], drift: bool, rng: np.random.Generator) -> tuple[dict, float]:
    """Build one request dict (+ its true sale price) from a test row."""
    actual = float(row["sale_price"])
    data = {k: v for k, v in row.to_dict().items() if k not in leakage and pd.notna(v)}

    if drift:
        # Macro regime shift: mortgage rate ~+2.5 pts above the training max (6.76).
        data["mortgage_rate"] = float(row.get("mortgage_rate", 6.0)) + 2.5 + float(rng.normal(0, 0.2))
        # Longer time on market (softer demand under higher rates).
        data["days_on_market"] = float(row.get("days_on_market", 45)) * 1.6 + 20
        # Older housing stock: age up ~25-40 years by moving YearBuilt back.
        shift = int(rng.integers(25, 40))
        if "year_built" in data:
            data["year_built"] = int(max(1872, int(data["year_built"]) - shift))
        if "year_remod_add" in data:
            data["year_remod_add"] = int(max(data["year_built"], int(data["year_remod_add"]) - shift))
        # Let the pipeline recompute age from the new year fields (see predict.py fix).
        data.pop("age_at_sale", None)
        data.pop("years_since_remodel", None)

    return data, actual


def _sample_indices(pool: pd.DataFrame, n: int, drift: bool, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(len(pool))
    replace = n > len(pool)
    if not drift:
        return rng.choice(idx, size=n, replace=replace)
    # Bias toward large, higher-value homes (the model's weaker, high-residual region).
    size = pool["gr_liv_area"].to_numpy(dtype=float)
    price = pool["sale_price"].to_numpy(dtype=float)
    w = (size - size.min()) / (np.ptp(size) + 1e-9) + (price - price.min()) / (np.ptp(price) + 1e-9)
    w = np.clip(w, 1e-3, None)
    w = w / w.sum()
    return rng.choice(idx, size=n, replace=replace, p=w)


def simulate(
    scenario: str,
    n: int,
    days: int,
    seed: int,
    actual_ratio: float,
    log_path: Path,
    time_offset_days: int = 0,
    market_shift: float = 1.0,
) -> int:
    rng = np.random.default_rng(seed)
    pool = _load_test_pool()
    leakage = _leakage_cols()
    drift = scenario == "drift"

    service = ModelService(log_path=log_path)
    service.warm_up()

    # Spread timestamps across a window; drift traffic is placed most recently.
    end = datetime.now(timezone.utc) - timedelta(days=time_offset_days)
    start = end - timedelta(days=days)
    chosen = _sample_indices(pool, n, drift, rng)

    written = 0
    for i, row_idx in enumerate(chosen):
        frac = i / max(n - 1, 1)
        ts = (start + (end - start) * frac).isoformat()
        request, actual = _row_to_request(pool.iloc[int(row_idx)], leakage, drift, rng)
        if drift:
            # Simulated market downturn (concept drift): realized prices fall
            # below what the model — trained on the old regime — expects, so the
            # model systematically overpredicts and realized error degrades.
            actual = actual * market_shift
        include_actual = rng.random() < actual_ratio
        try:
            service.predict(
                request,
                log=True,
                timestamp=ts,
                actual_sale_price=actual if include_actual else None,
            )
            written += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  skip row {row_idx}: {exc}")

    print(f"[{scenario}] wrote {written}/{n} predictions to {log_path}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate production traffic for Module 7 monitoring.")
    parser.add_argument("--scenario", choices=["stable", "drift", "both"], default="both")
    parser.add_argument("--n", type=int, default=300, help="Requests per scenario.")
    parser.add_argument("--days", type=int, default=21, help="Time window to spread requests over.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actual-ratio", type=float, default=1.0,
                        help="Fraction of requests with back-filled ground truth (0-1).")
    parser.add_argument("--market-shift", type=float, default=0.88,
                        help="Multiplier on realized prices in the drift scenario "
                             "(simulated downturn; <1 makes the model overpredict).")
    parser.add_argument("--reset", action="store_true", help="Clear the log before writing.")
    parser.add_argument("--log-path", default=str(default_log_path()))
    args = parser.parse_args()

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset and log_path.exists():
        log_path.unlink()
        print(f"Cleared {log_path}")

    total = 0
    if args.scenario in ("stable", "both"):
        # Stable traffic occupies the earlier part of the window.
        total += simulate("stable", args.n, args.days, args.seed, args.actual_ratio, log_path,
                           time_offset_days=args.days, market_shift=1.0)
    if args.scenario in ("drift", "both"):
        # Drift traffic is the most recent window.
        total += simulate("drift", args.n, args.days, args.seed + 1, args.actual_ratio, log_path,
                           time_offset_days=0, market_shift=args.market_shift)
    print(f"Done. {total} total predictions logged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
