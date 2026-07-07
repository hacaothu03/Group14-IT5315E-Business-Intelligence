"""Drift and performance metrics for Module 7 (pure pandas/numpy).

This is the dependency-free backbone of the monitoring dashboard: it always
works regardless of whether Evidently is installed. It provides:

- Population Stability Index (PSI) for numeric features and prediction drift,
- a categorical drift statistic (PSI over category frequencies),
- realized-performance metrics (RMSE/MAE/MAPE, % within +/-10/20%),
- rolling performance over time,
- retraining-trigger evaluation against the validation baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    ACTUAL_COL,
    CATEGORICAL_MONITOR_FEATURES,
    MAPE_CRITICAL,
    MAPE_WARN,
    N_DRIFTED_FEATURES_CRITICAL,
    NUMERIC_MONITOR_FEATURES,
    PREDICTION_COL,
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    TIMESTAMP_COL,
    WITHIN10_CRITICAL,
    WITHIN10_WARN,
    load_validation_baseline,
)

_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_predictions(log_path: str | Path) -> pd.DataFrame:
    """Load the JSONL prediction log into a DataFrame (empty if missing)."""
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", utc=True)
    return df


# --------------------------------------------------------------------------- #
# Drift statistics
# --------------------------------------------------------------------------- #
def numeric_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using quantile bins from the reference."""
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if len(expected) < 2 or len(actual) < 1:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # constant / near-constant feature
    edges[0], edges[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_perc = np.clip(e_counts / e_counts.sum(), _EPS, None)
    a_perc = np.clip(a_counts / a_counts.sum(), _EPS, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    """PSI over category frequencies (handles unseen categories)."""
    expected = expected.dropna().astype(str)
    actual = actual.dropna().astype(str)
    if expected.empty or actual.empty:
        return 0.0
    categories = set(expected.unique()) | set(actual.unique())
    e_freq = expected.value_counts(normalize=True)
    a_freq = actual.value_counts(normalize=True)
    total = 0.0
    for cat in categories:
        e_p = max(float(e_freq.get(cat, 0.0)), _EPS)
        a_p = max(float(a_freq.get(cat, 0.0)), _EPS)
        total += (a_p - e_p) * np.log(a_p / e_p)
    return float(total)


def _psi_label(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI-based drift table for every monitored feature + the prediction."""
    records: list[dict[str, Any]] = []

    for col in NUMERIC_MONITOR_FEATURES + [PREDICTION_COL]:
        if col in reference.columns and col in current.columns:
            psi = numeric_psi(reference[col], current[col])
            records.append(
                {
                    "feature": col,
                    "type": "prediction" if col == PREDICTION_COL else "numeric",
                    "psi": round(psi, 4),
                    "status": _psi_label(psi),
                    "drifted": psi >= PSI_SIGNIFICANT,
                }
            )
    for col in CATEGORICAL_MONITOR_FEATURES:
        if col in reference.columns and col in current.columns:
            psi = categorical_psi(reference[col], current[col])
            records.append(
                {
                    "feature": col,
                    "type": "categorical",
                    "psi": round(psi, 4),
                    "status": _psi_label(psi),
                    "drifted": psi >= PSI_SIGNIFICANT,
                }
            )

    report = pd.DataFrame.from_records(records)
    if not report.empty:
        report = report.sort_values("psi", ascending=False).reset_index(drop=True)
    return report


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def performance_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Realized regression metrics on rows where actual price is known."""
    if df.empty or ACTUAL_COL not in df.columns or PREDICTION_COL not in df.columns:
        return {"n_with_actual": 0}
    d = df.dropna(subset=[ACTUAL_COL, PREDICTION_COL]).copy()
    d = d[pd.to_numeric(d[ACTUAL_COL], errors="coerce").notna()]
    if d.empty:
        return {"n_with_actual": 0}

    actual = pd.to_numeric(d[ACTUAL_COL], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(d[PREDICTION_COL], errors="coerce").to_numpy(dtype=float)
    residual = actual - pred            # >0 => model underpredicts
    abs_err = np.abs(residual)
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_pct = np.where(actual != 0, abs_err / np.abs(actual), np.nan)

    return {
        "n_with_actual": int(len(d)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(abs_err)),
        "mape": float(np.nanmean(abs_pct)),
        "within_10_pct": float(np.nanmean(abs_pct <= 0.10)),
        "within_20_pct": float(np.nanmean(abs_pct <= 0.20)),
        "mean_residual": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
    }


def rolling_performance(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Performance metrics bucketed by time (default weekly) for trend charts."""
    if df.empty or TIMESTAMP_COL not in df.columns or ACTUAL_COL not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=[ACTUAL_COL, TIMESTAMP_COL]).copy()
    if d.empty:
        return pd.DataFrame()
    d[TIMESTAMP_COL] = pd.to_datetime(d[TIMESTAMP_COL], errors="coerce", utc=True)
    d = d.dropna(subset=[TIMESTAMP_COL])
    out = []
    for period, group in d.groupby(pd.Grouper(key=TIMESTAMP_COL, freq=freq)):
        if group.empty:
            continue
        m = performance_metrics(group)
        if m.get("n_with_actual", 0) == 0:
            continue
        m["period"] = period
        out.append(m)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Retraining triggers
# --------------------------------------------------------------------------- #
def evaluate_triggers(
    drift_report: pd.DataFrame,
    perf: dict[str, Any],
    baseline: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate the Module 5 handoff retraining triggers. Returns one row per
    trigger with severity: 'ok' | 'warning' | 'critical'."""
    baseline = baseline or load_validation_baseline()
    triggers: list[dict[str, Any]] = []

    # 1. MAPE
    mape = perf.get("mape")
    if mape is None:
        triggers.append({"trigger": "MAPE", "severity": "ok",
                         "detail": "No actuals yet — cannot evaluate.", "value": None})
    else:
        sev = "critical" if mape > MAPE_CRITICAL else "warning" if mape > MAPE_WARN else "ok"
        triggers.append({"trigger": "MAPE", "severity": sev,
                         "detail": f"MAPE {mape:.2%} vs baseline {baseline['valid_mape']:.2%} "
                                   f"(warn >{MAPE_WARN:.0%}, critical >{MAPE_CRITICAL:.0%}).",
                         "value": round(mape, 4)})

    # 2. % within +/-10%
    w10 = perf.get("within_10_pct")
    if w10 is None:
        triggers.append({"trigger": "Within +/-10%", "severity": "ok",
                         "detail": "No actuals yet — cannot evaluate.", "value": None})
    else:
        sev = "critical" if w10 < WITHIN10_CRITICAL else "warning" if w10 < WITHIN10_WARN else "ok"
        triggers.append({"trigger": "Within +/-10%", "severity": sev,
                         "detail": f"{w10:.1%} within +/-10% vs baseline "
                                   f"{baseline['valid_within_10_pct']:.1%} (warn <{WITHIN10_WARN:.0%}).",
                         "value": round(w10, 4)})

    # 3. Prediction-distribution drift
    pred_row = drift_report.loc[drift_report["feature"].eq(PREDICTION_COL)] if not drift_report.empty else pd.DataFrame()
    if pred_row.empty:
        triggers.append({"trigger": "Prediction drift", "severity": "ok",
                         "detail": "No prediction-drift signal available.", "value": None})
    else:
        psi = float(pred_row.iloc[0]["psi"])
        sev = "critical" if psi >= PSI_SIGNIFICANT else "warning" if psi >= PSI_MODERATE else "ok"
        triggers.append({"trigger": "Prediction drift", "severity": sev,
                         "detail": f"Predicted-price PSI {psi:.3f} "
                                   f"(warn >={PSI_MODERATE}, critical >={PSI_SIGNIFICANT}).",
                         "value": psi})

    # 4. Input feature drift
    if drift_report.empty:
        triggers.append({"trigger": "Feature drift", "severity": "ok",
                         "detail": "No feature-drift signal available.", "value": None})
    else:
        feats = drift_report.loc[drift_report["feature"].ne(PREDICTION_COL)]
        drifted = feats.loc[feats["drifted"]]
        n = int(len(drifted))
        sev = "critical" if n >= N_DRIFTED_FEATURES_CRITICAL else "warning" if n >= 1 else "ok"
        names = ", ".join(drifted["feature"].tolist()) if n else "none"
        triggers.append({"trigger": "Feature drift", "severity": sev,
                         "detail": f"{n} feature(s) with significant drift (PSI>={PSI_SIGNIFICANT}): {names}.",
                         "value": n})

    return triggers


def overall_status(triggers: list[dict[str, Any]]) -> str:
    severities = {t["severity"] for t in triggers}
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    return "ok"


def recommend_retraining(triggers: list[dict[str, Any]]) -> bool:
    """Retrain when any trigger is critical, or when >= 2 are in warning."""
    severities = [t["severity"] for t in triggers]
    return ("critical" in severities) or (severities.count("warning") >= 2)
