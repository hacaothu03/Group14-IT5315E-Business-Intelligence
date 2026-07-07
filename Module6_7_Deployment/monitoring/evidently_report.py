"""Generate an Evidently HTML monitoring report (Module 7).

Produces the rich, shareable Evidently report the project brief suggests: data
drift across the monitored features plus regression-quality metrics where ground
truth is available. Written defensively so a missing/incompatible Evidently
install never breaks the pipeline — the dependency-free dashboard
(``monitoring/dashboard.py``) still works in that case.

Usage (from Module6_7_Deployment/)::

    python -m monitoring.evidently_report            # uses default log + reference
    python -m monitoring.evidently_report --out monitoring/reports/report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_DEPLOY_DIR = Path(__file__).resolve().parents[1]
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

from monitoring.common import (  # noqa: E402
    ACTUAL_COL,
    CATEGORICAL_MONITOR_FEATURES,
    NUMERIC_MONITOR_FEATURES,
    PREDICTION_COL,
    default_log_path,
)
from monitoring.drift import load_predictions  # noqa: E402
from monitoring.reference import load_reference  # noqa: E402


def evidently_available() -> bool:
    try:
        import evidently  # noqa: F401

        return True
    except Exception:
        return False


def build_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Build and save an Evidently HTML report. Requires Evidently 0.7.x."""
    from evidently import DataDefinition, Dataset, Regression, Report
    from evidently.presets import DataDriftPreset, RegressionPreset

    numeric = [c for c in NUMERIC_MONITOR_FEATURES if c in reference.columns and c in current.columns]
    categorical = [c for c in CATEGORICAL_MONITOR_FEATURES if c in reference.columns and c in current.columns]

    metrics = [DataDriftPreset()]

    # Include regression quality only when ground truth is present in both frames.
    has_actual = (
        ACTUAL_COL in current.columns
        and ACTUAL_COL in reference.columns
        and current[ACTUAL_COL].notna().any()
    )
    reg = None
    cur = current.copy()
    ref = reference.copy()
    if has_actual:
        cur = cur[cur[ACTUAL_COL].notna()]
        reg = [Regression(target=ACTUAL_COL, prediction=PREDICTION_COL)]
        metrics.append(RegressionPreset())

    # Restrict BOTH frames to exactly the monitored columns. Evidently requires
    # reference and current to share the same schema; the production log carries
    # extra columns (timestamp, request_id, raw_input, ...) that must be dropped.
    keep = numeric + categorical + [PREDICTION_COL] + ([ACTUAL_COL] if has_actual else [])
    keep = [c for c in dict.fromkeys(keep) if c in ref.columns and c in cur.columns]
    ref = ref[keep]
    cur = cur[keep]

    data_def = DataDefinition(
        numerical_columns=[c for c in numeric if c != PREDICTION_COL] + [PREDICTION_COL],
        categorical_columns=categorical,
        regression=reg,
    )

    ref_ds = Dataset.from_pandas(ref, data_definition=data_def)
    cur_ds = Dataset.from_pandas(cur, data_definition=data_def)

    report = Report(metrics=metrics)
    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_path))
    return out_path


def generate(
    log_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path | None:
    if not evidently_available():
        print("Evidently is not installed; skipping HTML report. "
              "Install with `pip install evidently==0.7.14` or use the Streamlit dashboard.")
        return None

    reference = load_reference()
    current = load_predictions(log_path or default_log_path())
    if current.empty:
        print("No predictions logged yet. Run monitoring.simulate_production first.")
        return None

    out = Path(out_path) if out_path else (_DEPLOY_DIR / "monitoring" / "reports" / "evidently_report.html")
    saved = build_report(reference, current, out)
    print(f"Wrote Evidently report: {saved}")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an Evidently monitoring report.")
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = generate(log_path=args.log_path, out_path=args.out)
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
