from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC = REPO_ROOT / "Model_Development" / "src"
os.environ.setdefault("HOUSE_PRICE_MODULE_DIR", str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from house_price import config
from house_price.data import load_training_data
from house_price.diagnostics import compute_vif_report
from house_price.features import get_engineered_feature_frame
from house_price.utils import configure_logging, ensure_directories


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Module 4 engineered feature list outputs.")
    parser.add_argument("--cleaned-train", default=None)
    parser.add_argument("--include-context-features", type=parse_bool, default=True)
    parser.add_argument("--include-days-on-market", type=parse_bool, default=True)
    parser.add_argument("--include-distance-to-center", type=parse_bool, default=True)
    parser.add_argument("--include-neighborhood", type=parse_bool, default=True)
    parser.add_argument("--include-time-features", type=parse_bool, default=True)
    parser.add_argument("--include-mortgage-rate", type=parse_bool, default=True)
    parser.add_argument("--allow-raw-fallback", action="store_true")
    args = parser.parse_args()

    configure_logging()
    ensure_directories([config.OUTPUT_DIR])
    try:
        train_df, resolved = load_training_data(args.cleaned_train, args.allow_raw_fallback)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return 2

    logging.info("Loaded data from %s", resolved.path)
    full = get_engineered_feature_frame(
        train_df,
        include_derived=True,
        include_context_features=args.include_context_features,
        include_days_on_market=args.include_days_on_market,
        include_distance_to_center=args.include_distance_to_center,
        include_neighborhood=args.include_neighborhood,
        include_time_features=args.include_time_features,
        include_mortgage_rate=args.include_mortgage_rate,
        feature_set="full",
    )
    reduced = get_engineered_feature_frame(
        train_df,
        include_derived=True,
        include_context_features=args.include_context_features,
        include_days_on_market=args.include_days_on_market,
        include_distance_to_center=args.include_distance_to_center,
        include_neighborhood=args.include_neighborhood,
        include_time_features=args.include_time_features,
        include_mortgage_rate=args.include_mortgage_rate,
        feature_set="reduced_linear",
    )
    full.to_csv(config.OUTPUT_DIR / "engineered_train_full.csv", index=False)
    full.head(0).transpose().reset_index(names="feature").to_csv(config.OUTPUT_DIR / "feature_list_full.csv", index=False)
    reduced.head(0).transpose().reset_index(names="feature").to_csv(
        config.OUTPUT_DIR / "feature_list_reduced_linear.csv", index=False
    )
    compute_vif_report(reduced).to_csv(config.OUTPUT_DIR / "vif_report.csv", index=False)
    logging.info("Saved engineered feature outputs to %s", config.OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
