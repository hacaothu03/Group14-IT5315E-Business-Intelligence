from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from house_price import config
from house_price.data import load_test_data
from house_price.utils import configure_logging, load_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission with a trained final pipeline.")
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--model", default=str(config.MODEL_DIR / "final_pipeline.pkl"))
    args = parser.parse_args()

    configure_logging()
    model = load_artifact(Path(args.model))
    test_df, resolved = load_test_data(args.test_data)
    test_ids = test_df[config.ID_COLUMN] if config.ID_COLUMN in test_df.columns else pd.Series(np.arange(1, len(test_df) + 1))
    pred_log = model.predict(test_df)
    submission = pd.DataFrame({"Id": test_ids.astype(int), "SalePrice": np.maximum(np.expm1(pred_log), 0)})
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.SUBMISSION_DIR / "kaggle_submission.csv", index=False)
    submission.to_csv(config.OUTPUT_DIR / "kaggle_submission.csv", index=False)
    logging.info("Saved submission using %s", resolved.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
