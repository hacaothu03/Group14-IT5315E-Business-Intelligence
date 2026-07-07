"""Data loading and validation helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from house_price import config
from house_price.features import normalize_column_names

LOGGER = logging.getLogger(__name__)

CLEANED_NOT_FOUND_MESSAGE = (
    "Cleaned data from Module 3 not found. Expected CleanData/data-v2/processed/train_cleaned.csv "
    "or CLEANED_TRAIN_PATH. Use --allow-raw-fallback only for local smoke tests."
)


@dataclass(frozen=True)
class ResolvedPath:
    path: Path
    source: str
    used_raw_fallback: bool = False


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_clean_data_files() -> tuple[list[Path], list[Path]]:
    """Discover likely cleaned train/test CSVs without modifying CleanData."""
    root = config.CLEANDATA_DIR
    if not root.exists():
        return [], []
    train_candidates = sorted(root.glob("**/*train*clean*.csv")) + sorted(root.glob("**/train.csv"))
    test_candidates = sorted(root.glob("**/*test*clean*.csv")) + sorted(root.glob("**/test.csv"))
    return train_candidates, test_candidates


def _kaggle_candidates(file_name: str) -> list[Path]:
    root = Path("/kaggle/input")
    if not root.exists():
        return []
    return sorted(root.glob(f"**/{file_name}"))


def resolve_train_path(cleaned_train: str | Path | None = None, allow_raw_fallback: bool = False) -> ResolvedPath:
    """Resolve training input without silently falling back to raw data."""
    if cleaned_train:
        candidate = Path(cleaned_train)
        if candidate.exists():
            return ResolvedPath(candidate, "explicit cleaned train path")

    discovered_train, _ = discover_clean_data_files()
    default_cleaned = _first_existing(config.DEFAULT_CLEANED_TRAIN_PATHS + discovered_train)
    if default_cleaned:
        return ResolvedPath(default_cleaned, "default cleaned train path")

    env_path = os.environ.get("CLEANED_TRAIN_PATH")
    if env_path and Path(env_path).exists():
        return ResolvedPath(Path(env_path), "CLEANED_TRAIN_PATH")

    kaggle_cleaned = _kaggle_candidates("train_cleaned.csv")
    if kaggle_cleaned:
        return ResolvedPath(kaggle_cleaned[0], "Kaggle cleaned train path")

    if allow_raw_fallback:
        raw_train = _first_existing(config.RAW_TRAIN_PATHS)
        if raw_train:
            LOGGER.warning("Using raw %s because --allow-raw-fallback was set.", raw_train)
            return ResolvedPath(raw_train, "raw fallback", used_raw_fallback=True)
        kaggle_raw = _kaggle_candidates("train.csv")
        if kaggle_raw:
            LOGGER.warning("Using raw Kaggle train.csv because --allow-raw-fallback was set.")
            return ResolvedPath(kaggle_raw[0], "Kaggle raw fallback", used_raw_fallback=True)

    raise FileNotFoundError(CLEANED_NOT_FOUND_MESSAGE)


def resolve_test_path(test_data: str | Path | None = None) -> ResolvedPath:
    if test_data:
        candidate = Path(test_data)
        if candidate.exists():
            return ResolvedPath(candidate, "explicit test path")

    _, discovered_test = discover_clean_data_files()
    default_test = _first_existing(config.DEFAULT_TEST_PATHS + discovered_test + config.RAW_TEST_PATHS)
    if default_test:
        return ResolvedPath(default_test, "default test path")

    kaggle_test = _kaggle_candidates("test.csv")
    if kaggle_test:
        return ResolvedPath(kaggle_test[0], "Kaggle test path")

    raise FileNotFoundError("Test data not found. Expected data/processed/test_cleaned.csv or data/test.csv.")


def read_csv_normalized(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_column_names(df)
    return df


def merge_context_features_if_available(df: pd.DataFrame) -> pd.DataFrame:
    """Merge known contextual columns from housing_initial_data.csv by Id when available."""
    if not config.CONTEXT_SOURCE_PATH.exists() or config.ID_COLUMN not in df.columns:
        return df

    context_df = read_csv_normalized(config.CONTEXT_SOURCE_PATH)
    if config.ID_COLUMN not in context_df.columns:
        return df

    context_cols = [col for col in config.SUPPLEMENTARY_CONTEXT_FEATURES if col in context_df.columns and col not in df.columns]
    if not context_cols:
        return df

    LOGGER.info("Merging contextual placeholder columns from %s: %s", config.CONTEXT_SOURCE_PATH, context_cols)
    return df.merge(context_df[[config.ID_COLUMN] + context_cols], on=config.ID_COLUMN, how="left")


def load_training_data(
    cleaned_train: str | Path | None = None,
    allow_raw_fallback: bool = False,
) -> tuple[pd.DataFrame, ResolvedPath]:
    resolved = resolve_train_path(cleaned_train, allow_raw_fallback=allow_raw_fallback)
    df = read_csv_normalized(resolved.path)
    df = merge_context_features_if_available(df)
    if resolved.used_raw_fallback:
        df = remove_high_leverage_outliers_if_present(df)
    validate_training_data(df, resolved)
    return df, resolved


def load_test_data(test_data: str | Path | None = None) -> tuple[pd.DataFrame, ResolvedPath]:
    resolved = resolve_test_path(test_data)
    df = read_csv_normalized(resolved.path)
    df = merge_context_features_if_available(df)
    return df, resolved


def validate_training_data(df: pd.DataFrame, resolved: ResolvedPath | None = None) -> None:
    if config.TARGET_COLUMN not in df.columns:
        source = f" in {resolved.path}" if resolved else ""
        raise ValueError(f"Training data{source} must contain SalePrice / sale_price.")
    if config.ID_COLUMN in df.columns and df[config.ID_COLUMN].duplicated().any():
        LOGGER.warning("Duplicate Id values found in training data.")


def remove_high_leverage_outliers_if_present(df: pd.DataFrame) -> pd.DataFrame:
    """Drop only EDA-approved training outliers if a non-cleaned fallback still contains them."""
    if config.ID_COLUMN not in df.columns:
        return df
    outlier_ids = {524, 1299}
    mask = df[config.ID_COLUMN].isin(outlier_ids)
    if mask.any():
        LOGGER.warning("Dropping EDA-approved training outliers from fallback data: %s", sorted(df.loc[mask, config.ID_COLUMN].tolist()))
        return df.loc[~mask].copy()
    return df


def split_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_training_data(df)
    y = df[config.TARGET_COLUMN]
    x = df.drop(columns=[config.TARGET_COLUMN])
    return x, y
