"""Data cleansing pipeline for the Kaggle Ames house-prices dataset.

The pipeline is intentionally model-agnostic: it produces cleaned tabular data
that can feed simple regressors such as linear regression or decision trees.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AUGMENTED_FEATURE_COLUMNS = [
    "AgeAtSale",
    "YearsSinceRemodel",
    "MortgageRate",
    "DaysOnMarket",
    "DistanceToCenter",
]

NONE_CATEGORICAL_COLUMNS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

ZERO_NUMERIC_COLUMNS = [
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
]

ABSENT_CATEGORY = "NoFeature"

MODE_CATEGORICAL_COLUMNS = [
    "MSZoning",
    "Utilities",
    "Exterior1st",
    "Exterior2nd",
    "Electrical",
    "KitchenQual",
    "Functional",
    "SaleType",
]

KNOWN_TRAIN_OUTLIER_RULE = (
    "(GrLivArea > 4000) & (SalePrice < 300000)"
)


def _load_raw_data(
    raw_dir: Path,
    train_file: Path | None = None,
    test_file: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    train_path = train_file or raw_dir / (
        "train_v2.csv" if (raw_dir / "train_v2.csv").exists() else "train.csv"
    )
    test_path = test_file or raw_dir / (
        "test_v2.csv" if (raw_dir / "test_v2.csv").exists() else "test.csv"
    )
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train/test input files inside {raw_dir.resolve()}"
        )

    input_files = {
        "train": str(train_path),
        "test": str(test_path),
    }
    return pd.read_csv(train_path), pd.read_csv(test_path), input_files


def _missing_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        column: int(count)
        for column, count in df.isna().sum().sort_values(ascending=False).items()
        if count > 0
    }


def _mode(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.mode(dropna=True).iloc[0]


def _normalize_categorical_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[column] = cleaned[column].str.strip()
        cleaned[column] = cleaned[column].replace({"": np.nan})
    return cleaned


def _impute_domain_absence(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in NONE_CATEGORICAL_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(ABSENT_CATEGORY)

    for column in ZERO_NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(0)

    if "GarageYrBlt" in cleaned.columns:
        cleaned["GarageYrBlt"] = cleaned["GarageYrBlt"].fillna(0)

    return cleaned


def _impute_lot_frontage(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train_features, test_features], ignore_index=True)
    if not {"Neighborhood", "LotFrontage"}.issubset(combined.columns):
        return train_features, test_features

    neighborhood_medians = combined.groupby("Neighborhood")["LotFrontage"].median()
    global_median = float(combined["LotFrontage"].median())

    def fill_frame(df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        by_neighborhood = cleaned["Neighborhood"].map(neighborhood_medians)
        cleaned["LotFrontage"] = cleaned["LotFrontage"].fillna(by_neighborhood)
        cleaned["LotFrontage"] = cleaned["LotFrontage"].fillna(global_median)
        return cleaned

    return fill_frame(train_features), fill_frame(test_features)


def _impute_true_missing(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cleaned_train = train_features.copy()
    cleaned_test = test_features.copy()
    imputations: dict[str, Any] = {}

    for column in MODE_CATEGORICAL_COLUMNS:
        if column in cleaned_train.columns:
            fill_value = "Typ" if column == "Functional" else _mode(cleaned_train[column])
            if fill_value is None:
                fill_value = _mode(cleaned_test[column])
            cleaned_train[column] = cleaned_train[column].fillna(fill_value)
            cleaned_test[column] = cleaned_test[column].fillna(fill_value)
            imputations[column] = fill_value

    numeric_columns = [
        column
        for column in cleaned_train.select_dtypes(include=[np.number]).columns
        if column not in {"Id"}
    ]
    for column in numeric_columns:
        if cleaned_train[column].isna().any() or cleaned_test[column].isna().any():
            fill_value = float(cleaned_train[column].median())
            cleaned_train[column] = cleaned_train[column].fillna(fill_value)
            cleaned_test[column] = cleaned_test[column].fillna(fill_value)
            imputations[column] = fill_value

    return cleaned_train, cleaned_test, imputations


def _fill_by_month_rate(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cleaned_train = train_features.copy()
    cleaned_test = test_features.copy()
    metadata: dict[str, Any] = {}

    if "MortgageRate" not in cleaned_train.columns:
        return cleaned_train, cleaned_test, metadata
    if "MortgageRate" in cleaned_test.columns and not cleaned_test["MortgageRate"].isna().any():
        return cleaned_train, cleaned_test, metadata
    if not {"YrSold", "MoSold"}.issubset(cleaned_train.columns):
        return cleaned_train, cleaned_test, metadata

    rate_source = cleaned_train.dropna(subset=["MortgageRate"])
    rate_by_month = rate_source.groupby(["YrSold", "MoSold"])["MortgageRate"].median()
    global_rate = float(rate_source["MortgageRate"].median())

    if "MortgageRate" not in cleaned_test.columns:
        cleaned_test["MortgageRate"] = np.nan
        metadata["MortgageRate"] = "created_for_test_from_train_year_month_rates"

    missing_rate = cleaned_test["MortgageRate"].isna()
    if missing_rate.any():
        mapped_rates = pd.MultiIndex.from_frame(
            cleaned_test.loc[missing_rate, ["YrSold", "MoSold"]]
        ).map(rate_by_month)
        cleaned_test.loc[missing_rate, "MortgageRate"] = mapped_rates
        cleaned_test["MortgageRate"] = cleaned_test["MortgageRate"].fillna(global_rate)
        metadata["MortgageRateFallback"] = global_rate

    return cleaned_train, cleaned_test, metadata


def _ensure_augmented_features(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cleaned_train = train_features.copy()
    cleaned_test = test_features.copy()
    metadata: dict[str, Any] = {}

    for frame_name, frame in [("train", cleaned_train), ("test", cleaned_test)]:
        if "AgeAtSale" not in frame.columns and {"YrSold", "YearBuilt"}.issubset(frame.columns):
            frame["AgeAtSale"] = frame["YrSold"] - frame["YearBuilt"]
            metadata[f"{frame_name}.AgeAtSale"] = "derived_from_YrSold_minus_YearBuilt"

        if (
            "YearsSinceRemodel" not in frame.columns
            and {"YrSold", "YearRemodAdd"}.issubset(frame.columns)
        ):
            frame["YearsSinceRemodel"] = frame["YrSold"] - frame["YearRemodAdd"]
            metadata[f"{frame_name}.YearsSinceRemodel"] = (
                "derived_from_YrSold_minus_YearRemodAdd"
            )

    cleaned_train, cleaned_test, rate_metadata = _fill_by_month_rate(
        cleaned_train, cleaned_test
    )
    metadata.update(rate_metadata)

    for column in ["DaysOnMarket", "DistanceToCenter"]:
        if column in cleaned_train.columns and column not in cleaned_test.columns:
            fill_value = float(cleaned_train[column].median())
            cleaned_test[column] = fill_value
            metadata[f"test.{column}"] = (
                "test_v2_missing; filled_with_train_median_for_schema_alignment"
            )
            metadata[f"test.{column}.fill_value"] = fill_value

    return cleaned_train, cleaned_test, metadata


def _normalize_domain_rules(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    if {"MasVnrType", "MasVnrArea"}.issubset(cleaned.columns):
        no_masonry = cleaned["MasVnrType"].eq(ABSENT_CATEGORY)
        cleaned.loc[no_masonry, "MasVnrArea"] = cleaned.loc[
            no_masonry, "MasVnrArea"
        ].fillna(0)
        cleaned.loc[cleaned["MasVnrArea"].eq(0), "MasVnrType"] = ABSENT_CATEGORY

    if {"Fireplaces", "FireplaceQu"}.issubset(cleaned.columns):
        cleaned.loc[cleaned["Fireplaces"].eq(0), "FireplaceQu"] = ABSENT_CATEGORY

    if {"PoolArea", "PoolQC"}.issubset(cleaned.columns):
        cleaned.loc[cleaned["PoolArea"].eq(0), "PoolQC"] = ABSENT_CATEGORY

    garage_absent = (
        cleaned.get("GarageType", pd.Series(index=cleaned.index, dtype=object))
        .fillna(ABSENT_CATEGORY)
        .eq(ABSENT_CATEGORY)
    )
    for column in ["GarageFinish", "GarageQual", "GarageCond"]:
        if column in cleaned.columns:
            cleaned.loc[garage_absent, column] = ABSENT_CATEGORY
    for column in ["GarageYrBlt", "GarageCars", "GarageArea"]:
        if column in cleaned.columns:
            cleaned.loc[garage_absent, column] = 0

    basement_area_columns = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]
    if set(basement_area_columns).issubset(cleaned.columns):
        basement_absent = cleaned[basement_area_columns].sum(axis=1).eq(0)
        for column in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]:
            if column in cleaned.columns:
                cleaned.loc[basement_absent, column] = ABSENT_CATEGORY

    if {"GarageYrBlt", "YearBuilt", "YrSold"}.issubset(cleaned.columns):
        impossible_garage_year = cleaned["GarageYrBlt"].gt(cleaned["YrSold"])
        cleaned.loc[impossible_garage_year, "GarageYrBlt"] = cleaned.loc[
            impossible_garage_year, "YearBuilt"
        ]

    if "MSSubClass" in cleaned.columns:
        cleaned["MSSubClass"] = cleaned["MSSubClass"].astype(str)

    return cleaned


def _remove_training_outliers(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not {"GrLivArea", "SalePrice"}.issubset(train.columns):
        return train, pd.DataFrame()

    mask = train["GrLivArea"].gt(4000) & train["SalePrice"].lt(300000)
    return train.loc[~mask].copy(), train.loc[mask].copy()


def _validate_augmented_features(df: pd.DataFrame) -> dict[str, int]:
    validation: dict[str, int] = {}

    if {"AgeAtSale", "YrSold", "YearBuilt"}.issubset(df.columns):
        validation["AgeAtSaleMismatch"] = int(
            (df["AgeAtSale"] != df["YrSold"] - df["YearBuilt"]).sum()
        )
        validation["NegativeAgeAtSale"] = int(df["AgeAtSale"].lt(0).sum())

    if {"YearsSinceRemodel", "YrSold", "YearRemodAdd"}.issubset(df.columns):
        validation["YearsSinceRemodelMismatch"] = int(
            (df["YearsSinceRemodel"] != df["YrSold"] - df["YearRemodAdd"]).sum()
        )
        validation["NegativeYearsSinceRemodel"] = int(
            df["YearsSinceRemodel"].lt(0).sum()
        )

    if {"MortgageRate", "YrSold", "MoSold"}.issubset(df.columns):
        rates_per_month = df.groupby(["YrSold", "MoSold"])["MortgageRate"].nunique()
        validation["MortgageRateMonthPairsWithMultipleRates"] = int(
            rates_per_month.gt(1).sum()
        )

    return validation


def _build_report(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    clean_train: pd.DataFrame,
    clean_test: pd.DataFrame,
    removed_outliers: pd.DataFrame,
    imputations: dict[str, Any],
    input_files: dict[str, str],
    augmented_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "raw_shape": {
            "train": list(raw_train.shape),
            "test": list(raw_test.shape),
        },
        "clean_shape": {
            "train": list(clean_train.shape),
            "test": list(clean_test.shape),
        },
        "input_files": input_files,
        "augmented_features": {
            "available_columns": [
                column
                for column in AUGMENTED_FEATURE_COLUMNS
                if column in clean_train.columns or column in clean_test.columns
            ],
            "handling": augmented_metadata,
            "validation": {
                "train": _validate_augmented_features(clean_train),
                "test": _validate_augmented_features(clean_test),
            },
        },
        "missing_before": {
            "train": _missing_counts(raw_train),
            "test": _missing_counts(raw_test),
        },
        "missing_after": {
            "train": _missing_counts(clean_train),
            "test": _missing_counts(clean_test),
        },
        "duplicate_ids_before": {
            "train": int(raw_train["Id"].duplicated().sum()),
            "test": int(raw_test["Id"].duplicated().sum()),
        },
        "removed_training_outliers": {
            "rule": KNOWN_TRAIN_OUTLIER_RULE,
            "count": int(len(removed_outliers)),
            "ids": removed_outliers.get("Id", pd.Series(dtype=int)).astype(int).tolist(),
        },
        "imputations": imputations,
    }


def _write_markdown_summary(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Data Cleaning Summary",
        "",
        "## Shapes",
        f"- Raw train/test: {report['raw_shape']['train']} / {report['raw_shape']['test']}",
        f"- Clean train/test: {report['clean_shape']['train']} / {report['clean_shape']['test']}",
        "",
        "## Main Decisions",
        "- Filled domain-specific absence values (`No basement`, `No garage`, `No pool`, etc.) with `NoFeature` or `0` instead of treating them as random missing values.",
        "- Imputed `LotFrontage` by neighborhood median with a global median fallback.",
        "- Imputed remaining true categorical gaps from training modes, with `Functional` defaulting to `Typ`.",
        "- Preserved or derived augmented features (`AgeAtSale`, `YearsSinceRemodel`, `MortgageRate`, `DaysOnMarket`, `DistanceToCenter`) when v2 inputs are available.",
        "- Normalized inconsistent domain relationships, including garage/basement absence, zero pool/fireplace quality, masonry veneer area/type, and impossible year values.",
        f"- Removed training outliers using `{report['removed_training_outliers']['rule']}`.",
        "",
        "## Outliers Removed",
        f"- Count: {report['removed_training_outliers']['count']}",
        f"- Ids: {report['removed_training_outliers']['ids']}",
        "",
        "## Missing Values After Cleaning",
        f"- Train: {report['missing_after']['train']}",
        f"- Test: {report['missing_after']['test']}",
        "",
        "## Augmented Features",
        f"- Available: {report['augmented_features']['available_columns']}",
        f"- Handling: {report['augmented_features']['handling']}",
        f"- Validation: {report['augmented_features']['validation']}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_cleaning_pipeline(
    raw_dir: Path,
    output_dir: Path,
    train_file: Path | None = None,
    test_file: Path | None = None,
) -> dict[str, Any]:
    raw_train, raw_test, input_files = _load_raw_data(raw_dir, train_file, test_file)
    raw_train = _normalize_categorical_strings(raw_train)
    raw_test = _normalize_categorical_strings(raw_test)

    raw_train = raw_train.drop_duplicates(subset=["Id"]).copy()
    raw_test = raw_test.drop_duplicates(subset=["Id"]).copy()

    train_target = raw_train["SalePrice"].copy()
    train_features = raw_train.drop(columns=["SalePrice"])
    test_features = raw_test.copy()
    test_target = None
    if "SalePrice" in test_features.columns:
        test_target = test_features["SalePrice"].copy()
        test_features = test_features.drop(columns=["SalePrice"])

    train_features, test_features, augmented_metadata = _ensure_augmented_features(
        train_features, test_features
    )
    train_features, test_features = _impute_lot_frontage(train_features, test_features)
    train_features = _impute_domain_absence(train_features)
    test_features = _impute_domain_absence(test_features)
    train_features, test_features, imputations = _impute_true_missing(
        train_features, test_features
    )
    train_features = _normalize_domain_rules(train_features)
    test_features = _normalize_domain_rules(test_features)

    clean_train = pd.concat([train_features, train_target], axis=1)
    clean_train, removed_outliers = _remove_training_outliers(clean_train)
    clean_test = test_features
    if test_target is not None:
        clean_test = pd.concat([clean_test, test_target], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_train.to_csv(output_dir / "train_cleaned.csv", index=False)
    clean_test.to_csv(output_dir / "test_cleaned.csv", index=False)

    report = _build_report(
        raw_train=raw_train,
        raw_test=raw_test,
        clean_train=clean_train,
        clean_test=clean_test,
        removed_outliers=removed_outliers,
        imputations=imputations,
        input_files=input_files,
        augmented_metadata=augmented_metadata,
    )
    (output_dir / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    _write_markdown_summary(report, output_dir / "cleaning_summary.md")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the Kaggle Ames house-price train/test datasets."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--test-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_cleaning_pipeline(
        args.raw_dir,
        args.output_dir,
        train_file=args.train_file,
        test_file=args.test_file,
    )
    print(json.dumps(report["clean_shape"], indent=2))
    print(f"Wrote cleaned data and reports to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
