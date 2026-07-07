"""Feature engineering and sklearn preprocessing pipeline helpers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price import config


def to_snake_case(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = text.strip("_").lower()
    replacements = {
        "1st": "first",
        "2nd": "second",
        "3ssn": "three_ssn",
        "3_ssn": "three_ssn",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"_+", "_", text)
    return text


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [to_snake_case(col) for col in out.columns]
    aliases = {
        "mssub_class": "ms_sub_class",
        "mszoning": "ms_zoning",
        "condition1": "condition_1",
        "condition2": "condition_2",
        "exteriorfirst": "exterior_1st",
        "exteriorsecond": "exterior_2nd",
        "bsmt_fin_type1": "bsmt_fin_type_1",
        "bsmt_fin_type2": "bsmt_fin_type_2",
        "bsmt_fin_sf1": "bsmt_fin_sf_1",
        "bsmt_fin_sf2": "bsmt_fin_sf_2",
    }
    out = out.rename(columns={old: new for old, new in aliases.items() if old in out.columns})
    if "years_since_remod" in out.columns and "years_since_remodel" not in out.columns:
        out = out.rename(columns={"years_since_remod": "years_since_remodel"})
    return out


def _sum_existing(df: pd.DataFrame, columns: list[str]) -> pd.Series | None:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return None
    return df[existing].fillna(0).sum(axis=1)


def _safe_binary(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(int)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Normalize columns, create EDA-driven features, and remove leakage columns."""

    def __init__(
        self,
        include_derived: bool = True,
        include_context_features: bool = True,
        include_days_on_market: bool = True,
        include_distance_to_center: bool = True,
        include_neighborhood: bool = True,
        include_time_features: bool = True,
        include_mortgage_rate: bool = True,
        feature_set: str = "full",
        apply_log_to_skewed: bool = False,
    ) -> None:
        self.include_derived = include_derived
        self.include_context_features = include_context_features
        self.include_days_on_market = include_days_on_market
        self.include_distance_to_center = include_distance_to_center
        self.include_neighborhood = include_neighborhood
        self.include_time_features = include_time_features
        self.include_mortgage_rate = include_mortgage_rate
        self.feature_set = feature_set
        self.apply_log_to_skewed = apply_log_to_skewed

    def fit(self, x: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        df = normalize_column_names(pd.DataFrame(x).copy())
        if "yr_sold" in df.columns and df["yr_sold"].notna().any():
            self.min_train_year_ = int(pd.to_numeric(df["yr_sold"], errors="coerce").min())
        else:
            self.min_train_year_ = 0
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        df = normalize_column_names(pd.DataFrame(x).copy())
        df = self._fill_structural_absence(df)

        if self.include_derived:
            df = self._create_derived_features(df)
        else:
            df = self._drop_known_derived_features(df)

        df = self._encode_ordinals(df)
        df = self._prepare_nominals(df)
        df = self._filter_feature_groups(df)
        df = self._drop_leakage_and_identifiers(df)

        if self.feature_set == "reduced_linear":
            df = df.drop(columns=[col for col in config.REDUCED_LINEAR_DROP_COLUMNS if col in df.columns], errors="ignore")
        elif self.feature_set != "full":
            raise ValueError("feature_set must be 'full' or 'reduced_linear'.")

        if self.apply_log_to_skewed:
            df = self._log_transform_skewed(df)

        return df

    def _fill_structural_absence(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in config.STRUCTURAL_CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna("NoFeature")
        for col in config.STRUCTURAL_NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "yr_sold" in df.columns:
            df["sale_year"] = df["yr_sold"]
        if "mo_sold" in df.columns:
            df["sale_month"] = df["mo_sold"]
            df["sale_quarter"] = np.ceil(pd.to_numeric(df["mo_sold"], errors="coerce") / 3).astype("Int64")
        if {"yr_sold", "mo_sold"}.issubset(df.columns):
            yr = pd.to_numeric(df["yr_sold"], errors="coerce")
            mo = pd.to_numeric(df["mo_sold"], errors="coerce")
            df["sale_ym_index"] = (yr - getattr(self, "min_train_year_", int(yr.min()))) * 12 + mo

        if {"yr_sold", "year_built"}.issubset(df.columns):
            if "age_at_sale" not in df.columns:
                df["age_at_sale"] = df["yr_sold"] - df["year_built"]
        if {"yr_sold", "year_remod_add"}.issubset(df.columns):
            if "years_since_remodel" not in df.columns:
                df["years_since_remodel"] = df["yr_sold"] - df["year_remod_add"]
        if {"yr_sold", "garage_yr_blt"}.issubset(df.columns):
            df["garage_age_at_sale"] = df["yr_sold"] - df["garage_yr_blt"]

        total_sf = _sum_existing(df, ["total_bsmt_sf", "first_flr_sf", "second_flr_sf"])
        if total_sf is not None:
            df["total_sf"] = total_sf
        above_sf = _sum_existing(df, ["first_flr_sf", "second_flr_sf"])
        if above_sf is not None:
            df["total_above_ground_sf"] = above_sf
        total_bath = _sum_existing(df, ["full_bath"])
        if total_bath is not None:
            df["total_bath"] = total_bath
            if "half_bath" in df.columns:
                df["total_bath"] += 0.5 * df["half_bath"].fillna(0)
            if "bsmt_full_bath" in df.columns:
                df["total_bath"] += df["bsmt_full_bath"].fillna(0)
            if "bsmt_half_bath" in df.columns:
                df["total_bath"] += 0.5 * df["bsmt_half_bath"].fillna(0)
        porch_sf = _sum_existing(df, ["wood_deck_sf", "open_porch_sf", "enclosed_porch", "three_ssn_porch", "screen_porch"])
        if porch_sf is not None:
            df["total_porch_sf"] = porch_sf

        if "total_bsmt_sf" in df.columns:
            df["has_basement"] = _safe_binary(df["total_bsmt_sf"] > 0)
        garage_flags = []
        if "garage_area" in df.columns:
            garage_flags.append(df["garage_area"].fillna(0) > 0)
        if "garage_cars" in df.columns:
            garage_flags.append(df["garage_cars"].fillna(0) > 0)
        if garage_flags:
            df["has_garage"] = _safe_binary(pd.concat(garage_flags, axis=1).any(axis=1))
        if "pool_area" in df.columns:
            df["has_pool"] = _safe_binary(df["pool_area"] > 0)
        if "fireplaces" in df.columns:
            df["has_fireplace"] = _safe_binary(df["fireplaces"] > 0)
        if "fence" in df.columns:
            df["has_fence"] = _safe_binary(~df["fence"].fillna("NoFeature").isin(["NoFeature", "None", "NA"]))
        if "total_porch_sf" in df.columns:
            df["has_porch"] = _safe_binary(df["total_porch_sf"] > 0)
        new_flags = []
        if "sale_type" in df.columns:
            new_flags.append(df["sale_type"].eq("New"))
        if "sale_condition" in df.columns:
            new_flags.append(df["sale_condition"].eq("Partial"))
        if new_flags:
            df["is_new_house"] = _safe_binary(pd.concat(new_flags, axis=1).any(axis=1))

        if {"overall_qual", "gr_liv_area"}.issubset(df.columns):
            df["qual_gr_liv_area"] = df["overall_qual"] * df["gr_liv_area"]
        if {"overall_qual", "total_sf"}.issubset(df.columns):
            df["qual_total_sf"] = df["overall_qual"] * df["total_sf"]
        if {"overall_qual", "age_at_sale"}.issubset(df.columns):
            df["age_qual_interaction"] = df["age_at_sale"] * df["overall_qual"]

        df = self._clip_negative_age_features(df)
        return df

    def _clip_negative_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        age_columns = {
            "age_at_sale": "age_negative_flag",
            "years_since_remodel": "remodel_negative_flag",
            "garage_age_at_sale": "garage_age_negative_flag",
        }
        for age_col, flag_col in age_columns.items():
            if age_col in df.columns:
                df[flag_col] = _safe_binary(df[age_col] < 0)
                df[age_col] = df[age_col].clip(lower=0)
        return df

    def _drop_known_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[col for col in config.DERIVED_FEATURES if col in df.columns], errors="ignore")

    def _encode_ordinals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, mapping in config.ORDINAL_MAPPINGS.items():
            if col in df.columns:
                df[col] = df[col].fillna("NoFeature").map(mapping).fillna(0).astype(float)
        return df

    def _prepare_nominals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ms_sub_class" in df.columns:
            numeric_subclass = pd.to_numeric(df["ms_sub_class"], errors="coerce")
            if numeric_subclass.notna().any():
                df["ms_sub_class"] = numeric_subclass.astype("Int64").astype(str).replace("<NA>", np.nan)
            else:
                df["ms_sub_class"] = df["ms_sub_class"].astype(str).replace({"nan": np.nan, "None": np.nan})
        return df

    def _filter_feature_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols: list[str] = []
        context_cols = [col for col in config.SUPPLEMENTARY_CONTEXT_FEATURES if col in df.columns]
        if not self.include_context_features:
            drop_cols.extend(context_cols)
        else:
            if not self.include_days_on_market:
                drop_cols.append("days_on_market")
            if not self.include_distance_to_center:
                drop_cols.append("distance_to_center")
            if not self.include_mortgage_rate:
                drop_cols.append("mortgage_rate")

        if not self.include_neighborhood:
            drop_cols.append("neighborhood")
        if not self.include_time_features:
            drop_cols.extend([col for col in config.TIME_FEATURES if col in df.columns])

        if drop_cols:
            df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
        return df

    def _drop_leakage_and_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        explicit = [config.ID_COLUMN] + config.LEAKAGE_COLUMNS
        price_like = [
            col
            for col in df.columns
            if col != config.TARGET_COLUMN
            and ("price_per" in col or col in {"saleprice", "sale_price_log", "log_sale_price"})
        ]
        return df.drop(columns=[col for col in explicit + price_like if col in df.columns], errors="ignore")

    def _log_transform_skewed(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in config.SKEWED_NUMERIC_COLUMNS:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.min(skipna=True) >= 0:
                    df[col] = np.log1p(numeric)
        return df


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _make_one_hot_encoder()),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
            ("cat", Pipeline(categorical_steps), make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator: Any,
    include_derived: bool = True,
    include_context_features: bool = True,
    include_days_on_market: bool = True,
    include_distance_to_center: bool = True,
    include_neighborhood: bool = True,
    include_time_features: bool = True,
    include_mortgage_rate: bool = True,
    feature_set: str = "full",
    scale_numeric: bool = False,
    apply_log_to_skewed: bool = False,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FeatureEngineer(
                    include_derived=include_derived,
                    include_context_features=include_context_features,
                    include_days_on_market=include_days_on_market,
                    include_distance_to_center=include_distance_to_center,
                    include_neighborhood=include_neighborhood,
                    include_time_features=include_time_features,
                    include_mortgage_rate=include_mortgage_rate,
                    feature_set=feature_set,
                    apply_log_to_skewed=apply_log_to_skewed,
                ),
            ),
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def get_engineered_feature_frame(
    df: pd.DataFrame,
    include_derived: bool = True,
    include_context_features: bool = True,
    include_days_on_market: bool = True,
    include_distance_to_center: bool = True,
    include_neighborhood: bool = True,
    include_time_features: bool = True,
    include_mortgage_rate: bool = True,
    feature_set: str = "full",
    apply_log_to_skewed: bool = False,
) -> pd.DataFrame:
    transformer = FeatureEngineer(
        include_derived=include_derived,
        include_context_features=include_context_features,
        include_days_on_market=include_days_on_market,
        include_distance_to_center=include_distance_to_center,
        include_neighborhood=include_neighborhood,
        include_time_features=include_time_features,
        include_mortgage_rate=include_mortgage_rate,
        feature_set=feature_set,
        apply_log_to_skewed=apply_log_to_skewed,
    )
    return transformer.fit_transform(df)


def get_transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []
