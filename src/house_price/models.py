"""Model definitions and training orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from house_price import config
from house_price.evaluation import regression_metrics
from house_price.features import build_model_pipeline, get_transformed_feature_names

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    feature_set: str
    scale_numeric: bool
    apply_log_to_skewed: bool = False


def available_model_specs(selected: Iterable[str] | None = None) -> list[ModelSpec]:
    selected_set = {name.lower() for name in selected} if selected else None
    specs = [
        ModelSpec("linear_regression", LinearRegression(), "reduced_linear", True, True),
        ModelSpec("ridge_regression", RidgeCV(alphas=np.logspace(-3, 3, 25)), "reduced_linear", True, True),
        ModelSpec(
            "lasso_regression",
            LassoCV(alphas=np.logspace(-4, 0, 25), cv=5, random_state=config.RANDOM_SEED, max_iter=20000, n_jobs=-1),
            "reduced_linear",
            True,
            True,
        ),
        ModelSpec(
            "random_forest",
            RandomForestRegressor(
                n_estimators=350,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            ),
            "full",
            False,
            False,
        ),
    ]

    xgb_spec = _optional_boosting_spec()
    if xgb_spec is not None:
        specs.append(xgb_spec)

    if selected_set:
        specs = [spec for spec in specs if spec.name.lower() in selected_set]
    return specs


def _optional_boosting_spec() -> ModelSpec | None:
    try:
        from xgboost import XGBRegressor

        return ModelSpec(
            "xgboost",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=500,
                learning_rate=0.035,
                max_depth=3,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=config.RANDOM_SEED,
                n_jobs=-1,
            ),
            "full",
            False,
            False,
        )
    except Exception:
        try:
            from lightgbm import LGBMRegressor

            return ModelSpec(
                "lightgbm",
                LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.035,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=config.RANDOM_SEED,
                    n_jobs=-1,
                ),
                "full",
                False,
                False,
            )
        except Exception:
            LOGGER.info("XGBoost/LightGBM not installed; skipping optional boosting model.")
            return None


def fit_and_evaluate_models(
    x: pd.DataFrame,
    y_price: pd.Series,
    include_context_features: bool,
    include_derived: bool = True,
    selected_models: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, object, str, pd.DataFrame, pd.Series, np.ndarray]:
    y_log = np.log1p(y_price)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y_log,
        test_size=0.20,
        random_state=config.RANDOM_SEED,
    )

    metrics_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    best_pipeline = None
    best_name = ""
    best_score = float("inf")
    best_valid_predictions = None

    for spec in available_model_specs(selected_models):
        LOGGER.info("Training %s", spec.name)
        pipeline = build_model_pipeline(
            clone(spec.estimator),
            include_derived=include_derived,
            include_context_features=include_context_features,
            feature_set=spec.feature_set,
            scale_numeric=spec.scale_numeric,
            apply_log_to_skewed=spec.apply_log_to_skewed,
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
        cv_pred = cross_val_predict(pipeline, x, y_log, cv=cv, n_jobs=None)
        fold_metrics = []
        for fold, (_, valid_idx) in enumerate(cv.split(x), start=1):
            fold_metric = regression_metrics(y_log.iloc[valid_idx], cv_pred[valid_idx])
            fold_metric.update({"model": spec.name, "fold": fold})
            cv_rows.append(fold_metric)
            fold_metrics.append(fold_metric)

        pipeline.fit(x_train, y_train)
        train_pred = pipeline.predict(x_train)
        valid_pred = pipeline.predict(x_valid)
        train_metrics = regression_metrics(y_train, train_pred)
        valid_metrics = regression_metrics(y_valid, valid_pred)
        cv_rmse = [row["rmse"] for row in fold_metrics]
        cv_mae = [row["mae"] for row in fold_metrics]
        cv_mape = [row["mape"] for row in fold_metrics]

        metrics_rows.append(
            {
                "model": spec.name,
                "include_derived": include_derived,
                "include_context_features": include_context_features,
                "feature_set": spec.feature_set,
                "train_rmse": train_metrics["rmse"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
                "valid_mape": valid_metrics["mape"],
                "valid_r2": valid_metrics["r2"],
                "valid_within_10_pct": valid_metrics["within_10_pct"],
                "valid_within_20_pct": valid_metrics["within_20_pct"],
                "valid_log_rmse": valid_metrics["log_rmse"],
                "cv_rmse_mean": float(np.mean(cv_rmse)),
                "cv_rmse_std": float(np.std(cv_rmse, ddof=1)),
                "cv_mae_mean": float(np.mean(cv_mae)),
                "cv_mape_mean": float(np.mean(cv_mape)),
                "generalization_gap_rmse": valid_metrics["rmse"] - train_metrics["rmse"],
            }
        )

        if valid_metrics["rmse"] < best_score:
            best_score = valid_metrics["rmse"]
            best_pipeline = pipeline
            best_name = spec.name
            best_valid_predictions = valid_pred

    if best_pipeline is None or best_valid_predictions is None:
        raise RuntimeError("No model was trained. Check selected model names and optional dependencies.")

    best_pipeline.fit(x, y_log)
    metrics = pd.DataFrame(metrics_rows).sort_values(["valid_rmse", "cv_rmse_mean"])
    cv_results = pd.DataFrame(cv_rows)
    return metrics, cv_results, best_pipeline, best_name, x_valid, y_valid, best_valid_predictions


def run_ablation_experiments(x: pd.DataFrame, y_price: pd.Series) -> pd.DataFrame:
    y_log = np.log1p(y_price)
    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    experiments = [
        ("A_base_cleaned_features_only", False, False),
        ("B_base_plus_derived_features", True, False),
        ("C_base_plus_derived_plus_contextual", True, True),
    ]
    rows = []
    for experiment, include_derived, include_context in experiments:
        pipeline = build_model_pipeline(
            RidgeCV(alphas=np.logspace(-3, 3, 25)),
            include_derived=include_derived,
            include_context_features=include_context,
            feature_set="reduced_linear",
            scale_numeric=True,
            apply_log_to_skewed=True,
        )
        pred = cross_val_predict(pipeline, x, y_log, cv=cv, n_jobs=None)
        metrics = regression_metrics(y_log, pred)
        rows.append(
            {
                "experiment": experiment,
                "model": "ridge_regression",
                "include_derived": include_derived,
                "include_context_features": include_context,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def feature_importance_frame(pipeline: object) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    feature_names = get_transformed_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        kind = "feature_importance"
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        kind = "coefficient_abs"
    else:
        return pd.DataFrame()
    if len(feature_names) != len(values):
        feature_names = [f"feature_{i}" for i in range(len(values))]
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.abs(values),
            "raw_value": values,
            "importance_type": kind,
        }
    ).sort_values("importance", ascending=False)
