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
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict, train_test_split

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
    include_days_on_market: bool = True,
    include_distance_to_center: bool = True,
    include_neighborhood: bool = True,
    include_time_features: bool = True,
    include_mortgage_rate: bool = True,
    selected_models: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, object, str, pd.DataFrame, pd.Series, np.ndarray, dict[str, object]]:
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
    fitted_pipelines: dict[str, object] = {}

    for spec in available_model_specs(selected_models):
        LOGGER.info("Training %s", spec.name)
        pipeline = build_model_pipeline(
            clone(spec.estimator),
            include_derived=include_derived,
            include_context_features=include_context_features,
            include_days_on_market=include_days_on_market,
            include_distance_to_center=include_distance_to_center,
            include_neighborhood=include_neighborhood,
            include_time_features=include_time_features,
            include_mortgage_rate=include_mortgage_rate,
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
                "include_days_on_market": include_days_on_market,
                "include_distance_to_center": include_distance_to_center,
                "include_neighborhood": include_neighborhood,
                "include_time_features": include_time_features,
                "include_mortgage_rate": include_mortgage_rate,
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

        full_pipeline = build_model_pipeline(
            clone(spec.estimator),
            include_derived=include_derived,
            include_context_features=include_context_features,
            include_days_on_market=include_days_on_market,
            include_distance_to_center=include_distance_to_center,
            include_neighborhood=include_neighborhood,
            include_time_features=include_time_features,
            include_mortgage_rate=include_mortgage_rate,
            feature_set=spec.feature_set,
            scale_numeric=spec.scale_numeric,
            apply_log_to_skewed=spec.apply_log_to_skewed,
        )
        full_pipeline.fit(x, y_log)
        fitted_pipelines[spec.name] = full_pipeline

        if valid_metrics["rmse"] < best_score:
            best_score = valid_metrics["rmse"]
            best_pipeline = full_pipeline
            best_name = spec.name
            best_valid_predictions = valid_pred

    if best_pipeline is None or best_valid_predictions is None:
        raise RuntimeError("No model was trained. Check selected model names and optional dependencies.")

    metrics = pd.DataFrame(metrics_rows).sort_values(["valid_rmse", "cv_rmse_mean"])
    cv_results = pd.DataFrame(cv_rows)
    return metrics, cv_results, best_pipeline, best_name, x_valid, y_valid, best_valid_predictions, fitted_pipelines


def tune_tree_models(
    x: pd.DataFrame,
    y_price: pd.Series,
    include_context_features: bool,
    include_derived: bool = True,
    include_days_on_market: bool = True,
    include_distance_to_center: bool = True,
    include_neighborhood: bool = True,
    include_time_features: bool = True,
    include_mortgage_rate: bool = True,
    n_iter: int = 12,
    cv_splits: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, np.ndarray], pd.DataFrame, pd.Series]:
    """Light randomized tuning for tree/boosted models."""
    y_log = np.log1p(y_price)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y_log,
        test_size=0.20,
        random_state=config.RANDOM_SEED,
    )

    candidates: list[tuple[str, object, dict[str, list[object]]]] = [
        (
            "random_forest_tuned",
            RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=-1),
            {
                "model__n_estimators": [250, 400, 600],
                "model__max_depth": [None, 12, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", 0.5, 0.8],
            },
        )
    ]
    try:
        from xgboost import XGBRegressor

        candidates.append(
            (
                "xgboost_tuned",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=config.RANDOM_SEED,
                    n_jobs=-1,
                ),
                {
                    "model__n_estimators": [300, 500, 800],
                    "model__learning_rate": [0.02, 0.035, 0.05, 0.08],
                    "model__max_depth": [2, 3, 4],
                    "model__subsample": [0.75, 0.85, 1.0],
                    "model__colsample_bytree": [0.75, 0.85, 1.0],
                    "model__reg_lambda": [1, 3, 5, 10],
                    "model__min_child_weight": [1, 3, 5],
                },
            )
        )
    except Exception:
        LOGGER.info("XGBoost not installed; skipping XGBoost tuning.")

    metrics_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    tuned_pipelines: dict[str, object] = {}
    valid_predictions: dict[str, np.ndarray] = {}

    for model_name, estimator, param_distributions in candidates:
        LOGGER.info("Tuning %s with RandomizedSearchCV", model_name)
        pipeline = build_model_pipeline(
            estimator,
            include_derived=include_derived,
            include_context_features=include_context_features,
            include_days_on_market=include_days_on_market,
            include_distance_to_center=include_distance_to_center,
            include_neighborhood=include_neighborhood,
            include_time_features=include_time_features,
            include_mortgage_rate=include_mortgage_rate,
            feature_set="full",
            scale_numeric=False,
            apply_log_to_skewed=False,
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        search.fit(x_train, y_train)
        train_pred = search.predict(x_train)
        valid_pred = search.predict(x_valid)
        train_metrics = regression_metrics(y_train, train_pred)
        valid_metrics = regression_metrics(y_valid, valid_pred)

        best_params = dict(search.best_params_)
        tuning_rows.append(
            {
                "model": model_name,
                "best_cv_log_rmse": float(-search.best_score_),
                "n_iter": n_iter,
                "cv_splits": cv_splits,
                "best_params": best_params,
            }
        )
        metrics_rows.append(
            {
                "model": model_name,
                "include_derived": include_derived,
                "include_context_features": include_context_features,
                "include_days_on_market": include_days_on_market,
                "include_distance_to_center": include_distance_to_center,
                "include_neighborhood": include_neighborhood,
                "include_time_features": include_time_features,
                "include_mortgage_rate": include_mortgage_rate,
                "feature_set": "full",
                "train_rmse": train_metrics["rmse"],
                "valid_rmse": valid_metrics["rmse"],
                "valid_mae": valid_metrics["mae"],
                "valid_mape": valid_metrics["mape"],
                "valid_r2": valid_metrics["r2"],
                "valid_within_10_pct": valid_metrics["within_10_pct"],
                "valid_within_20_pct": valid_metrics["within_20_pct"],
                "valid_log_rmse": valid_metrics["log_rmse"],
                "cv_rmse_mean": np.nan,
                "cv_rmse_std": np.nan,
                "cv_mae_mean": np.nan,
                "cv_mape_mean": np.nan,
                "generalization_gap_rmse": valid_metrics["rmse"] - train_metrics["rmse"],
                "tuning_best_cv_log_rmse": float(-search.best_score_),
            }
        )

        final_estimator = clone(estimator)
        estimator_params = {key.replace("model__", ""): value for key, value in best_params.items()}
        final_estimator.set_params(**estimator_params)
        final_pipeline = build_model_pipeline(
            final_estimator,
            include_derived=include_derived,
            include_context_features=include_context_features,
            include_days_on_market=include_days_on_market,
            include_distance_to_center=include_distance_to_center,
            include_neighborhood=include_neighborhood,
            include_time_features=include_time_features,
            include_mortgage_rate=include_mortgage_rate,
            feature_set="full",
            scale_numeric=False,
            apply_log_to_skewed=False,
        )
        final_pipeline.fit(x, y_log)
        tuned_pipelines[model_name] = final_pipeline
        valid_predictions[model_name] = valid_pred

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(tuning_rows),
        tuned_pipelines,
        valid_predictions,
        x_valid,
        y_valid,
    )


def run_ablation_experiments(x: pd.DataFrame, y_price: pd.Series) -> pd.DataFrame:
    y_log = np.log1p(y_price)
    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    experiments = [
        {
            "experiment": "A_base_cleaned_features_only",
            "include_derived": False,
            "include_context_features": False,
            "include_days_on_market": False,
            "include_distance_to_center": False,
            "include_neighborhood": True,
            "include_time_features": True,
            "include_mortgage_rate": False,
        },
        {
            "experiment": "B_base_plus_derived_features",
            "include_derived": True,
            "include_context_features": False,
            "include_days_on_market": False,
            "include_distance_to_center": False,
            "include_neighborhood": True,
            "include_time_features": True,
            "include_mortgage_rate": False,
        },
        {
            "experiment": "C_derived_plus_all_supplementary",
            "include_derived": True,
            "include_context_features": True,
            "include_days_on_market": True,
            "include_distance_to_center": True,
            "include_neighborhood": True,
            "include_time_features": True,
            "include_mortgage_rate": True,
        },
        {
            "experiment": "D_without_days_on_market",
            "include_derived": True,
            "include_context_features": True,
            "include_days_on_market": False,
            "include_distance_to_center": True,
            "include_neighborhood": True,
            "include_time_features": True,
            "include_mortgage_rate": True,
        },
        {
            "experiment": "E_location_neighborhood_only",
            "include_derived": True,
            "include_context_features": True,
            "include_days_on_market": True,
            "include_distance_to_center": False,
            "include_neighborhood": True,
            "include_time_features": True,
            "include_mortgage_rate": True,
        },
        {
            "experiment": "F_location_distance_only",
            "include_derived": True,
            "include_context_features": True,
            "include_days_on_market": True,
            "include_distance_to_center": True,
            "include_neighborhood": False,
            "include_time_features": True,
            "include_mortgage_rate": True,
        },
        {
            "experiment": "G_location_neighborhood_and_distance",
            "include_derived": True,
            "include_context_features": True,
            "include_days_on_market": True,
            "include_distance_to_center": True,
            "include_neighborhood": True,
            "include_time_features": True,
            "include_mortgage_rate": True,
        },
        {
            "experiment": "H_without_time_and_mortgage",
            "include_derived": True,
            "include_context_features": True,
            "include_days_on_market": True,
            "include_distance_to_center": True,
            "include_neighborhood": True,
            "include_time_features": False,
            "include_mortgage_rate": False,
        },
        {
            "experiment": "I_with_time_without_mortgage",
            "include_derived": True,
            "include_context_features": True,
            "include_days_on_market": True,
            "include_distance_to_center": True,
            "include_neighborhood": True,
            "include_time_features": True,
            "include_mortgage_rate": False,
        },
    ]
    rows = []
    for exp in experiments:
        pipeline = build_model_pipeline(
            RidgeCV(alphas=np.logspace(-3, 3, 25)),
            include_derived=exp["include_derived"],
            include_context_features=exp["include_context_features"],
            include_days_on_market=exp["include_days_on_market"],
            include_distance_to_center=exp["include_distance_to_center"],
            include_neighborhood=exp["include_neighborhood"],
            include_time_features=exp["include_time_features"],
            include_mortgage_rate=exp["include_mortgage_rate"],
            feature_set="reduced_linear",
            scale_numeric=True,
            apply_log_to_skewed=True,
        )
        pred = cross_val_predict(pipeline, x, y_log, cv=cv, n_jobs=None)
        metrics = regression_metrics(y_log, pred)
        rows.append(
            {
                "experiment": exp["experiment"],
                "model": "ridge_regression",
                **exp,
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
