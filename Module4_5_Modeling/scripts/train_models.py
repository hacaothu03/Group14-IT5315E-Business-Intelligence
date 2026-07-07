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
from house_price.data import load_test_data, load_training_data, split_target
from house_price.diagnostics import compute_vif_report
from house_price.evaluation import residual_frame, summarize_errors
from house_price.features import get_engineered_feature_frame
from house_price.models import feature_importance_frame, fit_and_evaluate_models, run_ablation_experiments, tune_tree_models
from house_price.plots import (
    plot_actual_vs_predicted,
    plot_error_by_price_bucket,
    plot_feature_importance,
    plot_residual_distribution,
    plot_residuals_vs_predicted,
)
from house_price.utils import configure_logging, ensure_directories, save_artifact


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Module 5 house price regression models.")
    parser.add_argument("--cleaned-train", type=str, default=None, help="Path to Module 3 cleaned train CSV.")
    parser.add_argument("--test-data", type=str, default=None, help="Path to test CSV for Kaggle submission.")
    parser.add_argument("--include-context-features", type=parse_bool, default=True)
    parser.add_argument("--include-days-on-market", type=parse_bool, default=True)
    parser.add_argument("--include-distance-to-center", type=parse_bool, default=True)
    parser.add_argument("--include-neighborhood", type=parse_bool, default=True)
    parser.add_argument("--include-time-features", type=parse_bool, default=True)
    parser.add_argument("--include-mortgage-rate", type=parse_bool, default=True)
    parser.add_argument("--tune-tree-models", type=parse_bool, default=True)
    parser.add_argument("--tuning-iter", type=int, default=12)
    parser.add_argument("--allow-raw-fallback", action="store_true", help="Allow local raw data smoke tests before Module 3.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model subset: linear_regression ridge_regression lasso_regression random_forest xgboost lightgbm.",
    )
    return parser.parse_args()


def save_feature_lists(train_df: pd.DataFrame, args: argparse.Namespace) -> None:
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
    pd.DataFrame({"feature": full.columns}).to_csv(config.OUTPUT_DIR / "feature_list_full.csv", index=False)
    pd.DataFrame({"feature": reduced.columns}).to_csv(config.OUTPUT_DIR / "feature_list_reduced_linear.csv", index=False)
    compute_vif_report(reduced).to_csv(config.OUTPUT_DIR / "vif_report.csv", index=False)


def make_submission(best_pipeline: object, test_data_arg: str | None) -> None:
    try:
        test_df, resolved_test = load_test_data(test_data_arg)
    except FileNotFoundError as exc:
        logging.warning("Skipping submission because test data was not found: %s", exc)
        return

    test_ids = test_df[config.ID_COLUMN] if config.ID_COLUMN in test_df.columns else pd.Series(np.arange(1, len(test_df) + 1))
    pred_log = best_pipeline.predict(test_df)
    sale_price = np.maximum(np.expm1(pred_log), 0)
    submission = pd.DataFrame({"Id": test_ids.astype(int), "SalePrice": sale_price})
    submission_path = config.SUBMISSION_DIR / "kaggle_submission.csv"
    output_copy_path = config.OUTPUT_DIR / "kaggle_submission.csv"
    submission.to_csv(submission_path, index=False)
    submission.to_csv(output_copy_path, index=False)
    logging.info("Saved Kaggle submission from %s to %s", resolved_test.path, submission_path)


def save_model_checkpoints(fitted_pipelines: dict[str, object], best_name: str) -> pd.DataFrame:
    checkpoint_dir = config.MODEL_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, pipeline in fitted_pipelines.items():
        pipeline_path = checkpoint_dir / f"{model_name}_pipeline.pkl"
        model_path = checkpoint_dir / f"{model_name}_model.pkl"
        save_artifact(pipeline, pipeline_path)
        save_artifact(pipeline.named_steps["model"], model_path)
        try:
            display_pipeline_path = pipeline_path.relative_to(config.REPO_ROOT)
            display_model_path = model_path.relative_to(config.REPO_ROOT)
        except ValueError:
            display_pipeline_path = pipeline_path
            display_model_path = model_path
        rows.append(
            {
                "model": model_name,
                "is_best_model": model_name == best_name,
                "pipeline_path": str(display_pipeline_path),
                "model_path": str(display_model_path),
                "training_scope": "full_clean_training_data",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(config.OUTPUT_DIR / "model_artifacts.csv", index=False)
    return manifest


def main() -> int:
    args = parse_args()
    configure_logging()
    ensure_directories([config.OUTPUT_DIR, config.FIGURE_DIR, config.MODEL_DIR, config.SUBMISSION_DIR])

    try:
        train_df, resolved_train = load_training_data(args.cleaned_train, allow_raw_fallback=args.allow_raw_fallback)
    except FileNotFoundError as exc:
        logging.error(str(exc))
        return 2

    logging.info("Loaded training data from %s (%s)", resolved_train.path, resolved_train.source)
    if resolved_train.used_raw_fallback:
        logging.warning("This run used raw Kaggle train.csv. Do not use these results as final Module 4-5 results.")

    save_feature_lists(train_df, args)
    x, y_price = split_target(train_df)

    metrics, cv_results, best_pipeline, best_name, x_valid, y_valid, valid_pred, fitted_pipelines = fit_and_evaluate_models(
        x,
        y_price,
        include_context_features=args.include_context_features,
        include_derived=True,
        include_days_on_market=args.include_days_on_market,
        include_distance_to_center=args.include_distance_to_center,
        include_neighborhood=args.include_neighborhood,
        include_time_features=args.include_time_features,
        include_mortgage_rate=args.include_mortgage_rate,
        selected_models=args.models,
    )
    tuning_results = pd.DataFrame()
    if args.tune_tree_models and (args.models is None):
        tuned_metrics, tuning_results, tuned_pipelines, tuned_valid_predictions, tuned_x_valid, tuned_y_valid = tune_tree_models(
            x,
            y_price,
            include_context_features=args.include_context_features,
            include_derived=True,
            include_days_on_market=args.include_days_on_market,
            include_distance_to_center=args.include_distance_to_center,
            include_neighborhood=args.include_neighborhood,
            include_time_features=args.include_time_features,
            include_mortgage_rate=args.include_mortgage_rate,
            n_iter=args.tuning_iter,
            cv_splits=3,
        )
        if not tuned_metrics.empty:
            metrics = pd.concat([metrics, tuned_metrics], ignore_index=True, sort=False)
            metrics = metrics.sort_values(["valid_rmse", "cv_rmse_mean"], na_position="last")
            fitted_pipelines.update(tuned_pipelines)
            tuned_best = metrics.iloc[0]["model"]
            if tuned_best in tuned_pipelines:
                best_name = tuned_best
                best_pipeline = tuned_pipelines[tuned_best]
                x_valid = tuned_x_valid
                y_valid = tuned_y_valid
                valid_pred = tuned_valid_predictions[tuned_best]

    ablation = run_ablation_experiments(x, y_price)

    metrics.to_csv(config.OUTPUT_DIR / "metrics_summary.csv", index=False)
    cv_results.to_csv(config.OUTPUT_DIR / "cv_results.csv", index=False)
    ablation.to_csv(config.OUTPUT_DIR / "ablation_results.csv", index=False)
    tuning_results.to_csv(config.OUTPUT_DIR / "tuning_results.csv", index=False)

    residuals = residual_frame(x_valid, y_valid, valid_pred)
    residuals.describe(include="all").transpose().to_csv(config.OUTPUT_DIR / "residual_summary.csv")
    error_by_bucket = summarize_errors(residuals, "price_bucket")
    error_by_neighborhood = summarize_errors(residuals, "neighborhood")
    error_by_overallqual = summarize_errors(residuals, "overall_qual")
    error_by_bucket.to_csv(config.OUTPUT_DIR / "error_by_price_bucket.csv", index=False)
    error_by_neighborhood.to_csv(config.OUTPUT_DIR / "error_by_neighborhood.csv", index=False)
    error_by_overallqual.to_csv(config.OUTPUT_DIR / "error_by_overallqual.csv", index=False)

    importance = feature_importance_frame(best_pipeline)
    importance.to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False)
    save_model_checkpoints(fitted_pipelines, best_name)

    plot_actual_vs_predicted(residuals, config.FIGURE_DIR / "actual_vs_predicted.png")
    plot_residuals_vs_predicted(residuals, config.FIGURE_DIR / "residual_plot.png")
    plot_residual_distribution(residuals, config.FIGURE_DIR / "residual_distribution.png")
    plot_error_by_price_bucket(error_by_bucket, config.FIGURE_DIR / "residuals_by_price_bucket.png")
    plot_feature_importance(importance, config.FIGURE_DIR / "top_feature_importance.png")

    save_artifact(best_pipeline.named_steps["preprocessor"], config.MODEL_DIR / "preprocessing_pipeline.pkl")
    save_artifact(best_pipeline.named_steps["model"], config.MODEL_DIR / "best_model.pkl")
    save_artifact(best_pipeline, config.MODEL_DIR / "final_pipeline.pkl")
    make_submission(best_pipeline, args.test_data)

    logging.info("Best model: %s", best_name)
    logging.info("Saved metrics, residual analysis, figures, model artifacts, and submission outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
