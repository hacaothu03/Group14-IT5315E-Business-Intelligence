"""Plotting helpers for residual analysis and feature importance."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(residuals: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(residuals["actual"], residuals["predicted"], alpha=0.65, s=24)
    lower = min(residuals["actual"].min(), residuals["predicted"].min())
    upper = max(residuals["actual"].max(), residuals["predicted"].max())
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    ax.set_xlabel("Actual SalePrice")
    ax.set_ylabel("Predicted SalePrice")
    ax.set_title("Actual vs Predicted")
    _save(fig, path)


def plot_residuals_vs_predicted(residuals: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(residuals["predicted"], residuals["residual"], alpha=0.65, s=24)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted SalePrice")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residual Plot")
    _save(fig, path)


def plot_residual_distribution(residuals: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals["residual"], bins=35, color="#4C78A8", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    _save(fig, path)


def plot_error_by_price_bucket(error_by_bucket: pd.DataFrame, path: Path) -> None:
    if error_by_bucket.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(error_by_bucket["price_bucket"].astype(str), error_by_bucket["mape"], color="#59A14F")
    ax.set_xlabel("Actual price bucket")
    ax.set_ylabel("Mean absolute percentage error")
    ax.set_title("Error by Price Bucket")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, path)


def plot_feature_importance(importance: pd.DataFrame, path: Path, top_n: int = 25) -> None:
    if importance.empty:
        return
    top = importance.sort_values("importance", ascending=False).head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance"], color="#F28E2B")
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importance")
    _save(fig, path)
