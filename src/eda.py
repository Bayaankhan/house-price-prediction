"""Exploratory data analysis charts. Each function returns a Matplotlib Figure.

We use the Figure class directly (not pyplot) so the code works the same in a
notebook, a script, and on a server with no display.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.config import AREA_UNIT, NUMERIC_FEATURES, REPORTS_DIR, TARGET

_MILLION = 1_000_000


def plot_price_distribution(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    sns.histplot(df[TARGET] / _MILLION, bins=40, kde=True, ax=ax)
    ax.set_title("Distribution of house prices")
    ax.set_xlabel("Price (million PKR)")
    ax.set_ylabel("Number of houses")
    fig.tight_layout()
    return fig


def plot_area_vs_price(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    plot_df = df.assign(price_million=df[TARGET] / _MILLION)
    sns.scatterplot(
        data=plot_df, x="area_marla", y="price_million", hue="city", alpha=0.6, ax=ax
    )
    ax.set_title(f"Area vs price ({AREA_UNIT})")
    ax.set_xlabel(f"Area ({AREA_UNIT})")
    ax.set_ylabel("Price (million PKR)")
    fig.tight_layout()
    return fig


def plot_price_by_city(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    plot_df = df.assign(price_million=df[TARGET] / _MILLION)
    sns.boxplot(data=plot_df, x="city", y="price_million", ax=ax)
    ax.set_title("Price by city")
    ax.set_xlabel("City")
    ax.set_ylabel("Price (million PKR)")
    fig.tight_layout()
    return fig


def plot_price_by_bedrooms(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    plot_df = df.assign(price_million=df[TARGET] / _MILLION)
    sns.barplot(data=plot_df, x="bedrooms", y="price_million", errorbar=None, ax=ax)
    ax.set_title("Average price by number of bedrooms")
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Average price (million PKR)")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(7, 5.5))
    ax = fig.subplots()
    corr = df[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation between numeric columns")
    fig.tight_layout()
    return fig


def save_all_eda_charts(df: pd.DataFrame, reports_dir: Path = REPORTS_DIR) -> list[Path]:
    """Create every EDA chart and save it as a PNG in reports/."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "eda_price_distribution.png": plot_price_distribution,
        "eda_area_vs_price.png": plot_area_vs_price,
        "eda_price_by_city.png": plot_price_by_city,
        "eda_price_by_bedrooms.png": plot_price_by_bedrooms,
        "eda_correlation_heatmap.png": plot_correlation_heatmap,
    }
    saved = []
    for filename, make_chart in charts.items():
        path = reports_dir / filename
        make_chart(df).savefig(path, dpi=130)
        saved.append(path)
    return saved


def describe_by_city(df: pd.DataFrame) -> pd.DataFrame:
    """Average and median price per city, in million PKR."""
    summary = df.groupby("city")[TARGET].agg(["count", "mean", "median"])
    summary[["mean", "median"]] = (summary[["mean", "median"]] / _MILLION).round(2)
    return summary.rename(
        columns={"mean": "mean_price_million", "median": "median_price_million"}
    )
