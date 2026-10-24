"""Metrics, comparison chart, and re-evaluation of the saved model.

Run (from the project root) to re-check the saved model on the same test split:
    python -m src.evaluate
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import CLEAN_DATA_PATH, METADATA_PATH, MODEL_PATH, REPORTS_DIR
from src.data_preprocessing import split_data


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE and R2 for one set of predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_model_comparison(results: pd.DataFrame) -> Figure:
    """Bar charts of test RMSE and test R2 for every model.

    `results` must have columns: model, test_RMSE, test_R2.
    """
    fig = Figure(figsize=(10, 4))
    ax_rmse, ax_r2 = fig.subplots(1, 2)

    ax_rmse.bar(results["model"], results["test_RMSE"] / 1_000_000, color="#4C78A8")
    ax_rmse.set_title("Test RMSE (lower is better)")
    ax_rmse.set_ylabel("Million PKR")
    ax_rmse.tick_params(axis="x", rotation=25)

    ax_r2.bar(results["model"], results["test_R2"], color="#59A14F")
    ax_r2.set_title("Test R² (higher is better)")
    ax_r2.set_ylim(min(0, results["test_R2"].min()) - 0.05, 1.0)
    ax_r2.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true, y_pred, model_name: str) -> Figure:
    """Scatter of real vs predicted price. Points near the line are good."""
    fig = Figure(figsize=(5.5, 5.5))
    ax = fig.subplots()
    actual = np.asarray(y_true) / 1_000_000
    predicted = np.asarray(y_pred) / 1_000_000
    ax.scatter(actual, predicted, alpha=0.5, s=14)
    limit = max(actual.max(), predicted.max())
    ax.plot([0, limit], [0, limit], color="red", linestyle="--", label="perfect prediction")
    ax.set_xlabel("Actual price (million PKR)")
    ax.set_ylabel("Predicted price (million PKR)")
    ax.set_title(f"{model_name}: actual vs predicted (test set)")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    if not MODEL_PATH.exists() or not CLEAN_DATA_PATH.exists():
        raise SystemExit("Model or cleaned data not found. Run `python -m src.train` first.")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    pipeline = joblib.load(MODEL_PATH)
    clean_df = pd.read_csv(CLEAN_DATA_PATH)

    # Same seed + same cleaned data => exactly the same test rows as in training.
    _, X_test, _, y_test = split_data(clean_df)
    predictions = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, predictions)

    print(f"Model: {metadata['model_name']}  (data: {metadata['data_label']})")
    for name, value in metrics.items():
        print(f"  {name}: {value:,.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_actual_vs_predicted(y_test, predictions, metadata["model_name"]).savefig(
        REPORTS_DIR / "final_model_actual_vs_predicted.png", dpi=130
    )
    (REPORTS_DIR / "final_model_test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved chart and metrics to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
