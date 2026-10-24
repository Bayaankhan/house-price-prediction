"""Train, compare and save the house-price models.

Run (from the project root):
    python -m src.train                          # uses the synthetic sample data
    python -m src.train --data data/raw/my.csv   # uses your own CSV
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    AREA_UNIT,
    CLEAN_DATA_PATH,
    CURRENCY,
    CV_FOLDS,
    METADATA_PATH,
    MIN_ROWS_PER_CITY,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    REPORTS_DIR,
    SAMPLE_DATA_PATH,
    TARGET,
)
from src.data_loader import load_csv
from src.data_preprocessing import build_preprocessor, clean_data, split_data
from src.eda import save_all_eda_charts
from src.evaluate import plot_actual_vs_predicted, plot_model_comparison, regression_metrics


def get_models() -> dict:
    """name -> (estimator, scale_numeric).

    Baseline settings only: with ~1000 rows, heavy tuning would mostly overfit.
    Only Linear Regression gets scaled inputs; trees do not need scaling.
    """
    return {
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree": (
            DecisionTreeRegressor(min_samples_leaf=5, random_state=RANDOM_SEED),
            False,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=200, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
            ),
            False,
        ),
        "Hist Gradient Boosting": (
            HistGradientBoostingRegressor(random_state=RANDOM_SEED),
            False,
        ),
    }


def make_pipeline(estimator, scale_numeric: bool) -> Pipeline:
    """Preprocessing + model in ONE object, so they are always fitted together."""
    return Pipeline(
        [("preprocess", build_preprocessor(scale_numeric)), ("model", estimator)]
    )


def train(
    data_path: Path = SAMPLE_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    reports_dir: Path = REPORTS_DIR,
    processed_path: Path = CLEAN_DATA_PATH,
    cv_folds: int = CV_FOLDS,
    make_charts: bool = True,
) -> dict:
    """Full workflow. Returns a dict with the results table and the chosen model."""
    data_path = Path(data_path)
    raw_df = load_csv(data_path)
    clean_df, cleaning_report = clean_data(raw_df)
    print("Cleaning report:", json.dumps(cleaning_report, indent=2))

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(processed_path, index=False)

    # Split BEFORE fitting anything. The test set stays untouched until the end.
    X_train, X_test, y_train, y_test = split_data(clean_df)
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows, fitted = [], {}
    for name, (estimator, scale_numeric) in get_models().items():
        pipeline = make_pipeline(estimator, scale_numeric)

        # Cross-validation on the TRAINING rows only: a more stable estimate
        # than a single split, and it never touches the test set.
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

        pipeline.fit(X_train, y_train)
        train_metrics = regression_metrics(y_train, pipeline.predict(X_train))
        test_metrics = regression_metrics(y_test, pipeline.predict(X_test))

        fitted[name] = pipeline
        rows.append(
            {
                "model": name,
                "cv_RMSE": -cv_scores["test_rmse"].mean(),
                "cv_RMSE_std": cv_scores["test_rmse"].std(),
                "cv_MAE": -cv_scores["test_mae"].mean(),
                "cv_R2": cv_scores["test_r2"].mean(),
                "train_R2": train_metrics["R2"],
                "test_MAE": test_metrics["MAE"],
                "test_MSE": test_metrics["MSE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_R2": test_metrics["R2"],
            }
        )
        print(f"  trained {name}: cv_RMSE={rows[-1]['cv_RMSE']:,.0f} test_R2={test_metrics['R2']:.4f}")

    results = pd.DataFrame(rows)

    # Choose by CROSS-VALIDATION error (training data), not by looking at the
    # test set. The test set then gives an honest final score for that choice.
    best_name = results.loc[results["cv_RMSE"].idxmin(), "model"]
    best_pipeline = fitted[best_name]
    print(f"Selected model (lowest CV RMSE): {best_name}")

    reports_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(reports_dir / "model_comparison.csv", index=False)
    if make_charts:
        save_all_eda_charts(clean_df, reports_dir)
        plot_model_comparison(results).savefig(reports_dir / "model_comparison.png", dpi=130)
        plot_actual_vs_predicted(
            y_test, best_pipeline.predict(X_test), best_name
        ).savefig(reports_dir / "final_model_actual_vs_predicted.png", dpi=130)

    # Save the WHOLE pipeline (preprocessing + model) so prediction needs no retraining.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    city_counts = clean_df["city"].value_counts()
    best_row = results[results["model"] == best_name].iloc[0]
    metadata = {
        "model_name": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_file": data_path.name,
        "is_synthetic": "synthetic" in data_path.name.lower(),
        "data_label": (
            "SYNTHETIC demo data (invented, not real listings)"
            if "synthetic" in data_path.name.lower()
            else f"User-supplied dataset: {data_path.name}"
        ),
        "currency": CURRENCY,
        "area_unit": AREA_UNIT,
        "n_rows_clean": int(len(clean_df)),
        "cities": sorted(city_counts.index.tolist()),
        "city_counts": {k: int(v) for k, v in city_counts.items()},
        "property_types": sorted(clean_df["property_type"].unique().tolist()),
        "numeric_ranges": {
            col: {"min": float(clean_df[col].min()), "max": float(clean_df[col].max())}
            for col in NUMERIC_FEATURES
        },
        "min_rows_per_city": MIN_ROWS_PER_CITY,
        "test_metrics": {
            "MAE": float(best_row["test_MAE"]),
            "RMSE": float(best_row["test_RMSE"]),
            "R2": float(best_row["test_R2"]),
        },
        "cleaning_report": cleaning_report,
        "random_seed": RANDOM_SEED,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"results": results, "best_model": best_name, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the house price models.")
    parser.add_argument(
        "--data", type=Path, default=SAMPLE_DATA_PATH, help="Path to the CSV dataset."
    )
    args = parser.parse_args()

    outcome = train(args.data)

    display = outcome["results"].copy()
    for col in ["cv_RMSE", "cv_RMSE_std", "cv_MAE", "test_MAE", "test_MSE", "test_RMSE"]:
        display[col] = display[col].round(0).astype("int64")
    for col in ["cv_R2", "train_R2", "test_R2"]:
        display[col] = display[col].round(4)
    print("\nModel comparison (errors in PKR):")
    print(display.to_string(index=False))
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved reports -> {REPORTS_DIR}")


if __name__ == "__main__":
    main()
