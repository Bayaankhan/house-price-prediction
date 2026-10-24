"""Cleaning, splitting and the scikit-learn preprocessing pipeline.

Order of operations (this order prevents data leakage):
1. clean_data()   - row-level fixes that need no statistics from the data
                    (trim text, remove duplicates, drop impossible values).
2. split_data()   - split into train / test BEFORE anything is learned.
3. build_preprocessor() - imputers, scalers and encoders. These LEARN numbers
                    (medians, means, categories), so they live inside the model
                    pipeline and are fitted on the training rows only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    VALID_RANGES,
)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). The report counts what was changed or removed.

    Outliers are only FLAGGED in the report, never deleted automatically:
    a 20-marla house in Islamabad can be legitimately very expensive.
    """
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Expected columns: {FEATURES + [TARGET]}"
        )

    report: dict = {"rows_before": int(len(df))}
    out = df[FEATURES + [TARGET]].copy()

    # 1) Standardise text so "peshawar " and "Peshawar" are the same city.
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("string").str.strip().str.title()

    # 2) Force numeric columns to numbers; unreadable entries become NaN.
    numeric_cols = NUMERIC_FEATURES + [TARGET]
    before_nan = out[numeric_cols].isna().sum().sum()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    report["non_numeric_entries_set_to_nan"] = int(
        out[numeric_cols].isna().sum().sum() - before_nan
    )

    # 3) Exact duplicate rows. Removed BEFORE the split so the same listing
    #    cannot appear in both train and test (that would be leakage).
    n = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(n - len(out))

    # 4) Rows without a usable target cannot be used for learning.
    n = len(out)
    out = out[out[TARGET].notna() & (out[TARGET] > 0)]
    report["missing_or_nonpositive_price_removed"] = int(n - len(out))

    # 5) Impossible feature values (negative area, 99 bedrooms, ...).
    #    Missing values (NaN) are NOT removed here; the pipeline imputes them.
    n = len(out)
    valid = pd.Series(True, index=out.index)
    for col, (low, high) in VALID_RANGES.items():
        value = out[col]
        in_range = (value >= low) & (value <= high)
        valid &= in_range | value.isna()
    out = out[valid].reset_index(drop=True)
    report["invalid_values_removed"] = int(n - len(out))

    report["rows_after"] = int(len(out))
    report["missing_values_remaining"] = {
        col: int(count) for col, count in out.isna().sum().items() if count > 0
    }
    report["price_outliers_flagged_not_removed"] = int(find_price_outliers(out).sum())
    return out, report


def find_price_outliers(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of prices outside 1.5 x IQR on the log scale (flag only)."""
    log_price = np.log(df[TARGET])
    q1, q3 = log_price.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (log_price < q1 - 1.5 * iqr) | (log_price > q3 + 1.5 * iqr)


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into X_train, X_test, y_train, y_test with a fixed seed."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Imputation + encoding (+ optional scaling).

    scale_numeric=True  for Linear Regression (sensitive to feature scale).
    scale_numeric=False for tree models (they ignore scale, so skip it).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": an unseen city becomes all-zeros instead of crashing.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
        ]
    )
