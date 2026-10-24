"""Stage 1 inspection report. Works on ANY CSV, real or synthetic.

Usage (from the project root):
    python -m src.inspect_data                      # inspects the synthetic sample
    python -m src.inspect_data data/raw/your_file.csv
"""
import sys

import pandas as pd

from src.config import SAMPLE_DATA_PATH
from src.data_loader import load_csv


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def inspect(df: pd.DataFrame) -> None:
    section("1. Shape (rows, columns)")
    print(df.shape)

    section("2. Column names")
    print(list(df.columns))

    section("3. First 5 rows")
    print(df.head().to_string())

    section("4. Data types and non-null counts")
    df.info()

    section("5. Missing values per column")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values")

    section("6. Fully duplicated rows")
    print(df.duplicated().sum())

    section("7. Descriptive statistics (numeric columns)")
    print(df.describe().T.to_string())

    section("8. Categorical columns: unique values and top counts")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head(10).to_string())


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_DATA_PATH
    print(f"Inspecting: {path}")
    inspect(load_csv(path))


if __name__ == "__main__":
    main()
