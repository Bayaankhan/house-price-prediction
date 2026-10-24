"""Load a CSV dataset safely, with clear error messages."""
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Raises a readable error if the file is missing, empty, or unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Place your CSV inside {RAW_DIR} or pass the correct path."
        )
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        # Some scraped files are not UTF-8; latin-1 reads almost anything.
        df = pd.read_csv(path, encoding="latin-1")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The file is empty: {path}") from exc

    if df.empty:
        raise ValueError(f"The file has headers but no rows: {path}")
    return df
