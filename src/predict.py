"""Reusable prediction code: validate the input, load the model, return a price.

Example (from the project root):
    python -m src.predict
"""
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import joblib
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    INTEGER_FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    VALID_RANGES,
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file does not exist yet."""


class PredictionInputError(ValueError):
    """Raised when the property details are missing or invalid."""


@lru_cache(maxsize=2)
def _load_model_cached(model_path: str):
    return joblib.load(model_path)


def load_model(model_path: Path = MODEL_PATH):
    """Load the saved pipeline (preprocessing + model). Cached after first load."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at {model_path}. "
            "Train it first with:  python -m src.train"
        )
    return _load_model_cached(str(model_path))


def load_metadata(metadata_path: Path = METADATA_PATH) -> dict:
    """Load facts about the trained model (cities, ranges, test scores...)."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise ModelNotFoundError(
            f"Model metadata not found at {metadata_path}. "
            "Train the model first with:  python -m src.train"
        )
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_input(details: Mapping) -> pd.DataFrame:
    """Check the property details and return a one-row DataFrame for the model.

    Raises PredictionInputError with a clear message for any problem.
    """
    if not isinstance(details, Mapping):
        raise PredictionInputError("Property details must be a dictionary.")

    missing = [name for name in FEATURES if name not in details]
    if missing:
        raise PredictionInputError(f"Missing required fields: {missing}")
    unexpected = [name for name in details if name not in FEATURES]
    if unexpected:
        raise PredictionInputError(
            f"Unexpected fields: {unexpected}. Allowed fields: {FEATURES}"
        )

    row: dict = {}
    for name in CATEGORICAL_FEATURES:
        value = details[name]
        if not isinstance(value, str) or not value.strip():
            raise PredictionInputError(f"'{name}' must be a non-empty text value.")
        row[name] = value.strip().title()  # same standardisation as in cleaning

    for name in NUMERIC_FEATURES:
        value = details[name]
        if isinstance(value, bool):
            raise PredictionInputError(f"'{name}' must be a number, not True/False.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise PredictionInputError(f"'{name}' must be a number, got {value!r}.")
        if not math.isfinite(number):
            raise PredictionInputError(f"'{name}' must be a finite number.")

        low, high = VALID_RANGES[name]
        if number < low or number > high:
            raise PredictionInputError(
                f"'{name}' must be between {low} and {high}, got {number}."
            )
        if name in INTEGER_FEATURES:
            if number != int(number):
                raise PredictionInputError(f"'{name}' must be a whole number, got {number}.")
            number = int(number)
        row[name] = number

    return pd.DataFrame([row], columns=FEATURES)


def coverage_warnings(details: Mapping, metadata: dict) -> list[str]:
    """Plain-English warnings when the input is poorly covered by the training data."""
    warnings: list[str] = []
    city = str(details.get("city", "")).strip().title()
    counts = metadata.get("city_counts", {})

    if city not in counts:
        warnings.append(
            f"'{city}' was not in the training data, so the model has never seen it. "
            "The estimate ignores the city and is unreliable."
        )
    elif counts[city] < metadata.get("min_rows_per_city", 50):
        warnings.append(
            f"Only {counts[city]} training examples exist for {city}. "
            "The estimate for this city is less reliable."
        )

    property_type = str(details.get("property_type", "")).strip().title()
    if property_type not in metadata.get("property_types", []):
        warnings.append(f"Property type '{property_type}' was not in the training data.")

    for name, bounds in metadata.get("numeric_ranges", {}).items():
        try:
            value = float(details[name])
        except (KeyError, TypeError, ValueError):
            continue
        if value < bounds["min"] or value > bounds["max"]:
            warnings.append(
                f"{name} = {value:g} is outside the range seen in training "
                f"({bounds['min']:g} to {bounds['max']:g}); extrapolation is unreliable."
            )

    if metadata.get("is_synthetic"):
        warnings.append(
            "This model was trained on SYNTHETIC demo data. The price is NOT a real "
            "market estimate."
        )
    return warnings


def predict_price(details: Mapping, model_path: Path = MODEL_PATH) -> float:
    """Return the estimated price (in PKR) for one property."""
    features = validate_input(details)
    pipeline = load_model(model_path)
    predicted = float(pipeline.predict(features)[0])
    return max(predicted, 0.0)  # a price can never be negative


if __name__ == "__main__":
    example = {
        "city": "Peshawar",
        "property_type": "House",
        "area_marla": 5,
        "bedrooms": 3,
        "bathrooms": 3,
        "floors": 2,
        "age_years": 10,
    }
    try:
        price = predict_price(example)
        print(f"Input: {example}")
        print(f"Estimated price: PKR {price:,.0f}")
        for warning in coverage_warnings(example, load_metadata()):
            print("WARNING:", warning)
    except (ModelNotFoundError, PredictionInputError) as error:
        print("Error:", error)
