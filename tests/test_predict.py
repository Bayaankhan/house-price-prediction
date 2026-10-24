import copy

import pytest

from src.make_sample_data import build_sample_data
from src.predict import (
    ModelNotFoundError,
    PredictionInputError,
    coverage_warnings,
    load_metadata,
    predict_price,
    validate_input,
)
from src.train import train

VALID = {
    "city": "Peshawar",
    "property_type": "House",
    "area_marla": 5,
    "bedrooms": 3,
    "bathrooms": 2,
    "floors": 2,
    "age_years": 10,
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a small model into a temporary folder (never touches real models/)."""
    folder = tmp_path_factory.mktemp("run")
    data_path = folder / "tiny_SYNTHETIC.csv"
    build_sample_data(n_rows=300).to_csv(data_path, index=False)
    paths = {
        "model_path": folder / "model.joblib",
        "metadata_path": folder / "metadata.json",
        "reports_dir": folder / "reports",
        "processed_path": folder / "clean.csv",
    }
    train(data_path, cv_folds=3, make_charts=False, **paths)
    return paths


def test_prediction_returns_positive_number(trained):
    price = predict_price(VALID, model_path=trained["model_path"])
    assert isinstance(price, float)
    assert price > 0


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.train"):
        predict_price(VALID, model_path=tmp_path / "nope.joblib")


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("area_marla", -5, "between"),
        ("bedrooms", -1, "between"),
        ("bedrooms", 2.5, "whole number"),
        ("floors", 0, "between"),
        ("area_marla", "big", "must be a number"),
        ("area_marla", float("nan"), "finite"),
        ("city", "", "non-empty"),
        ("age_years", True, "not True/False"),
    ],
)
def test_invalid_inputs_are_rejected(field, value, message):
    bad = copy.deepcopy(VALID)
    bad[field] = value
    with pytest.raises(PredictionInputError, match=message):
        validate_input(bad)


def test_missing_and_unexpected_fields_are_rejected():
    incomplete = copy.deepcopy(VALID)
    del incomplete["bedrooms"]
    with pytest.raises(PredictionInputError, match="Missing"):
        validate_input(incomplete)

    extra = {**VALID, "swimming_pool": 1}
    with pytest.raises(PredictionInputError, match="Unexpected"):
        validate_input(extra)


def test_validated_row_has_training_column_order():
    from src.config import FEATURES

    assert list(validate_input(VALID).columns) == FEATURES


def test_coverage_warnings(trained):
    metadata = load_metadata(trained["metadata_path"])

    unknown_city = {**VALID, "city": "Atlantis"}
    assert any("not in the training data" in w for w in coverage_warnings(unknown_city, metadata))

    huge_area = {**VALID, "area_marla": 400}
    assert any("outside the range" in w for w in coverage_warnings(huge_area, metadata))

    assert any("SYNTHETIC" in w for w in coverage_warnings(VALID, metadata))
