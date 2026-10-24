import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.data_preprocessing import build_preprocessor, clean_data, split_data
from src.make_sample_data import build_sample_data


@pytest.fixture(scope="module")
def raw_df():
    return build_sample_data(n_rows=300)


def test_clean_data_removes_duplicates(raw_df):
    clean_df, report = clean_data(raw_df)
    assert not clean_df.duplicated().any()
    assert report["duplicates_removed"] == 10
    assert report["rows_after"] == len(clean_df)


def test_clean_data_drops_invalid_and_missing_target_rows(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "area_marla"] = -5          # impossible area
    bad.loc[1, "bedrooms"] = 99            # impossible bedrooms
    bad.loc[2, TARGET] = np.nan            # no target
    clean_df, report = clean_data(bad)
    assert report["invalid_values_removed"] >= 2
    assert report["missing_or_nonpositive_price_removed"] >= 1
    assert (clean_df["area_marla"] > 0).all()
    assert clean_df[TARGET].notna().all()


def test_clean_data_standardises_city_text(raw_df):
    messy = raw_df.copy()
    messy.loc[0, "city"] = "  peshawar "
    clean_df, _ = clean_data(messy)
    assert "peshawar" not in clean_df["city"].tolist()
    assert "  peshawar " not in clean_df["city"].tolist()


def test_clean_data_requires_all_columns(raw_df):
    with pytest.raises(ValueError, match="missing required columns"):
        clean_data(raw_df.drop(columns=["city"]))


def test_split_is_reproducible_and_disjoint(raw_df):
    clean_df, _ = clean_data(raw_df)
    a_train, a_test, _, _ = split_data(clean_df)
    b_train, b_test, _, _ = split_data(clean_df)
    assert a_test.index.equals(b_test.index)          # same seed -> same split
    assert set(a_train.index).isdisjoint(a_test.index)  # no row in both sets


def test_preprocessor_handles_missing_and_unseen_values(raw_df):
    clean_df, _ = clean_data(raw_df)
    X_train, _, _, _ = split_data(clean_df)
    preprocessor = build_preprocessor(scale_numeric=True).fit(X_train)

    new_row = pd.DataFrame(
        [{"city": "Atlantis", "property_type": "House", "area_marla": 5,
          "bedrooms": 3, "bathrooms": 2, "floors": 1, "age_years": np.nan}],
        columns=FEATURES,
    )
    transformed = preprocessor.transform(new_row)  # must not raise
    assert not np.isnan(transformed).any()
