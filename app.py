"""Streamlit web app for house price estimates.

Run (from the project root):
    streamlit run app.py
"""
import streamlit as st

from src.config import AREA_UNIT, CURRENCY, VALID_RANGES
from src.predict import (
    ModelNotFoundError,
    PredictionInputError,
    coverage_warnings,
    load_metadata,
    load_model,
    predict_price,
)

st.set_page_config(page_title="House Price Estimator", page_icon="\U0001F3E0", layout="centered")


def format_pkr(amount: float) -> str:
    """12345678 -> 'PKR 12,345,678'."""
    return f"{CURRENCY} {amount:,.0f}"


def format_pkr_words(amount: float) -> str:
    """Pakistani style: 12345678 -> '1.23 crore'; 4500000 -> '45.00 lakh'."""
    if amount >= 10_000_000:
        return f"about {amount / 10_000_000:,.2f} crore {CURRENCY}"
    if amount >= 100_000:
        return f"about {amount / 100_000:,.2f} lakh {CURRENCY}"
    return f"about {format_pkr(amount)}"


@st.cache_resource(show_spinner=False)
def get_model_and_metadata():
    """Load once per server session instead of on every click."""
    return load_model(), load_metadata()


st.title("House Price Estimator")
st.write(
    "Enter a property's details to get an **estimated** price from a machine "
    "learning model."
)

try:
    _, metadata = get_model_and_metadata()
except ModelNotFoundError as error:
    st.error(f"{error}")
    st.stop()
except Exception as error:  # corrupted file, version mismatch, etc.
    st.error(f"The model could not be loaded: {error}")
    st.stop()

if metadata.get("is_synthetic"):
    st.warning(
        "**Demo mode:** this model was trained on SYNTHETIC (invented) data. "
        "The prices below are not real market values."
    )

with st.form("property_form"):
    col_a, col_b = st.columns(2)
    with col_a:
        city = st.selectbox("City", metadata["cities"])
        property_type = st.selectbox("Property type", metadata["property_types"])
        area = st.number_input(
            f"Area ({AREA_UNIT})",
            min_value=float(VALID_RANGES["area_marla"][0]),
            max_value=float(VALID_RANGES["area_marla"][1]),
            value=5.0,
            step=0.5,
        )
        age = st.number_input(
            "House age (years)",
            min_value=VALID_RANGES["age_years"][0],
            max_value=VALID_RANGES["age_years"][1],
            value=10,
            step=1,
        )
    with col_b:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=VALID_RANGES["bedrooms"][0],
            max_value=VALID_RANGES["bedrooms"][1],
            value=3,
            step=1,
        )
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=VALID_RANGES["bathrooms"][0],
            max_value=VALID_RANGES["bathrooms"][1],
            value=3,
            step=1,
        )
        floors = st.number_input(
            "Floors",
            min_value=VALID_RANGES["floors"][0],
            max_value=VALID_RANGES["floors"][1],
            value=2,
            step=1,
        )
    submitted = st.form_submit_button("Estimate price", type="primary")

if submitted:
    details = {
        "city": city,
        "property_type": property_type,
        "area_marla": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "age_years": age,
    }
    try:
        price = predict_price(details)
    except PredictionInputError as error:
        st.error(f"Please fix the input: {error}")
    except ModelNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Something went wrong while predicting: {error}")
    else:
        st.subheader("Estimated price")
        st.metric(label=f"Estimate ({CURRENCY})", value=format_pkr(price))
        st.caption(format_pkr_words(price))

        for message in coverage_warnings(details, metadata):
            # The synthetic notice is already shown at the top of the page.
            if "SYNTHETIC" not in message:
                st.warning(message)

        st.info(
            "This is a statistical **estimate**, not an official valuation and not a "
            "guaranteed selling price. Real prices depend on the exact location, "
            "condition, negotiation and market timing."
        )

with st.expander("About this model and its data"):
    test = metadata["test_metrics"]
    st.write(
        f"- **Model:** {metadata['model_name']}\n"
        f"- **Data:** {metadata['data_label']}\n"
        f"- **Training rows (after cleaning):** {metadata['n_rows_clean']:,}\n"
        f"- **Trained at (UTC):** {metadata['trained_at_utc']}\n"
        f"- **Test-set error (MAE):** {format_pkr(test['MAE'])}\n"
        f"- **Test-set R²:** {test['R2']:.3f}"
    )
    st.write(
        "Results depend on the quality, coverage and date of the training data. "
        "Cities with few training examples, or values outside the training range, "
        "give less reliable estimates."
    )
