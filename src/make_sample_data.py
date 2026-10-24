"""Generate a SYNTHETIC demonstration dataset.

WARNING: Every row is invented by a random generator. Prices follow a simple
made-up formula and are NOT real Pakistani market values. Use this file only to
learn the workflow. Never present results from it as real market findings.

Run from the project root:
    python -m src.make_sample_data
"""
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_DIR, SAMPLE_DATA_PATH


def build_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)  # fixed seed -> same data each run

    cities = ["Peshawar", "Mardan", "Islamabad", "Lahore"]
    # Invented price level per marla in PKR (NOT real values)
    base_per_marla = {
        "Peshawar": 3_500_000,
        "Mardan": 2_500_000,
        "Islamabad": 6_000_000,
        "Lahore": 5_000_000,
    }

    city = rng.choice(cities, n_rows)
    property_type = rng.choice(["House", "Flat"], n_rows, p=[0.8, 0.2])
    area_marla = rng.choice([3, 5, 7, 10, 20], n_rows, p=[0.15, 0.30, 0.20, 0.25, 0.10])
    bedrooms = np.clip(
        np.round(area_marla / 2.5 + rng.integers(-1, 2, n_rows)), 1, 8
    ).astype(int)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n_rows), 1, 8)
    floors = rng.choice([1, 2, 3], n_rows, p=[0.4, 0.5, 0.1])
    age_years = rng.integers(0, 40, n_rows).astype(float)

    price = (
        np.array([base_per_marla[c] for c in city])
        * area_marla
        * (1 + 0.05 * (bedrooms - 3))
        * (1 + 0.10 * (floors - 1))
        * (1 - 0.008 * age_years)
        * np.where(property_type == "Flat", 0.8, 1.0)
        * rng.normal(1, 0.08, n_rows)  # random noise
    )
    price = np.round(price, -3)

    df = pd.DataFrame(
        {
            "city": city,
            "property_type": property_type,
            "area_marla": area_marla,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "floors": floors,
            "age_years": age_years,
            "price_pkr": price,
        }
    )

    # Add deliberate mess so we can practise cleaning in Stage 2
    missing_idx = rng.choice(n_rows, 30, replace=False)
    df.loc[missing_idx, "age_years"] = np.nan
    df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)  # duplicates
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = build_sample_data()
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    print(f"Saved {df.shape[0]} rows x {df.shape[1]} columns -> {SAMPLE_DATA_PATH}")
    print("REMINDER: this is SYNTHETIC demo data, not real property listings.")


if __name__ == "__main__":
    main()
