"""Central place for paths and constants, so every script uses the same ones."""
from pathlib import Path

# Project root = the folder that contains `src/`. No hardcoded absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Synthetic demo dataset (clearly NOT real market data)
SAMPLE_DATA_PATH = RAW_DIR / "sample_houses_SYNTHETIC.csv"

# Outputs created by the pipeline
CLEAN_DATA_PATH = PROCESSED_DIR / "houses_clean.csv"
MODEL_PATH = MODELS_DIR / "house_price_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ---- Schema: the dataset must provide these columns ----
TARGET = "price_pkr"
CATEGORICAL_FEATURES = ["city", "property_type"]
NUMERIC_FEATURES = ["area_marla", "bedrooms", "bathrooms", "floors", "age_years"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

CURRENCY = "PKR"
AREA_UNIT = "marla"

# Hard limits for a value to be considered valid (used in cleaning AND in the app).
# (min, max), both inclusive, except area which must be strictly greater than 0.
VALID_RANGES = {
    "area_marla": (0.5, 500.0),
    "bedrooms": (1, 20),
    "bathrooms": (1, 20),
    "floors": (1, 10),
    "age_years": (0, 100),
}
INTEGER_FEATURES = ["bedrooms", "bathrooms", "floors"]

# Modelling settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# A city with fewer training rows than this gets a "poorly represented" warning.
MIN_ROWS_PER_CITY = 50
