"""Central configuration: paths, constants, and modelling choices.

Every decision that was made from the exploratory analysis is recorded here with
its justification so the pipeline is auditable end to end.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = OUTPUT_DIR / "models"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUT_PATH = DATA_DIR / "december_chart_inputs.csv"

PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.csv"
DECEMBER_OUTPUT_PATH = OUTPUT_DIR / "december_chart_inputs.csv"

# --------------------------------------------------------------------------- schema
ID_COLUMN = "load_id"
TARGET = "posted_rate"
DATE_COLUMN = "date"
FEATURE_COLUMNS = [
    "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "equipment", "weight", "date", "market_index", "quote_signal",
]
EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

# --------------------------------------------------------------------------- cleaning rules
# Rate-per-mile bounds that separate genuine loads from corrupted targets.
# In the months where quote_signal is an exact copy of the true rate-per-mile,
# every corrupted row (rate multiplied or divided by 2.3-5.6x) has a rate-per-mile
# outside [1.25, 4.2] while every genuine row is inside [1.57, 3.57].  The bounds
# below sit in the empty gap between the two populations.
RPM_INLIER_BOUNDS = (1.2, 4.0)

# quote_signal is excluded from modelling on purpose.  It equals the true
# rate-per-mile (+-1%) in Jan-Mar, Jun and Sep, but is a degraded, downward
# biased signal in Apr-May, Jul-Aug and Oct, and in the Nov-Dec scoring set it
# explains 0.2% of its own variance against load features (i.e. it is noise).
# A model that learns to trust it would fail out of time.
EXCLUDED_FEATURES = {"quote_signal": "regime-dependent leak; pure noise in Nov-Dec"}

# --------------------------------------------------------------------------- validation design
DEV_START = "2025-01-01"
DEV_END = "2025-10-31"
# Primary holdout mirrors the real task: the last two months of development data
# are held out, exactly the horizon between the end of train_test (Oct 31) and
# the end of validation.csv (Dec 31).
HOLDOUT_START = "2025-09-01"
# Rolling-origin backtest: five two-month test windows, each trained on
# everything strictly before it.  Used to check stability across the
# quote-signal regimes and the market-index cycle.
BACKTEST_FOLDS = [
    ("2025-05-01", "2025-07-01"),
    ("2025-06-01", "2025-08-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-08-01", "2025-10-01"),
    ("2025-09-01", "2025-11-01"),
]

# --------------------------------------------------------------------------- December chart
DECEMBER_FIXED = {
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360.0,
    "equipment": "Dry Van",
    "weight": 32_000.0,
}

RANDOM_SEED = 42
