"""Loading, validation and cleaning of the load-level data.

Cleaning is deliberately *flag-and-fix*: every repaired value gets a companion
indicator column, and a :class:`QualityReport` records what was done so the
numbers can be quoted in the write-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

EARTH_RADIUS_MILES = 3958.8


# ----------------------------------------------------------------------------- loading
def load_frame(path: Path, *, require_target: bool) -> pd.DataFrame:
    """Read one of the provided CSVs and enforce the expected schema."""
    frame = pd.read_csv(path)
    expected = [config.ID_COLUMN, *config.FEATURE_COLUMNS] + ([config.TARGET] if require_target else [])
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame[config.DATE_COLUMN] = pd.to_datetime(frame[config.DATE_COLUMN], errors="coerce")
    if frame[config.DATE_COLUMN].isna().any():
        raise ValueError(f"{path.name} contains unparseable dates")
    if frame[config.ID_COLUMN].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate {config.ID_COLUMN} values")
    return frame


# ----------------------------------------------------------------------------- reference tables
def build_city_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """One row per city with its (synthetic but internally consistent) coordinates.

    Both the pickup and delivery columns are pooled so that cities which only
    appear on one side of a lane are still covered.  Raises if a city ever
    appears with two different coordinates, which would signal a data problem.
    """
    parts = []
    for frame in frames:
        parts.append(frame[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(frame[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    table = pd.concat(parts, ignore_index=True).drop_duplicates()
    if table["city"].duplicated().any():
        clashes = sorted(table.loc[table["city"].duplicated(), "city"].unique())
        raise ValueError(f"cities with inconsistent coordinates: {clashes}")
    return table.sort_values("city").set_index("city")


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles between two arrays of coordinates."""
    rad = np.pi / 180.0
    d_lat = (lat2 - lat1) * rad
    d_lon = (lon2 - lon1) * rad
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(d_lon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


class MarketCalendar:
    """Daily market index derived from the loads themselves.

    ``market_index`` is a market-wide daily series with a small per-load jitter
    (within-day std 0.025 versus 0.165 across days), so the daily mean is a
    faithful reconstruction and is used both to impute the ~0.8% of loads whose
    value is missing and to describe days for which we have no load of our own
    (the fixed December chart).  Dates outside the observed range fall back to a
    day-of-week profile scaled to the most recent four weeks.
    """

    def __init__(self, *frames: pd.DataFrame):
        pooled = pd.concat([f[[config.DATE_COLUMN, "market_index"]] for f in frames], ignore_index=True)
        daily = pooled.groupby(config.DATE_COLUMN)["market_index"].mean().sort_index()
        full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        self.daily = daily.reindex(full_index).interpolate(limit_direction="both")
        profile = self.daily.groupby(self.daily.index.dayofweek).mean() / self.daily.mean()
        self.dow_profile = profile.reindex(range(7)).fillna(1.0)
        self.recent_level = float(self.daily.iloc[-28:].mean())

    def lookup(self, dates: pd.Series) -> pd.Series:
        values = pd.Series(pd.to_datetime(dates).map(self.daily).to_numpy(), index=dates.index, dtype=float)
        unknown = values.isna()
        if unknown.any():
            profile = pd.to_datetime(dates[unknown]).dt.dayofweek.map(self.dow_profile)
            values[unknown] = profile.to_numpy() * self.recent_level
        return values

    def rolling(self, dates: pd.Series, window: int = 7) -> pd.Series:
        """Trailing ``window``-day mean of the daily index (rates react to a smoothed market)."""
        extended = self.daily.copy()
        needed = pd.to_datetime(dates).max()
        if needed > extended.index.max():
            extra = pd.date_range(extended.index.max() + pd.Timedelta(days=1), needed, freq="D")
            filled = self.lookup(pd.Series(extra)).to_numpy()
            extended = pd.concat([extended, pd.Series(filled, index=extra)])
        smoothed = extended.rolling(window, min_periods=1).mean()
        return pd.Series(pd.to_datetime(dates).map(smoothed).to_numpy(), index=dates.index, dtype=float)


# ----------------------------------------------------------------------------- cleaning
@dataclass
class QualityReport:
    """Counts of every data-quality repair, per dataset."""

    name: str
    rows: int = 0
    negative_weight: int = 0
    missing_weight: int = 0
    missing_market_index: int = 0
    target_outliers: int = 0
    unseen_cities: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"### {self.name} ({self.rows:,} rows)",
            f"- negative weights (sign flipped, fixed with abs): {self.negative_weight:,}",
            f"- missing weights (left as NaN + indicator; median-filled for the linear stage): {self.missing_weight:,}",
            f"- missing market_index (imputed from the same-day market mean): {self.missing_market_index:,}",
        ]
        if self.target_outliers:
            lines.append(f"- corrupted targets (rate-per-mile outside {config.RPM_INLIER_BOUNDS}, excluded from fitting): {self.target_outliers:,}")
        if self.unseen_cities:
            lines.append(f"- cities not present in training data: {', '.join(self.unseen_cities)}")
        lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


def clean_loads(frame: pd.DataFrame, calendar: MarketCalendar, *, name: str, is_train: bool,
                known_cities: set[str] | None = None) -> tuple[pd.DataFrame, QualityReport]:
    """Repair known data issues and flag them.  Returns (clean frame, report)."""
    df = frame.copy()
    report = QualityReport(name=name, rows=len(df))

    # weight: some rows carry a sign-flipped weight; the magnitudes match the normal distribution.
    negative = df["weight"] < 0
    report.negative_weight = int(negative.sum())
    df["weight_negative"] = negative.astype(int)
    df["weight"] = df["weight"].abs()
    df["weight_missing"] = df["weight"].isna().astype(int)
    report.missing_weight = int(df["weight_missing"].sum())

    # market_index: daily market series -> same-day mean is the natural imputation.
    missing_market = df["market_index"].isna()
    report.missing_market_index = int(missing_market.sum())
    df["market_index_missing"] = missing_market.astype(int)
    df.loc[missing_market, "market_index"] = calendar.lookup(df.loc[missing_market, config.DATE_COLUMN])

    # geometry sanity: distance must be at least the great-circle distance.
    great_circle = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    df["circuity"] = df["distance"] / np.maximum(great_circle, 1.0)
    if (df["distance"] < great_circle * 0.99).any():
        report.notes.append("some distances are shorter than the great-circle distance")

    if known_cities is not None:
        seen = set(df["pickup"]) | set(df["delivery"])
        report.unseen_cities = sorted(seen - known_cities)

    if is_train:
        df["rate_per_mile"] = df[config.TARGET] / df["distance"]
        low, high = config.RPM_INLIER_BOUNDS
        outlier = ~df["rate_per_mile"].between(low, high)
        df["is_target_outlier"] = outlier.astype(int)
        report.target_outliers = int(outlier.sum())

    return df, report


def load_all() -> dict:
    """Load, cross-reference and clean every provided file in one call."""
    train_raw = load_frame(config.TRAIN_PATH, require_target=True)
    validation_raw = load_frame(config.VALIDATION_PATH, require_target=False)
    calendar = MarketCalendar(train_raw, validation_raw)
    cities = build_city_table(train_raw, validation_raw)
    train, train_report = clean_loads(train_raw, calendar, name="train_test.csv", is_train=True)
    known = set(train_raw["pickup"]) | set(train_raw["delivery"])
    validation, validation_report = clean_loads(validation_raw, calendar, name="validation.csv", is_train=False, known_cities=known)
    return {
        "train": train,
        "validation": validation,
        "calendar": calendar,
        "cities": cities,
        "reports": [train_report, validation_report],
    }
