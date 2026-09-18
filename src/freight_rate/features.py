"""Feature engineering.

Two groups of features are produced from a cleaned load frame:

* **structural** features describe the load itself (distance, equipment, weight,
  lane geometry).  They are consumed by the gradient-boosted stage.
* **time** features describe *when* the load moves (trend, market index,
  quarter-end ramp, weekday).  They are consumed by the parametric stage, which
  is the only part of the model allowed to extrapolate beyond the training
  window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import MarketCalendar

TIME_ORIGIN = pd.Timestamp(config.DEV_START)

STRUCTURAL_FEATURES = [
    "distance", "log_distance", "equipment", "weight", "weight_missing",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "delta_lat", "delta_lon",
]
CALENDAR_FEATURES = ["days_to_quarter_end"]
TIME_FEATURES = ["t", "market_index", "dow", "dom_frac", "quarter_end_month", "quarter_end_ramp"]
CATEGORICAL_FEATURES = ["equipment"]


class FeatureBuilder:
    """Stateless transformer from a cleaned load frame to the model feature matrix."""

    def __init__(self, calendar: MarketCalendar, cities: pd.DataFrame):
        self.calendar = calendar
        self.cities = cities

    # -- public ---------------------------------------------------------------
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame
        dates = pd.to_datetime(df[config.DATE_COLUMN])
        X = pd.DataFrame(index=df.index)

        # structural
        X["distance"] = df["distance"].astype(float)
        X["log_distance"] = np.log(X["distance"])
        X["equipment"] = df["equipment"].map(config.EQUIPMENT_CODES).astype("Int64").astype(float)
        if X["equipment"].isna().any():
            unknown = sorted(df.loc[X["equipment"].isna(), "equipment"].unique())
            raise ValueError(f"unknown equipment types: {unknown}")
        X["weight"] = df["weight"].abs()
        X["weight_missing"] = df["weight"].isna().astype(int)
        for column in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
            X[column] = df[column].astype(float)
        X["delta_lat"] = X["delivery_lat"] - X["pickup_lat"]
        X["delta_lon"] = X["delivery_lon"] - X["pickup_lon"]

        # time
        X["t"] = (dates - TIME_ORIGIN).dt.days.astype(float)
        market = df["market_index"].astype(float)
        if market.isna().any():
            market = market.fillna(self.calendar.lookup(dates))
        X["market_index"] = market
        X["dow"] = dates.dt.dayofweek
        X["dom_frac"] = (dates.dt.day - 1) / (dates.dt.days_in_month - 1)
        X["quarter_end_month"] = (dates.dt.month % 3 == 0).astype(int)
        X["quarter_end_ramp"] = X["quarter_end_month"] * X["dom_frac"]
        quarter_end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
        X["days_to_quarter_end"] = (quarter_end - dates).dt.days.astype(float)
        return X

    def lookup_city(self, city: str) -> tuple[float, float]:
        if city not in self.cities.index:
            raise KeyError(f"unknown city {city!r}; known cities come from train_test.csv and validation.csv")
        row = self.cities.loc[city]
        return float(row["lat"]), float(row["lon"])
