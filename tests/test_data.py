import numpy as np
import pandas as pd
import pytest

from freight_rate import config
from freight_rate.data import MarketCalendar, build_city_table, clean_loads, haversine_miles


def _frame(n=6, with_target=True):
    dates = pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"] * 3)
    df = pd.DataFrame({
        "load_id": [f"X-{i}" for i in range(n)],
        "pickup": ["A"] * n, "delivery": ["B"] * n,
        "pickup_lat": [30.0] * n, "pickup_lon": [-90.0] * n,
        "delivery_lat": [35.0] * n, "delivery_lon": [-85.0] * n,
        "distance": [500.0] * n, "equipment": ["Dry Van"] * n,
        "weight": [30000.0, -25000.0, np.nan, 31000.0, 32000.0, 33000.0],
        "date": dates,
        "market_index": [1.0, 1.1, np.nan, 0.8, 0.9, 1.0],
        "quote_signal": [2.0] * n,
    })
    if with_target:
        df["posted_rate"] = [1000.0, 1100.0, 1050.0, 5000.0, 150.0, 1020.0]
    return df


def test_negative_weight_is_flagged_and_fixed():
    df = _frame()
    clean, report = clean_loads(df, MarketCalendar(df), name="t", is_train=True)
    assert report.negative_weight == 1
    assert clean.loc[1, "weight"] == 25000.0 and clean.loc[1, "weight_negative"] == 1
    assert report.missing_weight == 1 and clean.loc[2, "weight_missing"] == 1


def test_market_index_imputed_with_same_day_mean():
    df = _frame()
    clean, report = clean_loads(df, MarketCalendar(df), name="t", is_train=True)
    assert report.missing_market_index == 1
    assert clean.loc[2, "market_index"] == pytest.approx(1.05)  # mean of the other Jan-01 loads
    assert clean.loc[2, "market_index_missing"] == 1


def test_target_outliers_flagged_by_rate_per_mile():
    df = _frame()
    clean, report = clean_loads(df, MarketCalendar(df), name="t", is_train=True)
    assert report.target_outliers == 2
    assert clean["is_target_outlier"].tolist() == [0, 0, 0, 1, 1, 0]


def test_city_table_rejects_inconsistent_coordinates():
    df = _frame()
    bad = df.copy(); bad.loc[0, "pickup_lat"] = 31.0
    with pytest.raises(ValueError):
        build_city_table(df, bad)
    table = build_city_table(df)
    assert set(table.index) == {"A", "B"}


def test_calendar_falls_back_for_unknown_dates():
    df = _frame()
    calendar = MarketCalendar(df)
    future = pd.Series(pd.to_datetime(["2025-03-01"]))
    assert np.isfinite(calendar.lookup(future)).all()
    assert np.isfinite(calendar.rolling(future, 7)).all()


def test_haversine_symmetry_and_zero():
    assert haversine_miles(30, -90, 30, -90) == pytest.approx(0.0)
    assert haversine_miles(30, -90, 35, -85) == pytest.approx(haversine_miles(35, -85, 30, -90))


def test_unseen_cities_reported():
    train = _frame(); val = _frame(with_target=False); val.loc[0, "pickup"] = "Z"
    clean, report = clean_loads(val, MarketCalendar(train, val), name="v", is_train=False, known_cities={"A", "B"})
    assert report.unseen_cities == ["Z"]
