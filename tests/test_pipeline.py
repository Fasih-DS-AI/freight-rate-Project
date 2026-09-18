"""End-to-end smoke tests on a small synthetic dataset (fast, no real data needed)."""
import numpy as np
import pandas as pd

from freight_rate import config
from freight_rate.data import MarketCalendar, build_city_table, clean_loads
from freight_rate.features import FeatureBuilder, STRUCTURAL_FEATURES, TIME_FEATURES
from freight_rate.model import HybridConfig, HybridRateModel
from freight_rate.predict import build_december_loads, predict_december, predict_validation

CITIES = {"Lexington": (36.99, -85.0), "Fort Wayne": (41.32, -85.36), "Dallas": (31.83, -94.38), "Atlanta": (34.85, -86.29)}


def synthetic(n=600, seed=0, with_target=True):
    rng = np.random.default_rng(seed)
    names = list(CITIES)
    pick = rng.choice(names, n); deliv = np.array([rng.choice([c for c in names if c != p]) for p in pick])
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    distance = rng.uniform(100, 2500, n)
    equipment = rng.choice(list(config.EQUIPMENT_CODES), n)
    weight = rng.uniform(10000, 45000, n)
    market = 1 + 0.1 * np.sin(dates.dayofweek) + rng.normal(0, 0.02, n)
    df = pd.DataFrame({
        "load_id": [f"S-{i:05d}" for i in range(n)], "pickup": pick, "delivery": deliv,
        "pickup_lat": [CITIES[c][0] for c in pick], "pickup_lon": [CITIES[c][1] for c in pick],
        "delivery_lat": [CITIES[c][0] for c in deliv], "delivery_lon": [CITIES[c][1] for c in deliv],
        "distance": distance, "equipment": equipment, "weight": weight, "date": dates,
        "market_index": market, "quote_signal": rng.uniform(1.5, 2.5, n),
    })
    if with_target:
        rpm = 2.0 * distance ** -0.1 * np.exp(0.15 * (market - 1)) * (1 + 0.1 * (equipment == "Reefer")) * np.exp(rng.normal(0, 0.02, n))
        df["posted_rate"] = rpm * distance
    return df


def _bundle():
    train_raw, val_raw = synthetic(600, 0), synthetic(200, 1, with_target=False)
    calendar = MarketCalendar(train_raw, val_raw); cities = build_city_table(train_raw, val_raw)
    train, _ = clean_loads(train_raw, calendar, name="t", is_train=True)
    val, _ = clean_loads(val_raw, calendar, name="v", is_train=False)
    return train, val, calendar, cities


def test_feature_builder_columns():
    train, _, calendar, cities = _bundle()
    X = FeatureBuilder(calendar, cities).transform(train)
    for column in STRUCTURAL_FEATURES + TIME_FEATURES + ["days_to_quarter_end"]:
        assert column in X.columns
    assert X["quarter_end_ramp"].between(0, 1).all()
    assert (X.loc[train["date"].dt.month == 3, "quarter_end_month"] == 1).all()


def test_model_fits_and_predicts_positive_rates():
    train, val, calendar, cities = _bundle()
    builder = FeatureBuilder(calendar, cities)
    fit = train[train["is_target_outlier"] == 0]
    model = HybridRateModel(HybridConfig(n_seeds=1, num_boost_round=50)).fit(builder.transform(fit), fit["posted_rate"], fit["distance"])
    pred = model.predict(builder.transform(val), val["distance"])
    assert np.isfinite(pred).all() and (pred > 0).all()
    # in-sample fit should be decent on the synthetic generator
    train_pred = model.predict(builder.transform(fit), fit["distance"])
    assert np.mean(np.abs(train_pred - fit["posted_rate"]) / fit["posted_rate"]) < 0.05
    assert "trend_pct_per_month" in model.summary()


def test_trend_modes_differ_only_beyond_training_window():
    train, val, calendar, cities = _bundle()
    builder = FeatureBuilder(calendar, cities)
    fit = train[train["is_target_outlier"] == 0]
    X = builder.transform(fit)
    full = HybridRateModel(HybridConfig(n_seeds=1, num_boost_round=20, trend_mode="full")).fit(X, fit["posted_rate"], fit["distance"])
    none = HybridRateModel(HybridConfig(n_seeds=1, num_boost_round=20, trend_mode="none")).fit(X, fit["posted_rate"], fit["distance"])
    inside = builder.transform(fit.head(20))
    assert np.allclose(full.time_component(inside), none.time_component(inside))
    future = fit.head(20).copy(); future["date"] = pd.Timestamp("2025-12-15")
    Xf = builder.transform(future)
    assert not np.allclose(full.time_component(Xf), none.time_component(Xf))


def test_predict_validation_respects_template_order():
    train, val, calendar, cities = _bundle()
    builder = FeatureBuilder(calendar, cities)
    fit = train[train["is_target_outlier"] == 0]
    model = HybridRateModel(HybridConfig(n_seeds=1, num_boost_round=20)).fit(builder.transform(fit), fit["posted_rate"], fit["distance"])
    template = pd.DataFrame({"load_id": val["load_id"].iloc[::-1].to_numpy(), "predicted_rate": np.nan})
    out = predict_validation(model, builder, val, template)
    assert list(out.columns) == ["load_id", "predicted_rate"]
    assert out["load_id"].tolist() == template["load_id"].tolist()
    assert (out["predicted_rate"] > 0).all()


def test_december_frame_and_prediction():
    train, val, calendar, cities = _bundle()
    builder = FeatureBuilder(calendar, cities)
    fit = train[train["is_target_outlier"] == 0]
    model = HybridRateModel(HybridConfig(n_seeds=1, num_boost_round=20)).fit(builder.transform(fit), fit["posted_rate"], fit["distance"])
    template = pd.DataFrame({
        "pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360, "equipment": "Dry Van", "weight": 32000,
        "date": pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d"), "predicted_rate": np.nan,
    })
    loads = build_december_loads(template, calendar, builder)
    assert np.isfinite(loads["market_index"]).all()
    out = predict_december(model, builder, calendar, template)
    assert list(out.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert len(out) == 31 and (out["predicted_rate"] > 0).all()
