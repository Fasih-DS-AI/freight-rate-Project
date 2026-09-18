"""Produce the two deliverable files: validation_predictions.csv and the filled December chart inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import MarketCalendar
from .features import FeatureBuilder
from .model import HybridRateModel


def predict_validation(model: HybridRateModel, builder: FeatureBuilder, validation: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Fill the template's predicted_rate column, preserving its row order."""
    pred = pd.Series(model.predict(builder.transform(validation), validation["distance"]), index=validation[config.ID_COLUMN].to_numpy())
    missing = set(template[config.ID_COLUMN]) - set(pred.index)
    if missing:
        raise ValueError(f"validation.csv has no rows for {len(missing)} template ids, e.g. {sorted(missing)[:3]}")
    out = template[[config.ID_COLUMN]].copy()
    out["predicted_rate"] = pred.reindex(out[config.ID_COLUMN]).round(2).to_numpy()
    if out["predicted_rate"].isna().any() or (out["predicted_rate"] <= 0).any():
        raise ValueError("predictions contain missing or non-positive values")
    return out


def build_december_loads(template: pd.DataFrame, calendar: MarketCalendar, builder: FeatureBuilder) -> pd.DataFrame:
    """Expand the seven-column December template into the full load schema the model expects.

    Coordinates come from the city table, the market index from the daily market
    calendar (validation.csv contains loads on every December day), and
    quote_signal is left empty because the model does not use it.
    """
    df = template.copy()
    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN])
    plat, plon = builder.lookup_city(config.DECEMBER_FIXED["pickup"])
    dlat, dlon = builder.lookup_city(config.DECEMBER_FIXED["delivery"])
    df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"] = plat, plon, dlat, dlon
    df["market_index"] = calendar.lookup(df[config.DATE_COLUMN]).to_numpy()
    df["quote_signal"] = np.nan
    df["weight"] = df["weight"].astype(float); df["distance"] = df["distance"].astype(float)
    df["weight_missing"] = 0; df["market_index_missing"] = 0
    return df


def predict_december(model: HybridRateModel, builder: FeatureBuilder, calendar: MarketCalendar, template: pd.DataFrame) -> pd.DataFrame:
    loads = build_december_loads(template, calendar, builder)
    pred = model.predict(builder.transform(loads), loads["distance"])
    out = template.copy()
    out["predicted_rate"] = np.round(pred, 2)
    return out[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]]


def write_outputs(validation_predictions: pd.DataFrame, december: pd.DataFrame, root: Path = config.ROOT) -> dict[str, Path]:
    """Write to outputs/ and to the exact locations the provided scorer expects."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "validation_predictions": config.PREDICTIONS_PATH,
        "validation_predictions_root": root / "validation_predictions.csv",
        "december": config.DECEMBER_OUTPUT_PATH,
        "december_scorer": config.DECEMBER_INPUT_PATH,
    }
    validation_predictions.to_csv(paths["validation_predictions"], index=False)
    validation_predictions.to_csv(paths["validation_predictions_root"], index=False)
    december.to_csv(paths["december"], index=False)
    december.to_csv(paths["december_scorer"], index=False)
    return paths
