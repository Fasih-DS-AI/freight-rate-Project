"""Hybrid rate model: parametric time component + gradient-boosted structural component.

The model works on ``log(rate per mile)`` and decomposes it as::

    log(rate / distance) = time(date, market_index) + structure(load) + noise

* ``time`` is a small linear model (trend, market index, quarter-end ramp,
  weekday).  Being linear it can be extrapolated two months past the end of the
  development data, which trees cannot do.
* ``structure`` is a LightGBM ensemble fitted on the residual using only
  load-level features (distance, equipment, weight, lane geometry).  It never
  sees a date, so it cannot memorise daily levels.

The two stages are fitted jointly enough for the purpose: stage one includes
coarse structural controls so the time coefficients are not confounded, then
only its *time* terms are kept and everything else is left to stage two.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .features import CALENDAR_FEATURES, CATEGORICAL_FEATURES, STRUCTURAL_FEATURES

TIME_TERMS = ["t", "market_index", "quarter_end_ramp", "quarter_end_month", "dom_frac"] + [f"dow_{k}" for k in range(1, 7)]


@dataclass
class HybridConfig:
    trend_mode: str = "full"          # 'full' | 'damped' | 'none' -- how the linear trend extends past training
    damping: float = 0.5              # slope multiplier used when trend_mode == 'damped'
    n_seeds: int = 3                  # LightGBM models averaged (different bagging/feature seeds)
    num_boost_round: int = 1500
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_data_in_leaf: int = 40
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    gbm_features: tuple[str, ...] = tuple(STRUCTURAL_FEATURES + CALENDAR_FEATURES)
    base_seed: int = 42

    def lgb_params(self, seed: int) -> dict:
        return dict(
            objective="l2", learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_data_in_leaf, feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction, bagging_freq=1, lambda_l2=self.lambda_l2,
            verbose=-1, seed=seed, num_threads=0,
        )


@dataclass
class HybridRateModel:
    config: HybridConfig = field(default_factory=HybridConfig)

    # fitted state
    time_coef_: pd.Series | None = None
    t_end_: float | None = None
    weight_fill_: float | None = None
    boosters_: list = field(default_factory=list)

    # ------------------------------------------------------------------ stage 1
    def _stage1_design(self, X: pd.DataFrame) -> pd.DataFrame:
        D = pd.DataFrame(index=X.index)
        D["t"] = X["t"] / 100.0
        D["market_index"] = X["market_index"]
        D["quarter_end_ramp"] = X["quarter_end_ramp"]
        D["quarter_end_month"] = X["quarter_end_month"]
        D["dom_frac"] = X["dom_frac"]
        for k in range(1, 7):
            D[f"dow_{k}"] = (X["dow"] == k).astype(float)
        # structural controls (kept only during fitting so time terms are not confounded)
        D["log_distance"] = X["log_distance"]
        D["log_distance_sq"] = X["log_distance"] ** 2
        D["weight"] = X["weight"].fillna(self.weight_fill_) / 1e4
        D["equip_reefer"] = (X["equipment"] == 1).astype(float)
        D["equip_flatbed"] = (X["equipment"] == 2).astype(float)
        return D

    def time_component(self, X: pd.DataFrame) -> pd.Series:
        """Time effect in log space, with the trend handled according to ``trend_mode`` past ``t_end_``."""
        D = self._stage1_design(X)
        component = D[TIME_TERMS].mul(self.time_coef_[TIME_TERMS], axis=1).sum(axis=1)
        beyond = (D["t"] - self.t_end_).clip(lower=0.0)
        multiplier = {"full": 1.0, "damped": self.config.damping, "none": 0.0}[self.config.trend_mode]
        component = component - beyond * self.time_coef_["t"] * (1.0 - multiplier)
        return component

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X: pd.DataFrame, rate: pd.Series, distance: pd.Series) -> "HybridRateModel":
        y = np.log(rate / distance)
        self.weight_fill_ = float(X["weight"].median())
        D = self._stage1_design(X)
        linear = LinearRegression().fit(D, y)
        self.time_coef_ = pd.Series(linear.coef_, index=D.columns)
        self.t_end_ = float(D["t"].max())

        residual = y - self.time_component(X)
        features = list(self.config.gbm_features)
        self.boosters_ = []
        for i in range(self.config.n_seeds):
            dataset = lgb.Dataset(X[features], residual, categorical_feature=[c for c in CATEGORICAL_FEATURES if c in features], free_raw_data=False)
            booster = lgb.train(self.config.lgb_params(self.config.base_seed + i), dataset, num_boost_round=self.config.num_boost_round)
            self.boosters_.append(booster)
        return self

    def predict_log_rpm(self, X: pd.DataFrame) -> np.ndarray:
        features = list(self.config.gbm_features)
        structural = np.mean([b.predict(X[features]) for b in self.boosters_], axis=0)
        return structural + self.time_component(X).to_numpy()

    def predict(self, X: pd.DataFrame, distance: pd.Series) -> np.ndarray:
        return np.exp(self.predict_log_rpm(X)) * distance.to_numpy()

    # ------------------------------------------------------------------ introspection
    def summary(self) -> dict:
        c = self.time_coef_
        return {
            "trend_pct_per_month": float(c["t"] / 100.0 * 30 * 100),
            "market_index_elasticity_log_pts": float(c["market_index"]),
            "quarter_end_ramp_pct": float(c["quarter_end_ramp"] * 100),
            "quarter_end_month_shift_pct": float(c["quarter_end_month"] * 100),
            "dom_ramp_other_months_pct": float(c["dom_frac"] * 100),
            "weekday_effect_pct_vs_monday": {k: float(c[f"dow_{k}"] * 100) for k in range(1, 7)},
            "trend_mode": self.config.trend_mode,
            "n_boosters": len(self.boosters_),
        }

    def feature_importance(self) -> pd.DataFrame:
        gains = np.mean([b.feature_importance("gain") for b in self.boosters_], axis=0)
        names = self.boosters_[0].feature_name()
        table = pd.DataFrame({"feature": names, "gain": gains}).sort_values("gain", ascending=False)
        table["share"] = table["gain"] / table["gain"].sum()
        return table.reset_index(drop=True)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "HybridRateModel":
        with open(path, "rb") as handle:
            return pickle.load(handle)
