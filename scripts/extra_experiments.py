"""Two supporting experiments quoted in the report, run through the real pipeline.

1. Hybrid model vs a single LightGBM that receives the date features directly
   (same folds, same boosting settings) -- shows why the time component is separated.
2. Unseen-city simulation: eight cities removed from training entirely, error measured
   on holdout loads that touch them vs loads that do not.

Writes reports/extra_experiments.json and reports/extra_experiments.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freight_rate import config  # noqa: E402
from freight_rate.data import load_all  # noqa: E402
from freight_rate.features import CALENDAR_FEATURES, FeatureBuilder, STRUCTURAL_FEATURES  # noqa: E402
from freight_rate.model import HybridConfig, HybridRateModel  # noqa: E402
from freight_rate.validate import regression_metrics  # noqa: E402

PURE_FEATURES = STRUCTURAL_FEATURES + CALENDAR_FEATURES + ["t", "market_index", "dow", "dom_frac", "quarter_end_ramp"]


def pure_gbm_predict(cfg: HybridConfig, Xtr, ytr, Xte) -> np.ndarray:
    preds = []
    for i in range(cfg.n_seeds):
        ds = lgb.Dataset(Xtr[PURE_FEATURES], ytr, categorical_feature=["equipment"], free_raw_data=False)
        preds.append(lgb.train(cfg.lgb_params(cfg.base_seed + i), ds, num_boost_round=cfg.num_boost_round).predict(Xte[PURE_FEATURES]))
    return np.mean(preds, axis=0)


def main() -> None:
    bundle = load_all(); train = bundle["train"]; builder = FeatureBuilder(bundle["calendar"], bundle["cities"])
    cfg = HybridConfig()
    rows = []
    for start, end in config.BACKTEST_FOLDS:
        trn = train[(train[config.DATE_COLUMN] < pd.Timestamp(start)) & (train["is_target_outlier"] == 0)]
        tst = train[(train[config.DATE_COLUMN] >= pd.Timestamp(start)) & (train[config.DATE_COLUMN] < pd.Timestamp(end))]
        Xtr, Xte = builder.transform(trn), builder.transform(tst); ytr = np.log(trn[config.TARGET] / trn["distance"])
        inl = (tst["is_target_outlier"] == 0).to_numpy(); y = tst[config.TARGET].to_numpy()
        hybrid = HybridRateModel(cfg).fit(Xtr, trn[config.TARGET], trn["distance"]).predict(Xte, tst["distance"])
        pure = np.exp(pure_gbm_predict(cfg, Xtr, ytr, Xte)) * tst["distance"].to_numpy()
        for name, pred in [("hybrid", hybrid), ("single LightGBM with date features", pure)]:
            m = regression_metrics(y, pred, inl); rows.append({"model": name, "test_window": f"{start} to {end}", **m})
        print(f"fold {start}: hybrid {rows[-2]['inlier_MAPE']*100:.2f}% / pure {rows[-1]['inlier_MAPE']*100:.2f}%")
    comp = pd.DataFrame(rows)
    summary = comp.groupby("model").agg(MAE=("MAE", "mean"), inlier_MAPE=("inlier_MAPE", "mean"), abs_inlier_bias=("inlier_bias_pct", lambda s: s.abs().mean())).reset_index()

    rng = np.random.default_rng(config.RANDOM_SEED)
    cities = sorted(set(train["pickup"]) | set(train["delivery"])); hold = [str(c) for c in rng.choice(cities, 8, replace=False)]
    split = pd.Timestamp(config.HOLDOUT_START)
    trn = train[(train[config.DATE_COLUMN] < split) & (train["is_target_outlier"] == 0) & ~train["pickup"].isin(hold) & ~train["delivery"].isin(hold)]
    tst = train[train[config.DATE_COLUMN] >= split]; touch = (tst["pickup"].isin(hold) | tst["delivery"].isin(hold)).to_numpy()
    pred = HybridRateModel(cfg).fit(builder.transform(trn), trn[config.TARGET], trn["distance"]).predict(builder.transform(tst), tst["distance"])
    inl = (tst["is_target_outlier"] == 0).to_numpy(); y = tst[config.TARGET].to_numpy()
    unseen = {"held_out_cities": hold,
              "seen": regression_metrics(y[~touch], pred[~touch], inl[~touch]),
              "unseen": regression_metrics(y[touch], pred[touch], inl[touch])}

    out = {"backtest_comparison": comp.to_dict(orient="records"), "backtest_summary": summary.to_dict(orient="records"), "unseen_cities": unseen}
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "extra_experiments.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# Supporting experiments", "", "## Hybrid vs single LightGBM with date features (rolling backtest means)", "",
          summary.to_markdown(index=False, floatfmt=".4f"), "", "## Unseen-city simulation (holdout Sep-Oct, 8 cities removed from training)", "",
          f"held-out cities: {', '.join(hold)}", "",
          pd.DataFrame([{"loads": "seen cities", **unseen["seen"]}, {"loads": "touching unseen cities", **unseen["unseen"]}]).to_markdown(index=False, floatfmt=".4f")]
    (config.REPORT_DIR / "extra_experiments.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string(index=False)); print(json.dumps(unseen, indent=1)[:600])


if __name__ == "__main__":
    main()
