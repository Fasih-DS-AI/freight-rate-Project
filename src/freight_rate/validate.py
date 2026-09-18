"""Time-aware validation: primary holdout, rolling-origin backtest, baselines and diagnostics.

All splits are strictly chronological: every test window is later than
everything the model was trained on, mirroring the real task (train through
October, predict November-December).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .features import FeatureBuilder
from .model import HybridRateModel

ModelFactory = Callable[[], HybridRateModel]


# ----------------------------------------------------------------------------- metrics
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, inlier: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    out = {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE": float(np.mean(np.abs(error) / y_true)),
        "R2": float(1 - np.sum(error ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "bias_pct": float(np.mean(error / y_true) * 100),
        "n": int(len(y_true)),
    }
    if inlier is not None:
        inlier = np.asarray(inlier, dtype=bool)
        e, y = error[inlier], y_true[inlier]
        out.update({
            "inlier_MAE": float(np.mean(np.abs(e))),
            "inlier_RMSE": float(np.sqrt(np.mean(e ** 2))),
            "inlier_MAPE": float(np.mean(np.abs(e) / y)),
            "inlier_bias_pct": float(np.mean(e / y) * 100),
            "inlier_n": int(inlier.sum()),
        })
    return out


# ----------------------------------------------------------------------------- baselines
def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Simple reference predictors so the model's gain is quantified."""
    fit = train[train["is_target_outlier"] == 0]
    bins = [0, 200, 400, 600, 800, 1000, 1500, 2000, 2500, 4000]
    key = lambda df: pd.cut(df["distance"], bins)
    table = fit.assign(bin=key(fit)).groupby(["equipment", "bin"], observed=True)["rate_per_mile"].median()
    lookup = pd.MultiIndex.from_arrays([test["equipment"], key(test)])
    rpm_bin = table.reindex(lookup).to_numpy()
    rpm_bin = np.where(np.isnan(rpm_bin), fit["rate_per_mile"].median(), rpm_bin)
    return {
        "median rate/mile x distance": fit["rate_per_mile"].median() * test["distance"].to_numpy(),
        "equipment x distance-bin median rate/mile": rpm_bin * test["distance"].to_numpy(),
        "quote_signal x distance (the tempting leak)": (test["quote_signal"] * test["distance"]).to_numpy(),
    }


# ----------------------------------------------------------------------------- evaluation
def fit_and_predict(train: pd.DataFrame, test: pd.DataFrame, builder: FeatureBuilder, make_model: ModelFactory) -> tuple[HybridRateModel, np.ndarray]:
    fit = train[train["is_target_outlier"] == 0]
    model = make_model().fit(builder.transform(fit), fit[config.TARGET], fit["distance"])
    return model, model.predict(builder.transform(test), test["distance"])


def run_holdout(train_all: pd.DataFrame, builder: FeatureBuilder, make_model: ModelFactory,
                holdout_start: str = config.HOLDOUT_START) -> dict:
    """Primary validation: train before ``holdout_start``, test on everything after."""
    start = pd.Timestamp(holdout_start)
    train = train_all[train_all[config.DATE_COLUMN] < start]
    test = train_all[train_all[config.DATE_COLUMN] >= start].copy()
    model, pred = fit_and_predict(train, test, builder, make_model)
    inlier = test["is_target_outlier"].to_numpy() == 0
    test["predicted_rate"] = pred
    test["error_pct"] = (pred - test[config.TARGET]) / test[config.TARGET]

    rows = [{"model": "hybrid (this submission)", **regression_metrics(test[config.TARGET], pred, inlier)}]
    for name, base in baseline_predictions(train, test).items():
        rows.append({"model": name, **regression_metrics(test[config.TARGET], base, inlier)})
    comparison = pd.DataFrame(rows)

    inl = test[inlier]
    cols = [config.TARGET, "predicted_rate"]
    score = lambda g: pd.Series(regression_metrics(g[config.TARGET], g["predicted_rate"]))
    by_month = inl.groupby(inl[config.DATE_COLUMN].dt.to_period("M"))[cols].apply(score).reset_index()
    by_equipment = inl.groupby("equipment")[cols].apply(score).reset_index()
    by_distance = inl.groupby(pd.cut(inl["distance"], [0, 300, 600, 1000, 1500, 2500, 4000]), observed=True)[cols].apply(score).reset_index()
    return {
        "train_range": (str(train[config.DATE_COLUMN].min().date()), str(train[config.DATE_COLUMN].max().date())),
        "test_range": (str(test[config.DATE_COLUMN].min().date()), str(test[config.DATE_COLUMN].max().date())),
        "comparison": comparison,
        "by_month": by_month,
        "by_equipment": by_equipment,
        "by_distance": by_distance,
        "predictions": test[[config.ID_COLUMN, config.DATE_COLUMN, "equipment", "distance", config.TARGET, "predicted_rate", "error_pct", "is_target_outlier"]],
        "model_summary": model.summary(),
        "feature_importance": model.feature_importance(),
    }


def run_backtest(train_all: pd.DataFrame, builder: FeatureBuilder, make_model: ModelFactory,
                 folds=config.BACKTEST_FOLDS) -> pd.DataFrame:
    """Rolling-origin backtest: each fold trains on everything before its two-month window."""
    rows = []
    for start, end in folds:
        train = train_all[train_all[config.DATE_COLUMN] < pd.Timestamp(start)]
        test = train_all[(train_all[config.DATE_COLUMN] >= pd.Timestamp(start)) & (train_all[config.DATE_COLUMN] < pd.Timestamp(end))]
        model, pred = fit_and_predict(train, test, builder, make_model)
        metrics = regression_metrics(test[config.TARGET], pred, test["is_target_outlier"].to_numpy() == 0)
        summary = model.summary()
        rows.append({"train_until": start, "test_window": f"{start} to {end}", "n_train": len(train), "n_test": len(test),
                     "trend_pct_per_month": summary["trend_pct_per_month"], "quarter_end_ramp_pct": summary["quarter_end_ramp_pct"], **metrics})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- reporting
def short_window(window: str) -> str:
    """'2025-05-01 to 2025-07-01' -> 'May-Jun 2025'."""
    start, end = window.split(" to ")
    a = pd.Timestamp(start); b = pd.Timestamp(end) - pd.Timedelta(days=1)
    return f"{a.strftime('%b')}-{b.strftime('%b')} {b.year}"


def _fmt(df: pd.DataFrame, digits: int = 3) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f") if hasattr(df, "to_markdown") else df.to_string(index=False)


def write_validation_report(holdout: dict, backtest: pd.DataFrame, report_dir: Path = config.REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = report_dir / "figures"; figures.mkdir(exist_ok=True)
    holdout["comparison"].to_csv(report_dir / "holdout_comparison.csv", index=False)
    holdout["by_month"].to_csv(report_dir / "holdout_by_month.csv", index=False)
    holdout["predictions"].to_csv(report_dir / "holdout_predictions.csv", index=False)
    holdout["feature_importance"].to_csv(report_dir / "feature_importance.csv", index=False)
    backtest.to_csv(report_dir / "backtest.csv", index=False)

    # figure 1: predicted vs actual (holdout)
    p = holdout["predictions"]; inl = p["is_target_outlier"] == 0
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(p.loc[inl, config.TARGET], p.loc[inl, "predicted_rate"], s=4, alpha=0.3, label="genuine loads")
    ax.scatter(p.loc[~inl, config.TARGET], p.loc[~inl, "predicted_rate"], s=8, alpha=0.6, color="crimson", label="corrupted targets")
    lim = [0, p[config.TARGET].max() * 1.02]; ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual posted_rate ($)"); ax.set_ylabel("predicted rate ($)"); ax.set_title("Holdout (Sep-Oct 2025): predicted vs actual"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "holdout_pred_vs_actual.png"); plt.close(fig)

    # figure 2: daily mean actual vs predicted rate-per-mile over the holdout
    d = p[inl].assign(rpm=lambda x: x[config.TARGET] / x["distance"], prpm=lambda x: x["predicted_rate"] / x["distance"]).groupby(config.DATE_COLUMN)[["rpm", "prpm"]].mean()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(d.index, d["rpm"], label="actual (daily mean $/mile)", lw=1.8); ax.plot(d.index, d["prpm"], label="predicted", lw=1.8, ls="--")
    ax.set_title("Holdout: daily mean rate per mile, actual vs predicted"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figures / "holdout_daily_rpm.png"); plt.close(fig)

    # figure 3: backtest (only when it was run)
    has_backtest = len(backtest) > 0
    if has_backtest:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        labels = [short_window(w) for w in backtest["test_window"]]
        ax.bar(labels, backtest["inlier_MAPE"] * 100, color="#064A56")
        for i, (m, b) in enumerate(zip(backtest["inlier_MAPE"], backtest["inlier_bias_pct"])):
            ax.text(i, m * 100 + 0.04, f"{m*100:.2f}%\nbias {b:+.2f}%", ha="center", fontsize=8)
        ax.set_ylim(0, backtest["inlier_MAPE"].max() * 100 * 1.35)
        ax.set_ylabel("MAPE on genuine loads (%)"); ax.set_xlabel("test window (trained on everything before it)")
        ax.set_title("Rolling-origin backtest, two-month horizon", loc="left", fontweight="bold")
        fig.tight_layout(); fig.savefig(figures / "backtest_mape.png"); plt.close(fig)
    backtest_lines = (
        ["## Rolling-origin backtest", "", _fmt(backtest[["test_window", "n_train", "n_test", "MAE", "MAPE", "inlier_MAE", "inlier_MAPE", "inlier_bias_pct", "trend_pct_per_month", "quarter_end_ramp_pct"]]), "",
         f"Backtest mean: MAE {backtest['MAE'].mean():.1f}, inlier MAPE {backtest['inlier_MAPE'].mean()*100:.2f}%, mean |inlier bias| {backtest['inlier_bias_pct'].abs().mean():.2f}%", ""]
        if has_backtest else ["## Rolling-origin backtest", "", "Skipped (`--skip-backtest`).", ""])

    # markdown summary
    s = holdout["model_summary"]
    lines = [
        "# Validation summary", "",
        f"Primary holdout: train {holdout['train_range'][0]} to {holdout['train_range'][1]}, test {holdout['test_range'][0]} to {holdout['test_range'][1]} (same two-month horizon as the real task).", "",
        "## Holdout comparison", "", _fmt(holdout["comparison"]), "",
        "## Holdout by month (genuine loads)", "", _fmt(holdout["by_month"]), "",
        "## Holdout by equipment (genuine loads)", "", _fmt(holdout["by_equipment"]), "",
        "## Holdout by distance band (genuine loads)", "", _fmt(holdout["by_distance"]), "",
        *backtest_lines,
        "## Fitted time component (holdout model)", "",
        f"- trend: {s['trend_pct_per_month']:+.2f}% per month (mode: {s['trend_mode']})",
        f"- market index elasticity: {s['market_index_elasticity_log_pts']:.3f} log-points per index unit",
        f"- quarter-end ramp: {s['quarter_end_ramp_pct']:+.2f}% from the first to the last day of Mar/Jun/Sep/Dec",
        f"- weekday effect vs Monday (%): " + ", ".join(f"{k}: {v:+.2f}" for k, v in s["weekday_effect_pct_vs_monday"].items()), "",
        "## Structural feature importance (LightGBM gain share)", "", _fmt(holdout["feature_importance"]), "",
    ]
    path = report_dir / "validation_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
