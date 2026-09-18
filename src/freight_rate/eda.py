"""Exploratory-analysis figures and numbers used in the report and the walkthrough.

Everything here is model-free so it can be shown *before* the modelling story.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

TEAL = "#064A56"; RUST = "#B5432B"; GREY = "#9DAFB3"


def _prep(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    df["rpm"] = df[config.TARGET] / df["distance"]
    df["quote_ratio"] = df[config.TARGET] / (df["quote_signal"] * df["distance"])
    df["month"] = df[config.DATE_COLUMN].dt.month
    return df


def fig_quote_regimes(df: pd.DataFrame, path: Path) -> dict:
    daily = df.groupby(config.DATE_COLUMN).agg(exact=("quote_ratio", lambda s: ((s - 1).abs() < 0.03).mean()),
                                                median_ratio=("quote_ratio", "median"))
    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.plot(daily.index, daily["exact"] * 100, color=TEAL, lw=1.8)
    ax.set_ylabel("% loads with rate = quote x distance (+-3%)")
    ax.set_title("quote_signal is exact in some months and broken in others", loc="left", fontweight="bold")
    for m in (4, 5, 7, 8, 10):
        start = pd.Timestamp(2025, m, 1); end = start + pd.offsets.MonthEnd(1)
        ax.axvspan(start, end, color=RUST, alpha=0.08)
    ax.text(pd.Timestamp("2025-04-05"), 50, "shaded: degraded-quote months", color=RUST, fontsize=9)
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    clean_months = daily.index[daily["exact"] > 0.9].month.unique().tolist()
    return {"clean_months": sorted(clean_months), "noisy_months": sorted(set(range(1, 11)) - set(clean_months))}


def fig_quote_vs_distance(df: pd.DataFrame, validation: pd.DataFrame, path: Path) -> dict:
    clean = df[df["month"].isin([1, 2, 3, 6, 9])]; noisy = df[~df["month"].isin([1, 2, 3, 6, 9])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=150, sharey=True)
    for ax, (name, frame) in zip(axes, [("train: exact-quote months", clean), ("train: degraded-quote months", noisy), ("validation.csv (Nov-Dec)", validation)]):
        sample = frame.sample(min(4000, len(frame)), random_state=0)
        ax.scatter(sample["distance"], sample["quote_signal"], s=3, alpha=0.25, color=TEAL)
        corr = np.corrcoef(np.log(frame["distance"]), frame["quote_signal"])[0, 1]
        ax.set_title(f"{name}\ncorr(quote, log distance) = {corr:+.2f}", fontsize=10); ax.set_xlabel("distance (mi)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("quote_signal ($/mile)")
    fig.suptitle("In the scoring period quote_signal carries no structure at all", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return {"corr_clean": float(np.corrcoef(np.log(clean["distance"]), clean["quote_signal"])[0, 1]),
            "corr_noisy": float(np.corrcoef(np.log(noisy["distance"]), noisy["quote_signal"])[0, 1]),
            "corr_validation": float(np.corrcoef(np.log(validation["distance"]), validation["quote_signal"])[0, 1])}


def fig_rate_per_mile(df: pd.DataFrame, path: Path) -> dict:
    low, high = config.RPM_INLIER_BOUNDS
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.hist(df["rpm"], bins=np.logspace(np.log10(0.3), np.log10(15), 120), color=TEAL)
    ax.set_xscale("log"); ax.set_yscale("log")
    for b in (low, high):
        ax.axvline(b, color=RUST, ls="--", lw=1.2)
    ax.set_xlabel("posted_rate / distance ($ per mile, log scale)"); ax.set_ylabel("loads (log scale)")
    ax.set_title(f"Three populations: genuine loads between {low} and {high} $/mi, corrupted targets outside", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    out = ~df["rpm"].between(low, high)
    return {"n_outliers": int(out.sum()), "pct_outliers": float(out.mean() * 100),
            "high_multiplier_median": float((df.loc[out & (df["rpm"] > high), "quote_ratio"]).median()),
            "low_multiplier_median": float((1 / df.loc[out & (df["rpm"] < low), "quote_ratio"]).median())}


def fig_market_index(train: pd.DataFrame, validation: pd.DataFrame, path: Path) -> dict:
    pooled = pd.concat([train[[config.DATE_COLUMN, "market_index"]], validation[[config.DATE_COLUMN, "market_index"]]])
    daily = pooled.groupby(config.DATE_COLUMN)["market_index"].mean()
    fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
    ax.plot(daily.index, daily.values, color=TEAL, lw=1.2, label="daily mean market_index")
    ax.plot(daily.index, daily.rolling(7, center=True).mean(), color=RUST, lw=2, label="7-day mean")
    ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=GREY, alpha=0.2); ax.text(pd.Timestamp("2025-11-03"), daily.max() * 0.98, "validation period", fontsize=9)
    ax.set_title("market_index is a market-wide daily series with a weekly cycle (peak Thu, trough Sun)", loc="left", fontweight="bold", fontsize=11)
    ax.legend(loc="lower left"); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    dow = daily.groupby(daily.index.dayofweek).mean()
    return {"weekly_cycle_amplitude_pct": float((dow.max() - dow.min()) / dow.mean() * 100), "nov_dec_level": float(daily["2025-11-01":].mean()), "jan_oct_level": float(daily[:"2025-10-31"].mean())}


def fig_quarter_end_ramp(df: pd.DataFrame, path: Path) -> dict:
    d = df[df["rpm"].between(*config.RPM_INLIER_BOUNDS)].copy()
    d["rel"] = d["rpm"] / d.groupby(["month", "equipment"])["rpm"].transform("mean")
    d["dom"] = d[config.DATE_COLUMN].dt.day; d["qe"] = d["month"] % 3 == 0
    prof = d.groupby(["qe", "dom"])["rel"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(prof.index, (prof[True] - 1) * 100, color=RUST, lw=2, marker="o", ms=3, label="quarter-end months (Mar, Jun, Sep)")
    ax.plot(prof.index, (prof[False] - 1) * 100, color=TEAL, lw=2, marker="o", ms=3, label="other months")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("day of month"); ax.set_ylabel("rate/mile vs. month mean (%)")
    ax.set_title("Rates ramp up through quarter-end months (December is one)", loc="left", fontweight="bold", fontsize=11)
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return {"quarter_end_ramp_first_to_last_pct": float((prof[True].iloc[-3:].mean() - prof[True].iloc[:3].mean()) * 100),
            "other_months_ramp_pct": float((prof[False].iloc[-3:].mean() - prof[False].iloc[:3].mean()) * 100)}


def fig_monthly_profile(df: pd.DataFrame, path: Path) -> None:
    d = df[df["rpm"].between(*config.RPM_INLIER_BOUNDS)]
    monthly = d.groupby("month").agg(rpm=("rpm", "mean"), market=("market_index", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150); ax2 = ax.twinx()
    ax.plot(monthly.index, monthly["rpm"], color=TEAL, lw=2, marker="o", label="mean $/mile (genuine loads)")
    ax2.plot(monthly.index, monthly["market"], color=RUST, lw=2, marker="s", ls="--", label="mean market_index")
    ax.set_xlabel("month (2025)"); ax.set_ylabel("$ per mile", color=TEAL); ax2.set_ylabel("market_index", color=RUST)
    ax.set_title("Rates follow the market index plus a slow upward trend", loc="left", fontweight="bold", fontsize=11)
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def run_eda(train: pd.DataFrame, validation: pd.DataFrame, report_dir: Path = config.REPORT_DIR) -> Path:
    figures = report_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    df = _prep(train)
    facts = {}
    facts["regimes"] = fig_quote_regimes(df, figures / "eda_quote_regimes.png")
    facts["quote_vs_distance"] = fig_quote_vs_distance(df, validation, figures / "eda_quote_vs_distance.png")
    facts["outliers"] = fig_rate_per_mile(df, figures / "eda_rate_per_mile.png")
    facts["market"] = fig_market_index(train, validation, figures / "eda_market_index.png")
    facts["quarter_end"] = fig_quarter_end_ramp(df, figures / "eda_quarter_end_ramp.png")
    fig_monthly_profile(df, figures / "eda_monthly_profile.png")

    inl = df[df["rpm"].between(*config.RPM_INLIER_BOUNDS)]
    elasticity = np.polyfit(np.log(inl["distance"]), np.log(inl["rpm"]), 1)[0]
    lines = [
        "# EDA summary (train_test.csv, Jan-Oct 2025; validation.csv, Nov-Dec 2025)", "",
        f"- development loads: {len(train):,} over {train[config.DATE_COLUMN].nunique()} days; scoring loads: {len(validation):,} over {validation[config.DATE_COLUMN].nunique()} days (strictly later).",
        f"- posted_rate is essentially rate-per-mile x distance; rate-per-mile falls with distance (elasticity {elasticity:+.3f}) and is higher for Reefer/Flatbed.",
        f"- quote_signal equals the true rate-per-mile (+-1%) in months {facts['regimes']['clean_months']} but is degraded in months {facts['regimes']['noisy_months']}; in validation.csv its correlation with log distance is {facts['quote_vs_distance']['corr_validation']:+.3f} (vs {facts['quote_vs_distance']['corr_clean']:+.2f} in exact months): it is noise there and is excluded from the model.",
        f"- {facts['outliers']['n_outliers']:,} training targets ({facts['outliers']['pct_outliers']:.2f}%) are corrupted by a multiplicative factor (median x{facts['outliers']['high_multiplier_median']:.1f} up or /{facts['outliers']['low_multiplier_median']:.1f} down); they sit outside {config.RPM_INLIER_BOUNDS} $/mile and are excluded from fitting.",
        f"- market_index is a market-wide daily series (weekly cycle amplitude {facts['market']['weekly_cycle_amplitude_pct']:.0f}%); Nov-Dec level {facts['market']['nov_dec_level']:.3f} vs Jan-Oct {facts['market']['jan_oct_level']:.3f}. Missing values are imputed from the same-day mean.",
        f"- rates ramp up by about {facts['quarter_end']['quarter_end_ramp_first_to_last_pct']:+.1f}% through quarter-end months (Mar/Jun/Sep) versus {facts['quarter_end']['other_months_ramp_pct']:+.1f}% in other months; December is a quarter-end month.",
        "- weights: some negative (sign-flipped) and some missing; distances are consistent with the (synthetic, clipped) coordinates; eight validation cities never appear in training, so geography is modelled with coordinates rather than city identity.",
        "", "Figures: `reports/figures/eda_*.png`.",
    ]
    path = report_dir / "eda_summary.md"; path.write_text("\n".join(lines), encoding="utf-8")
    return path
