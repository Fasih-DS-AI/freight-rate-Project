"""Build the assessment report (DOCX + HTML + PDF) from the generated reports/ artefacts.

    python scripts/build_report.py --candidate "Your Name" --repo-url https://github.com/you/repo

The PDF is printed from the HTML with headless Microsoft Edge/Chrome when available.
Every number in the narrative is read from reports/*.csv|json, outputs/ and the saved model,
except a handful of exploratory findings that are stated with their source.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from freight_rate import config  # noqa: E402
from freight_rate.model import HybridRateModel  # noqa: E402
from freight_rate.validate import short_window  # noqa: E402

FIG = config.REPORT_DIR / "figures"
TEAL = RGBColor(0x06, 0x4A, 0x56)


# ----------------------------------------------------------------------------- numbers
def load_numbers() -> dict:
    n = {}
    comp = pd.read_csv(config.REPORT_DIR / "holdout_comparison.csv").set_index("model")
    n["holdout"] = comp
    n["by_month"] = pd.read_csv(config.REPORT_DIR / "holdout_by_month.csv")
    n["backtest"] = pd.read_csv(config.REPORT_DIR / "backtest.csv")
    n["importance"] = pd.read_csv(config.REPORT_DIR / "feature_importance.csv")
    n["extra"] = json.loads((config.REPORT_DIR / "extra_experiments.json").read_text(encoding="utf-8"))
    model = HybridRateModel.load(config.MODEL_DIR / "hybrid_rate_model.pkl"); n["model"] = model.summary()
    preds = pd.read_csv(config.PREDICTIONS_PATH); n["pred_stats"] = preds["predicted_rate"].describe()
    dec = pd.read_csv(config.DECEMBER_OUTPUT_PATH); dec["date"] = pd.to_datetime(dec["date"]); n["december"] = dec
    train = pd.read_csv(config.TRAIN_PATH); train["date"] = pd.to_datetime(train["date"])
    lf = train[(train.pickup == "Lexington") & (train.delivery == "Fort Wayne") & (train.equipment == "Dry Van")]
    n["lex_fw"] = lf["posted_rate"].agg(["count", "min", "median", "max"])
    n["quality"] = (config.REPORT_DIR / "data_quality.md").read_text(encoding="utf-8")
    return n


def pct(x, d=2): return f"{x*100:.{d}f}%"
def money(x): return f"${x:,.0f}"


# ----------------------------------------------------------------------------- content model
def build_content(n: dict, candidate: str, repo_url: str, loom_url: str = "") -> list:
    h = n["holdout"]; hy = h.loc["hybrid (this submission)"]; bt = n["backtest"]; ms = n["model"]; ex = n["extra"]
    base_bin = h.loc["equipment x distance-bin median rate/mile"]; base_quote = h.loc["quote_signal x distance (the tempting leak)"]; base_med = h.loc["median rate/mile x distance"]
    pure = next(r for r in ex["backtest_summary"] if r["model"].startswith("single")); hyb = next(r for r in ex["backtest_summary"] if r["model"] == "hybrid")
    un = ex["unseen_cities"]; dec = n["december"]; ps = n["pred_stats"]; lf = n["lex_fw"]
    wd = ms["weekday_effect_pct_vs_monday"]
    c = []
    A = c.append

    # ---- title
    A(("title", "Freight Rate Prediction"))
    A(("subtitle", "Machine Learning Engineer assessment, Spotter"))
    A(("meta", f"{candidate}  |  {date.today().strftime('%d %B %Y')}  |  Code: {repo_url}" + (f"  |  Walkthrough: {loom_url}" if loom_url else "")))

    # ---- 1 summary
    A(("h1", "1. Summary"))
    A(("p", [("Task. ", "b"), ("Predict the posted rate of 12,000 truckload freight loads in November-December 2025 from lane, equipment, weight, date and a market index, after learning from 48,000 labelled loads in January-October 2025. Because every scoring load is later than every training load, this is a two-month-ahead forecast and it is validated as one.", None)]))
    A(("p", [("Approach. ", "b"), ("Log rate-per-mile is decomposed into a small linear time component (trend, market index, quarter-end ramp, weekday) that can be extrapolated past the training window, plus a LightGBM structural component fitted on the residual with load features only. The quote signal provided in the data is deliberately excluded: it is an exact copy of the rate in some months, degraded in others, and pure noise in the scoring period.", None)]))
    A(("p", [("Result. ", "b"), (f"On a chronological holdout (train January-August, test September-October) the model reaches a mean absolute percentage error of {pct(hy['inlier_MAPE'])} on genuine loads (MAE {money(hy['inlier_MAE'])}), against {pct(base_bin['inlier_MAPE'])} for an equipment-by-distance-band median and {pct(base_quote['inlier_MAPE'])} for quote signal times distance. A five-window rolling backtest gives {pct(bt['inlier_MAPE'].min())} to {pct(bt['inlier_MAPE'].max())} with absolute bias under 1%. About 1.4% of targets are corrupted by large multiplicative errors and are reported separately, since no model can predict them.", None)]))
    A(("p", [("Deliverables. ", "b"), ("validation_predictions.csv (12,000 rows, validated by the provided scorer), the fixed December chart produced by score.py (section 6), and a repository that reproduces everything with one command (appendix).", None)]))

    # ---- 2 data
    A(("h1", "2. The data and what it revealed"))
    A(("p", "Both files share the same schema: pickup and delivery city with coordinates, distance, equipment (Dry Van, Reefer, Flatbed), weight, date, a market index and a quote signal. The development file has the posted rate; the scoring file does not. The coordinates are synthetic but internally consistent per city, and distances are consistent with them, so both are used as given."))
    A(("table", pd.DataFrame([
        ["train_test.csv", "48,000", "1 Jan - 31 Oct 2025 (304 days)", "64", "yes"],
        ["validation.csv", "12,000", "1 Nov - 31 Dec 2025 (61 days)", "72 (8 unseen in training)", "no"],
    ], columns=["file", "loads", "dates", "cities", "posted_rate"]), "Table 1. The two files. Every scoring date is later than every training date."))
    A(("h2", "2.1 The posted rate is rate-per-mile times distance"))
    A(("p", "Rate-per-mile falls with distance (elasticity about -0.13 in log-log terms), is about 8% higher for Flatbed and 12% higher for Reefer than for Dry Van, rises about 12% from the lightest to the heaviest loads, and is a few percent higher for lanes touching the South-Central states than the West Coast. Modelling rate-per-mile rather than the dollar rate removes most of the scale and makes the remaining structure nearly additive in log space, which is what the model exploits."))
    A(("h2", "2.2 The quote signal is a trap"))
    A(("p", "In January-March, June and September, quote_signal times distance reproduces posted_rate within 1% for 98% of loads (Figure 1). In April-May, July-August and October it is a degraded signal: lower on average, with the sign of its relationship to distance flipped (Figure 2, middle). In the scoring file it has no relationship with distance, equipment, weight or region at all: a regression of quote_signal on those features explains 0.2% of its variance, versus 90% in the exact months (Figure 2, right). A model trained on the development data learns to lean on it (it is the single best in-sample feature) and then fails on exactly the loads that are scored. Section 4 quantifies the damage; the feature is excluded from the model."))
    A(("figure", FIG / "eda_quote_regimes.png", "Figure 1. Share of loads per day whose posted rate equals quote_signal x distance within 3%. The signal switches between an exact regime and a broken one at month boundaries.", 6.4))
    A(("figure", FIG / "eda_quote_vs_distance.png", "Figure 2. quote_signal against distance in the exact months, the degraded months and the scoring file. In the scoring file it is noise around $2.05 per mile.", 6.6))
    A(("h2", "2.3 About 1.4% of the training targets are corrupted"))
    A(("p", "675 development loads (1.41%) carry a posted rate that has been multiplied by a factor between about 2.3 and 5.6, or divided by one. Because the quote signal is exact in half the months, the corruption can be characterised precisely there: genuine loads have 1.57 to 3.57 dollars per mile, corrupted ones fall below 1.25 or above 4.2, and there is an empty gap between the populations (Figure 3). A rate-per-mile filter of 1.2 to 4.0 therefore removes them cleanly, and the same rule identifies the corrupted loads in the months where the quote cannot be used. The scoring file presumably contains the same 1.4%, which sets a floor on any raw error metric; all metrics in this report are given on all loads and on genuine loads."))
    A(("figure", FIG / "eda_rate_per_mile.png", "Figure 3. Distribution of posted_rate / distance on log axes. The dashed lines are the filter bounds; everything outside them is a corrupted target.", 6.2))
    A(("h2", "2.4 The market index is a daily market series with a weekly cycle"))
    A(("p", "market_index varies mostly between days (standard deviation 0.165 across daily means) and very little within a day (0.025), so it is a market-wide series with a small per-load jitter. It has a strong weekly cycle (peak Thursday, trough Sunday, about 19% peak-to-trough) on top of a slow swing that peaked in May and settled around 0.93 in November-December, lower than the development-period mean of 1.08 (Figure 4). Rates respond to it with an elasticity of about 0.15 log-points per index unit. The daily series is used to impute the 0.8% of loads with a missing value and to supply values for the fixed December chart."))
    A(("figure", FIG / "eda_market_index.png", "Figure 4. Daily mean market index with a 7-day mean; the shaded area is the scoring period.", 6.6))
    A(("h2", "2.5 Rates ramp up through quarter-end months, on top of a slow trend"))
    A(("p", f"After controlling for load features and the market index, rates rise steadily through March, June and September, ending the month about 4% above where they started, while other months are flat (Figure 5). There is also a slow upward drift of about {ms['trend_pct_per_month']:.1f}% per month over the year. Both effects matter for the scoring period: December is a quarter-end month, and the trend has to be extended two months past the data."))
    A(("figure", FIG / "eda_quarter_end_ramp.png", "Figure 5. Rate-per-mile relative to its month mean by day of month, quarter-end months versus the rest (model-free view; the fitted ramp is +4.1%).", 6.2))

    # ---- 3 quality
    A(("h1", "3. Data-quality issues and how they were handled"))
    A(("p", "Every repair leaves an indicator column and is counted in a quality report generated by the pipeline (reports/data_quality.md). Cleaning is applied identically to both files, except that the corrupted-target rule needs the target and so applies to the development file only."))
    A(("table", pd.DataFrame([
        ["Negative weight", "292 train / 145 validation", "Sign flip: the magnitudes match the normal weight distribution", "abs(weight) + indicator"],
        ["Missing weight", "300 / 165", "Missing at random", "Left as NaN for LightGBM (native handling) + indicator; median-filled in the linear stage"],
        ["Missing market_index", "374 / 249", "Market-wide daily series", "Same-day mean of the other loads (day-of-week profile fallback for unseen dates)"],
        ["Corrupted posted_rate", "675 (1.41%) / unknown", "Multiplicative x2.3-5.6 up or down", "Flagged by rate-per-mile outside 1.2-4.0; excluded from fitting, kept in evaluation"],
        ["quote_signal", "all rows", "Exact in 5 months, degraded in 5, noise in scoring file", "Excluded from the feature set"],
        ["Cities only in validation", "8 cities, 1,447 loads (12%)", "Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego", "No city identity features; geography via coordinates and lane geometry (tested in 4.4)"],
        ["Clipped coordinates, distance floor", "Boston/Providence lon -69.5; 69 loads at 70 mi", "Synthetic generator limits", "Distances are consistent with the coordinates, so used as given"],
    ], columns=["issue", "count", "diagnosis", "handling"]), "Table 2. Data-quality issues, their diagnosis and the fix applied."))

    # ---- 4 validation
    A(("h1", "4. Validation design and data split"))
    A(("p", "The scoring loads are all later than the development loads, so a random split would be misleading in two ways: it would let the model see the same days it is tested on, and it would reward the quote signal. Every split used here is chronological and every test window is later than everything the model was trained on."))
    A(("h2", "4.1 Primary holdout"))
    A(("p", f"Train on 1 January to 31 August ({int(bt.loc[4, 'n_train']):,} genuine loads after excluding corrupted targets) and test on 1 September to 31 October ({int(bt.loc[4, 'n_test']):,} loads, corrupted targets included so that the reported numbers are honest). This is exactly the horizon of the real task: two months of predictions starting the day after the training data ends. Three reference predictors are evaluated on the same split: the global median rate-per-mile times distance, the median rate-per-mile within equipment and distance band, and the quote signal times distance."))
    hold_tbl = pd.DataFrame([
        [name, money(r["MAE"]), pct(r["MAPE"]), f"{r['R2']:.3f}", money(r["inlier_MAE"]), pct(r["inlier_MAPE"]), f"{r['inlier_bias_pct']:+.2f}%"]
        for name, r in h.iterrows()], columns=["model", "MAE (all)", "MAPE (all)", "R² (all)", "MAE (genuine)", "MAPE (genuine)", "bias (genuine)"])
    A(("table", hold_tbl, "Table 3. Holdout results, September-October 2025. 'Genuine' excludes the 1.5% of corrupted targets in the test window; bias is the mean signed percentage error."))
    A(("p", f"The quote-signal predictor is the cautionary case: it explains posted rates almost perfectly in the exact months, yet on the holdout it is worse than a distance-band median because October is a degraded-quote month. In the scoring period, where the signal is pure noise, it would be worse still. By month, the model's genuine-load MAPE is {pct(n['by_month'].loc[0, 'MAPE'])} in September and {pct(n['by_month'].loc[1, 'MAPE'])} in October, with a bias of {n['by_month'].loc[1, 'bias_pct']:+.2f}% in October: the second month ahead is harder, as expected for a forecast, and the model slightly under-predicts it because the trend is a little steeper late in the year than the linear fit."))
    A(("figure", FIG / "holdout_daily_rpm.png", "Figure 6. Daily mean rate-per-mile on the holdout, actual versus predicted. The model tracks the weekly market cycle and the September quarter-end ramp.", 6.6))
    A(("h2", "4.2 Rolling-origin backtest"))
    A(("p", "One holdout is one draw. To check that the result is stable across the quote regimes and the market cycle, the same procedure is repeated over five two-month windows, each trained on everything strictly before it (Table 4, Figure 7). The fitted trend and quarter-end ramp are reported per window to show the time component is stable as data accumulates."))
    bt_tbl = pd.DataFrame([[short_window(r["test_window"]), f"{int(r['n_train']):,}", money(r["MAE"]), money(r["inlier_MAE"]), pct(r["inlier_MAPE"]), f"{r['inlier_bias_pct']:+.2f}%", f"{r['trend_pct_per_month']:+.2f}%", f"{r['quarter_end_ramp_pct']:+.1f}%"] for _, r in bt.iterrows()],
                          columns=["test window", "train loads", "MAE (all)", "MAE (genuine)", "MAPE (genuine)", "bias", "trend / month", "quarter-end ramp"])
    A(("table", bt_tbl, f"Table 4. Rolling-origin backtest. Mean MAE {money(bt['MAE'].mean())}, mean genuine-load MAPE {pct(bt['inlier_MAPE'].mean())}, mean absolute bias {bt['inlier_bias_pct'].abs().mean():.2f}%."))
    A(("figure", FIG / "backtest_mape.png", "Figure 7. Genuine-load MAPE and bias per backtest window.", 5.8))
    A(("h2", "4.3 Why the time component is separated from the trees"))
    A(("p", f"A single LightGBM given the same features plus the date features is the natural alternative. On the same five windows it reaches a genuine-load MAPE of {pct(pure['inlier_MAPE'])} with a mean absolute bias of {pure['abs_inlier_bias']:.2f}%, against {pct(hyb['inlier_MAPE'])} and {hyb['abs_inlier_bias']:.2f}% for the hybrid. Two things go wrong with the single model: trees cannot extrapolate the trend, so every future window is predicted at the last level seen; and the daily market index acts as a day identifier that lets trees memorise daily levels in-sample, which is why weekday and day-of-month features receive no importance in that model and the ramp is lost out of sample. Separating a linear time component that is fitted with structural controls, and giving the trees no date at all, fixes both."))
    A(("h2", "4.4 Unseen cities"))
    A(("p", f"Twelve percent of scoring loads touch a city that never appears in training. To measure the cost, eight training cities ({', '.join(un['held_out_cities'])}) were removed from the training data entirely and the holdout was scored separately for loads touching them: genuine-load MAPE {pct(un['unseen']['inlier_MAPE'])} on those loads against {pct(un['seen']['inlier_MAPE'])} on the rest, with the same bias. Coordinates and lane geometry carry the regional effect; a city-identity feature was tried and does not help even for seen cities."))
    A(("h2", "4.5 Leakage controls"))
    A(("bullets", ["No feature is derived from the target other than the corrupted-target flag, which is used only to choose fitting rows in the training window.",
                   "The market calendar is built from the market_index feature, which is available in both files; no target information enters it.",
                   "No target encoding of cities or lanes (tested; no gain, and unsafe for unseen cities).",
                   "Hyper-parameters were chosen on the holdout only; the backtest was run once with the final configuration."]))

    # ---- 5 model
    A(("h1", "5. Model"))
    A(("p", [("Decomposition. ", "b"), ("log(rate / distance) = time(date, market index) + structure(load). Multiplying the exponentiated sum by distance gives the rate.", None)]))
    A(("p", [("Time component. ", "b"), (f"Ordinary least squares on log rate-per-mile with a linear trend, the load's market index, a quarter-end ramp (zero outside March/June/September/December, rising from 0 to 1 through those months), a day-of-month term and weekday indicators, plus coarse structural controls (log distance and its square, weight, equipment) so that the time coefficients are not confounded. Only the time terms are kept. Fitted on all development data: trend {ms['trend_pct_per_month']:+.2f}% per month, market elasticity {ms['market_index_elasticity_log_pts']:.3f} log-points per index unit, quarter-end ramp {ms['quarter_end_ramp_pct']:+.1f}% first to last day, weekday effects within {min(wd.values()):+.2f}% to {max(wd.values()):+.2f}% of Monday. The trend is extended linearly past October; damped and frozen alternatives are available as switches and were worse on the backtest, as was a quadratic trend (which over-predicts historically).", None)]))
    A(("p", [("Structural component. ", "b"), ("LightGBM regression on the residual with distance, log distance, equipment, weight and a missing-weight indicator, pickup and delivery coordinates, their differences and days to quarter end. 63 leaves, minimum 40 loads per leaf, learning rate 0.03, 1,500 rounds, 80% feature and row subsampling, L2 regularisation 1.0; three boosters with different seeds are averaged. No date, trend or market feature is visible to the trees.", None)]))
    imp = n["importance"]; top = imp.head(6); rest = imp.iloc[6:]
    imp_tbl = pd.DataFrame([[r["feature"], pct(r["share"], 1)] for _, r in top.iterrows()], columns=["feature", "gain share"])
    A(("table", imp_tbl, f"Table 5. LightGBM gain share of the structural features (final model). Remaining features ({', '.join(rest['feature'])}) share the other {pct(rest['share'].sum(), 1)}."))
    A(("p", [("Alternatives tried and rejected. ", "b"), ("Using the quote signal (fails out of time); predicting the dollar rate directly instead of rate-per-mile (worse scale handling); Huber and L1 objectives (no gain once corrupted targets are excluded); city as a categorical feature and smoothed target encoding of cities and lanes (no gain, unsafe for unseen cities); LightGBM linear trees for extrapolation (worse); a smoothed market index in place of the load's own value (slightly worse); a quadratic trend (over-predicts on the backtest).", None)]))

    # ---- 6 december
    A(("h1", "6. Fixed December prediction chart"))
    A(("p", f"The chart required by the brief holds every input fixed (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb) and varies only the date across December 2025. The inputs were expanded to the model's schema exactly as a real load would be: coordinates from the city table, the market index from the daily calendar (validation.csv contains loads on every December day, so no market value is invented), quote signal left empty because the model does not use it. The predictions were written into the provided template and the chart was produced by the unmodified score.py."))
    A(("figure", ROOT / "scorer_results" / "candidate_december.png", "Figure 8. Output of score.py: predicted rate for the fixed load on each day of December 2025.", 6.6))
    A(("p", f"Reading the shape: the level starts at {money(dec['predicted_rate'].iloc[0])} on 1 December and ends at {money(dec['predicted_rate'].iloc[-1])} on 31 December ({money(dec['predicted_rate'].min())} to {money(dec['predicted_rate'].max())} overall). The rise is the quarter-end ramp (December is treated like March, June and September) plus two months of the fitted trend; the waves are the weekly market cycle, peaking mid-week when the market index peaks. For scale, the development data contains {int(lf['count'])} Dry Van loads on this lane at {money(lf['min'])} to {money(lf['max'])} (median {money(lf['median'])}), most of the later ones between $800 and $900, so the December level is consistent with the lane's history. No holiday effect is modelled because the development data shows none around New Year, Memorial Day, Independence Day or Labor Day; if Christmas week behaves differently in reality, the chart will not show it."))

    # ---- 7 predictions
    A(("h1", "7. Final predictions"))
    A(("p", f"The final model is refitted on all development data (47,325 genuine loads; 675 corrupted targets excluded) and applied to validation.csv after the same cleaning. The template's row order is preserved and every prediction is positive. Predicted rates have a mean of {money(ps['mean'])}, a median of {money(ps['50%'])} and range from {money(ps['min'])} to {money(ps['max'])}, in line with the development-data distribution. The provided scorer validates the file (12,000 ids, two columns) and the December file (31 days, fixed inputs)."))

    # ---- 8 limitations
    A(("h1", "8. Limitations and next steps"))
    A(("bullets", [
        "The trend is extrapolated linearly for two months. The backtest supports this, but the late-year slope looks slightly steeper than the full-year fit; if the scoring metric shows a negative bias, a locally estimated slope is the first thing to try.",
        "December is assumed to behave like the other quarter-end months. That is the natural reading of the data, but it is an assumption stated here rather than something the data can prove.",
        "Corrupted targets cannot be predicted, and they are presumably present in the scoring file at the same 1.4% rate; roughly half of any raw MAE on this data comes from them.",
        "Holiday effects are absent from the development data and therefore from the model.",
        "Prediction intervals were not produced; quantile boosters on the same residual would add them at little cost.",
    ]))

    # ---- appendix
    A(("h1", "Appendix: reproducing the results"))
    A(("code", "python -m pip install -r requirements.txt\npython -m pip install -e .\npython -m freight_rate.cli all        # quality, eda, validate, train, predict, score\npython -m pytest -q                   # 12 tests on synthetic data\npython scripts/extra_experiments.py   # section 4.3 and 4.4\npython scripts/build_report.py        # this document"))
    A(("p", "Repository layout: src/freight_rate/ (config, data, features, model, validate, eda, predict, cli), tests/, reports/ (tables and figures used here), outputs/ (predictions), data/ (provided files), docs/ (the brief), score.py (unmodified)."))
    return c


# ----------------------------------------------------------------------------- DOCX renderer
def render_docx(content: list, path: Path) -> None:
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9); s.top_margin = s.bottom_margin = Inches(0.8)
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(10.5)
    for lvl, size in ((1, 16), (2, 12.5)):
        hs = doc.styles[f"Heading {lvl}"]; hs.font.name = "Calibri"; hs.font.size = Pt(size); hs.font.color.rgb = TEAL; hs.font.bold = True

    def add_runs(par, text):
        if isinstance(text, str):
            par.add_run(text); return
        for seg, style in text:
            r = par.add_run(seg); r.bold = style == "b"; r.italic = style == "i"

    for kind, *args in content:
        if kind == "title":
            p = doc.add_paragraph(); r = p.add_run(args[0]); r.bold = True; r.font.size = Pt(24); r.font.color.rgb = TEAL
        elif kind == "subtitle":
            p = doc.add_paragraph(); r = p.add_run(args[0]); r.font.size = Pt(13); r.font.color.rgb = RGBColor(0x45, 0x5A, 0x60)
        elif kind == "meta":
            p = doc.add_paragraph(); r = p.add_run(args[0]); r.font.size = Pt(10); r.italic = True
            p.paragraph_format.space_after = Pt(14)
        elif kind == "h1":
            doc.add_heading(args[0], level=1)
        elif kind == "h2":
            doc.add_heading(args[0], level=2)
        elif kind == "p":
            p = doc.add_paragraph(); add_runs(p, args[0]); p.paragraph_format.space_after = Pt(6)
        elif kind == "bullets":
            for item in args[0]:
                doc.add_paragraph(item, style="List Bullet")
        elif kind == "code":
            for line in args[0].split("\n"):
                p = doc.add_paragraph(); r = p.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(9); p.paragraph_format.space_after = Pt(0)
            doc.add_paragraph()
        elif kind == "table":
            df, caption = args
            t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, col in enumerate(df.columns):
                cell = t.rows[0].cells[i]; cell.text = ""; run = cell.paragraphs[0].add_run(str(col)); run.bold = True; run.font.size = Pt(9)
            for _, row in df.iterrows():
                cells = t.add_row().cells
                for i, val in enumerate(row):
                    cells[i].text = ""; run = cells[i].paragraphs[0].add_run(str(val)); run.font.size = Pt(9)
            cap = doc.add_paragraph(); r = cap.add_run(caption); r.italic = True; r.font.size = Pt(9); cap.paragraph_format.space_after = Pt(10)
        elif kind == "figure":
            img, caption, width = args
            doc.add_picture(str(img), width=Inches(min(width, 6.7)))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(); r = cap.add_run(caption); r.italic = True; r.font.size = Pt(9)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER; cap.paragraph_format.space_after = Pt(10)
    doc.save(path)


# ----------------------------------------------------------------------------- HTML renderer
CSS = """
@page { size: Letter; margin: 0.8in 0.9in; }
body { font-family: Calibri, 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.38; color: #1e2a2e; margin: 0; }
h1 { font-size: 16pt; color: #064A56; margin: 22px 0 8px; page-break-after: avoid; }
h2 { font-size: 12.5pt; color: #064A56; margin: 16px 0 6px; page-break-after: avoid; }
p { margin: 0 0 8px; text-align: justify; }
.title { font-size: 24pt; font-weight: bold; color: #064A56; margin: 0 0 4px; }
.subtitle { font-size: 13pt; color: #455A60; margin: 0 0 4px; }
.meta { font-size: 10pt; font-style: italic; margin: 0 0 18px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 4px; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d4d7; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #e6eef0; color: #064A56; }
.caption { font-size: 9pt; font-style: italic; color: #455A60; margin: 2px 0 12px; }
figure { margin: 8px 0 12px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
pre { font-family: Consolas, monospace; font-size: 9pt; background: #f3f6f7; padding: 8px; border: 1px solid #dfe6e8; }
ul { margin: 0 0 8px 18px; padding: 0; } li { margin-bottom: 4px; }
"""


def render_html(content: list, path: Path) -> None:
    def esc(x): return html.escape(str(x))
    def runs(text):
        if isinstance(text, str): return esc(text)
        return "".join(f"<b>{esc(s)}</b>" if st == "b" else (f"<i>{esc(s)}</i>" if st == "i" else esc(s)) for s, st in text)
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>Freight Rate Prediction</title><style>{CSS}</style></head><body>"]
    for kind, *args in content:
        if kind in ("title", "subtitle", "meta"): out.append(f"<p class='{kind}'>{esc(args[0])}</p>")
        elif kind in ("h1", "h2"): out.append(f"<{kind}>{esc(args[0])}</{kind}>")
        elif kind == "p": out.append(f"<p>{runs(args[0])}</p>")
        elif kind == "bullets": out.append("<ul>" + "".join(f"<li>{esc(i)}</li>" for i in args[0]) + "</ul>")
        elif kind == "code": out.append(f"<pre>{esc(args[0])}</pre>")
        elif kind == "table":
            df, caption = args
            head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
            body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for _, row in df.iterrows())
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table><p class='caption'>{esc(caption)}</p>")
        elif kind == "figure":
            img, caption, width = args
            data = base64.b64encode(Path(img).read_bytes()).decode()
            out.append(f"<figure><img src='data:image/png;base64,{data}' style='width:{min(width, 6.7)}in'><figcaption class='caption'>{esc(caption)}</figcaption></figure>")
    out.append("</body></html>")
    path.write_text("\n".join(out), encoding="utf-8")


def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    candidates = [shutil.which("msedge"), shutil.which("chrome"), shutil.which("google-chrome"),
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    browser = next((c for c in candidates if c and Path(c).exists()), None)
    if browser is None:
        print("no headless browser found; PDF not produced (open the HTML and print to PDF)"); return False
    profile = Path(config.OUTPUT_DIR) / "_browser_profile"; profile.mkdir(parents=True, exist_ok=True)
    pdf_path.unlink(missing_ok=True)  # so that existence below is a true signal
    cmd = [browser, "--headless", "--disable-gpu", "--no-first-run", f"--user-data-dir={profile}", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    subprocess.run(cmd, check=True, timeout=180, capture_output=True)
    # the launcher process can return before the renderer has finished writing the file
    for _ in range(120):
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            break
        time.sleep(1)
    time.sleep(2)
    shutil.rmtree(profile, ignore_errors=True)
    return pdf_path.exists()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--candidate", default="Fasih Ur Rehman"); ap.add_argument("--repo-url", default="github.com (link in README)"); ap.add_argument("--loom-url", default="")
    ap.add_argument("--name", default="Freight_Rate_Report"); args = ap.parse_args()
    n = load_numbers(); content = build_content(n, args.candidate, args.repo_url, args.loom_url)
    docx_path = config.REPORT_DIR / f"{args.name}.docx"; html_path = config.REPORT_DIR / f"{args.name}.html"; pdf_path = config.REPORT_DIR / f"{args.name}.pdf"
    render_docx(content, docx_path); render_html(content, html_path); ok = html_to_pdf(html_path, pdf_path)
    print(f"wrote {docx_path}\nwrote {html_path}\n" + (f"wrote {pdf_path}" if ok else "PDF not written"))


if __name__ == "__main__":
    main()
