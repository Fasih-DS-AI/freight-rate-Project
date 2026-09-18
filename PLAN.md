# Plan: Spotter ML Engineer assessment

Goal: maximise the chance of an offer. The reviewers will look at four things in this order:
the write-up and Loom (does this person think like an ML engineer?), the code (would I want
this in my repo?), the December chart (does the model behave sensibly on a controlled input?),
and the validation score (is it accurate?). The plan below optimises all four, in that order.

## 1. What the assessment is really testing

The dataset is synthetic and every "trap" in it is deliberate. Finding and handling them
well *is* the assessment:

| trap | evidence | handling |
|---|---|---|
| Temporal extrapolation | train Jan-Oct, score Nov-Dec, chart for Dec | chronological holdout, rolling backtest, parametric trend |
| `quote_signal` leak | exact in Jan-Mar/Jun/Sep, degraded otherwise, pure noise in Nov-Dec | excluded, with a demonstration of what happens if you use it |
| Corrupted targets | 1.4% of rates x/÷ 2.3-5.6, cleanly separable by $/mile | flagged and excluded from fitting; reported separately in metrics |
| Sign-flipped and missing weights | 0.6% each | `abs` + indicator |
| Missing market index | 0.8% | same-day market mean (it is a daily series) |
| Unseen cities | 8 cities only in validation (12% of loads) | coordinates + lane geometry, no city identity; simulated by holding cities out |
| Quarter-end ramp | +4% through Mar/Jun/Sep | explicit ramp feature; December is a quarter-end month |
| Weekly market cycle | peak Thu, trough Sun | market index + weekday terms |
| Clipped coordinates / distance floor | Boston and Providence share lon -69.5; distance floor 70 mi | documented; distances are internally consistent so used as given |

## 2. Pipeline (built, in `src/freight_rate/`)

```
python -m freight_rate.cli all
   quality  -> flag-and-fix cleaning + counts          reports/data_quality.md
   eda      -> six model-free figures + summary         reports/eda_summary.md, figures/eda_*.png
   validate -> holdout + baselines + rolling backtest   reports/validation_summary.md, figures/holdout_*.png
   train    -> final fit on all Jan-Oct data            outputs/models/hybrid_rate_model.pkl
   predict  -> validation_predictions.csv + December    root + outputs/ + data/december_chart_inputs.csv
   score    -> Spotter's score.py                       scorer_results/candidate_december.png
```

Model: `log(rate/mile) = time(date, market) + structure(load)`; time part is linear (trend,
market index, quarter-end ramp, weekday) so it extrapolates; structural part is a 3-seed
LightGBM on the residual with no date features. Current numbers: holdout MAPE 1.47% on
genuine loads (bias -0.7%), backtest 1.31-1.60%.

## 3. Remaining work, in priority order

### A. Model refinements (half a day)
1. **Trend shape (checked).** October is under-predicted by ~1.5% in the holdout and the
   monthly levels look mildly convex (linear slope 0.6%/month, local slope near October
   ~1%/month). On the five-fold backtest a quadratic trend over-predicts (+0.54% mean bias)
   and a quadratic-in-sample/linear-extrapolation variant is only marginally better than
   linear (|bias| 0.46% vs 0.55%, MAPE 1.43% vs 1.45%), so the linear trend stays. Worth one
   sentence in the report as a stated assumption; `--trend damped` is the conservative switch.
2. **Light hyper-parameter search** on the holdout only (num_leaves 31/63/127, rounds with
   lr 0.015-0.03, min_data_in_leaf). Keep whatever changes MAPE by more than 0.05 pp.
3. **Unseen-city robustness**: re-run the hold-8-cities simulation through the pipeline and
   quote the number in the report (currently +0.2 pp MAPE on loads touching unseen cities).
4. **Sanity constraints**: monotone constraint on distance (rate must rise with distance) and
   a floor/ceiling on predicted $/mile (1.2-4.0) as a guard rail; confirm no change in score.
5. Optional: quantile (P10/P90) boosters for uncertainty bands in the report.

### B. Report (PDF, ~6 pages, half a day) - structure
1. Problem framing and data (one paragraph; the forecast horizon).
2. Data quality: the trap table above with counts; figures `eda_quote_regimes.png`,
   `eda_quote_vs_distance.png`, `eda_rate_per_mile.png`.
3. Key findings: `eda_market_index.png`, `eda_quarter_end_ramp.png`, `eda_monthly_profile.png`.
4. Validation design (required section): why chronological, the holdout, the backtest table,
   baselines, unseen-city simulation, metrics on all vs genuine loads. `backtest_mape.png`,
   `holdout_daily_rpm.png`, `holdout_pred_vs_actual.png`.
5. Model: the decomposition, why hybrid beats a pure GBM (1.97% -> 1.52%, bias 1.35% -> 0.53%),
   feature importance, fitted time coefficients (trend, elasticity, ramp, weekday).
6. December chart (required) with a plain-English reading of its shape and the assumptions
   (trend extended, December treated as quarter-end, market index from validation.csv).
7. Limitations and next steps (holiday effects not observable in the data, trend
   extrapolation risk, corrupted targets in the scoring set).
Build with `docx` -> PDF; keep every number sourced from `reports/`.

### C. Loom (2-3 minutes) - script
- 0:00-0:25 Framing: two-month-ahead forecast; all validation chronological.
- 0:25-1:05 Findings: quote_signal regimes (show `eda_quote_regimes.png`, then the
  validation panel of `eda_quote_vs_distance.png`); corrupted targets; market index cycle;
  quarter-end ramp.
- 1:05-1:35 Data-quality fixes: weights, market index, outlier exclusion, unseen cities.
- 1:35-2:10 Model choice: why the hybrid (extrapolation + no date memorisation), backtest
  numbers, baseline comparison.
- 2:10-2:40 Code walkthrough: `config.py` (decisions), `model.py` (`time_component`,
  `fit`), `validate.py` (`run_backtest`), `cli.py all`.
- 2:40-3:00 December chart and what its shape means.
Record after the report is final so numbers match.

### D. Repository polish (1-2 hours)
- Commit history with meaningful messages; push to a public GitHub repo; verify a fresh clone
  runs `pip install -r requirements.txt && python -m freight_rate.cli all` in a clean venv
  (pandas 2.x, since the provided requirements pin `pandas<3`).
- README: keep the numbers in sync with `reports/validation_summary.md`.
- Add `reports/` figures and the final PDF to the repo; add the Loom link to the README.
- Final checklist: `validation_predictions.csv` has exactly `load_id,predicted_rate`, 12,000
  rows, positive; `score.py` passes; chart embedded in the report.

## 4. Risks and how they are covered
- **Scoring metric unknown.** Predictions are unbiased in log space and evaluated on MAE, RMSE
  and MAPE; the corrupted-target floor is explained so a mediocre raw MAE is not misread.
- **Trend reverses in Nov-Dec.** The `--trend damped` and `--trend none` switches exist and
  the backtest shows the full trend is best historically; the report states the assumption.
- **Reviewer environment.** No exotic dependencies (pandas, numpy, scikit-learn, lightgbm,
  matplotlib); tests run on synthetic data in two seconds.
