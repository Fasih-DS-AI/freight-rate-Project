# Freight Rate Prediction (Spotter ML Engineer assessment)

Predicts the `posted_rate` of truckload freight from lane, equipment, weight, date and a
market index, for the 12,000 loads in `data/validation.csv` (Nov-Dec 2025) after learning
from the 48,000 labelled loads in `data/train_test.csv` (Jan-Oct 2025). The provided
scorer is untouched (`score.py`); everything else lives in `src/freight_rate/`.

## Deliverables

| item | where |
|---|---|
| Predictions | [`validation_predictions.csv`](validation_predictions.csv) (repo root; copy in `outputs/`) |
| Report (validation approach, data split, December chart) | [`reports/Freight_Rate_Report.pdf`](reports/Freight_Rate_Report.pdf) (also `.docx`) |
| December chart from `score.py` | [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) |
| Walkthrough video (Loom, ~3 min) | https://www.loom.com/share/aef38079b00342b2af36746f4e08f382 |

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; use source .venv/bin/activate elsewhere
python -m pip install -r requirements.txt
python -m pip install -e .                            # or: set PYTHONPATH=src
python -m freight_rate.cli all                        # ~3 minutes on a laptop CPU
```

`all` runs, in order:

| step | what it does | writes |
|---|---|---|
| `quality` | loads both CSVs, repairs and counts every data issue | `reports/data_quality.md` |
| `eda` | model-free exploratory figures and a summary | `reports/eda_summary.md`, `reports/figures/eda_*.png` |
| `validate` | chronological holdout (train Jan-Aug, test Sep-Oct) with baselines, plus a five-fold rolling-origin backtest | `reports/validation_summary.md`, `reports/*.csv`, `reports/figures/holdout_*.png`, `backtest_mape.png` |
| `train` | fits the final model on all development data (corrupted targets excluded) | `outputs/models/hybrid_rate_model.pkl` |
| `predict` | fills the prediction template and the December chart inputs | `validation_predictions.csv` (root and `outputs/`), `data/december_chart_inputs.csv`, `outputs/december_chart_inputs.csv` |
| `score` | runs Spotter's `score.py` on those files | `scorer_results/candidate_december.png` |

Each step can be run on its own (`python -m freight_rate.cli validate`). Options:
`--trend {full,damped,none}` (how the fitted trend extends past October), `--seeds N`,
`--rounds N`, `--lr X`, `--skip-backtest`. Unit tests: `python -m pytest -q`.

## Repository layout

```
data/                      provided inputs (README-conformant names); data/original/ keeps the untouched originals
docs/                      the assessment brief (PDF + original README)
src/freight_rate/
  config.py                paths, constants and every modelling decision with its justification
  data.py                  schema checks, city table, daily market calendar, flag-and-fix cleaning, quality report
  features.py              structural (load) features and time features
  model.py                 HybridRateModel: parametric time component + LightGBM structural component
  validate.py              chronological holdout, rolling backtest, baselines, report + figures
  eda.py                   exploratory figures
  predict.py               deliverable files (template filling, December inputs)
  cli.py                   command-line entry point
tests/                     unit and smoke tests (synthetic data, run in ~2 s)
reports/                   generated tables and figures used in the write-up
outputs/                   generated predictions and the saved model
score.py                   Spotter's scorer, unmodified
```

## What the data showed (details in `reports/eda_summary.md`)

- **The task is a two-month-ahead forecast.** Development data ends 31 Oct 2025; every
  scoring load is in Nov-Dec 2025. All validation is therefore chronological.
- **`quote_signal` is a trap.** It equals the true rate-per-mile within +-1% in Jan-Mar, Jun and
  Sep, is a degraded, downward-biased signal in Apr-May, Jul-Aug and Oct, and in the scoring
  file it has no relationship with distance, equipment, weight or region at all (R² = 0.002).
  A model that leans on it scores brilliantly in-sample and fails out of time. It is excluded.
- **1.4% of training targets are corrupted** by a multiplicative factor (about x2.3-5.6 up or
  down). Genuine loads have 1.57-3.57 $/mile; corrupted ones fall outside 1.25-4.2 $/mile with
  an empty gap between, so a rate-per-mile filter of (1.2, 4.0) removes them cleanly.
- **`market_index` is a market-wide daily series** (weekly cycle, peak Thursday) with a small
  per-load jitter. Missing values (0.8%) are imputed from the same-day mean, and the same
  calendar supplies December values for the fixed chart.
- **Negative weights** (0.6%) are sign flips and are fixed with `abs`; missing weights (0.6%)
  get an indicator.
- **Rates ramp up about 4% through quarter-end months** (Mar, Jun, Sep) and follow a slow
  upward trend (about +0.6% per month) on top of the market index. December is a quarter-end month.
- **Eight validation cities never appear in training** (12% of scoring loads), so geography is
  modelled with coordinates rather than city identity.

## Model

`log(rate per mile) = time(date, market_index) + structure(load)`

1. **Time component** (linear regression): trend, market index, quarter-end ramp, weekday.
   Linear so that it can be extrapolated two months past the training window.
2. **Structural component** (LightGBM, 3 seeds averaged) fitted on the residual with distance,
   equipment, weight, coordinates and lane geometry. It never sees a date, so it cannot
   memorise daily levels.

Multiply by distance to get the rate. On the chronological holdout (train Jan-Aug, predict
Sep-Oct) the model reaches a MAPE of about 1.5% on genuine loads, against 3.7% for an
equipment-by-distance-band median and 9.9% for `quote_signal x distance`. The rolling backtest
(five two-month windows) gives 1.3-1.6% with a bias under 1%. Exact numbers for the current
run are in `reports/validation_summary.md`.

About half of the raw MAE on any held-out period comes from the 1.4% of corrupted targets,
which no model can predict; metrics are reported both on all loads and on genuine loads.

## December chart

`data/december_chart_inputs.csv` is filled from the same model: coordinates from the city
table, the market index from the daily calendar (validation.csv has loads on every December
day), the trend extended linearly, and December treated as a quarter-end month. The result
shows the weekly market cycle on a rising ramp; see `scorer_results/candidate_december.png`.
