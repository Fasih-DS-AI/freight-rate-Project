# EDA summary (train_test.csv, Jan-Oct 2025; validation.csv, Nov-Dec 2025)

- development loads: 48,000 over 304 days; scoring loads: 12,000 over 61 days (strictly later).
- posted_rate is essentially rate-per-mile x distance; rate-per-mile falls with distance (elasticity -0.129) and is higher for Reefer/Flatbed.
- quote_signal equals the true rate-per-mile (+-1%) in months [1, 2, 3, 6, 9] but is degraded in months [4, 5, 7, 8, 10]; in validation.csv its correlation with log distance is +0.011 (vs -0.77 in exact months): it is noise there and is excluded from the model.
- 675 training targets (1.41%) are corrupted by a multiplicative factor (median x3.6 up or /3.3 down); they sit outside (1.2, 4.0) $/mile and are excluded from fitting.
- market_index is a market-wide daily series (weekly cycle amplitude 19%); Nov-Dec level 0.927 vs Jan-Oct 1.082. Missing values are imputed from the same-day mean.
- rates ramp up by about +3.4% through quarter-end months (Mar/Jun/Sep) versus +0.8% in other months; December is a quarter-end month.
- weights: some negative (sign-flipped) and some missing; distances are consistent with the (synthetic, clipped) coordinates; eight validation cities never appear in training, so geography is modelled with coordinates rather than city identity.

Figures: `reports/figures/eda_*.png`.