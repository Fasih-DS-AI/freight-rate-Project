### train_test.csv (48,000 rows)
- negative weights (sign flipped, fixed with abs): 292
- missing weights (left as NaN + indicator; median-filled for the linear stage): 300
- missing market_index (imputed from the same-day market mean): 374
- corrupted targets (rate-per-mile outside (1.2, 4.0), excluded from fitting): 675

### validation.csv (12,000 rows)
- negative weights (sign flipped, fixed with abs): 145
- missing weights (left as NaN + indicator; median-filled for the linear stage): 165
- missing market_index (imputed from the same-day market mean): 249
- cities not present in training data: Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego