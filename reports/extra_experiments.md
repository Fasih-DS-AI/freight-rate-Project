# Supporting experiments

## Hybrid vs single LightGBM with date features (rolling backtest means)

| model                              |     MAE |   inlier_MAPE |   abs_inlier_bias |
|:-----------------------------------|--------:|--------------:|------------------:|
| hybrid                             | 88.6046 |        0.0145 |            0.5511 |
| single LightGBM with date features | 98.8314 |        0.0187 |            1.2394 |

## Unseen-city simulation (holdout Sep-Oct, 8 cities removed from training)

held-out cities: Tulsa, Washington, San Antonio, Memphis, Toledo, Hartford, Bakersfield, New York

| loads                  |     MAE |     RMSE |   MAPE |     R2 |   bias_pct |    n |   inlier_MAE |   inlier_RMSE |   inlier_MAPE |   inlier_bias_pct |   inlier_n |
|:-----------------------|--------:|---------:|-------:|-------:|-----------:|-----:|-------------:|--------------:|--------------:|------------------:|-----------:|
| seen cities            | 89.4320 | 618.2592 | 0.0403 | 0.8319 |     0.6762 | 7007 |      34.3316 |       49.0213 |        0.0151 |           -0.7850 |       6902 |
| touching unseen cities | 98.8625 | 662.3009 | 0.0419 | 0.8229 |     0.6151 | 2516 |      39.9204 |       58.2490 |        0.0168 |           -0.8872 |       2477 |