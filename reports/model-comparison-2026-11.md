# Model comparison

Generated 2026-11-05 00:13 UTC by `python -m app.ml train`. Feature set `v1-079714a734`.

## Data and protocol

- Split: dates: validation from 2024-08-01, test from 2025-08-01 to 2026-06-01.
- Train: 2022-08-05 – 2024-05-19 (760 matches); validation: 2024-08-16 – 2025-05-25 (380 matches); test: 2025-08-15 – 2026-05-24 (380 matches).
- Tuning and selection metric: `cv_log_loss` (lower is better). Time-aware cross-validation over 2 rolling-origin folds; each fold is predicted by a model fitted only on earlier matches:
  1. fit 2022-08-27 – 2023-05-28 (350 matches) → predict 2023-08-11 – 2024-05-19 (380 matches)
  2. fit 2022-08-27 – 2024-05-19 (721 matches) → predict 2024-08-16 – 2025-05-25 (380 matches)
- The test period is used once, after selection, to report the tuned models refitted on train + validation. It influenced no choice.

## Selection (no test data)

| Model | Type | Selection score | Validation log loss | CV log loss | Grid points | Chosen hyperparameters |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **elo** (selected) | EloModel | 0.9746 | 1.0008 | 0.9746 | 1 | – |
| random_forest | RandomForestModel | 0.9884 | 1.0096 | 0.9884 | 6 | max_depth=4, min_samples_leaf=60 |
| poisson | PoissonModel | 0.9942 | 1.0195 | 0.9942 | 4 | alpha=10.0 |
| lightgbm | LightGBMModel | 0.9955 | 1.0196 | 0.9955 | 6 | num_leaves=3, n_estimators=100 |
| xgboost | XGBoostModel | 1.0006 | 1.0228 | 1.0006 | 6 | max_depth=2, n_estimators=100 |
| logistic_regression | LogisticModel | 1.0133 | 1.0585 | 1.0133 | 5 | c=0.003, goal_alpha=10.0 |
| historical_baseline | HistoricalBaseline | 1.0710 | 1.0862 | 1.0710 | 1 | smoothing=1.0 |

## Validation metrics

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0862 | 0.6596 | 0.2372 | 40.8% | 1.281 | 1.620 | 11.8% | 28.2% | 41.8% | -3.088 |
| **elo** (selected) | 1.0008 | 0.5991 | 0.2059 | 51.6% | 1.353 | 1.685 | 12.1% | 28.2% | 45.3% | -3.021 |
| logistic_regression | 1.0585 | 0.6363 | 0.2242 | 50.3% | 1.331 | 1.677 | 12.4% | 28.4% | 40.8% | -3.061 |
| poisson | 1.0195 | 0.6104 | 0.2123 | 51.1% | 1.321 | 1.670 | 10.8% | 31.6% | 45.5% | -3.021 |
| random_forest | 1.0096 | 0.6042 | 0.2095 | 51.1% | 1.313 | 1.631 | 11.6% | 29.7% | 46.3% | -2.988 |
| xgboost | 1.0228 | 0.6140 | 0.2143 | 50.8% | 1.306 | 1.626 | 12.1% | 27.9% | 45.0% | -3.006 |
| lightgbm | 1.0196 | 0.6109 | 0.2123 | 51.1% | 1.306 | 1.626 | 11.6% | 29.7% | 46.3% | -3.001 |

## Cross-validation metrics (pooled over folds)

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0710 | 0.6490 | 0.2358 | 43.4% | 1.324 | 1.665 | 8.8% | 26.1% | 38.7% | -3.197 |
| **elo** (selected) | 0.9746 | 0.5802 | 0.2013 | 54.1% | 1.355 | 1.680 | 12.0% | 27.8% | 44.1% | -3.062 |
| logistic_regression | 1.0133 | 0.6050 | 0.2125 | 52.9% | 1.355 | 1.696 | 11.7% | 26.7% | 40.9% | -3.100 |
| poisson | 0.9942 | 0.5921 | 0.2074 | 54.3% | 1.350 | 1.693 | 11.1% | 29.1% | 42.9% | -3.081 |
| random_forest | 0.9884 | 0.5892 | 0.2058 | 54.3% | 1.340 | 1.659 | 11.3% | 27.5% | 43.2% | -3.057 |
| xgboost | 1.0006 | 0.5974 | 0.2096 | 52.9% | 1.334 | 1.660 | 11.3% | 26.8% | 43.2% | -3.075 |
| lightgbm | 0.9955 | 0.5938 | 0.2074 | 53.6% | 1.334 | 1.656 | 10.8% | 27.4% | 43.3% | -3.067 |

## Test metrics (reported only)

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0843 | 0.6561 | 0.2278 | 42.6% | 1.222 | 1.592 | 12.4% | 27.9% | 41.8% | -2.976 |
| **elo** (selected) | 1.0305 | 0.6203 | 0.2096 | 48.2% | 1.308 | 1.640 | 9.2% | 28.9% | 47.1% | -2.929 |
| logistic_regression | 1.0439 | 0.6225 | 0.2088 | 51.1% | 1.241 | 1.586 | 10.5% | 31.6% | 49.2% | -2.907 |
| poisson | 1.0375 | 0.6224 | 0.2110 | 50.5% | 1.237 | 1.585 | 11.3% | 32.1% | 46.1% | -2.901 |
| random_forest | 1.0306 | 0.6196 | 0.2094 | 48.9% | 1.268 | 1.594 | 10.3% | 32.4% | 49.7% | -2.899 |
| xgboost | 1.0336 | 0.6197 | 0.2092 | 49.2% | 1.259 | 1.597 | 11.1% | 30.3% | 47.1% | -2.899 |
| lightgbm | 1.0344 | 0.6206 | 0.2094 | 49.5% | 1.261 | 1.599 | 11.3% | 30.3% | 48.2% | -2.901 |

## Derived markets on test (from each model's score grid)

Over/under and both-teams-to-score probabilities are sums over the same score grid as the 1X2 and exact-score figures; lower is better.

| Model | O/U 1.5 Brier | O/U 2.5 Brier | O/U 2.5 log loss | O/U 3.5 Brier | BTTS Brier | BTTS log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 0.1665 | 0.2482 | 0.6896 | 0.2100 | 0.2465 | 0.6861 |
| **elo** (selected) | 0.1742 | 0.2644 | 0.7234 | 0.2167 | 0.2529 | 0.6994 |
| logistic_regression | 0.1653 | 0.2474 | 0.6880 | 0.2092 | 0.2449 | 0.6830 |
| poisson | 0.1656 | 0.2484 | 0.6900 | 0.2087 | 0.2451 | 0.6833 |
| random_forest | 0.1663 | 0.2518 | 0.6970 | 0.2111 | 0.2466 | 0.6863 |
| xgboost | 0.1662 | 0.2508 | 0.6950 | 0.2099 | 0.2474 | 0.6881 |
| lightgbm | 0.1663 | 0.2512 | 0.6958 | 0.2101 | 0.2477 | 0.6886 |

## Test log loss compared with the selected model (elo)

Mean per-match difference (negative = better than the selected model), with a 95% paired bootstrap interval over test matches (5000 resamples). An interval that contains 0 means the test set cannot distinguish the two models.

| Model | Difference | 95% interval | Resamples where better |
| --- | ---: | --- | ---: |
| random_forest | +0.0001 | [-0.0151, +0.0156] | 49% |
| xgboost | +0.0031 | [-0.0169, +0.0234] | 38% |
| lightgbm | +0.0039 | [-0.0148, +0.0229] | 34% |
| poisson | +0.0070 | [-0.0170, +0.0312] | 29% |
| logistic_regression | +0.0134 | [-0.0121, +0.0423] | 17% |
| historical_baseline | +0.0538 | [+0.0148, +0.0938] | 0% |

## Most important features (tree models, 1X2 classifier of the final model)

| Rank | random_forest | xgboost | lightgbm |
| ---: | --- | --- | --- |
| 1 | elo_diff (0.205) | elo_home_win_expectancy (0.059) | elo_diff (0.127) |
| 2 | elo_home_win_expectancy (0.194) | elo_diff (0.057) | home_attack_strength (0.093) |
| 3 | rating_expected_home_goals (0.128) | rating_expected_home_goals (0.049) | elo_home_win_expectancy (0.073) |
| 4 | home_attack_strength (0.075) | home_elo (0.034) | rating_expected_away_goals (0.065) |
| 5 | rating_expected_away_goals (0.060) | home_attack_strength (0.029) | rating_expected_home_goals (0.063) |
| 6 | home_elo (0.055) | rating_expected_away_goals (0.024) | competition_draw_rate (0.045) |
| 7 | home_defence_weakness (0.021) | away_form5_win_rate (0.021) | away_form10_draw_rate (0.037) |
| 8 | home_attack_rating (0.018) | home_form10_points (0.020) | away_defence_rating (0.037) |
| 9 | h2h_weighted_points (0.012) | away_elo (0.020) | home_elo (0.035) |
| 10 | away_elo (0.012) | away_form10_loss_rate (0.019) | away_form5_goals_for (0.032) |
