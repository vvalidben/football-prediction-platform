# Model comparison

Generated 2026-11-05 01:30 UTC by `python -m app.ml train`. Feature set `v1-f33b9f0f5a`.

## Data and protocol

- Split: dates: validation from 2024-08-01, test from 2025-08-01 to 2026-06-01.
- Train: 2022-08-05 – 2024-07-28 (4326 matches); validation: 2024-08-03 – 2025-07-29 (3614 matches); test: 2025-08-02 – 2026-05-31 (3663 matches).
- Tuning and selection metric: `cv_log_loss` (lower is better). Time-aware cross-validation over 2 rolling-origin folds; each fold is predicted by a model fitted only on earlier matches:
  1. fit 2022-08-27 – 2024-06-14 (3740 matches) → predict 2024-06-14 – 2024-07-28 (164 matches)
  2. fit 2022-08-27 – 2024-07-28 (3868 matches) → predict 2024-08-03 – 2025-07-29 (3614 matches)
- The test period is used once, after selection, to report the tuned models refitted on train + validation. It influenced no choice.

## Selection (no test data)

| Model | Type | Selection score | Validation log loss | CV log loss | Grid points | Chosen hyperparameters |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **poisson** (selected) | PoissonModel | 0.9987 | 0.9953 | 0.9987 | 4 | alpha=1.0 |
| random_forest | RandomForestModel | 1.0012 | 0.9991 | 1.0012 | 6 | max_depth=4, min_samples_leaf=60 |
| elo | EloModel | 1.0041 | 1.0016 | 1.0041 | 1 | – |
| xgboost | XGBoostModel | 1.0041 | 1.0024 | 1.0041 | 6 | max_depth=2, n_estimators=250 |
| lightgbm | LightGBMModel | 1.0043 | 1.0024 | 1.0043 | 6 | num_leaves=3, n_estimators=250 |
| logistic_regression | LogisticModel | 1.0068 | 1.0035 | 1.0068 | 5 | c=0.003, goal_alpha=10.0 |
| historical_baseline | HistoricalBaseline | 1.0697 | 1.0703 | 1.0697 | 1 | smoothing=1.0 |

## Validation metrics

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0703 | 0.6473 | 0.2294 | 44.4% | 1.322 | 1.650 | 12.1% | 32.0% | 46.0% | -3.003 |
| elo | 1.0016 | 0.5989 | 0.2055 | 51.3% | 1.307 | 1.641 | 12.3% | 32.2% | 49.7% | -2.918 |
| logistic_regression | 1.0035 | 0.5986 | 0.2048 | 51.1% | 1.303 | 1.624 | 12.8% | 34.3% | 49.8% | -2.907 |
| **poisson** (selected) | 0.9953 | 0.5944 | 0.2036 | 51.5% | 1.301 | 1.617 | 12.7% | 33.6% | 50.3% | -2.892 |
| random_forest | 0.9991 | 0.5970 | 0.2047 | 51.1% | 1.299 | 1.616 | 12.6% | 33.5% | 50.0% | -2.895 |
| xgboost | 1.0024 | 0.5985 | 0.2051 | 51.3% | 1.299 | 1.616 | 12.9% | 33.8% | 49.8% | -2.899 |
| lightgbm | 1.0024 | 0.5987 | 0.2051 | 51.2% | 1.299 | 1.616 | 12.7% | 32.9% | 49.6% | -2.899 |

## Cross-validation metrics (pooled over folds)

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0697 | 0.6469 | 0.2291 | 44.5% | 1.310 | 1.635 | 12.3% | 32.4% | 46.6% | -2.991 |
| elo | 1.0041 | 0.6004 | 0.2063 | 51.1% | 1.295 | 1.626 | 12.5% | 32.4% | 50.1% | -2.909 |
| logistic_regression | 1.0068 | 0.6010 | 0.2060 | 50.7% | 1.291 | 1.611 | 13.0% | 34.6% | 50.1% | -2.899 |
| **poisson** (selected) | 0.9987 | 0.5967 | 0.2048 | 51.2% | 1.291 | 1.603 | 12.9% | 33.7% | 50.6% | -2.885 |
| random_forest | 1.0012 | 0.5983 | 0.2054 | 50.8% | 1.286 | 1.601 | 12.8% | 33.9% | 50.4% | -2.886 |
| xgboost | 1.0041 | 0.5997 | 0.2057 | 51.1% | 1.286 | 1.601 | 13.2% | 34.2% | 50.3% | -2.889 |
| lightgbm | 1.0043 | 0.5999 | 0.2058 | 51.0% | 1.287 | 1.601 | 13.0% | 33.4% | 50.1% | -2.889 |

## Test metrics (reported only)

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0690 | 0.6463 | 0.2286 | 44.7% | 1.307 | 1.641 | 11.9% | 30.5% | 45.0% | -3.001 |
| elo | 1.0073 | 0.6021 | 0.2073 | 50.5% | 1.307 | 1.642 | 12.0% | 32.5% | 48.6% | -2.927 |
| logistic_regression | 1.0156 | 0.6062 | 0.2085 | 50.5% | 1.290 | 1.617 | 12.4% | 32.4% | 49.1% | -2.916 |
| **poisson** (selected) | 1.0075 | 0.6020 | 0.2071 | 50.5% | 1.286 | 1.609 | 12.2% | 32.8% | 49.4% | -2.903 |
| random_forest | 1.0051 | 0.6006 | 0.2067 | 50.4% | 1.288 | 1.612 | 12.5% | 32.8% | 49.2% | -2.903 |
| xgboost | 1.0058 | 0.6007 | 0.2067 | 50.4% | 1.289 | 1.614 | 12.4% | 33.0% | 49.5% | -2.904 |
| lightgbm | 1.0063 | 0.6011 | 0.2069 | 50.6% | 1.290 | 1.616 | 12.4% | 32.8% | 48.9% | -2.906 |

## Derived markets on test (from each model's score grid)

Over/under and both-teams-to-score probabilities are sums over the same score grid as the 1X2 and exact-score figures; lower is better.

| Model | O/U 1.5 Brier | O/U 2.5 Brier | O/U 2.5 log loss | O/U 3.5 Brier | BTTS Brier | BTTS log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 0.1780 | 0.2492 | 0.6916 | 0.2086 | 0.2485 | 0.6902 |
| elo | 0.1801 | 0.2499 | 0.6933 | 0.2093 | 0.2514 | 0.6964 |
| logistic_regression | 0.1758 | 0.2447 | 0.6824 | 0.2054 | 0.2473 | 0.6876 |
| **poisson** (selected) | 0.1754 | 0.2434 | 0.6796 | 0.2041 | 0.2468 | 0.6866 |
| random_forest | 0.1759 | 0.2443 | 0.6815 | 0.2052 | 0.2467 | 0.6866 |
| xgboost | 0.1760 | 0.2448 | 0.6825 | 0.2053 | 0.2470 | 0.6871 |
| lightgbm | 0.1761 | 0.2450 | 0.6830 | 0.2056 | 0.2474 | 0.6879 |

## Test log loss compared with the selected model (poisson)

Mean per-match difference (negative = better than the selected model), with a 95% paired bootstrap interval over test matches (5000 resamples). An interval that contains 0 means the test set cannot distinguish the two models.

| Model | Difference | 95% interval | Resamples where better |
| --- | ---: | --- | ---: |
| random_forest | -0.0024 | [-0.0062, +0.0013] | 90% |
| xgboost | -0.0017 | [-0.0053, +0.0019] | 82% |
| lightgbm | -0.0012 | [-0.0051, +0.0026] | 72% |
| elo | -0.0002 | [-0.0047, +0.0045] | 53% |
| logistic_regression | +0.0080 | [+0.0038, +0.0124] | 0% |
| historical_baseline | +0.0615 | [+0.0502, +0.0725] | 0% |

## Most important features (tree models, 1X2 classifier of the final model)

| Rank | random_forest | xgboost | lightgbm |
| ---: | --- | --- | --- |
| 1 | elo_diff (0.283) | elo_home_win_expectancy (0.080) | elo_diff (0.123) |
| 2 | elo_home_win_expectancy (0.237) | elo_diff (0.065) | elo_home_win_expectancy (0.069) |
| 3 | rating_expected_home_goals (0.107) | rating_expected_home_goals (0.046) | rating_expected_home_goals (0.061) |
| 4 | home_elo (0.074) | home_elo (0.043) | rating_expected_away_goals (0.053) |
| 5 | rating_expected_away_goals (0.053) | away_elo (0.029) | home_elo (0.047) |
| 6 | away_elo (0.032) | away_form5_win_rate (0.026) | rest_days_diff (0.046) |
| 7 | away_attack_rating (0.027) | rating_expected_away_goals (0.024) | home_rest_days (0.040) |
| 8 | home_attack_rating (0.023) | away_form10_win_rate (0.020) | away_attack_rating (0.038) |
| 9 | home_attack_strength (0.014) | away_attack_rating (0.020) | away_venue_matches (0.037) |
| 10 | home_defence_rating (0.009) | home_attack_rating (0.018) | away_elo (0.031) |
