# Model comparison

Generated 2026-09-23 20:49 UTC by `python -m app.ml train`. Feature set `v1-079714a734`.

## Data and protocol

- Split: seasons: 1 train, 1 validation, 1 test.
- Train: 2022-08-05 – 2023-05-28 (380 matches); validation: 2023-08-11 – 2024-05-19 (380 matches); test: 2024-08-16 – 2025-05-25 (380 matches).
- Tuning and selection metric: `validation_log_loss` (lower is better). Hyperparameters and the model are chosen on the validation period only.
- The test period is used once, after selection, to report the tuned models refitted on train + validation. It influenced no choice.

## Selection (no test data)

| Model | Type | Selection score | Validation log loss | CV log loss | Grid points | Chosen hyperparameters |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **elo** (selected) | EloModel | 0.9485 | 0.9485 | – | 1 | – |
| poisson | PoissonModel | 0.9537 | 0.9537 | – | 4 | alpha=1.0 |
| random_forest | RandomForestModel | 0.9603 | 0.9603 | – | 6 | max_depth=4, min_samples_leaf=10 |
| logistic_regression | LogisticModel | 0.9681 | 0.9681 | – | 5 | c=0.003, goal_alpha=10.0 |
| lightgbm | LightGBMModel | 0.9713 | 0.9713 | – | 6 | num_leaves=3, n_estimators=250 |
| xgboost | XGBoostModel | 0.9719 | 0.9719 | – | 6 | max_depth=2, n_estimators=250 |
| historical_baseline | HistoricalBaseline | 1.0558 | 1.0558 | – | 1 | smoothing=1.0 |

## Validation metrics

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0558 | 0.6383 | 0.2345 | 46.1% | 1.367 | 1.710 | 5.8% | 23.9% | 35.5% | -3.305 |
| **elo** (selected) | 0.9485 | 0.5613 | 0.1967 | 56.6% | 1.358 | 1.675 | 11.8% | 27.4% | 42.9% | -3.104 |
| logistic_regression | 0.9681 | 0.5738 | 0.2008 | 55.5% | 1.379 | 1.716 | 11.1% | 25.0% | 41.1% | -3.138 |
| poisson | 0.9537 | 0.5641 | 0.1976 | 56.6% | 1.487 | 1.857 | 10.0% | 24.2% | 38.2% | -3.235 |
| random_forest | 0.9603 | 0.5698 | 0.1993 | 57.6% | 1.376 | 1.703 | 11.3% | 25.0% | 39.2% | -3.124 |
| xgboost | 0.9719 | 0.5766 | 0.2013 | 55.5% | 1.368 | 1.705 | 10.0% | 23.7% | 39.7% | -3.137 |
| lightgbm | 0.9713 | 0.5766 | 0.2006 | 54.2% | 1.376 | 1.706 | 11.1% | 23.7% | 38.9% | -3.137 |

## Test metrics (reported only)

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0856 | 0.6591 | 0.2369 | 40.8% | 1.283 | 1.621 | 11.8% | 28.2% | 41.3% | -3.084 |
| **elo** (selected) | 1.0010 | 0.5992 | 0.2059 | 52.1% | 1.354 | 1.686 | 12.1% | 28.4% | 45.3% | -3.021 |
| logistic_regression | 1.0566 | 0.6347 | 0.2234 | 50.3% | 1.331 | 1.677 | 12.6% | 29.5% | 41.3% | -3.058 |
| poisson | 1.0804 | 0.6438 | 0.2280 | 48.4% | 1.365 | 1.716 | 11.3% | 27.9% | 42.4% | -3.114 |
| random_forest | 1.0227 | 0.6134 | 0.2130 | 51.8% | 1.318 | 1.641 | 11.8% | 28.2% | 45.3% | -3.012 |
| xgboost | 1.0406 | 0.6256 | 0.2197 | 51.1% | 1.303 | 1.631 | 13.2% | 29.5% | 44.7% | -3.028 |
| lightgbm | 1.0457 | 0.6284 | 0.2205 | 51.3% | 1.313 | 1.631 | 10.8% | 28.9% | 43.9% | -3.029 |

## Derived markets on test (from each model's score grid)

Over/under and both-teams-to-score probabilities are sums over the same score grid as the 1X2 and exact-score figures; lower is better.

| Model | O/U 1.5 Brier | O/U 2.5 Brier | O/U 2.5 log loss | O/U 3.5 Brier | BTTS Brier | BTTS log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 0.1520 | 0.2461 | 0.6854 | 0.2263 | 0.2446 | 0.6822 |
| **elo** (selected) | 0.1549 | 0.2490 | 0.6926 | 0.2390 | 0.2516 | 0.6971 |
| logistic_regression | 0.1536 | 0.2501 | 0.7006 | 0.2308 | 0.2450 | 0.6833 |
| poisson | 0.1543 | 0.2505 | 0.7053 | 0.2356 | 0.2465 | 0.6881 |
| random_forest | 0.1533 | 0.2460 | 0.6855 | 0.2306 | 0.2460 | 0.6852 |
| xgboost | 0.1534 | 0.2431 | 0.6796 | 0.2274 | 0.2458 | 0.6846 |
| lightgbm | 0.1525 | 0.2435 | 0.6804 | 0.2282 | 0.2441 | 0.6811 |

## Test log loss compared with the selected model (elo)

Mean per-match difference (negative = better than the selected model), with a 95% paired bootstrap interval over test matches (5000 resamples). An interval that contains 0 means the test set cannot distinguish the two models.

| Model | Difference | 95% interval | Resamples where better |
| --- | ---: | --- | ---: |
| random_forest | +0.0216 | [-0.0055, +0.0468] | 5% |
| xgboost | +0.0396 | [+0.0079, +0.0703] | 1% |
| lightgbm | +0.0446 | [+0.0134, +0.0740] | 0% |
| logistic_regression | +0.0555 | [+0.0226, +0.0912] | 0% |
| poisson | +0.0794 | [+0.0373, +0.1279] | 0% |
| historical_baseline | +0.0846 | [+0.0352, +0.1315] | 0% |

## Most important features (tree models, 1X2 classifier of the final model)

| Rank | random_forest | xgboost | lightgbm |
| ---: | --- | --- | --- |
| 1 | elo_diff (0.134) | elo_home_win_expectancy (0.047) | away_form3_goals_for (0.070) |
| 2 | elo_home_win_expectancy (0.117) | home_elo (0.040) | home_attack_strength (0.065) |
| 3 | rating_expected_home_goals (0.080) | elo_diff (0.040) | elo_diff (0.058) |
| 4 | home_attack_strength (0.071) | rating_expected_home_goals (0.034) | competition_home_win_rate (0.055) |
| 5 | home_elo (0.061) | home_attack_strength (0.027) | home_rest_days (0.050) |
| 6 | rating_expected_away_goals (0.027) | away_elo (0.018) | away_form10_draw_rate (0.048) |
| 7 | home_attack_rating (0.023) | away_form5_points (0.018) | rating_expected_home_goals (0.045) |
| 8 | away_elo (0.023) | away_form3_goals_for (0.018) | away_defence_rating (0.036) |
| 9 | away_attack_rating (0.022) | rating_expected_away_goals (0.017) | competition_draw_rate (0.033) |
| 10 | competition_home_win_rate (0.020) | away_form10_win_rate (0.017) | home_elo (0.027) |
