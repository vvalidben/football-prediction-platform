# Model comparison

Generated 2026-09-22 14:11 UTC by `python -m app.ml train`. Feature set `v1-079714a734`.

> **Context (added by hand; the rest of this file is generated).** Data: eight real
> Premier League seasons (2018/19 to 2025/26, from football-data.co.uk), loaded into a
> throwaway database through the normal ingestion path, because no API-Football key
> is configured. Command: `python -m app.ml train --validation-start 2024-07-01
> --test-start 2025-07-01 --with-odds --cv-folds 4 --ensemble --report ...`. Bookmaker reference,
> closing odds with the margin removed (not a candidate): 1X2 log loss validation 0.966,
> test 1.012; over 2.5 Brier on test 0.2452 (market average, 380 matches). Interpretation: [models.md](models.md#results-on-real-data) and
> [ensembles.md](ensembles.md).

## Data and protocol

- Split: dates: validation from 2024-07-01, test from 2025-07-01.
- Train: 2018-08-10 – 2024-05-19 (2280 matches); validation: 2024-08-16 – 2025-05-25 (380 matches); test: 2025-08-15 – 2026-05-24 (380 matches).
- Tuning and selection metric: `cv_log_loss` (lower is better). Time-aware cross-validation over 4 rolling-origin folds; each fold is predicted by a model fitted only on earlier matches:
  1. fit 2018-09-01 – 2021-05-23 (1095 matches) → predict 2021-08-13 – 2022-05-22 (380 matches)
  2. fit 2018-09-01 – 2022-05-22 (1472 matches) → predict 2022-08-05 – 2023-05-28 (380 matches)
  3. fit 2018-09-01 – 2023-05-28 (1849 matches) → predict 2023-08-11 – 2024-05-19 (380 matches)
  4. fit 2018-09-01 – 2024-05-19 (2226 matches) → predict 2024-08-16 – 2025-05-25 (380 matches)
- The test period is used once, after selection, to report the tuned models refitted on train + validation. It influenced no choice.

## Selection (no test data)

| Model | Type | Selection score | Validation log loss | CV log loss | Grid points | Chosen hyperparameters |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **elo** (selected) | EloModel | 0.9656 | 0.9889 | 0.9656 | 1 | – |
| random_forest | RandomForestModel | 0.9738 | 0.9962 | 0.9738 | 6 | max_depth=4, min_samples_leaf=60 |
| poisson | PoissonModel | 0.9747 | 0.9928 | 0.9747 | 4 | alpha=3.0 |
| xgboost | XGBoostModel | 0.9785 | 1.0003 | 0.9785 | 6 | max_depth=2, n_estimators=250 |
| lightgbm | LightGBMModel | 0.9786 | 1.0018 | 0.9786 | 6 | num_leaves=3, n_estimators=250 |
| logistic_regression_with_odds | LogisticModel | 0.9801 | 0.9929 | 0.9801 | 3 | c=0.01, goal_alpha=10.0 |
| logistic_regression | LogisticModel | 0.9832 | 1.0004 | 0.9832 | 5 | c=0.003, goal_alpha=10.0 |
| historical_baseline | HistoricalBaseline | 1.0647 | 1.0806 | 1.0647 | 1 | smoothing=1.0 |

## Validation metrics

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0806 | 0.6552 | 0.2349 | 40.8% | 1.279 | 1.617 | 11.8% | 28.2% | 40.5% | -3.076 |
| **elo** (selected) | 0.9889 | 0.5912 | 0.2027 | 53.2% | 1.348 | 1.678 | 10.8% | 29.2% | 46.1% | -3.005 |
| logistic_regression | 1.0004 | 0.5994 | 0.2064 | 51.1% | 1.278 | 1.620 | 9.7% | 31.1% | 46.1% | -2.979 |
| poisson | 0.9928 | 0.5929 | 0.2044 | 51.3% | 1.285 | 1.627 | 11.8% | 31.1% | 46.6% | -2.975 |
| random_forest | 0.9962 | 0.5944 | 0.2045 | 51.8% | 1.289 | 1.621 | 11.6% | 29.2% | 47.4% | -2.969 |
| xgboost | 1.0003 | 0.5977 | 0.2059 | 51.1% | 1.286 | 1.616 | 12.4% | 29.7% | 44.7% | -2.973 |
| lightgbm | 1.0018 | 0.5984 | 0.2061 | 51.1% | 1.286 | 1.617 | 12.4% | 29.5% | 44.7% | -2.973 |
| logistic_regression_with_odds | 0.9929 | 0.5915 | 0.2024 | 55.0% | 1.278 | 1.618 | 12.6% | 32.4% | 45.5% | -2.970 |

## Cross-validation metrics (pooled over folds)

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0647 | 0.6442 | 0.2334 | 44.5% | 1.356 | 1.691 | 10.7% | 27.3% | 40.6% | -3.148 |
| **elo** (selected) | 0.9656 | 0.5736 | 0.1986 | 55.1% | 1.360 | 1.697 | 11.4% | 30.0% | 45.7% | -3.015 |
| logistic_regression | 0.9832 | 0.5848 | 0.2031 | 53.7% | 1.344 | 1.676 | 10.4% | 29.3% | 44.5% | -3.023 |
| poisson | 0.9747 | 0.5795 | 0.2015 | 54.3% | 1.349 | 1.682 | 11.5% | 30.5% | 44.8% | -3.014 |
| random_forest | 0.9738 | 0.5777 | 0.2004 | 54.5% | 1.346 | 1.672 | 10.9% | 29.4% | 45.9% | -3.007 |
| xgboost | 0.9785 | 0.5806 | 0.2013 | 54.5% | 1.345 | 1.670 | 11.1% | 29.3% | 44.7% | -3.010 |
| lightgbm | 0.9786 | 0.5805 | 0.2010 | 54.6% | 1.346 | 1.672 | 10.4% | 29.5% | 44.9% | -3.013 |
| logistic_regression_with_odds | 0.9801 | 0.5802 | 0.1999 | 54.5% | 1.343 | 1.673 | 10.9% | 29.6% | 44.0% | -3.017 |

## Test metrics (reported only)

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | Goals RMSE | Exact score | Top-3 | Top-5 | log P(score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 1.0862 | 0.6571 | 0.2280 | 42.6% | 1.226 | 1.573 | 12.4% | 27.9% | 42.6% | -2.967 |
| **elo** (selected) | 1.0295 | 0.6194 | 0.2092 | 48.4% | 1.307 | 1.639 | 9.5% | 30.0% | 46.6% | -2.927 |
| logistic_regression | 1.0444 | 0.6247 | 0.2113 | 48.2% | 1.236 | 1.576 | 10.3% | 29.2% | 47.4% | -2.905 |
| poisson | 1.0332 | 0.6198 | 0.2101 | 48.7% | 1.244 | 1.579 | 11.8% | 31.3% | 47.9% | -2.899 |
| random_forest | 1.0312 | 0.6201 | 0.2097 | 46.8% | 1.267 | 1.590 | 12.1% | 29.5% | 47.1% | -2.897 |
| xgboost | 1.0366 | 0.6225 | 0.2100 | 48.2% | 1.265 | 1.591 | 11.8% | 28.2% | 46.3% | -2.902 |
| lightgbm | 1.0404 | 0.6250 | 0.2105 | 47.4% | 1.266 | 1.594 | 11.1% | 27.9% | 47.4% | -2.905 |
| logistic_regression_with_odds | 1.0412 | 0.6191 | 0.2077 | 48.7% | 1.236 | 1.570 | 10.3% | 29.2% | 46.6% | -2.899 |

## Derived markets on test (from each model's score grid)

Over/under and both-teams-to-score probabilities are sums over the same score grid as the 1X2 and exact-score figures; lower is better.

| Model | O/U 1.5 Brier | O/U 2.5 Brier | O/U 2.5 log loss | O/U 3.5 Brier | BTTS Brier | BTTS log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| historical_baseline | 0.1662 | 0.2475 | 0.6882 | 0.2052 | 0.2471 | 0.6874 |
| **elo** (selected) | 0.1740 | 0.2638 | 0.7223 | 0.2164 | 0.2523 | 0.6982 |
| logistic_regression | 0.1662 | 0.2478 | 0.6888 | 0.2070 | 0.2453 | 0.6837 |
| poisson | 0.1670 | 0.2496 | 0.6925 | 0.2072 | 0.2457 | 0.6843 |
| random_forest | 0.1669 | 0.2516 | 0.6964 | 0.2100 | 0.2471 | 0.6873 |
| xgboost | 0.1666 | 0.2516 | 0.6965 | 0.2098 | 0.2475 | 0.6882 |
| lightgbm | 0.1668 | 0.2523 | 0.6978 | 0.2100 | 0.2473 | 0.6877 |
| logistic_regression_with_odds | 0.1658 | 0.2473 | 0.6877 | 0.2062 | 0.2454 | 0.6839 |

## Test log loss compared with the selected model (elo)

Mean per-match difference (negative = better than the selected model), with a 95% paired bootstrap interval over test matches (5000 resamples). An interval that contains 0 means the test set cannot distinguish the two models.

| Model | Difference | 95% interval | Resamples where better |
| --- | ---: | --- | ---: |
| random_forest | +0.0017 | [-0.0079, +0.0110] | 36% |
| poisson | +0.0037 | [-0.0143, +0.0227] | 35% |
| xgboost | +0.0070 | [-0.0042, +0.0183] | 11% |
| lightgbm | +0.0109 | [-0.0012, +0.0230] | 4% |
| logistic_regression_with_odds | +0.0117 | [-0.0197, +0.0510] | 28% |
| logistic_regression | +0.0148 | [-0.0069, +0.0414] | 11% |
| historical_baseline | +0.0567 | [+0.0192, +0.0937] | 0% |

## Most important features (tree models, 1X2 classifier of the final model)

| Rank | random_forest | xgboost | lightgbm |
| ---: | --- | --- | --- |
| 1 | elo_home_win_expectancy (0.256) | elo_home_win_expectancy (0.068) | elo_diff (0.105) |
| 2 | elo_diff (0.239) | elo_diff (0.060) | home_elo (0.067) |
| 3 | rating_expected_home_goals (0.116) | rating_expected_home_goals (0.046) | elo_home_win_expectancy (0.061) |
| 4 | home_elo (0.062) | away_elo (0.032) | rating_expected_home_goals (0.053) |
| 5 | rating_expected_away_goals (0.058) | home_elo (0.030) | h2h_weighted_goal_diff (0.035) |
| 6 | home_attack_rating (0.023) | rating_expected_away_goals (0.028) | rating_expected_away_goals (0.033) |
| 7 | h2h_weighted_goal_diff (0.019) | home_form10_win_rate (0.018) | home_attack_rating (0.033) |
| 8 | h2h_weighted_points (0.019) | home_form10_loss_rate (0.017) | home_rest_days (0.033) |
| 9 | away_defence_rating (0.018) | away_defence_rating (0.016) | h2h_weighted_points (0.032) |
| 10 | away_elo (0.016) | home_form10_goals_for (0.016) | away_defence_rating (0.029) |

## Ensembles

Members: logistic_regression, random_forest, xgboost, lightgbm, poisson, elo. Each ensemble combines the members' score grids, so every derived market stays consistent.

- **simple_average**: the mean of the grids (nothing fitted).
- **weighted_average**: weights >= 0 summing to 1 that minimise the 1X2 log loss of the mixture.
- **stacking**: a multinomial logistic regression on the members' 1X2 log-odds; the average grid is rescaled to its 1X2.

Weights and meta-models are fitted only on **out-of-fold validation predictions**: every match below was predicted by a member fitted only on earlier matches. The test period is not used.

### Forward-chained validation (the basis for every choice)

Each block is predicted with combiners fitted on the earlier blocks only; the first block only trains the combiners. Individual models are scored on the same matches.

1. 2021-08-13 – 2022-05-22 (380 matches) (combiner training only)
2. 2022-08-05 – 2023-05-28 (380 matches)
3. 2023-08-11 – 2024-05-19 (380 matches)
4. 2024-08-16 – 2025-05-25 (380 matches)

Pooled over 1140 matches:

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | log P(score) | O/U 2.5 Brier | BTTS Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **weighted_average (ensemble)** (best) | 0.9684 | 0.5754 | 0.1994 | 55.6% | 1.365 | -3.038 | 0.2414 | 0.2482 |
| elo (individual) | 0.9686 | 0.5756 | 0.1994 | 54.9% | 1.369 | -3.041 | 0.2419 | 0.2486 |
| simple_average (ensemble) | 0.9717 | 0.5771 | 0.1999 | 54.7% | 1.344 | -3.022 | 0.2405 | 0.2460 |
| random_forest (individual) | 0.9737 | 0.5782 | 0.2006 | 54.4% | 1.348 | -3.027 | 0.2410 | 0.2471 |
| poisson (individual) | 0.9748 | 0.5798 | 0.2016 | 54.1% | 1.356 | -3.036 | 0.2441 | 0.2473 |
| xgboost (individual) | 0.9780 | 0.5808 | 0.2012 | 54.8% | 1.349 | -3.028 | 0.2415 | 0.2465 |
| lightgbm (individual) | 0.9783 | 0.5807 | 0.2008 | 55.0% | 1.349 | -3.031 | 0.2414 | 0.2469 |
| logistic_regression (individual) | 0.9835 | 0.5855 | 0.2031 | 53.6% | 1.351 | -3.043 | 0.2434 | 0.2472 |
| stacking (ensemble) | 0.9871 | 0.5869 | 0.2039 | 54.7% | 1.345 | -3.038 | 0.2409 | 0.2458 |

Forward-validation log loss minus the best individual model (elo), with a 95% paired bootstrap interval over the 1140 matches. An interval containing 0 means validation cannot tell the entry apart from that model:

| Model | Difference | 95% interval | Resamples where better |
| --- | ---: | --- | ---: |
| weighted_average | -0.0002 | [-0.0008, +0.0004] | 70% |
| simple_average | +0.0031 | [-0.0034, +0.0094] | 17% |
| random_forest | +0.0051 | [-0.0016, +0.0118] | 6% |
| poisson | +0.0062 | [-0.0028, +0.0152] | 9% |
| xgboost | +0.0093 | [-0.0000, +0.0185] | 3% |
| lightgbm | +0.0097 | [+0.0007, +0.0187] | 2% |
| logistic_regression | +0.0149 | [+0.0033, +0.0263] | 1% |
| stacking | +0.0185 | [+0.0058, +0.0316] | 0% |

**Recommended: elo.** Rule: ensemble only if its 95% forward-validation interval vs the best individual model lies entirely below 0. Reason: weighted_average does not measurably beat elo on forward validation (difference -0.0002, 95% interval [-0.0008, +0.0004]); the single model is simpler.

### Weights

Fitted on all validation blocks (used by the final ensembles):

| Member | simple_average | weighted_average | stacking |
| --- | ---: | ---: | ---: |
| logistic_regression | 0.167 | 0.000 | (meta-model) |
| random_forest | 0.167 | 0.000 | (meta-model) |
| xgboost | 0.167 | 0.000 | (meta-model) |
| lightgbm | 0.167 | 0.000 | (meta-model) |
| poisson | 0.167 | 0.139 | (meta-model) |
| elo | 0.167 | 0.861 | (meta-model) |

Weighted-average weights as fitted at each forward step (stability check):

| Fitted on blocks | logistic_regression | random_forest | xgboost | lightgbm | poisson | elo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 1-2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.090 | 0.910 |
| 1-3 | 0.003 | 0.000 | 0.000 | 0.011 | 0.068 | 0.918 |

### Test (reported only)

| Model | Log loss | Brier | RPS | Accuracy | Goals MAE | log P(score) | O/U 2.5 Brier | BTTS Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| elo (individual) | 1.0295 | 0.6194 | 0.2092 | 48.4% | 1.307 | -2.927 | 0.2638 | 0.2523 |
| logistic_regression (individual) | 1.0444 | 0.6247 | 0.2113 | 48.2% | 1.236 | -2.905 | 0.2478 | 0.2453 |
| poisson (individual) | 1.0332 | 0.6198 | 0.2101 | 48.7% | 1.244 | -2.899 | 0.2496 | 0.2457 |
| random_forest (individual) | 1.0312 | 0.6201 | 0.2097 | 46.8% | 1.267 | -2.897 | 0.2516 | 0.2471 |
| xgboost (individual) | 1.0366 | 0.6225 | 0.2100 | 48.2% | 1.265 | -2.902 | 0.2516 | 0.2475 |
| lightgbm (individual) | 1.0404 | 0.6250 | 0.2105 | 47.4% | 1.266 | -2.905 | 0.2523 | 0.2473 |
| simple_average (ensemble) | 1.0311 | 0.6197 | 0.2092 | 47.6% | 1.259 | -2.895 | 0.2516 | 0.2465 |
| weighted_average (ensemble) | 1.0282 | 0.6184 | 0.2088 | 48.2% | 1.296 | -2.917 | 0.2610 | 0.2506 |
| stacking (ensemble) | 1.0267 | 0.6183 | 0.2088 | 50.0% | 1.258 | -2.891 | 0.2514 | 0.2472 |

Test log loss of each ensemble minus elo (95% paired bootstrap interval):

| Ensemble | Difference | 95% interval | Resamples where better |
| --- | ---: | --- | ---: |
| simple_average | +0.0016 | [-0.0075, +0.0105] | 36% |
| weighted_average | -0.0013 | [-0.0038, +0.0010] | 86% |
| stacking | -0.0028 | [-0.0123, +0.0067] | 71% |
