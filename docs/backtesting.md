# Backtesting and calibration

Sprint 9: a walk-forward backtest that reproduces real prediction conditions, and
probability calibration applied only within that temporal protocol. Code:
`backend/app/ml/backtest.py`, `backtest_store.py`, `backtest_report.py`,
`calibration.py`. Full generated tables: [backtest-report.md](backtest-report.md).

## How a backtest runs

For each step time t (weekly by default), for every model:

1. **Only previous information.** A fresh model is fitted on the matches whose
   results were known at t: kickoff + 3 h (the feature system's result delay) ≤ t.
   With a rolling window, only matches with kickoff ≥ t − window are used.
2. **Predict** every match kicking off in [t, t + 1 week), from the feature
   snapshots taken at kickoff (Sprint 4). The snapshots themselves only contain
   information from before kickoff.
3. **Store** the predictions immediately: `predictions` (with `predicted_at` = t),
   the full 11 × 11 exact-score grid, and over/under lines 0.5 to 3.5. Each model
   and calibration variant is its own inactive model version,
   `backtest_<model>_<variant>`.
4. **Reveal** a result only at kickoff + 3 h. From then on it can be used for
   training at later steps, for calibration, and for its `prediction_results` row
   (`evaluated_at` = kickoff + 3 h, actual score read from `matches`).
5. **Metrics** are computed from the stored predictions and revealed results.
   `metrics_from_store` recomputes them from the database rows alone.
6. Move forward one step.

**Guards.** Every step checks that the newest training result was known at t, that
no predicted match has kicked off, and that rolling-window training stays inside the
window. A violation raises `BacktestLeakError`.

```bash
python -m app.ml backtest --start 2021-07-01 --model elo --model poisson:alpha=3 \
    --window expanding --report ../docs/backtest-report.md
python -m app.ml backtest --start 2021-07-01 --window rolling --window-days 1095 --no-store
```

Options:
- `--retrain-every-days` (default 7);
- `--calibration none,platt,isotonic`;
- `--min-calibration-matches` (default 300);
- `--min-training-matches` (default 300; smaller steps are skipped and counted);
- `--model name:param=value,...` for any model, with fixed hyperparameters.

## Calibration within temporal validation

- **Measurement.**
  - Reliability tables: bins of predicted probability against observed frequency.
  - Expected calibration error (ECE): the match-weighted gap over 10 bins. For 1X2
    it is the mean of the home, draw and away ECE.
  - `evaluate()` now reports ECE for 1X2, over/under and BTTS, plus yes/no
    accuracy for those markets.
- **Platt scaling.** One logistic regression per outcome on the logit of its
  probability; the three results are renormalised.
- **Isotonic regression.** One monotone fit per outcome, renormalised, with a floor
  so no outcome gets probability 0.
- **Used correctly in time.** At each step the calibrator is fitted only on the
  same model's *earlier backtest predictions* whose results were known at t. Those
  predictions were genuinely out of sample when made, and never include the matches
  being predicted.
- **Fallback.** Until at least 300 such predictions exist, and all three outcomes
  have occurred among them, the calibrated variant keeps the raw probabilities. It
  records `calibrated_on = 0`.
- **Consistency.** The calibrated 1X2 is applied to the score grid with the Sprint 7
  rescaling, so every market stays consistent.

## Results: Premier League, 2021/22 to 2025/26

1,900 matches, 179 weekly steps, no match skipped. Four models, each raw, with Platt
and with isotonic.

**Look-ahead caveat.** The hyperparameters (Poisson α = 3, logistic C = 0.003,
LightGBM 3 leaves / 250 trees) were chosen in Sprint 6 on data through 2024/25. For
2021/22 to 2024/25 that is a mild look-ahead in hyperparameter choice; 2025/26 is
fully clean. Elo has no tuned hyperparameters, so its whole backtest is clean.

### Overall (expanding window)

| Model | Log loss | Brier | ECE | Accuracy | Goals MAE | Exact | Top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **elo** | **0.9782** | 0.5826 | 0.0234 | 53.8% | 1.350 | 11.1% | 46.1% |
| elo + Platt | 0.9780 | 0.5824 | **0.0143** | 53.4% | 1.349 | 10.9% | 46.6% |
| poisson (α = 3) | 0.9850 | 0.5867 | 0.0200 | 53.4% | 1.322 | 11.5% | 45.9% |
| lightgbm | 0.9889 | 0.5878 | 0.0247 | 53.4% | 1.326 | 11.6% | 45.6% |
| logistic regression | 0.9919 | 0.5905 | 0.0295 | 52.6% | 1.319 | 10.8% | 46.0% |
| *bookmakers, closing odds* | *0.956* (mean of seasons) | | | | | | |

### Calibration (1,591 matches with an active calibrator)

| Model | Platt: log loss change, 95% interval | ECE raw → Platt | Isotonic: log loss change, 95% interval |
| --- | --- | --- | --- |
| elo | −0.0003 [−0.0037, +0.0033] | 0.025 → 0.014 | +0.0302 [+0.0100, +0.0535] |
| poisson | +0.0001 [−0.0028, +0.0032] | 0.024 → 0.015 | +0.0180 [+0.0020, +0.0376] |
| logistic regression | −0.0015 [−0.0052, +0.0021] | 0.034 → 0.022 | +0.0111 [−0.0025, +0.0277] |
| lightgbm | −0.0035 [−0.0079, +0.0008] | 0.027 → 0.018 | +0.0350 [+0.0106, +0.0628] |

- **Platt scaling improves calibration by 33–44% (ECE)** for every model, at no
  measurable cost or gain in log loss. The models' probabilities were slightly
  miscalibrated but their ranking was right. Platt fixes the first without touching
  the second.
- **Isotonic regression makes log loss worse** for every model; for three of the
  four, the interval excludes 0. With about 300 to 1,900 matches it overfits: its
  step function turns noise in the history into jumps in the probabilities. It is
  not recommended at this data size.
- The rolling 3-season window gives the same picture (Platt ECE 0.013–0.018,
  isotonic log loss +0.027 to +0.035).

### By season (Elo, expanding window)

| Season | Log loss | Accuracy | ECE | Closing odds log loss |
| --- | --- | --- | --- | --- |
| 2021/22 | 0.9573 | 55.5% | 0.037 | 0.9367 |
| 2022/23 | 0.9839 | 54.5% | 0.045 | 0.9620 |
| 2023/24 | 0.9319 | 57.1% | 0.032 | 0.9007 |
| 2024/25 | 0.9888 | 52.9% | 0.035 | 0.9667 |
| 2025/26 | 1.0292 | 48.9% | 0.052 | 1.0118 |

The gap to the market is 0.017–0.031 in every season. Seasons differ far more from
each other (0.93 to 1.03) than models do within a season. The Elo figure for
2025/26 (1.0292, retrained weekly) matches the Sprint 6 test figure (1.0295,
trained once), as it should.

### Expanding vs rolling window

The rolling 3-season window is slightly worse for the models that benefit from more
data, and about the same for Elo:

| Model | Expanding | Rolling (3 seasons) |
| --- | --- | --- |
| elo | 0.9782 | 0.9786 |
| poisson | 0.9850 | 0.9859 |
| logistic regression | 0.9919 | 0.9968 |
| lightgbm | 0.9889 | 0.9892 |

There is no evidence that forgetting old seasons helps.

### By prediction type (Elo)

| Prediction | Accuracy | Brier | Log loss | ECE |
| --- | --- | --- | --- | --- |
| 1X2 | 53.8% | 0.5826 | 0.9782 | 0.023 |
| Over/under 1.5 | 79.8% | 0.1650 | 0.5124 | 0.046 |
| Over/under 2.5 | 55.2% | 0.2490 | 0.6919 | 0.068 |
| Over/under 3.5 | 64.7% | 0.2272 | 0.6492 | 0.055 |
| Both teams to score | 54.7% | 0.2499 | 0.6936 | 0.054 |
| Exact score | 11.1% (top-3 30.4%, top-5 46.1%) | | | |
| Goals (total) | MAE 1.350, RMSE 1.687 | | | |

The goal markets are clearly less calibrated than 1X2 (ECE 0.046 to 0.068 vs 0.023).
Calibrating 1X2 does not fix them, because the 1X2 calibration only rescales the
outcome regions of the grid. Calibrating goal markets directly is future work. The
report has the same table for every model and variant.

### Storage check on real data

- 22,800 predictions stored (4 models × 3 variants × 1,900 matches), each with 121
  score cells and 4 over/under lines, and 22,800 results.
- **0** predictions were stored after kickoff; **0** results were evaluated before
  kickoff + 3 h.
- For all 12 variants, metrics recomputed from the database rows equal the reported
  ones.

## Tests

`test_backtest.py` (24 tests), `test_calibration.py` (14), 3 PostgreSQL integration
tests:

- **No look-ahead.**
  - Every prediction's newest training result was known at prediction time, and
    every prediction was made before kickoff.
  - **Poisoning every result after a time T leaves every earlier prediction (raw and
    calibrated) bit-for-bit unchanged.**
  - A match that kicked off 1 h before a step is not trained on.
  - The guard rejects each kind of look-ahead.
- **Walking forward.**
  - Each match is predicted once per variant, in order.
  - The expanding window grows; the rolling window stays inside its span.
  - Skipped steps are counted.
- **Calibration in time.**
  - A calibrator uses exactly the earlier predictions revealed by t.
  - This is tested on a schedule where one prediction per step is still unrevealed.
    The first version of the test did not have such a case, and a mutation check
    showed it would miss a leak.
  - Calibrated predictions stay consistent.
  - A history without every outcome keeps the raw probabilities.
  - On simulated overconfident forecasts, Platt and isotonic reduce ECE and log loss
    on *later* data. On a calibrated model they change almost nothing.
- **Mutation checks.** Removing the result delay and letting the calibrator see
  unrevealed results are both caught.
- **Storage** (PostgreSQL).
  - Rows are written in real-world order, with 121 cells and 4 lines per prediction.
  - Stored metrics equal in-memory metrics: probabilities to 1e-4; goals MAE to
    2e-3, because rounded tail cells are weighted by up to 10 goals; ECE to 5e-3,
    because a rounded probability can change bin.
  - The CLI, with its report and options, is tested end to end.

## Limitations

- One league; hyperparameters not re-tuned inside the backtest (see the caveat
  above).
- Weekly retraining means a match can be predicted up to 7 days before kickoff with
  a model that ignores results from the days in between. The features, computed at
  kickoff, do include those results.
- Calibration covers 1X2 only; goal markets are not calibrated.
- Ensembles are not backtested: their weights would need their own walk-forward
  fitting, and Sprint 8 found no measurable gain over Elo.
