# Prediction models

Four baseline models (Sprint 5) and three tree-ensemble models (Sprint 6), all
evaluated chronologically and tracked in MLflow. Ensembles of these models (Sprint 8)
are described in [ensembles.md](ensembles.md). Code: `backend/app/ml/`. The full
metric tables from real data are in [model-comparison.md](model-comparison.md).

## What every model predicts

Every model outputs a full **exact-score distribution** (0 to 10 goals per side,
normalised). Everything else is derived from it, so a model's outputs can never
contradict each other. The goal prediction engine ([prediction-engine.md](prediction-engine.md))
turns it into the full prediction object, including over/under and both teams to
score:

- home / draw / away probabilities: sums over the score grid;
- expected home and away goals: the grid's marginal means;
- exact-score probabilities and top-k scores: the grid itself.

## The models

| Model | 1X2 from | Score grid | Tuned |
| --- | --- | --- | --- |
| `historical_baseline` | the grid | Empirical score frequencies from training, smoothed with an independent-Poisson prior; the same for every match | none |
| `elo` | Ordered logit on `elo_diff` (Sprint 4 Elo, home advantage included): P(away) = σ(c₁ − s·x), P(home) = 1 − σ(c₂ − s·x) | Poisson from the Elo system's attack/defence ratings, rescaled to the ordered-logit 1X2 | none (3 fitted parameters) |
| `logistic_regression` | Multinomial logistic regression on the feature vector (median imputation + missing indicators, standardised) | Poisson GLMs for home and away goals, rescaled to the logistic 1X2 | `C` |
| `poisson` | the grid | Two Poisson GLMs (home and away goals) with a fitted Dixon–Coles low-score correction ρ | `alpha` |
| `random_forest` | scikit-learn random forest classifier | Random forests with the Poisson criterion for home and away goals, rescaled to the classifier's 1X2 | `max_depth`, `min_samples_leaf` |
| `xgboost` | XGBoost `multi:softprob` classifier | XGBoost `count:poisson` regressors, rescaled | `max_depth`, `n_estimators` |
| `lightgbm` | LightGBM multiclass classifier | LightGBM Poisson regressors, rescaled | `num_leaves`, `n_estimators` |

*Rescaled* means the Poisson grid's home-win, draw and away-win regions are scaled
to the classifier's probabilities, keeping the relative likelihood of scores within
each region.

Tree models (`app/ml/trees.py`):

- The classifier and both goal regressors share one hyperparameter set; the fixed
  values (learning rate 0.02, row/column subsampling, L2 penalty, 500 forest trees)
  are in each class's `DEFAULTS` and are tracked with every run.
- Missing values are handled natively by all three libraries; no imputation.
- Seeds are fixed, and random-forest predictions run single-threaded. Parallel
  prediction changed the last bit of probabilities between calls, so without this
  a saved model would not reproduce its own predictions exactly. Refits and
  reloaded models now reproduce predictions bit for bit (tested).
- `feature_importances()` gives the 1X2 classifier's normalised importances, logged
  to MLflow as `feature_importances.json`.

**Features.** All models read the same stored Sprint 4 feature vector
(`app/ml/feature_sets.py`, `CORE_FEATURES`). No model engineers its own features.
Excluded from model inputs:

- `competition_id` (categorical);
- bookmaker odds (never mandatory; a separate `logistic_regression_with_odds`
  candidate uses them for comparison);
- counts that grow without bound over time (`home/away_rated_matches`,
  `competition_matches`). In testing, those made later seasons fall outside the
  training range and pushed goal predictions up by about 0.6 per match.

Features that never have a value in the training data (e.g. xG when the source has
none) are dropped automatically and recorded.

## Common model interface

Every model, baseline or tree, is a `Predictor` (`app/ml/predictors.py`):

| Method | Returns |
| --- | --- |
| `train(data)` (alias `fit`) | the model, fitted |
| `predict(data)` | one `MatchPrediction` (full score grid) per match |
| `predict_proba(data)` | array (matches × 3): home, draw, away |
| `save(directory, metadata)` | writes `model.joblib` + `metadata.json`; **refuses a directory that already holds a model** |
| `Model.load(directory, feature_set_version=…)` | the model; refuses another class or another feature-set version |

## Chronological evaluation

- **No shuffling, ever.** `split_by_dates` and `split_by_seasons` assign matches by
  kickoff. `ChronologicalSplit` refuses any split where training overlaps or
  follows validation, or validation overlaps or follows test.
- **Warm-up.** Training rows where either team has fewer than 3 rated matches are
  dropped (ratings are still at defaults). Validation, held-out folds and test keep
  every match.
- **Time-aware cross-validation** (`--cv-folds n`, Sprint 6). Tuning and selection
  use the log loss pooled over n rolling-origin folds: the validation period plus
  the last n−1 training seasons. Each fold is predicted by a model fitted only on
  matches before it (expanding window). Folds lie entirely inside train +
  validation. With `--cv-folds 1` (the default) tuning uses the validation period
  alone, as in Sprint 5.
- **Selection never sees the test set.** `tune_and_select` receives only training
  and validation data (a test checks its signature). `evaluate_on_test` then refits
  each tuned model on train + validation and evaluates it on the test period once,
  for reporting. Tests invert every test-set result, with and without
  cross-validation, and check that the selection, the chosen hyperparameters and
  every validation and CV figure are unchanged. A mutation check confirmed that
  test fails if the folds are built from data that includes the test period.

## Metrics (`app/ml/metrics.py`)

| Group | Metrics |
| --- | --- |
| 1X2 | accuracy, log loss (selection metric), Brier score (multi-class, 0–2), ranked probability score |
| Goals | MAE and RMSE of expected vs actual goals: home, away, total |
| Exact score | accuracy of the most likely score, top-3 and top-5 coverage, mean log probability of the actual score |
| Derived markets (Sprint 7) | Brier score and log loss for over 1.5 / 2.5 / 3.5 goals and both teams to score, from the same score grid ([prediction-engine.md](prediction-engine.md)) |

The comparison report adds, for each model, the **paired bootstrap 95% interval** of
its per-match test log-loss difference from the selected model (5,000 resamples of
test matches). An interval containing 0 means the test set cannot tell the two
models apart.

## Experiment tracking (MLflow)

Every tuning trial and every final model is an MLflow run:

| Stored | Where |
| --- | --- |
| Model type | tag and param `model_type` (e.g. `XGBoostModel`); tag `candidate` |
| Hyperparameters | params: tuned values and every fixed default |
| Feature version | param `feature_set_version`; artifact `features.json` (features actually used) |
| Training period | tuning runs: `train_start`, `train_end`, `train_matches`; final runs: `fit_start`, `fit_end`, `fit_matches`, plus `dataset.json` with all periods |
| Validation metrics | `validation_*`; with cross-validation also `cv_*` (pooled) and param `cv_folds` |
| Test metrics | `test_*` (final runs only) |
| Model | artifact `model/` (`model.joblib` + `metadata.json` with library versions) |
| Tree models | artifact `feature_importances.json` |
| Selection | tags `stage` (tuning/final), `selected`, `selection_metric` |

The store defaults to `backend/mlruns/mlflow.db` with artifacts in
`backend/mlruns/artifacts/` (git-ignored). Set `MLFLOW_TRACKING_URI` or pass
`--tracking-uri` to use a server.

## Model versions: nothing is overwritten

Full version management (active model, history, rollback, loading) is in
[model-registry.md](model-registry.md).

- Each final model is stored in its own MLflow run (a new run ID every time).
- `save()` refuses to write into a directory that already holds a model.
- `--register` inserts a **new** `model_versions` row (version
  `YYYYmmddHHMMSS-<run id>`, unique per name) that is **inactive**. It never
  updates, replaces or deactivates an existing row. An integration test registers
  twice next to active "production" rows and checks they are untouched. The
  database also allows at most one active version per model name. Activation stays a
  deliberate, separate step: `python -m app.ml registry activate <name> <version>`
  ([model-registry.md](model-registry.md)).

```bash
cd backend
pip install -e ".[dev,ml]"          # ML dependencies are an optional extra
python -m app.features generate     # snapshots at kickoff (Sprint 4)
python -m app.ml train              # season split: last season = test
python -m app.ml train --validation-start 2024-07-01 --test-start 2025-07-01 \
    --with-odds --cv-folds 4 --report ../docs/model-comparison.md --register
python -m app.ml train --models elo,xgboost      # a subset of candidates
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db   # browse runs
```

Saved models are joblib pickles: only load files this project produced.

## Results on real data

No API key is configured, so the models were verified on eight real Premier League
seasons (2018/19 to 2025/26, 3,040 matches, from football-data.co.uk). The data was
loaded into a throwaway database through the normal ingestion path; it is not an
integrated provider. Split: **train 2018/19–2023/24** (2,280 matches), **validation
2024/25** (380), **test 2025/26** (380). Tuning used 4 rolling-origin folds
(2021/22, 2022/23, 2023/24, 2024/25: 1,520 held-out matches). Full tables:
[model-comparison.md](model-comparison.md).

| Model | CV log loss (selection) | Val log loss | Test log loss | Test − Elo, 95% interval | Test RPS | Test goals MAE | Test top-5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **elo** (selected) | **0.9656** | 0.9889 | 1.0295 | – | 0.2092 | 1.307 | 46.6% |
| random_forest | 0.9738 | 0.9962 | 1.0312 | +0.0017 [−0.008, +0.011] | 0.2097 | 1.267 | 47.1% |
| poisson | 0.9747 | 0.9928 | 1.0332 | +0.0037 [−0.014, +0.023] | 0.2101 | 1.244 | 47.9% |
| xgboost | 0.9785 | 1.0003 | 1.0366 | +0.0070 [−0.004, +0.018] | 0.2100 | 1.265 | 46.3% |
| lightgbm | 0.9786 | 1.0018 | 1.0404 | +0.0109 [−0.001, +0.023] | 0.2105 | 1.266 | 47.4% |
| logistic regression + odds | 0.9801 | 0.9929 | 1.0412 | +0.0117 [−0.020, +0.051] | 0.2077 | 1.236 | 46.6% |
| logistic regression | 0.9832 | 1.0004 | 1.0444 | +0.0148 [−0.007, +0.041] | 0.2113 | 1.236 | 47.4% |
| historical baseline | 1.0647 | 1.0806 | 1.0862 | +0.0567 [+0.019, +0.094] | 0.2280 | 1.226 | 42.6% |
| *bookmakers, closing (reference only)* | | *0.966* | *1.012* | | *0.205* | | |

What the numbers support:

- **Elo was selected, and the tree models did not beat it.** Elo had the lowest
  pooled CV log loss (0.9656), ahead of random forest (0.9738), Poisson (0.9747),
  XGBoost (0.9785) and LightGBM (0.9786). Selection used no test data.
- **On test, no feature-based model is distinguishable from Elo.** Every 95%
  interval for the difference contains 0. Only the historical baseline is clearly
  worse. The test season has 380 matches; differences of ±0.01 log loss are within
  noise at that size.
- **Tree models vs the other baselines:** random forest's CV log loss is 0.0009
  better than Poisson's and 0.009 better than logistic regression's. XGBoost and
  LightGBM fall between Poisson and logistic regression. These gaps are small, and
  none has been shown to be significant.
- **Tuning picked the most constrained tree configurations.** Gradient boosting
  chose depth-2 trees or 3 leaves; the forest chose 60 samples per leaf. The
  feature importances are dominated by the Elo and goal-rating features. With about
  2,000 training matches, the extra flexibility of trees finds little signal that
  the Elo rating does not already contain.
- **Goals:** the tree models' total-goals MAE on test (1.265–1.267) is better than
  Elo's (1.307) but worse than the Poisson and logistic GLMs (1.236–1.244). None
  beats the constant baseline (1.226) on test.
- **Selection is robust to the tuning protocol.** Validation-only tuning
  (`--cv-folds 1`) also selects Elo (validation log loss 0.9889). The order below
  it changes (Poisson 0.9921, then logistic + odds 0.9929, then random forest
  0.9958), which shows how little separates those models on one season. With CV,
  two tuned hyperparameters differ: Poisson α 3 instead of 1, forest depth 4
  instead of 8.
- **Market gap:** the best models are about 0.02 log loss behind closing odds on
  both validation (0.989 vs 0.966) and test (1.030 vs 1.012). Pinnacle closing odds
  exist for only 210 of the 380 test matches, so the test reference uses the
  market average (380 matches; Bet365: 1.013).

- **Goal markets (Sprint 7).** Elo, selected on 1X2, is the weakest model for
  over/under and BTTS (over 2.5 Brier 0.264 vs 0.247–0.252 for the others). No model
  clearly beats a constant forecast on goal totals. Details:
  [prediction-engine.md](prediction-engine.md#how-good-are-the-derived-markets).

Sprint 5's results (validation-only tuning) are reproduced exactly by the
`--cv-folds 1` run: the baseline code did not change.

## Known limitations

- Verified on one league. The live API is still unverified, so there is no xG, and
  absences did not contribute.
- Hyperparameter grids are small (6 points per tree model); no randomised or
  Bayesian search, and no probability calibration step.
- Tree models share hyperparameters between the 1X2 classifier and the goal
  regressors.
- The Elo and goal-rating parameters from Sprint 4 are fixed, not tuned.
- The Poisson grids assume independent home and away goals, apart from the
  Dixon–Coles correction (Poisson model only).
- The bootstrap resamples matches independently; matches on the same weekend are
  not fully independent, so the intervals are, if anything, slightly narrow.
- The ML dependencies (now including XGBoost and LightGBM) are an optional extra
  and are not installed in the backend Docker image. Training runs locally for
  now; a training worker image is future work.
