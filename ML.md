# Machine learning

How a model is trained, chosen, measured and served — and what the platform
refuses to claim. This is the guide; the reference pages under `docs/` go deeper.

| Topic | Reference |
| --- | --- |
| The models and their comparison | [docs/models.md](docs/models.md), [docs/model-comparison.md](docs/model-comparison.md) |
| Turning a model into markets | [docs/prediction-engine.md](docs/prediction-engine.md) |
| Ensembles | [docs/ensembles.md](docs/ensembles.md) |
| Walk-forward backtesting and calibration | [docs/backtesting.md](docs/backtesting.md), [docs/backtest-report.md](docs/backtest-report.md) |
| Versions, activation, rollback | [docs/model-registry.md](docs/model-registry.md) |
| Per-prediction explanations | [docs/explainability.md](docs/explainability.md) |
| What the tests guarantee | [docs/testing.md](docs/testing.md) |

## The one idea everything rests on

Every model — baseline or gradient boosted — produces **one 11×11 exact-score
distribution** per match. Every market is derived from that single grid:

```
score grid ──┬── 1X2 (sum the halves and the diagonal)
             ├── expected goals (the mean of each margin)
             ├── most probable score (the largest cell)
             ├── over/under any line (sum by total goals)
             └── both teams to score (sum off the zero row and column)
```

So the figures can never contradict each other: over 2.5 plus under 2.5 is 1
because they are two sums of the same numbers, not two separate models.

## Models

| Model | What it is |
| --- | --- |
| `historical_baseline` | The smoothed empirical score distribution. The floor every real model must beat. |
| `elo` | Ordered logit on the Elo difference; goals from the rating system's attack/defence. |
| `poisson` | Two Poisson GLMs with a fitted Dixon-Coles low-score correction. |
| `logistic_regression` | Multinomial logistic for 1X2, Poisson regressions for the grid. |
| `random_forest`, `xgboost`, `lightgbm` | Tree ensembles: a 1X2 classifier plus two goal regressors, same inputs, same outputs. |
| `ensemble` | A fitted combination of the above — only when it measurably wins. |

They share one interface (`train` / `predict` / `predict_proba` / `save` / `load`),
take the stored feature snapshots as input, and do no feature engineering of their
own.

## Training

```bash
cd backend
python -m app.features generate                 # prerequisite: snapshots at kickoff
python -m app.ml train                          # season split: last = test, previous = validation
python -m app.ml train --validation-start 2023-08-01 --test-start 2024-08-01
python -m app.ml train --cv-folds 3 --report reports/model-comparison.md
python -m app.ml train --models elo,xgboost --with-odds --register
python -m app.ml train --cv-folds 4 --ensemble --report report.md
```

Splits are **chronological**, never shuffled: validation follows training, test
follows validation. Hyperparameters are tuned with time-aware cross-validation on
the training period only.

**The test set is never used to choose anything.** A test asserts this by poisoning
the test set and checking the selection does not change.

Every run is tracked in MLflow: model type, hyperparameters, feature version,
training period, validation metrics and test metrics.

## Evaluation

Metrics are those that reward honest probabilities, not just a correct guess:
accuracy, log loss, Brier score, ranked probability score, exact-score accuracy.
Comparisons come with **paired bootstrap intervals**, so "better" means measurably
better rather than better on this particular split.

An ensemble is recommended only when its 95% interval against the best single model
lies below zero. More models are not assumed to be better.

## Backtesting

A walk-forward backtest re-runs history the way it actually happened:

```bash
python -m app.ml backtest --start 2021-07-01 --model elo --model poisson:alpha=3
python -m app.ml backtest --start 2021-07-01 --window rolling --window-days 1095
```

At each step the model sees only matches whose results were knowable then, predicts
the next block, and the prediction is stored **before** the result is revealed.
Calibration (Platt, isotonic) is fitted only on earlier predictions. A guard raises
`BacktestLeakError` if any step would train on something later.

Reports break results down by competition, season and prediction type.

## Serving a model

```bash
python -m app.ml registry list
python -m app.ml registry activate elo 20260922141211-20c9f0ee --reason "better log loss"
python -m app.ml registry rollback elo
```

One active version per model name, with an activation history and rollback. The API
loads the active version's artifact, and every prediction names the version that
produced it and how fresh its data was. A model trained for a different feature-set
version is refused rather than served with the wrong inputs.

Predictions are stored immutably and scored once the match is played
([docs/prediction-history.md](docs/prediction-history.md)).

## Explanations

Per-prediction feature contributions, where they can be computed exactly: linear
Shapley values for the linear models, TreeSHAP for the tree models, and a refusal
with a reason where no method is valid (the historical baseline reads no features;
an ensemble's mixture has no additive decomposition).

An explanation describes **the model's behaviour, not causality** — the wording in
the API and the UI says so, and a test asserts the rendered text never claims a
feature caused a result.

## What the platform will not claim

- No model is called "best" without a quantitative comparison behind it.
- No prediction is produced for a match whose features cannot be computed; the API
  returns 422 with the reason.
- No number is invented to fill a gap — a missing metric is `null`, and the UI
  shows a dash.
- Retraining does not happen after every match: it runs on a configurable cadence
  and only when enough new results exist ([docs/scheduled-jobs.md](docs/scheduled-jobs.md)).
