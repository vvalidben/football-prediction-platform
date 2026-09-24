# Explainability

What the platform says about *why* a model predicted what it did. Code:
`backend/app/ml/explain.py`, `backend/app/services/explanations.py`,
`frontend/src/components/Explanation.tsx`.

## What an explanation claims, and what it does not

An explanation describes **the model's behaviour**: how this fitted model turned
this match's feature vector into its output, measured against a stated baseline.

- "The rating difference contributed +0.21 to the model's home-win score" is a
  statement about arithmetic inside the model.
- "The rating difference caused the home team to win" is **not** something this
  platform says, or could support. Contributions are not causes: they say nothing
  about what would happen in the world if a feature were different.
- Features move together. Recent form, rating difference and expected goals all
  carry the same underlying information, and a fitted model splits the credit
  between them in a way that depends on how it was fitted, not on football. A
  contribution can therefore land on a feature that a person would not have
  chosen, and the sign of one correlated feature's contribution need not match
  intuition even when the prediction is sound.

Every API response carries this as a `caveat` field, and the UI prints it under
the figures rather than hiding it behind a tooltip. The wording on the page is
"These features contributed to the model's prediction."

## Methods, and where each one is valid

Attribution is only reported where it can be computed exactly for the model in
question. Nothing is approximated silently.

| Model | Method | Baseline | Exactness |
| --- | --- | --- | --- |
| `logistic_regression` | Shapley values of a linear model: coefficient × standardised value | Average training match (the fitted intercept) | Exact: baseline + contributions = the model's score |
| `poisson` | The same, on each side's log goal rate | Average training match | Exact |
| `elo` | One feature, so its contribution is the whole difference from the baseline | Two equally rated teams | Exact by definition |
| `random_forest`, `xgboost`, `lightgbm` | TreeSHAP (the `shap` package) over the 1X2 classifier | The model's average training prediction | Exact for the tree model |
| `historical_baseline` | — | — | Refused: it reads no features |
| `ensemble` | — | — | Refused: see below |

The linear method is SHAP's linear explainer under the independent-features
assumption. Because `StandardScaler` centres the training data exactly, the
contributions are the coefficients times the standardised values and they add up
to the model's own score with no residual — which the tests check, per outcome,
to 1e-9.

**Why ensembles are refused.** An ensemble mixes its members' score grids. The
mixture's probability is not the weighted sum of its members' log-odds, so there
is no additive decomposition of the ensemble's output to report. Rather than
present member contributions as if they were the ensemble's, the endpoint returns
409 with that reason. Explain a member version instead.

**Why the tree path can be unavailable.** TreeSHAP needs the `shap` package,
which is in the `ml` extra, not `serve`. The API image cannot load a tree model at
all (it has no xgboost or lightgbm), so it says so instead of approximating.

## What is explained

For every model with a 1X2 classifier, the contributions are the classifier's, and
the classifier's output *is* the prediction's match-result probabilities — the
score grid is rescaled to them (docs/prediction-engine.md). The goal regressors
that shape the rest of the score distribution are not covered.

`poisson` has no classifier: its two log goal rates are explained instead, and
every market follows from those rates.

An API test asserts that the explained probabilities equal the stored
prediction's probabilities, so the figures on screen explain the number next to
them and not some other number.

## Missing features

A linear pipeline imputes a missing value with the training median and adds a
missingness flag, and that flag is part of what the model used. Both appear:

- the feature itself, with `value: null` — the UI shows "no value for this match"
  rather than a number;
- `<feature>__missing`, labelled "… — not available for this match", with the
  contribution the model attaches to not knowing.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/predictions/{prediction_id}/explanation` | Contributions behind one stored prediction, per outcome. `top` (1–50, default 12) limits the list; the rest is summed into `other_contribution` so the parts still add up. |
| GET | `/models/{model_id}/importance` | What a model version relies on across all matches. `top` (1–100, default 12), with `other_importance` for the tail. |

Both return 409 with a reason when no valid method applies, 503 when the model
artifact cannot be loaded, and 404 for an unknown prediction or model.

Explanations are computed from the **feature snapshot the prediction was stored
with**, so they are reproducible: the same prediction explained tomorrow gives the
same numbers, whatever has happened since. A prediction stored without its
snapshot is refused (409) rather than explained from today's data.

`importance` is global model behaviour, from the fitted model itself: impurity
importance for the tree models (the library's own measure) and the average
absolute standardised coefficient for the linear ones — how far the model's score
moves per standard deviation of a feature. It is not a ranking of what matters in
football, and the UI says so beside the chart.

## In the UI

- **Match prediction page** — a "Why this prediction" card: pick the outcome, and
  the contributions appear as bars either side of a centre line, green to the
  right for what pushed the model's score up and red to the left for what pushed
  it down. A line above says where the model started (the baseline probability)
  and where it arrived, and the method and caveat sit underneath. When the model
  cannot be explained, the card states that plainly — it is not shown as an error.
- **Model performance page** — a "What this model relies on" card with the
  importance ranking, the tail summed, and the same caveat.

## Tests

- `backend/tests/ml/test_explain.py` (39 tests): exact additivity per target and
  per model; the explained output equals the model's own `predict_proba` and goal
  rates; each contribution equals the coefficient times the standardised value for
  the column **sklearn itself names** (the guard against attributing a number to
  the wrong feature — a mutation that shifts the mapping fails three tests while
  the additivity checks still pass); nudging one feature changes only that
  feature's contribution; missing-value flags; refusals for the baseline, an
  ensemble, an unfitted model and an out-of-range index; TreeSHAP additivity for
  all three tree models (skipped when `shap` is absent), and the honest refusal
  when it is absent.
- `backend/tests/api/test_explanations.py` (13 tests): the endpoints end to end
  against PostgreSQL and a real fitted model, including that the explained
  probabilities are the stored prediction's own, that truncation keeps the parts
  adding up, and that an unexplainable model gives 409 with its reason.
- `frontend/src/components/Explanation.test.tsx` (12 tests): the wording, the
  direction of each bar, outcome switching, the "other features" line, the 409
  state shown as a statement rather than an error, and that the rendered text
  never says a feature caused anything.

## Limitations

- Contributions explain the 1X2 classifier (or the goal rates), not the full
  score grid: over/under and both-teams-to-score are derived from the grid and are
  not attributed separately.
- The linear method assumes independent features. With correlated features the
  split of credit between them is one of several defensible splits, not the only
  one.
- TreeSHAP runs on request. For the tree models this is fast at one match, but no
  explanation is precomputed or cached.
- `shap` is not installed in the serving image, and the serving image cannot load
  tree models in any case.
