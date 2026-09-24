# Testing

How the platform is tested, what the tests are allowed to assume, and where each
guarantee lives. Run everything with:

```bash
python scripts/check.py                 # static checks + unit tests
python scripts/check.py --integration   # also PostgreSQL- and Redis-backed tests
python scripts/check.py --audit         # also pip-audit and npm audit
```

## What the suite covers

| Area | Where | What it proves |
| --- | --- | --- |
| API | `backend/tests/api/` | Every endpoint's success and failure paths against a real database and a real fitted model: catalog, predictions, batch, history, explanations, model registry, jobs, data status, rate limits and request ids, and that the OpenAPI document matches the routes that exist |
| Database | `backend/tests/db/` | The migration applies, round-trips and matches the models; every foreign key is indexed and both hot orderings have an index the planner chooses; natural keys, CHECK constraints and prediction storage |
| Ingestion | `backend/tests/ingestion/`, `backend/tests/providers/`, `backend/tests/quality/` | Provider retries, backoff, `Retry-After` and rate limiting; normalisation and validation; first sync, idempotent re-sync, partial and total failure, run logs, cross-provider linking, concurrent syncs; the historical backfill and the data-quality issue log |
| Feature engineering | `backend/tests/features/` | Leakage proofs (below), hand-calculated feature values, snapshot immutability, drift, reproducibility and versioning |
| Models | `backend/tests/ml/` | Score-grid maths, hand-computed metrics, chronological splits, training and prediction for every model on synthetic data with a known signal, selection that cannot see the test set, persistence round trips, ensembles, calibration, explanations |
| Prediction engine | `backend/tests/ml/test_score_engine.py`, `backend/tests/api/test_predictions.py` | Every market derives from one score grid and none can contradict another; the API returns what the engine produced |
| Background jobs | `backend/tests/tasks/` | Each scheduled job against a real database and a mock provider; retries, backoff, permanent failures, job status rows, the retraining policy, and the operator CLI (`run`, `status`, `schedule`) |
| Frontend | `frontend/src/**/*.test.ts(x)` | Components, hooks, routing and every page: what is rendered from a full response, the loading state, the empty state, and a backend failure |

Counts as of Sprint 17: **887 backend tests** and **145 frontend tests**; backend
line coverage is 97%.

## The eight guarantees

`backend/tests/ml/test_critical_guarantees.py` states them in one place, and
`backend/tests/api/test_prediction_guarantees.py` repeats the two that can only be
proven end to end — through the database, the feature builder and a stored model.

1. **No future information enters features.** `tests/features/test_no_leakage.py`
   proves it for future matches, in-progress matches, future and post-match
   statistics, the target's own result, odds and absences after the cutoff, plus a
   whole-dataset canary and batch-vs-single agreement. These are mutation-checked:
   removing the protection makes them fail.
2. **No post-match information enters predictions.** Predict a scheduled match,
   then record a 7–0 result and force a fresh prediction: the numbers must not
   move by more than 1e-9. The cutoff for a played match is its kickoff, and the
   newest result behind a prediction is always older than its cutoff.
3. **Probability distributions sum to 1.** The 11×11 grid, the 1X2 outcomes,
   both-teams-to-score, and every half-goal over/under line, for every model.
   `predict_proba` must agree with the grid it came from.
4. **Predictions are reproducible.** The same model gives identical grids twice;
   a model saved and loaded again predicts identically; refitting on the same data
   gives the same model. (Random forests predict single-threaded for this reason.)
5. **Missing data is handled.** A match with every feature missing still gets a
   valid distribution; blanking one feature moves the prediction only as far as
   that feature's weight allows; a feature never observed in training is dropped
   rather than imputed into nonsense.
6. **Unknown teams are handled.** Two clubs with no history at all are predicted
   through the whole stack: both start from the default rating, so `elo_diff` is
   exactly the home-advantage constant, and two different unknown fixtures get the
   same answer — nothing may leak in to tell them apart. An unknown team *name* is
   a 404, not a guess.
7. **Model loading works.** An artifact round-trips through disk; one built for a
   different feature set is refused rather than used with the wrong inputs; an
   unfitted model refuses to predict; the registry loads the active version.
8. **Historical backtesting is chronological.** A dataset out of order is refused
   outright. `tests/ml/test_backtest.py` proves the rest in depth: expanding and
   rolling windows, the result delay, the leak guard, and that changing a future
   result changes no earlier prediction.

## Conventions

- Tests that need PostgreSQL or Redis are marked `@pytest.mark.integration` and
  run only with `RUN_INTEGRATION_TESTS=1`. They use a throwaway `<db>_test`
  database, one rolled-back transaction per test.
- Model tests use `tests/ml/synthetic.py`: teams with hidden strengths, Poisson
  goals, and features that expose the strength gap noisily. A model that uses the
  features must beat the historical baseline on it.
- Tests assert behaviour a person would care about, and say so in the test name.
  Where an assertion encodes an intuition rather than a guarantee, it is checked
  against the model's own coefficients instead — correlated features mean the
  "obvious" sign is not always the right one (see docs/explainability.md).
- Frontend tests mock `fetch` by path prefix (`frontend/src/test/render.tsx`) and
  404 anything a test did not declare, so an accidental call is visible.

## Known gaps

- No end-to-end browser test: the stack smoke test (`tests/smoke/check_stack.py`)
  checks that the SPA is served and the proxy reaches the API, but no test drives
  a real browser through a prediction.
- The API-Football adapter is tested against recorded-style fixtures and a
  contract suite, not the live provider; the coverage probe needs a real key.
- `app/providers/api_football/coverage.py` and the ingestion CLI have the lowest
  coverage: both are operator tools whose remaining branches need a live provider.
  The jobs CLI was at 0% before this sprint and is now covered.
- Load and soak testing has not been done; rate limits and resource limits are
  starting points (docs/operations.md).
