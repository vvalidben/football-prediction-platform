# Prediction history

Every prediction the platform has served, with the result it was scored against.
Code: `backend/app/services/history.py`, `backend/app/services/evaluation.py`,
`frontend/src/pages/History.tsx`.

## The lifecycle of one prediction

1. **Stored when served.** A prediction row is written with its 1X2
   probabilities, expected goals, most probable score, the full score grid, the
   over/under lines, the model version and the feature snapshot it was computed
   from. Rows are immutable: a later prediction for the same match is a new row.
2. **The match is played.** The ingestion jobs write the result to the match.
   The score is *not* copied into the prediction tables — the match stays the
   single source of truth, and the history reads it from there.
3. **Evaluated.** Once the match is finished and its result would be known
   (kickoff plus the feature system's result delay), `evaluate_pending` writes one
   `prediction_results` row: that is what "marked evaluated" means here, and it is
   what the scheduled job does every time results arrive.
4. **Errors and metrics recorded** in that row, from the stored probabilities and
   grid, so the score always describes exactly what was served:
   - `outcome_correct` — the most likely outcome happened;
   - `exact_score_correct`, `btts_correct`;
   - `brier_score`, `log_loss`, `ranked_probability_score`;
   - `actual_score_probability` — what the model gave to the score that happened;
   - `metrics` — per-line over/under correctness, and the goal errors
     (`expected goals − actual goals`, per side, signed).

Nothing in this sprint recalculates any of that. The history endpoint joins the
three records — prediction, match, evaluation — and returns them.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/predictions/history` | The history, newest kickoff first, with a summary |
| GET | `/seasons` | Seasons, so a caller can offer the season filter |

Filters, which combine: `competition_id`, `season_id`, `team_id` (home **or**
away), `model_name`, `model_version_id`, `from_date`, `to_date` (inclusive, on
kickoff), `evaluated` (true: scored only, false: still waiting) and
`outcome_correct`. Paging is `limit` (1–100, default 25) and `offset`.

Each row carries what was predicted (probabilities, expected goals, most probable
score, the predicted outcome), what happened (status, score, actual outcome), how
it did (the evaluation fields above, plus the probability the model gave to the
outcome that happened, picked from the prediction's own three), and which model
version made it.

A prediction whose match has not been played is still listed, with its result
fields null. The history shows what exists rather than hiding what is pending.

### The summary

`summary` covers **every row matching the filters**, not the page returned, so a
filtered view reports its own accuracy. Counts (`predictions`, `evaluated`,
`pending`) cover all matching rows; the averages cover evaluated rows only,
because an unplayed match has no error to average. With nothing evaluated, every
average is `null` — never `0` standing in for "no data".

Averaging a boolean needs `CASE` rather than a cast: PostgreSQL refuses
`boolean::double precision`. The same latent defect was fixed in
`evaluation.accuracy_since`, which had no caller and no test; it now has both.

## The page

`/history` in the web application. Filters sit in the URL, so a filtered view can
be linked to and survives a reload. Above the table is the backend's summary
(predictions, scored, pending, accuracy, exact score, log loss, Brier, RPS, mean
absolute goal error, and the period covered). The table shows, per row:

- kickoff and season;
- the match, linking to its prediction page;
- the prediction: most probable score, predicted outcome, expected goals;
- the three probabilities, with the predicted one in bold and the one that
  happened underlined;
- the result, with a Hit/Miss badge, or "Not scored" for a played match that has
  not been evaluated yet;
- the error: log loss and the two goal errors, signed;
- the model name and version, linking to its performance page.

## Tests

- `backend/tests/api/test_history.py` (21 tests, PostgreSQL): rows pair what was
  predicted with what happened; goal errors equal expected minus actual; each
  filter, and filters combined; the summary follows the filters and reports `null`
  rather than invented averages; a pending prediction becomes evaluated when its
  result arrives, and re-evaluating changes nothing; the seasons endpoint;
  `accuracy_since`.
- `frontend/src/pages/History.test.tsx` (11 tests): the rendered row, the
  probabilities, pending versus scored, the summary, every filter reaching the
  right URL, URL-driven filters, clearing, paging, the empty state and a backend
  failure.

## Limitations

- One row per prediction, not per match: a match predicted by two model versions
  appears twice. Filter by model to compare like with like.
- The team filter is a list of up to 100 teams (narrowed by competition when one
  is chosen); there is no type-ahead yet.
- Over/under correctness is stored per line in `metrics` but is not yet surfaced
  in the history row or the page.
- There is no export (CSV/JSON download) from the page; the API returns JSON.
