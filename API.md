# API

The REST interface over the data and models. Code: `backend/app/api/routes/`,
`backend/app/services/`, `backend/app/schemas/`.

- Interactive documentation: `/docs` — **withheld when `ENVIRONMENT=production`**,
  together with `/openapi.json`.
- Every response carries `X-Request-ID`. A 500 returns that id and nothing else
  about the failure.
- Callers are rate limited per address: 300 requests a minute, and 30 a minute for
  predictions, answered with `429` and `Retry-After`. See
  [docs/operations.md](docs/operations.md).
- Through the bundled nginx the browser talks to one origin and the API is reached
  at `/api/...`; the paths below are the backend's own.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service status plus PostgreSQL and Redis checks (503 when degraded) |
| GET | `/competitions` | Competitions, with the current season; filter by `country` |
| GET | `/seasons` | Seasons, newest first; filter by `competition_id` |
| GET | `/teams` | Teams; filter by `search` and `competition_id` |
| GET | `/teams/{team_id}` | A team's record, home and away splits, form, recent matches and goal trend |
| GET | `/matches` | Matches played or scheduled, newest first; filter by competition, team, `status` and dates |
| GET | `/matches/upcoming` | Matches not yet played |
| GET | `/matches/{match_id}` | One match with teams, result so far and stored predictions |
| POST | `/predictions/match` | Predict one match |
| POST | `/predictions/batch` | Predict up to 50 matches in one call |
| GET | `/predictions` | Stored predictions with their outcome; filter by `match_id`, `model_name`; `latest_per_match=true` keeps each match's newest |
| GET | `/predictions/history` | History with results, errors and a summary; filter by competition, season, team, model, date and whether it has been scored |
| GET | `/predictions/{prediction_id}` | A stored prediction, unchanged |
| GET | `/predictions/{prediction_id}/explanation` | Which features moved that prediction, per outcome |
| GET | `/models` | Registered model versions; filter by `name`, `active_only` |
| GET | `/models/{model_id}` | One model version's metadata |
| GET | `/models/{model_id}/metrics` | The metrics recorded when it was registered |
| GET | `/models/{model_id}/history` | When that model name was served by which version |
| GET | `/models/{model_id}/performance` | Measured accuracy, scores, calibration bins and a monthly series |
| GET | `/models/{model_id}/importance` | What that model version relies on in general |
| GET | `/data/status` | Synchronisation, coverage per competition, gaps, providers, job runs and open issues |
| GET | `/data/freshness` | Last import, last job, newest result and feature snapshot |
| GET | `/jobs`, `/jobs/summary`, `/jobs/{run_id}` | Status of the scheduled background jobs |

List endpoints return `{total, limit, offset, items}`; `limit` is 1–100.

## Asking for a prediction

By match id:

```bash
curl -X POST http://localhost:8080/api/predictions/match \
  -H 'Content-Type: application/json' -d '{"match_id": 12345}'
```

or by teams, optionally narrowed by competition and date:

```json
{"home_team": "Arsenal", "away_team": "Chelsea",
 "competition": "Premier League", "date": "2026-05-24"}
```

Names are matched after normalisation (case, accents, punctuation and club affixes
are ignored), so `"arsenal"` finds `Arsenal`. Without a date, the next scheduled
meeting is used; if several fixtures match, the response is `409` and lists them.
Giving both a `match_id` and team names is a `422`.

Optional: `model_name` (which registered model to use) and `refresh` (compute a new
prediction instead of reusing the stored one).

## What comes back

```json
{
  "id": 1,
  "match_id": 3031,
  "kickoff_at": "2026-05-24T15:00:00Z",
  "competition": "Premier League",
  "home_team": "Brighton",
  "away_team": "Man United",
  "expected_home_goals": 1.361,
  "expected_away_goals": 1.351,
  "most_probable_score": {"home_goals": 1, "away_goals": 1, "probability": 0.124329},
  "top_scores": [{"home_goals": 1, "away_goals": 1, "probability": 0.124329}, "…"],
  "score_probability_matrix": [[0.067, 0.093, "… 11 x 11"]],
  "max_goals": 10,
  "home_win_probability": 0.36492,
  "draw_probability": 0.262259,
  "away_win_probability": 0.372821,
  "over_under_probabilities": [{"line": 0.5, "over": 0.932538, "under": 0.067462}, "…"],
  "btts_probability": {"yes": 0.552858, "no": 0.447142},
  "model": {"id": 1, "name": "elo", "version": "20260922201641-08e237bb",
            "algorithm": "EloModel", "feature_set_version": "v1-079714a734"},
  "predicted_at": "2026-09-22T20:16:42Z",
  "data_freshness": {"feature_cutoff_at": "2026-05-24T15:00:00Z",
                     "latest_result_at": "2026-05-24T11:30:00Z",
                     "last_ingestion_at": "2026-09-22T20:15:10Z",
                     "age_seconds": 10250742.0,
                     "match_started": true},
  "reused": false
}
```

Every market comes from the one score matrix
([docs/prediction-engine.md](docs/prediction-engine.md)), so 1X2, over/under and
BTTS always agree. Stored values are rounded to six decimals, so sums are within a
few millionths of 1.

**Data freshness** says what the prediction knew:

- `feature_cutoff_at` — no information after this moment was used;
- `latest_result_at` — the most recent finished match included (kickoff + 3 h ≤ cutoff);
- `last_ingestion_at` — when data was last imported successfully;
- `age_seconds` — how old the cutoff is now;
- `match_started` — whether the match has kicked off.

## How a prediction is produced

1. The match is resolved from the request.
2. Features are computed on demand through the leak-free path, with the cutoff at
   now for a match that has not started, or at kickoff for one that has. The cutoff
   is floored to the minute and the snapshot is saved.
3. The active version of the requested (or only) model is loaded from its artifact
   and cached per version.
4. The model produces one score grid; every field above is derived from it.
5. The prediction is stored — with its full grid and over/under lines — and returned.

Asking again from the same feature snapshot returns the stored prediction with
`reused: true`. `refresh: true` forces a new one.

## Batch

```json
{"matches": [{"match_id": 1}, {"home_team": "Arsenal", "away_team": "Chelsea"}]}
```

Returns `{requested, predicted, failed, items}`. Each item carries either a
`prediction` or an `error`, in request order, so one bad entry does not fail the
rest. Features for the whole batch are computed in one pass.

## History and explanations

`GET /predictions/history` pairs what was predicted with what happened, and adds a
summary over **every row matching the filters** — not just the page returned. Its
averages cover evaluated rows only, and are `null` when nothing has been scored
([docs/prediction-history.md](docs/prediction-history.md)).

`GET /predictions/{id}/explanation` returns per-feature contributions for that
stored prediction, computed from the snapshot it was made from. A model with no
valid method (the historical baseline, an ensemble) returns `409` with the reason
rather than an approximation. Contributions describe **model behaviour, not
causality**; every response carries that caveat
([docs/explainability.md](docs/explainability.md)).

## Errors

| Status | When |
| --- | --- |
| 400/422 | The request is not valid: neither or both ways of identifying a match, a bad page size, an unparseable id |
| 404 | No such match, team, competition, model or prediction |
| 409 | Several fixtures match; no model is active; several are active (name one); this model cannot be explained |
| 422 | Features could not be computed for the match |
| 429 | Rate limit reached; `Retry-After` says when to come back |
| 503 | The model artifact is missing, or this service cannot load that model type |

Errors are `{"detail": "..."}` and say what to do, for example
`"no model is active; activate one with python -m app.ml registry activate"`.
Unhandled errors are logged with a stack trace and returned as
`{"detail": "Internal server error", "request_id": "…"}`.

## Caching

Only `/data/status` and `/data/freshness` are cached (30 s by default, in Redis,
with a matching `Cache-Control`). Predictions, history, explanations and model
metrics are never cached: they are records, and a caller should see what is stored
now. The cache fails open — if Redis is unavailable the answer is simply computed.

## Serving requirements

The API image installs the `serve` extra (numpy, scipy, scikit-learn) and not the
full ML stack:

- artifacts are read straight from their `file://` location, so MLflow is not needed;
- a registered XGBoost or LightGBM model cannot be loaded by this image and returns
  `503` explaining that, instead of failing obscurely. Serving those needs the `ml`
  extra in the image ([DEPLOYMENT.md](DEPLOYMENT.md)).

The model served is whichever version is **active** in the registry. Activation
stays a deliberate step ([docs/model-registry.md](docs/model-registry.md)).

## Authentication

There is none, by design: the deployment is a public, read-only view with no
accounts and no personal data, and the only state a caller creates is a stored
prediction. The expensive path is bounded by rate limiting instead. If this ever
serves private or paid access, authentication belongs at the edge so the browser
app stays free of credentials — the reasoning is in
[docs/operations.md](docs/operations.md).

## Tests

`backend/tests/api/` covers every endpoint's success and failure paths against a
real PostgreSQL database and a real fitted model, including consistency of every
market against the returned matrix, and that the OpenAPI document matches the
routes that exist. See [docs/testing.md](docs/testing.md).

## Limitations

- Predictions are computed synchronously; a batch of 50 unseen matches computes
  features for all of them inside the request.
- `/matches/upcoming` needs scheduled matches: a database of finished matches only
  returns an empty list.
- No API versioning yet: the path has no `/v1` prefix, and a breaking change would
  need one.
- Rate-limit windows are fixed rather than sliding.
