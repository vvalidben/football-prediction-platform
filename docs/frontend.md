# Frontend

The React application people use: fixtures, predictions, teams, model performance
and the state of the data behind all of it. Code: `frontend/src/`.

## Pages

| Route | Page | What it shows |
| --- | --- | --- |
| `/` | Dashboard | Data freshness, the next matches, the latest predictions, the active models, and the live health of the API, PostgreSQL and Redis |
| `/matches/:matchId` | Match prediction | Predicted score, expected goals, 1X2, exact-score distribution, total goals, over/under, both-teams-to-score, why the model predicted it ([explainability.md](explainability.md)), and both teams' recent form and statistics |
| `/history` | Prediction history | Every stored prediction with its result, error and model version, filtered by competition, season, team, model, date or whether it has been scored ([prediction-history.md](prediction-history.md)) |
| `/teams` | Teams | A searchable index, the way into a team page |
| `/teams/:teamId` | Team | Record, scoring and conceding, home and away splits, xG where a provider supplied it, a goal trend and the recent matches |
| `/models` | Model performance | Measured accuracy, log loss, Brier, RPS and exact-score accuracy, a reliability diagram, a monthly history, what the model relies on, and separately the metrics recorded at registration |
| `/data` | Data status | Last synchronisation, coverage per competition, matches without features, provider status, import and job runs, and open data-quality issues |

Routing is `react-router-dom`; unknown paths redirect to the dashboard. Every page
is reachable from the header, and matches, teams and models link to one another.

## Data

Each page reads the REST API described in [api.md](api.md). Nothing is computed in
the browser that the backend does not already provide, and nothing is filled in
when data is missing:

- `frontend/src/api/client.ts` — one `fetch` wrapper. Failures become an `ApiError`
  carrying the HTTP status and the FastAPI `detail` (a validation error's messages
  are joined), so pages show what the backend actually said. A request that cannot
  be made at all reports "The backend is unreachable".
- `frontend/src/api/endpoints.ts` — one function per endpoint, so a URL is written
  once.
- `frontend/src/api/types.ts` — the response shapes as the pages read them.
- `frontend/src/hooks/useApi.ts` — `AsyncState<T>` is `loading | success | error`.
  The hook aborts the request in flight when its dependencies change or the
  component unmounts, so a slow response can never overwrite a newer one, and takes
  an optional refresh interval (the dashboard and data status refresh every 30 s).

## States

A render error in one view would otherwise unmount the whole application and
leave a blank page, so the routes are wrapped in an `ErrorBoundary`
(`src/components/ErrorBoundary.tsx`): the layout and navigation stay usable, the
message is shown, and navigating elsewhere clears it.


`frontend/src/components/states.tsx` and `Async.tsx` give every page the same three
states:

- **Loading** — `role="status"`, `aria-live="polite"`, with what is being loaded.
- **Error** — `role="alert"`, the backend's message, the status code, and a
  "Try again" button that re-runs only the failed request. One failing endpoint
  does not blank the page: the dashboard still lists fixtures when the freshness
  panel is down.
- **Empty** — a plain statement that there is nothing yet, and why. No placeholder
  numbers, no sample fixtures, no invented predictions. Examples: "No predictions
  yet", "No evaluated predictions yet" on a model that has never been scored,
  "No provider supplied xG", and a coverage bar showing `0 / 1100` statistics.

## Charts

Hand-drawn SVG and flexbox in `frontend/src/components/charts/Charts.tsx` — no
charting dependency, which keeps the bundle small and every chart labelled:

| Chart | Used for |
| --- | --- |
| `BarChart` | 1X2 probabilities, most likely scores |
| `ColumnChart` | Total-goals distribution |
| `LineChart` | A team's rolling goals, a model's monthly accuracy and log loss |
| `ContributionChart` | Feature contributions either side of a centre line: green right for what pushed the model's score up, red left for what pushed it down |
| `CalibrationChart` | Reliability diagram: predicted probability against observed frequency, point size by number of cases, dashed diagonal for perfect calibration |

The exact-score distribution is a colour-scaled table of the score grid with the
most probable score ringed; every cell carries its probability as text and as a
tooltip. Each chart has an `aria-label`, so the figures are reachable without
seeing the drawing.

## Conventions

- Tailwind v4 with the `pitch-*` palette; layouts are single-column on a phone and
  two-column from `lg` upwards.
- Dates and numbers are formatted in one place (`frontend/src/format.ts`): absolute
  times use the viewer's locale, and "how current is this" is phrased relatively
  ("3 hours ago", "never").
- Probabilities are shown as percentages with one decimal, goal averages with two.
- A missing number renders as `—`, never as `0`.

## Tests

Vitest with jsdom and Testing Library, next to the code (145 tests). Each page has a
test file that mocks `fetch` through `frontend/src/test/render.tsx`, which answers
by path prefix and 404s anything a test did not declare. The tests cover, per page:
what is rendered from a full response, the empty state, and a backend failure. They
also check that xG only appears once a provider supplies it, that the recompute
button asks for `refresh: true`, and that changing the model or the match window
re-requests the right URL. The explanation panel has its own file covering the
wording, the direction of each bar and the case where a model cannot be explained
([explainability.md](explainability.md)).

```bash
cd frontend
npm test          # vitest run
npm run lint      # eslint
npm run typecheck # tsc -b
npm run build     # tsc -b && vite build
```

## Limitations

- The match page asks the API for a prediction when it opens. The backend reuses a
  stored prediction built from the same features and model version, so this is not
  a new computation each time; "Recompute" forces one.
- There is no authentication, by design: every page is public and read-only, and
  the API bounds the expensive path with rate limiting instead. The reasoning is in
  [operations.md](operations.md).
- Lists are capped (8 on the dashboard, 50 teams) and there is no pagination
  control yet; the API supports `limit`/`offset` already.
- The team index searches on every keystroke without debouncing.
