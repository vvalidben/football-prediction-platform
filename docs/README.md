# Documentation

The guides at the repository root are the entry points:
[ARCHITECTURE](../ARCHITECTURE.md), [DATA](../DATA.md), [ML](../ML.md),
[API](../API.md), [DEPLOYMENT](../DEPLOYMENT.md) and
[CONTRIBUTING](../CONTRIBUTING.md), and [AUDIT](../AUDIT.md) for the final review.
The pages here are the detail behind them.

- [../README.md](../README.md): setup, environment variables, running and testing.
- [../ARCHITECTURE.md](../ARCHITECTURE.md): services, backend and frontend structure,
  design decisions, testing strategy, known limitations.
- [database.md](database.md): PostgreSQL schema, entity relationship diagram,
  provider-ID design, constraints and database limitations.
- [data-providers.md](data-providers.md): provider selection, provider interface,
  API-Football adapter and its limitations, ingestion, run log and CLI.
- [historical-pipeline.md](historical-pipeline.md): historical backfill,
  normalisation, validation policy, data-quality issues, audit and report.
- [features.md](features.md): leakage prevention and its tests, the feature list,
  versioning, snapshots and the real-data verification.
- [models.md](models.md): the baseline and tree-ensemble models, the common model
  interface, chronological evaluation and time-aware cross-validation, metrics,
  MLflow tracking, model versioning and results on eight real seasons.
- [prediction-engine.md](prediction-engine.md): the goal prediction engine: the score
  matrix, the maximum score, derived markets, the prediction object, consistency
  tests and derived-market results.
- [scheduled-jobs.md](scheduled-jobs.md): the background jobs, their schedules,
  retries and rate-limit handling, job status and the retraining policy.
- [api.md](api.md): moved to [../API.md](../API.md).
- [testing.md](testing.md): what the suite covers, the eight guarantees and where
  each is proven, the conventions tests follow, and the known gaps.
- [operations.md](operations.md): request ids and logging, rate limiting, why
  there is no authentication, caching, the measured index work, Docker and
  secret handling, and the dependency review.
- [prediction-history.md](prediction-history.md): the life of a prediction from
  served to scored, the history endpoint and its filters, what the summary
  averages, and the history page.
- [explainability.md](explainability.md): what a feature contribution claims and
  what it does not, the method used per model, why ensembles are refused, the two
  endpoints and how the figures are shown.
- [frontend.md](frontend.md): the five pages, how they load data, their loading,
  error and empty states, the charts and the page tests.
- [model-registry.md](model-registry.md): model versions, the active model,
  activation history, rollback, loading and the MLflow Model Registry mirror.
- [backtesting.md](backtesting.md): the walk-forward backtest, calibration within
  temporal validation, and results by season, competition and prediction type;
  [backtest-report.md](backtest-report.md) is the full generated report.
- [ensembles.md](ensembles.md): ensembles of the models: validation-only weights,
  forward-chained evaluation, the recommendation rule and the results.
- [model-comparison.md](model-comparison.md): the generated comparison report
  (all metrics, paired bootstrap intervals, feature importances) on real data.

Put further design notes, data-source descriptions and model documentation in this
folder as the project grows, for example `data-sources.md` or `models/<model>.md`.
