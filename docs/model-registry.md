# Model registry

Version management for trained models: what every version records, which one is
active, what was active before, and how to roll back and load. Code:
`backend/app/registry/` (service and MLflow mirror, no ML dependencies),
`backend/app/ml/registry.py` (registering trained models, loading artifacts).

## What a version records

`model_versions` (one row per model version):

| Field | Column | Example |
| --- | --- | --- |
| Model ID | `id` (and `name`, unique with `version`) | 2, `elo` |
| Version | `version` | `20260922193601-80eab58a` (time + MLflow run) |
| Algorithm | `algorithm` | `EloModel`, `XGBoostModel`, `EnsembleModel` |
| Feature version | `feature_set_version` | `v1-079714a734` |
| Dataset version | `dataset_version` | `ds-911414f244768710` |
| Training period | `training_data_start`, `training_data_end` | 2018-09-01 .. 2025-05-25 |
| Competitions | `competitions` (JSONB) | `[{"id": 1, "name": "Premier League"}]` |
| Hyperparameters | `hyperparameters` (JSONB) | `{"feature": "elo_diff", "model_type": "EloModel"}` |
| Evaluation metrics | `training_metrics` (JSONB) | validation, cross-validation and test metrics, MLflow run id |
| Creation timestamp | `created_at` (and `trained_at`) | |
| Artifact | `artifact_uri` | the MLflow run's `model/` directory |
| Active flag | `is_active` | one active version per name (partial unique index) |
| MLflow link | `mlflow_model_name`, `mlflow_model_version` | `elo`, `2` |

**Dataset version** is a fingerprint of the exact training data: a hash of the
feature-set version and of every match's id, kickoff and result, in order
(`Dataset.fingerprint()`). Two runs on the same matches give the same dataset
version; adding, removing or correcting one match changes it.

Every field is required when registering. A registration without an algorithm,
dataset version, feature version, training period or competitions is refused.

## Active, previous and rollback

`model_activations` records every period a version was active: the version, the one
it replaced, when it started, when it ended, the reason, and whether it was a
rollback. The open row (no `deactivated_at`) is the current activation; a partial
unique index allows only one per model name.

- **Registering never activates.** New versions arrive inactive; nothing existing is
  modified.
- **Activating** deactivates the current version and opens a new activation in the
  same transaction, recording which version it replaced.
- **Only loadable versions can be activated.** A version without an artifact (a
  backtest record, for instance) is refused.
- **Rollback** re-activates the version that was active before the current one (or
  `--to VERSION`, which must have been active before) and is marked as a rollback.
  Rolling back twice returns to where you started.
- **Concurrency.** Activation for a model name takes a transaction-level PostgreSQL
  advisory lock, so two simultaneous activations cannot interleave.
- **Older databases.** A version that was active before this sprint has no
  activation row. The registry treats the `is_active` flag as authoritative and
  writes the missing row when that version is next changed, so it still appears in
  the history and can be rolled back to. The migration backfills it, and both paths
  are tested.

## MLflow

MLflow tracking was already in place (Sprint 5). Sprint 10 adds the **MLflow Model
Registry**: registering a version also creates an MLflow model version pointing at
the same artifact, tagged with the database version, algorithm, dataset version and
feature version. The alias `active` follows activations and rollbacks, so the MLflow
UI shows the same active model as the database.

The database is the source of truth. MLflow mirroring is best effort: if MLflow is
unavailable or not installed, the registry still works and the command says the
alias was not updated.

## Commands

```bash
python -m app.ml registry list [--name elo]
python -m app.ml registry show elo [--version 20260922193601-80eab58a]
python -m app.ml registry history elo
python -m app.ml registry activate elo 20260922193601-80eab58a --reason "retrained"
python -m app.ml registry rollback elo [--to VERSION] [--reason "regression"]
python -m app.ml registry deactivate elo --reason "paused for review"
python -m app.ml registry load elo [--feature-version v1-079714a734]
```

`python -m app.ml train --register` registers the selected model (and, with
`--ensemble`, the best ensemble as its own version).

## Loading a model

```python
from app.ml.registry import load_model_version

loaded = load_model_version(session, "elo", feature_set_version=version)
loaded.model.predict_proba(dataset)      # any Predictor: baselines, trees, ensembles
loaded.version.dataset_version           # what it was trained on
```

The artifact is downloaded from its MLflow URI and unpickled. Loading refuses a
model whose feature-set version differs from the features it will be given, and a
version with no artifact.

## For the API (later)

`app/registry/service.py` has no ML dependencies, so the API image (which has
neither scikit-learn nor MLflow) can read the registry. `metadata(row)` returns a
`ModelVersionMetadata` (`app/schemas/model_registry.py`), a Pydantic model ready to
be returned by an endpoint; `activation_record(activation)` does the same for
history. No endpoints exist yet: exposing them is a later sprint.

## Verification on real data

Eight Premier League seasons, through the real CLI: two training runs registered two
versions of `elo` with identical dataset versions (same fit data), then activate,
activate again, roll back, and load:

```
elo   20260922193601-80eab58a   EloModel   ds-911414f244768710   2018-09-01 .. 2025-05-25   Premier League
elo   20260922193548-813bd003   EloModel   ds-911414f244768710   2018-09-01 .. 2025-05-25   Premier League

activate …3bd003  → MLflow alias 'active' → version 1
activate …ab58a   → MLflow alias 'active' → version 2
rollback          → MLflow alias 'active' → version 1   (recorded as a rollback)
load              → EloModel, 3 features, feature set v1-079714a734
                    prediction: home 0.348, draw 0.262, away 0.391, most likely 1-1
```

## Tests (`backend/tests/registry/`, 36 tests)

- **Metadata:** every required field is stored and returned; incomplete
  registrations are refused (missing algorithm, dataset version, feature version,
  version, competitions, a competition without an id, an inverted training period);
  the same version cannot be registered twice; metadata serialises to JSON.
- **Active model:** activating records the change; activating another version
  replaces and closes the previous activation; activating the active version is a
  no-op; versions without artifacts are refused; each name has its own active
  version; the database itself refuses two active versions of a name; deactivation
  leaves none.
- **Rollback:** restores the previous version, is marked as a rollback, can be
  undone by rolling back again, accepts an explicit earlier version, and reports
  when there is nowhere to go (never active, already active, no history).
- **Legacy rows:** a version active without an activation row is handled, recorded
  in history and can be rolled back to.
- **MLflow:** registering mirrors the version with its tags and source; the `active`
  alias follows activation and rollback; the alias can be cleared; failures to sync
  are reported, never raised.
- **Loading:** the active model loads and predicts; a named version loads without
  being active; other feature versions and missing artifacts are refused.
- **CLI:** list, show, history, activate, rollback and load end to end on
  PostgreSQL, plus error reporting.
- **Migration:** a database written before this sprint upgrades with the algorithm
  backfilled from the recorded model type, `dataset_version` set to `unrecorded`,
  and an open activation row for the version that was active; the downgrade removes
  everything.

## Limitations

- Activation is a manual step: there is no automatic promotion from evaluation
  results, by design.
- Rollback is "undo the last activation" (or an explicit target); there is no
  scheduled or time-based rollback.
- The registry does not verify that an artifact still exists when activating; `load`
  is how you check that.
- Metrics are stored as JSON rather than in typed columns, so the shape can change
  between sprints.
