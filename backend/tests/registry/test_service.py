"""The model registry: required metadata, the active model, history and rollback."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import ModelActivation, ModelVersion
from app.registry.service import (
    ModelRegistry,
    NewModelVersion,
    RegistryError,
    activation_record,
    metadata,
)

pytestmark = pytest.mark.integration

START = datetime(2026, 1, 1, tzinfo=UTC)


class Clock:
    """A clock that advances a minute on every call, for deterministic timestamps."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        self.now += timedelta(minutes=1)
        return self.now


def new(**overrides: object) -> NewModelVersion:
    """A complete registration, with any field replaced."""
    values: dict[str, object] = {
        "name": "elo",
        "version": "v1",
        "algorithm": "EloModel",
        "feature_set_version": "v1-abc",
        "dataset_version": "ds-123",
        "training_data_start": date(2018, 8, 10),
        "training_data_end": date(2024, 5, 19),
        "competitions": [{"id": 1, "name": "Premier League"}],
        "hyperparameters": {"feature": "elo_diff"},
        "metrics": {"validation": {"log_loss": 0.99}},
        "artifact_uri": "file:///models/elo",
        "trained_at": START,
        "description": "selected on cv_log_loss",
    }
    values.update(overrides)
    return NewModelVersion(**values)  # type: ignore[arg-type]


@pytest.fixture
def registry(session: Session) -> ModelRegistry:
    return ModelRegistry(session, clock=Clock())


class TestRegistration:
    def test_a_version_records_everything_required(self, registry: ModelRegistry) -> None:
        row = registry.register(new())

        data = metadata(row)
        assert data.id == row.id
        assert (data.name, data.version, data.algorithm) == ("elo", "v1", "EloModel")
        assert data.feature_set_version == "v1-abc"
        assert data.dataset_version == "ds-123"
        assert (data.training_period.start, data.training_period.end) == (
            date(2018, 8, 10),
            date(2024, 5, 19),
        )
        assert data.competitions == [{"id": 1, "name": "Premier League"}]
        assert data.hyperparameters == {"feature": "elo_diff"}
        assert data.metrics == {"validation": {"log_loss": 0.99}}
        assert data.created_at is not None and data.trained_at == START
        assert data.artifact_uri == "file:///models/elo"
        assert data.is_active is False  # registering never activates

    def test_metadata_is_json_ready(self, registry: ModelRegistry) -> None:
        import json

        dumped = json.dumps(metadata(registry.register(new())).model_dump(mode="json"))

        assert json.loads(dumped)["dataset_version"] == "ds-123"

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"algorithm": " "}, "needs a algorithm"),
            ({"dataset_version": ""}, "needs a dataset version"),
            ({"feature_set_version": ""}, "needs a feature set version"),
            ({"version": ""}, "needs a version"),
            ({"competitions": []}, "competitions it was trained on"),
            ({"competitions": [{"name": "x"}]}, "every competition needs an id"),
            (
                {"training_data_start": date(2025, 1, 1), "training_data_end": date(2024, 1, 1)},
                "ends before it starts",
            ),
        ],
    )
    def test_incomplete_metadata_is_refused(
        self, registry: ModelRegistry, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(RegistryError, match=message):
            registry.register(new(**overrides))

    def test_the_same_version_cannot_be_registered_twice(self, registry: ModelRegistry) -> None:
        registry.register(new())

        with pytest.raises(RegistryError, match="already registered"):
            registry.register(new(description="second attempt"))

        registry.register(new(version="v2"))  # a new version is fine

    def test_versions_are_listed_newest_first_per_name(self, registry: ModelRegistry) -> None:
        registry.register(new(version="v1"))
        registry.register(new(version="v2"))
        registry.register(new(name="poisson", version="p1", algorithm="PoissonModel"))

        assert [r.version for r in registry.versions("elo")] == ["v2", "v1"]
        assert registry.names() == ["elo", "poisson"]
        assert len(registry.versions()) == 3


class TestActiveModel:
    def test_activating_records_the_change(self, registry: ModelRegistry) -> None:
        registry.register(new(version="v1"))

        activation = registry.activate("elo", "v1", reason="first release")

        assert registry.active("elo") is not None
        assert registry.get("elo").version == "v1"  # the active one by default
        record = activation_record(activation)
        assert (record.version, record.previous_version) == ("v1", None)
        assert record.reason == "first release" and record.is_rollback is False
        assert record.deactivated_at is None

    def test_activating_another_version_replaces_the_active_one(
        self, registry: ModelRegistry, session: Session
    ) -> None:
        registry.register(new(version="v1"))
        registry.register(new(version="v2"))
        registry.activate("elo", "v1")

        registry.activate("elo", "v2", reason="better validation log loss")

        assert registry.get("elo").version == "v2"
        assert [r.version for r in registry.previous("elo")] == ["v1"]
        history = [activation_record(a) for a in registry.history("elo")]
        assert [(h.version, h.previous_version) for h in history] == [("v2", "v1"), ("v1", None)]
        assert history[1].deactivated_at == history[0].activated_at
        assert session.scalars(
            select(ModelVersion.version).where(ModelVersion.name == "elo", ModelVersion.is_active)
        ).all() == ["v2"]

    def test_activating_the_active_version_changes_nothing(self, registry: ModelRegistry) -> None:
        registry.register(new(version="v1"))
        first = registry.activate("elo", "v1")

        again = registry.activate("elo", "v1", reason="no-op")

        assert again.id == first.id
        assert len(registry.history("elo")) == 1

    def test_models_without_an_artifact_cannot_be_activated(self, registry: ModelRegistry) -> None:
        registry.register(new(name="backtest_elo_none", version="b1", artifact_uri=None))

        with pytest.raises(RegistryError, match="no model artifact"):
            registry.activate("backtest_elo_none", "b1")

    def test_each_name_has_its_own_active_version(self, registry: ModelRegistry) -> None:
        registry.register(new(version="v1"))
        registry.register(new(name="ensemble_weighted_average", version="e1"))
        registry.activate("elo", "v1")
        registry.activate("ensemble_weighted_average", "e1")

        assert registry.get("elo").version == "v1"
        assert registry.get("ensemble_weighted_average").version == "e1"

    def test_the_database_refuses_two_active_versions_of_a_name(
        self, registry: ModelRegistry, session: Session
    ) -> None:
        registry.register(new(version="v1"))
        second = registry.register(new(version="v2"))
        registry.activate("elo", "v1")

        second.is_active = True
        with pytest.raises(IntegrityError):
            session.flush()

    def test_deactivating_leaves_no_active_version(self, registry: ModelRegistry) -> None:
        registry.register(new(version="v1"))
        registry.activate("elo", "v1")

        registry.deactivate("elo", reason="paused for review")

        assert registry.active("elo") is None
        with pytest.raises(RegistryError, match="no active version"):
            registry.get("elo")
        record = activation_record(registry.history("elo")[0])
        assert record.deactivated_at is not None
        assert "paused for review" in (record.reason or "")
        assert [r.version for r in registry.previous("elo")] == ["v1"]

    def test_unknown_names_and_versions_are_reported(self, registry: ModelRegistry) -> None:
        with pytest.raises(RegistryError, match="no active version"):
            registry.get("missing")
        registry.register(new(version="v1"))
        with pytest.raises(RegistryError, match="no version 'v9'"):
            registry.get("elo", "v9")
        with pytest.raises(RegistryError, match="no active version"):
            registry.deactivate("elo")


class TestRollback:
    def test_rollback_restores_the_previous_version(self, registry: ModelRegistry) -> None:
        registry.register(new(version="v1"))
        registry.register(new(version="v2"))
        registry.activate("elo", "v1")
        registry.activate("elo", "v2")

        activation = registry.rollback("elo")

        assert registry.get("elo").version == "v1"
        record = activation_record(activation)
        assert record.is_rollback is True
        assert record.previous_version == "v2"
        assert record.reason == "rollback from v2 to v1"

    def test_rollback_can_be_undone_by_rolling_back_again(self, registry: ModelRegistry) -> None:
        registry.register(new(version="v1"))
        registry.register(new(version="v2"))
        registry.activate("elo", "v1")
        registry.activate("elo", "v2")
        registry.rollback("elo")

        registry.rollback("elo")

        assert registry.get("elo").version == "v2"

    def test_rollback_to_a_named_earlier_version(self, registry: ModelRegistry) -> None:
        for version in ("v1", "v2", "v3"):
            registry.register(new(version=version))
            registry.activate("elo", version)

        registry.rollback("elo", to_version="v1", reason="v2 and v3 both regressed")

        assert registry.get("elo").version == "v1"
        assert activation_record(registry.history("elo")[0]).reason == "v2 and v3 both regressed"

    def test_rollback_needs_somewhere_to_go(self, registry: ModelRegistry) -> None:
        registry.register(new(version="v1"))
        registry.register(new(version="v2"))

        with pytest.raises(RegistryError, match="no active version to roll back"):
            registry.rollback("elo")

        registry.activate("elo", "v1")
        with pytest.raises(RegistryError, match="no previous version"):
            registry.rollback("elo")
        with pytest.raises(RegistryError, match="never active"):
            registry.rollback("elo", to_version="v2")
        with pytest.raises(RegistryError, match="already active"):
            registry.rollback("elo", to_version="v1")


def test_a_version_activated_before_activations_were_recorded_still_works(
    registry: ModelRegistry, session: Session
) -> None:
    """Rows from before this migration are active without an activation row."""
    legacy = registry.register(new(version="legacy"))
    legacy.is_active = True
    session.flush()
    registry.register(new(version="v2"))

    registry.activate("elo", "v2", reason="replacing the legacy model")

    assert registry.get("elo").version == "v2"
    assert [r.version for r in registry.previous("elo")] == ["legacy"]
    closed = session.scalars(
        select(ModelActivation).where(ModelActivation.deactivated_at.is_not(None))
    ).all()
    assert [a.model_version.version for a in closed] == ["legacy"]
    assert registry.rollback("elo").model_version.version == "legacy"
