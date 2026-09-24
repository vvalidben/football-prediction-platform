"""MLflow Model Registry mirroring, loading a registered model, and the CLI."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from app.ml.__main__ import main as ml_main
from app.ml.experiment import Candidate, run_experiment
from app.ml.feature_sets import CORE_FEATURES
from app.ml.predictors import EloModel, LogisticModel
from app.ml.registry import load_model_version, register_selected
from app.ml.splits import split_by_seasons
from app.ml.tracking import Tracker
from app.models import ModelVersion
from app.registry.mlflow_sync import ACTIVE_ALIAS, MlflowRegistry, sync_active
from app.registry.service import ModelRegistry
from tests.ml.synthetic import make_dataset
from tests.registry.test_service import Clock, new

pytestmark = pytest.mark.integration


@pytest.fixture
def tracker(tmp_path: Path) -> Tracker:
    return Tracker("registry-test", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")


@pytest.fixture
def db(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """A session on the factory's connection, so the CLI sees the same data."""
    with session_factory() as db_session:
        yield db_session
        db_session.commit()


@pytest.fixture
def trained(db: Session, tracker: Tracker) -> ModelVersion:
    """A real trained model: tracked in MLflow, then registered."""
    split = split_by_seasons(make_dataset(seasons=4, teams=8))
    result = run_experiment(
        split,
        [
            Candidate("elo", EloModel),
            Candidate("logistic_regression", lambda **p: LogisticModel(CORE_FEATURES, **p)),
        ],
        tracker=tracker,
    )
    db.execute(
        text(
            "INSERT INTO competitions (id, name, type) OVERRIDING SYSTEM VALUE "
            "VALUES (1, 'Synthetic League', 'league')"
        )
    )
    version_id = register_selected(db, result, tracker)
    db.commit()
    return db.get(ModelVersion, version_id)  # type: ignore[return-value]


class TestMlflowMirror:
    def test_registering_mirrors_the_version_into_mlflow(
        self, trained: ModelVersion, tracker: Tracker
    ) -> None:
        assert trained.mlflow_model_name == trained.name
        assert trained.mlflow_model_version == "1"

        client = MlflowRegistry(tracker.tracking_uri).client
        mirrored = client.get_model_version(trained.name, "1")
        assert mirrored.tags["db_version"] == trained.version
        assert mirrored.tags["algorithm"] == trained.algorithm
        assert mirrored.tags["dataset_version"] == trained.dataset_version
        assert mirrored.source == trained.artifact_uri

    def test_the_active_alias_follows_activation_and_rollback(
        self, db: Session, trained: ModelVersion, tracker: Tracker
    ) -> None:
        registry = ModelRegistry(db, clock=Clock())
        second = registry.register(
            new(name=trained.name, version="v-next", artifact_uri=trained.artifact_uri)
        )
        mlflow_registry = MlflowRegistry(tracker.tracking_uri)
        second.mlflow_model_name = trained.name
        second.mlflow_model_version = mlflow_registry.register(
            trained.name,
            source=str(trained.artifact_uri),
            run_id=None,
            tags={"db_version": "v-next"},
        )

        registry.activate(trained.name, trained.version)
        sync_active(trained.mlflow_model_name, trained.mlflow_model_version, tracker.tracking_uri)
        assert mlflow_registry.active_version(trained.name) == "1"

        registry.activate(trained.name, "v-next")
        sync_active(second.mlflow_model_name, second.mlflow_model_version, tracker.tracking_uri)
        assert mlflow_registry.active_version(trained.name) == "2"

        registry.rollback(trained.name)
        sync_active(trained.mlflow_model_name, trained.mlflow_model_version, tracker.tracking_uri)
        assert mlflow_registry.active_version(trained.name) == "1"
        assert registry.get(trained.name).version == trained.version

    def test_syncing_never_raises_when_mlflow_cannot_be_updated(self, tmp_path: Path) -> None:
        assert "not mirrored in MLflow" in sync_active(None, None)

        message = sync_active("missing-model", "7", f"sqlite:///{(tmp_path / 'x.db').as_posix()}")

        assert "alias not updated" in message

    def test_alias_can_be_cleared(self, trained: ModelVersion, tracker: Tracker) -> None:
        mlflow_registry = MlflowRegistry(tracker.tracking_uri)
        mlflow_registry.set_active(trained.name, "1")

        mlflow_registry.set_active(trained.name, None)

        assert mlflow_registry.active_version(trained.name) is None
        assert ACTIVE_ALIAS == "active"


class TestLoading:
    def test_the_active_model_can_be_loaded_and_used(
        self, db: Session, trained: ModelVersion, tracker: Tracker
    ) -> None:
        ModelRegistry(db).activate(trained.name, trained.version, reason="test")
        data = make_dataset(seasons=1, teams=8)

        loaded = load_model_version(
            db,
            trained.name,
            feature_set_version=trained.feature_set_version,
            tracking_uri=tracker.tracking_uri,
        )

        assert loaded.version.id == trained.id
        assert type(loaded.model).__name__ == trained.algorithm
        probabilities = loaded.model.predict_proba(data)
        assert probabilities.shape == (len(data), 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        assert loaded.metadata["feature_set_version"] == trained.feature_set_version

    def test_a_named_version_can_be_loaded_without_being_active(
        self, db: Session, trained: ModelVersion, tracker: Tracker
    ) -> None:
        loaded = load_model_version(
            db, trained.name, trained.version, tracking_uri=tracker.tracking_uri
        )

        assert loaded.version.version == trained.version
        assert ModelRegistry(db).active(trained.name) is None

    def test_loading_refuses_other_features_and_missing_models(
        self, db: Session, trained: ModelVersion, tracker: Tracker
    ) -> None:
        with pytest.raises(ValueError, match="trained on features"):
            load_model_version(
                db,
                trained.name,
                trained.version,
                feature_set_version="v1-other",
                tracking_uri=tracker.tracking_uri,
            )
        without_artifact = ModelRegistry(db).register(
            new(name="backtest_elo_none", version="b1", artifact_uri=None)
        )
        with pytest.raises(ValueError, match="no model artifact"):
            load_model_version(db, without_artifact.name, "b1")


class TestRegistryCommand:
    def test_list_show_history_activate_rollback_and_load(
        self,
        session_factory: sessionmaker[Session],
        db: Session,
        trained: ModelVersion,
        tracker: Tracker,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        registry = ModelRegistry(db)
        older = registry.register(
            new(
                name=trained.name,
                version="v-old",
                algorithm=trained.algorithm,
                # The same artifact, so it describes what it points at.
                artifact_uri=trained.artifact_uri,
                feature_set_version=trained.feature_set_version,
            )
        )
        db.commit()
        common = ["registry", "--tracking-uri", tracker.tracking_uri]

        assert ml_main([*common, "list"], session_factory=session_factory) == 0
        listing = capsys.readouterr().out
        assert trained.version in listing and trained.algorithm in listing
        assert "Synthetic League" in listing

        assert (
            ml_main(
                [*common, "activate", trained.name, older.version], session_factory=session_factory
            )
            == 0
        )
        assert (
            ml_main(
                [*common, "activate", trained.name, trained.version, "--reason", "better"],
                session_factory=session_factory,
            )
            == 0
        )
        assert f"active version {trained.version}" in capsys.readouterr().out

        assert ml_main([*common, "show", trained.name], session_factory=session_factory) == 0
        shown = capsys.readouterr().out
        assert f'"algorithm": "{trained.algorithm}"' in shown and '"is_active": true' in shown

        assert ml_main([*common, "history", trained.name], session_factory=session_factory) == 0
        history = capsys.readouterr().out
        assert history.count("previous:") == 2 and "better" in history

        assert ml_main([*common, "rollback", trained.name], session_factory=session_factory) == 0
        assert f"active version {older.version}" in capsys.readouterr().out
        with session_factory() as check:
            assert ModelRegistry(check).get(trained.name).version == older.version

        assert ml_main([*common, "load", trained.name], session_factory=session_factory) == 0
        assert "features" in capsys.readouterr().out

    def test_errors_are_reported(
        self,
        session_factory: sessionmaker[Session],
        db: Session,
        trained: ModelVersion,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db.commit()

        code = ml_main(
            ["registry", "activate", trained.name, "v9"], session_factory=session_factory
        )

        assert code == 2
        assert "no version 'v9'" in capsys.readouterr().err


def test_registered_versions_are_visible_to_the_database_only_api(
    db: Session, trained: ModelVersion
) -> None:
    """The service reads everything the API will need, without MLflow or sklearn."""
    rows = db.scalars(select(ModelVersion).where(ModelVersion.name == trained.name)).all()

    assert [r.algorithm for r in rows] == [trained.algorithm]
    assert rows[0].competitions == [{"id": 1, "name": "Synthetic League"}]
    assert rows[0].dataset_version.startswith("ds-")
    assert rows[0].training_data_start is not None and rows[0].training_data_end is not None
    assert mlflow.get_tracking_uri()  # the ML side set one; the service never needs it
