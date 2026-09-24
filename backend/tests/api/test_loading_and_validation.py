"""Artifact loading, the model cache and request validation (no database)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.registry.loading import ArtifactUnavailableError, ModelCache, local_path
from app.schemas.predictions import BatchPredictionRequest, MatchSelector, PredictionRequest


def _name(loaded: tuple[Any, Any]) -> str:
    """The cache is typed for real models; the test stores a name instead."""
    return str(loaded[0])


def version(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "id": 1,
        "name": "elo",
        "version": "v1",
        "algorithm": "EloModel",
        "artifact_uri": "file:///models/elo",
        "feature_set_version": "v1-abc",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestLocalPath:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("file:///models/elo", Path("/models/elo")),
            ("file:///C:/models/elo", Path("C:/models/elo")),
            ("file:///with%20space/elo", Path("/with space/elo")),
            ("/plain/path", Path("/plain/path")),
        ],
    )
    def test_local_uris_become_paths(self, uri: str, expected: Path) -> None:
        assert local_path(uri) == expected

    @pytest.mark.parametrize("uri", ["s3://bucket/model", "http://host/model"])
    def test_remote_uris_are_refused_with_an_explanation(self, uri: str) -> None:
        with pytest.raises(ArtifactUnavailableError, match="not on a local filesystem"):
            local_path(uri)


class TestModelCache:
    def test_a_model_is_loaded_once_per_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loads = []

        def fake_load(row: Any) -> tuple[str, dict[str, Any]]:
            loads.append(row.version)
            return f"model-{row.version}", {}

        monkeypatch.setattr("app.registry.loading.load_artifact", fake_load)
        cache = ModelCache()

        assert _name(cache.get(version())) == "model-v1"
        assert _name(cache.get(version())) == "model-v1"
        assert loads == ["v1"]

        # A new version of the same row is loaded again.
        assert _name(cache.get(version(version="v2"))) == "model-v2"
        assert loads == ["v1", "v2"]

        cache.clear()
        cache.get(version(version="v2"))
        assert loads == ["v1", "v2", "v2"]

    def test_a_model_this_environment_cannot_unpickle_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The serving image has scikit-learn but not XGBoost or LightGBM."""
        directory = tmp_path / "xgb"
        directory.mkdir()

        def missing_library(*args: Any, **kwargs: Any) -> Any:
            raise ImportError("No module named 'xgboost'")

        monkeypatch.setattr("app.ml.persistence.load_model", missing_library)
        from app.registry.loading import load_artifact

        with pytest.raises(ArtifactUnavailableError, match="cannot load XGBoostModel models"):
            load_artifact(version(algorithm="XGBoostModel", artifact_uri=directory.as_uri()))

    def test_missing_artifacts_are_reported(self, tmp_path: Path) -> None:
        from app.registry.loading import load_artifact

        with pytest.raises(ArtifactUnavailableError, match="has no model artifact"):
            load_artifact(version(artifact_uri=None))
        with pytest.raises(ArtifactUnavailableError, match="does not exist"):
            load_artifact(version(artifact_uri=(tmp_path / "gone").as_uri()))


class TestRequestValidation:
    def test_a_match_id_alone_is_valid(self) -> None:
        assert PredictionRequest(match_id=7).match_id == 7

    def test_team_names_with_optional_context_are_valid(self) -> None:
        request = PredictionRequest(
            home_team="Arsenal", away_team="Chelsea", competition="Premier League"
        )

        assert (request.home_team, request.away_team) == ("Arsenal", "Chelsea")
        assert request.match_id is None
        assert request.refresh is False

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ({}, "give match_id, or home_team and away_team"),
            ({"home_team": "Arsenal"}, "give match_id, or home_team and away_team"),
            ({"match_id": 1, "home_team": "A", "away_team": "B"}, "not both"),
            ({"match_id": 1, "date": "2026-05-01"}, "not both"),
            ({"match_id": 0}, "greater than or equal to 1"),
            ({"home_team": "", "away_team": "B"}, "at least 1 character"),
        ],
    )
    def test_invalid_requests_explain_themselves(
        self, payload: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            PredictionRequest(**payload)

    def test_batches_are_bounded(self) -> None:
        one = BatchPredictionRequest(matches=[MatchSelector(match_id=1)])
        assert one.matches[0].match_id == 1

        with pytest.raises(ValidationError, match="at least 1 item"):
            BatchPredictionRequest(matches=[])
        with pytest.raises(ValidationError, match="at most 50 items"):
            BatchPredictionRequest(matches=[MatchSelector(match_id=1)] * 51)
