"""The explanation endpoints: faithful to the stored prediction, honest when refused."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.features.config import FeatureConfig
from app.models import ModelVersion
from app.registry.service import ModelRegistry, NewModelVersion

pytestmark = pytest.mark.integration


def register(
    session_factory: sessionmaker[Session],
    directory: Path,
    *,
    name: str,
    algorithm: str,
    competition_id: int,
    build: Any,
) -> int:
    """Fit `build()` on the league, save it and activate it under `name`."""
    from app.ml.dataset import load_dataset

    version = FeatureConfig().version()
    with session_factory() as session:
        data = load_dataset(session, version)
    model = build().fit(data)
    model.save(directory, {"feature_set_version": version})
    with session_factory() as session, session.begin():
        row = ModelRegistry(session).register(
            NewModelVersion(
                name=name,
                version="v1",
                algorithm=algorithm,
                feature_set_version=version,
                dataset_version=data.fingerprint(),
                training_data_start=data.examples[0].kickoff_at.date(),
                training_data_end=data.examples[-1].kickoff_at.date(),
                competitions=[{"id": competition_id, "name": "Premier League"}],
                hyperparameters={},
                metrics={},
                artifact_uri=directory.as_uri(),
                trained_at=datetime.now(UTC),
                description=f"{name} for explanation tests",
            )
        )
        ModelRegistry(session).activate(name, "v1", reason="tests")
        session.flush()
        return int(row.id)


@pytest.fixture
def logistic_model(
    league: dict[str, int], session_factory: sessionmaker[Session], tmp_path: Path
) -> int:
    from app.ml.feature_sets import CORE_FEATURES
    from app.ml.predictors import LogisticModel

    return register(
        session_factory,
        tmp_path / "logistic-model",
        name="logistic_regression",
        algorithm="LogisticModel",
        competition_id=league["competition"],
        build=lambda: LogisticModel(CORE_FEATURES, c=0.1),
    )


@pytest.fixture
def baseline_model(
    league: dict[str, int], session_factory: sessionmaker[Session], tmp_path: Path
) -> int:
    from app.ml.predictors import HistoricalBaseline

    return register(
        session_factory,
        tmp_path / "baseline-model",
        name="historical_baseline",
        algorithm="HistoricalBaseline",
        competition_id=league["competition"],
        build=HistoricalBaseline,
    )


def predict(client: TestClient, match_id: int, model_name: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"match_id": match_id}
    if model_name:
        body["model_name"] = model_name
    response = client.post("/predictions/match", json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())


class TestExplainAPrediction:
    def test_reports_the_features_that_moved_each_outcome(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        prediction = predict(client, league["next_match"])

        response = client.get(f"/predictions/{prediction['id']}/explanation")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["prediction_id"] == prediction["id"]
        assert body["home_team"] == prediction["home_team"]
        assert body["model_name"] == "elo"
        assert [target["key"] for target in body["targets"]] == ["home_win", "draw", "away_win"]
        for target in body["targets"]:
            assert target["contributions"]
            for item in target["contributions"]:
                assert item["label"] and item["group"]
                assert item["direction"] == (
                    "positive" if item["contribution"] >= 0 else "negative"
                )

    def test_the_explained_probabilities_are_the_prediction_s_own(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        """An explanation of a different number would be worse than none."""
        prediction = predict(client, league["next_match"])

        body = client.get(f"/predictions/{prediction['id']}/explanation").json()

        explained = {target["key"]: target["predicted_output"] for target in body["targets"]}
        assert explained["home_win"] == pytest.approx(prediction["home_win_probability"], abs=1e-6)
        assert explained["draw"] == pytest.approx(prediction["draw_probability"], abs=1e-6)
        assert explained["away_win"] == pytest.approx(prediction["away_win_probability"], abs=1e-6)

    def test_the_parts_add_up_to_the_whole_even_when_the_list_is_cut_short(
        self, client: TestClient, league: dict[str, int], logistic_model: int
    ) -> None:
        prediction = predict(client, league["next_match"], model_name="logistic_regression")

        body = client.get(f"/predictions/{prediction['id']}/explanation", params={"top": 3}).json()

        for target in body["targets"]:
            assert len(target["contributions"]) == 3
            assert target["other_features"] == body["features_used"] - 3
            listed = sum(item["contribution"] for item in target["contributions"])
            total = target["baseline"] + listed + target["other_contribution"]
            assert total == pytest.approx(target["predicted"], abs=1e-9)

    def test_contributions_are_ordered_with_the_largest_first(
        self, client: TestClient, league: dict[str, int], logistic_model: int
    ) -> None:
        prediction = predict(client, league["next_match"], model_name="logistic_regression")

        body = client.get(f"/predictions/{prediction['id']}/explanation").json()

        sizes = [abs(item["contribution"]) for item in body["targets"][0]["contributions"]]
        assert sizes == sorted(sizes, reverse=True)

    def test_the_response_says_it_describes_the_model_not_causes(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        prediction = predict(client, league["next_match"])

        body = client.get(f"/predictions/{prediction['id']}/explanation").json()

        assert "not causes" in body["caveat"]
        assert body["method"] and body["method_detail"]
        assert body["baseline_description"]

    def test_explaining_a_prediction_twice_gives_the_same_answer(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        prediction = predict(client, league["next_match"])
        path = f"/predictions/{prediction['id']}/explanation"

        assert client.get(path).json() == client.get(path).json()

    def test_an_unknown_prediction_is_404(self, client: TestClient, league: dict[str, int]) -> None:
        response = client.get("/predictions/424242/explanation")

        assert response.status_code == 404
        assert "was not found" in response.json()["detail"]

    def test_a_model_without_a_valid_method_is_refused_with_the_reason(
        self, client: TestClient, league: dict[str, int], baseline_model: int
    ) -> None:
        """No approximation is invented for a model that cannot be explained."""
        prediction = predict(client, league["next_match"], model_name="historical_baseline")

        response = client.get(f"/predictions/{prediction['id']}/explanation")

        assert response.status_code == 409
        assert "reads no features" in response.json()["detail"]


class TestModelImportance:
    def test_reports_what_the_model_relies_on_overall(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.get(f"/models/{active_model.id}/importance")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model_name"] == "elo"
        assert [entry["feature"] for entry in body["entries"]] == ["elo_diff"]
        assert body["entries"][0]["importance"] == pytest.approx(1.0)
        assert body["entries"][0]["label"] == "Home Elo + home advantage - away Elo"
        assert "not causes" in body["caveat"]

    def test_a_long_ranking_is_cut_with_the_rest_summed(
        self, client: TestClient, league: dict[str, int], logistic_model: int
    ) -> None:
        body = client.get(f"/models/{logistic_model}/importance", params={"top": 4}).json()

        assert len(body["entries"]) == 4
        assert body["other_features"] > 0
        listed = sum(entry["importance"] for entry in body["entries"])
        assert listed + body["other_importance"] == pytest.approx(1.0)
        assert all(
            body["entries"][i]["importance"] >= body["entries"][i + 1]["importance"]
            for i in range(3)
        )

    def test_a_model_without_an_importance_measure_is_refused(
        self, client: TestClient, league: dict[str, int], baseline_model: int
    ) -> None:
        response = client.get(f"/models/{baseline_model}/importance")

        assert response.status_code == 409
        assert "importance" in response.json()["detail"]

    def test_an_unknown_model_is_404(self, client: TestClient, league: dict[str, int]) -> None:
        assert client.get("/models/424242/importance").status_code == 404


class TestDocumentation:
    def test_both_endpoints_are_documented(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]

        explanation = paths["/predictions/{prediction_id}/explanation"]["get"]
        assert explanation["tags"] == ["predictions"]
        assert "not a cause" in explanation["description"]
        assert {"404", "409", "503"} <= set(explanation["responses"])
        importance = paths["/models/{model_id}/importance"]["get"]
        assert "model behaviour" in importance["description"]
