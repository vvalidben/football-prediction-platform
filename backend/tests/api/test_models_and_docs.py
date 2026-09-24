"""GET /models, /models/{id}/metrics, and the generated API documentation."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models import ModelVersion
from app.registry.service import ModelRegistry
from tests.api.test_predictions import _copy_of

pytestmark = pytest.mark.integration


class TestModels:
    def test_lists_registered_versions_with_every_field(
        self, client: TestClient, active_model: ModelVersion
    ) -> None:
        response = client.get("/models")

        assert response.status_code == 200
        [model] = response.json()
        assert model["id"] == active_model.id
        assert model["name"] == "elo" and model["version"] == "v1"
        assert model["algorithm"] == "EloModel"
        assert model["feature_set_version"] == active_model.feature_set_version
        assert model["dataset_version"].startswith("ds-")
        assert model["training_period"]["start"] and model["training_period"]["end"]
        assert model["competitions"][0]["name"] == "Premier League"
        assert model["hyperparameters"] == {"feature": "elo_diff"}
        assert model["metrics"]["validation"]["log_loss"] == 0.98
        assert model["is_active"] is True
        assert model["created_at"] and model["trained_at"]

    def test_filters_by_name_and_active(
        self,
        client: TestClient,
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
    ) -> None:
        with session_factory() as session, session.begin():
            row = session.get(ModelVersion, active_model.id)
            assert row is not None
            ModelRegistry(session).register(_copy_of(row, name="poisson", version="p1"))

        assert len(client.get("/models").json()) == 2
        assert [m["name"] for m in client.get("/models", params={"name": "poisson"}).json()] == [
            "poisson"
        ]
        active = client.get("/models", params={"active_only": True}).json()
        assert [m["name"] for m in active] == ["elo"]

    def test_reads_one_version_and_its_metrics(
        self, client: TestClient, active_model: ModelVersion
    ) -> None:
        one = client.get(f"/models/{active_model.id}")
        metrics = client.get(f"/models/{active_model.id}/metrics")

        assert one.status_code == 200 and one.json()["version"] == "v1"
        assert metrics.status_code == 200
        body = metrics.json()
        assert body["model"]["name"] == "elo"
        assert body["model"]["algorithm"] == "EloModel"
        assert body["model"]["is_active"] is True
        assert body["metrics"]["validation"] == {"log_loss": 0.98, "accuracy": 0.52}

    def test_history_shows_activations(
        self, client: TestClient, active_model: ModelVersion
    ) -> None:
        history = client.get(f"/models/{active_model.id}/history").json()

        assert len(history) == 1
        assert history[0]["version"] == "v1"
        assert history[0]["reason"] == "tests"
        assert history[0]["deactivated_at"] is None

    def test_unknown_model_is_404(self, client: TestClient, league: dict[str, int]) -> None:
        for path in ("/models/4242", "/models/4242/metrics", "/models/4242/history"):
            response = client.get(path)
            assert response.status_code == 404, path
            assert "was not found" in response.json()["detail"]


class TestDocumentation:
    def test_every_documented_endpoint_is_present(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()

        assert set(spec["paths"]) == {
            "/health",
            "/competitions",
            "/teams",
            "/seasons",
            "/teams/{team_id}",
            "/matches",
            "/matches/upcoming",
            "/matches/{match_id}",
            "/predictions",
            "/predictions/match",
            "/predictions/batch",
            "/predictions/{prediction_id}",
            "/predictions/{prediction_id}/explanation",
            "/predictions/history",
            "/models",
            "/models/{model_id}",
            "/models/{model_id}/metrics",
            "/models/{model_id}/history",
            "/models/{model_id}/performance",
            "/models/{model_id}/importance",
            "/jobs",
            "/jobs/summary",
            "/jobs/{run_id}",
            "/data/status",
            "/data/freshness",
        }

    def test_endpoints_are_described_and_tagged(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()

        predict = spec["paths"]["/predictions/match"]["post"]
        assert predict["summary"] == "Predict one match"
        assert "match_id" in predict["description"]
        assert predict["tags"] == ["predictions"]
        assert {"404", "409", "422", "503"} <= set(predict["responses"])
        assert {tag["name"] for tag in spec["tags"]} == {
            "health",
            "catalog",
            "predictions",
            "models",
            "jobs",
            "status",
        }
        assert "one exact-score distribution" in spec["info"]["description"]

    def test_the_request_schema_documents_both_ways_to_pick_a_match(
        self, client: TestClient
    ) -> None:
        spec = client.get("/openapi.json").json()

        request = spec["components"]["schemas"]["PredictionRequest"]
        assert set(request["properties"]) >= {
            "match_id",
            "home_team",
            "away_team",
            "competition",
            "date",
            "model_name",
            "refresh",
        }
        assert request["examples"][0] == {"match_id": 12345}

    def test_the_response_schema_documents_every_promised_field(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()

        response = spec["components"]["schemas"]["PredictionRead"]
        assert set(response["properties"]) >= {
            "home_team",
            "away_team",
            "expected_home_goals",
            "expected_away_goals",
            "most_probable_score",
            "score_probability_matrix",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "over_under_probabilities",
            "btts_probability",
            "model",
            "predicted_at",
            "data_freshness",
        }

    def test_interactive_docs_are_served_outside_production(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200
