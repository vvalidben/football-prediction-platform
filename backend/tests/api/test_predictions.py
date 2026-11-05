"""POST /predictions/match, POST /predictions/batch and GET /predictions/{id}."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import FeatureSnapshot, Match, ModelVersion, Prediction
from app.models.enums import MatchStatus
from app.registry.service import ModelRegistry

pytestmark = pytest.mark.integration


def _prediction_and_snapshot_counts(
    session_factory: sessionmaker[Session], match_id: int
) -> tuple[int, int]:
    with session_factory() as session:
        predictions = session.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.match_id == match_id)
        )
        snapshots = session.scalar(
            select(func.count())
            .select_from(FeatureSnapshot)
            .where(FeatureSnapshot.match_id == match_id)
        )
    return predictions or 0, snapshots or 0


def _predict_later(
    client: TestClient,
    session_factory: sessionmaker[Session],
    match_id: int,
    model_version: ModelVersion,
) -> Any:
    """The same match predicted an hour from now, so a new cutoff is due."""
    service = client.app.state.predictions  # type: ignore[attr-defined]
    with session_factory() as session, session.begin():
        match = session.get(Match, match_id)
        version = session.get(ModelVersion, model_version.id)
        assert match is not None and version is not None
        (outcome,) = service.predict(
            session, [match], version, now=datetime.now(UTC) + timedelta(hours=1)
        )
    return outcome


def assert_consistent(body: dict[str, Any]) -> None:
    """Every market must agree with the one score matrix it came from."""
    matrix = body["score_probability_matrix"]
    size = body["max_goals"] + 1
    assert len(matrix) == size and all(len(row) == size for row in matrix)
    total = sum(sum(row) for row in matrix)
    assert total == pytest.approx(1.0, abs=1e-4)

    outcomes = (
        body["home_win_probability"],
        body["draw_probability"],
        body["away_win_probability"],
    )
    assert sum(outcomes) == pytest.approx(1.0, abs=1e-5)
    assert all(0 <= p <= 1 for p in outcomes)
    home = sum(matrix[h][a] for h in range(size) for a in range(size) if h > a)
    assert body["home_win_probability"] == pytest.approx(home, abs=1e-3)

    for line in body["over_under_probabilities"]:
        assert line["over"] + line["under"] == pytest.approx(1.0, abs=1e-4)
        over = sum(matrix[h][a] for h in range(size) for a in range(size) if h + a > line["line"])
        assert line["over"] == pytest.approx(over, abs=1e-3)
    assert [line["line"] for line in body["over_under_probabilities"]] == [0.5, 1.5, 2.5, 3.5]

    btts = body["btts_probability"]
    assert btts["yes"] + btts["no"] == pytest.approx(1.0, abs=1e-5)
    yes = sum(matrix[h][a] for h in range(1, size) for a in range(1, size))
    assert btts["yes"] == pytest.approx(yes, abs=1e-3)

    best = body["most_probable_score"]
    assert best["probability"] == pytest.approx(max(max(row) for row in matrix), abs=1e-6)
    assert body["top_scores"][0]["probability"] == pytest.approx(best["probability"])
    assert [s["probability"] for s in body["top_scores"]] == sorted(
        (s["probability"] for s in body["top_scores"]), reverse=True
    )
    expected_home = sum(h * sum(matrix[h]) for h in range(size))
    assert body["expected_home_goals"] == pytest.approx(expected_home, abs=1e-2)


class TestPredictMatch:
    def test_predicting_by_match_id_returns_every_field(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.post("/predictions/match", json={"match_id": league["next_match"]})

        assert response.status_code == 200
        body = response.json()
        assert body["match_id"] == league["next_match"]
        assert body["home_team"] == "Arsenal" and body["away_team"] == "Chelsea"
        assert body["competition"] == "Premier League"
        assert 0 < body["expected_home_goals"] < 6
        assert 0 < body["expected_away_goals"] < 6
        assert body["model"]["name"] == "elo" and body["model"]["version"] == "v1"
        assert body["model"]["algorithm"] == "EloModel"
        assert body["model"]["id"] == active_model.id
        assert body["reused"] is False
        predicted_at = datetime.fromisoformat(body["predicted_at"])
        assert abs((datetime.now(UTC) - predicted_at).total_seconds()) < 120
        assert_consistent(body)

    def test_data_freshness_describes_the_information_used(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        body = client.post("/predictions/match", json={"match_id": league["next_match"]}).json()

        freshness = body["data_freshness"]
        cutoff = datetime.fromisoformat(freshness["feature_cutoff_at"])
        kickoff = datetime.fromisoformat(body["kickoff_at"])
        assert cutoff <= datetime.now(UTC) and cutoff < kickoff
        assert freshness["match_started"] is False
        assert freshness["age_seconds"] >= 0
        latest = datetime.fromisoformat(freshness["latest_result_at"])
        assert latest < cutoff  # only results known by the cutoff were used

    def test_predicting_by_team_names_and_date(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        kickoff = datetime.fromisoformat(
            client.get(f"/matches/{league['next_match']}").json()["kickoff_at"]
        )

        response = client.post(
            "/predictions/match",
            json={
                "home_team": "arsenal",  # case and spacing are normalised
                "away_team": "Chelsea",
                "competition": "Premier League",
                "date": kickoff.date().isoformat(),
            },
        )

        assert response.status_code == 200
        assert response.json()["match_id"] == league["next_match"]

    def test_team_names_without_a_date_use_the_next_meeting(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.post(
            "/predictions/match",
            json={"home_team": "Arsenal", "away_team": "Chelsea", "competition": "Premier League"},
        )

        assert response.status_code == 200
        assert response.json()["match_id"] == league["next_match"]

    def test_a_finished_match_is_predicted_as_of_its_kickoff(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        body = client.post("/predictions/match", json={"match_id": league["finished_match"]}).json()

        freshness = body["data_freshness"]
        assert freshness["match_started"] is True
        assert freshness["feature_cutoff_at"] == body["kickoff_at"]
        assert_consistent(body)

    def test_the_same_request_reuses_the_stored_prediction(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        first = client.post("/predictions/match", json={"match_id": league["next_match"]}).json()

        second = client.post("/predictions/match", json={"match_id": league["next_match"]}).json()

        assert second["id"] == first["id"]
        assert second["reused"] is True and first["reused"] is False
        assert second["home_win_probability"] == first["home_win_probability"]

    def test_a_later_request_with_nothing_new_reuses_the_prediction(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Opening a match page again must not save another copy of the same forecast."""
        first = client.post("/predictions/match", json={"match_id": league["next_match"]}).json()
        counts = _prediction_and_snapshot_counts(session_factory, league["next_match"])

        outcome = _predict_later(client, session_factory, league["next_match"], active_model)

        assert outcome.created is False
        assert _prediction_and_snapshot_counts(session_factory, league["next_match"]) == counts
        assert first["reused"] is False

    def test_a_new_result_makes_a_new_prediction(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
    ) -> None:
        client.post("/predictions/match", json={"match_id": league["next_match"]})
        with session_factory() as session, session.begin():
            upcoming = session.get(Match, league["next_match"])
            assert upcoming is not None
            team = upcoming.home_team_id
            played = session.scalars(
                select(Match)
                .where(
                    Match.status == MatchStatus.FINISHED,
                    or_(Match.home_team_id == team, Match.away_team_id == team),
                )
                .order_by(Match.kickoff_at.desc())
                .limit(1)
            ).one()
            played.home_score = (played.home_score or 0) + 5

        outcome = _predict_later(client, session_factory, league["next_match"], active_model)

        assert outcome.created is True

    def test_refresh_makes_a_new_prediction(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        first = client.post("/predictions/match", json={"match_id": league["next_match"]}).json()

        second = client.post(
            "/predictions/match", json={"match_id": league["next_match"], "refresh": True}
        ).json()

        assert second["id"] != first["id"]
        assert second["reused"] is False
        assert second["home_win_probability"] == pytest.approx(first["home_win_probability"])

    def test_the_stored_prediction_is_returned_unchanged(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        created = client.post("/predictions/match", json={"match_id": league["next_match"]}).json()

        read = client.get(f"/predictions/{created['id']}")

        assert read.status_code == 200
        body = read.json()
        for field in (
            "id",
            "match_id",
            "expected_home_goals",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "score_probability_matrix",
            "most_probable_score",
            "predicted_at",
        ):
            assert body[field] == created[field], field
        assert body["model"] == created["model"]

    def test_the_match_detail_lists_the_prediction(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        created = client.post("/predictions/match", json={"match_id": league["next_match"]}).json()

        detail = client.get(f"/matches/{league['next_match']}").json()

        assert [p["id"] for p in detail["predictions"]] == [created["id"]]
        assert detail["predictions"][0]["model"] == "elo"


class TestPredictionErrors:
    def test_a_request_must_identify_a_match(self, client: TestClient) -> None:
        empty = client.post("/predictions/match", json={})
        both = client.post(
            "/predictions/match",
            json={"match_id": 1, "home_team": "Arsenal", "away_team": "Chelsea"},
        )

        assert empty.status_code == 422
        assert "give match_id, or home_team and away_team" in str(empty.json()["detail"])
        assert both.status_code == 422
        assert "not both" in str(both.json()["detail"])

    def test_unknown_match_team_and_competition(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        missing_match = client.post("/predictions/match", json={"match_id": 987654})
        missing_team = client.post(
            "/predictions/match", json={"home_team": "Madrid", "away_team": "Chelsea"}
        )
        missing_competition = client.post(
            "/predictions/match",
            json={"home_team": "Arsenal", "away_team": "Chelsea", "competition": "La Liga"},
        )

        assert missing_match.status_code == 404
        assert "match 987654 was not found" in missing_match.json()["detail"]
        assert missing_team.status_code == 404
        assert "'Madrid' was not found" in missing_team.json()["detail"]
        assert missing_competition.status_code == 404

    def test_no_scheduled_meeting_is_reported(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.post(
            "/predictions/match", json={"home_team": "Fulham", "away_team": "Everton"}
        )

        assert response.status_code == 404
        assert "none scheduled" in response.json()["detail"]

    def test_an_ambiguous_fixture_is_a_conflict(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.post(
            "/predictions/match", json={"home_team": "Arsenal", "away_team": "Chelsea"}
        )

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "several fixtures" in detail and "give a date or a match_id" in detail

    def test_without_an_active_model(
        self, client: TestClient, league: dict[str, int], session_factory: sessionmaker[Session]
    ) -> None:
        response = client.post("/predictions/match", json={"match_id": league["next_match"]})

        assert response.status_code == 409
        assert "no model is active" in response.json()["detail"]

    def test_with_several_active_models(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
        tmp_path: Any,
    ) -> None:
        with session_factory() as session, session.begin():
            registry = ModelRegistry(session)
            other = session.get(ModelVersion, active_model.id)
            assert other is not None
            registry.register(
                _copy_of(other, name="poisson", version="p1"),
            )
            registry.activate("poisson", "p1", reason="second active model")

        response = client.post("/predictions/match", json={"match_id": league["next_match"]})
        named = client.post(
            "/predictions/match",
            json={"match_id": league["next_match"], "model_name": "poisson"},
        )

        assert response.status_code == 409
        assert "several models are active (elo, poisson)" in response.json()["detail"]
        assert named.status_code == 200
        assert named.json()["model"]["name"] == "poisson"

    def test_an_unknown_model_name_is_404(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.post(
            "/predictions/match",
            json={"match_id": league["next_match"], "model_name": "transformer"},
        )

        assert response.status_code == 404
        assert "transformer" in response.json()["detail"]

    def test_a_missing_artifact_is_503(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
    ) -> None:
        with session_factory() as session, session.begin():
            row = session.get(ModelVersion, active_model.id)
            assert row is not None
            row.artifact_uri = "file:///models/gone"

        response = client.post("/predictions/match", json={"match_id": league["next_match"]})

        assert response.status_code == 503
        assert "does not exist" in response.json()["detail"]

    def test_unknown_prediction_is_404(self, client: TestClient, league: dict[str, int]) -> None:
        response = client.get("/predictions/123456")

        assert response.status_code == 404
        assert "prediction 123456 was not found" in response.json()["detail"]


def _copy_of(row: ModelVersion, *, name: str, version: str) -> Any:
    from app.registry.service import NewModelVersion

    return NewModelVersion(
        name=name,
        version=version,
        algorithm=row.algorithm,
        feature_set_version=row.feature_set_version or "",
        dataset_version=row.dataset_version,
        training_data_start=row.training_data_start or datetime.now(UTC).date(),
        training_data_end=row.training_data_end or datetime.now(UTC).date(),
        competitions=row.competitions,
        hyperparameters=row.hyperparameters,
        metrics=row.training_metrics,
        artifact_uri=row.artifact_uri,
        trained_at=row.trained_at or datetime.now(UTC),
    )


class TestBatch:
    def test_predicts_several_matches_in_one_call(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.post(
            "/predictions/batch",
            json={
                "matches": [
                    {"match_id": league["next_match"]},
                    {"match_id": league["later_match"]},
                    {"match_id": league["cup_match"]},
                ]
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert (body["requested"], body["predicted"], body["failed"]) == (3, 3, 0)
        assert [item["index"] for item in body["items"]] == [0, 1, 2]
        ids = [item["prediction"]["match_id"] for item in body["items"]]
        assert ids == [league["next_match"], league["later_match"], league["cup_match"]]
        for item in body["items"]:
            assert item["error"] is None
            assert_consistent(item["prediction"])

    def test_failures_are_reported_per_item(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.post(
            "/predictions/batch",
            json={
                "matches": [
                    {"match_id": 999999},
                    {"match_id": league["next_match"]},
                    {"home_team": "Nobody", "away_team": "Chelsea"},
                ]
            },
        )

        body = response.json()
        assert (body["requested"], body["predicted"], body["failed"]) == (3, 1, 2)
        assert [item["index"] for item in body["items"]] == [0, 1, 2]
        assert "was not found" in body["items"][0]["error"]
        assert body["items"][1]["prediction"]["match_id"] == league["next_match"]
        assert "'Nobody' was not found" in body["items"][2]["error"]

    def test_batch_size_is_limited(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        too_many = client.post(
            "/predictions/batch",
            json={"matches": [{"match_id": league["next_match"]}] * 51},
        )
        empty = client.post("/predictions/batch", json={"matches": []})

        assert too_many.status_code == 422
        assert empty.status_code == 422

    def test_batch_predictions_are_stored_and_readable(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        body = client.post(
            "/predictions/batch",
            json={"matches": [{"match_id": league["next_match"]}]},
        ).json()

        prediction_id = body["items"][0]["prediction"]["id"]
        assert client.get(f"/predictions/{prediction_id}").status_code == 200


def test_a_prediction_uses_only_information_before_its_cutoff(
    client: TestClient, league: dict[str, int], active_model: ModelVersion
) -> None:
    """The match a prediction is for is never part of the data behind it."""
    body = client.post("/predictions/match", json={"match_id": league["finished_match"]}).json()

    cutoff = datetime.fromisoformat(body["data_freshness"]["feature_cutoff_at"])
    kickoff = datetime.fromisoformat(body["kickoff_at"])
    latest = datetime.fromisoformat(body["data_freshness"]["latest_result_at"])
    assert cutoff == kickoff
    assert latest + timedelta(hours=3) <= cutoff


class TestBatchOrdering:
    """A batch must not depend on the order the caller happens to send.

    Found by running the platform on real data: a batch ordered newest-first
    raised `ValueError: dataset examples must be in chronological order` inside the
    service and came back as a 500 — a server error for a valid request.
    """

    def test_a_batch_in_reverse_chronological_order_is_predicted(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        newest_first = client.get("/matches", params={"status": "finished", "limit": 5}).json()
        ids = [item["id"] for item in newest_first["items"]]
        assert len(ids) == 5

        response = client.post(
            "/predictions/batch", json={"matches": [{"match_id": i} for i in ids]}
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["predicted"] == 5 and body["failed"] == 0

    def test_the_answers_come_back_in_the_order_they_were_asked_for(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        newest_first = client.get("/matches", params={"status": "finished", "limit": 4}).json()
        ids = [item["id"] for item in newest_first["items"]]

        body = client.post(
            "/predictions/batch", json={"matches": [{"match_id": i} for i in ids]}
        ).json()

        returned = [item["prediction"]["match_id"] for item in body["items"]]
        assert returned == ids

    def test_each_match_keeps_its_own_prediction(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        """Reordering for the model must not hand a match another match's answer."""
        newest_first = client.get("/matches", params={"status": "finished", "limit": 4}).json()
        ids = [item["id"] for item in newest_first["items"]]

        batched = client.post(
            "/predictions/batch", json={"matches": [{"match_id": i} for i in ids]}
        ).json()

        for item in batched["items"]:
            match_id = item["prediction"]["match_id"]
            alone = client.post("/predictions/match", json={"match_id": match_id}).json()
            assert item["prediction"]["home_win_probability"] == pytest.approx(
                alone["home_win_probability"], abs=1e-9
            )
            assert item["prediction"]["home_team"] == alone["home_team"]


class TestLatestPerMatch:
    def test_the_list_can_show_each_match_once(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        for refresh in (False, True):
            client.post(
                "/predictions/match", json={"match_id": league["next_match"], "refresh": refresh}
            )
        newest = client.post(
            "/predictions/match", json={"match_id": league["next_match"], "refresh": True}
        ).json()
        client.post("/predictions/match", json={"match_id": league["later_match"]})

        everything = client.get("/predictions").json()
        once = client.get("/predictions", params={"latest_per_match": True}).json()

        assert everything["total"] == 4
        assert once["total"] == 2
        assert sorted(item["match_id"] for item in once["items"]) == sorted(
            [league["next_match"], league["later_match"]]
        )
        assert newest["id"] in {item["id"] for item in once["items"]}
