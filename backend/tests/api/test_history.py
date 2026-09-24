"""Prediction history: the record of what was predicted, what happened, and the errors."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Match, ModelVersion
from app.models.enums import MatchStatus

pytestmark = pytest.mark.integration


def predict(client: TestClient, match_id: int) -> dict[str, Any]:
    response = client.post("/predictions/match", json={"match_id": match_id})
    assert response.status_code == 200, response.text
    return dict(response.json())


def evaluate(session_factory: sessionmaker[Session]) -> int:
    from app.services.evaluation import evaluate_pending

    with session_factory() as session, session.begin():
        return evaluate_pending(session).evaluated


@pytest.fixture
def predicted_history(
    client: TestClient,
    league: dict[str, int],
    active_model: ModelVersion,
    session_factory: sessionmaker[Session],
) -> dict[str, Any]:
    """Three predictions: two on played matches (evaluated), one still to come."""
    finished = predict(client, league["finished_match"])
    upcoming = predict(client, league["next_match"])
    # A second played match, so filters have more than one row to work with.
    with session_factory() as session:
        other = session.scalars(
            select(Match)
            .where(
                Match.status == MatchStatus.FINISHED,
                Match.id != league["finished_match"],
                Match.home_team_id == league["Arsenal"],
            )
            .order_by(Match.kickoff_at.desc())
            .limit(1)
        ).one()
        other_id, other_season = other.id, other.season_id
    second = predict(client, other_id)
    evaluated = evaluate(session_factory)
    return {
        "finished": finished,
        "second": second,
        "upcoming": upcoming,
        "evaluated": evaluated,
        "season_id": other_season,
    }


class TestHistoryRows:
    def test_lists_what_was_predicted_beside_what_happened(
        self, client: TestClient, predicted_history: dict[str, Any]
    ) -> None:
        body = client.get("/predictions/history").json()

        assert body["total"] == 3
        played = next(
            row
            for row in body["items"]
            if row["prediction_id"] == predicted_history["finished"]["id"]
        )
        assert played["home_score"] is not None and played["away_score"] is not None
        assert played["actual_outcome"] in {"home", "draw", "away"}
        assert played["predicted_outcome"] in {"home", "draw", "away"}
        assert played["model_name"] == "elo" and played["model_version"] == "v1"
        assert played["competition"] == "Premier League"
        assert played["season"]

    def test_reports_the_errors_of_an_evaluated_prediction(
        self, client: TestClient, predicted_history: dict[str, Any]
    ) -> None:
        body = client.get("/predictions/history", params={"evaluated": True}).json()

        assert body["total"] == predicted_history["evaluated"] == 2
        for row in body["items"]:
            assert row["evaluated_at"]
            assert row["outcome_correct"] is not None
            assert row["log_loss"] > 0 and row["brier_score"] >= 0
            assert 0 <= row["ranked_probability_score"] <= 1
            assert row["home_goals_error"] == pytest.approx(
                row["expected_home_goals"] - row["home_score"], abs=1e-6
            )
            assert row["away_goals_error"] == pytest.approx(
                row["expected_away_goals"] - row["away_score"], abs=1e-6
            )
            probabilities = {
                "home": row["home_win_probability"],
                "draw": row["draw_probability"],
                "away": row["away_win_probability"],
            }
            assert row["actual_outcome_probability"] == probabilities[row["actual_outcome"]]
            assert row["outcome_correct"] == (row["predicted_outcome"] == row["actual_outcome"])

    def test_a_prediction_on_an_unplayed_match_is_listed_without_a_result(
        self, client: TestClient, predicted_history: dict[str, Any]
    ) -> None:
        body = client.get("/predictions/history", params={"evaluated": False}).json()

        assert [row["prediction_id"] for row in body["items"]] == [
            predicted_history["upcoming"]["id"]
        ]
        row = body["items"][0]
        assert row["home_score"] is None
        assert row["actual_outcome"] is None
        assert row["evaluated_at"] is None
        assert row["log_loss"] is None and row["home_goals_error"] is None
        assert row["home_win_probability"] > 0  # what was predicted is still there

    def test_rows_are_newest_kickoff_first(
        self, client: TestClient, predicted_history: dict[str, Any]
    ) -> None:
        kickoffs = [row["kickoff_at"] for row in client.get("/predictions/history").json()["items"]]

        assert kickoffs == sorted(kickoffs, reverse=True)


class TestFilters:
    def test_by_competition(self, client: TestClient, predicted_history: dict[str, Any]) -> None:
        competition_id = client.get("/predictions/history").json()["items"][0]["competition_id"]

        matching = client.get(
            "/predictions/history", params={"competition_id": competition_id}
        ).json()
        other = client.get("/predictions/history", params={"competition_id": 999999}).json()

        assert matching["total"] == 3
        assert other["total"] == 0 and other["items"] == []

    def test_by_season(self, client: TestClient, predicted_history: dict[str, Any]) -> None:
        season_id = predicted_history["season_id"]

        body = client.get("/predictions/history", params={"season_id": season_id}).json()

        assert body["total"] >= 1
        assert {row["season_id"] for row in body["items"]} == {season_id}

    def test_by_team_covers_home_and_away(
        self, client: TestClient, league: dict[str, int], predicted_history: dict[str, Any]
    ) -> None:
        body = client.get("/predictions/history", params={"team_id": league["Arsenal"]}).json()

        assert body["total"] >= 1
        for row in body["items"]:
            assert league["Arsenal"] in (row["home_team_id"], row["away_team_id"])

    def test_by_date_range(self, client: TestClient, predicted_history: dict[str, Any]) -> None:
        rows = client.get("/predictions/history").json()["items"]
        newest = datetime.fromisoformat(rows[0]["kickoff_at"])

        on_the_day = client.get(
            "/predictions/history",
            params={"from_date": newest.date().isoformat(), "to_date": newest.date().isoformat()},
        ).json()
        before = client.get(
            "/predictions/history",
            params={"to_date": (newest.date() - timedelta(days=1)).isoformat()},
        ).json()

        assert on_the_day["total"] >= 1
        assert all(
            datetime.fromisoformat(row["kickoff_at"]).date() == newest.date()
            for row in on_the_day["items"]
        )
        assert before["total"] == 3 - on_the_day["total"]

    def test_by_model(
        self, client: TestClient, active_model: ModelVersion, predicted_history: dict[str, Any]
    ) -> None:
        by_name = client.get("/predictions/history", params={"model_name": "elo"}).json()
        by_version = client.get(
            "/predictions/history", params={"model_version_id": active_model.id}
        ).json()
        unknown = client.get("/predictions/history", params={"model_name": "nothing"}).json()

        assert by_name["total"] == by_version["total"] == 3
        assert unknown["total"] == 0

    def test_by_whether_the_model_got_it_right(
        self, client: TestClient, predicted_history: dict[str, Any]
    ) -> None:
        right = client.get("/predictions/history", params={"outcome_correct": True}).json()
        wrong = client.get("/predictions/history", params={"outcome_correct": False}).json()

        assert right["total"] + wrong["total"] == predicted_history["evaluated"]
        assert all(row["outcome_correct"] for row in right["items"])
        assert all(row["outcome_correct"] is False for row in wrong["items"])

    def test_filters_combine(
        self, client: TestClient, league: dict[str, int], predicted_history: dict[str, Any]
    ) -> None:
        body = client.get(
            "/predictions/history",
            params={"team_id": league["Arsenal"], "model_name": "elo", "evaluated": True},
        ).json()

        for row in body["items"]:
            assert league["Arsenal"] in (row["home_team_id"], row["away_team_id"])
            assert row["model_name"] == "elo"
            assert row["evaluated_at"]


class TestSummary:
    def test_summarises_every_row_that_matches_not_just_the_page(
        self, client: TestClient, predicted_history: dict[str, Any]
    ) -> None:
        page = client.get("/predictions/history", params={"limit": 1}).json()

        assert len(page["items"]) == 1
        summary = page["summary"]
        assert summary["predictions"] == 3
        assert summary["evaluated"] == 2
        assert summary["pending"] == 1
        assert 0 <= summary["accuracy"] <= 1
        assert summary["log_loss"] > 0
        assert summary["mean_absolute_goals_error"] >= 0
        assert summary["first_kickoff_at"] < summary["last_kickoff_at"]

    def test_the_summary_follows_the_filters(
        self, client: TestClient, predicted_history: dict[str, Any]
    ) -> None:
        everything = client.get("/predictions/history").json()["summary"]
        only_pending = client.get("/predictions/history", params={"evaluated": False}).json()[
            "summary"
        ]

        assert everything["evaluated"] == 2
        assert only_pending["predictions"] == 1
        assert only_pending["evaluated"] == 0
        assert only_pending["accuracy"] is None  # nothing scored, so nothing averaged
        assert only_pending["mean_absolute_goals_error"] is None

    def test_the_accuracy_is_the_share_of_correct_rows(
        self, client: TestClient, predicted_history: dict[str, Any]
    ) -> None:
        body = client.get("/predictions/history", params={"evaluated": True}).json()

        correct = sum(1 for row in body["items"] if row["outcome_correct"])
        assert body["summary"]["accuracy"] == pytest.approx(correct / len(body["items"]))

    def test_an_empty_history_reports_zeros_not_invented_averages(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        body = client.get("/predictions/history").json()

        assert body["total"] == 0 and body["items"] == []
        assert body["summary"] == {
            "predictions": 0,
            "evaluated": 0,
            "pending": 0,
            "accuracy": None,
            "exact_score_accuracy": None,
            "btts_accuracy": None,
            "log_loss": None,
            "brier_score": None,
            "ranked_probability_score": None,
            "mean_home_goals_error": None,
            "mean_away_goals_error": None,
            "mean_absolute_goals_error": None,
            "first_kickoff_at": None,
            "last_kickoff_at": None,
        }


class TestEvaluationOnFinish:
    def test_a_prediction_is_scored_once_its_match_is_played(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
    ) -> None:
        """The history turns from pending to evaluated when the result arrives."""
        prediction = predict(client, league["next_match"])
        path = "/predictions/history"
        assert client.get(path).json()["items"][0]["evaluated_at"] is None

        with session_factory() as session, session.begin():
            match = session.get(Match, league["next_match"])
            assert match is not None
            match.status = MatchStatus.FINISHED
            match.home_score, match.away_score = 2, 1
            match.kickoff_at = datetime.now(UTC) - timedelta(days=1)
        evaluate(session_factory)

        row = next(
            item
            for item in client.get(path).json()["items"]
            if item["prediction_id"] == prediction["id"]
        )
        assert row["home_score"] == 2 and row["away_score"] == 1
        assert row["actual_outcome"] == "home"
        assert row["evaluated_at"] is not None
        assert row["outcome_correct"] == (row["predicted_outcome"] == "home")
        assert row["home_goals_error"] == pytest.approx(row["expected_home_goals"] - 2, abs=1e-6)

    def test_evaluating_twice_does_not_change_the_history(
        self, client: TestClient, predicted_history: dict[str, Any], session_factory: Any
    ) -> None:
        before = client.get("/predictions/history").json()

        assert evaluate(session_factory) == 0  # nothing left to score
        assert client.get("/predictions/history").json() == before


class TestAccuracySince:
    """Used by the reporting job; it averages a boolean, which PostgreSQL is fussy about."""

    def test_averages_what_has_been_evaluated_since_a_moment(
        self, predicted_history: dict[str, Any], session_factory: sessionmaker[Session]
    ) -> None:
        from app.services.evaluation import accuracy_since

        with session_factory() as session:
            recent = accuracy_since(session, since=datetime.now(UTC) - timedelta(hours=1))
            ancient = accuracy_since(session, since=datetime.now(UTC) + timedelta(hours=1))

        assert recent["evaluated"] == predicted_history["evaluated"]
        assert 0 <= recent["accuracy"] <= 1
        assert recent["log_loss"] > 0
        assert ancient == {"evaluated": 0, "accuracy": 0.0, "log_loss": 0.0}


class TestSeasonsEndpoint:
    def test_lists_seasons_for_the_filter(self, client: TestClient, league: dict[str, int]) -> None:
        body = client.get("/seasons").json()

        assert body["total"] >= 3
        names = [item["name"] for item in body["items"]]
        assert names == sorted(names, reverse=True)
        assert {item["competition"] for item in body["items"]} == {"Premier League", "FA Cup"}
        assert sum(1 for item in body["items"] if item["is_current"]) == 1

    def test_filters_by_competition(self, client: TestClient, league: dict[str, int]) -> None:
        body = client.get("/seasons", params={"competition_id": league["cup"]}).json()

        assert {item["competition"] for item in body["items"]} == {"FA Cup"}


class TestDocumentation:
    def test_the_history_endpoints_are_documented(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]

        history = paths["/predictions/history"]["get"]
        assert history["tags"] == ["predictions"]
        parameters = {parameter["name"] for parameter in history["parameters"]}
        assert {
            "competition_id",
            "season_id",
            "team_id",
            "model_name",
            "from_date",
            "to_date",
        } <= parameters
        assert "/seasons" in paths
