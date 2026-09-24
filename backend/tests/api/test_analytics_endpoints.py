"""The endpoints the frontend pages need: matches, teams, predictions, performance, status."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import DataSource, Match, ModelVersion, TeamMatchStatistics
from app.models.enums import MatchStatus

pytestmark = pytest.mark.integration


class TestMatchList:
    def test_lists_matches_newest_first(self, client: TestClient, league: dict[str, int]) -> None:
        body = client.get("/matches", params={"limit": 5}).json()

        assert body["total"] > 5
        kickoffs = [item["kickoff_at"] for item in body["items"]]
        assert kickoffs == sorted(kickoffs, reverse=True)

    def test_filters_by_status_team_and_competition(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        finished = client.get("/matches", params={"status": "finished", "limit": 3}).json()
        assert [item["status"] for item in finished["items"]] == ["finished"] * 3

        by_team = client.get("/matches", params={"team_id": league["Arsenal"], "limit": 100}).json()
        for item in by_team["items"]:
            assert league["Arsenal"] in (item["home_team"]["id"], item["away_team"]["id"])

        cup = client.get("/matches", params={"competition_id": league["cup"]}).json()
        assert [item["competition"]["name"] for item in cup["items"]] == ["FA Cup"]

    def test_an_unknown_status_is_rejected_with_the_options(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        response = client.get("/matches", params={"status": "abandoned_by_referee"})

        assert response.status_code == 422
        assert "known:" in response.json()["detail"]


class TestTeamDetail:
    def test_returns_record_form_and_trend(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        response = client.get(f"/teams/{league['Arsenal']}", params={"matches": 8})

        assert response.status_code == 200
        body = response.json()
        assert body["team"]["name"] == "Arsenal"
        record = body["record"]
        assert record["played"] == record["wins"] + record["draws"] + record["losses"]
        assert record["played"] == record["home"]["played"] + record["away"]["played"]
        assert record["goals_for"] == record["home"]["scored"] + record["away"]["scored"]
        assert record["goals_for_per_match"] == pytest.approx(
            record["goals_for"] / record["played"], abs=1e-3
        )
        assert set(record["form"]) <= {"W", "D", "L"}
        assert len(record["form"]) <= 5
        assert len(body["recent_matches"]) == 8
        trend = body["trend"]
        assert len(trend) == 8
        assert [entry["kickoff_at"] for entry in trend] == sorted(
            entry["kickoff_at"] for entry in trend
        )  # oldest first, for the chart
        first = trend[0]
        assert first["rolling_goals_for"] == pytest.approx(first["goals_for"])
        assert first["result"] in {"W", "D", "L"}

    def test_expected_goals_are_absent_until_a_provider_supplies_them(
        self, client: TestClient, league: dict[str, int], session_factory: sessionmaker[Session]
    ) -> None:
        body = client.get(f"/teams/{league['Arsenal']}").json()
        assert body["record"]["expected_goals_for_per_match"] is None
        assert body["record"]["matches_with_expected_goals"] == 0

        with session_factory() as session, session.begin():
            source = DataSource(code="test_provider", name="Test provider")
            session.add(source)
            session.flush()
            played = session.scalars(
                select(Match)
                .where(
                    Match.status == MatchStatus.FINISHED,
                    Match.home_team_id == league["Arsenal"],
                )
                .order_by(Match.kickoff_at.desc())
                .limit(3)
            ).all()
            for match in played:
                session.add(
                    TeamMatchStatistics(
                        match_id=match.id,
                        team_id=league["Arsenal"],
                        data_source_id=source.id,
                        expected_goals=1.5,
                    )
                )

        updated = client.get(f"/teams/{league['Arsenal']}").json()
        assert updated["record"]["matches_with_expected_goals"] == 3
        assert updated["record"]["expected_goals_for_per_match"] == pytest.approx(1.5)

    def test_unknown_team_is_404(self, client: TestClient, league: dict[str, int]) -> None:
        response = client.get("/teams/999999")

        assert response.status_code == 404
        assert "team 999999 was not found" in response.json()["detail"]


class TestStoredPredictions:
    def test_lists_predictions_with_their_outcome(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        created = client.post(
            "/predictions/match", json={"match_id": league["finished_match"]}
        ).json()

        body = client.get("/predictions").json()

        assert body["total"] == 1
        item = body["items"][0]
        assert item["id"] == created["id"]
        assert item["home_team"] == created["home_team"]
        assert item["model_name"] == "elo"
        assert item["most_probable_score"].count("-") == 1
        assert item["actual_score"] is not None  # the match has been played
        assert item["outcome_correct"] is None  # not evaluated yet

    def test_filters_by_match_and_model(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        client.post("/predictions/match", json={"match_id": league["next_match"]})

        by_match = client.get("/predictions", params={"match_id": league["next_match"]}).json()
        by_model = client.get("/predictions", params={"model_name": "nothing"}).json()

        assert by_match["total"] == 1
        assert by_match["items"][0]["match_id"] == league["next_match"]
        assert by_model["total"] == 0

    def test_an_upcoming_match_has_no_actual_score(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        client.post("/predictions/match", json={"match_id": league["next_match"]})

        item = client.get("/predictions").json()["items"][0]

        assert item["actual_score"] is None and item["outcome_correct"] is None


class TestModelPerformance:
    def test_is_empty_until_predictions_have_been_evaluated(
        self, client: TestClient, active_model: ModelVersion
    ) -> None:
        body = client.get(f"/models/{active_model.id}/performance").json()

        assert body["model"]["name"] == "elo"
        assert body["evaluated"] == 0
        assert body["accuracy"] is None
        assert body["calibration"] == [] and body["monthly"] == []

    def test_reports_accuracy_calibration_and_a_monthly_series(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
    ) -> None:
        for match_id in (league["finished_match"], league["next_match"]):
            client.post("/predictions/match", json={"match_id": match_id})
        with session_factory() as session, session.begin():
            upcoming = session.get(Match, league["next_match"])
            assert upcoming is not None
            upcoming.status = MatchStatus.FINISHED
            upcoming.home_score, upcoming.away_score = 3, 0
            upcoming.kickoff_at = datetime.now(UTC) - timedelta(days=2)
        from app.services.evaluation import evaluate_pending

        with session_factory() as session, session.begin():
            evaluate_pending(session)

        body = client.get(f"/models/{active_model.id}/performance").json()

        assert body["evaluated"] == 2
        assert 0 <= body["accuracy"] <= 1
        assert body["log_loss"] > 0 and body["brier"] > 0
        assert sum(bucket["count"] for bucket in body["calibration"]) == 6  # 3 outcomes each
        for bucket in body["calibration"]:
            assert bucket["low"] <= bucket["mean_predicted"] <= bucket["high"]
            assert 0 <= bucket["observed"] <= 1
        assert sum(entry["matches"] for entry in body["monthly"]) == 2
        assert body["first_match"] and body["last_match"]

    def test_unknown_model_is_404(self, client: TestClient, league: dict[str, int]) -> None:
        assert client.get("/models/9999/performance").status_code == 404


class TestDataStatus:
    def test_reports_coverage_gaps_and_providers(
        self, client: TestClient, league: dict[str, int]
    ) -> None:
        response = client.get("/data/status")

        assert response.status_code == 200
        body = response.json()
        assert body["feature_set_version"].startswith("v1-")
        coverage = {entry["competition"]: entry for entry in body["coverage"]}
        assert set(coverage) == {"Premier League", "FA Cup"}
        premier = coverage["Premier League"]
        assert premier["matches"] > 0
        assert premier["finished"] <= premier["matches"]
        assert premier["with_features"] > 0  # the fixture generated snapshots
        assert premier["with_statistics"] == 0  # none imported in this database
        assert body["matches_missing_features"] >= 0
        assert isinstance(body["issues"], list)
        assert isinstance(body["providers"], list)

    def test_freshness_summarises_the_dashboard_panel(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        body = client.get("/data/freshness").json()

        assert body["upcoming_matches"] == 3
        assert body["active_models"] == 1
        assert body["latest_finished_match_at"] is not None
        assert body["latest_feature_snapshot_at"] is not None
        assert body["predictions_last_24h"] == 0

    def test_the_new_endpoints_are_documented(self, client: TestClient) -> None:
        paths = client.get("/openapi.json").json()["paths"]

        for path in ("/matches", "/teams/{team_id}", "/predictions", "/data/status"):
            assert path in paths, path
        assert paths["/data/status"]["get"]["tags"] == ["status"]
