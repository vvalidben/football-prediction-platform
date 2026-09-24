"""Guarantees 2 and 6, end to end: a real database, real features, a real model.

`tests/ml/test_critical_guarantees.py` states these against the model in isolation.
Here they run through the whole path the API takes — resolve the match, build the
features from the database, load the artifact, predict, store — because that is
where a leak or a crash would actually happen.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Match, ModelVersion, Season, Team
from app.models.enums import MatchStatus

pytestmark = pytest.mark.integration


def predict(client: TestClient, match_id: int, *, refresh: bool = False) -> dict[str, Any]:
    response = client.post("/predictions/match", json={"match_id": match_id, "refresh": refresh})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def probabilities(body: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(body["home_win_probability"]),
        float(body["draw_probability"]),
        float(body["away_win_probability"]),
    )


class TestNoPostMatchInformation:
    """2. No post-match information enters predictions."""

    def test_the_result_arriving_does_not_change_the_prediction(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
    ) -> None:
        """The strongest form of the guarantee: predict a match while it is still
        scheduled, then record a lopsided result and force a fresh prediction. If
        anything about the match leaked in, the numbers would move."""
        match_id = league["next_match"]
        before = predict(client, match_id)

        with session_factory() as session, session.begin():
            match = session.get(Match, match_id)
            assert match is not None
            match.status = MatchStatus.FINISHED
            match.home_score, match.away_score = 7, 0
            match.kickoff_at = datetime.now(UTC) - timedelta(days=1)

        after = predict(client, match_id, refresh=True)

        assert probabilities(after) == pytest.approx(probabilities(before), abs=1e-9)
        assert after["expected_home_goals"] == pytest.approx(
            before["expected_home_goals"], abs=1e-9
        )

    def test_a_played_match_is_predicted_as_of_its_kickoff(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        body = predict(client, league["finished_match"])

        freshness = body["data_freshness"]
        assert isinstance(freshness, dict)
        cutoff = datetime.fromisoformat(str(freshness["feature_cutoff_at"]))
        kickoff = datetime.fromisoformat(str(body["kickoff_at"]))
        assert cutoff == kickoff
        assert freshness["match_started"] is True

    def test_results_from_other_matches_after_the_cutoff_are_excluded(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: ModelVersion,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Not just the match's own result: nothing that happened anywhere after
        the cutoff may be behind the prediction."""
        body = predict(client, league["finished_match"])

        freshness = body["data_freshness"]
        assert isinstance(freshness, dict)
        cutoff = datetime.fromisoformat(str(freshness["feature_cutoff_at"]))
        latest_used = datetime.fromisoformat(str(freshness["latest_result_at"]))
        with session_factory() as session:
            newest_overall = session.scalar(
                select(Match.kickoff_at)
                .where(Match.status == MatchStatus.FINISHED)
                .order_by(Match.kickoff_at.desc())
                .limit(1)
            )

        assert newest_overall is not None
        assert latest_used < cutoff
        assert latest_used <= newest_overall


def _unknown_fixture(
    session_factory: sessionmaker[Session], league: dict[str, int], home: str, away: str
) -> int:
    """Two brand-new clubs and a scheduled match between them: no history at all."""
    with session_factory() as session, session.begin():
        season = session.scalars(
            select(Season)
            .where(Season.competition_id == league["competition"])
            .order_by(Season.start_date.desc())
        ).first()
        assert season is not None
        teams = [Team(name=home, country="England"), Team(name=away, country="England")]
        session.add_all(teams)
        session.flush()
        match = Match(
            competition_id=league["competition"],
            season_id=season.id,
            home_team_id=teams[0].id,
            away_team_id=teams[1].id,
            kickoff_at=datetime.now(UTC) + timedelta(days=5),
            status=MatchStatus.SCHEDULED,
        )
        session.add(match)
        session.flush()
        return int(match.id)


class TestUnknownTeamsEndToEnd:
    """6. Unknown teams are handled."""

    @pytest.fixture
    def newcomer_match(self, league: dict[str, int], session_factory: sessionmaker[Session]) -> int:
        """A fixture between two clubs the platform has never seen play."""
        return _unknown_fixture(session_factory, league, "Promoted FC", "Newcomers United")

    @pytest.fixture
    def second_newcomer_match(
        self, league: dict[str, int], session_factory: sessionmaker[Session]
    ) -> int:
        return _unknown_fixture(session_factory, league, "Another New FC", "Late Arrivals")

    def test_a_match_between_two_teams_with_no_history_is_predicted(
        self, client: TestClient, newcomer_match: int, active_model: ModelVersion
    ) -> None:
        body = predict(client, newcomer_match)

        home, draw, away = probabilities(body)
        assert home + draw + away == pytest.approx(1.0, abs=1e-5)
        assert all(0 < p < 1 for p in (home, draw, away))
        assert body["expected_home_goals"] > 0 and body["expected_away_goals"] > 0

    def test_the_score_distribution_is_still_complete(
        self, client: TestClient, newcomer_match: int, active_model: ModelVersion
    ) -> None:
        body = predict(client, newcomer_match)

        matrix = body["score_probability_matrix"]
        assert isinstance(matrix, list)
        total = sum(sum(row) for row in matrix)
        assert total == pytest.approx(1.0, abs=1e-5)
        assert body["most_probable_score"]["probability"] > 0

    def test_a_team_with_no_history_starts_from_the_default_rating(
        self, client: TestClient, newcomer_match: int, active_model: ModelVersion
    ) -> None:
        """Both sides are unrated, so the only thing separating them is the home
        advantage the rating system adds — not a guess about either club."""
        from app.features.config import FeatureConfig

        body = predict(client, newcomer_match)

        explanation = client.get(f"/predictions/{body['id']}/explanation").json()
        elo = [
            item
            for target in explanation["targets"]
            for item in target["contributions"]
            if item["feature"] == "elo_diff"
        ]
        assert elo, "the Elo model explains its one feature"
        assert {item["value"] for item in elo} == {FeatureConfig().elo_home_advantage}

    def test_two_different_unknown_fixtures_get_the_same_answer(
        self,
        client: TestClient,
        newcomer_match: int,
        second_newcomer_match: int,
        active_model: ModelVersion,
    ) -> None:
        """Nothing distinguishes one pair of unknown clubs from another, so the
        model must not appear to know something about either."""
        first = predict(client, newcomer_match)
        second = predict(client, second_newcomer_match)

        assert probabilities(second) == pytest.approx(probabilities(first), abs=1e-9)

    def test_an_unknown_team_name_is_refused_rather_than_guessed(
        self, client: TestClient, league: dict[str, int], active_model: ModelVersion
    ) -> None:
        response = client.post(
            "/predictions/match",
            json={"home_team": "Club That Does Not Exist", "away_team": "Arsenal"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
