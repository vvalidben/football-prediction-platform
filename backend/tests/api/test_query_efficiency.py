"""Listing endpoints must not issue one query per row.

The fixtures elsewhere use six teams and two competitions, which hides lazy
loading: the identity map satisfies the repeats. These tests use enough distinct
entities that an N+1 shows up as what it is.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import Competition, Match, Season, Team
from app.models.enums import CompetitionType, MatchStatus

pytestmark = pytest.mark.integration

#: Distinct teams in the fixture below: enough that per-row loading is visible.
TEAM_COUNT = 24


class QueryCounter:
    """Counts the SQL statements a block of work issues."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.statements: list[str] = []

    def __enter__(self) -> "QueryCounter":
        event.listen(self._engine, "before_cursor_execute", self._record)
        return self

    def __exit__(self, *exc: object) -> None:
        event.remove(self._engine, "before_cursor_execute", self._record)

    def _record(
        self, conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        self.statements.append(statement.split("\n")[0][:80])

    def __len__(self) -> int:
        return len(self.statements)


@pytest.fixture
def crowded_league(session_factory: sessionmaker[Session]) -> dict[str, int]:
    """Many teams across two competitions, so lazy loading cannot be masked."""
    with session_factory() as session, session.begin():
        competitions = [
            Competition(name="Audit League", type=CompetitionType.LEAGUE, country="Nowhere"),
            Competition(name="Audit Cup", type=CompetitionType.CUP, country="Nowhere"),
        ]
        teams = [Team(name=f"Audit Team {index}", country="Nowhere") for index in range(TEAM_COUNT)]
        session.add_all([*competitions, *teams])
        session.flush()
        seasons = [
            Season(
                competition_id=competition.id,
                name="2026/2027",
                start_date=datetime(2026, 8, 1, tzinfo=UTC).date(),
                end_date=datetime(2027, 5, 31, tzinfo=UTC).date(),
            )
            for competition in competitions
        ]
        session.add_all(seasons)
        session.flush()
        kickoff = datetime.now(UTC) + timedelta(days=1)
        for index in range(TEAM_COUNT // 2):
            which = index % 2
            session.add(
                Match(
                    competition_id=competitions[which].id,
                    season_id=seasons[which].id,
                    home_team_id=teams[index * 2].id,
                    away_team_id=teams[index * 2 + 1].id,
                    kickoff_at=kickoff + timedelta(hours=index),
                    status=MatchStatus.SCHEDULED,
                )
            )
        session.flush()
        return {"competition": competitions[0].id, "team": teams[0].id}


def counts_for(client: TestClient, engine: Engine, path: str, **params: Any) -> int:
    with QueryCounter(engine) as counter:
        response = client.get(path, params=params)
        assert response.status_code == 200, response.text
        assert response.json()["items"], "the fixture should produce rows to serialise"
    return len(counter)


class TestListingsDoNotScaleWithRows:
    def test_listing_matches_is_a_fixed_number_of_queries(
        self, client: TestClient, crowded_league: dict[str, int], db_engine: Engine
    ) -> None:
        """Every row names a competition, a season and two teams. Loading those per
        row is four extra queries a match; joining them is none."""
        few = counts_for(client, db_engine, "/matches", limit=2)
        many = counts_for(client, db_engine, "/matches", limit=12)

        assert many == few, f"{many} queries for 12 rows vs {few} for 2: loading per row"
        assert many <= 4, f"{many} queries to list matches"

    def test_upcoming_matches_is_a_fixed_number_of_queries(
        self, client: TestClient, crowded_league: dict[str, int], db_engine: Engine
    ) -> None:
        few = counts_for(client, db_engine, "/matches/upcoming", limit=2)
        many = counts_for(client, db_engine, "/matches/upcoming", limit=12)

        assert many == few, f"{many} queries for 12 rows vs {few} for 2"
        assert many <= 4

    def test_listing_teams_is_a_fixed_number_of_queries(
        self, client: TestClient, crowded_league: dict[str, int], db_engine: Engine
    ) -> None:
        few = counts_for(client, db_engine, "/teams", limit=2)
        many = counts_for(client, db_engine, "/teams", limit=20)

        assert many == few
        assert many <= 4

    def test_the_history_endpoint_already_joins_what_it_needs(
        self, client: TestClient, crowded_league: dict[str, int], db_engine: Engine
    ) -> None:
        """It was written with explicit joins; this keeps it that way."""
        with QueryCounter(db_engine) as counter:
            assert client.get("/predictions/history", params={"limit": 20}).status_code == 200

        assert len(counter) <= 6, counter.statements


class TestMatchDetail:
    def test_one_match_with_its_predictions_is_a_fixed_number_of_queries(
        self,
        client: TestClient,
        league: dict[str, int],
        active_model: Any,
        db_engine: Engine,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Each stored prediction names its model version. Loading that per
        prediction is one query each."""
        match_id = league["finished_match"]
        for _ in range(3):
            client.post("/predictions/match", json={"match_id": match_id, "refresh": True})
        with session_factory() as session:
            stored = session.scalars(select(Match).where(Match.id == match_id).limit(1)).one()
            assert stored is not None

        with QueryCounter(db_engine) as counter:
            response = client.get(f"/matches/{match_id}")

        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 3
        assert len(counter) <= 5, counter.statements
