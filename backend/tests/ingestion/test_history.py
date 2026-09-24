"""Historical backfill through the API-Football adapter into PostgreSQL."""

import copy
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.ingestion.history import HistoricalImport
from app.ingestion.sync import ProviderSync
from app.models import (
    DataQualityIssue,
    Match,
    MatchExternalId,
    Season,
    Team,
    TeamMatchStatistics,
)
from app.models.enums import IngestionStatus, IssueEntity, IssueType
from tests.providers.fake_api import FakeApiFootball, body, envelope, load_fixture

pytestmark = pytest.mark.integration


def count(factory: sessionmaker[Session], model: type) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def open_issues(factory: sessionmaker[Session], issue_type: IssueType) -> list[DataQualityIssue]:
    with factory() as session:
        return list(
            session.scalars(
                select(DataQualityIssue).where(
                    DataQualityIssue.issue_type == issue_type,
                    DataQualityIssue.resolved_at.is_(None),
                )
            )
        )


def stats_requests(api: FakeApiFootball) -> list[str]:
    return [r.url.params["ids"] for r in api.requests if "ids" in r.url.params]


@pytest.fixture
def api() -> FakeApiFootball:
    return FakeApiFootball()


@pytest.fixture
def importer(api: FakeApiFootball, session_factory: sessionmaker[Session]) -> HistoricalImport:
    return HistoricalImport(ProviderSync(api.provider(), session_factory))


class TestBackfill:
    def test_imports_every_season_the_provider_offers(
        self, importer: HistoricalImport, session_factory: sessionmaker[Session]
    ) -> None:
        report = importer.run("39")

        assert report.seasons == ["39:2022", "39:2023"]  # oldest first
        assert [(r.job, r.inserted) for r in report.runs] == [
            ("competitions", 1),
            ("seasons", 2),
            ("teams", 20),
            ("matches", 10),
            ("team_statistics", 0),  # 2022 fixtures have no details in the fixtures
            ("teams", 3),  # Burnley, Luton, Sheffield Utd are new in 2023
            ("matches", 10),
            ("team_statistics", 18),  # 9 of 10 fixtures, both teams
        ]
        assert count(session_factory, Season) == 2
        assert count(session_factory, Team) == 23
        assert count(session_factory, Match) == 20
        assert count(session_factory, TeamMatchStatistics) == 18

    def test_exact_kickoff_timestamps_are_preserved(
        self, importer: HistoricalImport, session_factory: sessionmaker[Session]
    ) -> None:
        importer.run("39")

        source = {
            str(item["fixture"]["id"]): datetime.fromisoformat(item["fixture"]["date"])
            for name in (
                "fixtures_league_39_season_2022.json",
                "fixtures_league_39_season_2023.json",
            )
            for item in load_fixture(name)["response"]
        }
        with session_factory() as session:
            rows = session.execute(
                select(MatchExternalId.external_id, Match.kickoff_at).join(
                    Match, Match.id == MatchExternalId.match_id
                )
            ).all()
        stored: dict[str, datetime] = {external_id: kickoff for external_id, kickoff in rows}
        assert stored == source
        assert all(kickoff.utcoffset() == timedelta(0) for kickoff in stored.values())

    def test_missing_statistics_are_recorded_not_invented(
        self, importer: HistoricalImport, session_factory: sessionmaker[Session]
    ) -> None:
        importer.run("39")

        missing = {
            issue.external_id
            for issue in open_issues(session_factory, IssueType.MISSING_STATISTICS)
        }
        # 10 matches of 2022 (no details available) + Man Utd v Wolves in 2023.
        assert len(missing) == 11
        assert "1035046" in missing
        assert all(
            i.entity_type is IssueEntity.MATCH
            for i in open_issues(session_factory, IssueType.MISSING_STATISTICS)
        )

    def test_rerun_is_idempotent_and_does_not_re_request_known_gaps(
        self,
        api: FakeApiFootball,
        importer: HistoricalImport,
        session_factory: sessionmaker[Session],
    ) -> None:
        importer.run("39")
        before = {
            m: count(session_factory, m)
            for m in (Team, Match, TeamMatchStatistics, DataQualityIssue)
        }
        requests_before = len(stats_requests(api))

        report = importer.run("39")

        assert {m: count(session_factory, m) for m in before} == before
        assert all(run.inserted == 0 and run.updated == 0 for run in report.runs)
        assert len(stats_requests(api)) == requests_before  # nothing left worth asking for

    def test_retry_missing_statistics_asks_again_and_resolves_filled_gaps(
        self,
        api: FakeApiFootball,
        importer: HistoricalImport,
        session_factory: sessionmaker[Session],
    ) -> None:
        importer.run("39")
        # The provider has since published statistics for Man Utd (33) v Wolves (39):
        # reuse another fixture's statistics blocks under this fixture's teams.
        filled = copy.deepcopy(api.fixture_details[1035046])
        filled["statistics"] = copy.deepcopy(api.fixture_details[1035045]["statistics"])
        filled["statistics"][0]["team"] = {"id": 33, "name": "Manchester United"}
        filled["statistics"][1]["team"] = {"id": 39, "name": "Wolves"}
        api.fixture_details[1035046] = filled

        importer.run("39", seasons=["39:2023"], retry_missing_statistics=True)

        assert count(session_factory, TeamMatchStatistics) == 20
        still_missing = {
            i.external_id for i in open_issues(session_factory, IssueType.MISSING_STATISTICS)
        }
        assert "1035046" not in still_missing
        with session_factory() as session:
            resolved = session.scalars(
                select(DataQualityIssue).where(
                    DataQualityIssue.external_id == "1035046",
                    DataQualityIssue.issue_type == IssueType.MISSING_STATISTICS,
                )
            ).one()
        assert resolved.resolved_at is not None  # kept, not deleted

    def test_year_range_limits_seasons(
        self, importer: HistoricalImport, session_factory: sessionmaker[Session]
    ) -> None:
        report = importer.run("39", first_year=2023, last_year=2023)

        assert report.seasons == ["39:2023"]
        assert count(session_factory, Match) == 10

    def test_seasons_without_provider_statistics_are_skipped_and_noted(
        self,
        api: FakeApiFootball,
        importer: HistoricalImport,
        session_factory: sessionmaker[Session],
    ) -> None:
        league = copy.deepcopy(load_fixture("leagues_id_39.json")["response"][0])
        league["seasons"][0]["coverage"]["fixtures"]["statistics_fixtures"] = False
        api.serve_json("/leagues", {"id": "39"}, envelope([league]))

        report = importer.run("39")

        assert report.skipped_statistics == ["39:2022"]
        assert all("868" not in ids for ids in stats_requests(api))  # 2022 never requested
        (gap,) = [
            i
            for i in open_issues(session_factory, IssueType.MISSING_STATISTICS)
            if i.entity_type is IssueEntity.SEASON
        ]
        assert gap.external_id == "39:2022"

    def test_quota_exhaustion_stops_the_import(
        self,
        api: FakeApiFootball,
        importer: HistoricalImport,
        session_factory: sessionmaker[Session],
    ) -> None:
        quota = body(
            envelope([], errors={"requests": "You have reached the request limit for the day."})
        )
        api.routes[("/teams", frozenset({("league", "39"), ("season", "2022")}))] = quota

        report = importer.run("39")

        assert report.stopped is not None and "QuotaExceededError" in report.stopped
        assert report.status is IngestionStatus.FAILED
        assert count(session_factory, Match) == 0  # nothing after the stop
