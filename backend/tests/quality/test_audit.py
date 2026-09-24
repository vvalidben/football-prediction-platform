"""The database audit finds cross-record problems and never changes football data."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.models import (
    Competition,
    DataQualityIssue,
    DataSource,
    Match,
    Season,
    Team,
    TeamMatchStatistics,
)
from app.models.enums import CompetitionType, IssueType, MatchStatus
from app.quality.audit import run_audit

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 22, tzinfo=UTC)
FOOTBALL_TABLES = (Competition, Season, Team, Match, TeamMatchStatistics)


def kickoff(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}+00:00")


@pytest.fixture
def seeded(session_factory: sessionmaker[Session]) -> dict[str, int]:
    """A small league with one example of each problem the audit looks for."""
    with session_factory() as session, session.begin():
        source = DataSource(code="audit_test", name="Audit test")
        league = Competition(name="Test League", type=CompetitionType.LEAGUE, country="Spain")
        season = Season(
            competition=league,
            name="2023/2024",
            start_date=date(2023, 8, 11),
            end_date=date(2024, 5, 19),
        )
        Season(
            competition=league,
            name="2024",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )  # overlaps 2023/2024
        a, b, c = (Team(name=n, country="Spain") for n in ("Alpha", "Beta", "Gamma"))
        atletico = Team(name="Atlético Madrid", country="Spain")
        atletico_again = Team(name="Atletico Madrid FC", country="Spain")
        session.add_all([source, league, a, b, c, atletico, atletico_again])

        def match(home: Team, away: Team, when: str, **values: object) -> Match:
            row = Match(
                competition=league,
                season=season,
                home_team=home,
                away_team=away,
                kickoff_at=kickoff(when),
                **values,
            )
            session.add(row)
            return row

        finished = {"status": MatchStatus.FINISHED}
        first = match(
            a,
            b,
            "2023-09-01T15:00",
            home_score=1,
            away_score=0,
            home_half_time_score=0,
            away_half_time_score=0,
            **finished,
        )
        near_duplicate = match(a, b, "2023-09-02T12:00", status=MatchStatus.SCHEDULED)
        clash = match(
            a,
            c,
            "2023-09-01T20:00",
            home_score=2,
            away_score=1,
            home_half_time_score=3,
            away_half_time_score=0,
            **finished,
        )
        outside = match(b, c, "2024-12-01T15:00", home_score=0, away_score=0, **finished)
        session.flush()
        session.add_all(
            [
                TeamMatchStatistics(
                    match_id=first.id, team_id=a.id, data_source=source, possession=Decimal(60)
                ),
                TeamMatchStatistics(
                    match_id=first.id, team_id=b.id, data_source=source, possession=Decimal(50)
                ),
                TeamMatchStatistics(match_id=first.id, team_id=c.id, data_source=source),
            ]
        )
        session.flush()
        return {
            "first": first.id,
            "near_duplicate": near_duplicate.id,
            "clash": clash.id,
            "outside": outside.id,
            "atletico_again": atletico_again.id,
        }


def counts(factory: sessionmaker[Session]) -> dict[str, int]:
    with factory() as session:
        return {
            m.__tablename__: session.scalar(select(func.count()).select_from(m)) or 0
            for m in FOOTBALL_TABLES
        }


def test_finds_each_kind_of_problem(
    seeded: dict[str, int], session_factory: sessionmaker[Session]
) -> None:
    result = run_audit(session_factory, now=NOW)

    by_key = {(i.issue_type, i.key.split(":")[0], i.entity_id) for i in result.issues}
    ids = seeded
    assert (IssueType.DUPLICATE_MATCH, "near_duplicate_of", ids["near_duplicate"]) in by_key
    assert (IssueType.DUPLICATE_MATCH, "team_clash_with", ids["clash"]) in by_key
    assert (IssueType.IMPOSSIBLE_SCORE, "half_time_exceeds_full_time", ids["clash"]) in by_key
    assert (IssueType.INVALID_DATE, "outside_season", ids["outside"]) in by_key
    assert (IssueType.INCOMPLETE_RECORD, "stale_scheduled_status", ids["near_duplicate"]) in by_key
    assert (IssueType.INCONSISTENT_STATISTICS, "possession_sum", ids["first"]) in by_key
    assert (IssueType.POSSIBLE_DUPLICATE_TEAM, "same_name_as", ids["atletico_again"]) in by_key
    kinds = {(issue_type, key) for issue_type, key, _ in by_key}
    assert (IssueType.INCONSISTENT_STATISTICS, "team_not_in_match") in kinds
    assert (IssueType.INCONSISTENT_COMPETITION, "overlaps") in kinds


def test_audit_changes_no_football_data(
    seeded: dict[str, int], session_factory: sessionmaker[Session]
) -> None:
    before = counts(session_factory)

    run_audit(session_factory, now=NOW)

    assert counts(session_factory) == before


def test_rerun_updates_open_issues_and_resolves_fixed_ones(
    seeded: dict[str, int], session_factory: sessionmaker[Session]
) -> None:
    first = run_audit(session_factory, now=NOW)
    with session_factory() as session:
        stored_after_first = session.scalar(select(func.count()).select_from(DataQualityIssue))

    # Correct the possession figures, then audit again.
    with session_factory() as session, session.begin():
        session.execute(
            update(TeamMatchStatistics)
            .where(TeamMatchStatistics.possession == Decimal(50))
            .values(possession=Decimal(40))
        )
    second = run_audit(session_factory, now=NOW)

    assert len(second.issues) == len(first.issues) - 1
    assert second.resolved == 1
    with session_factory() as session:
        total = session.scalar(select(func.count()).select_from(DataQualityIssue))
        resolved = session.scalars(
            select(DataQualityIssue).where(DataQualityIssue.resolved_at.is_not(None))
        ).all()
        repeated = session.scalar(select(func.max(DataQualityIssue.occurrences)))
    assert total == stored_after_first  # nothing deleted, nothing duplicated
    assert [r.issue_type for r in resolved] == [IssueType.INCONSISTENT_STATISTICS]
    assert repeated == 2


def test_clean_database_has_no_issues(session_factory: sessionmaker[Session]) -> None:
    assert run_audit(session_factory, now=NOW).issues == []
