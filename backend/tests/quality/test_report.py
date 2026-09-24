"""The data-quality report and the CLI commands, on a real backfill (over fixtures)."""

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.ingestion.__main__ import main
from app.ingestion.history import HistoricalImport
from app.ingestion.sync import ProviderSync
from app.quality.report import build_report, render_text
from tests.providers.fake_api import FakeApiFootball

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 22, tzinfo=UTC)


@pytest.fixture
def imported(session_factory: sessionmaker[Session]) -> sessionmaker[Session]:
    """Two Premier League seasons: 2022/23 without statistics, 2023/24 with 9 of 10."""
    HistoricalImport(ProviderSync(FakeApiFootball().provider(), session_factory)).run("39")
    return session_factory


def test_totals_and_missing_statistics(imported: sessionmaker[Session]) -> None:
    with imported() as session:
        report = build_report(session, NOW)

    assert report.totals == {
        "competitions": 1,
        "seasons": 2,
        "teams": 23,
        "matches": 20,
        "finished_matches": 20,
        "team_statistics_rows": 18,
        "data_sources": 1,
    }
    assert report.missing_statistics == {
        "finished_matches_without_statistics": 11,
        "finished_matches_with_one_team_only": 0,
        "finished_matches_without_xg": 11,
    }
    assert report.duplicates == {
        "duplicate_match": 0,
        "duplicate_provider_id": 0,
        "possible_duplicate_team": 0,
    }
    assert report.open_issues == {"missing_statistics": {"info": 11}}


def test_incomplete_records(imported: sessionmaker[Session]) -> None:
    with imported() as session:
        incomplete = build_report(session, NOW).incomplete

    assert incomplete["matches_without_referee"] == 20  # fixtures carry no referee
    assert incomplete["matches_without_venue"] == 0
    assert incomplete["finished_matches_without_half_time_score"] == 0
    assert incomplete["past_matches_still_scheduled"] == 0
    assert incomplete["seasons_without_dates"] == 0


def test_coverage_by_competition_and_season(imported: sessionmaker[Session]) -> None:
    with imported() as session:
        report = build_report(session, NOW)

    (competition,) = report.by_competition
    assert competition.label == "Premier League (England)"
    assert (competition.seasons, competition.matches, competition.finished) == (2, 20, 20)
    assert (competition.statistics_pct, competition.xg_pct) == (45.0, 45.0)
    assert competition.first_kickoff == datetime(2022, 8, 5, 19, tzinfo=UTC)
    assert competition.last_kickoff == datetime(2023, 8, 14, 19, tzinfo=UTC)

    by_label = {row.label: row for row in report.by_season}
    assert list(by_label) == ["Premier League 2022/2023", "Premier League 2023/2024"]
    assert by_label["Premier League 2022/2023"].statistics_pct == 0.0
    assert by_label["Premier League 2023/2024"].statistics_pct == 90.0


def test_empty_database_report(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        report = build_report(session, NOW)

    assert set(report.totals.values()) == {0}
    assert report.by_competition == []
    assert "(no matches)" in render_text(report)


def test_text_and_json_rendering(imported: sessionmaker[Session]) -> None:
    with imported() as session:
        report = build_report(session, NOW)

    text = render_text(report)
    for heading in (
        "Totals",
        "Missing statistics",
        "Duplicate records",
        "Incomplete records",
        "Open data-quality issues",
        "Coverage by competition",
        "Coverage by season",
    ):
        assert heading in text
    assert "Premier League 2023/2024" in text
    assert text.isascii()

    data = json.loads(report.to_json())
    assert data["totals"]["matches"] == 20
    assert data["by_season"][1]["statistics_pct"] == 90.0


class TestCommands:
    def test_backfill_audit_and_report(
        self, session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
    ) -> None:
        provider = FakeApiFootball().provider()

        assert (
            main(
                ["backfill", "--competition", "39"],
                provider=provider,
                session_factory=session_factory,
            )
            == 0
        )
        assert "Competition 39: 2 season(s) selected" in capsys.readouterr().out

        assert main(["audit"], session_factory=session_factory) == 0
        assert "Audit found 0 issue(s)" in capsys.readouterr().out

        assert main(["report", "--json"], session_factory=session_factory) == 0
        assert json.loads(capsys.readouterr().out)["totals"]["matches"] == 20

    def test_backfill_year_filter(
        self, session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
    ) -> None:
        provider = FakeApiFootball().provider()

        code = main(
            ["backfill", "--competition", "39", "--from", "2023", "--no-statistics"],
            provider=provider,
            session_factory=session_factory,
        )

        assert code == 0
        output = capsys.readouterr().out
        assert "1 season(s) selected" in output
        assert "team_statistics" not in output

    def test_report_needs_no_provider_credentials(
        self, session_factory: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["report"], session_factory=session_factory) == 0
        assert "Totals" in capsys.readouterr().out
