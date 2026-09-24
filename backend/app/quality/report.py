"""Data-quality report: what is stored, what is missing, and what is questionable.

Every number comes from a query over the stored data. Nothing is estimated.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Select, and_, case, distinct, exists, func, select
from sqlalchemy.orm import Session

from app.models import (
    Competition,
    DataQualityIssue,
    DataSource,
    Match,
    Season,
    Team,
    TeamMatchStatistics,
)
from app.models.enums import IssueType, MatchStatus

DUPLICATE_ISSUES = (
    IssueType.DUPLICATE_MATCH,
    IssueType.DUPLICATE_PROVIDER_ID,
    IssueType.POSSIBLE_DUPLICATE_TEAM,
)


@dataclass
class Coverage:
    label: str
    seasons: int
    matches: int
    finished: int
    with_statistics: int
    with_xg: int
    first_kickoff: datetime | None
    last_kickoff: datetime | None

    @property
    def statistics_pct(self) -> float | None:
        return round(100 * self.with_statistics / self.finished, 1) if self.finished else None

    @property
    def xg_pct(self) -> float | None:
        return round(100 * self.with_xg / self.finished, 1) if self.finished else None


@dataclass
class QualityReport:
    generated_at: datetime
    totals: dict[str, int]
    missing_statistics: dict[str, int]
    duplicates: dict[str, int]
    incomplete: dict[str, int]
    open_issues: dict[str, dict[str, int]]
    by_competition: list[Coverage] = field(default_factory=list)
    by_season: list[Coverage] = field(default_factory=list)

    def to_json(self) -> str:
        def encode(value: Any) -> Any:
            return value.isoformat() if isinstance(value, datetime) else str(value)

        data = asdict(self)
        for key in ("by_competition", "by_season"):
            for row, coverage in zip(data[key], getattr(self, key), strict=True):
                row["statistics_pct"] = coverage.statistics_pct
                row["xg_pct"] = coverage.xg_pct
        return json.dumps(data, indent=2, default=encode)


def _count(session: Session, query: Select[Any]) -> int:
    return int(session.scalar(select(func.count()).select_from(query.subquery())) or 0)


def _has_statistics() -> Any:
    return exists().where(TeamMatchStatistics.match_id == Match.id)


def _has_xg() -> Any:
    return exists().where(
        TeamMatchStatistics.match_id == Match.id,
        TeamMatchStatistics.expected_goals.is_not(None),
    )


def _coverage_query(*group: Any) -> Select[Any]:
    finished = Match.status == MatchStatus.FINISHED
    return select(
        *group,
        func.count(distinct(Match.season_id)),
        func.count(Match.id),
        func.count(case((finished, 1))),
        func.count(case((and_(finished, _has_statistics()), 1))),
        func.count(case((and_(finished, _has_xg()), 1))),
        func.min(Match.kickoff_at),
        func.max(Match.kickoff_at),
    )


def build_report(session: Session, now: datetime) -> QualityReport:
    finished = select(Match.id).where(Match.status == MatchStatus.FINISHED)
    stats_teams = (
        select(func.count(distinct(TeamMatchStatistics.team_id)))
        .where(TeamMatchStatistics.match_id == Match.id)
        .scalar_subquery()
    )
    totals = {
        "competitions": _count(session, select(Competition.id)),
        "seasons": _count(session, select(Season.id)),
        "teams": _count(session, select(Team.id)),
        "matches": _count(session, select(Match.id)),
        "finished_matches": _count(session, finished),
        "team_statistics_rows": _count(session, select(TeamMatchStatistics.id)),
        "data_sources": _count(session, select(DataSource.id)),
    }
    missing_statistics = {
        "finished_matches_without_statistics": _count(session, finished.where(~_has_statistics())),
        "finished_matches_with_one_team_only": _count(session, finished.where(stats_teams == 1)),
        "finished_matches_without_xg": _count(session, finished.where(~_has_xg())),
    }
    open_issue = DataQualityIssue.resolved_at.is_(None)
    duplicates = {
        issue_type.value: _count(
            session,
            select(DataQualityIssue.id).where(
                open_issue, DataQualityIssue.issue_type == issue_type
            ),
        )
        for issue_type in DUPLICATE_ISSUES
    }
    finished_rows = Match.status == MatchStatus.FINISHED
    incomplete = {
        "finished_matches_without_half_time_score": _count(
            session, select(Match.id).where(finished_rows, Match.home_half_time_score.is_(None))
        ),
        "matches_without_venue": _count(session, select(Match.id).where(Match.venue.is_(None))),
        "matches_without_referee": _count(session, select(Match.id).where(Match.referee.is_(None))),
        "matches_without_matchday_or_stage": _count(
            session, select(Match.id).where(Match.matchday.is_(None), Match.stage.is_(None))
        ),
        "past_matches_still_scheduled": _count(
            session,
            select(Match.id).where(Match.status == MatchStatus.SCHEDULED, Match.kickoff_at < now),
        ),
        "seasons_without_dates": _count(
            session,
            select(Season.id).where((Season.start_date.is_(None)) | (Season.end_date.is_(None))),
        ),
        "teams_without_country": _count(session, select(Team.id).where(Team.country.is_(None))),
        "statistics_rows_without_shots": _count(
            session,
            select(TeamMatchStatistics.id).where(TeamMatchStatistics.shots.is_(None)),
        ),
        "statistics_rows_without_possession": _count(
            session,
            select(TeamMatchStatistics.id).where(TeamMatchStatistics.possession.is_(None)),
        ),
    }
    open_issues: dict[str, dict[str, int]] = {}
    for issue_type, severity, count in session.execute(
        select(DataQualityIssue.issue_type, DataQualityIssue.severity, func.count())
        .where(open_issue)
        .group_by(DataQualityIssue.issue_type, DataQualityIssue.severity)
        .order_by(DataQualityIssue.issue_type)
    ):
        open_issues.setdefault(issue_type.value, {})[severity.value] = count

    by_competition = [
        Coverage(f"{name} ({country or 'international'})", *values)
        for name, country, *values in session.execute(
            _coverage_query(Competition.name, Competition.country)
            .join(Competition, Competition.id == Match.competition_id)
            .group_by(Competition.id, Competition.name, Competition.country)
            .order_by(Competition.name)
        )
    ]
    by_season = [
        Coverage(f"{competition} {season}", *values)
        for competition, season, *values in session.execute(
            _coverage_query(Competition.name, Season.name)
            .join(Season, Season.id == Match.season_id)
            .join(Competition, Competition.id == Match.competition_id)
            .group_by(Competition.name, Season.id, Season.name, Season.start_date)
            .order_by(Competition.name, Season.start_date, Season.name)
        )
    ]
    return QualityReport(
        generated_at=now,
        totals=totals,
        missing_statistics=missing_statistics,
        duplicates=duplicates,
        incomplete=incomplete,
        open_issues=open_issues,
        by_competition=by_competition,
        by_season=by_season,
    )


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def _day(value: datetime | None) -> str:
    return "-" if value is None else f"{value:%Y-%m-%d}"


def _section(title: str, values: dict[str, int]) -> list[str]:
    lines = [title]
    lines += [f"  {name.replace('_', ' '):<44}{count:>8}" for name, count in values.items()]
    return [*lines, ""]


def _coverage_table(title: str, rows: list[Coverage]) -> list[str]:
    header = (
        f"  {'':<40}{'seasons':>8}{'matches':>9}{'finished':>9}"
        f"{'stats':>8}{'xG':>8}  {'first':<11}{'last':<11}"
    )
    lines = [title, header]
    for row in rows:
        lines.append(
            f"  {row.label[:40]:<40}{row.seasons:>8}{row.matches:>9}{row.finished:>9}"
            f"{_pct(row.statistics_pct):>8}{_pct(row.xg_pct):>8}"
            f"  {_day(row.first_kickoff):<11}{_day(row.last_kickoff):<11}"
        )
    if not rows:
        lines.append("  (no matches)")
    return [*lines, ""]


def render_text(report: QualityReport) -> str:
    lines = [f"Data-quality report - {report.generated_at:%Y-%m-%d %H:%M} UTC", ""]
    lines += _section("Totals", report.totals)
    lines += _section("Missing statistics", report.missing_statistics)
    lines += _section("Duplicate records (open issues)", report.duplicates)
    lines += _section("Incomplete records", report.incomplete)
    lines.append("Open data-quality issues (by type and severity)")
    if report.open_issues:
        for issue_type, severities in report.open_issues.items():
            detail = ", ".join(f"{sev}={n}" for sev, n in sorted(severities.items()))
            lines.append(f"  {issue_type.replace('_', ' '):<44}{detail}")
    else:
        lines.append("  none")
    lines.append("")
    lines += _coverage_table(
        "Coverage by competition (stats/xG = % of finished matches)", report.by_competition
    )
    lines += _coverage_table("Coverage by season", report.by_season)
    return "\n".join(lines)
