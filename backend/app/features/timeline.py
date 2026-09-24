"""Point-in-time access to match history: the core defence against data leakage.

A `Timeline` holds finished matches ordered by the moment their result became
known (`available_at` = kickoff + result delay). `advance(cutoff)` feeds consumers
(Elo, form indexes, …) every match known by `cutoff`, and nothing else. It never
moves backwards, and it refuses to proceed if the target match itself would be
included. Feature code only ever reads consumer state, so it cannot see the
future: that includes future matches, post-match statistics, and the target's own
result.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from sqlalchemy import and_, select
from sqlalchemy.orm import Session, aliased

from app.models import (
    Competition,
    DataQualityIssue,
    Match,
    Season,
    TeamMatchStatistics,
)
from app.models.enums import CompetitionType, IssueEntity, IssueSeverity, MatchStatus


class LeakageError(RuntimeError):
    """Raised when data from at or after the cutoff would reach a feature."""


@dataclass(frozen=True)
class PastMatch:
    """A finished match as the feature code sees it."""

    match_id: int
    competition_id: int
    season_id: int
    kickoff_at: datetime
    available_at: datetime
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    home_xg: float | None
    away_xg: float | None


@dataclass(frozen=True)
class Target:
    """A match to compute features for, as of `cutoff` (≤ kickoff)."""

    match_id: int
    competition_id: int
    competition_type: CompetitionType
    season_id: int
    season_start: date | None
    season_end: date | None
    kickoff_at: datetime
    cutoff: datetime
    home_team_id: int
    away_team_id: int
    matchday: int | None

    def __post_init__(self) -> None:
        if self.cutoff > self.kickoff_at:
            raise LeakageError(
                f"cutoff {self.cutoff} is after kickoff {self.kickoff_at} of match {self.match_id}"
            )


class Consumer(Protocol):
    def add(self, match: PastMatch) -> None: ...


class Timeline:
    def __init__(self, matches: Iterable[PastMatch], consumers: Sequence[Consumer]) -> None:
        self._matches = sorted(matches, key=lambda m: (m.available_at, m.match_id))
        self._consumers = consumers
        self._position = 0
        self._seen: set[int] = set()
        self.cutoff: datetime | None = None

    def advance(self, cutoff: datetime) -> None:
        """Feed every match whose result is known at `cutoff` (inclusive)."""
        if self.cutoff is not None and cutoff < self.cutoff:
            raise LeakageError(f"timeline cannot move back from {self.cutoff} to {cutoff}")
        while (
            self._position < len(self._matches)
            and self._matches[self._position].available_at <= cutoff
        ):
            match = self._matches[self._position]
            for consumer in self._consumers:
                consumer.add(match)
            self._seen.add(match.match_id)
            self._position += 1
        self.cutoff = cutoff

    def prepare(self, target: Target) -> None:
        """Advance to the target's cutoff and prove its own result is unseen."""
        self.advance(target.cutoff)
        if target.match_id in self._seen:
            raise LeakageError(f"match {target.match_id}'s own result is in the history")

    @property
    def latest_available(self) -> datetime | None:
        if self._position == 0:
            return None
        return self._matches[self._position - 1].available_at


def _flagged_matches(session: Session) -> set[int]:
    """Matches with open error-level data-quality issues; kept out of history."""
    return set(
        session.scalars(
            select(DataQualityIssue.entity_id).where(
                DataQualityIssue.entity_type == IssueEntity.MATCH,
                DataQualityIssue.severity == IssueSeverity.ERROR,
                DataQualityIssue.resolved_at.is_(None),
                DataQualityIssue.entity_id.is_not(None),
            )
        )
    )


def load_history(
    session: Session, result_delay: timedelta, *, exclude_flagged: bool = True
) -> list[PastMatch]:
    """All finished matches with a score, with each team's xG where recorded.

    When several sources report xG for one match, the lowest data source id wins,
    so the choice is deterministic.
    """
    home_stats, away_stats = aliased(TeamMatchStatistics), aliased(TeamMatchStatistics)
    rows = session.execute(
        select(
            Match.id,
            Match.competition_id,
            Match.season_id,
            Match.kickoff_at,
            Match.home_team_id,
            Match.away_team_id,
            Match.home_score,
            Match.away_score,
            home_stats.expected_goals,
            away_stats.expected_goals,
        )
        .outerjoin(
            home_stats,
            and_(home_stats.match_id == Match.id, home_stats.team_id == Match.home_team_id),
        )
        .outerjoin(
            away_stats,
            and_(
                away_stats.match_id == Match.id,
                away_stats.team_id == Match.away_team_id,
                away_stats.data_source_id == home_stats.data_source_id,
            ),
        )
        .where(
            Match.status == MatchStatus.FINISHED,
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
        )
        .order_by(Match.id, home_stats.data_source_id)
    ).all()
    flagged = _flagged_matches(session) if exclude_flagged else set()
    history: dict[int, PastMatch] = {}
    for match_id, comp, season, kickoff, home, away, hg, ag, hxg, axg in rows:
        if match_id in history or match_id in flagged:
            continue  # first row per match = lowest data source id
        history[match_id] = PastMatch(
            match_id=match_id,
            competition_id=comp,
            season_id=season,
            kickoff_at=kickoff,
            available_at=kickoff + result_delay,
            home_team_id=home,
            away_team_id=away,
            home_goals=hg,
            away_goals=ag,
            home_xg=float(hxg) if hxg is not None else None,
            away_xg=float(axg) if axg is not None else None,
        )
    return list(history.values())


def load_targets(
    session: Session,
    *,
    match_ids: Sequence[int] | None = None,
    competition_id: int | None = None,
    season_id: int | None = None,
    finished_only: bool = False,
    lead_time: timedelta = timedelta(0),
) -> list[Target]:
    """Matches to build features for, each with cutoff = kickoff - lead_time."""
    query = (
        select(Match, Competition.type, Season.start_date, Season.end_date)
        .join(Competition, Competition.id == Match.competition_id)
        .join(Season, Season.id == Match.season_id)
        .order_by(Match.kickoff_at, Match.id)
    )
    if match_ids is not None:
        query = query.where(Match.id.in_(match_ids))
    if competition_id is not None:
        query = query.where(Match.competition_id == competition_id)
    if season_id is not None:
        query = query.where(Match.season_id == season_id)
    if finished_only:
        query = query.where(Match.status == MatchStatus.FINISHED)
    return [
        Target(
            match_id=match.id,
            competition_id=match.competition_id,
            competition_type=competition_type,
            season_id=match.season_id,
            season_start=start,
            season_end=end,
            kickoff_at=match.kickoff_at,
            cutoff=match.kickoff_at - lead_time,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            matchday=match.matchday,
        )
        for match, competition_type, start, end in session.execute(query)
    ]
