"""Indexes over known results: per team, per competition and per pairing.

Each index is a `Timeline` consumer, filled only with matches known at the cutoff.
The feature functions in `builder` read from these indexes and nothing else.
"""

import bisect
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.features.timeline import PastMatch


@dataclass(frozen=True)
class TeamMatch:
    """One past match from one team's point of view."""

    match_id: int
    kickoff_at: datetime
    competition_id: int
    season_id: int
    opponent_id: int
    is_home: bool
    goals_for: int
    goals_against: int
    xg_for: float | None
    xg_against: float | None

    @property
    def points(self) -> int:
        if self.goals_for > self.goals_against:
            return 3
        return 1 if self.goals_for == self.goals_against else 0


class TeamIndex:
    """Each team's known matches, oldest first."""

    def __init__(self) -> None:
        self._matches: dict[int, list[TeamMatch]] = defaultdict(list)

    def add(self, match: PastMatch) -> None:
        for team, opponent, is_home, gf, ga, xgf, xga in (
            (
                match.home_team_id,
                match.away_team_id,
                True,
                match.home_goals,
                match.away_goals,
                match.home_xg,
                match.away_xg,
            ),
            (
                match.away_team_id,
                match.home_team_id,
                False,
                match.away_goals,
                match.home_goals,
                match.away_xg,
                match.home_xg,
            ),
        ):
            self._matches[team].append(
                TeamMatch(
                    match_id=match.match_id,
                    kickoff_at=match.kickoff_at,
                    competition_id=match.competition_id,
                    season_id=match.season_id,
                    opponent_id=opponent,
                    is_home=is_home,
                    goals_for=gf,
                    goals_against=ga,
                    xg_for=xgf,
                    xg_against=xga,
                )
            )

    def matches(self, team_id: int) -> list[TeamMatch]:
        return self._matches.get(team_id, [])

    def last(self, team_id: int, n: int, *, is_home: bool | None = None) -> list[TeamMatch]:
        matches = self.matches(team_id)
        if is_home is not None:
            matches = [m for m in matches if m.is_home is is_home]
        return matches[-n:]


@dataclass(frozen=True)
class CompetitionAverages:
    matches: int
    home_win_rate: float | None
    draw_rate: float | None
    home_goals: float | None
    away_goals: float | None


class CompetitionIndex:
    """Results per competition, for home advantage and goal-rate baselines."""

    def __init__(self) -> None:
        self._kickoffs: dict[int | None, list[datetime]] = defaultdict(list)
        self._results: dict[int | None, list[tuple[int, int]]] = defaultdict(list)

    def add(self, match: PastMatch) -> None:
        for key in (match.competition_id, None):  # None = all competitions
            self._kickoffs[key].append(match.kickoff_at)
            self._results[key].append((match.home_goals, match.away_goals))

    def averages(
        self, competition_id: int | None, before: datetime, window: timedelta
    ) -> CompetitionAverages:
        kickoffs = self._kickoffs.get(competition_id, [])
        start = bisect.bisect_left(kickoffs, before - window)
        end = bisect.bisect_left(kickoffs, before)
        results = self._results.get(competition_id, [])[start:end]
        n = len(results)
        if n == 0:
            return CompetitionAverages(0, None, None, None, None)
        return CompetitionAverages(
            matches=n,
            home_win_rate=sum(h > a for h, a in results) / n,
            draw_rate=sum(h == a for h, a in results) / n,
            home_goals=sum(h for h, _ in results) / n,
            away_goals=sum(a for _, a in results) / n,
        )


class PairIndex:
    """Meetings between two teams, oldest first, regardless of venue."""

    def __init__(self) -> None:
        self._meetings: dict[frozenset[int], list[PastMatch]] = defaultdict(list)

    def add(self, match: PastMatch) -> None:
        self._meetings[frozenset((match.home_team_id, match.away_team_id))].append(match)

    def meetings(self, team_a: int, team_b: int) -> list[PastMatch]:
        return self._meetings.get(frozenset((team_a, team_b)), [])
