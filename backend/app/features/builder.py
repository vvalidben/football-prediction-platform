"""Builds leak-free feature vectors for target matches.

    builder = FeatureBuilder(session_factory)
    rows = builder.build(targets)      # any order; processed by cutoff

All history-based features are read from consumers fed by one `Timeline`, which
is advanced to each target's cutoff in chronological order. Batch generation
(many targets, one timeline) and single-match generation (one target, fresh
timeline) therefore run the same code on the same data.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session, sessionmaker

from app.features import sources
from app.features.config import FeatureConfig
from app.features.history import CompetitionIndex, PairIndex, TeamIndex, TeamMatch
from app.features.ratings import RatingEngine
from app.features.registry import CONGESTION_DAYS, FEATURE_NAMES, FORM_WINDOWS
from app.features.timeline import (
    LeakageError,
    PastMatch,
    Target,
    Timeline,
    load_history,
)
from app.models.enums import CompetitionType

Value = float | int | None
Features = dict[str, Value]


@dataclass(frozen=True)
class FeatureRow:
    target: Target
    version: str
    values: Features
    #: When the newest result used became known; always ≤ the cutoff.
    latest_information_at: datetime | None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: Value) -> Value:
    """Stable, JSON-friendly numbers (identical across recomputation)."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    return value


class FeatureBuilder:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        config: FeatureConfig | None = None,
        *,
        exclude_flagged: bool = True,
    ) -> None:
        self.session_factory = session_factory
        self.config = config or FeatureConfig()
        self.version = self.config.version()
        self.exclude_flagged = exclude_flagged

    def build(
        self, targets: Sequence[Target], history: Sequence[PastMatch] | None = None
    ) -> list[FeatureRow]:
        """Features for each target, computed as of its cutoff."""
        with self.session_factory() as session:
            if history is None:
                history = load_history(
                    session, self.config.result_delay, exclude_flagged=self.exclude_flagged
                )
            teams, competitions, pairs = TeamIndex(), CompetitionIndex(), PairIndex()
            ratings = RatingEngine(self.config)
            timeline = Timeline(history, [ratings, teams, competitions, pairs])

            rows = []
            for target in sorted(targets, key=lambda t: (t.cutoff, t.match_id)):
                timeline.prepare(target)
                latest = timeline.latest_available
                if latest is not None and latest > target.cutoff:  # defence in depth
                    raise LeakageError(f"history newer than cutoff for match {target.match_id}")
                values: Features = {}
                values.update(self._form(teams, target))
                values.update(self._venue(teams, competitions, target))
                values.update(self._strength(ratings, target))
                values.update(self._context(teams, competitions, target))
                values.update(self._head_to_head(pairs, target))
                values.update(
                    sources.absence_features(
                        session,
                        target,
                        result_delay=self.config.result_delay,
                        trust_provider_timing=self.config.trust_provider_absence_timing,
                    )
                )
                values.update(sources.odds_features(session, target))
                rows.append(
                    FeatureRow(target, self.version, self._finalise(values, target), latest)
                )
            return rows

    def _finalise(self, values: Features, target: Target) -> Features:
        unexpected = set(values) - set(FEATURE_NAMES)
        missing = set(FEATURE_NAMES) - set(values)
        if unexpected or missing:
            raise ValueError(
                f"features for match {target.match_id} do not match the registry: "
                f"unexpected={sorted(unexpected)} missing={sorted(missing)}"
            )
        return {name: _round(values[name]) for name in FEATURE_NAMES}

    # -- feature groups ------------------------------------------------------------

    def _form(self, teams: TeamIndex, target: Target) -> Features:
        values: Features = {}
        for side, team_id in (("home", target.home_team_id), ("away", target.away_team_id)):
            for n in FORM_WINDOWS:
                recent = teams.last(team_id, n)
                prefix = f"{side}_form{n}"
                xg_pairs = [
                    (m.xg_for, m.xg_against)
                    for m in recent
                    if m.xg_for is not None and m.xg_against is not None
                ]
                count = len(recent)
                values[f"{prefix}_matches"] = count
                values[f"{prefix}_goals_for"] = _mean([m.goals_for for m in recent])
                values[f"{prefix}_goals_against"] = _mean([m.goals_against for m in recent])
                values[f"{prefix}_points"] = _mean([m.points for m in recent])
                values[f"{prefix}_win_rate"] = _mean([m.points == 3 for m in recent])
                values[f"{prefix}_draw_rate"] = _mean([m.points == 1 for m in recent])
                values[f"{prefix}_loss_rate"] = _mean([m.points == 0 for m in recent])
                values[f"{prefix}_xg_for"] = _mean([xg for xg, _ in xg_pairs])
                values[f"{prefix}_xg_against"] = _mean([xga for _, xga in xg_pairs])
                values[f"{prefix}_xg_matches"] = len(xg_pairs)
        return values

    def _baseline(self, competitions: CompetitionIndex, target: Target) -> tuple[float, float]:
        """Home and away goals per match for the target's competition (or all
        competitions when it has too little history), shrunk toward a prior so the
        baseline is never zero."""
        config = self.config
        window = config.competition_window
        averages = competitions.averages(target.competition_id, target.kickoff_at, window)
        if averages.matches < config.competition_min_matches:
            averages = competitions.averages(None, target.kickoff_at, window)
        n, m = averages.matches, config.baseline_prior_matches
        home = (n * (averages.home_goals or 0.0) + m * config.baseline_prior_home_goals) / (n + m)
        away = (n * (averages.away_goals or 0.0) + m * config.baseline_prior_away_goals) / (n + m)
        return home, away

    def _venue(self, teams: TeamIndex, competitions: CompetitionIndex, target: Target) -> Features:
        league_home, league_away = self._baseline(competitions, target)
        k = self.config.venue_prior_matches

        def strength(matches: list[TeamMatch], scored: bool, baseline: float) -> float:
            goals = sum(m.goals_for if scored else m.goals_against for m in matches)
            return (goals + k * baseline) / (len(matches) + k) / baseline

        home = teams.last(target.home_team_id, self.config.venue_window, is_home=True)
        away = teams.last(target.away_team_id, self.config.venue_window, is_home=False)
        return {
            "home_attack_strength": strength(home, True, league_home),
            "home_defence_weakness": strength(home, False, league_away),
            "away_attack_strength": strength(away, True, league_away),
            "away_defence_weakness": strength(away, False, league_home),
            "home_venue_matches": len(home),
            "away_venue_matches": len(away),
        }

    def _strength(self, ratings: RatingEngine, target: Target) -> Features:
        home, away = ratings.rating(target.home_team_id), ratings.rating(target.away_team_id)
        exp_home, exp_away = ratings.expected_goals(target.home_team_id, target.away_team_id)
        return {
            "home_elo": home.elo,
            "away_elo": away.elo,
            "elo_diff": home.elo + self.config.elo_home_advantage - away.elo,
            "elo_home_win_expectancy": ratings.expected_score(
                target.home_team_id, target.away_team_id
            ),
            "home_attack_rating": home.attack,
            "home_defence_rating": home.defence,
            "away_attack_rating": away.attack,
            "away_defence_rating": away.defence,
            "rating_expected_home_goals": exp_home,
            "rating_expected_away_goals": exp_away,
            "home_rated_matches": home.matches,
            "away_rated_matches": away.matches,
        }

    def _context(
        self, teams: TeamIndex, competitions: CompetitionIndex, target: Target
    ) -> Features:
        averages = competitions.averages(
            target.competition_id, target.kickoff_at, self.config.competition_window
        )
        values: Features = {
            "competition_id": target.competition_id,
            "competition_is_cup": int(target.competition_type is not CompetitionType.LEAGUE),
            "competition_home_win_rate": averages.home_win_rate,
            "competition_draw_rate": averages.draw_rate,
            "competition_home_goals": averages.home_goals,
            "competition_away_goals": averages.away_goals,
            "competition_matches": averages.matches,
            "matchday": target.matchday,
            "season_progress": self._season_progress(target),
        }
        rest: dict[str, float | None] = {}
        for side, team_id in (("home", target.home_team_id), ("away", target.away_team_id)):
            past = teams.matches(team_id)
            rest[side] = (
                (target.kickoff_at - past[-1].kickoff_at).total_seconds() / 86400 if past else None
            )
            values[f"{side}_rest_days"] = rest[side]
            for days in CONGESTION_DAYS:
                since = target.kickoff_at - timedelta(days=days)
                values[f"{side}_matches_last_{days}d"] = sum(
                    1 for m in past if since <= m.kickoff_at < target.kickoff_at
                )
            values[f"{side}_season_matches"] = sum(
                1 for m in past if m.season_id == target.season_id
            )
        home_rest, away_rest = rest["home"], rest["away"]
        values["rest_days_diff"] = (
            home_rest - away_rest if home_rest is not None and away_rest is not None else None
        )
        return values

    @staticmethod
    def _season_progress(target: Target) -> float | None:
        """Position in the season from its published dates (known in advance)."""
        if target.season_start is None or target.season_end is None:
            return None
        length = (target.season_end - target.season_start).days
        if length <= 0:
            return None
        elapsed = (target.kickoff_at.date() - target.season_start).days
        return min(1.0, max(0.0, elapsed / length))

    def _head_to_head(self, pairs: PairIndex, target: Target) -> Features:
        cutoff_age = target.kickoff_at - self.config.h2h_max_age
        meetings = [
            m
            for m in pairs.meetings(target.home_team_id, target.away_team_id)
            if m.kickoff_at >= cutoff_age
        ][-self.config.h2h_max_matches :]
        if not meetings:
            return {
                "h2h_matches": 0,
                "h2h_weighted_goal_diff": None,
                "h2h_weighted_points": None,
                "h2h_weight": 0.0,
                "h2h_days_since_last": None,
            }
        weighted_diff = weighted_points = total = 0.0
        for meeting in meetings:
            age_days = (target.kickoff_at - meeting.kickoff_at).total_seconds() / 86400
            weight = 0.5 ** (age_days / self.config.h2h_half_life_days)
            home_view = meeting.home_team_id == target.home_team_id
            gf = meeting.home_goals if home_view else meeting.away_goals
            ga = meeting.away_goals if home_view else meeting.home_goals
            cap = self.config.h2h_goal_diff_cap
            weighted_diff += weight * max(-cap, min(cap, gf - ga))
            weighted_points += weight * (3 if gf > ga else 1 if gf == ga else 0)
            total += weight
        return {
            "h2h_matches": len(meetings),
            "h2h_weighted_goal_diff": weighted_diff / total,
            "h2h_weighted_points": weighted_points / total,
            "h2h_weight": total,
            "h2h_days_since_last": (target.kickoff_at - meetings[-1].kickoff_at).total_seconds()
            / 86400,
        }
