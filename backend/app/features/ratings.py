"""Elo-like team strength, updated one known result at a time.

- **Overall rating** (Elo): expected score from the rating gap plus home advantage;
  updates are scaled by the goal margin (World Football Elo style).
- **Attack / defence ratings** in log-goals: a team's expected goals are
  `base_rate * exp(attack - opponent_defence (+ home advantage))`. After each match
  both ratings move by the Poisson log-likelihood gradient (actual - expected
  goals). A higher defence rating means fewer goals conceded.
- When a team plays its first match of a new season, all three ratings are pulled
  part of the way back to the average, since squads change between seasons.

Ratings only change in `add`, which the `Timeline` calls for matches whose result
is already known, so a rating read for a target reflects only the past.
"""

import math
from dataclasses import dataclass

from app.features.config import FeatureConfig
from app.features.timeline import PastMatch


@dataclass
class TeamRating:
    elo: float
    attack: float = 0.0
    defence: float = 0.0
    matches: int = 0
    season_id: int | None = None


class RatingEngine:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.teams: dict[int, TeamRating] = {}
        self._goals = 0
        self._team_matches = 0

    @property
    def base_rate(self) -> float:
        """Mean goals per team per match so far (prior until data exists)."""
        prior_weight = 50
        return (self._goals + self.config.goal_prior_rate * prior_weight) / (
            self._team_matches + prior_weight
        )

    def rating(self, team_id: int) -> TeamRating:
        return self.teams.get(team_id) or TeamRating(elo=self.config.elo_initial)

    def expected_score(self, home_id: int, away_id: int) -> float:
        gap = self.rating(home_id).elo + self.config.elo_home_advantage - self.rating(away_id).elo
        return 1.0 / (1.0 + 10 ** (-gap / 400.0))

    def expected_goals(self, home_id: int, away_id: int) -> tuple[float, float]:
        home, away = self.rating(home_id), self.rating(away_id)
        base = self.base_rate
        return (
            base * math.exp(home.attack - away.defence + self.config.goal_home_advantage),
            base * math.exp(away.attack - home.defence),
        )

    def _enter_season(self, team_id: int, season_id: int) -> TeamRating:
        rating = self.teams.setdefault(team_id, TeamRating(elo=self.config.elo_initial))
        if rating.season_id is not None and rating.season_id != season_id:
            keep = 1.0 - self.config.elo_season_regression
            rating.elo = self.config.elo_initial + keep * (rating.elo - self.config.elo_initial)
            rating.attack *= keep
            rating.defence *= keep
        rating.season_id = season_id
        return rating

    def add(self, match: PastMatch) -> None:
        home = self._enter_season(match.home_team_id, match.season_id)
        away = self._enter_season(match.away_team_id, match.season_id)

        # Overall Elo.
        expected = self.expected_score(match.home_team_id, match.away_team_id)
        margin = abs(match.home_goals - match.away_goals)
        actual = 1.0 if match.home_goals > match.away_goals else 0.5 if margin == 0 else 0.0
        multiplier = 1.0 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
        change = self.config.elo_k * multiplier * (actual - expected)
        home.elo += change
        away.elo -= change

        # Attack and defence (computed from pre-match ratings).
        exp_home, exp_away = self.expected_goals(match.home_team_id, match.away_team_id)
        lr = self.config.goal_learning_rate
        home_error, away_error = match.home_goals - exp_home, match.away_goals - exp_away
        home.attack += lr * home_error
        away.defence -= lr * home_error
        away.attack += lr * away_error
        home.defence -= lr * away_error

        home.matches += 1
        away.matches += 1
        self._goals += match.home_goals + match.away_goals
        self._team_matches += 2
