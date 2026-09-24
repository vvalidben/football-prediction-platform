"""Synthetic datasets with a known signal, for model tests (no database).

Each team has a hidden strength; goals are Poisson with rates depending on the
strength gap, and features expose that gap (noisily) the way Sprint 4 features do.
Models that use the features should therefore beat the historical baseline.
"""

import math
from datetime import UTC, datetime, timedelta

import numpy as np

from app.features.registry import FEATURE_NAMES
from app.ml.dataset import Dataset, Example

VERSION = "v1-synthetic"
START = datetime(2019, 8, 1, 15, tzinfo=UTC)


def make_dataset(
    seasons: int = 4, teams: int = 10, seed: int = 7, matches_per_season: int | None = None
) -> Dataset:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 0.35, teams)
    examples = []
    match_id = 0
    for season in range(seasons):
        pairs = [(h, a) for h in range(teams) for a in range(teams) if h != a]
        rng.shuffle(pairs)
        pairs = pairs[: matches_per_season or len(pairs)]
        season_start = START + timedelta(days=365 * season)
        for index, (home, away) in enumerate(pairs):
            gap = strength[home] - strength[away]
            home_rate, away_rate = math.exp(0.25 + 0.2 + gap), math.exp(0.25 - gap)
            hg, ag = int(rng.poisson(home_rate)), int(rng.poisson(away_rate))
            features: dict[str, float | int | None] = dict.fromkeys(FEATURE_NAMES)
            features.update(
                {
                    "elo_diff": 60 + 400 * gap + rng.normal(0, 30),
                    "rating_expected_home_goals": home_rate * rng.uniform(0.85, 1.15),
                    "rating_expected_away_goals": away_rate * rng.uniform(0.85, 1.15),
                    "home_form5_goals_for": home_rate + rng.normal(0, 0.4),
                    "away_form5_goals_for": away_rate + rng.normal(0, 0.4),
                    "home_form5_points": float(rng.uniform(0, 3)),  # pure noise
                    "home_rated_matches": 10 + season * 30,
                    "away_rated_matches": 10 + season * 30,
                    "competition_is_cup": 0,
                }
            )
            examples.append(
                Example(
                    match_id=match_id,
                    kickoff_at=season_start + timedelta(hours=6 * index),
                    season_id=season + 1,
                    competition_id=1,
                    features=features,
                    home_goals=hg,
                    away_goals=ag,
                )
            )
            match_id += 1
    return Dataset(tuple(examples), VERSION)
