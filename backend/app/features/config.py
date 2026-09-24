"""Feature-set parameters and versioning.

The version string combines a human name with a hash of every parameter and every
feature name. Changing a window length, an Elo constant or the list of features
therefore produces a new version automatically, so snapshots made with different
definitions can never be mixed up.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta

from app.features.registry import FEATURE_NAMES


@dataclass(frozen=True)
class FeatureConfig:
    name: str = "v1"
    # Form windows and congestion windows are defined in `registry` (they shape the
    # feature names, which are part of the version hash).

    #: A result is treated as known this long after kickoff (covers extra time and
    #: publication delay). Matches kicking off within this window before the cutoff
    #: are invisible, including the target match itself.
    result_delay: timedelta = timedelta(hours=3)

    #: Home (or away) matches used for venue-specific strengths.
    venue_window: int = 10
    #: Pseudo-matches at the competition average, shrinking small samples.
    venue_prior_matches: float = 3.0
    #: Competition averages (home advantage, goal rates) use this much history.
    competition_window: timedelta = timedelta(days=365)
    #: Below this many matches, competition averages fall back to all competitions.
    competition_min_matches: int = 20
    #: Goal baselines for venue strengths are shrunk toward these rates, weighted as
    #: this many matches, so early-history baselines are neither noisy nor zero.
    baseline_prior_home_goals: float = 1.5
    baseline_prior_away_goals: float = 1.2
    baseline_prior_matches: float = 20.0

    elo_initial: float = 1500.0
    elo_k: float = 20.0
    elo_home_advantage: float = 60.0
    #: Fraction of the distance to the mean removed when a team starts a new season.
    elo_season_regression: float = 0.2
    #: Goal (attack/defence) ratings: learning rate and home advantage in log-goals.
    goal_learning_rate: float = 0.05
    goal_home_advantage: float = 0.15
    goal_prior_rate: float = 1.35

    h2h_max_matches: int = 6
    h2h_max_age: timedelta = timedelta(days=5 * 365)
    #: Weight of a meeting halves every this many days: a 4-year-old meeting counts
    #: 1/16 as much as one played today.
    h2h_half_life_days: float = 365.0
    #: Goal difference per meeting is capped, so one blowout cannot dominate.
    h2h_goal_diff_cap: int = 3

    #: Per-fixture absence lists are only used if ingested before the cutoff,
    #: unless the provider is trusted to publish them before kickoff.
    trust_provider_absence_timing: bool = False

    def version(self) -> str:
        definition = {
            **{k: str(v) if isinstance(v, timedelta) else v for k, v in asdict(self).items()},
            "features": list(FEATURE_NAMES),
        }
        digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        return f"{self.name}-{digest[:10]}"
