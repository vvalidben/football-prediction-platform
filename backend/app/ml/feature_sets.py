"""Which stored features each model family uses.

Bookmaker odds are excluded from the default set: they must never be mandatory, and
a model that mostly copies the market says little about our own signal. A separate
set with odds exists for comparison.
"""

from app.features.registry import FEATURE_NAMES

#: Not model inputs:
#: - identifiers are categorical and have no meaning as numbers;
#: - counts that grow without bound as history accumulates. Later seasons always
#:   lie outside the training range, so any weight on them extrapolates (found in
#:   testing: goal predictions drifted up by 0.6 per match). They remain available
#:   as confidence information, e.g. for the training warm-up filter.
EXCLUDED = frozenset(
    {"competition_id", "home_rated_matches", "away_rated_matches", "competition_matches"}
)
ODDS = tuple(name for name in FEATURE_NAMES if name.startswith("odds_"))

CORE_FEATURES: tuple[str, ...] = tuple(
    name for name in FEATURE_NAMES if name not in EXCLUDED and name not in ODDS
)
WITH_ODDS: tuple[str, ...] = CORE_FEATURES + tuple(
    name for name in ODDS if name.startswith("odds_latest") and name.endswith("_prob")
)
