"""Every feature the builder produces, with its group and meaning.

The builder refuses to emit a feature that is not listed here, and every listed
feature is always present (None when the data does not exist). The list is part
of the feature-set version hash.

Prefixes: `home_` / `away_` = the home or away team of the target match.
"""

from dataclasses import dataclass

FORM_WINDOWS = (3, 5, 10)
CONGESTION_DAYS = (7, 14, 30)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    description: str


def _form_specs() -> list[FeatureSpec]:
    specs = []
    for side in ("home", "away"):
        for n in FORM_WINDOWS:
            p = f"{side}_form{n}"
            specs += [
                FeatureSpec(
                    f"{p}_matches", "form", f"{side} team: matches available (of last {n})"
                ),
                FeatureSpec(
                    f"{p}_goals_for", "form", f"{side} team: goals scored per match, last {n}"
                ),
                FeatureSpec(
                    f"{p}_goals_against", "form", f"{side} team: goals conceded per match, last {n}"
                ),
                FeatureSpec(f"{p}_points", "form", f"{side} team: points per match, last {n}"),
                FeatureSpec(f"{p}_win_rate", "form", f"{side} team: share of wins, last {n}"),
                FeatureSpec(f"{p}_draw_rate", "form", f"{side} team: share of draws, last {n}"),
                FeatureSpec(f"{p}_loss_rate", "form", f"{side} team: share of losses, last {n}"),
                FeatureSpec(
                    f"{p}_xg_for", "form", f"{side} team: xG per match, last {n} (where known)"
                ),
                FeatureSpec(
                    f"{p}_xg_against", "form", f"{side} team: xGA per match, last {n} (where known)"
                ),
                FeatureSpec(
                    f"{p}_xg_matches", "form", f"{side} team: matches with xG, of last {n}"
                ),
            ]
    return specs


SPECS: tuple[FeatureSpec, ...] = (
    *_form_specs(),
    # Venue-specific strengths, relative to the competition average (1.0 = average).
    FeatureSpec(
        "home_attack_strength", "venue", "home team scoring at home vs league home average"
    ),
    FeatureSpec(
        "home_defence_weakness", "venue", "home team conceding at home vs league away average"
    ),
    FeatureSpec("away_attack_strength", "venue", "away team scoring away vs league away average"),
    FeatureSpec(
        "away_defence_weakness", "venue", "away team conceding away vs league home average"
    ),
    FeatureSpec("home_venue_matches", "venue", "home matches behind home_* strengths"),
    FeatureSpec("away_venue_matches", "venue", "away matches behind away_* strengths"),
    # Elo-like ratings.
    FeatureSpec("home_elo", "strength", "home team Elo rating"),
    FeatureSpec("away_elo", "strength", "away team Elo rating"),
    FeatureSpec("elo_diff", "strength", "home Elo + home advantage - away Elo"),
    FeatureSpec("elo_home_win_expectancy", "strength", "Elo expected score of the home team"),
    FeatureSpec("home_attack_rating", "strength", "home team attack rating (log-goals)"),
    FeatureSpec(
        "home_defence_rating", "strength", "home team defence rating (log-goals, higher = better)"
    ),
    FeatureSpec("away_attack_rating", "strength", "away team attack rating"),
    FeatureSpec("away_defence_rating", "strength", "away team defence rating"),
    FeatureSpec(
        "rating_expected_home_goals", "strength", "goals implied by attack/defence ratings"
    ),
    FeatureSpec(
        "rating_expected_away_goals", "strength", "goals implied by attack/defence ratings"
    ),
    FeatureSpec("home_rated_matches", "strength", "matches behind the home team's ratings"),
    FeatureSpec("away_rated_matches", "strength", "matches behind the away team's ratings"),
    # Context.
    FeatureSpec("competition_id", "context", "internal competition id (categorical)"),
    FeatureSpec("competition_is_cup", "context", "1 for cups and other knockout competitions"),
    FeatureSpec(
        "competition_home_win_rate", "context", "home win share in the competition, past year"
    ),
    FeatureSpec("competition_draw_rate", "context", "draw share in the competition, past year"),
    FeatureSpec("competition_home_goals", "context", "home goals per match in the competition"),
    FeatureSpec("competition_away_goals", "context", "away goals per match in the competition"),
    FeatureSpec("competition_matches", "context", "matches behind the competition averages"),
    FeatureSpec("home_rest_days", "context", "days since the home team's previous match"),
    FeatureSpec("away_rest_days", "context", "days since the away team's previous match"),
    FeatureSpec("rest_days_diff", "context", "home rest days - away rest days"),
    *(
        FeatureSpec(
            f"{side}_matches_last_{d}d", "context", f"{side} team matches in the previous {d} days"
        )
        for side in ("home", "away")
        for d in CONGESTION_DAYS
    ),
    FeatureSpec("matchday", "context", "round number within the season, where known"),
    FeatureSpec(
        "season_progress", "context", "0 at season start, 1 at season end (published dates)"
    ),
    FeatureSpec("home_season_matches", "context", "home team matches already played this season"),
    FeatureSpec("away_season_matches", "context", "away team matches already played this season"),
    # Head-to-head, time-decayed so old meetings count little.
    FeatureSpec("h2h_matches", "h2h", "recent meetings used (capped, within 5 years)"),
    FeatureSpec(
        "h2h_weighted_goal_diff", "h2h", "decayed mean goal difference (capped at +/-3), home view"
    ),
    FeatureSpec("h2h_weighted_points", "h2h", "decayed mean points for the home team"),
    FeatureSpec("h2h_weight", "h2h", "total decay weight (evidence strength)"),
    FeatureSpec("h2h_days_since_last", "h2h", "days since the teams last met"),
    # Player availability.
    FeatureSpec("home_absences_out", "players", "home players known to be out"),
    FeatureSpec("home_absences_doubtful", "players", "home players known to be doubtful"),
    FeatureSpec("away_absences_out", "players", "away players known to be out"),
    FeatureSpec("away_absences_doubtful", "players", "away players known to be doubtful"),
    FeatureSpec(
        "home_absent_contribution",
        "players",
        "share of home goals+assists this season by absent players",
    ),
    FeatureSpec(
        "away_absent_contribution",
        "players",
        "share of away goals+assists this season by absent players",
    ),
    # Bookmaker odds (optional; match-winner market, overround removed).
    FeatureSpec("odds_bookmakers", "odds", "bookmakers with match-winner odds before the cutoff"),
    FeatureSpec("odds_opening_home_prob", "odds", "implied home probability, opening odds"),
    FeatureSpec("odds_opening_draw_prob", "odds", "implied draw probability, opening odds"),
    FeatureSpec("odds_opening_away_prob", "odds", "implied away probability, opening odds"),
    FeatureSpec(
        "odds_latest_home_prob", "odds", "implied home probability, latest odds before cutoff"
    ),
    FeatureSpec(
        "odds_latest_draw_prob", "odds", "implied draw probability, latest odds before cutoff"
    ),
    FeatureSpec(
        "odds_latest_away_prob", "odds", "implied away probability, latest odds before cutoff"
    ),
    FeatureSpec("odds_latest_overround", "odds", "bookmaker margin of the latest odds"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in SPECS)
GROUPS: dict[str, str] = {spec.name: spec.group for spec in SPECS}
DESCRIPTIONS: dict[str, str] = {spec.name: spec.description for spec in SPECS}

if len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):  # pragma: no cover - import-time guard
    raise RuntimeError("duplicate feature names in registry")
