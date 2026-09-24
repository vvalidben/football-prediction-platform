"""Features read directly from the database, each query bounded by the cutoff.

Absences and odds are not part of the match timeline, so every query here filters
explicitly on time:

- **odds**: `recorded_at < cutoff` (and never after kickoff, since cutoff ≤ kickoff);
- **absences**: an injury period must have *started before the cutoff date* and not
  ended before kickoff. A per-fixture absence list for the target match is used
  only if we had ingested it before the cutoff (`created_at`). The table does not
  record when a provider published an absence, and ingestion time is the only
  proof we knew it in time. `trust_provider_absence_timing` relaxes this for
  providers known to publish absences before kickoff;
- **player contributions**: only matches whose result was known at the cutoff.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app.features.timeline import Target
from app.models import Bookmaker, Injury, Match, Odds, PlayerMatchStatistics
from app.models.enums import AvailabilityStatus, MatchStatus, OddsMarket

Values = dict[str, float | int | None]
SELECTIONS = ("home", "draw", "away")


def absence_features(
    session: Session, target: Target, *, result_delay: timedelta, trust_provider_timing: bool
) -> Values:
    cutoff_day = target.cutoff.date()
    kickoff_day = target.kickoff_at.date()
    period_known = and_(
        Injury.match_id.is_(None),
        Injury.start_date < cutoff_day,
        or_(Injury.end_date.is_(None), Injury.end_date >= kickoff_day),
        or_(Injury.expected_return_date.is_(None), Injury.expected_return_date > kickoff_day),
    )
    fixture_list = Injury.match_id == target.match_id
    if not trust_provider_timing:
        fixture_list = and_(fixture_list, Injury.created_at <= target.cutoff)
    rows = session.execute(
        select(Injury.team_id, Injury.player_id, Injury.status).where(
            Injury.team_id.in_((target.home_team_id, target.away_team_id)),
            or_(period_known, fixture_list),
        )
    ).all()

    absent: dict[int, dict[int, AvailabilityStatus]] = defaultdict(dict)
    for team_id, player_id, status in rows:
        current = absent[team_id].get(player_id)
        if current is None or status is AvailabilityStatus.OUT:
            absent[team_id][player_id] = status

    values: Values = {}
    for side, team_id in (("home", target.home_team_id), ("away", target.away_team_id)):
        statuses = list(absent.get(team_id, {}).values())
        # Zero absences is only a fact if absence data exists for *this* team.
        known = team_id in absent or _has_absence_data(session, team_id, target.cutoff)
        values[f"{side}_absences_out"] = statuses.count(AvailabilityStatus.OUT) if known else None
        values[f"{side}_absences_doubtful"] = (
            statuses.count(AvailabilityStatus.DOUBTFUL) if known else None
        )
        values[f"{side}_absent_contribution"] = _absent_contribution(
            session, target, team_id, set(absent.get(team_id, {})), result_delay
        )
    return values


def _has_absence_data(session: Session, team_id: int, cutoff: datetime) -> bool:
    """Whether any absence data for this team was ingested by the cutoff. Without
    it, zero absences would be a guess, so the features stay None."""
    return (
        session.scalar(
            select(func.count())
            .select_from(Injury)
            .where(Injury.team_id == team_id, Injury.created_at <= cutoff)
        )
        or 0
    ) > 0


def _absent_contribution(
    session: Session,
    target: Target,
    team_id: int,
    absent_players: set[int],
    result_delay: timedelta,
) -> float | None:
    """Share of the team's goals + assists this season scored by absent players."""
    rows = session.execute(
        select(
            PlayerMatchStatistics.player_id,
            func.coalesce(func.sum(PlayerMatchStatistics.goals), 0)
            + func.coalesce(func.sum(PlayerMatchStatistics.assists), 0),
        )
        .join(Match, Match.id == PlayerMatchStatistics.match_id)
        .where(
            PlayerMatchStatistics.team_id == team_id,
            Match.season_id == target.season_id,
            Match.status == MatchStatus.FINISHED,
            Match.kickoff_at <= target.cutoff - result_delay,
        )
        .group_by(PlayerMatchStatistics.player_id)
    ).all()
    total = sum(int(contribution) for _, contribution in rows)
    if not rows or total == 0:
        return None
    return sum(int(c) for player, c in rows if player in absent_players) / total


def odds_features(session: Session, target: Target) -> Values:
    """Opening and latest match-winner odds known before the cutoff, averaged over
    bookmakers after removing each bookmaker's margin."""
    rows = session.execute(
        select(Odds.bookmaker_id, Odds.selection, Odds.price, Odds.recorded_at)
        .join(Bookmaker, Bookmaker.id == Odds.bookmaker_id)
        .where(
            Odds.match_id == target.match_id,
            Odds.market == OddsMarket.MATCH_WINNER,
            Odds.recorded_at < target.cutoff,
        )
        .order_by(Odds.recorded_at)
    ).all()
    snapshots: dict[int, dict[datetime, dict[str, Decimal]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for bookmaker, selection, price, recorded_at in rows:
        snapshots[bookmaker][recorded_at][selection] = price

    opening: list[dict[str, float]] = []
    latest: list[tuple[dict[str, float], float]] = []
    for by_time in snapshots.values():
        complete = [
            prices for _, prices in sorted(by_time.items()) if set(prices) >= set(SELECTIONS)
        ]
        if not complete:
            continue
        opening.append(_implied(complete[0])[0])
        latest.append(_implied(complete[-1]))

    values: Values = {"odds_bookmakers": len(latest) or None}
    for phase, books in (("opening", opening), ("latest", [probs for probs, _ in latest])):
        for selection in SELECTIONS:
            values[f"odds_{phase}_{selection}_prob"] = (
                sum(book[selection] for book in books) / len(books) if books else None
            )
    values["odds_latest_overround"] = (
        sum(margin for _, margin in latest) / len(latest) if latest else None
    )
    return values


def _implied(prices: dict[str, Decimal]) -> tuple[dict[str, float], float]:
    raw = {selection: 1.0 / float(prices[selection]) for selection in SELECTIONS}
    total = sum(raw.values())
    return {selection: p / total for selection, p in raw.items()}, total - 1.0
