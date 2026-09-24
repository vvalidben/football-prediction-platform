"""Per-match statistics for teams and players.

Common statistics are typed, nullable columns: NULL means the provider did not
report the value, which is different from 0. Anything else a provider supplies goes
in `extra` (JSONB) and can be promoted to a column once it is used widely.

Expected goals against (xGA) is not stored: a team's xGA in a match is the
opponent's `expected_goals` in the same match and source, so storing it would
duplicate data that can disagree.
"""

from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Numeric,
    SmallInteger,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.match import Match
from app.models.mixins import IdMixin, ProviderRecordMixin, TimestampMixin
from app.models.player import Player
from app.models.team import Team

TEAM_COUNT_COLUMNS = (
    "shots",
    "shots_on_target",
    "shots_off_target",
    "blocked_shots",
    "shots_inside_box",
    "corners",
    "fouls",
    "offsides",
    "yellow_cards",
    "red_cards",
    "saves",
    "passes",
    "passes_completed",
)

PLAYER_COUNT_COLUMNS = (
    "minutes_played",
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "passes",
    "passes_completed",
    "key_passes",
    "tackles",
    "interceptions",
    "fouls_committed",
    "yellow_cards",
    "red_cards",
    "saves",
)


def _non_negative(columns: tuple[str, ...]) -> CheckConstraint:
    # NULL comparisons are unknown, which CHECK treats as passing.
    return CheckConstraint(
        " AND ".join(f"{column} >= 0" for column in columns), name="counts_non_negative"
    )


def _count() -> Mapped[int | None]:
    return mapped_column(SmallInteger)


def _xg() -> Mapped[Decimal | None]:
    return mapped_column(Numeric(6, 3))


class TeamMatchStatistics(IdMixin, ProviderRecordMixin, TimestampMixin, Base):
    __tablename__ = "team_match_statistics"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint("match_id", "team_id", "data_source_id"),
        _non_negative(TEAM_COUNT_COLUMNS),
        CheckConstraint("possession BETWEEN 0 AND 100", name="possession_percentage"),
        CheckConstraint("shots_on_target <= shots", name="shots_on_target_within_shots"),
        CheckConstraint("passes_completed <= passes", name="passes_completed_within_passes"),
        CheckConstraint("expected_goals >= 0", name="expected_goals_non_negative"),
    )

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    team_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT"), index=True
    )
    possession: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))  # percent, 0-100
    shots: Mapped[int | None] = _count()
    shots_on_target: Mapped[int | None] = _count()
    shots_off_target: Mapped[int | None] = _count()
    blocked_shots: Mapped[int | None] = _count()
    shots_inside_box: Mapped[int | None] = _count()
    corners: Mapped[int | None] = _count()
    fouls: Mapped[int | None] = _count()
    offsides: Mapped[int | None] = _count()
    yellow_cards: Mapped[int | None] = _count()
    red_cards: Mapped[int | None] = _count()
    saves: Mapped[int | None] = _count()
    passes: Mapped[int | None] = _count()
    passes_completed: Mapped[int | None] = _count()
    expected_goals: Mapped[Decimal | None] = _xg()
    extra: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))

    match: Mapped[Match] = relationship()
    team: Mapped[Team] = relationship()


class PlayerMatchStatistics(IdMixin, ProviderRecordMixin, TimestampMixin, Base):
    __tablename__ = "player_match_statistics"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint(
            "match_id", "player_id", "data_source_id", name="uq_player_match_statistics_natural_key"
        ),
        _non_negative(PLAYER_COUNT_COLUMNS),
        CheckConstraint("shots_on_target <= shots", name="shots_on_target_within_shots"),
        CheckConstraint("passes_completed <= passes", name="passes_completed_within_passes"),
        CheckConstraint(
            "expected_goals >= 0 AND expected_assists >= 0", name="expected_values_non_negative"
        ),
        CheckConstraint("rating BETWEEN 0 AND 10", name="rating_range"),
    )

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.id", ondelete="RESTRICT"), index=True
    )
    team_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT"), index=True
    )
    minutes_played: Mapped[int | None] = _count()
    goals: Mapped[int | None] = _count()
    assists: Mapped[int | None] = _count()
    shots: Mapped[int | None] = _count()
    shots_on_target: Mapped[int | None] = _count()
    passes: Mapped[int | None] = _count()
    passes_completed: Mapped[int | None] = _count()
    key_passes: Mapped[int | None] = _count()
    tackles: Mapped[int | None] = _count()
    interceptions: Mapped[int | None] = _count()
    fouls_committed: Mapped[int | None] = _count()
    yellow_cards: Mapped[int | None] = _count()
    red_cards: Mapped[int | None] = _count()
    saves: Mapped[int | None] = _count()
    expected_goals: Mapped[Decimal | None] = _xg()
    expected_assists: Mapped[Decimal | None] = _xg()
    rating: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))  # provider rating, 0-10
    extra: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))

    match: Mapped[Match] = relationship()
    player: Mapped[Player] = relationship()
    team: Mapped[Team] = relationship()
