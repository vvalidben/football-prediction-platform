from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Computed,
    ForeignKey,
    SmallInteger,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import IdMixin, ProviderRecordMixin, TimestampMixin
from app.models.team import Team

if TYPE_CHECKING:
    from app.models.competition import Season


class Standing(IdMixin, ProviderRecordMixin, TimestampMixin, Base):
    """A team's league-table row as of a date, as reported by one provider.

    Snapshots over time let models use the table as it stood before each match.
    """

    __tablename__ = "standings"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint(
            "season_id",
            "group_name",
            "team_id",
            "data_source_id",
            "as_of_date",
            name="uq_standings_natural_key",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint("position > 0", name="position_positive"),
        CheckConstraint(
            "won >= 0 AND drawn >= 0 AND lost >= 0 AND goals_for >= 0 AND goals_against >= 0",
            name="counts_non_negative",
        ),
        CheckConstraint("played = won + drawn + lost", name="played_equals_results"),
    )

    season_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("seasons.id", ondelete="RESTRICT")
    )
    team_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT"), index=True
    )
    group_name: Mapped[str | None]  # for group stages; NULL for a single table
    as_of_date: Mapped[date]
    position: Mapped[int] = mapped_column(SmallInteger)
    played: Mapped[int] = mapped_column(SmallInteger)
    won: Mapped[int] = mapped_column(SmallInteger)
    drawn: Mapped[int] = mapped_column(SmallInteger)
    lost: Mapped[int] = mapped_column(SmallInteger)
    goals_for: Mapped[int] = mapped_column(SmallInteger)
    goals_against: Mapped[int] = mapped_column(SmallInteger)
    goal_difference: Mapped[int] = mapped_column(
        SmallInteger, Computed("goals_for - goals_against", persisted=True)
    )
    # Not derived from results: competitions can deduct points.
    points: Mapped[int] = mapped_column(SmallInteger)
    form: Mapped[str | None]  # most recent results, e.g. "WWDLW"

    season: Mapped["Season"] = relationship(back_populates="standings")
    team: Mapped[Team] = relationship()
