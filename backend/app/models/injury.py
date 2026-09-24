from datetime import date

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import AbsenceType, AvailabilityStatus
from app.models.match import Match
from app.models.mixins import IdMixin, ProviderRecordMixin, TimestampMixin, str_enum
from app.models.player import Player
from app.models.team import Team


class Injury(IdMixin, ProviderRecordMixin, TimestampMixin, Base):
    """A player's unavailability: injury, illness or suspension.

    Providers report absences either per fixture (`match_id` set) or as a period
    (`start_date` set). The NULLS NOT DISTINCT natural key deduplicates both styles.
    """

    __tablename__ = "injuries"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint(
            "player_id",
            "data_source_id",
            "match_id",
            "start_date",
            name="uq_injuries_natural_key",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "match_id IS NOT NULL OR start_date IS NOT NULL", name="match_or_start_date"
        ),
        CheckConstraint("end_date >= start_date", name="end_after_start"),
        CheckConstraint("expected_return_date >= start_date", name="expected_return_after_start"),
        Index("ix_injuries_team_id_start_date", "team_id", "start_date"),
    )

    player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.id", ondelete="RESTRICT")
    )
    team_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT")
    )
    match_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("matches.id", ondelete="CASCADE"), index=True
    )
    absence_type: Mapped[AbsenceType] = mapped_column(str_enum(AbsenceType, "absence_type"))
    status: Mapped[AvailabilityStatus] = mapped_column(
        str_enum(AvailabilityStatus, "availability_status")
    )
    reason: Mapped[str | None]  # provider text, e.g. "Hamstring", "Red card"
    start_date: Mapped[date | None]
    expected_return_date: Mapped[date | None]
    end_date: Mapped[date | None]

    player: Mapped[Player] = relationship()
    team: Mapped[Team | None] = relationship()
    match: Mapped[Match | None] = relationship()
