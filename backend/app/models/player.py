from datetime import date

from sqlalchemy import BigInteger, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import PlayerPosition
from app.models.mixins import ExternalIdMixin, IdMixin, TimestampMixin, str_enum
from app.models.team import Team


class Player(IdMixin, TimestampMixin, Base):
    __tablename__ = "players"

    name: Mapped[str]  # display name as commonly reported
    first_name: Mapped[str | None]
    last_name: Mapped[str | None]
    date_of_birth: Mapped[date | None]
    nationality: Mapped[str | None]
    position: Mapped[PlayerPosition | None] = mapped_column(
        str_enum(PlayerPosition, "player_position")
    )
    # Latest known club. Historical membership is recorded per match in lineups.
    current_team_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="SET NULL"), index=True
    )

    current_team: Mapped[Team | None] = relationship()
    external_ids: Mapped[list["PlayerExternalId"]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )


class PlayerExternalId(ExternalIdMixin, Base):
    __tablename__ = "player_external_ids"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint("player_id", "data_source_id"),
    )

    player_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("players.id", ondelete="CASCADE"))
    player: Mapped[Player] = relationship(back_populates="external_ids")
