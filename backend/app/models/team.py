from sqlalchemy import BigInteger, ForeignKey, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.mixins import ExternalIdMixin, IdMixin, TimestampMixin


class Team(IdMixin, TimestampMixin, Base):
    __tablename__ = "teams"

    name: Mapped[str]
    short_name: Mapped[str | None]
    code: Mapped[str | None]  # three-letter abbreviation, e.g. "ARS"
    country: Mapped[str | None]
    logo_url: Mapped[str | None]
    is_national_team: Mapped[bool] = mapped_column(server_default=text("false"))

    external_ids: Mapped[list["TeamExternalId"]] = relationship(
        back_populates="team", cascade="all, delete-orphan"
    )


class TeamExternalId(ExternalIdMixin, Base):
    __tablename__ = "team_external_ids"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint("team_id", "data_source_id"),
    )

    team_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("teams.id", ondelete="CASCADE"))
    team: Mapped[Team] = relationship(back_populates="external_ids")
