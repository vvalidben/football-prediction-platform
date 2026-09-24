from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import CompetitionType
from app.models.mixins import ExternalIdMixin, IdMixin, TimestampMixin, str_enum

if TYPE_CHECKING:
    from app.models.match import Match
    from app.models.standing import Standing


class Competition(IdMixin, TimestampMixin, Base):
    __tablename__ = "competitions"

    name: Mapped[str]
    short_name: Mapped[str | None]
    type: Mapped[CompetitionType] = mapped_column(str_enum(CompetitionType, "competition_type"))
    country: Mapped[str | None]  # NULL for international competitions
    logo_url: Mapped[str | None]

    seasons: Mapped[list["Season"]] = relationship(back_populates="competition")
    external_ids: Mapped[list["CompetitionExternalId"]] = relationship(
        back_populates="competition", cascade="all, delete-orphan"
    )


class CompetitionExternalId(ExternalIdMixin, Base):
    __tablename__ = "competition_external_ids"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint("competition_id", "data_source_id"),
    )

    competition_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("competitions.id", ondelete="CASCADE")
    )
    competition: Mapped[Competition] = relationship(back_populates="external_ids")


class Season(IdMixin, TimestampMixin, Base):
    """One edition of a competition, e.g. Premier League 2024/2025."""

    __tablename__ = "seasons"
    __table_args__ = (
        UniqueConstraint("competition_id", "name"),
        # Target for matches' composite FK, guaranteeing match.competition_id agrees
        # with its season's competition.
        UniqueConstraint("id", "competition_id"),
        CheckConstraint(
            "start_date IS NULL OR end_date IS NULL OR end_date >= start_date",
            name="end_after_start",
        ),
        Index(
            "uq_seasons_one_current_per_competition",
            "competition_id",
            unique=True,
            postgresql_where=text("is_current"),
        ),
    )

    competition_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("competitions.id", ondelete="RESTRICT")
    )
    name: Mapped[str]  # e.g. "2024/2025" or "2026"
    start_date: Mapped[date | None]
    end_date: Mapped[date | None]
    is_current: Mapped[bool] = mapped_column(server_default=text("false"))

    competition: Mapped[Competition] = relationship(back_populates="seasons")
    matches: Mapped[list["Match"]] = relationship(
        back_populates="season",
        primaryjoin="Match.season_id == Season.id",
        foreign_keys="Match.season_id",
    )
    standings: Mapped[list["Standing"]] = relationship(back_populates="season")
    external_ids: Mapped[list["SeasonExternalId"]] = relationship(
        back_populates="season", cascade="all, delete-orphan"
    )


class SeasonExternalId(ExternalIdMixin, Base):
    __tablename__ = "season_external_ids"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint("season_id", "data_source_id"),
    )

    season_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("seasons.id", ondelete="CASCADE"))
    season: Mapped[Season] = relationship(back_populates="external_ids")
