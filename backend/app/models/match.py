from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    SmallInteger,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.competition import Competition, Season
from app.models.enums import MatchEventType, MatchPeriod, MatchStatus
from app.models.mixins import (
    CreatedAtMixin,
    ExternalIdMixin,
    IdMixin,
    ProviderRecordMixin,
    TimestampMixin,
    str_enum,
)
from app.models.player import Player
from app.models.team import Team


def _score_pair_checks(prefix: str) -> tuple[CheckConstraint, CheckConstraint]:
    home, away = f"home_{prefix}score", f"away_{prefix}score"
    return (
        CheckConstraint(f"{home} >= 0 AND {away} >= 0", name=f"{prefix}scores_non_negative"),
        CheckConstraint(f"({home} IS NULL) = ({away} IS NULL)", name=f"{prefix}scores_paired"),
    )


class Match(IdMixin, TimestampMixin, Base):
    __tablename__ = "matches"
    __table_args__ = (
        ForeignKeyConstraint(
            ["season_id", "competition_id"],
            ["seasons.id", "seasons.competition_id"],
            name="fk_matches_season_id_competition_id_seasons",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("home_team_id", "away_team_id", "kickoff_at"),
        CheckConstraint("home_team_id <> away_team_id", name="distinct_teams"),
        CheckConstraint(
            "status <> 'finished' OR (home_score IS NOT NULL AND away_score IS NOT NULL)",
            name="finished_has_score",
        ),
        CheckConstraint("matchday IS NULL OR matchday > 0", name="matchday_positive"),
        *_score_pair_checks(""),
        *_score_pair_checks("half_time_"),
        *_score_pair_checks("penalty_"),
        Index("ix_matches_season_id_competition_id", "season_id", "competition_id"),
        Index("ix_matches_competition_id_kickoff_at", "competition_id", "kickoff_at"),
        Index("ix_matches_home_team_id_kickoff_at", "home_team_id", "kickoff_at"),
        Index("ix_matches_away_team_id_kickoff_at", "away_team_id", "kickoff_at"),
        Index("ix_matches_status_kickoff_at", "status", "kickoff_at"),
    )

    competition_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("competitions.id", ondelete="RESTRICT")
    )
    season_id: Mapped[int] = mapped_column(BigInteger)
    home_team_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT")
    )
    away_team_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT")
    )
    kickoff_at: Mapped[datetime]  # scheduled kickoff, UTC
    status: Mapped[MatchStatus] = mapped_column(
        str_enum(MatchStatus, "match_status"), server_default=MatchStatus.SCHEDULED.value
    )
    matchday: Mapped[int | None] = mapped_column(SmallInteger)
    stage: Mapped[str | None]  # e.g. "regular_season", "group_stage", "final"
    # Score after regular and extra time; penalty shootout goals are stored separately.
    home_score: Mapped[int | None] = mapped_column(SmallInteger)
    away_score: Mapped[int | None] = mapped_column(SmallInteger)
    home_half_time_score: Mapped[int | None] = mapped_column(SmallInteger)
    away_half_time_score: Mapped[int | None] = mapped_column(SmallInteger)
    home_penalty_score: Mapped[int | None] = mapped_column(SmallInteger)
    away_penalty_score: Mapped[int | None] = mapped_column(SmallInteger)
    venue: Mapped[str | None]
    referee: Mapped[str | None]

    competition: Mapped[Competition] = relationship()
    season: Mapped[Season] = relationship(
        back_populates="matches",
        primaryjoin="Match.season_id == Season.id",
        foreign_keys=[season_id],
    )
    home_team: Mapped[Team] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped[Team] = relationship(foreign_keys=[away_team_id])
    external_ids: Mapped[list["MatchExternalId"]] = relationship(
        back_populates="match", cascade="all, delete-orphan"
    )
    events: Mapped[list["MatchEvent"]] = relationship(
        back_populates="match", cascade="all, delete-orphan", passive_deletes=True
    )
    lineups: Mapped[list["Lineup"]] = relationship(
        back_populates="match", cascade="all, delete-orphan", passive_deletes=True
    )


class MatchExternalId(ExternalIdMixin, Base):
    __tablename__ = "match_external_ids"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint("match_id", "data_source_id"),
    )

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    match: Mapped[Match] = relationship(back_populates="external_ids")


class MatchEvent(IdMixin, ProviderRecordMixin, CreatedAtMixin, Base):
    """A timeline event: goal, card, substitution, VAR decision, …"""

    __tablename__ = "match_events"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        CheckConstraint("minute >= 0", name="minute_non_negative"),
        CheckConstraint("added_time IS NULL OR added_time >= 0", name="added_time_non_negative"),
        Index("ix_match_events_match_id_minute", "match_id", "minute"),
    )

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    type: Mapped[MatchEventType] = mapped_column(str_enum(MatchEventType, "match_event_type"))
    period: Mapped[MatchPeriod | None] = mapped_column(str_enum(MatchPeriod, "match_period"))
    minute: Mapped[int] = mapped_column(SmallInteger)
    added_time: Mapped[int | None] = mapped_column(SmallInteger)  # the "+3" in 90+3
    team_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT"), index=True
    )
    player_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("players.id", ondelete="RESTRICT"), index=True
    )
    # Assist provider for goals, player replaced for substitutions.
    related_player_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("players.id", ondelete="RESTRICT"), index=True
    )
    detail: Mapped[str | None]  # provider's free-text qualifier, e.g. "Header"

    match: Mapped[Match] = relationship(back_populates="events")
    team: Mapped[Team | None] = relationship()
    player: Mapped[Player | None] = relationship(foreign_keys=[player_id])
    related_player: Mapped[Player | None] = relationship(foreign_keys=[related_player_id])


class Lineup(IdMixin, ProviderRecordMixin, TimestampMixin, Base):
    """A team's announced line-up for one match, as reported by one provider."""

    __tablename__ = "lineups"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint("match_id", "team_id", "data_source_id"),
    )

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    team_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("teams.id", ondelete="RESTRICT"), index=True
    )
    formation: Mapped[str | None]  # e.g. "4-3-3"
    coach_name: Mapped[str | None]

    match: Mapped[Match] = relationship(back_populates="lineups")
    team: Mapped[Team] = relationship()
    players: Mapped[list["LineupPlayer"]] = relationship(
        back_populates="lineup", cascade="all, delete-orphan", passive_deletes=True
    )


class LineupPlayer(IdMixin, CreatedAtMixin, Base):
    __tablename__ = "lineup_players"
    __table_args__ = (
        UniqueConstraint("lineup_id", "player_id"),
        CheckConstraint(
            "shirt_number IS NULL OR shirt_number BETWEEN 0 AND 99", name="shirt_number_range"
        ),
    )

    lineup_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("lineups.id", ondelete="CASCADE"))
    player_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("players.id", ondelete="RESTRICT"), index=True
    )
    is_starter: Mapped[bool]
    position: Mapped[str | None]  # provider label, e.g. "G", "CB", "LW"
    shirt_number: Mapped[int | None] = mapped_column(SmallInteger)

    lineup: Mapped[Lineup] = relationship(back_populates="players")
    player: Mapped[Player] = relationship()
