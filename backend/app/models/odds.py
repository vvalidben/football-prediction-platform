from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Numeric,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.enums import OddsMarket
from app.models.match import Match
from app.models.mixins import CreatedAtMixin, IdMixin, ProviderRecordMixin, str_enum


class Bookmaker(IdMixin, CreatedAtMixin, Base):
    __tablename__ = "bookmakers"

    name: Mapped[str] = mapped_column(unique=True)


class Odds(IdMixin, ProviderRecordMixin, CreatedAtMixin, Base):
    """One bookmaker price for one selection, observed at `recorded_at`.

    Rows are append-only snapshots, so price movement over time is preserved.
    Selections: match_winner → home/draw/away, over_under → over/under with `line`,
    both_teams_to_score → yes/no, correct_score → "2-1", other markets are free text.
    """

    __tablename__ = "odds"
    __table_args__ = (
        UniqueConstraint("data_source_id", "external_id"),
        UniqueConstraint(
            "match_id",
            "bookmaker_id",
            "data_source_id",
            "market",
            "selection",
            "line",
            "recorded_at",
            name="uq_odds_natural_key",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint("price > 1", name="price_above_one"),
        CheckConstraint(
            "market <> 'match_winner' OR selection IN ('home', 'draw', 'away')",
            name="match_winner_selection",
        ),
        CheckConstraint(
            "market <> 'over_under' OR (selection IN ('over', 'under') AND line IS NOT NULL)",
            name="over_under_selection",
        ),
        CheckConstraint(
            "market <> 'both_teams_to_score' OR selection IN ('yes', 'no')",
            name="btts_selection",
        ),
        CheckConstraint(
            "market <> 'correct_score' OR selection ~ '^[0-9]+-[0-9]+$'",
            name="correct_score_selection",
        ),
    )

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    bookmaker_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("bookmakers.id", ondelete="RESTRICT"), index=True
    )
    market: Mapped[OddsMarket] = mapped_column(str_enum(OddsMarket, "odds_market"))
    selection: Mapped[str]
    line: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))  # e.g. 2.5 goals, -0.25
    price: Mapped[Decimal] = mapped_column(Numeric(8, 3))  # decimal odds
    is_closing: Mapped[bool] = mapped_column(server_default=text("false"))
    recorded_at: Mapped[datetime]

    match: Mapped[Match] = relationship()
    bookmaker: Mapped[Bookmaker] = relationship()
