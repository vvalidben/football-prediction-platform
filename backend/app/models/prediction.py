"""Model outputs for matches and their evaluation against actual results."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.match import Match
from app.models.mixins import CreatedAtMixin, IdMixin
from app.models.ml import FeatureSnapshot, ModelVersion

# Probabilities carry six decimal places; outcome probabilities must sum to 1
# within this tolerance to allow for rounding.
PROBABILITY = Numeric(7, 6)
PROBABILITY_SUM_TOLERANCE = "0.00001"


def _probability_range(*columns: str, name: str) -> CheckConstraint:
    return CheckConstraint(
        " AND ".join(f"{column} BETWEEN 0 AND 1" for column in columns), name=name
    )


class Prediction(IdMixin, CreatedAtMixin, Base):
    """One model version's forecast for one match. Rows are immutable."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("match_id", "model_version_id", "predicted_at"),
        # `GET /predictions` and the dashboard read the newest predictions first;
        # without this the whole table is sorted on every request.
        Index("ix_predictions_predicted_at", text("predicted_at DESC"), text("id DESC")),
        _probability_range(
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "btts_probability",
            name="probabilities_in_range",
        ),
        CheckConstraint(
            "abs(home_win_probability + draw_probability + away_win_probability - 1)"
            f" <= {PROBABILITY_SUM_TOLERANCE}",
            name="outcome_probabilities_sum_to_one",
        ),
        CheckConstraint(
            "expected_home_goals >= 0 AND expected_away_goals >= 0",
            name="expected_goals_non_negative",
        ),
        CheckConstraint(
            "predicted_home_goals >= 0 AND predicted_away_goals >= 0",
            name="predicted_goals_non_negative",
        ),
    )

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="RESTRICT"))
    model_version_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("model_versions.id", ondelete="RESTRICT"), index=True
    )
    feature_snapshot_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("feature_snapshots.id", ondelete="RESTRICT"), index=True
    )
    predicted_at: Mapped[datetime]
    # Most likely exact score.
    predicted_home_goals: Mapped[int] = mapped_column(SmallInteger)
    predicted_away_goals: Mapped[int] = mapped_column(SmallInteger)
    # Expected goals (the mean of each team's goal distribution).
    expected_home_goals: Mapped[Decimal] = mapped_column(Numeric(6, 3))
    expected_away_goals: Mapped[Decimal] = mapped_column(Numeric(6, 3))
    home_win_probability: Mapped[Decimal] = mapped_column(PROBABILITY)
    draw_probability: Mapped[Decimal] = mapped_column(PROBABILITY)
    away_win_probability: Mapped[Decimal] = mapped_column(PROBABILITY)
    btts_probability: Mapped[Decimal] = mapped_column(PROBABILITY)

    match: Mapped[Match] = relationship()
    model_version: Mapped[ModelVersion] = relationship()
    feature_snapshot: Mapped[FeatureSnapshot | None] = relationship()
    score_probabilities: Mapped[list["PredictionScoreProbability"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="(PredictionScoreProbability.home_goals, PredictionScoreProbability.away_goals)",
    )
    over_under_probabilities: Mapped[list["PredictionOverUnderProbability"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="PredictionOverUnderProbability.line",
    )
    result: Mapped["PredictionResult | None"] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", passive_deletes=True
    )


class PredictionScoreProbability(Base):
    """One cell of a prediction's exact-score distribution."""

    __tablename__ = "prediction_score_probabilities"
    __table_args__ = (
        CheckConstraint("home_goals >= 0 AND away_goals >= 0", name="goals_non_negative"),
        _probability_range("probability", name="probability_in_range"),
    )

    prediction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("predictions.id", ondelete="CASCADE"), primary_key=True
    )
    home_goals: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    away_goals: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    probability: Mapped[Decimal] = mapped_column(PROBABILITY)

    prediction: Mapped[Prediction] = relationship(back_populates="score_probabilities")


class PredictionOverUnderProbability(Base):
    """Total-goals probabilities for one line, e.g. over/under 2.5.

    Over and under are stored separately because on whole-number lines the stake is
    refunded when the total equals the line, so they need not sum to 1.
    """

    __tablename__ = "prediction_over_under"
    __table_args__ = (
        CheckConstraint("line >= 0", name="line_non_negative"),
        _probability_range("over_probability", "under_probability", name="probabilities_in_range"),
        CheckConstraint(
            f"over_probability + under_probability <= 1 + {PROBABILITY_SUM_TOLERANCE}",
            name="probabilities_at_most_one",
        ),
    )

    prediction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("predictions.id", ondelete="CASCADE"), primary_key=True
    )
    line: Mapped[Decimal] = mapped_column(Numeric(4, 2), primary_key=True)
    over_probability: Mapped[Decimal] = mapped_column(PROBABILITY)
    under_probability: Mapped[Decimal] = mapped_column(PROBABILITY)

    prediction: Mapped[Prediction] = relationship(back_populates="over_under_probabilities")


class PredictionResult(CreatedAtMixin, Base):
    """Evaluation of a prediction once its match has finished (one per prediction).

    The actual score is not copied here; it is read from the match, so there is a
    single source of truth.
    """

    __tablename__ = "prediction_results"
    __table_args__ = (
        # Reports ask what has been scored since a moment.
        Index("ix_prediction_results_evaluated_at", "evaluated_at"),
        CheckConstraint(
            "brier_score >= 0 AND log_loss >= 0 AND ranked_probability_score >= 0",
            name="scores_non_negative",
        ),
        _probability_range("actual_score_probability", name="actual_score_probability_in_range"),
    )

    prediction_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("predictions.id", ondelete="CASCADE"), primary_key=True
    )
    evaluated_at: Mapped[datetime]
    outcome_correct: Mapped[bool]  # predicted home/draw/away matched
    exact_score_correct: Mapped[bool]
    btts_correct: Mapped[bool]
    brier_score: Mapped[Decimal] = mapped_column(Numeric(8, 6))  # home/draw/away
    log_loss: Mapped[Decimal] = mapped_column(Numeric(10, 6))
    ranked_probability_score: Mapped[Decimal] = mapped_column(Numeric(8, 6))
    # Probability the model gave to the score that actually happened.
    actual_score_probability: Mapped[Decimal | None] = mapped_column(PROBABILITY)
    # Additional metrics, e.g. per-line over/under correctness.
    metrics: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))

    prediction: Mapped[Prediction] = relationship(back_populates="result")
