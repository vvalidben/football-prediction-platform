"""Prediction history: what was predicted, what happened, and how wrong it was.

Rows are read from the stored prediction and its evaluation; the actual score comes
from the match, which stays the single source of truth. A prediction whose match has
not been played (or not yet evaluated) is still listed, with its result fields null —
the history shows what exists rather than hiding what is pending.
"""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.catalog import Page


class PredictionHistoryRow(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    prediction_id: int
    match_id: int
    kickoff_at: dt.datetime
    competition_id: int | None
    competition: str | None
    season_id: int | None
    season: str | None
    home_team_id: int
    away_team_id: int
    home_team: str
    away_team: str

    # What was predicted.
    predicted_home_goals: int
    predicted_away_goals: int
    expected_home_goals: float
    expected_away_goals: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    btts_probability: float
    #: "home", "draw" or "away": the outcome the model gave the highest probability.
    predicted_outcome: str

    # What happened.
    status: str
    home_score: int | None
    away_score: int | None
    actual_outcome: str | None

    # How it did. Null until the prediction has been evaluated.
    evaluated_at: dt.datetime | None
    outcome_correct: bool | None
    exact_score_correct: bool | None
    btts_correct: bool | None
    brier_score: float | None
    log_loss: float | None
    ranked_probability_score: float | None
    actual_score_probability: float | None
    #: Expected goals minus actual goals, per side; positive means the model was high.
    home_goals_error: float | None
    away_goals_error: float | None
    #: Probability the model gave to the outcome that happened.
    actual_outcome_probability: float | None

    model_id: int
    model_name: str
    model_version: str
    predicted_at: dt.datetime


class HistorySummary(BaseModel):
    """Totals over every row matching the filters, not just the page shown."""

    model_config = ConfigDict(frozen=True)

    predictions: int
    evaluated: int
    pending: int = Field(description="Stored but not yet scored against a result")
    accuracy: float | None
    exact_score_accuracy: float | None
    btts_accuracy: float | None
    log_loss: float | None
    brier_score: float | None
    ranked_probability_score: float | None
    #: Mean signed and absolute goal errors, over evaluated rows.
    mean_home_goals_error: float | None
    mean_away_goals_error: float | None
    mean_absolute_goals_error: float | None
    first_kickoff_at: dt.datetime | None
    last_kickoff_at: dt.datetime | None


class PredictionHistoryPage(Page):
    items: list[PredictionHistoryRow]
    summary: HistorySummary
