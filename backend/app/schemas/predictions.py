"""Prediction requests and responses.

A prediction is one model version's forecast for one match, derived from a single
exact-score distribution (see docs/prediction-engine.md), so the 1X2, over/under and
both-teams-to-score figures can never contradict each other.
"""

import datetime as dt
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_BATCH = 50


class MatchSelector(BaseModel):
    """Which match to predict: either `match_id`, or both team names.

    `competition` and `date` narrow the search when the same teams meet more than
    once; without a date, the next scheduled meeting is used.
    """

    match_id: int | None = Field(default=None, ge=1)
    home_team: str | None = Field(default=None, min_length=1, max_length=120)
    away_team: str | None = Field(default=None, min_length=1, max_length=120)
    competition: str | None = Field(default=None, min_length=1, max_length=120)
    date: dt.date | None = None

    @model_validator(mode="after")
    def check_identification(self) -> Self:
        by_teams = bool(self.home_team) and bool(self.away_team)
        if self.match_id is not None and (by_teams or self.competition or self.date):
            raise ValueError("give either match_id or team names, not both")
        if self.match_id is None and not by_teams:
            raise ValueError("give match_id, or home_team and away_team")
        return self


class PredictionRequest(MatchSelector):
    model_config = ConfigDict(
        frozen=True,
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {"match_id": 12345},
                {
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "competition": "Premier League",
                    "date": "2026-05-24",
                },
            ]
        },
    )

    model_name: str | None = Field(
        default=None,
        description="Registered model name; defaults to the only active model",
        max_length=120,
    )
    refresh: bool = Field(
        default=False,
        description="Recompute even if a prediction from the same features exists",
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    matches: list[MatchSelector] = Field(min_length=1, max_length=MAX_BATCH)
    model_name: str | None = Field(default=None, max_length=120)
    refresh: bool = False


class ScoreProbability(BaseModel):
    model_config = ConfigDict(frozen=True)

    home_goals: int
    away_goals: int
    probability: float


class OverUnderProbability(BaseModel):
    model_config = ConfigDict(frozen=True)

    line: float
    over: float
    under: float


class BothTeamsToScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    yes: float
    no: float


class ModelReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    version: str
    algorithm: str
    feature_set_version: str | None


class DataFreshness(BaseModel):
    """How current the information behind the prediction is."""

    model_config = ConfigDict(frozen=True)

    feature_cutoff_at: dt.datetime = Field(description="No information after this time was used")
    latest_result_at: dt.datetime | None = Field(
        default=None, description="Kickoff of the most recent finished match included"
    )
    last_ingestion_at: dt.datetime | None = Field(
        default=None, description="When data was last imported successfully"
    )
    age_seconds: float = Field(description="Seconds between the feature cutoff and now")
    match_started: bool


class PredictionRead(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    id: int
    match_id: int
    kickoff_at: dt.datetime
    competition: str | None
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    most_probable_score: ScoreProbability
    top_scores: list[ScoreProbability]
    #: matrix[home goals][away goals], 0..max_goals per side; sums to 1.
    score_probability_matrix: list[list[float]]
    max_goals: int
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    over_under_probabilities: list[OverUnderProbability]
    btts_probability: BothTeamsToScore
    model: ModelReference
    predicted_at: dt.datetime
    data_freshness: DataFreshness
    reused: bool = Field(
        default=False, description="True when an identical stored prediction was returned"
    )


class BatchItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    prediction: PredictionRead | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    requested: int
    predicted: int
    failed: int
    items: list[BatchItem]


class StoredPredictionRead(BaseModel):
    """A prediction as it appears in lists: the headline numbers and its outcome."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    id: int
    match_id: int
    kickoff_at: dt.datetime
    competition: str | None
    home_team: str
    away_team: str
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    most_probable_score: str
    expected_home_goals: float
    expected_away_goals: float
    model_name: str
    model_version: str
    predicted_at: dt.datetime
    #: Filled once the match has been played and the prediction scored.
    actual_score: str | None = None
    outcome_correct: bool | None = None


class StoredPredictionPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    limit: int
    offset: int
    items: list[StoredPredictionRead]
