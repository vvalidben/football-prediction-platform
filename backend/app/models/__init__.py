"""ORM models. Importing this package registers every table on `Base.metadata`."""

from app.models.competition import Competition, CompetitionExternalId, Season, SeasonExternalId
from app.models.data_source import DataIngestionRun, DataSource
from app.models.injury import Injury
from app.models.jobs import JobRun
from app.models.match import Lineup, LineupPlayer, Match, MatchEvent, MatchExternalId
from app.models.ml import FeatureSnapshot, ModelActivation, ModelVersion
from app.models.odds import Bookmaker, Odds
from app.models.player import Player, PlayerExternalId
from app.models.prediction import (
    Prediction,
    PredictionOverUnderProbability,
    PredictionResult,
    PredictionScoreProbability,
)
from app.models.quality import DataQualityIssue
from app.models.standing import Standing
from app.models.statistics import PlayerMatchStatistics, TeamMatchStatistics
from app.models.team import Team, TeamExternalId

__all__ = [
    "Bookmaker",
    "Competition",
    "CompetitionExternalId",
    "DataIngestionRun",
    "DataQualityIssue",
    "DataSource",
    "FeatureSnapshot",
    "Injury",
    "JobRun",
    "Lineup",
    "LineupPlayer",
    "Match",
    "MatchEvent",
    "MatchExternalId",
    "ModelActivation",
    "ModelVersion",
    "Odds",
    "Player",
    "PlayerExternalId",
    "PlayerMatchStatistics",
    "Prediction",
    "PredictionOverUnderProbability",
    "PredictionResult",
    "PredictionScoreProbability",
    "Season",
    "SeasonExternalId",
    "Standing",
    "Team",
    "TeamExternalId",
    "TeamMatchStatistics",
]
