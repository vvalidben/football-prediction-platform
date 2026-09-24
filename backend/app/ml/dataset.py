"""Supervised datasets built from stored feature snapshots.

Models never read raw match tables: inputs are exactly the leak-free features stored
in `feature_snapshots` (Sprint 4) with cutoff = kickoff, and targets are the final
scores. A dataset is therefore reproducible from its feature-set version and period.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import FeatureSnapshot, Match
from app.models.enums import MatchStatus

HOME, DRAW, AWAY = 0, 1, 2
OUTCOME_LABELS = ("home", "draw", "away")

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Example:
    match_id: int
    kickoff_at: datetime
    season_id: int
    competition_id: int
    features: dict[str, Any]
    home_goals: int
    away_goals: int

    @property
    def outcome(self) -> int:
        if self.home_goals > self.away_goals:
            return HOME
        return DRAW if self.home_goals == self.away_goals else AWAY


@dataclass(frozen=True)
class Dataset:
    """Chronologically ordered examples from one feature-set version."""

    examples: tuple[Example, ...]
    feature_set_version: str

    def __post_init__(self) -> None:
        kickoffs = [e.kickoff_at for e in self.examples]
        if kickoffs != sorted(kickoffs):
            raise ValueError("dataset examples must be in chronological order")

    def __len__(self) -> int:
        return len(self.examples)

    def matrix(self, feature_names: Sequence[str]) -> FloatArray:
        """Feature matrix; missing values (None) become NaN."""
        rows = [
            [
                np.nan if e.features.get(name) is None else float(e.features[name])
                for name in feature_names
            ]
            for e in self.examples
        ]
        return np.asarray(rows, dtype=np.float64).reshape(len(self.examples), len(feature_names))

    @property
    def outcomes(self) -> IntArray:
        return np.asarray([e.outcome for e in self.examples], dtype=np.int64)

    @property
    def home_goals(self) -> IntArray:
        return np.asarray([e.home_goals for e in self.examples], dtype=np.int64)

    @property
    def away_goals(self) -> IntArray:
        return np.asarray([e.away_goals for e in self.examples], dtype=np.int64)

    @property
    def period(self) -> tuple[datetime, datetime] | None:
        if not self.examples:
            return None
        return self.examples[0].kickoff_at, self.examples[-1].kickoff_at

    def where(self, keep: Callable[[Example], bool]) -> "Dataset":
        return Dataset(tuple(e for e in self.examples if keep(e)), self.feature_set_version)

    def __add__(self, other: "Dataset") -> "Dataset":
        if other.feature_set_version != self.feature_set_version:
            raise ValueError("cannot combine datasets of different feature-set versions")
        return Dataset(self.examples + other.examples, self.feature_set_version)

    def fingerprint(self) -> str:
        """Dataset version: a hash of the feature-set version and every match's id,
        kickoff and result, in order. The same training data always gives the same
        version; any added, removed or corrected match changes it."""
        digest = hashlib.sha256(self.feature_set_version.encode())
        for e in self.examples:
            digest.update(
                f"|{e.match_id},{e.kickoff_at.isoformat()},{e.home_goals},{e.away_goals}".encode()
            )
        return f"ds-{digest.hexdigest()[:16]}"

    @property
    def competition_ids(self) -> list[int]:
        return sorted({e.competition_id for e in self.examples})

    def describe(self) -> dict[str, Any]:
        period = self.period
        return {
            "matches": len(self),
            "start": period[0].isoformat() if period else None,
            "end": period[1].isoformat() if period else None,
            "home_win_rate": float(np.mean(self.outcomes == HOME)) if len(self) else None,
        }


def load_dataset(session: Session, feature_set_version: str) -> Dataset:
    """Finished matches with a snapshot of this version taken at kickoff."""
    rows = session.execute(
        select(FeatureSnapshot.features, Match)
        .join(Match, Match.id == FeatureSnapshot.match_id)
        .where(
            FeatureSnapshot.feature_set_version == feature_set_version,
            FeatureSnapshot.data_cutoff_at == Match.kickoff_at,
            Match.status == MatchStatus.FINISHED,
            Match.home_score.is_not(None),
            Match.away_score.is_not(None),
        )
        .order_by(Match.kickoff_at, Match.id)
    ).all()
    examples = tuple(
        Example(
            match_id=match.id,
            kickoff_at=match.kickoff_at,
            season_id=match.season_id,
            competition_id=match.competition_id,
            features=features,
            home_goals=match.home_score,
            away_goals=match.away_score,
        )
        for features, match in rows
    )
    return Dataset(examples, feature_set_version)
