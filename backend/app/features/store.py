"""Persisting and reproducing feature snapshots.

A snapshot is keyed by (match, feature-set version, cutoff) and is **immutable**:
predictions reference it, so it must keep describing exactly what a model saw.
Regenerating an existing snapshot never overwrites it. If the recomputed values
differ (because stored history was corrected since), the snapshot is left alone and
reported as *drifted*.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.features.builder import FeatureBuilder, FeatureRow, Features
from app.features.timeline import load_targets
from app.models import FeatureSnapshot


@dataclass
class SaveResult:
    created: int = 0
    unchanged: int = 0
    drifted: list[int] = field(default_factory=list)  # match ids


class FeatureStore:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def save(self, rows: Sequence[FeatureRow]) -> SaveResult:
        result = SaveResult()
        with self.session_factory() as session, session.begin():
            for row in rows:
                existing = session.scalars(
                    select(FeatureSnapshot).where(
                        FeatureSnapshot.match_id == row.target.match_id,
                        FeatureSnapshot.feature_set_version == row.version,
                        FeatureSnapshot.data_cutoff_at == row.target.cutoff,
                    )
                ).one_or_none()
                if existing is None:
                    session.add(
                        FeatureSnapshot(
                            match_id=row.target.match_id,
                            feature_set_version=row.version,
                            data_cutoff_at=row.target.cutoff,
                            features=dict(row.values),
                        )
                    )
                    result.created += 1
                elif existing.features == row.values:
                    result.unchanged += 1
                else:
                    result.drifted.append(row.target.match_id)
        return result


def reproduce(builder: FeatureBuilder, snapshot: FeatureSnapshot) -> Features:
    """Recompute a stored snapshot with the current data and definitions.

    Raises if the snapshot was made with a different feature-set version, since the
    current code could not reproduce it.
    """
    if snapshot.feature_set_version != builder.version:
        raise ValueError(
            f"snapshot has version {snapshot.feature_set_version}, "
            f"this code builds {builder.version}"
        )
    with builder.session_factory() as session:
        (target,) = load_targets(session, match_ids=[snapshot.match_id])
    lead_time = target.kickoff_at - snapshot.data_cutoff_at
    with builder.session_factory() as session:
        (target,) = load_targets(session, match_ids=[snapshot.match_id], lead_time=lead_time)
    (row,) = builder.build([target])
    return row.values
