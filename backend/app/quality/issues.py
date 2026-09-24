"""Recording data-quality issues without duplicating or deleting them."""

import hashlib
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from app.models import DataQualityIssue
from app.models.enums import IssueEntity, IssueSeverity, IssueType

INGESTION = "ingestion"
AUDIT = "audit"


def ingestion_detector(job: str) -> str:
    """Detector name for one ingestion operation, e.g. "ingestion:matches".

    Scoping issues by operation matters: re-syncing matches must only resolve
    issues the match sync can detect, not e.g. statistics gaps on the same match.
    """
    return f"{INGESTION}:{job}"


@dataclass(frozen=True)
class IssueDraft:
    """A detected problem, before it is stored.

    The fingerprint identifies *the problem*, not one sighting of it: the same
    problem found again updates the existing open issue. `key` distinguishes
    several problems of one type on one record (e.g. which rule fired).
    """

    issue_type: IssueType
    severity: IssueSeverity
    entity_type: IssueEntity
    message: str
    external_id: str | None = None
    entity_id: int | None = None
    key: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self, data_source_id: int | None) -> str:
        parts = (
            str(data_source_id or ""),
            self.issue_type.value,
            self.entity_type.value,
            self.external_id or "",
            str(self.entity_id or "") if self.external_id is None else "",
            self.key,
        )
        return hashlib.sha256("|".join(parts).encode()).hexdigest()


class IssueStore:
    """Upserts issues by fingerprint and resolves those no longer detected."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock

    def record(
        self,
        issues: Iterable[IssueDraft],
        *,
        detector: str,
        data_source_id: int | None = None,
        run_id: int | None = None,
    ) -> set[str]:
        """Store issues; return their fingerprints."""
        now = self._clock()
        fingerprints: set[str] = set()
        with self._session_factory() as session, session.begin():
            for issue in issues:
                fingerprint = issue.fingerprint(data_source_id)
                if fingerprint in fingerprints:
                    continue
                fingerprints.add(fingerprint)
                values = {
                    "data_source_id": data_source_id,
                    "ingestion_run_id": run_id,
                    "detector": detector,
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "entity_type": issue.entity_type,
                    "entity_id": issue.entity_id,
                    "external_id": issue.external_id,
                    "message": issue.message,
                    "details": issue.details,
                    "fingerprint": fingerprint,
                    "first_seen_at": now,
                    "last_seen_at": now,
                }
                statement = insert(DataQualityIssue).values(**values)
                session.execute(
                    statement.on_conflict_do_update(
                        index_elements=[DataQualityIssue.fingerprint],
                        index_where=DataQualityIssue.resolved_at.is_(None),
                        set_={
                            "last_seen_at": now,
                            "occurrences": DataQualityIssue.occurrences + 1,
                            "message": statement.excluded.message,
                            "details": statement.excluded.details,
                            "severity": statement.excluded.severity,
                            "entity_id": func.coalesce(
                                statement.excluded.entity_id, DataQualityIssue.entity_id
                            ),
                            "ingestion_run_id": statement.excluded.ingestion_run_id,
                        },
                    )
                )
        return fingerprints

    def resolve_absent(
        self,
        *,
        detector: str,
        present: Collection[str],
        data_source_id: int | None = None,
        scope: Collection[tuple[IssueEntity, str]] | None = None,
    ) -> int:
        """Mark open issues of `detector` as resolved unless their fingerprint is in
        `present`. With `scope`, only issues of those (entity type, external id)
        pairs are considered: the records that were just re-checked.
        """
        if scope is not None and not scope:
            return 0
        conditions = [
            DataQualityIssue.detector == detector,
            DataQualityIssue.resolved_at.is_(None),
        ]
        if present:
            conditions.append(DataQualityIssue.fingerprint.not_in(list(present)))
        if data_source_id is not None:
            conditions.append(DataQualityIssue.data_source_id == data_source_id)
        with self._session_factory() as session, session.begin():
            resolved = 0
            if scope is None:
                result = session.execute(
                    update(DataQualityIssue).where(*conditions).values(resolved_at=self._clock())
                )
                return int(result.rowcount)  # type: ignore[attr-defined]
            by_entity: dict[IssueEntity, list[str]] = {}
            for entity_type, external_id in scope:
                by_entity.setdefault(entity_type, []).append(external_id)
            for entity_type, external_ids in by_entity.items():
                result = session.execute(
                    update(DataQualityIssue)
                    .where(
                        *conditions,
                        DataQualityIssue.entity_type == entity_type,
                        DataQualityIssue.external_id.in_(external_ids),
                    )
                    .values(resolved_at=self._clock())
                )
                resolved += int(result.rowcount)  # type: ignore[attr-defined]
            return resolved
