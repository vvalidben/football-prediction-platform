"""Model registry and the feature inputs used to make predictions."""

from datetime import date, datetime
from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.match import Match
from app.models.mixins import CreatedAtMixin, IdMixin, TimestampMixin


class ModelVersion(IdMixin, TimestampMixin, Base):
    """A trained, versioned model. Predictions reference the exact version used."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version"),
        CheckConstraint(
            "training_data_start IS NULL OR training_data_end IS NULL"
            " OR training_data_end >= training_data_start",
            name="training_window_ordered",
        ),
        Index(
            "uq_model_versions_one_active_per_name",
            "name",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    name: Mapped[str]  # model family, e.g. "dixon_coles"
    version: Mapped[str]  # e.g. "1.0.0"
    description: Mapped[str | None]
    #: The implementation, e.g. "EloModel", "XGBoostModel", "EnsembleModel".
    algorithm: Mapped[str]
    feature_set_version: Mapped[str | None]
    #: Fingerprint of the exact training data (matches, results, feature version).
    dataset_version: Mapped[str]
    #: Competitions in the training data: [{"id": ..., "name": ...}].
    competitions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    training_metrics: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    # Match dates covered by the training data, to rule out evaluating on seen matches.
    training_data_start: Mapped[date | None]
    training_data_end: Mapped[date | None]
    trained_at: Mapped[datetime | None]
    artifact_uri: Mapped[str | None]  # where the serialised model is stored
    is_active: Mapped[bool] = mapped_column(server_default=text("false"))
    #: The same version in the MLflow Model Registry, when mirrored there.
    mlflow_model_name: Mapped[str | None]
    mlflow_model_version: Mapped[str | None]

    activations: Mapped[list["ModelActivation"]] = relationship(
        back_populates="model_version",
        foreign_keys="ModelActivation.model_version_id",
        order_by="ModelActivation.activated_at",
    )


class ModelActivation(IdMixin, CreatedAtMixin, Base):
    """One period during which a model version was the active version of its name.

    The open row (no `deactivated_at`) is the current activation; closed rows are the
    history used to list previous models and to roll back.
    """

    __tablename__ = "model_activations"
    __table_args__ = (
        CheckConstraint(
            "deactivated_at IS NULL OR deactivated_at >= activated_at",
            name="deactivation_after_activation",
        ),
        Index(
            "uq_model_activations_one_open_per_name",
            "model_name",
            unique=True,
            postgresql_where=text("deactivated_at IS NULL"),
        ),
        Index("ix_model_activations_model_name_activated_at", "model_name", "activated_at"),
    )

    model_name: Mapped[str]
    model_version_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("model_versions.id", ondelete="RESTRICT"), index=True
    )
    #: The version that was active just before this activation (None: first one).
    previous_version_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("model_versions.id", ondelete="RESTRICT"), index=True
    )
    activated_at: Mapped[datetime]
    deactivated_at: Mapped[datetime | None]
    reason: Mapped[str | None]
    is_rollback: Mapped[bool] = mapped_column(server_default=text("false"))

    model_version: Mapped[ModelVersion] = relationship(
        back_populates="activations", foreign_keys=[model_version_id]
    )
    previous_version: Mapped[ModelVersion | None] = relationship(foreign_keys=[previous_version_id])


class FeatureSnapshot(IdMixin, CreatedAtMixin, Base):
    """The feature vector for a match, computed from data available at `data_cutoff_at`.

    Storing the exact inputs makes every prediction reproducible, and the cut-off
    documents that no information from after it was used (no leakage).
    """

    __tablename__ = "feature_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "match_id",
            "feature_set_version",
            "data_cutoff_at",
            name="uq_feature_snapshots_natural_key",
        ),
        CheckConstraint("jsonb_typeof(features) = 'object'", name="features_is_object"),
    )

    match_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("matches.id", ondelete="CASCADE"))
    feature_set_version: Mapped[str]
    data_cutoff_at: Mapped[datetime]
    features: Mapped[dict[str, Any]]

    match: Mapped[Match] = relationship()
