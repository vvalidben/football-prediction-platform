"""Columns shared by many tables."""

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Enum, ForeignKey, Identity, func
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship

if TYPE_CHECKING:
    from app.models.data_source import DataSource


class IdMixin:
    """Surrogate primary key: a bigint identity column."""

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True, sort_order=-100
    )


class CreatedAtMixin:
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), sort_order=100)


class TimestampMixin(CreatedAtMixin):
    # onupdate applies to ORM updates; bulk SQL updates must set updated_at themselves.
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), sort_order=101
    )


class ProviderRecordMixin:
    """For rows that are one provider's observation (events, statistics, odds, …).

    The source is part of each table's natural key, so the same fact reported by two
    providers is stored twice, once per provider, and can be compared or reconciled.
    `external_id`, when the provider has one, is unique per provider.
    """

    @declared_attr
    def data_source_id(cls) -> Mapped[int]:
        return mapped_column(
            BigInteger, ForeignKey("data_sources.id", ondelete="RESTRICT"), sort_order=-90
        )

    @declared_attr
    def data_source(cls) -> Mapped["DataSource"]:
        return relationship()

    external_id: Mapped[str | None] = mapped_column(sort_order=-89)


def str_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Store a StrEnum as text guarded by a CHECK constraint.

    Preferred over native PostgreSQL enums, which cannot drop values and are awkward
    to alter in migrations. `name` becomes the CHECK constraint's name suffix.
    """
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )


class ExternalIdMixin(IdMixin, CreatedAtMixin):
    """Base for `<entity>_external_ids` tables mapping a canonical row to provider IDs.

    Each concrete table adds the FK to its entity plus two unique constraints:
    `(data_source_id, external_id)` so a provider ID resolves to exactly one entity,
    and `(<entity>_id, data_source_id)` so an entity has one ID per provider.
    """

    @declared_attr
    def data_source_id(cls) -> Mapped[int]:
        return mapped_column(
            BigInteger, ForeignKey("data_sources.id", ondelete="RESTRICT"), sort_order=-90
        )

    @declared_attr
    def data_source(cls) -> Mapped["DataSource"]:
        return relationship()

    external_id: Mapped[str] = mapped_column(sort_order=-89)
