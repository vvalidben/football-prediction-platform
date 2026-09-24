"""Initial schema: football data, provider mappings, ingestion log, models and predictions.

Revision ID: ea24dba4071d
Revises:
Create Date: 2026-09-21 23:14:49.460828+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "ea24dba4071d"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "bookmakers",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bookmakers")),
        sa.UniqueConstraint("name", name=op.f("uq_bookmakers_name")),
    )
    op.create_table(
        "competitions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("short_name", sa.Text(), nullable=True),
        sa.Column(
            "type",
            sa.Enum(
                "league",
                "cup",
                "super_cup",
                "friendly",
                "other",
                name="competition_type",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("country", sa.Text(), nullable=True),
        sa.Column("logo_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_competitions")),
    )
    op.create_table(
        "data_sources",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("code", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("base_url", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("code ~ '^[a-z0-9_]+$'", name=op.f("ck_data_sources_code_format")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_data_sources")),
        sa.UniqueConstraint("code", name=op.f("uq_data_sources_code")),
    )
    op.create_table(
        "model_versions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("feature_set_version", sa.Text(), nullable=True),
        sa.Column(
            "hyperparameters",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "training_metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("training_data_start", sa.Date(), nullable=True),
        sa.Column("training_data_end", sa.Date(), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "training_data_start IS NULL OR training_data_end IS NULL OR training_data_end >= training_data_start",
            name=op.f("ck_model_versions_training_window_ordered"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.UniqueConstraint("name", "version", name=op.f("uq_model_versions_name_version")),
    )
    op.create_index(
        "uq_model_versions_one_active_per_name",
        "model_versions",
        ["name"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )
    op.create_table(
        "teams",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("short_name", sa.Text(), nullable=True),
        sa.Column("code", sa.Text(), nullable=True),
        sa.Column("country", sa.Text(), nullable=True),
        sa.Column("logo_url", sa.Text(), nullable=True),
        sa.Column(
            "is_national_team", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_teams")),
    )
    op.create_table(
        "competition_external_ids",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competitions.id"],
            name=op.f("fk_competition_external_ids_competition_id_competitions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_competition_external_ids_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_competition_external_ids")),
        sa.UniqueConstraint(
            "competition_id",
            "data_source_id",
            name=op.f("uq_competition_external_ids_competition_id_data_source_id"),
        ),
        sa.UniqueConstraint(
            "data_source_id",
            "external_id",
            name=op.f("uq_competition_external_ids_data_source_id_external_id"),
        ),
    )
    op.create_table(
        "data_ingestion_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("job", sa.Text(), nullable=False),
        sa.Column(
            "parameters",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "running",
                "succeeded",
                "partial",
                "failed",
                name="ingestion_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="running",
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("records_fetched", sa.Integer(), server_default="0", nullable=False),
        sa.Column("records_inserted", sa.Integer(), server_default="0", nullable=False),
        sa.Column("records_updated", sa.Integer(), server_default="0", nullable=False),
        sa.Column("records_failed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(status = 'running') = (finished_at IS NULL)",
            name=op.f("ck_data_ingestion_runs_finished_at_matches_status"),
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR finished_at >= started_at",
            name=op.f("ck_data_ingestion_runs_finished_after_started"),
        ),
        sa.CheckConstraint(
            "records_fetched >= 0 AND records_inserted >= 0 AND records_updated >= 0 AND records_failed >= 0",
            name=op.f("ck_data_ingestion_runs_record_counts_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_data_ingestion_runs_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_data_ingestion_runs")),
    )
    op.create_index(
        "ix_data_ingestion_runs_running",
        "data_ingestion_runs",
        ["data_source_id"],
        unique=False,
        postgresql_where=sa.text("status = 'running'"),
    )
    op.create_index(
        "ix_data_ingestion_runs_source_job_started",
        "data_ingestion_runs",
        ["data_source_id", "job", "started_at"],
        unique=False,
    )
    op.create_table(
        "players",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("first_name", sa.Text(), nullable=True),
        sa.Column("last_name", sa.Text(), nullable=True),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("nationality", sa.Text(), nullable=True),
        sa.Column(
            "position",
            sa.Enum(
                "goalkeeper",
                "defender",
                "midfielder",
                "forward",
                name="player_position",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=True,
        ),
        sa.Column("current_team_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["current_team_id"],
            ["teams.id"],
            name=op.f("fk_players_current_team_id_teams"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_players")),
    )
    op.create_index(
        op.f("ix_players_current_team_id"), "players", ["current_team_id"], unique=False
    )
    op.create_table(
        "seasons",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("is_current", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "start_date IS NULL OR end_date IS NULL OR end_date >= start_date",
            name=op.f("ck_seasons_end_after_start"),
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competitions.id"],
            name=op.f("fk_seasons_competition_id_competitions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_seasons")),
        sa.UniqueConstraint("competition_id", "name", name=op.f("uq_seasons_competition_id_name")),
        sa.UniqueConstraint("id", "competition_id", name=op.f("uq_seasons_id_competition_id")),
    )
    op.create_index(
        "uq_seasons_one_current_per_competition",
        "seasons",
        ["competition_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )
    op.create_table(
        "team_external_ids",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_team_external_ids_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["teams.id"],
            name=op.f("fk_team_external_ids_team_id_teams"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_team_external_ids")),
        sa.UniqueConstraint(
            "data_source_id",
            "external_id",
            name=op.f("uq_team_external_ids_data_source_id_external_id"),
        ),
        sa.UniqueConstraint(
            "team_id", "data_source_id", name=op.f("uq_team_external_ids_team_id_data_source_id")
        ),
    )
    op.create_table(
        "matches",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("competition_id", sa.BigInteger(), nullable=False),
        sa.Column("season_id", sa.BigInteger(), nullable=False),
        sa.Column("home_team_id", sa.BigInteger(), nullable=False),
        sa.Column("away_team_id", sa.BigInteger(), nullable=False),
        sa.Column("kickoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "scheduled",
                "live",
                "finished",
                "postponed",
                "suspended",
                "cancelled",
                "abandoned",
                name="match_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            server_default="scheduled",
            nullable=False,
        ),
        sa.Column("matchday", sa.SmallInteger(), nullable=True),
        sa.Column("stage", sa.Text(), nullable=True),
        sa.Column("home_score", sa.SmallInteger(), nullable=True),
        sa.Column("away_score", sa.SmallInteger(), nullable=True),
        sa.Column("home_half_time_score", sa.SmallInteger(), nullable=True),
        sa.Column("away_half_time_score", sa.SmallInteger(), nullable=True),
        sa.Column("home_penalty_score", sa.SmallInteger(), nullable=True),
        sa.Column("away_penalty_score", sa.SmallInteger(), nullable=True),
        sa.Column("venue", sa.Text(), nullable=True),
        sa.Column("referee", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status <> 'finished' OR (home_score IS NOT NULL AND away_score IS NOT NULL)",
            name=op.f("ck_matches_finished_has_score"),
        ),
        sa.CheckConstraint(
            "(home_half_time_score IS NULL) = (away_half_time_score IS NULL)",
            name=op.f("ck_matches_half_time_scores_paired"),
        ),
        sa.CheckConstraint(
            "(home_penalty_score IS NULL) = (away_penalty_score IS NULL)",
            name=op.f("ck_matches_penalty_scores_paired"),
        ),
        sa.CheckConstraint(
            "(home_score IS NULL) = (away_score IS NULL)", name=op.f("ck_matches_scores_paired")
        ),
        sa.CheckConstraint(
            "home_half_time_score >= 0 AND away_half_time_score >= 0",
            name=op.f("ck_matches_half_time_scores_non_negative"),
        ),
        sa.CheckConstraint(
            "home_penalty_score >= 0 AND away_penalty_score >= 0",
            name=op.f("ck_matches_penalty_scores_non_negative"),
        ),
        sa.CheckConstraint(
            "home_score >= 0 AND away_score >= 0", name=op.f("ck_matches_scores_non_negative")
        ),
        sa.CheckConstraint("home_team_id <> away_team_id", name=op.f("ck_matches_distinct_teams")),
        sa.CheckConstraint(
            "matchday IS NULL OR matchday > 0", name=op.f("ck_matches_matchday_positive")
        ),
        sa.ForeignKeyConstraint(
            ["away_team_id"],
            ["teams.id"],
            name=op.f("fk_matches_away_team_id_teams"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competitions.id"],
            name=op.f("fk_matches_competition_id_competitions"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["home_team_id"],
            ["teams.id"],
            name=op.f("fk_matches_home_team_id_teams"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["season_id", "competition_id"],
            ["seasons.id", "seasons.competition_id"],
            name="fk_matches_season_id_competition_id_seasons",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_matches")),
        sa.UniqueConstraint(
            "home_team_id",
            "away_team_id",
            "kickoff_at",
            name=op.f("uq_matches_home_team_id_away_team_id_kickoff_at"),
        ),
    )
    op.create_index(
        "ix_matches_away_team_id_kickoff_at",
        "matches",
        ["away_team_id", "kickoff_at"],
        unique=False,
    )
    op.create_index(
        "ix_matches_competition_id_kickoff_at",
        "matches",
        ["competition_id", "kickoff_at"],
        unique=False,
    )
    op.create_index(
        "ix_matches_home_team_id_kickoff_at",
        "matches",
        ["home_team_id", "kickoff_at"],
        unique=False,
    )
    op.create_index(
        "ix_matches_season_id_competition_id",
        "matches",
        ["season_id", "competition_id"],
        unique=False,
    )
    op.create_index(
        "ix_matches_status_kickoff_at", "matches", ["status", "kickoff_at"], unique=False
    )
    op.create_table(
        "player_external_ids",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_player_external_ids_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["players.id"],
            name=op.f("fk_player_external_ids_player_id_players"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_player_external_ids")),
        sa.UniqueConstraint(
            "data_source_id",
            "external_id",
            name=op.f("uq_player_external_ids_data_source_id_external_id"),
        ),
        sa.UniqueConstraint(
            "player_id",
            "data_source_id",
            name=op.f("uq_player_external_ids_player_id_data_source_id"),
        ),
    )
    op.create_table(
        "season_external_ids",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("season_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_season_external_ids_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"],
            ["seasons.id"],
            name=op.f("fk_season_external_ids_season_id_seasons"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_season_external_ids")),
        sa.UniqueConstraint(
            "data_source_id",
            "external_id",
            name=op.f("uq_season_external_ids_data_source_id_external_id"),
        ),
        sa.UniqueConstraint(
            "season_id",
            "data_source_id",
            name=op.f("uq_season_external_ids_season_id_data_source_id"),
        ),
    )
    op.create_table(
        "standings",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("season_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("group_name", sa.Text(), nullable=True),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.Column("played", sa.SmallInteger(), nullable=False),
        sa.Column("won", sa.SmallInteger(), nullable=False),
        sa.Column("drawn", sa.SmallInteger(), nullable=False),
        sa.Column("lost", sa.SmallInteger(), nullable=False),
        sa.Column("goals_for", sa.SmallInteger(), nullable=False),
        sa.Column("goals_against", sa.SmallInteger(), nullable=False),
        sa.Column(
            "goal_difference",
            sa.SmallInteger(),
            sa.Computed("goals_for - goals_against", persisted=True),
            nullable=False,
        ),
        sa.Column("points", sa.SmallInteger(), nullable=False),
        sa.Column("form", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "played = won + drawn + lost", name=op.f("ck_standings_played_equals_results")
        ),
        sa.CheckConstraint("position > 0", name=op.f("ck_standings_position_positive")),
        sa.CheckConstraint(
            "won >= 0 AND drawn >= 0 AND lost >= 0 AND goals_for >= 0 AND goals_against >= 0",
            name=op.f("ck_standings_counts_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_standings_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["season_id"],
            ["seasons.id"],
            name=op.f("fk_standings_season_id_seasons"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name=op.f("fk_standings_team_id_teams"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_standings")),
        sa.UniqueConstraint(
            "data_source_id", "external_id", name=op.f("uq_standings_data_source_id_external_id")
        ),
        sa.UniqueConstraint(
            "season_id",
            "group_name",
            "team_id",
            "data_source_id",
            "as_of_date",
            name="uq_standings_natural_key",
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index(op.f("ix_standings_team_id"), "standings", ["team_id"], unique=False)
    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("feature_set_version", sa.Text(), nullable=False),
        sa.Column("data_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(features) = 'object'",
            name=op.f("ck_feature_snapshots_features_is_object"),
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.id"],
            name=op.f("fk_feature_snapshots_match_id_matches"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feature_snapshots")),
        sa.UniqueConstraint(
            "match_id",
            "feature_set_version",
            "data_cutoff_at",
            name="uq_feature_snapshots_natural_key",
        ),
    )
    op.create_table(
        "injuries",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=True),
        sa.Column("match_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "absence_type",
            sa.Enum(
                "injury",
                "illness",
                "suspension",
                "other",
                name="absence_type",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "out",
                "doubtful",
                name="availability_status",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("expected_return_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("end_date >= start_date", name=op.f("ck_injuries_end_after_start")),
        sa.CheckConstraint(
            "expected_return_date >= start_date",
            name=op.f("ck_injuries_expected_return_after_start"),
        ),
        sa.CheckConstraint(
            "match_id IS NOT NULL OR start_date IS NOT NULL",
            name=op.f("ck_injuries_match_or_start_date"),
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_injuries_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.id"],
            name=op.f("fk_injuries_match_id_matches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["players.id"],
            name=op.f("fk_injuries_player_id_players"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name=op.f("fk_injuries_team_id_teams"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_injuries")),
        sa.UniqueConstraint(
            "data_source_id", "external_id", name=op.f("uq_injuries_data_source_id_external_id")
        ),
        sa.UniqueConstraint(
            "player_id",
            "data_source_id",
            "match_id",
            "start_date",
            name="uq_injuries_natural_key",
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index(op.f("ix_injuries_match_id"), "injuries", ["match_id"], unique=False)
    op.create_index(
        "ix_injuries_team_id_start_date", "injuries", ["team_id", "start_date"], unique=False
    )
    op.create_table(
        "lineups",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("formation", sa.Text(), nullable=True),
        sa.Column("coach_name", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_lineups_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.id"],
            name=op.f("fk_lineups_match_id_matches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"], ["teams.id"], name=op.f("fk_lineups_team_id_teams"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lineups")),
        sa.UniqueConstraint(
            "data_source_id", "external_id", name=op.f("uq_lineups_data_source_id_external_id")
        ),
        sa.UniqueConstraint(
            "match_id",
            "team_id",
            "data_source_id",
            name=op.f("uq_lineups_match_id_team_id_data_source_id"),
        ),
    )
    op.create_index(op.f("ix_lineups_team_id"), "lineups", ["team_id"], unique=False)
    op.create_table(
        "match_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "goal",
                "own_goal",
                "penalty_goal",
                "penalty_missed",
                "yellow_card",
                "second_yellow_card",
                "red_card",
                "substitution",
                "var_decision",
                "other",
                name="match_event_type",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column(
            "period",
            sa.Enum(
                "first_half",
                "second_half",
                "extra_time_first_half",
                "extra_time_second_half",
                "penalty_shootout",
                name="match_period",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=True,
        ),
        sa.Column("minute", sa.SmallInteger(), nullable=False),
        sa.Column("added_time", sa.SmallInteger(), nullable=True),
        sa.Column("team_id", sa.BigInteger(), nullable=True),
        sa.Column("player_id", sa.BigInteger(), nullable=True),
        sa.Column("related_player_id", sa.BigInteger(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "added_time IS NULL OR added_time >= 0",
            name=op.f("ck_match_events_added_time_non_negative"),
        ),
        sa.CheckConstraint("minute >= 0", name=op.f("ck_match_events_minute_non_negative")),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_match_events_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.id"],
            name=op.f("fk_match_events_match_id_matches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["players.id"],
            name=op.f("fk_match_events_player_id_players"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["related_player_id"],
            ["players.id"],
            name=op.f("fk_match_events_related_player_id_players"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["teams.id"],
            name=op.f("fk_match_events_team_id_teams"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_match_events")),
        sa.UniqueConstraint(
            "data_source_id", "external_id", name=op.f("uq_match_events_data_source_id_external_id")
        ),
    )
    op.create_index(
        "ix_match_events_match_id_minute", "match_events", ["match_id", "minute"], unique=False
    )
    op.create_index(op.f("ix_match_events_player_id"), "match_events", ["player_id"], unique=False)
    op.create_index(
        op.f("ix_match_events_related_player_id"),
        "match_events",
        ["related_player_id"],
        unique=False,
    )
    op.create_index(op.f("ix_match_events_team_id"), "match_events", ["team_id"], unique=False)
    op.create_table(
        "match_external_ids",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_match_external_ids_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.id"],
            name=op.f("fk_match_external_ids_match_id_matches"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_match_external_ids")),
        sa.UniqueConstraint(
            "data_source_id",
            "external_id",
            name=op.f("uq_match_external_ids_data_source_id_external_id"),
        ),
        sa.UniqueConstraint(
            "match_id", "data_source_id", name=op.f("uq_match_external_ids_match_id_data_source_id")
        ),
    )
    op.create_table(
        "odds",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("bookmaker_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "market",
            sa.Enum(
                "match_winner",
                "double_chance",
                "over_under",
                "both_teams_to_score",
                "correct_score",
                "asian_handicap",
                name="odds_market",
                native_enum=False,
                create_constraint=True,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("selection", sa.Text(), nullable=False),
        sa.Column("line", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("price", sa.Numeric(precision=8, scale=3), nullable=False),
        sa.Column("is_closing", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "market <> 'both_teams_to_score' OR selection IN ('yes', 'no')",
            name=op.f("ck_odds_btts_selection"),
        ),
        sa.CheckConstraint(
            "market <> 'correct_score' OR selection ~ '^[0-9]+-[0-9]+$'",
            name=op.f("ck_odds_correct_score_selection"),
        ),
        sa.CheckConstraint(
            "market <> 'match_winner' OR selection IN ('home', 'draw', 'away')",
            name=op.f("ck_odds_match_winner_selection"),
        ),
        sa.CheckConstraint(
            "market <> 'over_under' OR (selection IN ('over', 'under') AND line IS NOT NULL)",
            name=op.f("ck_odds_over_under_selection"),
        ),
        sa.CheckConstraint("price > 1", name=op.f("ck_odds_price_above_one")),
        sa.ForeignKeyConstraint(
            ["bookmaker_id"],
            ["bookmakers.id"],
            name=op.f("fk_odds_bookmaker_id_bookmakers"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_odds_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"], ["matches.id"], name=op.f("fk_odds_match_id_matches"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_odds")),
        sa.UniqueConstraint(
            "data_source_id", "external_id", name=op.f("uq_odds_data_source_id_external_id")
        ),
        sa.UniqueConstraint(
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
    )
    op.create_index(op.f("ix_odds_bookmaker_id"), "odds", ["bookmaker_id"], unique=False)
    op.create_table(
        "player_match_statistics",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("minutes_played", sa.SmallInteger(), nullable=True),
        sa.Column("goals", sa.SmallInteger(), nullable=True),
        sa.Column("assists", sa.SmallInteger(), nullable=True),
        sa.Column("shots", sa.SmallInteger(), nullable=True),
        sa.Column("shots_on_target", sa.SmallInteger(), nullable=True),
        sa.Column("passes", sa.SmallInteger(), nullable=True),
        sa.Column("passes_completed", sa.SmallInteger(), nullable=True),
        sa.Column("key_passes", sa.SmallInteger(), nullable=True),
        sa.Column("tackles", sa.SmallInteger(), nullable=True),
        sa.Column("interceptions", sa.SmallInteger(), nullable=True),
        sa.Column("fouls_committed", sa.SmallInteger(), nullable=True),
        sa.Column("yellow_cards", sa.SmallInteger(), nullable=True),
        sa.Column("red_cards", sa.SmallInteger(), nullable=True),
        sa.Column("saves", sa.SmallInteger(), nullable=True),
        sa.Column("expected_goals", sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column("expected_assists", sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column("rating", sa.Numeric(precision=4, scale=2), nullable=True),
        sa.Column(
            "extra",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "expected_goals >= 0 AND expected_assists >= 0",
            name=op.f("ck_player_match_statistics_expected_values_non_negative"),
        ),
        sa.CheckConstraint(
            "minutes_played >= 0 AND goals >= 0 AND assists >= 0 AND shots >= 0 AND shots_on_target >= 0 AND passes >= 0 AND passes_completed >= 0 AND key_passes >= 0 AND tackles >= 0 AND interceptions >= 0 AND fouls_committed >= 0 AND yellow_cards >= 0 AND red_cards >= 0 AND saves >= 0",
            name=op.f("ck_player_match_statistics_counts_non_negative"),
        ),
        sa.CheckConstraint(
            "passes_completed <= passes",
            name=op.f("ck_player_match_statistics_passes_completed_within_passes"),
        ),
        sa.CheckConstraint(
            "rating BETWEEN 0 AND 10", name=op.f("ck_player_match_statistics_rating_range")
        ),
        sa.CheckConstraint(
            "shots_on_target <= shots",
            name=op.f("ck_player_match_statistics_shots_on_target_within_shots"),
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_player_match_statistics_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.id"],
            name=op.f("fk_player_match_statistics_match_id_matches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["players.id"],
            name=op.f("fk_player_match_statistics_player_id_players"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["teams.id"],
            name=op.f("fk_player_match_statistics_team_id_teams"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_player_match_statistics")),
        sa.UniqueConstraint(
            "data_source_id",
            "external_id",
            name=op.f("uq_player_match_statistics_data_source_id_external_id"),
        ),
        sa.UniqueConstraint(
            "match_id", "player_id", "data_source_id", name="uq_player_match_statistics_natural_key"
        ),
    )
    op.create_index(
        op.f("ix_player_match_statistics_player_id"),
        "player_match_statistics",
        ["player_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_player_match_statistics_team_id"),
        "player_match_statistics",
        ["team_id"],
        unique=False,
    )
    op.create_table(
        "team_match_statistics",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("data_source_id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("team_id", sa.BigInteger(), nullable=False),
        sa.Column("possession", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("shots", sa.SmallInteger(), nullable=True),
        sa.Column("shots_on_target", sa.SmallInteger(), nullable=True),
        sa.Column("shots_off_target", sa.SmallInteger(), nullable=True),
        sa.Column("blocked_shots", sa.SmallInteger(), nullable=True),
        sa.Column("shots_inside_box", sa.SmallInteger(), nullable=True),
        sa.Column("corners", sa.SmallInteger(), nullable=True),
        sa.Column("fouls", sa.SmallInteger(), nullable=True),
        sa.Column("offsides", sa.SmallInteger(), nullable=True),
        sa.Column("yellow_cards", sa.SmallInteger(), nullable=True),
        sa.Column("red_cards", sa.SmallInteger(), nullable=True),
        sa.Column("saves", sa.SmallInteger(), nullable=True),
        sa.Column("passes", sa.SmallInteger(), nullable=True),
        sa.Column("passes_completed", sa.SmallInteger(), nullable=True),
        sa.Column("expected_goals", sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column(
            "extra",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "expected_goals >= 0", name=op.f("ck_team_match_statistics_expected_goals_non_negative")
        ),
        sa.CheckConstraint(
            "passes_completed <= passes",
            name=op.f("ck_team_match_statistics_passes_completed_within_passes"),
        ),
        sa.CheckConstraint(
            "possession BETWEEN 0 AND 100",
            name=op.f("ck_team_match_statistics_possession_percentage"),
        ),
        sa.CheckConstraint(
            "shots >= 0 AND shots_on_target >= 0 AND shots_off_target >= 0 AND blocked_shots >= 0 AND shots_inside_box >= 0 AND corners >= 0 AND fouls >= 0 AND offsides >= 0 AND yellow_cards >= 0 AND red_cards >= 0 AND saves >= 0 AND passes >= 0 AND passes_completed >= 0",
            name=op.f("ck_team_match_statistics_counts_non_negative"),
        ),
        sa.CheckConstraint(
            "shots_on_target <= shots",
            name=op.f("ck_team_match_statistics_shots_on_target_within_shots"),
        ),
        sa.ForeignKeyConstraint(
            ["data_source_id"],
            ["data_sources.id"],
            name=op.f("fk_team_match_statistics_data_source_id_data_sources"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.id"],
            name=op.f("fk_team_match_statistics_match_id_matches"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["team_id"],
            ["teams.id"],
            name=op.f("fk_team_match_statistics_team_id_teams"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_team_match_statistics")),
        sa.UniqueConstraint(
            "data_source_id",
            "external_id",
            name=op.f("uq_team_match_statistics_data_source_id_external_id"),
        ),
        sa.UniqueConstraint(
            "match_id",
            "team_id",
            "data_source_id",
            name=op.f("uq_team_match_statistics_match_id_team_id_data_source_id"),
        ),
    )
    op.create_index(
        op.f("ix_team_match_statistics_team_id"), "team_match_statistics", ["team_id"], unique=False
    )
    op.create_table(
        "lineup_players",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("lineup_id", sa.BigInteger(), nullable=False),
        sa.Column("player_id", sa.BigInteger(), nullable=False),
        sa.Column("is_starter", sa.Boolean(), nullable=False),
        sa.Column("position", sa.Text(), nullable=True),
        sa.Column("shirt_number", sa.SmallInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "shirt_number IS NULL OR shirt_number BETWEEN 0 AND 99",
            name=op.f("ck_lineup_players_shirt_number_range"),
        ),
        sa.ForeignKeyConstraint(
            ["lineup_id"],
            ["lineups.id"],
            name=op.f("fk_lineup_players_lineup_id_lineups"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["player_id"],
            ["players.id"],
            name=op.f("fk_lineup_players_player_id_players"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lineup_players")),
        sa.UniqueConstraint(
            "lineup_id", "player_id", name=op.f("uq_lineup_players_lineup_id_player_id")
        ),
    )
    op.create_index(
        op.f("ix_lineup_players_player_id"), "lineup_players", ["player_id"], unique=False
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("model_version_id", sa.BigInteger(), nullable=False),
        sa.Column("feature_snapshot_id", sa.BigInteger(), nullable=True),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_home_goals", sa.SmallInteger(), nullable=False),
        sa.Column("predicted_away_goals", sa.SmallInteger(), nullable=False),
        sa.Column("expected_home_goals", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("expected_away_goals", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("home_win_probability", sa.Numeric(precision=7, scale=6), nullable=False),
        sa.Column("draw_probability", sa.Numeric(precision=7, scale=6), nullable=False),
        sa.Column("away_win_probability", sa.Numeric(precision=7, scale=6), nullable=False),
        sa.Column("btts_probability", sa.Numeric(precision=7, scale=6), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "abs(home_win_probability + draw_probability + away_win_probability - 1) <= 0.00001",
            name=op.f("ck_predictions_outcome_probabilities_sum_to_one"),
        ),
        sa.CheckConstraint(
            "expected_home_goals >= 0 AND expected_away_goals >= 0",
            name=op.f("ck_predictions_expected_goals_non_negative"),
        ),
        sa.CheckConstraint(
            "home_win_probability BETWEEN 0 AND 1 AND draw_probability BETWEEN 0 AND 1 AND away_win_probability BETWEEN 0 AND 1 AND btts_probability BETWEEN 0 AND 1",
            name=op.f("ck_predictions_probabilities_in_range"),
        ),
        sa.CheckConstraint(
            "predicted_home_goals >= 0 AND predicted_away_goals >= 0",
            name=op.f("ck_predictions_predicted_goals_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["feature_snapshot_id"],
            ["feature_snapshots.id"],
            name=op.f("fk_predictions_feature_snapshot_id_feature_snapshots"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["match_id"],
            ["matches.id"],
            name=op.f("fk_predictions_match_id_matches"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_predictions_model_version_id_model_versions"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
        sa.UniqueConstraint(
            "match_id",
            "model_version_id",
            "predicted_at",
            name=op.f("uq_predictions_match_id_model_version_id_predicted_at"),
        ),
    )
    op.create_index(
        op.f("ix_predictions_feature_snapshot_id"),
        "predictions",
        ["feature_snapshot_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_predictions_model_version_id"), "predictions", ["model_version_id"], unique=False
    )
    op.create_table(
        "prediction_over_under",
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("line", sa.Numeric(precision=4, scale=2), nullable=False),
        sa.Column("over_probability", sa.Numeric(precision=7, scale=6), nullable=False),
        sa.Column("under_probability", sa.Numeric(precision=7, scale=6), nullable=False),
        sa.CheckConstraint("line >= 0", name=op.f("ck_prediction_over_under_line_non_negative")),
        sa.CheckConstraint(
            "over_probability + under_probability <= 1 + 0.00001",
            name=op.f("ck_prediction_over_under_probabilities_at_most_one"),
        ),
        sa.CheckConstraint(
            "over_probability BETWEEN 0 AND 1 AND under_probability BETWEEN 0 AND 1",
            name=op.f("ck_prediction_over_under_probabilities_in_range"),
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name=op.f("fk_prediction_over_under_prediction_id_predictions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("prediction_id", "line", name=op.f("pk_prediction_over_under")),
    )
    op.create_table(
        "prediction_results",
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome_correct", sa.Boolean(), nullable=False),
        sa.Column("exact_score_correct", sa.Boolean(), nullable=False),
        sa.Column("btts_correct", sa.Boolean(), nullable=False),
        sa.Column("brier_score", sa.Numeric(precision=8, scale=6), nullable=False),
        sa.Column("log_loss", sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column("ranked_probability_score", sa.Numeric(precision=8, scale=6), nullable=False),
        sa.Column("actual_score_probability", sa.Numeric(precision=7, scale=6), nullable=True),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "actual_score_probability BETWEEN 0 AND 1",
            name=op.f("ck_prediction_results_actual_score_probability_in_range"),
        ),
        sa.CheckConstraint(
            "brier_score >= 0 AND log_loss >= 0 AND ranked_probability_score >= 0",
            name=op.f("ck_prediction_results_scores_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name=op.f("fk_prediction_results_prediction_id_predictions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("prediction_id", name=op.f("pk_prediction_results")),
    )
    op.create_table(
        "prediction_score_probabilities",
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("home_goals", sa.SmallInteger(), nullable=False),
        sa.Column("away_goals", sa.SmallInteger(), nullable=False),
        sa.Column("probability", sa.Numeric(precision=7, scale=6), nullable=False),
        sa.CheckConstraint(
            "home_goals >= 0 AND away_goals >= 0",
            name=op.f("ck_prediction_score_probabilities_goals_non_negative"),
        ),
        sa.CheckConstraint(
            "probability BETWEEN 0 AND 1",
            name=op.f("ck_prediction_score_probabilities_probability_in_range"),
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"],
            ["predictions.id"],
            name=op.f("fk_prediction_score_probabilities_prediction_id_predictions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "prediction_id",
            "home_goals",
            "away_goals",
            name=op.f("pk_prediction_score_probabilities"),
        ),
    )


def downgrade() -> None:
    op.drop_table("prediction_score_probabilities")
    op.drop_table("prediction_results")
    op.drop_table("prediction_over_under")
    op.drop_index(op.f("ix_predictions_model_version_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_feature_snapshot_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index(op.f("ix_lineup_players_player_id"), table_name="lineup_players")
    op.drop_table("lineup_players")
    op.drop_index(op.f("ix_team_match_statistics_team_id"), table_name="team_match_statistics")
    op.drop_table("team_match_statistics")
    op.drop_index(op.f("ix_player_match_statistics_team_id"), table_name="player_match_statistics")
    op.drop_index(
        op.f("ix_player_match_statistics_player_id"), table_name="player_match_statistics"
    )
    op.drop_table("player_match_statistics")
    op.drop_index(op.f("ix_odds_bookmaker_id"), table_name="odds")
    op.drop_table("odds")
    op.drop_table("match_external_ids")
    op.drop_index(op.f("ix_match_events_team_id"), table_name="match_events")
    op.drop_index(op.f("ix_match_events_related_player_id"), table_name="match_events")
    op.drop_index(op.f("ix_match_events_player_id"), table_name="match_events")
    op.drop_index("ix_match_events_match_id_minute", table_name="match_events")
    op.drop_table("match_events")
    op.drop_index(op.f("ix_lineups_team_id"), table_name="lineups")
    op.drop_table("lineups")
    op.drop_index("ix_injuries_team_id_start_date", table_name="injuries")
    op.drop_index(op.f("ix_injuries_match_id"), table_name="injuries")
    op.drop_table("injuries")
    op.drop_table("feature_snapshots")
    op.drop_index(op.f("ix_standings_team_id"), table_name="standings")
    op.drop_table("standings")
    op.drop_table("season_external_ids")
    op.drop_table("player_external_ids")
    op.drop_index("ix_matches_status_kickoff_at", table_name="matches")
    op.drop_index("ix_matches_season_id_competition_id", table_name="matches")
    op.drop_index("ix_matches_home_team_id_kickoff_at", table_name="matches")
    op.drop_index("ix_matches_competition_id_kickoff_at", table_name="matches")
    op.drop_index("ix_matches_away_team_id_kickoff_at", table_name="matches")
    op.drop_table("matches")
    op.drop_table("team_external_ids")
    op.drop_index(
        "uq_seasons_one_current_per_competition",
        table_name="seasons",
        postgresql_where=sa.text("is_current"),
    )
    op.drop_table("seasons")
    op.drop_index(op.f("ix_players_current_team_id"), table_name="players")
    op.drop_table("players")
    op.drop_index("ix_data_ingestion_runs_source_job_started", table_name="data_ingestion_runs")
    op.drop_index(
        "ix_data_ingestion_runs_running",
        table_name="data_ingestion_runs",
        postgresql_where=sa.text("status = 'running'"),
    )
    op.drop_table("data_ingestion_runs")
    op.drop_table("competition_external_ids")
    op.drop_table("teams")
    op.drop_index(
        "uq_model_versions_one_active_per_name",
        table_name="model_versions",
        postgresql_where=sa.text("is_active"),
    )
    op.drop_table("model_versions")
    op.drop_table("data_sources")
    op.drop_table("competitions")
    op.drop_table("bookmakers")
