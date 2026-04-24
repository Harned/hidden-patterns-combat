"""Initial schema: users, sources, analysis_runs (TASK_SPEC_006).

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-24 00:00:00

Описывает текущее состояние схемы после TASK_SPEC_002/003/004/005 +
поля фонового run'а (state/started_at/finished_at/error) и hmm_mode.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(length=320), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "owner_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("stored_path", sa.String(length=1024), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("mapping_config", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("owner_id", "id", name="uq_sources_owner_id"),
    )
    op.create_index("ix_sources_owner_id", "sources", ["owner_id"])

    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "source_id",
            sa.Integer(),
            sa.ForeignKey("sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "state",
            sa.String(length=16),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("status", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("algo_version", sa.String(length=32), nullable=False, server_default=""),
        sa.Column("result_json", sa.Text(), nullable=False, server_default=""),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("hmm_mode", sa.String(length=16), nullable=False, server_default="auto"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_analysis_runs_source_id", "analysis_runs", ["source_id"])


def downgrade() -> None:
    op.drop_index("ix_analysis_runs_source_id", table_name="analysis_runs")
    op.drop_table("analysis_runs")
    op.drop_index("ix_sources_owner_id", table_name="sources")
    op.drop_table("sources")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
