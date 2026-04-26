"""Add `preparation_state` to sources for the prep wizard.

Revision ID: 0004_source_preparation_state
Revises: 0003_user_consents_and_codes
Create Date: 2026-04-26 00:00:00

Источник теперь имеет жизненный цикл: после загрузки он переходит в
``draft`` и анализ запрещён, пока пользователь не подтвердит листы,
column mapping и при необходимости не отредактирует данные через
мастер предобработки. Финализация выставляет ``ready``.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_source_preparation_state"
down_revision: str | None = "0003_user_consents_and_codes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("sources") as batch:
        batch.add_column(
            sa.Column(
                "preparation_state",
                sa.String(length=16),
                nullable=False,
                server_default="ready",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("sources") as batch:
        batch.drop_column("preparation_state")
