"""TASK_SPEC_010: consents, email/reset codes, onboarding.

Revision ID: 0003_user_consents_and_codes
Revises: 0002_email_verified_at
Create Date: 2026-04-25 00:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0003_user_consents_and_codes"
down_revision: str | None = "0002_email_verified_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_NEW_COLUMNS = (
    ("terms_accepted_at", sa.DateTime(timezone=True), True),
    ("pdn_accepted_at", sa.DateTime(timezone=True), True),
    ("email_verification_code_hash", sa.String(length=255), True),
    ("email_verification_code_expires_at", sa.DateTime(timezone=True), True),
    ("email_verification_code_sent_at", sa.DateTime(timezone=True), True),
    ("password_reset_code_hash", sa.String(length=255), True),
    ("password_reset_code_expires_at", sa.DateTime(timezone=True), True),
    ("onboarding_completed_at", sa.DateTime(timezone=True), True),
)


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        for name, type_, nullable in _NEW_COLUMNS:
            batch.add_column(sa.Column(name, type_, nullable=nullable))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        for name, _type, _nullable in reversed(_NEW_COLUMNS):
            batch.drop_column(name)
