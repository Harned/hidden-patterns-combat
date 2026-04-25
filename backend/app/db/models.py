"""ORM-модели для auth, sources и сохранённых AnalysisResult."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    email_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # --- TASK_SPEC_010: согласия и коды ---

    # Время принятия двух согласий при регистрации (LEGAL-REG-1).
    terms_accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pdn_accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Email verification code (короткий код, не JWT). Активен один за раз.
    email_verification_code_hash: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    email_verification_code_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    email_verification_code_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Password reset code (отдельный сценарий, не путаем с verification).
    password_reset_code_hash: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    password_reset_code_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Онбординг-дисклеймер при первом входе после подтверждения email
    # (LEGAL-ONBOARD-1). NULL = ещё не показан / не подтверждён.
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    sources: Mapped[list[Source]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )


class Source(Base):
    """Загруженный пользователем Excel-источник."""

    __tablename__ = "sources"
    __table_args__ = (
        UniqueConstraint("owner_id", "id", name="uq_sources_owner_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    # JSON-сериализованный ColumnMappingConfig (TASK_SPEC_003). None, если
    # пользователь ещё не подтвердил сопоставление колонок.
    mapping_config: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    owner: Mapped[User] = relationship(back_populates="sources")
    analysis_runs: Mapped[list[AnalysisRun]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        order_by="AnalysisRun.id.desc()",
    )


class AnalysisRun(Base):
    """Результат вызова :func:`hpc_algo.analyze_source`.

    `state` — жизненный цикл фоновой задачи
    (``pending`` → ``running`` → ``done`` / ``failed``). `status` — это
    предметный статус результата (``baseline_only`` / ``hmm_ready`` / …)
    из AnalysisResult. Разделение важно: `state=done` не означает
    успешную диагностику.
    """

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    state: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
        doc="pending | running | done | failed",
    )
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    algo_version: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    result_json: Mapped[str] = mapped_column(Text, nullable=False, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    hmm_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="auto")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    source: Mapped[Source] = relationship(back_populates="analysis_runs")
