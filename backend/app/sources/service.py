"""Бизнес-операции источников (upload, list, get, delete)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Source, User
from app.sources.schemas import SourceSummary
from app.sources.storage import LocalStorage


class SourceError(Exception):
    """Ошибка операций над источником (не найден / не принадлежит / битый upload)."""


_ALLOWED_EXTS = {".xlsx", ".xls"}
_ALLOWED_MIME_PREFIXES = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/octet-stream",  # браузер иногда не знает точный mime
)
# Запрет .xlsm — макросы потенциально опасны и не нужны для MVP.
_FORBIDDEN_EXTS = {".xlsm", ".xlsb", ".csv", ".ods"}

# Магические байты ZIP (xlsx) и OLE (xls).
_XLSX_MAGIC = b"PK\x03\x04"
_XLS_MAGIC = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"


def validate_upload(
    filename: str,
    content_type: str,
    data: bytes,
    max_size: int,
) -> None:
    if not filename:
        raise SourceError("Имя файла не указано.")
    if len(data) == 0:
        raise SourceError("Файл пустой.")
    if len(data) > max_size:
        raise SourceError(
            f"Файл слишком большой: {len(data)} байт (лимит {max_size})."
        )

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in _FORBIDDEN_EXTS:
        raise SourceError(
            f"Расширение {ext} запрещено (макросы / неподдерживаемый формат)."
        )
    if ext not in _ALLOWED_EXTS:
        raise SourceError(
            f"Разрешены только .xlsx и .xls. Получено: {ext or '(без расширения)'}."
        )

    ctype = (content_type or "").lower()
    if not any(ctype.startswith(prefix) for prefix in _ALLOWED_MIME_PREFIXES):
        # Не фатально, но зафиксируем.
        # Предпочитаем магические байты.
        pass

    if ext == ".xlsx" and not data.startswith(_XLSX_MAGIC):
        raise SourceError(
            "Файл не похож на валидный .xlsx (отсутствует ZIP-сигнатура)."
        )
    if ext == ".xls" and not data.startswith(_XLS_MAGIC):
        raise SourceError(
            "Файл не похож на валидный .xls (отсутствует OLE-сигнатура)."
        )


def create_source(
    db: Session,
    storage: LocalStorage,
    user: User,
    original_filename: str,
    content_type: str,
    data: bytes,
    max_size: int,
) -> Source:
    validate_upload(original_filename, content_type, data, max_size)

    stored = storage.save(user.id, original_filename, data)
    source = Source(
        owner_id=user.id,
        original_filename=original_filename,
        stored_path=stored.relative_path,
        size_bytes=stored.size_bytes,
        sha256=stored.sha256,
        content_type=content_type or "",
        preparation_state="draft",
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


def list_user_sources(db: Session, user: User) -> list[SourceSummary]:
    stmt = select(Source).where(Source.owner_id == user.id).order_by(Source.id.desc())
    sources = db.execute(stmt).scalars().all()

    summaries: list[SourceSummary] = []
    for s in sources:
        last_run = s.analysis_runs[0] if s.analysis_runs else None
        summaries.append(
            SourceSummary(
                id=s.id,
                original_filename=s.original_filename,
                size_bytes=s.size_bytes,
                sha256=s.sha256,
                created_at=s.created_at,
                has_analysis=last_run is not None,
                last_analysis_status=last_run.status if last_run else None,
                has_mapping=bool(s.mapping_config),
                preparation_state=s.preparation_state,
            )
        )
    return summaries


def get_owned_source(db: Session, user: User, source_id: int) -> Source:
    source = db.get(Source, source_id)
    if source is None or source.owner_id != user.id:
        # Намеренно 404 — не даём узнать факт существования чужих источников.
        raise SourceError("Источник не найден.")
    return source


def delete_owned_source(
    db: Session,
    storage: LocalStorage,
    user: User,
    source_id: int,
) -> None:
    source = get_owned_source(db, user, source_id)
    storage.delete(source.stored_path)
    db.delete(source)
    db.commit()


# ---------------------------------------------------------------------------
# Column mapping (TASK_SPEC_003)
# ---------------------------------------------------------------------------


def save_mapping(db: Session, source: Source, mapping_json: str) -> None:
    """Сохранить сериализованный ColumnMappingConfig.

    Валидация (парсинг в pydantic-модель) выполняется в router, тут мы
    доверяем уже проверенной строке.
    """

    source.mapping_config = mapping_json
    db.commit()


def clear_mapping(db: Session, source: Source) -> None:
    source.mapping_config = None
    db.commit()


# ---------------------------------------------------------------------------
# Жизненный цикл источника: draft → ready
# ---------------------------------------------------------------------------


class FinalizeError(SourceError):
    """Источник нельзя финализировать в текущем состоянии."""


def finalize_source(db: Session, source: Source) -> Source:
    """Перевести источник из ``draft`` в ``ready``.

    Запретим финализацию, если у пользователя нет ни одного листа,
    помеченного для анализа: алгоритм использует ``config.sheets`` как
    whitelist и без него возвращает ``audit_only``. Это явная ошибка,
    о которой имеет смысл сказать в UI.
    """

    if not source.mapping_config:
        raise FinalizeError(
            "Перед подтверждением выполните preflight и сохраните column mapping."
        )
    from hpc_algo import ColumnMappingConfig

    try:
        cfg = ColumnMappingConfig.model_validate_json(source.mapping_config)
    except Exception as exc:  # noqa: BLE001
        raise FinalizeError(f"Сохранённый mapping повреждён: {exc}") from exc
    if not cfg.sheets:
        raise FinalizeError(
            "Выберите хотя бы один лист, который должен участвовать в анализе."
        )

    source.preparation_state = "ready"
    db.commit()
    db.refresh(source)
    return source


def refresh_storage_metadata(
    db: Session, storage: LocalStorage, source: Source
) -> Source:
    """Пересчитать ``size_bytes`` и ``sha256`` после правки файла на диске."""

    absolute: Path = storage.resolve(source.stored_path)
    data = absolute.read_bytes()
    source.size_bytes = len(data)
    source.sha256 = hashlib.sha256(data).hexdigest()
    db.commit()
    db.refresh(source)
    return source
