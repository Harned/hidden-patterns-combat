"""Локальный storage adapter. Абстрагирован, чтобы заменить на S3 позже."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\-а-яА-ЯёЁ ]+")


def _safe_filename(name: str) -> str:
    """Обезопасить имя файла, сохраняя кириллицу."""

    cleaned = _SAFE_FILENAME_RE.sub("_", name).strip() or "upload"
    return cleaned[:200]


@dataclass(frozen=True)
class StoredFile:
    relative_path: str
    absolute_path: Path
    size_bytes: int
    sha256: str


class LocalStorage:
    """Простейший FS-adapter: ``<root>/<user_id>/<uuid>__<safe_name>``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Settings) -> LocalStorage:
        return cls(root=settings.storage_root)

    def save(self, user_id: int, original_name: str, data: bytes) -> StoredFile:
        user_dir = self.root / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        unique = uuid.uuid4().hex
        safe = _safe_filename(original_name)
        filename = f"{unique}__{safe}"
        target = user_dir / filename
        target.write_bytes(data)

        sha256 = hashlib.sha256(data).hexdigest()
        return StoredFile(
            relative_path=str(Path(str(user_id)) / filename),
            absolute_path=target,
            size_bytes=len(data),
            sha256=sha256,
        )

    def resolve(self, relative_path: str) -> Path:
        return (self.root / relative_path).resolve()

    def delete(self, relative_path: str) -> None:
        target = self.resolve(relative_path)
        if target.exists() and target.is_file() and self.root in target.parents:
            target.unlink(missing_ok=True)
