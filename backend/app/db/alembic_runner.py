"""Запуск Alembic-миграций из кода (без shell).

`create_all` не добавляет колонки к уже существующим таблицам, поэтому
локальный dev с SQLite обязан применять миграции — либо вручную
(`make db-upgrade`), либо при старте (см. `app.main` при ``use_alembic``).

Старые dev-базы создавались через ``init_schema``/``create_all`` без
``alembic_version``. Тогда ``upgrade`` с нуля пытается снова создать
таблицы и падает. В этом случае делаем ``stamp`` на ревизию перед
последними дельтовыми миграциями и затем ``upgrade head``.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from app.config import Settings

# Ревизия сразу перед 0004 (добавление ``preparation_state``). Схема
# create_all() в старых dev должна соответствовать ей, если ORM
# обновлён вместе с миграциями 0001–0003.
_STAMP_FOR_LEGACY_NO_ALEMBIC = "0003_user_consents_and_codes"


def _make_alembic_config(settings: Settings) -> Config:
    backend_dir = Path(__file__).resolve().parent.parent.parent
    ini_path = backend_dir / "alembic.ini"
    if not ini_path.is_file():
        msg = f"Alembic config not found: {ini_path}"
        raise FileNotFoundError(msg)
    cfg = Config(str(ini_path))
    # В alembic.ini ``script_location = alembic`` относительный; при запуске
    # uvicorn из корня репозитория CWD != backend/, и Alembic ищет ./alembic
    # в CWD. Задаём каталог миграций абсолютно.
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    # Чтобы ``prepend_sys_path = .`` из ini не тянул не тот каталог.
    cfg.set_main_option("prepend_sys_path", str(backend_dir))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    return cfg


def run_alembic_upgrade_to_head(settings: Settings) -> None:
    """Применить миграции до ``head`` для ``settings.database_url``."""

    cfg = _make_alembic_config(settings)
    engine = create_engine(
        settings.database_url,
        future=True,
    )
    with engine.connect() as conn:
        insp = inspect(conn)
        mctx = MigrationContext.configure(conn)
        current = mctx.get_current_revision()
        # Пустой ``alembic_version`` (таблица есть, строк нет) или нет
        # миграций — схема от старого ``create_all``; иначе ``upgrade`` с
        # нуля пытается снова создать ``users``.
        if current is None and insp.has_table("users"):
            command.stamp(cfg, _STAMP_FOR_LEGACY_NO_ALEMBIC)
    command.upgrade(cfg, "head")
