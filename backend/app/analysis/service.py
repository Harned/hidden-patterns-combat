"""Тонкая обёртка над :func:`hpc_algo.analyze_source`.

Любая исследовательская логика **запрещена** в этом модуле. Он только:
* берёт путь к файлу пользователя,
* при наличии — прикладывает сохранённый ``ColumnMappingConfig``,
* управляет жизненным циклом ``AnalysisRun`` (pending → running →
  done | failed),
* вызывает независимый processing module,
* сохраняет результат.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AnalysisRun, Source
from app.db.session import get_sessionmaker


def _load_mapping_from_source(source: Source) -> Any | None:
    if not source.mapping_config:
        return None
    from hpc_algo import ColumnMappingConfig

    try:
        return ColumnMappingConfig.model_validate_json(source.mapping_config)
    except Exception:  # noqa: BLE001 — битый конфиг не должен ронять анализ
        return None


def create_pending_run(
    db: Session,
    source: Source,
    hmm_mode: str,
) -> AnalysisRun:
    run = AnalysisRun(
        source_id=source.id,
        state="pending",
        status="",
        algo_version="",
        result_json="",
        hmm_mode=hmm_mode,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _run_in_its_own_session(run_id: int, source_id: int, storage_resolve) -> None:
    """Исполнитель фоновой задачи.

    BackgroundTasks запускается после возврата HTTP-ответа, поэтому у
    нас нет уже открытой сессии БД. Создаём свою.
    """

    from hpc_algo import AnalyzeConfig, analyze_source

    SessionLocal = get_sessionmaker()
    with SessionLocal() as db:
        run = db.get(AnalysisRun, run_id)
        source = db.get(Source, source_id)
        if run is None or source is None:
            return

        run.state = "running"
        run.started_at = datetime.now(UTC)
        db.commit()

        try:
            mapping = _load_mapping_from_source(source)
            config = AnalyzeConfig(column_mapping=mapping, hmm_mode=run.hmm_mode)
            absolute: Path = storage_resolve(source.stored_path)
            result = analyze_source(absolute, config)

            run.status = result.status.value
            run.algo_version = result.algo_version
            run.result_json = result.model_dump_json()
            run.state = "done"
            run.error = None
            run.finished_at = datetime.now(UTC)
            db.commit()
        except Exception as exc:  # noqa: BLE001
            run.state = "failed"
            run.error = str(exc)
            run.finished_at = datetime.now(UTC)
            db.commit()


def schedule_run(
    background_tasks,
    run: AnalysisRun,
    storage_resolve,
) -> None:
    background_tasks.add_task(
        _run_in_its_own_session, run.id, run.source_id, storage_resolve
    )


def run_and_persist_sync(
    db: Session,
    source: Source,
    storage_resolve,
    hmm_mode: str = "auto",
) -> AnalysisRun:
    """Синхронный вариант запуска анализа — используется в CLI / тестах."""

    from hpc_algo import AnalyzeConfig, analyze_source

    absolute: Path = storage_resolve(source.stored_path)
    mapping = _load_mapping_from_source(source)
    config = AnalyzeConfig(column_mapping=mapping, hmm_mode=hmm_mode)
    result = analyze_source(absolute, config)
    now = datetime.now(UTC)
    run = AnalysisRun(
        source_id=source.id,
        state="done",
        status=result.status.value,
        algo_version=result.algo_version,
        result_json=result.model_dump_json(),
        hmm_mode=hmm_mode,
        started_at=now,
        finished_at=now,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def run_preflight(source: Source, storage_resolve) -> str:
    from hpc_algo import preflight_mapping

    absolute: Path = storage_resolve(source.stored_path)
    cfg = preflight_mapping(absolute)
    return cfg.model_dump_json()


def describe_sheet_columns(
    source: Source,
    storage_resolve,
    sheet_name: str,
    header_rows: list[int] | None,
) -> dict[str, Any]:
    from hpc_algo.mapping import describe_sheet_columns as _describe

    absolute: Path = storage_resolve(source.stored_path)
    used_rows, columns = _describe(absolute, sheet_name, header_rows=header_rows)
    return {
        "sheet": sheet_name,
        "header_rows": used_rows,
        "columns": columns,
    }
