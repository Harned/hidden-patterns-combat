"""Тонкая обёртка над :func:`hpc_algo.analyze_source`.

Любая исследовательская логика **запрещена** в этом модуле. Он только:
* берёт путь к файлу пользователя,
* при наличии — прикладывает сохранённый ``ColumnMappingConfig``,
* вызывает независимый processing module,
* сохраняет результат.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import AnalysisRun, Source


def _load_mapping_from_source(source: Source) -> Any | None:
    """Десериализовать сохранённый ``ColumnMappingConfig`` (если есть)."""

    if not source.mapping_config:
        return None
    # Импорт внутри функции — backend не должен тянуть hpc_algo в import-time.
    from hpc_algo import ColumnMappingConfig

    try:
        return ColumnMappingConfig.model_validate_json(source.mapping_config)
    except Exception:  # noqa: BLE001 — битый конфиг не должен ронять анализ
        return None


def run_and_persist(
    db: Session,
    source: Source,
    storage_resolve,
    hmm_mode: str = "auto",
) -> AnalysisRun:
    """Выполнить анализ и сохранить JSON-результат."""

    from hpc_algo import AnalyzeConfig, analyze_source

    absolute: Path = storage_resolve(source.stored_path)
    mapping = _load_mapping_from_source(source)
    config = AnalyzeConfig(column_mapping=mapping, hmm_mode=hmm_mode)

    result = analyze_source(absolute, config)

    run = AnalysisRun(
        source_id=source.id,
        status=result.status.value,
        algo_version=result.algo_version,
        result_json=result.model_dump_json(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def run_preflight(source: Source, storage_resolve) -> str:
    """Вернуть предложенный ``ColumnMappingConfig`` как JSON-строку."""

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
    """Описать колонки выбранного листа для редактора mapping."""

    from hpc_algo.mapping import describe_sheet_columns as _describe

    absolute: Path = storage_resolve(source.stored_path)
    used_rows, columns = _describe(absolute, sheet_name, header_rows=header_rows)
    return {
        "sheet": sheet_name,
        "header_rows": used_rows,
        "columns": columns,
    }
