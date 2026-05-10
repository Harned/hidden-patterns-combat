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


def run_preflight(
    source: Source,
    storage_resolve,
    sheet_names: list[str] | None = None,
) -> str:
    from hpc_algo import preflight_mapping

    absolute: Path = storage_resolve(source.stored_path)
    cfg = preflight_mapping(absolute, sheet_names=sheet_names)
    return cfg.model_dump_json()


def list_sheets(source: Source, storage_resolve) -> list[str]:
    from hpc_algo import list_workbook_sheets

    absolute: Path = storage_resolve(source.stored_path)
    return list_workbook_sheets(absolute)


def read_grid_fragment(
    source: Source,
    storage_resolve,
    *,
    sheet_name: str,
    start_row: int,
    start_col: int,
    n_rows: int,
    n_cols: int,
) -> dict[str, Any]:
    from hpc_algo.workbook_editor import read_grid

    absolute: Path = storage_resolve(source.stored_path)
    fragment = read_grid(
        absolute,
        sheet_name,
        start_row=start_row,
        start_col=start_col,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    return {
        "sheet": fragment.sheet,
        "start_row": fragment.start_row,
        "start_col": fragment.start_col,
        "n_rows": fragment.n_rows,
        "n_cols": fragment.n_cols,
        "total_rows": fragment.total_rows,
        "total_cols": fragment.total_cols,
        "cells": fragment.cells,
    }


def apply_grid_edits(
    source: Source,
    storage_resolve,
    *,
    sheet_name: str,
    edits: list[dict[str, Any]],
) -> int:
    from hpc_algo.workbook_editor import apply_cell_edits

    absolute: Path = storage_resolve(source.stored_path)
    return apply_cell_edits(absolute, sheet_name, edits)


def remove_empty_rows_in_sheet(
    source: Source,
    storage_resolve,
    *,
    sheet_name: str,
    header_rows: list[int] | None,
) -> int:
    from hpc_algo.workbook_editor import remove_empty_rows

    absolute: Path = storage_resolve(source.stored_path)
    return remove_empty_rows(absolute, sheet_name, header_rows=header_rows)


def count_empty_rows_in_sheet(
    source: Source,
    storage_resolve,
    *,
    sheet_name: str,
    header_rows: list[int] | None,
) -> int:
    from hpc_algo.workbook_editor import count_empty_rows

    absolute: Path = storage_resolve(source.stored_path)
    return count_empty_rows(absolute, sheet_name, header_rows=header_rows)


def athlete_forward_fill_suggestions(
    source: Source,
    storage_resolve,
    *,
    sheet_name: str,
    header_rows: list[int] | None,
    athlete_columns: list[str],
    episode_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Получить предложения forward-fill по колонке ФИО для одного листа.

    Тонкая обёртка вокруг :func:`hpc_algo.suggestions.forward_fill_athlete_suggestions`.
    Возвращает уже сериализованные значения, готовые для JSON-ответа.
    """

    from hpc_algo.suggestions import forward_fill_athlete_suggestions

    absolute: Path = storage_resolve(source.stored_path)
    report = forward_fill_athlete_suggestions(
        absolute,
        sheet_name,
        header_rows=header_rows or [0],
        athlete_columns=athlete_columns,
        episode_columns=episode_columns,
    )
    return {
        "suggestions": [
            {
                "row": s.row,
                "col": s.col,
                "proposed": s.proposed,
                "source_row": s.source_row,
                "message_ru": s.message_ru,
            }
            for s in report.suggestions
        ],
        "athlete_column": report.athlete_column,
        "athlete_columns": list(report.athlete_columns_seen),
        "warning": report.warning,
    }


def header_merge_fill_for_sheet(
    source: Source,
    storage_resolve,
    *,
    sheet_name: str,
    header_rows: list[int] | None,
) -> dict[str, Any]:
    """Получить предложения «материализовать merged-ячейки шапки».

    Тонкая обёртка вокруг :func:`hpc_algo.suggestions.header_merge_fill_suggestions`.
    Если ``header_rows`` не передан, вернётся предупреждение и пустой список —
    подбор header_rows здесь не делаем (за это отвечает отдельный endpoint).
    """

    from hpc_algo.suggestions import header_merge_fill_suggestions

    absolute: Path = storage_resolve(source.stored_path)
    report = header_merge_fill_suggestions(
        absolute,
        sheet_name,
        header_rows=header_rows or [],
    )
    return {
        "sheet": sheet_name,
        "header_rows": list(report.header_rows),
        "suggestions": [
            {
                "row": s.row,
                "col": s.col,
                "proposed": s.proposed,
                "source_row": s.source_row,
                "source_col": s.source_col,
                "message_ru": s.message_ru,
            }
            for s in report.suggestions
        ],
        "warning": report.warning,
    }


def suggest_sheet_header_rows(
    source: Source,
    storage_resolve,
    *,
    sheet_name: str,
    current_header_rows: list[int] | None,
) -> dict[str, Any]:
    """Подсказать ``header_rows`` для листа + flatten-превью имён."""

    from hpc_algo.mapping import suggest_header_rows as _suggest

    absolute: Path = storage_resolve(source.stored_path)
    suggestion = _suggest(
        absolute,
        sheet_name,
        current_header_rows=current_header_rows,
    )

    def _safe(value: Any) -> Any:
        import math as _math

        if value is None:
            return None
        if isinstance(value, float) and (_math.isnan(value) or _math.isinf(value)):
            return None
        if isinstance(value, (str, int, bool, float)):
            return value
        try:
            return value.isoformat()  # pd.Timestamp / datetime
        except Exception:  # noqa: BLE001
            return str(value)

    return {
        "sheet": sheet_name,
        "suggested_header_rows": list(suggestion.suggested),
        "current_header_rows": (
            list(suggestion.current) if suggestion.current is not None else None
        ),
        "matches_current": suggestion.matches_current,
        "preview": suggestion.preview,
        "raw_preview": [
            [_safe(v) for v in row] for row in suggestion.raw_preview
        ],
    }


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


def sheet_preview(
    source: Source,
    storage_resolve,
    sheet_name: str,
    header_rows: list[int] | None,
    rows: int,
) -> dict[str, Any]:
    """Первые ``rows`` строк листа после применения header_rows.

    Используется в UI-редакторе mapping, чтобы пользователь видел
    примерные значения колонок и понимал, какую роль им назначить.
    """

    import math

    import pandas as pd
    from hpc_algo.mapping import (
        column_levels,
        guess_header_rows,
    )

    absolute: Path = storage_resolve(source.stored_path)
    if header_rows is None:
        raw = pd.read_excel(
            absolute, sheet_name=sheet_name, header=None, engine="openpyxl"
        )
        header_rows = guess_header_rows(raw)

    header_arg = (
        header_rows if len(header_rows) > 1 else (header_rows[0] if header_rows else 0)
    )
    df = pd.read_excel(
        absolute, sheet_name=sheet_name, header=header_arg, engine="openpyxl"
    )
    flat_names = [
        " | ".join(column_levels(col, idx))
        for idx, col in enumerate(df.columns)
    ]
    df.columns = flat_names

    def _safe(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (str, int, bool)):
            return value
        if isinstance(value, float):
            return value
        # datetimes / Timestamps / numpy scalars — сериализуем как строку.
        try:
            return value.isoformat()  # pd.Timestamp / datetime
        except Exception:  # noqa: BLE001
            return str(value)

    preview: list[dict[str, Any]] = []
    for _, row in df.head(max(0, int(rows))).iterrows():
        preview.append({str(k): _safe(v) for k, v in row.items()})

    return {
        "sheet": sheet_name,
        "header_rows": list(header_rows),
        "columns": flat_names,
        "preview": preview,
    }


# ---------------------------------------------------------------------------
# Markov-пайплайн (TASK_SPEC_011 / 012 / 013)
# ---------------------------------------------------------------------------


def run_markov_individual(
    source: Source,
    storage_resolve,
    settings,
) -> dict[str, Any]:
    """Запустить 5-state Observable Markov Chain по источнику.

    Возвращает ``BuildIndividualSummary`` + per-athlete детали (матрица
    переходов, стационарка, episode-метрики) как сериализуемый dict.
    Никакой исследовательской логики здесь нет — только вызов
    независимого processing module.
    """

    import json as _json

    from hpc_algo.build_individual import build_individual_models
    from hpc_algo.episode_metrics import classify_style, load_style_thresholds
    from hpc_algo.episode_split import (
        detect_base_columns,
        read_episodes_sheet,
        split_into_bouts_and_episodes,
    )
    from hpc_algo.markov_individual import build_episode_sequence, fit_individual_markov
    from hpc_algo.state_groups import load_state_groups

    excel_path = storage_resolve(source.stored_path)
    state_groups_path = settings.markov_state_groups_path
    style_thresholds_path = settings.markov_style_thresholds_path
    output_dir = settings.markov_reports_dir / str(source.id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary + HTML reports via standard orchestrator.
    summary = build_individual_models(
        excel_path=excel_path,
        state_groups_path=state_groups_path,
        output_dir=output_dir,
        style_thresholds_path=style_thresholds_path if style_thresholds_path.exists() else None,
    )

    # 2. Per-athlete structured data for the API JSON response.
    cfg, _ = load_state_groups(state_groups_path)
    style_thresholds = None
    if style_thresholds_path.exists():
        style_thresholds, _ = load_style_thresholds(style_thresholds_path)

    df = read_episodes_sheet(excel_path, sheet=cfg.sheet, header_rows=None)
    base = detect_base_columns(df.columns)
    feature_cols = [c for cols in cfg.states.values() for c in cols]
    raw_episodes, _ = split_into_bouts_and_episodes(df, base, feature_cols)
    records = build_episode_sequence(raw_episodes, cfg)

    from collections import defaultdict

    from hpc_algo.episode_metrics import compute_episode_metrics

    rendered_set = set(summary.rendered_athletes)
    per_ath_raw: dict = defaultdict(list)
    per_ath_rec: dict = defaultdict(list)
    for r in raw_episodes:
        if r.athlete in rendered_set:
            per_ath_raw[r.athlete].append(r)
    for r in records:
        if r.athlete in rendered_set:
            per_ath_rec[r.athlete].append(r)

    athletes_data = []
    for athlete in summary.rendered_athletes:
        markov_result = fit_individual_markov(athlete, records, mode=cfg.mode)
        metrics = compute_episode_metrics(
            per_ath_raw.get(athlete, []),
            per_ath_rec.get(athlete, []),
        )
        style_label = None
        if style_thresholds is not None:
            label, _ = classify_style(metrics, style_thresholds)
            style_label = label.value if hasattr(label, "value") else str(label)

        athletes_data.append({
            "athlete": athlete,
            "episode_count": markov_result.episode_count,
            "bout_count": markov_result.bout_count,
            "transition_matrix": markov_result.transition_matrix,
            "stationary": {s: v for s, v in zip(markov_result.states, markov_result.stationary)},
            "visit_counts": markov_result.visit_counts,
            "state_labels": [s.value if hasattr(s, "value") else str(s) for s in markov_result.states],
            "style": style_label,
            "episode_metrics": _json.loads(metrics.model_dump_json()),
            "warnings_count": summary.per_athlete_warning_counts.get(athlete, 0),
        })

    return {
        "summary": _json.loads(summary.model_dump_json()),
        "athletes": athletes_data,
        "state_labels": list(cfg.states.keys()) + ["pause"],
        "reports_dir": str(output_dir),
    }
