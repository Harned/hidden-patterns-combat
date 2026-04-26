"""HTTP-слой анализа."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.analysis import service as analysis_service
from app.auth.deps import current_verified_user as current_user
from app.db.models import AnalysisRun, User
from app.db.session import get_db
from app.sources import service as sources_service
from app.sources.router import get_storage
from app.sources.service import SourceError
from app.sources.storage import LocalStorage

router = APIRouter(prefix="/sources/{source_id}", tags=["analysis"])


class AnalysisRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_id: int
    state: str
    status: str
    algo_version: str
    hmm_mode: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class AnalysisRunFull(AnalysisRunSummary):
    result: dict[str, Any]


class MappingResponse(BaseModel):
    mapping: dict[str, Any] | None


_VALID_HMM_MODES = {"auto", "detailed", "basic", "off"}


def _get_owned_source_or_404(db: Session, user: User, source_id: int):
    try:
        return sources_service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.post(
    "/analyze",
    response_model=AnalysisRunSummary,
)
def analyze(
    source_id: int,
    background_tasks: BackgroundTasks,
    mode: str = Query(default="auto"),
    wait: bool = Query(default=False, description="Синхронный запуск (для тестов)."),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> AnalysisRunSummary:
    if mode not in _VALID_HMM_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Неизвестный mode='{mode}'. Допустимые: "
                + ", ".join(sorted(_VALID_HMM_MODES))
            ),
        )
    source = _get_owned_source_or_404(db, user, source_id)

    if source.preparation_state != "ready":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Источник ещё не подтверждён. Завершите мастер предобработки "
                "(выбор листов, column mapping) перед запуском анализа."
            ),
        )

    if wait:
        run = analysis_service.run_and_persist_sync(
            db, source, storage.resolve, hmm_mode=mode
        )
        return AnalysisRunSummary.model_validate(run)

    run = analysis_service.create_pending_run(db, source, hmm_mode=mode)
    analysis_service.schedule_run(background_tasks, run, storage.resolve)
    return AnalysisRunSummary.model_validate(run)


@router.get("/runs/{run_id}", response_model=AnalysisRunSummary)
def get_run(
    source_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> AnalysisRunSummary:
    source = _get_owned_source_or_404(db, user, source_id)
    run = db.get(AnalysisRun, run_id)
    if run is None or run.source_id != source.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Запуск анализа не найден.",
        )
    return AnalysisRunSummary.model_validate(run)


@router.get("/runs", response_model=list[AnalysisRunSummary])
def list_runs(
    source_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> list[AnalysisRunSummary]:
    source = _get_owned_source_or_404(db, user, source_id)
    runs = source.analysis_runs[:limit]
    return [AnalysisRunSummary.model_validate(r) for r in runs]


@router.get("/runs/{run_id}/result", response_model=AnalysisRunFull)
def get_run_result(
    source_id: int,
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> AnalysisRunFull:
    source = _get_owned_source_or_404(db, user, source_id)
    run = db.get(AnalysisRun, run_id)
    if run is None or run.source_id != source.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Запуск анализа не найден.",
        )
    if run.state != "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Запуск в состоянии '{run.state}', результат ещё не готов.",
        )
    return AnalysisRunFull(
        id=run.id,
        source_id=run.source_id,
        state=run.state,
        status=run.status,
        algo_version=run.algo_version,
        hmm_mode=run.hmm_mode,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error=run.error,
        result=json.loads(run.result_json) if run.result_json else {},
    )


@router.get("/result", response_model=AnalysisRunFull)
def latest_result(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> AnalysisRunFull:
    source = _get_owned_source_or_404(db, user, source_id)

    completed = [r for r in source.analysis_runs if r.state == "done"]
    if not completed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Для этого источника ещё нет завершённого анализа.",
        )
    last = completed[0]
    return AnalysisRunFull(
        id=last.id,
        source_id=last.source_id,
        state=last.state,
        status=last.status,
        algo_version=last.algo_version,
        hmm_mode=last.hmm_mode,
        created_at=last.created_at,
        started_at=last.started_at,
        finished_at=last.finished_at,
        error=last.error,
        result=json.loads(last.result_json) if last.result_json else {},
    )


# ---------------------------------------------------------------------------
# Preflight и column mapping (TASK_SPEC_003)
# ---------------------------------------------------------------------------


class PreflightRequest(BaseModel):
    """Опциональный фильтр листов для preflight'а."""

    model_config = ConfigDict(extra="forbid")

    sheet_names: list[str] | None = None


@router.post("/preflight", response_model=MappingResponse)
def preflight(
    source_id: int,
    payload: PreflightRequest | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> MappingResponse:
    source = _get_owned_source_or_404(db, user, source_id)
    sheet_names = payload.sheet_names if payload else None
    try:
        mapping_json = analysis_service.run_preflight(
            source, storage.resolve, sheet_names=sheet_names
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Не удалось выполнить preflight: {exc}",
        ) from exc

    return MappingResponse(mapping=json.loads(mapping_json))


@router.get("/sheets")
def list_sheets(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    """Вернуть список листов Excel-файла (без чтения содержимого)."""

    source = _get_owned_source_or_404(db, user, source_id)
    try:
        names = analysis_service.list_sheets(source, storage.resolve)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Не удалось прочитать список листов: {exc}",
        ) from exc

    return {"sheet_names": names}


@router.get("/mapping", response_model=MappingResponse)
def get_mapping(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> MappingResponse:
    source = _get_owned_source_or_404(db, user, source_id)
    if not source.mapping_config:
        return MappingResponse(mapping=None)
    return MappingResponse(mapping=json.loads(source.mapping_config))


@router.put("/mapping", response_model=MappingResponse)
def put_mapping(
    source_id: int,
    payload: dict[str, Any],
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> MappingResponse:
    source = _get_owned_source_or_404(db, user, source_id)

    from hpc_algo import ColumnMappingConfig

    try:
        cfg = ColumnMappingConfig.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Некорректный column_mapping: {exc}",
        ) from exc

    sources_service.save_mapping(db, source, cfg.model_dump_json())
    return MappingResponse(mapping=json.loads(source.mapping_config or "null"))


@router.delete("/mapping", status_code=status.HTTP_204_NO_CONTENT)
def delete_mapping(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> None:
    source = _get_owned_source_or_404(db, user, source_id)
    sources_service.clear_mapping(db, source)


def _parse_header_rows(header_rows_param: str | None) -> list[int] | None:
    if not header_rows_param:
        return None
    try:
        return [
            int(piece)
            for piece in header_rows_param.split(",")
            if piece.strip() != ""
        ]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="header_rows должен быть списком целых чисел через запятую.",
        ) from exc


@router.get("/sheets/{sheet_name}/columns")
def list_sheet_columns(
    source_id: int,
    sheet_name: str,
    header_rows: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    source = _get_owned_source_or_404(db, user, source_id)
    parsed_rows = _parse_header_rows(header_rows)
    try:
        return analysis_service.describe_sheet_columns(
            source, storage.resolve, sheet_name, parsed_rows
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Не удалось описать лист '{sheet_name}': {exc}",
        ) from exc


@router.get("/sheets/{sheet_name}/suggestions/header-rows")
def get_header_rows_suggestion(
    source_id: int,
    sheet_name: str,
    header_rows: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    """Подсказка ``header_rows`` для листа + превью flatten-имён.

    MVP «не пишет в файл»: возвращает только рекомендацию, а применять её
    через PUT mapping будет пользователь. Источник правды по текущим
    ``header_rows`` — переданный query-параметр; если он не задан, берём
    значение из сохранённого ``mapping_config`` (если есть).
    """

    source = _get_owned_source_or_404(db, user, source_id)
    parsed_rows = _parse_header_rows(header_rows)

    if parsed_rows is None and source.mapping_config:
        try:
            cfg = json.loads(source.mapping_config)
        except json.JSONDecodeError:
            cfg = None
        if isinstance(cfg, dict):
            sheet_cfg = (cfg.get("sheets") or {}).get(sheet_name) or {}
            stored = sheet_cfg.get("header_rows")
            if isinstance(stored, list):
                parsed_rows = [int(x) for x in stored]

    try:
        return analysis_service.suggest_sheet_header_rows(
            source,
            storage.resolve,
            sheet_name=sheet_name,
            current_header_rows=parsed_rows,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Не удалось подсказать header_rows для '{sheet_name}': {exc}"
            ),
        ) from exc


@router.get("/sheets/{sheet_name}/preview")
def get_sheet_preview(
    source_id: int,
    sheet_name: str,
    header_rows: str | None = None,
    rows: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    """Первые `rows` строк листа после применения header_rows."""

    source = _get_owned_source_or_404(db, user, source_id)
    parsed_rows = _parse_header_rows(header_rows)
    try:
        return analysis_service.sheet_preview(
            source, storage.resolve, sheet_name, parsed_rows, rows
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Не удалось получить preview листа '{sheet_name}': {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Grid I/O для мастера предобработки (виртуализированная таблица в UI)
# ---------------------------------------------------------------------------


def _ensure_draft(source) -> None:
    if source.preparation_state != "draft":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Редактирование данных доступно только в режиме черновика. "
                "Источник уже подтверждён."
            ),
        )


@router.get("/sheets/{sheet_name}/grid")
def get_sheet_grid(
    source_id: int,
    sheet_name: str,
    start_row: int = Query(default=1, ge=1),
    start_col: int = Query(default=1, ge=1),
    n_rows: int = Query(default=100, ge=0, le=500),
    n_cols: int = Query(default=50, ge=0, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    source = _get_owned_source_or_404(db, user, source_id)
    try:
        return analysis_service.read_grid_fragment(
            source,
            storage.resolve,
            sheet_name=sheet_name,
            start_row=start_row,
            start_col=start_col,
            n_rows=n_rows,
            n_cols=n_cols,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Не удалось прочитать сетку листа '{sheet_name}': {exc}",
        ) from exc


class CellEditPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: int
    col: int
    value: Any | None = None


class GridEditsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edits: list[CellEditPayload]


@router.put("/sheets/{sheet_name}/grid")
def put_sheet_grid_edits(
    source_id: int,
    sheet_name: str,
    payload: GridEditsRequest,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    source = _get_owned_source_or_404(db, user, source_id)
    _ensure_draft(source)
    try:
        applied = analysis_service.apply_grid_edits(
            source,
            storage.resolve,
            sheet_name=sheet_name,
            edits=[e.model_dump() for e in payload.edits],
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Не удалось применить правки к '{sheet_name}': {exc}",
        ) from exc

    sources_service.refresh_storage_metadata(db, storage, source)
    return {"applied": applied}


class RemoveEmptyRowsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    header_rows: list[int] | None = None


@router.get("/sheets/{sheet_name}/empty-rows-count")
def count_empty_rows(
    source_id: int,
    sheet_name: str,
    header_rows: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    """Сколько полностью пустых data-строк сейчас в листе.

    Используется на шаге редактирования, чтобы заранее предложить
    пользователю удалить мусор без ручного просмотра всей таблицы.
    """

    source = _get_owned_source_or_404(db, user, source_id)
    parsed_rows = _parse_header_rows(header_rows)
    try:
        count = analysis_service.count_empty_rows_in_sheet(
            source,
            storage.resolve,
            sheet_name=sheet_name,
            header_rows=parsed_rows,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Не удалось посчитать пустые строки на листе '{sheet_name}': {exc}"
            ),
        ) from exc

    return {"count": count}


@router.get("/sheets/{sheet_name}/suggestions/athlete-forward-fill")
def get_athlete_forward_fill_suggestions(
    source_id: int,
    sheet_name: str,
    header_rows: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    """Предложить заполнить пустые ячейки в колонке ФИО значением «выше».

    Источник правды по тому, какие колонки относятся к роли ``athlete``,
    — saved mapping (`source.mapping_config`). Без сохранённого mapping
    эндпоинт возвращает пустой список и подсказку для пользователя.
    """

    source = _get_owned_source_or_404(db, user, source_id)
    _ensure_draft(source)
    parsed_rows = _parse_header_rows(header_rows)

    athlete_columns: list[str] = []
    episode_columns: list[str] = []
    if source.mapping_config:
        try:
            cfg = json.loads(source.mapping_config)
        except json.JSONDecodeError:
            cfg = None
        if isinstance(cfg, dict):
            sheets = cfg.get("sheets") or {}
            sheet_cfg = sheets.get(sheet_name) or {}
            if parsed_rows is None:
                stored_rows = sheet_cfg.get("header_rows")
                if isinstance(stored_rows, list):
                    parsed_rows = [int(x) for x in stored_rows]
            roles = sheet_cfg.get("roles") or {}
            cols = roles.get("athlete") or []
            if isinstance(cols, list):
                athlete_columns = [str(c) for c in cols]
            eps = roles.get("episode") or []
            if isinstance(eps, list):
                episode_columns = [str(c) for c in eps]

    try:
        return analysis_service.athlete_forward_fill_suggestions(
            source,
            storage.resolve,
            sheet_name=sheet_name,
            header_rows=parsed_rows,
            athlete_columns=athlete_columns,
            episode_columns=episode_columns or None,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Не удалось получить предложения по ФИО для '{sheet_name}': {exc}"
            ),
        ) from exc


@router.post("/sheets/{sheet_name}/remove-empty-rows")
def remove_empty_rows(
    source_id: int,
    sheet_name: str,
    payload: RemoveEmptyRowsRequest | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> dict[str, Any]:
    source = _get_owned_source_or_404(db, user, source_id)
    _ensure_draft(source)
    header_rows = payload.header_rows if payload else None
    try:
        deleted = analysis_service.remove_empty_rows_in_sheet(
            source,
            storage.resolve,
            sheet_name=sheet_name,
            header_rows=header_rows,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Не удалось удалить пустые строки на листе '{sheet_name}': {exc}"
            ),
        ) from exc

    sources_service.refresh_storage_metadata(db, storage, source)
    return {"deleted": deleted}
