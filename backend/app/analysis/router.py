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


@router.post("/preflight", response_model=MappingResponse)
def preflight(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
    storage: LocalStorage = Depends(get_storage),
) -> MappingResponse:
    source = _get_owned_source_or_404(db, user, source_id)
    try:
        mapping_json = analysis_service.run_preflight(source, storage.resolve)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Не удалось выполнить preflight: {exc}",
        ) from exc

    return MappingResponse(mapping=json.loads(mapping_json))


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
