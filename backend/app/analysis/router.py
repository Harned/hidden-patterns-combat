"""HTTP-слой анализа."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.analysis import service as analysis_service
from app.auth.deps import current_user
from app.db.models import User
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
    status: str
    algo_version: str
    created_at: datetime


class AnalysisRunFull(AnalysisRunSummary):
    result: dict[str, Any]


class MappingResponse(BaseModel):
    mapping: dict[str, Any] | None


_VALID_HMM_MODES = {"auto", "detailed", "basic", "off"}


@router.post(
    "/analyze",
    response_model=AnalysisRunSummary,
    status_code=status.HTTP_201_CREATED,
)
def analyze(
    source_id: int,
    mode: str = "auto",
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
    try:
        source = sources_service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    run = analysis_service.run_and_persist(db, source, storage.resolve, hmm_mode=mode)
    return AnalysisRunSummary.model_validate(run)


@router.get("/result", response_model=AnalysisRunFull)
def latest_result(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> AnalysisRunFull:
    try:
        source = sources_service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    if not source.analysis_runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Для этого источника анализ ещё не запускался.",
        )
    last = source.analysis_runs[0]
    return AnalysisRunFull(
        id=last.id,
        source_id=last.source_id,
        status=last.status,
        algo_version=last.algo_version,
        created_at=last.created_at,
        result=json.loads(last.result_json),
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
    """Вернуть предлагаемый ColumnMappingConfig для источника."""

    try:
        source = sources_service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

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
    try:
        source = sources_service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
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
    try:
        source = sources_service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    # Валидация через hpc_algo — единственная точка доверия к mapping.
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
    try:
        source = sources_service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
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
    except ValueError as exc:  # noqa: BLE001
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
    """Полный список flatten-колонок листа для UI-редактора mapping."""

    try:
        source = sources_service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

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
