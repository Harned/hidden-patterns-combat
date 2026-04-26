"""HTTP-слой /sources. Никакой логики алгоритма здесь нет."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.auth.deps import current_verified_user
from app.config import Settings, get_settings
from app.db.models import User
from app.db.session import get_db
from app.sources import service
from app.sources.schemas import SourceSummary
from app.sources.service import SourceError
from app.sources.storage import LocalStorage

router = APIRouter(prefix="/sources", tags=["sources"])


def get_storage(settings: Settings = Depends(get_settings)) -> LocalStorage:
    return LocalStorage.from_settings(settings)


@router.get("", response_model=list[SourceSummary])
def list_sources(
    db: Session = Depends(get_db),
    user: User = Depends(current_verified_user),
) -> list[SourceSummary]:
    return service.list_user_sources(db, user)


@router.post("", response_model=SourceSummary, status_code=status.HTTP_201_CREATED)
async def upload_source(
    file: UploadFile = File(..., description="Excel-источник (.xlsx или .xls)."),
    confirm_upload: bool = Form(
        False,
        description=(
            "UPLOAD-GATE-1: подтверждение пользователя, что данные обезличены, "
            "загрузка правомерна и сервис используется в исследовательском режиме."
        ),
    ),
    db: Session = Depends(get_db),
    user: User = Depends(current_verified_user),
    storage: LocalStorage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> SourceSummary:
    if not confirm_upload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Загрузка возможна только после подтверждения, что вы "
                "удалили персональные данные и подтверждаете правомерность "
                "загрузки. Установите флаг `confirm_upload=true`."
            ),
        )
    data = await file.read()
    try:
        source = service.create_source(
            db=db,
            storage=storage,
            user=user,
            original_filename=file.filename or "upload.xlsx",
            content_type=file.content_type or "",
            data=data,
            max_size=settings.max_upload_size_bytes,
        )
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return SourceSummary(
        id=source.id,
        original_filename=source.original_filename,
        size_bytes=source.size_bytes,
        sha256=source.sha256,
        created_at=source.created_at,
        has_analysis=False,
        last_analysis_status=None,
        has_mapping=False,
        preparation_state=source.preparation_state,
    )


def _summary_for(source) -> SourceSummary:
    last_run = source.analysis_runs[0] if source.analysis_runs else None
    return SourceSummary(
        id=source.id,
        original_filename=source.original_filename,
        size_bytes=source.size_bytes,
        sha256=source.sha256,
        created_at=source.created_at,
        has_analysis=last_run is not None,
        last_analysis_status=last_run.status if last_run else None,
        has_mapping=bool(source.mapping_config),
        preparation_state=source.preparation_state,
    )


@router.get("/{source_id}", response_model=SourceSummary)
def get_source(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_verified_user),
) -> SourceSummary:
    try:
        source = service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    return _summary_for(source)


@router.post("/{source_id}/finalize", response_model=SourceSummary)
def finalize_source(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_verified_user),
) -> SourceSummary:
    """Подтвердить источник после прохождения мастера предобработки."""

    try:
        source = service.get_owned_source(db, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc

    try:
        source = service.finalize_source(db, source)
    except service.FinalizeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    return _summary_for(source)


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_source(
    source_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_verified_user),
    storage: LocalStorage = Depends(get_storage),
) -> None:
    try:
        service.delete_owned_source(db, storage, user, source_id)
    except SourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
