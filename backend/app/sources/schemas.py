"""DTO для /sources."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SourceSummary(BaseModel):
    """Краткая карточка источника для левой панели UI."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    size_bytes: int
    sha256: str
    created_at: datetime
    has_analysis: bool = False
    last_analysis_status: str | None = None
    has_mapping: bool = False
