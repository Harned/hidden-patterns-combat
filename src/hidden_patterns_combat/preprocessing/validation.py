from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .data_dictionary import DataDictionary


@dataclass
class ValidationReport:
    is_valid: bool
    missing_blocks: list[str]
    present_blocks: list[str]
    row_count: int
    column_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_tidy_structure(
    cleaned: pd.DataFrame,
    mapping: pd.DataFrame,
    data_dictionary: DataDictionary | None = None,
) -> ValidationReport:
    dd = data_dictionary or DataDictionary.default()

    present = sorted(set(mapping["block"].astype(str).tolist()))
    missing = [b for b in dd.required_blocks if b not in present]
    is_valid = len(missing) == 0 and len(cleaned) > 0

    return ValidationReport(
        is_valid=is_valid,
        missing_blocks=missing,
        present_blocks=present,
        row_count=len(cleaned),
        column_count=len(cleaned.columns),
    )
