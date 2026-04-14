from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .data_dictionary import DataDictionary
from .export import export_preprocessing_outputs
from .ingestion import SheetSelector, load_excel_for_preprocessing
from .transform import transform_raw_to_tidy
from .validation import ValidationReport, validate_tidy_structure

ParserMode = Literal["auto", "table", "matrix"]


@dataclass
class PreprocessingReport:
    excel_path: str
    sheets_loaded: list[str]
    rows_raw: int
    rows_cleaned: int
    columns_cleaned: int
    validation: ValidationReport
    exports: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["validation"] = self.validation.to_dict()
        return payload


def run_preprocessing(
    excel_path: str | Path,
    sheet_selector: SheetSelector = None,
    header_depth: int = 2,
    output_dir: str | Path = "data/processed/preprocessing",
    save_parquet: bool = False,
    data_dictionary: DataDictionary | None = None,
    parser_mode: ParserMode = "auto",
    force_matrix_parser: bool | None = None,
) -> PreprocessingReport:
    ingestion = load_excel_for_preprocessing(
        excel_path=excel_path,
        sheet_selector=sheet_selector,
        header_depth=header_depth,
        parser_mode=parser_mode,
        force_matrix_parser=force_matrix_parser,
    )

    transform = transform_raw_to_tidy(ingestion.raw_combined, data_dictionary=data_dictionary)
    validation = validate_tidy_structure(
        transform.cleaned,
        transform.mapping,
        data_dictionary=data_dictionary,
    )

    exports = export_preprocessing_outputs(
        output_dir=output_dir,
        raw_df=ingestion.raw_combined,
        cleaned_df=transform.cleaned,
        mapping_df=transform.mapping,
        validation=validation,
        save_parquet=save_parquet,
        episodes_tidy_df=ingestion.episodes_tidy,
        matrix_label_mapping_df=ingestion.matrix_label_mapping,
    )

    return PreprocessingReport(
        excel_path=str(Path(excel_path)),
        sheets_loaded=ingestion.sheets_loaded,
        rows_raw=len(ingestion.raw_combined),
        rows_cleaned=len(transform.cleaned),
        columns_cleaned=len(transform.cleaned.columns),
        validation=validation,
        exports=exports,
    )
