from pathlib import Path

from hidden_patterns_combat.preprocessing.pipeline import run_preprocessing


def test_preprocessing_exports_episodes_tidy_for_matrix_sheet(matrix_excel_path: Path, tmp_path: Path):
    out_dir = tmp_path / "preprocessing"
    report = run_preprocessing(
        excel_path=matrix_excel_path,
        sheet_selector="Matrix",
        output_dir=out_dir,
    )

    exports = report.exports
    assert "episodes_tidy_csv" in exports
    assert Path(exports["episodes_tidy_csv"]).exists()
    assert "matrix_label_mapping_csv" in exports
    assert Path(exports["matrix_label_mapping_csv"]).exists()
