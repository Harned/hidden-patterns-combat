from hidden_patterns_combat.preprocessing.ingestion import load_excel_for_preprocessing


def test_ingestion_supports_single_list_and_all(demo_excel_path):
    one = load_excel_for_preprocessing(demo_excel_path, sheet_selector="Общее", header_depth=2)
    assert one.sheets_loaded == ["Общее"]
    assert one.raw_combined["_sheet"].nunique() == 1

    many = load_excel_for_preprocessing(demo_excel_path, sheet_selector=["Общее", "48"], header_depth=2)
    assert many.raw_combined["_sheet"].nunique() == 2

    all_sheets = load_excel_for_preprocessing(demo_excel_path, sheet_selector=None, header_depth=2)
    assert set(all_sheets.sheets_loaded) == {"Общее", "48"}


def test_ingestion_raises_for_unknown_sheet(demo_excel_path):
    try:
        load_excel_for_preprocessing(demo_excel_path, sheet_selector=["UNKNOWN"], header_depth=2)
    except ValueError as exc:
        assert "Unknown sheet(s)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown sheet selection")
