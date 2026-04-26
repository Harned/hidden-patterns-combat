from __future__ import annotations

from pathlib import Path

import pandas as pd

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.mapping import (
    describe_sheet_columns,
    flatten_columns,
    guess_header_rows,
    suggest_header_rows,
)
from hpc_algo.schema import AnalysisStatus, ColumnMappingConfig, HiddenGroup, SheetMapping


def test_flatten_columns_ignores_unnamed_and_nan() -> None:
    out = flatten_columns(
        [
            ("Баллы", "Unnamed: 1", "ЗАП"),
            ("Unnamed: 2", "Правосторонняя стойка (ПС)", "Вперед-влево"),
            ("Unnamed: 3", "Unnamed: 4", "Unnamed: 5"),
        ]
    )
    assert out[0] == "Баллы | ЗАП"
    assert out[1] == "Правосторонняя стойка (ПС) | Вперед-влево"
    # Все уровни пустые — fallback на детерминированное имя.
    assert out[2].startswith("column_")


def test_guess_header_rows_detects_three_rows(multirow_header_excel: Path) -> None:
    raw = pd.read_excel(multirow_header_excel, header=None, engine="openpyxl")
    rows = guess_header_rows(raw)
    # Первые три строки — заголовки, затем идут числовые данные.
    assert 0 in rows and 1 in rows and 2 in rows
    assert 3 not in rows


def test_suggest_header_rows_returns_preview_and_matches_current(
    multirow_header_excel: Path,
) -> None:
    s = suggest_header_rows(multirow_header_excel, "48", current_header_rows=[0, 1, 2])
    assert s.suggested == [0, 1, 2]
    assert s.matches_current is True
    assert len(s.preview) > 0
    # Превью имён содержит хотя бы одно flatten-имя с разделителем " | "
    assert any(" | " in item["name"] for item in s.preview)
    # raw_preview ограничен 5 строками.
    assert len(s.raw_preview) <= 5


def test_suggest_header_rows_flags_disagreement_with_current(
    multirow_header_excel: Path,
) -> None:
    s = suggest_header_rows(multirow_header_excel, "48", current_header_rows=[0])
    assert s.suggested == [0, 1, 2]
    assert s.current == [0]
    assert s.matches_current is False


def test_preflight_detects_zap_and_manevr(multirow_header_excel: Path) -> None:
    cfg = preflight_mapping(multirow_header_excel)
    assert "48" in cfg.sheets
    sm = cfg.sheets["48"]

    assert sm.header_rows == [0, 1, 2]
    assert HiddenGroup.ZAP in sm.roles and sm.roles[HiddenGroup.ZAP]
    assert HiddenGroup.MANEUVERING in sm.roles and sm.roles[HiddenGroup.MANEUVERING]
    assert HiddenGroup.KFV in sm.roles and sm.roles[HiddenGroup.KFV]
    assert HiddenGroup.VUP in sm.roles and sm.roles[HiddenGroup.VUP]
    assert HiddenGroup.TIME in sm.roles and sm.roles[HiddenGroup.TIME]
    assert HiddenGroup.ATHLETE in sm.roles and sm.roles[HiddenGroup.ATHLETE]


def test_analyze_with_preflight_returns_baseline_only(
    multirow_header_excel: Path,
) -> None:
    cfg = preflight_mapping(multirow_header_excel)
    result = analyze_source(
        multirow_header_excel,
        AnalyzeConfig(column_mapping=cfg),
    )

    assert result.status == AnalysisStatus.BASELINE_ONLY
    assert result.applied_mapping is not None

    # Все 4 предметные группы должны набрать наблюдений.
    totals = result.basic_statistics.hidden_group_totals
    for group in (
        HiddenGroup.ZAP.value,
        HiddenGroup.MANEUVERING.value,
        HiddenGroup.KFV.value,
        HiddenGroup.VUP.value,
    ):
        assert totals.get(group, 0) > 0, (
            f"Ожидались наблюдения для группы {group}, получено: {totals}"
        )

    # Time-статистика — непустая.
    assert result.basic_statistics.time_statistics
    # Число эпизодов на лист рассчитано.
    assert result.basic_statistics.episodes_per_sheet["48"] >= 2

    # Инвариант: HMM-полей всё ещё нет.
    dumped = result.model_dump()
    for forbidden in ("viterbi_path", "hidden_states", "gamma"):
        assert forbidden not in dumped


def test_analyze_with_empty_mapping_falls_back_to_heuristic(
    multirow_header_excel: Path,
) -> None:
    empty = ColumnMappingConfig(sheets={})
    result = analyze_source(multirow_header_excel, AnalyzeConfig(column_mapping=empty))
    # Эвристика на multi-row header при single-row парсинге может
    # увидеть ЗАП-подобный контент в колонках, попавших в данные
    # (sub-header «ЗАП» оказался в данных, а маркер «Баллы» теперь
    # распознаётся правилом ZAP). В любом случае mapping не применён.
    assert result.status in {
        AnalysisStatus.NEEDS_COLUMN_MAPPING,
        AnalysisStatus.BASELINE_ONLY,
    }
    assert result.applied_mapping is None


def test_analyze_flags_unknown_columns(multirow_header_excel: Path) -> None:
    cfg = ColumnMappingConfig(
        sheets={
            "48": SheetMapping(
                header_rows=[0, 1, 2],
                roles={
                    HiddenGroup.ZAP: ["Баллы | ЗАП"],
                    HiddenGroup.MANEUVERING: ["Не существующая колонка"],
                },
            )
        }
    )
    result = analyze_source(multirow_header_excel, AnalyzeConfig(column_mapping=cfg))
    codes = {w.code for w in result.warnings}
    assert "mapping.unknown_column" in codes


def test_describe_sheet_columns_returns_all_with_samples(
    multirow_header_excel: Path,
) -> None:
    header_rows, columns = describe_sheet_columns(multirow_header_excel, "48")
    assert header_rows == [0, 1, 2]
    names = [c["name"] for c in columns]

    # Все ожидаемые колонки фикстуры присутствуют.
    assert "ФИО борца" in names
    assert any("Время эпизода, с." in n for n in names)
    assert "Баллы | ЗАП" in names
    assert any(n.endswith("Вперед-влево") for n in names)
    assert "ВУП | Передний" in names

    # role_hint корректно проставлен для основных ролей.
    hints = {c["name"]: c["role_hint"] for c in columns}
    assert hints["ФИО борца"] == HiddenGroup.ATHLETE.value
    assert hints["Баллы | ЗАП"] == HiddenGroup.ZAP.value
    assert hints["ВУП | Передний"] == HiddenGroup.VUP.value

    # sample_values есть хотя бы у одной колонки и сериализуемы.
    any_samples = any(c["sample_values"] for c in columns)
    assert any_samples


def test_no_zap_values_yields_needs_mapping(multirow_header_excel: Path) -> None:
    cfg = ColumnMappingConfig(
        sheets={
            "48": SheetMapping(
                header_rows=[0, 1, 2],
                roles={
                    HiddenGroup.ZAP: ["Баллы | Правосторонняя стойка (ПС) | Вперед-влево"],
                    # Это колонка маневрирования, но мы «подсунули» её как ЗАП —
                    # значения там 0/1. Это не пусто, так что это даст baseline_only.
                    # Чтобы проверить needs_column_mapping, укажем заведомо отсутствующую.
                },
            )
        }
    )
    # Переопределяем через несуществующую колонку — тогда zap-values пусто.
    cfg2 = ColumnMappingConfig(
        sheets={
            "48": SheetMapping(
                header_rows=[0, 1, 2],
                roles={HiddenGroup.ZAP: ["НесуществующаяЗАП"]},
            )
        }
    )
    r2 = analyze_source(multirow_header_excel, AnalyzeConfig(column_mapping=cfg2))
    assert r2.status == AnalysisStatus.NEEDS_COLUMN_MAPPING
    codes = {w.code for w in r2.warnings}
    assert "mapping.no_zap_values" in codes
    # cfg с нашей "ложной ЗАП-колонкой" не обязателен в финальной проверке.
    _ = cfg
