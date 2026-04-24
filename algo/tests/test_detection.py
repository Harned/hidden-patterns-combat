from __future__ import annotations

from pathlib import Path

from hpc_algo.detection import detect_columns, strong_zap_candidates
from hpc_algo.loading import load_excel
from hpc_algo.schema import HiddenGroup


def test_detects_strong_zap(zap_only_excel: Path) -> None:
    loaded = load_excel(zap_only_excel)
    report = detect_columns(loaded)

    zap = strong_zap_candidates(report)
    assert zap, "должен быть хотя бы один уверенный ЗАП-кандидат"
    assert all(c.group == HiddenGroup.ZAP for c in zap)
    assert HiddenGroup.ZAP in report.detected_groups


def test_detects_all_required_groups(full_structure_excel: Path) -> None:
    loaded = load_excel(full_structure_excel)
    report = detect_columns(loaded)

    required = {
        HiddenGroup.ZAP,
        HiddenGroup.MANEUVERING,
        HiddenGroup.KFV,
        HiddenGroup.VUP,
        HiddenGroup.TIME,
    }
    assert required.issubset(set(report.detected_groups)), (
        f"Ожидались все обязательные группы, получены: {report.detected_groups}"
    )


def test_opaque_headers_yield_no_detection(opaque_excel: Path) -> None:
    loaded = load_excel(opaque_excel)
    report = detect_columns(loaded)

    assert report.candidates == []
    assert report.detected_groups == []
    # Все обязательные группы должны быть в missing.
    for g in (
        HiddenGroup.ZAP,
        HiddenGroup.MANEUVERING,
        HiddenGroup.KFV,
        HiddenGroup.VUP,
        HiddenGroup.TIME,
    ):
        assert g in report.missing_groups
