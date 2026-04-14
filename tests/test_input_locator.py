from __future__ import annotations

from pathlib import Path

import pytest

from hidden_patterns_combat.app.input_locator import resolve_input_excel


def test_resolve_input_excel_prefers_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)

    default_path = project_root / "data" / "raw" / "episodes.xlsx"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text("default", encoding="utf-8")

    env_path = tmp_path / "custom" / "episodes.xlsx"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("env", encoding="utf-8")

    monkeypatch.setenv("HPC_INPUT_XLSX", str(env_path))

    resolved = resolve_input_excel(project_root)
    assert resolved.input_path == env_path.resolve()
    assert resolved.source == "env:HPC_INPUT_XLSX"


def test_resolve_input_excel_uses_standard_data_raw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("HPC_INPUT_XLSX", raising=False)

    project_root = tmp_path / "repo"
    target = project_root / "data" / "raw" / "episodes.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("xlsx", encoding="utf-8")

    resolved = resolve_input_excel(project_root)
    assert resolved.input_path == target.resolve()
    assert str(target.resolve()) in [str(p) for p in resolved.checked_paths]


def test_resolve_input_excel_error_message_is_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("HPC_INPUT_XLSX", raising=False)
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError) as err:
        resolve_input_excel(project_root)

    msg = str(err.value)
    assert "Checked paths:" in msg
    assert "data/raw/episodes.xlsx" in msg
    assert "HPC_INPUT_XLSX" in msg
