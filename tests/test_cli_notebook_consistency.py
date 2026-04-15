from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cli_uses_public_inverse_entrypoint() -> None:
    cli_path = _repo_root() / "src" / "hidden_patterns_combat" / "cli.py"
    text = cli_path.read_text(encoding="utf-8")

    assert "from hidden_patterns_combat.app.inverse_diagnostic_cycle import run_inverse_diagnostic_cycle" in text
    assert "run_inverse_diagnostic_cycle(" in text


def test_notebook_uses_same_entrypoint_without_business_logic_duplication() -> None:
    notebook_path = _repo_root() / "notebooks" / "inverse_diagnostic_demo.ipynb"
    assert notebook_path.exists()

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    source_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if isinstance(cell, dict)
    )

    assert "run_inverse_diagnostic_cycle" in source_text
    assert "build_observed_zap_classes(" not in source_text
    assert "build_canonical_episode_table(" not in source_text
    assert "InverseDiagnosticHMM(" not in source_text
