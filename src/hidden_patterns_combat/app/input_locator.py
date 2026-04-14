from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class InputExcelResolution:
    input_path: Path
    checked_paths: list[Path]
    source: str


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        resolved = p.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def input_search_candidates(
    project_root: Path,
    *,
    env_var: str = "HPC_INPUT_XLSX",
    extra_candidates: list[Path] | None = None,
) -> list[Path]:
    candidates: list[Path] = []

    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        candidates.append(Path(env_value))

    candidates.extend(
        [
            project_root / "data" / "raw" / "episodes.xlsx",
            project_root / "episodes.xlsx",
            project_root / "data" / "episodes.xlsx",
            project_root / "notebooks" / "episodes.xlsx",
            project_root / "input" / "episodes.xlsx",
        ]
    )

    if extra_candidates:
        candidates.extend(extra_candidates)

    return _dedupe_paths(candidates)


def resolve_input_excel(
    project_root: Path,
    *,
    env_var: str = "HPC_INPUT_XLSX",
    extra_candidates: list[Path] | None = None,
) -> InputExcelResolution:
    checked = input_search_candidates(
        project_root,
        env_var=env_var,
        extra_candidates=extra_candidates,
    )

    for p in checked:
        if p.exists() and p.is_file():
            source = f"env:{env_var}" if os.environ.get(env_var, "").strip() and p == Path(os.environ[env_var]).expanduser().resolve() else "default_candidates"
            return InputExcelResolution(input_path=p, checked_paths=checked, source=source)

    checked_lines = "\n".join(f"- {p}" for p in checked)
    raise FileNotFoundError(
        "Input Excel file not found for notebook run.\n"
        "Checked paths:\n"
        f"{checked_lines}\n\n"
        "How to fix:\n"
        "1) Put workbook at data/raw/episodes.xlsx (recommended), or\n"
        f"2) Set environment variable {env_var}=/absolute/path/to/episodes.xlsx"
    )
