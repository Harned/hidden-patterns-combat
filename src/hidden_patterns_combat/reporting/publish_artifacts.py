from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from typing import Any


PUBLISH_RELATIVE_PATHS: tuple[str, ...] = (
    "features/episode_features.csv",
    "diagnostics/run_summary.json",
    "diagnostics/quality_diagnostics.json",
    "diagnostics/state_profile.csv",
    "diagnostics/transition_matrix.csv",
    "diagnostics/emission_params.json",
    "diagnostics/per_episode_viterbi.csv",
    "reports/inverse_diagnostic_report.md",
    "run_manifest.json",
)


@dataclass
class PublishResult:
    source_run_dir: Path
    target_run_dir: Path
    run_id: str
    copied_files: list[str]
    missing_files: list[str]
    summary_path: Path


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sanitize_run_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._-")
    return text or "published_run"


def _resolve_run_dir(source: str | Path) -> Path:
    src = Path(source)
    if src.is_file():
        raise ValueError(f"source must be a directory, got file: {src}")
    if not src.exists():
        raise FileNotFoundError(f"source directory does not exist: {src}")

    manifest_here = src / "run_manifest.json"
    if manifest_here.exists():
        return src

    candidates = [path.parent for path in src.glob("**/run_manifest.json")]
    if not candidates:
        raise FileNotFoundError(
            f"No run_manifest.json found under source: {src}. "
            "Point --source to a run directory or a parent folder that contains run directories."
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def publish_inverse_artifacts(
    *,
    source: str | Path,
    target_root: str | Path = "analytics/runs",
    run_id: str | None = None,
    include_plots: bool = False,
) -> PublishResult:
    source_run_dir = _resolve_run_dir(source)
    manifest_path = source_run_dir / "run_manifest.json"
    manifest = _load_manifest(manifest_path)

    resolved_run_id = _sanitize_run_id(run_id or manifest.get("run_id") or source_run_dir.name)
    target_run_dir = Path(target_root) / resolved_run_id
    target_run_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    missing_files: list[str] = []

    for rel in PUBLISH_RELATIVE_PATHS:
        src_path = source_run_dir / rel
        dst_path = target_run_dir / rel
        if not src_path.exists():
            missing_files.append(rel)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        copied_files.append(rel)

    if include_plots:
        for png in sorted((source_run_dir / "plots").glob("*.png")):
            rel = str(png.relative_to(source_run_dir))
            dst = target_run_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png, dst)
            copied_files.append(rel)

    summary = {
        "published_at": _utcnow_iso(),
        "source_run_dir": str(source_run_dir),
        "target_run_dir": str(target_run_dir),
        "run_id": resolved_run_id,
        "copied_files": copied_files,
        "missing_files": missing_files,
        "manifest_snapshot": {
            "status": manifest.get("status"),
            "run_id": manifest.get("run_id"),
            "run_fingerprint": manifest.get("run_fingerprint"),
            "input_file_name": manifest.get("input_file_name"),
            "input_file_hash": manifest.get("input_file_hash"),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
        },
    }
    summary_path = target_run_dir / "publish_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return PublishResult(
        source_run_dir=source_run_dir,
        target_run_dir=target_run_dir,
        run_id=resolved_run_id,
        copied_files=copied_files,
        missing_files=missing_files,
        summary_path=summary_path,
    )
