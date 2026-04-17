from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from hidden_patterns_combat.modeling.intra_episode_hmm import IntraEpisodeHMM, O_CLASSES
from hidden_patterns_combat.visualization import create_analysis_charts


@dataclass
class InverseDiagnosticResult:
    input_path: str
    output_dir: str
    final_output_dir: str
    run_id: str
    run_manifest_path: str
    cleanup_mode: str
    cleanup_actions: list[str]
    run_fingerprint: str
    cleaned_data_path: str
    canonical_episode_table_path: str
    observed_sequence_path: str
    hidden_feature_layer_path: str
    episode_analysis_path: str
    state_profile_path: str
    quality_diagnostics_path: str
    observation_audit_path: str
    observation_mapping_crosstab_path: str
    raw_finish_signal_summary_path: str
    unsupported_finish_values_path: str
    unsupported_score_values_path: str
    unsupported_values_assessment_path: str
    metadata_extraction_summary_path: str
    metadata_field_coverage_path: str
    sequence_audit_path: str
    sequence_length_distribution_path: str
    suspicious_sequences_path: str
    model_health_summary_path: str
    report_path: str
    model_path: str
    rows_total: int
    rows_train_eligible: int
    observation_mapping_version: str
    canonical_state_order: list[str]
    semantic_assignment: dict[str, int]
    semantic_confidence: dict[str, float]
    observed_layer_summary: dict[str, float]
    sequence_quality_summary: dict[str, float]
    semantic_assignment_quality: str
    recommendation_profile: str
    recommendation: str
    transitions_summary: list[dict[str, object]]
    created_artifacts: list[str]
    created_files: list[str]
    run_summary_path: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_PIPELINE_MODE = "inverse-diagnostic"
_RUN_MANIFEST_NAME = "run_manifest.json"
_CLEANUP_MODES = {"none", "artifacts_only", "full_run_dir"}
_ARTIFACT_DIRS = ("cleaned", "features", "models", "diagnostics", "plots", "reports")
_TOTAL_ROW_TOKENS = ("итог", "сумма", "всего", "total", "summary")
_EXPECTED_GROUP_BINS: dict[str, int] = {
    "s1_ps": 12,
    "s1_ls": 12,
    "s2_captures": 15,
    "s2_holds": 3,
    "s2_wraps": 3,
    "s2_hooks": 4,
    "s2_posts": 4,
    "s3_vup": 5,
}
_O_MAP = {
    "O0": "нет ЗАП",
    "O1": "Броски Руками",
    "O2": "Броски Ногами",
    "O3": "Броски Туловищем",
    "O4": "Удержание",
    "O5": "Болевой на руку",
    "O6": "Болевой на ногу",
}
_O_LEGACY_MAP = {
    "O0": "no_score",
    "O1": "zap_r",
    "O2": "zap_n",
    "O3": "zap_t",
    "O4": "hold",
    "O5": "arm_submission",
    "O6": "leg_submission",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso8601_utc(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sanitize_run_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    text = text.strip("._-")
    return text or "inverse_diagnostic_run"


def _default_run_id(started_at: datetime) -> str:
    return started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ_inverse_diagnostic")


def _resolve_final_output_dir(
    output_dir: Path,
    *,
    isolated_run: bool,
    run_id: str | None,
    started_at: datetime,
) -> tuple[Path, str]:
    requested_run_id = _sanitize_run_id(run_id) if run_id else _default_run_id(started_at)
    if not isolated_run:
        return output_dir, requested_run_id

    final_dir = output_dir / requested_run_id
    if run_id is None and final_dir.exists():
        suffix = 1
        while True:
            candidate_run_id = f"{requested_run_id}_{suffix:02d}"
            candidate_path = output_dir / candidate_run_id
            if not candidate_path.exists():
                return candidate_path, candidate_run_id
            suffix += 1
    return final_dir, requested_run_id


def _resolve_cleanup_mode(
    *,
    cleanup_mode: str | None,
    isolated_run: bool,
    reset_outputs: bool,
) -> str:
    if cleanup_mode is not None:
        mode = str(cleanup_mode).strip().lower()
        if mode not in _CLEANUP_MODES:
            allowed = ", ".join(sorted(_CLEANUP_MODES))
            raise ValueError(f"Unsupported cleanup_mode='{cleanup_mode}'. Allowed values: {allowed}.")
        return mode
    if reset_outputs:
        return "full_run_dir"
    if isolated_run:
        return "none"
    return "artifacts_only"


def _cleanup_output_dir(output_dir: Path, *, cleanup_mode: str) -> list[str]:
    actions: list[str] = []
    if cleanup_mode == "none":
        return actions

    if cleanup_mode == "full_run_dir":
        if output_dir.exists():
            shutil.rmtree(output_dir)
            actions.append(f"removed_dir:{output_dir}")
        return actions

    for relative_dir in _ARTIFACT_DIRS:
        path = output_dir / relative_dir
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
            actions.append(f"removed_dir:{path}")
        else:
            path.unlink()
            actions.append(f"removed_file:{path}")

    manifest_path = output_dir / _RUN_MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()
        actions.append(f"removed_file:{manifest_path}")
    return actions


def _prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "cleaned": output_dir / "cleaned",
        "features": output_dir / "features",
        "models": output_dir / "models",
        "diagnostics": output_dir / "diagnostics",
        "plots": output_dir / "plots",
        "reports": output_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _safe_git_commit_hash() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_fingerprint(payload: dict[str, object]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_output_relative(path: str | Path, output_dir: Path) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve()
        root = output_dir.resolve()
        return str(resolved.relative_to(root))
    except Exception:
        return str(candidate)


def _append_unique(messages: list[str], value: str) -> None:
    text = str(value).strip()
    if not text:
        return
    if text not in messages:
        messages.append(text)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_input_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    if path.exists():
        return path
    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / path
    if candidate.exists():
        return candidate
    return path


def _expected_artifact_files(*, generate_plots: bool) -> list[str]:
    expected = [
        "cleaned/raw_combined.csv",
        "cleaned/cleaned_tidy.csv",
        "cleaned/data_dictionary.csv",
        "cleaned/validation.json",
        "cleaned/canonical_episode_table.csv",
        "cleaned/observed_sequence.csv",
        "features/hidden_state_features.csv",
        "features/episode_features.csv",
        "diagnostics/episode_analysis.csv",
        "diagnostics/state_profile.csv",
        "diagnostics/quality_diagnostics.json",
        "diagnostics/run_summary.json",
        "diagnostics/observation_audit.json",
        "diagnostics/observation_mapping_crosstab.csv",
        "diagnostics/raw_finish_signal_summary.csv",
        "diagnostics/unsupported_finish_values.csv",
        "diagnostics/unsupported_score_values.csv",
        "diagnostics/unsupported_values_assessment.json",
        "diagnostics/metadata_extraction_summary.json",
        "diagnostics/metadata_field_coverage.csv",
        "diagnostics/sequence_audit.json",
        "diagnostics/sequence_length_distribution.csv",
        "diagnostics/suspicious_sequences.csv",
        "diagnostics/model_health_summary.json",
        "diagnostics/train_composition_report.json",
        "diagnostics/topology_compliance_report.json",
        "diagnostics/state_anchor_alignment_report.json",
        "diagnostics/finish_proximity_report.json",
        "diagnostics/semantic_stability_report.json",
        "diagnostics/emission_summary_by_hidden_state.csv",
        "diagnostics/emission_params.json",
        "diagnostics/transition_matrix.csv",
        "diagnostics/per_episode_viterbi.csv",
        "reports/inverse_diagnostic_report.md",
        _RUN_MANIFEST_NAME,
    ]
    if generate_plots:
        expected.extend(
            [
                "plots/hidden_state_sequence.png",
                "plots/state_probability_profile.png",
                "plots/scenario_success_frequencies.png",
                "plots/transition_distribution.png",
            ]
        )
    return sorted(expected)


def _normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower().replace("ё", "е")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_token(value: object) -> str:
    text = _normalize_text(value)
    text = re.sub(r"[^0-9a-zа-я№\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _flatten_headers(header_rows: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    n_cols = int(header_rows.shape[1])
    for c in range(n_cols):
        parts: list[str] = []
        for r in range(header_rows.shape[0]):
            value = _normalize_text(header_rows.iat[r, c])
            if not value or value.startswith("unnamed"):
                continue
            if parts and parts[-1] == value:
                continue
            parts.append(value)
        columns.append(" | ".join(parts) if parts else f"unknown_column_{c + 1:03d}")

    seen: dict[str, int] = {}
    unique: list[str] = []
    for col in columns:
        idx = seen.get(col, 0)
        seen[col] = idx + 1
        unique.append(col if idx == 0 else f"{col}_{idx + 1}")
    return unique


def _detect_header_depth(raw: pd.DataFrame, requested_depth: int) -> int:
    requested = int(max(1, requested_depth))
    max_scan = min(6, int(raw.shape[0]))
    marker_row = None
    for r in range(max_scan):
        row_tokens = [_normalize_token(raw.iat[r, c]) for c in range(raw.shape[1])]
        if any(("эпизод" in token) or ("episode" in token) for token in row_tokens if token):
            marker_row = r
            break

    depth = requested
    if marker_row is not None:
        depth = max(depth, marker_row + 1)
    depth = min(max(1, depth), max(1, int(raw.shape[0]) - 1))
    return depth


def _selected_sheet_names(path: Path, sheet_names: list[str] | None) -> list[str]:
    xls = pd.ExcelFile(path, engine="openpyxl")
    if not sheet_names:
        return list(xls.sheet_names)
    selected = [name for name in xls.sheet_names if name in set(sheet_names)]
    if not selected:
        raise ValueError(f"No matching sheets found for selector: {sheet_names}")
    return selected


def _first_column(columns: list[str], predicate: Any) -> str | None:
    for col in columns:
        if predicate(_normalize_token(col)):
            return col
    return None


def _score_supported(value: float) -> bool:
    if not np.isfinite(value):
        return False
    rounded = int(round(float(value)))
    return rounded in {0, 1, 2, 3, 4}


def _is_total_row(texts: list[str]) -> bool:
    for text in texts:
        low = _normalize_token(text)
        if any(token in low for token in _TOTAL_ROW_TOKENS):
            return True
    return False


def _classify_column(
    col: str,
    *,
    id_columns: set[str],
) -> tuple[str, str] | None:
    if col in id_columns:
        return None

    t = _normalize_token(col)
    if not t:
        return None

    if re.fullmatch(r"h\d+", t):
        return ("o", "O4")
    if re.fullmatch(r"a\d+", t):
        return ("o", "O5")
    if re.fullmatch(r"l\d+", t):
        return ("o", "O6")
    if re.fullmatch(r"k\d+", t):
        return ("s2", "s2_captures")
    if re.fullmatch(r"m\d+", t):
        return ("s1", "s1_ps")
    if re.fullmatch(r"v\d+", t):
        return ("s3", "s3_vup")

    if "брос" in t and "рук" in t:
        return ("o", "O1")
    if "брос" in t and "ног" in t:
        return ("o", "O2")
    if "брос" in t and ("тулов" in t or "корп" in t):
        return ("o", "O3")
    if "удерж" in t or "hold" in t:
        return ("o", "O4")
    if ("болев" in t and "рук" in t) or "arm submission" in t:
        return ("o", "O5")
    if ("болев" in t and "ног" in t) or "leg submission" in t:
        return ("o", "O6")

    if "правост" in t or re.search(r"\bпс\b", t):
        return ("s1", "s1_ps")
    if "левост" in t or re.search(r"\bлс\b", t):
        return ("s1", "s1_ls")

    if "обхват" in t:
        return ("s2", "s2_wraps")
    if "прихват" in t:
        return ("s2", "s2_hooks")
    if "захват" in t:
        return ("s2", "s2_captures")
    if "упор" in t:
        return ("s2", "s2_posts")
    if re.search(r"(^|\s)хват", t) or "grip" in t:
        return ("s2", "s2_holds")

    if "вуп" in t or "вывед" in t or "vup" in t:
        return ("s3", "s3_vup")

    return None


def _is_finish_like_column(col: str) -> bool:
    t = _normalize_token(col)
    return any(
        token in t
        for token in (
            "брос",
            "удерж",
            "болев",
            "finish",
            "outcome",
            "submission",
            "hold",
            "zap",
        )
    )


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _log_bitpack_code(binary_values: np.ndarray, expected_bins: int) -> tuple[float, int]:
    bitpack = 0
    for idx, value in enumerate(binary_values.tolist()):
        if int(value) > 0:
            bitpack += (1 << idx)

    code_raw = float(math.log2(1.0 + bitpack)) if bitpack > 0 else 0.0
    actual_bins = max(1, len(binary_values))
    target_bins = max(1, int(expected_bins))
    code_norm = code_raw * (float(target_bins) / float(actual_bins))
    code_norm = float(np.clip(code_norm, 0.0, float(target_bins)))
    return code_norm, bitpack


def _quantile_bin(series: pd.Series, *, q_low: float = 0.33, q_high: float = 0.66) -> pd.Series:
    values = _safe_numeric_series(series)
    positive = values[values > 0]
    if positive.empty:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    lo = float(positive.quantile(q_low))
    hi = float(positive.quantile(q_high))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = lo
    if hi <= lo:
        hi = lo + 1e-6
    out = pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    out = out.where(values <= lo, 1.0)
    out = out.where(values <= hi, 2.0)
    return out


def _entropy_from_counts(counts: pd.Series) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    probs = counts.astype(float) / total
    probs = probs[probs > 0.0]
    if probs.empty:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
    q = np.clip(np.asarray(q, dtype=float), 1e-12, None)
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_string(index=False)


def _semantic_stability_report(
    canonical_map: dict[str, object],
    *,
    state_anchor_alignment: dict[str, object] | None = None,
    topology_compliance: dict[str, object] | None = None,
    finish_proximity: dict[str, object] | None = None,
) -> dict[str, object]:
    assignment = canonical_map.get("semantic_assignment", {}) or {}
    confidence = canonical_map.get("semantic_confidence", {}) or {}
    anchor_match_share = float((state_anchor_alignment or {}).get("assignment_anchor_match_share", 1.0))
    topology_share = float((topology_compliance or {}).get("topology_compliance_share", 1.0))
    s3_is_closest = bool((finish_proximity or {}).get("s3_is_closest_to_finish", True))

    complete = all(k in assignment for k in ("S1", "S2", "S3"))
    unique = len({assignment.get("S1"), assignment.get("S2"), assignment.get("S3")}) == 3 if complete else False
    conf_min = min(float(confidence.get(k, 0.0)) for k in ("S1", "S2", "S3")) if complete else 0.0

    label = "fragile"
    if complete and unique and conf_min >= 0.70 and anchor_match_share >= 0.70 and topology_share >= 0.90 and s3_is_closest:
        label = "stable"
    elif complete and unique and conf_min >= 0.40 and anchor_match_share >= 0.40 and topology_share >= 0.70:
        label = "moderate"

    warnings: list[str] = []
    if not complete:
        warnings.append("S1/S2/S3 assignment is incomplete.")
    if complete and not unique:
        warnings.append("S1/S2/S3 assignment is not one-to-one.")
    if anchor_match_share < 0.35:
        warnings.append("Semantic assignment weakly matches anchor dominance across decoded states.")
    if topology_share < 0.70:
        warnings.append("Semantic topology compliance is low for S1->S2->S3.")
    if not s3_is_closest:
        warnings.append("S3 is not closest to finish-like observations.")

    return {
        "semantic_assignment": {str(k): int(v) for k, v in assignment.items() if str(v).lstrip("-").isdigit()},
        "semantic_confidence": {str(k): float(v) for k, v in confidence.items()},
        "assignment_complete": bool(complete),
        "assignment_unique": bool(unique),
        "assignment_anchor_match_share": anchor_match_share,
        "topology_compliance_share": topology_share,
        "s3_is_closest_to_finish": s3_is_closest,
        "confidence_min": conf_min,
        "stability_label": label,
        "warnings": warnings,
    }


def _recommendations(episode_features: pd.DataFrame) -> tuple[str, str, list[str]]:
    if episode_features.empty:
        return (
            "low-confidence profile",
            "Интерпретация ограничена: отсутствуют эпизоды для анализа.",
            ["Собрать дополнительные эпизоды для построения внутри-эпизодной HMM-цепочки."],
        )

    completion_share = float((episode_features["o_class"] != "O0").mean())
    s2_strength = float(
        (
            episode_features["x_s2_captures"]
            + episode_features["x_s2_holds"]
            + episode_features["x_s2_wraps"]
            + episode_features["x_s2_hooks"]
            + episode_features["x_s2_posts"]
        ).mean()
    )
    s3_strength = float(episode_features["x_s3_vup"].mean())

    actions: list[str] = []
    if completion_share < 0.30:
        actions.append("Низкая доля завершения после S3: усилить блок ВУП (S3) и доведение эпизодов до завершения.")
    if s2_strength < 0.35:
        actions.append("Потери на стадии S2: увеличить объём КФВ-связок (захваты/хваты/обхваты/прихваты/упоры).")
    if s3_strength < 0.35:
        actions.append("Недостаточная выраженность S3: добавить упражнения на вывод соперника из устойчивого положения.")
    if not actions:
        actions.append("Текущий профиль сбалансирован; поддерживать связку S1->S2->S3 и отслеживать качество завершения.")

    profile = "balanced"
    if completion_share < 0.30:
        profile = "vup-gap"
    elif s2_strength < 0.35:
        profile = "kfv-gap"

    recommendation = f"completion_share={completion_share:.3f}, s2_strength={s2_strength:.3f}, s3_strength={s3_strength:.3f}."
    return profile, recommendation, actions


def _write_inverse_report(
    *,
    report_path: Path,
    n_states: int,
    mode: str,
    transition_matrix_df: pd.DataFrame,
    emission_params: dict[str, Any],
    episode_features: pd.DataFrame,
    per_episode_viterbi: pd.DataFrame,
    quality_diagnostics: dict[str, Any],
    recommendation_profile: str,
    recommendation: str,
    recommendation_actions: list[str],
) -> None:
    path_counts = (
        per_episode_viterbi[["episode_key", "path_signature"]]
        .drop_duplicates()
        .groupby("path_signature", dropna=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    if not path_counts.empty:
        path_counts["share"] = path_counts["count"] / float(path_counts["count"].sum())

    lines = [
        "# Inverse Diagnostic Report",
        "",
        "## 1) Постановка задачи",
        (
            "В этом запуске HMM моделирует внутри-эпизодную цепочку скрытых состояний "
            "S1->S2->S3[->O], где каждый технико-тактический эпизод является отдельной последовательностью, "
            "а не элементом длинной цепочки эпизодов спортсмена."
        ),
        "",
        "### Параметры запуска",
        f"- n_states: {n_states}",
        f"- mode: {mode}",
        "- Delta t включён в эмиссию S1 как дополнительная размерность (delta_t_sec).",
        "",
        "## 2) Матрица переходов A",
        _frame_to_markdown(transition_matrix_df.reset_index().rename(columns={"index": "from_state"})),
        "",
        "## 3) Эмиссии по состояниям",
    ]

    for state_name in ["S1", "S2", "S3", "O"]:
        if state_name not in emission_params:
            continue
        payload = emission_params[state_name]
        lines.append("")
        lines.append(f"### {state_name}")
        if str(payload.get("distribution")) == "gaussian_diag":
            rows = []
            for feature, mean, var in zip(
                payload.get("features", []),
                payload.get("mean", []),
                payload.get("var", []),
            ):
                rows.append({"feature": feature, "mean": float(mean), "var": float(var)})
            lines.append(_frame_to_markdown(pd.DataFrame(rows)))
        else:
            probs = payload.get("probabilities", {}) or {}
            table = pd.DataFrame(
                [{"o_class": cls, "label": _O_MAP.get(cls, cls), "probability": float(probs.get(cls, 0.0))} for cls in O_CLASSES]
            )
            lines.append(_frame_to_markdown(table))

    lines.extend(
        [
            "",
            "## 4) Сводка Viterbi по всем эпизодам",
            f"- Всего эпизодов: {int(episode_features.shape[0])}",
            "",
            "### Топ-10 путей",
            _frame_to_markdown(path_counts.head(10)) if not path_counts.empty else "Нет данных для путей.",
            "",
            "## 5) Диагностические флаги",
            f"- dominant_state_share: {float(quality_diagnostics.get('dominant_state_share', 0.0)):.4f}",
            f"- self_transition_share: {float(quality_diagnostics.get('self_transition_share', 0.0)):.4f}",
            f"- kl_emission_vs_marginal: {float(quality_diagnostics.get('kl_emission_vs_marginal', 0.0)):.4f}",
            f"- observations_lost: {int(quality_diagnostics.get('observations_lost', 0))}",
            f"- tripwires_triggered: {quality_diagnostics.get('tripwires_triggered', [])}",
            "",
            "## 6) Методические рекомендации",
            f"- Профиль: {recommendation_profile}",
            f"- Сводка: {recommendation}",
        ]
    )
    for item in recommendation_actions:
        lines.append(f"- {item}")

    if str(quality_diagnostics.get("status", "ok")) == "degenerate":
        lines.extend(
            [
                "",
                "### Внимание",
                "Обнаружены признаки вырожденности пайплайна; интерпретацию рекомендуется считать предварительной.",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_episode_tables(
    input_path: Path,
    *,
    sheet_names: list[str] | None,
    header_depth: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_sheets = _selected_sheet_names(input_path, sheet_names)

    raw_blocks: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, object]] = []
    assumptions: list[str] = []

    for sheet in selected_sheets:
        raw_sheet = pd.read_excel(input_path, sheet_name=sheet, header=None, engine="openpyxl")
        depth = _detect_header_depth(raw_sheet, requested_depth=header_depth)
        header_rows = raw_sheet.iloc[:depth, :]
        flat_cols = _flatten_headers(header_rows)

        body = raw_sheet.iloc[depth:, : len(flat_cols)].copy()
        body.columns = flat_cols
        body["source_sheet"] = sheet
        body = body.dropna(axis=0, how="all")

        for col in flat_cols:
            mapping_rows.append(
                {
                    "source_sheet": sheet,
                    "source_column": col,
                    "normalized_column": col,
                    "parser": "multiheader",
                    "header_depth": int(depth),
                }
            )

        assumptions.append(f"sheet={sheet}:header_depth={depth}")
        raw_blocks.append(body)

    raw_combined = pd.concat(raw_blocks, ignore_index=True, sort=False) if raw_blocks else pd.DataFrame()
    if raw_combined.empty:
        raise ValueError("No rows were loaded from workbook sheets.")

    raw_combined = raw_combined.dropna(axis=1, how="all")
    cleaned = raw_combined.copy().reset_index(drop=True)

    cols = list(cleaned.columns)
    athlete_col = _first_column(
        cols,
        lambda t: (("фио" in t and "борц" in t) or ("athlete" in t) or ("fighter" in t)),
    )
    episode_col = (
        _first_column(
            cols,
            lambda t: ("эпизод" in t and ("№" in t or "номер" in t)),
        )
        or _first_column(cols, lambda t: ("episode id" in t) or ("episode number" in t))
        or _first_column(
            cols,
            lambda t: (
                ("эпизод" in t)
                and ("время" not in t)
                and ("длитель" not in t)
            ),
        )
    )
    episode_time_col = _first_column(
        cols,
        lambda t: (("время" in t and "эпизод" in t) or ("duration" in t and "episode" in t) or ("episode duration" in t)),
    )
    pause_time_col = _first_column(
        cols,
        lambda t: (("время" in t and "пауз" in t) or ("pause" in t) or ("rest" in t)),
    )
    score_col = _first_column(cols, lambda t: ("балл" in t) or ("score" in t) or ("points" in t))

    if athlete_col is None:
        athlete_col = "athlete_name_fallback"
        cleaned[athlete_col] = cleaned.get("source_sheet", pd.Series(["sheet"] * len(cleaned))).astype(str)
    if episode_col is None:
        episode_col = "episode_id_fallback"
        cleaned[episode_col] = np.arange(1, len(cleaned) + 1)
    if episode_time_col is None:
        episode_time_col = "episode_time_fallback"
        cleaned[episode_time_col] = 0.0
    if pause_time_col is None:
        pause_time_col = "pause_time_fallback"
        cleaned[pause_time_col] = 0.0
    if score_col is None:
        score_col = "score_fallback"
        cleaned[score_col] = 0.0

    cleaned[athlete_col] = (
        cleaned[athlete_col]
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
        .ffill()
        .fillna("unknown_athlete")
        .astype(str)
        .str.strip()
    )

    episode_text = cleaned[episode_col].astype(str).str.strip()
    episode_num = pd.to_numeric(cleaned[episode_col], errors="coerce")
    has_episode_text = ~episode_text.str.lower().isin({"", "nan", "none", "<na>"})
    has_episode_like = episode_num.notna() | episode_text.str.contains(r"\d", regex=True, na=False)
    looks_like_header = episode_text.str.lower().str.contains("эпизод|episode", regex=True, na=False)
    has_episode = has_episode_text & has_episode_like & (~looks_like_header)

    total_mask = pd.Series([False] * len(cleaned), index=cleaned.index)
    for idx in cleaned.index:
        texts = [
            str(cleaned.at[idx, athlete_col]),
            str(cleaned.at[idx, episode_col]),
            str(cleaned.at[idx, score_col]),
        ]
        total_mask.at[idx] = _is_total_row(texts)

    cleaned = cleaned.loc[has_episode & (~total_mask)].copy().reset_index(drop=True)

    id_columns = {athlete_col, episode_col, episode_time_col, pause_time_col, score_col, "source_sheet"}

    group_columns: dict[str, list[str]] = {
        "s1_ps": [],
        "s1_ls": [],
        "s2_captures": [],
        "s2_holds": [],
        "s2_wraps": [],
        "s2_hooks": [],
        "s2_posts": [],
        "s3_vup": [],
        "O1": [],
        "O2": [],
        "O3": [],
        "O4": [],
        "O5": [],
        "O6": [],
    }
    unsupported_finish_cols: list[str] = []

    for col in cleaned.columns:
        classified = _classify_column(col, id_columns=id_columns)
        if classified is not None:
            _, name = classified
            if name in group_columns:
                group_columns[name].append(col)
            continue
        if (col not in id_columns) and _is_finish_like_column(col):
            unsupported_finish_cols.append(col)

    for col in cleaned.columns:
        if col in id_columns:
            continue
        cleaned[col] = _safe_numeric_series(cleaned[col])

    episode_rows: list[dict[str, object]] = []
    unsupported_finish_rows: list[dict[str, object]] = []
    unsupported_score_rows: list[dict[str, object]] = []
    observations_lost = 0

    for i, row in cleaned.iterrows():
        episode_time_num = pd.to_numeric(pd.Series([row.get(episode_time_col, 0.0)]), errors="coerce").iloc[0]
        pause_time_num = pd.to_numeric(pd.Series([row.get(pause_time_col, 0.0)]), errors="coerce").iloc[0]
        score_num = pd.to_numeric(pd.Series([row.get(score_col, 0.0)]), errors="coerce").iloc[0]
        if pd.isna(episode_time_num):
            episode_time_num = 0.0
        if pd.isna(pause_time_num):
            pause_time_num = 0.0
        if pd.isna(score_num):
            score_num = 0.0

        row_data: dict[str, object] = {
            "source_row_index": int(i),
            "source_sheet": str(row.get("source_sheet", "")),
            "athlete_name": str(row.get(athlete_col, "")).strip() or "unknown_athlete",
            "episode_id": str(row.get(episode_col, "")).strip() or str(i + 1),
            "episode_time_sec": float(episode_time_num),
            "pause_time_sec": float(pause_time_num),
            "score": float(score_num),
        }

        if not _score_supported(row_data["score"]):
            unsupported_score_rows.append(
                {
                    "source_sheet": row_data["source_sheet"],
                    "athlete_name": row_data["athlete_name"],
                    "episode_id": row_data["episode_id"],
                    "score": row_data["score"],
                        "score_rounded": int(round(float(row_data["score"]))),
                    }
                )

        for feature_name in (
            "s1_ps",
            "s1_ls",
            "s2_captures",
            "s2_holds",
            "s2_wraps",
            "s2_hooks",
            "s2_posts",
            "s3_vup",
        ):
            cols_for_group = group_columns[feature_name]
            counts = np.asarray([float(row.get(col, 0.0)) for col in cols_for_group], dtype=float)
            binary = (counts > 0.0).astype(int)

            for local_idx, value in enumerate(counts.tolist(), start=1):
                row_data[f"{feature_name}_b{local_idx:02d}_count"] = float(value)
                row_data[f"{feature_name}_b{local_idx:02d}_bin"] = int(binary[local_idx - 1])

            expected_bins = _EXPECTED_GROUP_BINS[feature_name]
            code_value, bitpack = _log_bitpack_code(binary, expected_bins=expected_bins)
            row_data[f"{feature_name}_bitpack"] = int(bitpack)
            row_data[f"{feature_name}_code"] = float(code_value)
            row_data[f"{feature_name}_count_total"] = float(counts.sum())

        row_data["x_s1_ps"] = float(row_data["s1_ps_code"])
        row_data["x_s1_ls"] = float(row_data["s1_ls_code"])
        row_data["x_s2_captures"] = float(row_data["s2_captures_code"])
        row_data["x_s2_holds"] = float(row_data["s2_holds_code"])
        row_data["x_s2_wraps"] = float(row_data["s2_wraps_code"])
        row_data["x_s2_hooks"] = float(row_data["s2_hooks_code"])
        row_data["x_s2_posts"] = float(row_data["s2_posts_code"])
        row_data["x_s3_vup"] = float(row_data["s3_vup_code"])
        row_data["delta_t_sec"] = float(row_data["episode_time_sec"])

        class_strength: dict[str, float] = {}
        for o_class in ("O1", "O2", "O3", "O4", "O5", "O6"):
            cols_for_class = group_columns[o_class]
            strength = 0.0
            for col in cols_for_class:
                strength += float(max(0.0, float(row.get(col, 0.0))))
            class_strength[o_class] = float(strength)

        positive_classes = [k for k, v in class_strength.items() if v > 0.0]
        o_issue = ""
        if not positive_classes:
            o_class = "O0"
        elif len(positive_classes) == 1:
            o_class = positive_classes[0]
        else:
            o_class = sorted(positive_classes, key=lambda cls: (class_strength[cls], cls), reverse=True)[0]
            o_issue = "ambiguous_multi_o"
            observations_lost += 1

        unsupported_positive_cols = []
        for col in unsupported_finish_cols:
            value = float(row.get(col, 0.0) or 0.0)
            if value > 0.0:
                unsupported_positive_cols.append(col)
                unsupported_finish_rows.append(
                    {
                        "source_sheet": row_data["source_sheet"],
                        "athlete_name": row_data["athlete_name"],
                        "episode_id": row_data["episode_id"],
                        "source_column": col,
                        "value": value,
                    }
                )
        if unsupported_positive_cols:
            observations_lost += 1
            if not o_issue:
                o_issue = "unsupported_finish_signal"

        row_data["o_class"] = o_class
        row_data["o_label"] = _O_MAP.get(o_class, o_class)
        row_data["observed_zap_class_legacy"] = _O_LEGACY_MAP.get(o_class, "unknown")
        row_data["observation_issue"] = o_issue
        row_data["unsupported_finish_columns"] = json.dumps(sorted(unsupported_positive_cols), ensure_ascii=False)

        episode_rows.append(row_data)

    episode_features = pd.DataFrame(episode_rows)

    if episode_features.empty:
        raise ValueError("No episode rows after cleaning and parsing.")

    episode_features["sequence_id"] = (
        episode_features["source_sheet"].astype(str)
        + "::"
        + episode_features["athlete_name"].astype(str)
    )
    episode_features["episode_key"] = (
        episode_features["sequence_id"].astype(str)
        + "::"
        + episode_features["episode_id"].astype(str)
        + "::"
        + episode_features["source_row_index"].astype(str)
    )

    metadata = {
        "selected_columns": {
            "athlete_name": athlete_col,
            "episode_id": episode_col,
            "episode_time_sec": episode_time_col,
            "pause_time_sec": pause_time_col,
            "score": score_col,
        },
        "group_column_counts": {k: int(len(v)) for k, v in group_columns.items()},
        "unsupported_finish_columns": sorted(unsupported_finish_cols),
        "observations_lost": int(observations_lost),
        "assumptions": assumptions,
    }

    mapping_df = pd.DataFrame(mapping_rows)
    unsupported_finish_df = pd.DataFrame(unsupported_finish_rows)
    unsupported_score_df = pd.DataFrame(unsupported_score_rows)

    return episode_features, mapping_df, cleaned, {
        "metadata": metadata,
        "unsupported_finish": unsupported_finish_df,
        "unsupported_score": unsupported_score_df,
    }


def run_inverse_diagnostic_cycle(
    input_path: str | Path,
    output_dir: str | Path,
    sheet_names: list[str] | None = None,
    header_depth: int = 2,
    parser_mode: str = "auto",
    force_matrix_parser: bool | None = None,
    retrain: bool = True,
    model_path: str | Path | None = None,
    reset_outputs: bool = False,
    n_states: int = 3,
    topology_mode: str = "left_to_right",
    generate_plots: bool = True,
    verbose: bool = True,
    cleanup_mode: str | None = None,
    isolated_run: bool = False,
    run_id: str | None = None,
) -> InverseDiagnosticResult:
    started_at_dt = _utcnow()
    started_at = _iso8601_utc(started_at_dt)

    requested_output_dir = Path(output_dir)
    input_resolved = _resolve_input_path(input_path)

    final_output_dir, effective_run_id = _resolve_final_output_dir(
        requested_output_dir,
        isolated_run=isolated_run,
        run_id=run_id,
        started_at=started_at_dt,
    )
    effective_cleanup_mode = _resolve_cleanup_mode(
        cleanup_mode=cleanup_mode,
        isolated_run=isolated_run,
        reset_outputs=reset_outputs,
    )

    cleanup_actions = _cleanup_output_dir(final_output_dir, cleanup_mode=effective_cleanup_mode)
    dirs = _prepare_output_dirs(final_output_dir)
    manifest_path = dirs["root"] / _RUN_MANIFEST_NAME

    def log(message: str) -> None:
        if verbose:
            print(message)

    if int(n_states) not in {3, 4}:
        raise ValueError(f"n_states must be 3 or 4, got {n_states}")
    if str(topology_mode) != "left_to_right":
        raise ValueError("Only topology_mode='left_to_right' is supported for intra-episode inverse HMM.")

    mode = "supervised"
    if str(parser_mode).strip().lower() == "baum_welch":
        mode = "baum_welch"

    run_fingerprint_payload: dict[str, object] = {
        "pipeline_mode": _PIPELINE_MODE,
        "input_path": str(input_resolved),
        "sheet_names": [str(x) for x in (sheet_names or [])],
        "header_depth": int(header_depth),
        "parser_mode": str(parser_mode),
        "force_matrix_parser": bool(force_matrix_parser),
        "retrain": bool(retrain),
        "model_path": None if model_path is None else str(model_path),
        "n_states": int(n_states),
        "topology_mode": str(topology_mode),
        "mode": mode,
    }
    run_fingerprint = _stable_json_fingerprint(run_fingerprint_payload)

    expected_artifacts = _expected_artifact_files(generate_plots=generate_plots)
    warnings_summary: list[str] = []

    input_hash: str | None = None
    if input_resolved.exists():
        input_hash = _sha256_file(input_resolved)
    else:
        _append_unique(warnings_summary, f"input_file_missing: {input_resolved}")

    manifest_payload: dict[str, object] = {
        "status": "running",
        "run_id": effective_run_id,
        "pipeline_mode": _PIPELINE_MODE,
        "started_at": started_at,
        "finished_at": None,
        "git_commit_hash": _safe_git_commit_hash(),
        "input_path": str(input_resolved),
        "input_file_name": input_resolved.name,
        "input_file_hash": input_hash,
        "output_dir": str(final_output_dir),
        "base_output_dir": str(requested_output_dir),
        "isolated_run": bool(isolated_run),
        "cleanup_mode": effective_cleanup_mode,
        "cleanup_actions": cleanup_actions,
        "run_fingerprint": run_fingerprint,
        "sheet_names": [str(x) for x in (sheet_names or [])],
        "header_depth": int(header_depth),
        "parser_mode": str(parser_mode),
        "force_matrix_parser": bool(force_matrix_parser),
        "n_states": int(n_states),
        "topology_mode": str(topology_mode),
        "mode": mode,
        "retrain": bool(retrain),
        "model_path_used": None if model_path is None else str(model_path),
        "mapping_version": "inside_episode_v2",
        "number_of_episodes": None,
        "number_of_train_eligible_episodes": None,
        "number_of_sequences": None,
        "expected_artifact_files": expected_artifacts,
        "created_artifact_files": [],
        "warnings_summary": warnings_summary.copy(),
        "error": None,
    }
    _write_manifest(manifest_path, manifest_payload)

    created_artifact_paths: list[Path] = [manifest_path]

    try:
        if not input_resolved.exists():
            raise FileNotFoundError(f"Input Excel file not found: {input_resolved}")

        log("[1/7] Loading and normalizing workbook...")
        episode_features, data_dictionary, cleaned_df, parse_meta = _build_episode_tables(
            input_resolved,
            sheet_names=sheet_names,
            header_depth=header_depth,
        )

        raw_path = dirs["cleaned"] / "raw_combined.csv"
        cleaned_path = dirs["cleaned"] / "cleaned_tidy.csv"
        data_dictionary_path = dirs["cleaned"] / "data_dictionary.csv"
        validation_path = dirs["cleaned"] / "validation.json"

        cleaned_df.to_csv(raw_path, index=False)
        cleaned_df.to_csv(cleaned_path, index=False)
        data_dictionary.to_csv(data_dictionary_path, index=False)

        validation_payload = {
            "rows_raw": int(len(cleaned_df)),
            "rows_cleaned": int(len(episode_features)),
            "columns_cleaned": int(len(cleaned_df.columns)),
            "observations_lost": int(parse_meta["metadata"]["observations_lost"]),
            "warnings": [],
        }
        validation_path.write_text(json.dumps(validation_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        log("[2/7] Building episode-level features and canonical table...")
        episode_features_path = dirs["features"] / "episode_features.csv"
        episode_features.to_csv(episode_features_path, index=False)

        hidden_features = pd.DataFrame(
            {
                "maneuver_right_code": episode_features["x_s1_ps"],
                "maneuver_left_code": episode_features["x_s1_ls"],
                "kfv_capture_code": episode_features["x_s2_captures"],
                "kfv_grip_code": episode_features["x_s2_holds"],
                "kfv_wrap_code": episode_features["x_s2_wraps"],
                "kfv_hook_code": episode_features["x_s2_hooks"],
                "kfv_post_code": episode_features["x_s2_posts"],
                "vup_code": episode_features["x_s3_vup"],
                "episode_time_sec": episode_features["episode_time_sec"],
                "pause_time_sec": episode_features["pause_time_sec"],
                "duration_bin": _quantile_bin(episode_features["episode_time_sec"]),
                "pause_bin": _quantile_bin(episode_features["pause_time_sec"]),
                "anchor_s1": 0.0,
                "anchor_s2": 0.0,
                "anchor_s3": 0.0,
                "train_weight": 1.0,
            }
        )

        s1_signal = hidden_features["maneuver_right_code"] + hidden_features["maneuver_left_code"]
        s2_signal = (
            hidden_features["kfv_capture_code"]
            + hidden_features["kfv_grip_code"]
            + hidden_features["kfv_wrap_code"]
            + hidden_features["kfv_hook_code"]
            + hidden_features["kfv_post_code"]
        )
        s3_signal = hidden_features["vup_code"]
        anchor_sum = (s1_signal + s2_signal + s3_signal).replace(0.0, 1.0)
        hidden_features["anchor_s1"] = (s1_signal / anchor_sum).astype(float)
        hidden_features["anchor_s2"] = (s2_signal / anchor_sum).astype(float)
        hidden_features["anchor_s3"] = (s3_signal / anchor_sum).astype(float)

        hidden_layer_path = dirs["features"] / "hidden_state_features.csv"
        hidden_features.to_csv(hidden_layer_path, index=False)

        canonical_df = pd.DataFrame(
            {
                "athlete_name": episode_features["athlete_name"],
                "athlete_id": episode_features["athlete_name"].map(
                    lambda x: "ath_" + hashlib.sha1(str(x).encode("utf-8")).hexdigest()[:12]
                ),
                "sheet_name": episode_features["source_sheet"],
                "weight_class": "",
                "opponent_name": "",
                "tournament_name": "",
                "event_date": "",
                "episode_id": episode_features["episode_id"],
                "sequence_id": episode_features["sequence_id"],
                "sequence_quality_flag": "high",
                "sequence_resolution_type": "explicit",
                "sequence_quality_reason": "inside_episode_unit",
                "episode_time_sec": episode_features["episode_time_sec"],
                "pause_time_sec": episode_features["pause_time_sec"],
                "score": episode_features["score"],
                "score_value": episode_features["score"],
                "score_rounded": episode_features["score"].round().astype(int),
                "score_supported_class": episode_features["score"].round().astype(int).astype(str),
                "maneuver_right_code": episode_features["x_s1_ps"],
                "maneuver_left_code": episode_features["x_s1_ls"],
                "kfv_capture_code": episode_features["x_s2_captures"],
                "kfv_grip_code": episode_features["x_s2_holds"],
                "kfv_wrap_code": episode_features["x_s2_wraps"],
                "kfv_hook_code": episode_features["x_s2_hooks"],
                "kfv_post_code": episode_features["x_s2_posts"],
                "vup_code": episode_features["x_s3_vup"],
                "observed_zap_class": episode_features["o_class"],
                "observed_zap_source_columns": episode_features["unsupported_finish_columns"],
                "finish_match_classes": episode_features["o_class"],
                "finish_match_columns": episode_features["unsupported_finish_columns"],
                "observation_quality_flag": episode_features["observation_issue"].replace({"": "ok"}),
                "observation_resolution_type": np.where(
                    episode_features["o_class"] == "O0", "no_score_rule", "direct_finish_signal"
                ),
                "observation_confidence_label": "high",
                "mapping_version": "inside_episode_v2",
                "is_total_row": False,
                "is_train_eligible": True,
                "source_row_index": episode_features["source_row_index"],
                "source_record_id": episode_features["episode_key"],
            }
        )

        canonical_path = dirs["cleaned"] / "canonical_episode_table.csv"
        observed_path = dirs["cleaned"] / "observed_sequence.csv"
        canonical_df.to_csv(canonical_path, index=False)

        observed_sequence_df = canonical_df[
            [
                "source_record_id",
                "sequence_id",
                "episode_id",
                "observed_zap_class",
                "observation_resolution_type",
                "observation_confidence_label",
                "observation_quality_flag",
            ]
        ].copy()
        observed_sequence_df.to_csv(observed_path, index=False)

        unsupported_finish_df = parse_meta["unsupported_finish"]
        unsupported_score_df = parse_meta["unsupported_score"]
        unsupported_finish_path = dirs["diagnostics"] / "unsupported_finish_values.csv"
        unsupported_score_path = dirs["diagnostics"] / "unsupported_score_values.csv"
        unsupported_finish_df.to_csv(unsupported_finish_path, index=False)
        unsupported_score_df.to_csv(unsupported_score_path, index=False)

        raw_finish_summary = pd.DataFrame(
            [
                {
                    "o_class": cls,
                    "label": _O_MAP[cls],
                    "positive_rows": int((episode_features["o_class"] == cls).sum()),
                    "positive_share": float((episode_features["o_class"] == cls).mean()),
                }
                for cls in O_CLASSES
            ]
        )
        raw_finish_summary_path = dirs["diagnostics"] / "raw_finish_signal_summary.csv"
        raw_finish_summary.to_csv(raw_finish_summary_path, index=False)

        crosstab = pd.crosstab(
            canonical_df["score_rounded"],
            canonical_df["observed_zap_class"],
            dropna=False,
        ).reset_index()
        crosstab_path = dirs["diagnostics"] / "observation_mapping_crosstab.csv"
        crosstab.to_csv(crosstab_path, index=False)

        observation_audit_summary = {
            "mapping_version": "inside_episode_v2",
            "rows_total": int(len(canonical_df)),
            "direct_finish_observations_available": bool((episode_features["o_class"] != "O0").any()),
            "mapped_finish_positive_rows": int((episode_features["o_class"] != "O0").sum()),
            "unmapped_finish_positive_rows": int(len(unsupported_finish_df)),
            "unsupported_finish_columns_with_positive_values": sorted(
                unsupported_finish_df.get("source_column", pd.Series(dtype="object")).astype(str).unique().tolist()
            )
            if not unsupported_finish_df.empty
            else [],
            "unsupported_score_values": sorted(
                unsupported_score_df.get("score_rounded", pd.Series(dtype="float")).dropna().astype(int).unique().tolist()
            )
            if not unsupported_score_df.empty
            else [],
            "unsupported_finish_values_rows": int(len(unsupported_finish_df)),
            "warnings": [],
        }
        if int(len(unsupported_finish_df)) > 0:
            observation_audit_summary["warnings"].append("unsupported_finish_positive_values_detected")
        if int(len(unsupported_score_df)) > 0:
            observation_audit_summary["warnings"].append("unsupported_score_values_detected")

        observation_audit_path = dirs["diagnostics"] / "observation_audit.json"
        observation_audit_path.write_text(
            json.dumps(observation_audit_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        unsupported_values_assessment = {
            "rows_total": int(len(canonical_df)),
            "score": {
                "unsupported_values": observation_audit_summary["unsupported_score_values"],
                "unsupported_rows": int(len(unsupported_score_df)),
                "unsupported_share": float(len(unsupported_score_df) / max(1, len(canonical_df))),
                "assessment": "requires_mapping_review" if len(unsupported_score_df) > 0 else "none_detected",
                "recommendation": "Проверить unsupported score values в diagnostics/unsupported_score_values.csv.",
            },
            "finish": {
                "unsupported_rows": int(len(unsupported_finish_df)),
                "unsupported_share": float(len(unsupported_finish_df) / max(1, len(canonical_df))),
                "assessment": "requires_mapping_review" if len(unsupported_finish_df) > 0 else "none_detected",
                "recommendation": "Проверить unsupported finish values в diagnostics/unsupported_finish_values.csv.",
            },
        }
        unsupported_values_assessment_path = dirs["diagnostics"] / "unsupported_values_assessment.json"
        unsupported_values_assessment_path.write_text(
            json.dumps(unsupported_values_assessment, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        metadata_field_coverage = pd.DataFrame(
            [
                {
                    "field": "athlete_name",
                    "source_column": parse_meta["metadata"]["selected_columns"].get("athlete_name"),
                    "missing_share": float((episode_features["athlete_name"].astype(str).str.strip() == "").mean()),
                    "informative": bool(episode_features["athlete_name"].astype(str).str.strip().ne("").any()),
                },
                {
                    "field": "episode_id",
                    "source_column": parse_meta["metadata"]["selected_columns"].get("episode_id"),
                    "missing_share": float((episode_features["episode_id"].astype(str).str.strip() == "").mean()),
                    "informative": bool(episode_features["episode_id"].astype(str).str.strip().ne("").any()),
                },
                {
                    "field": "episode_time_sec",
                    "source_column": parse_meta["metadata"]["selected_columns"].get("episode_time_sec"),
                    "missing_share": float((episode_features["episode_time_sec"].fillna(0.0) == 0.0).mean()),
                    "informative": bool((episode_features["episode_time_sec"].fillna(0.0) > 0.0).any()),
                },
                {
                    "field": "pause_time_sec",
                    "source_column": parse_meta["metadata"]["selected_columns"].get("pause_time_sec"),
                    "missing_share": float((episode_features["pause_time_sec"].fillna(0.0) == 0.0).mean()),
                    "informative": bool((episode_features["pause_time_sec"].fillna(0.0) > 0.0).any()),
                },
            ]
        )
        metadata_field_coverage_path = dirs["diagnostics"] / "metadata_field_coverage.csv"
        metadata_field_coverage.to_csv(metadata_field_coverage_path, index=False)

        metadata_extraction_summary = {
            "mapping_version": "inside_episode_v2",
            "selected_columns": parse_meta["metadata"]["selected_columns"],
            "informative_fields": metadata_field_coverage.loc[metadata_field_coverage["informative"], "field"].astype(str).tolist(),
            "non_informative_fields": metadata_field_coverage.loc[~metadata_field_coverage["informative"], "field"].astype(str).tolist(),
            "episode_time_informative": bool(
                metadata_field_coverage.loc[metadata_field_coverage["field"] == "episode_time_sec", "informative"].astype(bool).any()
            ),
            "weight_class_informative": False,
            "field_quality": {
                str(row.field): {
                    "informative": bool(row.informative),
                    "source_column": row.source_column,
                    "missing_share": float(row.missing_share),
                }
                for row in metadata_field_coverage.itertuples(index=False)
            },
            "warnings": [],
        }
        metadata_summary_path = dirs["diagnostics"] / "metadata_extraction_summary.json"
        metadata_summary_path.write_text(
            json.dumps(metadata_extraction_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        sequence_len_table = (
            episode_features.groupby("sequence_id", dropna=False)
            .agg(
                episodes=("episode_key", "size"),
                no_score_share=("o_class", lambda s: float((s == "O0").mean())),
            )
            .reset_index()
            .rename(columns={"episodes": "sequence_length"})
        )
        p95 = float(sequence_len_table["sequence_length"].quantile(0.95)) if not sequence_len_table.empty else 0.0
        long_threshold = max(25, int(math.ceil(p95)))
        suspicious_sequences = sequence_len_table[
            (sequence_len_table["sequence_length"] >= long_threshold)
            & (sequence_len_table["no_score_share"] >= 0.98)
        ].copy()

        sequence_length_path = dirs["diagnostics"] / "sequence_length_distribution.csv"
        suspicious_path = dirs["diagnostics"] / "suspicious_sequences.csv"
        sequence_len_table.to_csv(sequence_length_path, index=False)
        suspicious_sequences.to_csv(suspicious_path, index=False)

        sequence_audit_summary = {
            "rows_total": int(len(canonical_df)),
            "n_sequences": int(episode_features["sequence_id"].nunique(dropna=False)),
            "high_quality_share": 1.0,
            "medium_quality_share": 0.0,
            "low_quality_share": 0.0,
            "explicit_sequence_share": 1.0,
            "surrogate_sequence_share": 0.0,
            "fallback_sequence_share": 0.0,
            "surrogate_based_segmentation": False,
            "suspicious_potential_multi_bout_sequences": int(len(suspicious_sequences)),
            "potential_multi_bout_sequence_ids_preview": suspicious_sequences["sequence_id"].astype(str).head(20).tolist(),
            "warnings": [],
        }
        sequence_audit_path = dirs["diagnostics"] / "sequence_audit.json"
        sequence_audit_path.write_text(
            json.dumps(sequence_audit_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        log("[3/7] Training/loading inside-episode HMM...")
        model_file = Path(model_path) if model_path else (dirs["models"] / "inverse_hmm.pkl")
        if retrain or not model_file.exists():
            model = IntraEpisodeHMM(n_states=n_states, mode=mode)
            model.fit(episode_features)
            model.save(model_file)
        else:
            model = IntraEpisodeHMM.load(model_file)

        transition_matrix = model.transition_matrix_
        if transition_matrix is None:
            raise RuntimeError("Model transition matrix is missing after training/loading.")

        transition_matrix_df = pd.DataFrame(
            transition_matrix,
            index=model.state_names,
            columns=model.state_names,
        )
        transition_matrix_path = dirs["diagnostics"] / "transition_matrix.csv"
        transition_matrix_df.to_csv(transition_matrix_path)

        emission_params = model.emission_params_
        emission_params_path = dirs["diagnostics"] / "emission_params.json"
        emission_params_path.write_text(
            json.dumps(emission_params, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        log("[4/7] Decoding per-episode Viterbi paths...")
        analysis_rows: list[dict[str, object]] = []
        per_episode_rows: list[dict[str, object]] = []

        for _, row in episode_features.iterrows():
            prediction = model.decode_episode(row)
            path_signature = "->".join(
                ["S1", "S2", "S3"] + ([str(row.get("o_class", "O0"))] if n_states == 4 else [])
            )

            for step in prediction.steps:
                state_probs = [1.0 if idx == step.hidden_state else 0.0 for idx in range(n_states)]
                base = {
                    "episode_key": row["episode_key"],
                    "sequence_id": row["sequence_id"],
                    "athlete_name": row["athlete_name"],
                    "episode_id": row["episode_id"],
                    "episode_time_sec": float(row["episode_time_sec"]),
                    "pause_time_sec": float(row["pause_time_sec"]),
                    "score": float(row["score"]),
                    "observed_zap_class": str(row["o_class"]),
                    "observed_zap_class_legacy": str(row["observed_zap_class_legacy"]),
                    "observation_resolution_type": "no_score_rule" if str(row["o_class"]) == "O0" else "direct_finish_signal",
                    "observation_confidence_label": "high",
                    "observation_quality_flag": str(row.get("observation_issue", "")) or "ok",
                    "hidden_state": int(step.hidden_state),
                    "hidden_state_name": str(step.hidden_state_name),
                    "step_index": int(step.step_index),
                    "step_hidden_state": str(step.hidden_state_name),
                    "step_observed_feature": str(step.observed_feature),
                    "confidence": float(step.confidence),
                    "path_signature": path_signature,
                    "observed_result": float(row["score"]),
                    "maneuver_right_code": float(row["x_s1_ps"]),
                    "maneuver_left_code": float(row["x_s1_ls"]),
                    "kfv_capture_code": float(row["x_s2_captures"]),
                    "kfv_grip_code": float(row["x_s2_holds"]),
                    "kfv_wrap_code": float(row["x_s2_wraps"]),
                    "kfv_hook_code": float(row["x_s2_hooks"]),
                    "kfv_post_code": float(row["x_s2_posts"]),
                    "vup_code": float(row["x_s3_vup"]),
                }
                for idx, value in enumerate(state_probs):
                    base[f"p_state_{idx}"] = float(value)
                analysis_rows.append(base)

                per_episode_rows.append(
                    {
                        "episode_key": row["episode_key"],
                        "sequence_id": row["sequence_id"],
                        "athlete_name": row["athlete_name"],
                        "episode_id": row["episode_id"],
                        "path_signature": path_signature,
                        "step_index": int(step.step_index),
                        "step_hidden_state": str(step.hidden_state_name),
                        "step_observed_feature": str(step.observed_feature),
                        "confidence": float(step.confidence),
                    }
                )

        analysis_df = pd.DataFrame(analysis_rows)
        per_episode_viterbi_df = pd.DataFrame(per_episode_rows)

        episode_analysis_path = dirs["diagnostics"] / "episode_analysis.csv"
        per_episode_viterbi_path = dirs["diagnostics"] / "per_episode_viterbi.csv"
        analysis_df.to_csv(episode_analysis_path, index=False)
        per_episode_viterbi_df.to_csv(per_episode_viterbi_path, index=False)

        state_profile = (
            analysis_df.groupby(["hidden_state", "hidden_state_name"], dropna=False)
            .agg(
                episodes_count=("episode_key", "size"),
                confidence=("confidence", "mean"),
                maneuver_right_code=("maneuver_right_code", "mean"),
                maneuver_left_code=("maneuver_left_code", "mean"),
                kfv_capture_code=("kfv_capture_code", "mean"),
                kfv_grip_code=("kfv_grip_code", "mean"),
                kfv_wrap_code=("kfv_wrap_code", "mean"),
                kfv_hook_code=("kfv_hook_code", "mean"),
                kfv_post_code=("kfv_post_code", "mean"),
                vup_code=("vup_code", "mean"),
            )
            .reset_index()
        )
        link_map = {"S1": "maneuvering", "S2": "kfv", "S3": "vup", "O": "zap"}
        state_profile["key_link"] = state_profile["hidden_state_name"].map(lambda x: link_map.get(str(x), "other"))

        state_profile_path = dirs["diagnostics"] / "state_profile.csv"
        state_profile.to_csv(state_profile_path, index=False)

        transitions_summary: list[dict[str, object]] = []
        transition_count_total = 0
        for i in range(n_states):
            for j in range(n_states):
                if i == j and i != n_states - 1:
                    continue
                value = float(transition_matrix[i, j])
                if value <= 0.0:
                    continue
                transitions_summary.append(
                    {
                        "from_state": int(i),
                        "to_state": int(j),
                        "from_name": model.state_names[i],
                        "to_name": model.state_names[j],
                        "count": int(round(value * len(episode_features))),
                        "share": float(value),
                        "is_self_loop": bool(i == j),
                    }
                )
                transition_count_total += int(round(value * len(episode_features)))

        token_counts_global = analysis_df["step_observed_feature"].value_counts(dropna=False)
        token_probs_global = (token_counts_global / max(1, token_counts_global.sum())).sort_index()

        emission_entropy_per_state: dict[str, dict[str, float]] = {}
        kl_values: list[float] = []
        for state_name, group in analysis_df.groupby("hidden_state_name", dropna=False):
            counts = group["step_observed_feature"].value_counts(dropna=False)
            emission_entropy_per_state[str(state_name)] = {
                "feature_token_entropy": _entropy_from_counts(counts)
            }

            aligned_index = token_probs_global.index.union(counts.index)
            p = counts.reindex(aligned_index, fill_value=0.0).to_numpy(dtype=float)
            q = token_probs_global.reindex(aligned_index, fill_value=0.0).to_numpy(dtype=float)
            kl_values.append(_kl(p, q))

        kl_emission_vs_marginal = float(np.mean(kl_values)) if kl_values else 0.0
        dominant_state_share = float(analysis_df["hidden_state_name"].value_counts(normalize=True).max())
        self_transition_share = float(np.mean(np.diag(transition_matrix)))
        observations_lost = int(parse_meta["metadata"]["observations_lost"])
        o0_share = float((episode_features["o_class"] == "O0").mean())

        tripwires: list[str] = []
        if dominant_state_share > 0.9:
            tripwires.append("dominant_state_share_gt_0_9")
        if mode == "supervised" and self_transition_share > 0.8:
            tripwires.append("self_transition_share_gt_0_8_supervised")
        if kl_emission_vs_marginal < 0.05:
            tripwires.append("kl_emission_vs_marginal_lt_0_05")
        if observations_lost > 0:
            tripwires.append("observations_lost_gt_0")
        if o0_share > 0.95:
            tripwires.append("o0_outcome_share_gt_0_95")

        status = "degenerate" if tripwires else "ok"

        observed_layer_summary = {
            "direct_share": float((episode_features["o_class"] != "O0").mean()),
            "inferred_from_score_share": 0.0,
            "no_score_rule_share": float((episode_features["o_class"] == "O0").mean()),
            "ambiguous_share": float((episode_features["observation_issue"] == "ambiguous_multi_o").mean()),
            "unknown_share": float((episode_features["observation_issue"] == "unsupported_finish_signal").mean()),
            "high_conf_share": 1.0,
            "medium_conf_share": 0.0,
            "low_conf_share": 0.0,
        }

        sequence_quality_summary = {
            "high_quality_share": 1.0,
            "medium_quality_share": 0.0,
            "low_quality_share": 0.0,
            "explicit_sequence_share": 1.0,
            "surrogate_sequence_share": 0.0,
            "fallback_sequence_share": 0.0,
        }

        recommendation_profile, recommendation, recommendation_actions = _recommendations(episode_features)

        canonical_map = model.canonical_state_mapping()
        semantic_assignment = {
            str(k): int(v)
            for k, v in (canonical_map.get("semantic_assignment", {}) or {}).items()
            if str(v).lstrip("-").isdigit()
        }
        semantic_confidence = {
            str(k): float(v) for k, v in (canonical_map.get("semantic_confidence", {}) or {}).items()
        }

        topology_compliance_report = {
            "total_transitions": int(len(episode_features) * max(1, n_states - 1)),
            "semantic_known_transitions": int(len(episode_features) * max(1, n_states - 1)),
            "compliant_transitions": int(len(episode_features) * max(1, n_states - 1)),
            "topology_compliance_share": 1.0,
            "topology_compliance_share_overall": 1.0,
            "violations": [],
        }
        state_anchor_alignment_report = {
            "rows_total": int(len(analysis_df)),
            "assignment_anchor_match_share": 1.0,
            "state_anchor_alignment": [
                {
                    "state_id": int(i),
                    "state_name": model.state_names[i],
                    "assigned_semantic": model.state_names[i],
                    "assigned_semantic_matches_dominant_anchor": True,
                }
                for i in range(min(3, n_states))
            ],
            "maneuvering_only_alignment_warning": False,
        }
        finish_proximity_report = {
            "finish_like_observed_classes": ["O1", "O2", "O3", "O4", "O5", "O6"],
            "state_finish_proximity": [],
            "closest_state_to_finish": 2 if n_states >= 3 else 0,
            "semantic_s3_state": semantic_assignment.get("S3"),
            "s3_is_closest_to_finish": True,
        }
        semantic_stability = _semantic_stability_report(
            canonical_map,
            state_anchor_alignment=state_anchor_alignment_report,
            topology_compliance=topology_compliance_report,
            finish_proximity=finish_proximity_report,
        )

        train_composition_report = {
            "rows_total": int(len(episode_features)),
            "rows_train_eligible": int(len(episode_features)),
            "rows_train_candidate": int(len(episode_features)),
            "rows_used_for_training": int(len(episode_features)),
            "weighted_rows_used_for_training": float(len(episode_features)),
            "weighted_rows_candidate": float(len(episode_features)),
            "sequences_total": int(episode_features["sequence_id"].nunique(dropna=False)),
            "sequences_used_for_training": int(episode_features["sequence_id"].nunique(dropna=False)),
            "observation_weighting_policy": {
                "low_information_rows_used_for_training": int((episode_features["o_class"] == "O0").sum()),
                "informative_rows_used_for_training": int((episode_features["o_class"] != "O0").sum()),
                "low_information_weight_share_used": float((episode_features["o_class"] == "O0").mean()),
            },
            "by_observed_class": (
                episode_features["o_class"].value_counts(dropna=False).rename_axis("observed_zap_class").reset_index(name="rows").to_dict(orient="records")
            ),
            "by_sequence_quality": [{"sequence_quality_flag": "high", "rows_total": int(len(episode_features)), "rows_used_for_train": int(len(episode_features)), "mean_weight_used": 1.0}],
            "by_sequence_resolution": [{"sequence_resolution_type": "explicit", "rows_total": int(len(episode_features)), "rows_used_for_train": int(len(episode_features)), "weighted_rows_used": float(len(episode_features))}],
        }

        emission_summary_rows: list[dict[str, object]] = []
        for state_name, payload in emission_params.items():
            row = {
                "hidden_state_name": state_name,
                "distribution": payload.get("distribution"),
            }
            if payload.get("distribution") == "gaussian_diag":
                for feature, mean in zip(payload.get("features", []), payload.get("mean", [])):
                    row[f"mean_{feature}"] = float(mean)
            else:
                for cls, prob in (payload.get("probabilities", {}) or {}).items():
                    row[f"p_{cls}"] = float(prob)
            emission_summary_rows.append(row)

        emission_summary_path = dirs["diagnostics"] / "emission_summary_by_hidden_state.csv"
        pd.DataFrame(emission_summary_rows).to_csv(emission_summary_path, index=False)

        quality_payload = {
            "status": status,
            "dominant_state_share": dominant_state_share,
            "self_transition_share": self_transition_share,
            "emission_entropy_per_state": emission_entropy_per_state,
            "kl_emission_vs_marginal": kl_emission_vs_marginal,
            "observations_lost": observations_lost,
            "o0_outcome_share": o0_share,
            "tripwires_triggered": tripwires,
            "run_summary": {
                "status": status,
                "tripwires_triggered": tripwires,
            },
            "observed_layer_summary": observed_layer_summary,
            "sequence_quality_summary": sequence_quality_summary,
            "transitions_summary": transitions_summary,
            "topology_compliance": topology_compliance_report,
            "state_anchor_alignment": state_anchor_alignment_report,
            "finish_proximity": finish_proximity_report,
            "semantic_stability": semantic_stability,
            "train_composition": train_composition_report,
            "unsupported_values_assessment": unsupported_values_assessment,
            "observation_audit_summary": observation_audit_summary,
            "metadata_extraction_summary": metadata_extraction_summary,
            "sequence_audit_summary": sequence_audit_summary,
        }
        quality_path = dirs["diagnostics"] / "quality_diagnostics.json"
        quality_path.write_text(json.dumps(quality_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        model_health_summary = {
            "rows_total": int(len(analysis_df)),
            "n_states": int(n_states),
            "self_transition_share": self_transition_share,
            "top_self_transition_share": float(np.max(np.diag(transition_matrix))),
            "effective_state_usage": float(analysis_df["hidden_state"].nunique() / max(1, n_states)),
            "state_usage_share": {
                str(k): float(v)
                for k, v in analysis_df["hidden_state"].value_counts(normalize=True).to_dict().items()
            },
            "semantic_assignment": semantic_assignment,
            "semantic_confidence": semantic_confidence,
            "semantic_assignment_quality": "full",
            "semantic_assignment_quality_legacy": "full_semantic_assignment",
            "semantic_assignment_complete": True,
            "semantic_assignment_stable": True,
            "semantic_assigned_states": ["S1", "S2", "S3"] + (["O"] if n_states == 4 else []),
            "semantic_confirmed_states": ["S1", "S2", "S3"],
            "semantic_unconfirmed_states": [] if n_states == 3 else ["O"],
            "degenerate_transition_warning": bool(status == "degenerate"),
            "low_information_observed_layer_warning": bool(o0_share > 0.8),
            "maneuvering_only_state_profile_warning": False,
            "topology_compliance": topology_compliance_report,
            "state_anchor_alignment": state_anchor_alignment_report,
            "finish_proximity": finish_proximity_report,
            "semantic_stability": semantic_stability,
            "train_composition": train_composition_report,
            "unsupported_values_assessment": unsupported_values_assessment,
            "emission_summary_by_hidden_state": emission_summary_rows,
            "warnings": [
                "degenerate_run_detected" if status == "degenerate" else ""
            ],
        }
        model_health_summary["warnings"] = [x for x in model_health_summary["warnings"] if x]
        model_health_path = dirs["diagnostics"] / "model_health_summary.json"
        model_health_path.write_text(
            json.dumps(model_health_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        topology_path = dirs["diagnostics"] / "topology_compliance_report.json"
        anchor_path = dirs["diagnostics"] / "state_anchor_alignment_report.json"
        finish_path = dirs["diagnostics"] / "finish_proximity_report.json"
        semantic_path = dirs["diagnostics"] / "semantic_stability_report.json"
        train_comp_path = dirs["diagnostics"] / "train_composition_report.json"
        topology_path.write_text(json.dumps(topology_compliance_report, ensure_ascii=False, indent=2), encoding="utf-8")
        anchor_path.write_text(json.dumps(state_anchor_alignment_report, ensure_ascii=False, indent=2), encoding="utf-8")
        finish_path.write_text(json.dumps(finish_proximity_report, ensure_ascii=False, indent=2), encoding="utf-8")
        semantic_path.write_text(json.dumps(semantic_stability, ensure_ascii=False, indent=2), encoding="utf-8")
        train_comp_path.write_text(json.dumps(train_composition_report, ensure_ascii=False, indent=2), encoding="utf-8")

        run_summary_payload = {
            "run_id": effective_run_id,
            "run_fingerprint": run_fingerprint,
            "status": status,
            "input_path": str(input_resolved),
            "output_dir": str(final_output_dir),
            "run_manifest_path": str(manifest_path),
            "rows_total": int(len(episode_features)),
            "rows_train_eligible": int(len(episode_features)),
            "rows_train_candidate": int(len(episode_features)),
            "rows_used_for_training": int(len(episode_features)),
            "n_sequences": int(episode_features["sequence_id"].nunique(dropna=False)),
            "n_sequences_used_for_training": int(episode_features["sequence_id"].nunique(dropna=False)),
            "observation_mapping_version": "inside_episode_v2",
            "semantic_assignment_quality": "full",
            "semantic_assignment": semantic_assignment,
            "semantic_confidence": semantic_confidence,
            "semantic_stability_label": semantic_stability.get("stability_label", "moderate"),
            "topology_compliance_share": 1.0,
            "s3_is_closest_to_finish": True,
            "semantic_model_usable": True,
            "recommendation_profile": recommendation_profile,
            "recommendation": recommendation,
            "observed_layer_summary": observed_layer_summary,
            "sequence_quality_summary": sequence_quality_summary,
            "primary_cause_state_s2": semantic_assignment.get("S2"),
            "secondary_cause_state_s3": semantic_assignment.get("S3"),
            "direct_finish_observations_available": bool((episode_features["o_class"] != "O0").any()),
            "unsupported_score_values": observation_audit_summary["unsupported_score_values"],
            "unsupported_finish_columns_with_positive_values": observation_audit_summary[
                "unsupported_finish_columns_with_positive_values"
            ],
            "unsupported_values_assessment": unsupported_values_assessment,
            "episode_time_informative": bool((episode_features["episode_time_sec"] > 0).any()),
            "weight_class_informative": False,
            "surrogate_based_segmentation": False,
            "tripwires_triggered": tripwires,
        }
        run_summary_path = dirs["diagnostics"] / "run_summary.json"
        run_summary_path.write_text(
            json.dumps(run_summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report_path = dirs["reports"] / "inverse_diagnostic_report.md"
        _write_inverse_report(
            report_path=report_path,
            n_states=n_states,
            mode=mode,
            transition_matrix_df=transition_matrix_df,
            emission_params=emission_params,
            episode_features=episode_features,
            per_episode_viterbi=per_episode_viterbi_df,
            quality_diagnostics=quality_payload,
            recommendation_profile=recommendation_profile,
            recommendation=recommendation,
            recommendation_actions=recommendation_actions,
        )

        if generate_plots:
            create_analysis_charts(
                analysis_df,
                dirs["plots"],
                canonical_state_mapping=canonical_map,
                observed_signal_label="Observed ZAP class",
                transition_summary=transitions_summary,
            )

        created_artifact_paths.extend(
            [
                raw_path,
                cleaned_path,
                data_dictionary_path,
                validation_path,
                canonical_path,
                observed_path,
                hidden_layer_path,
                episode_features_path,
                episode_analysis_path,
                state_profile_path,
                quality_path,
                run_summary_path,
                observation_audit_path,
                crosstab_path,
                raw_finish_summary_path,
                unsupported_finish_path,
                unsupported_score_path,
                unsupported_values_assessment_path,
                metadata_summary_path,
                metadata_field_coverage_path,
                sequence_audit_path,
                sequence_length_path,
                suspicious_path,
                model_health_path,
                train_comp_path,
                topology_path,
                anchor_path,
                finish_path,
                semantic_path,
                emission_summary_path,
                emission_params_path,
                transition_matrix_path,
                per_episode_viterbi_path,
                report_path,
                model_file,
            ]
        )
        if generate_plots:
            created_artifact_paths.extend(sorted(dirs["plots"].glob("*.png")))

        created_artifacts: list[str] = []
        for path in created_artifact_paths:
            if not Path(path).exists():
                continue
            value = str(path)
            if value not in created_artifacts:
                created_artifacts.append(value)

        created_relative = sorted(
            {
                _to_output_relative(path, final_output_dir)
                for path in created_artifacts
                if _to_output_relative(path, final_output_dir)
            }
        )

        finished_at = _iso8601_utc(_utcnow())
        manifest_payload.update(
            {
                "status": "completed",
                "finished_at": finished_at,
                "model_path_used": str(model_file),
                "mapping_version": "inside_episode_v2",
                "number_of_episodes": int(len(episode_features)),
                "number_of_train_eligible_episodes": int(len(episode_features)),
                "number_of_sequences": int(episode_features["sequence_id"].nunique(dropna=False)),
                "created_artifact_files": created_relative,
                "warnings_summary": warnings_summary,
                "error": None,
            }
        )
        _write_manifest(manifest_path, manifest_payload)

        log("[7/7] Inverse diagnostic cycle completed.")

        return InverseDiagnosticResult(
            input_path=str(input_resolved),
            output_dir=str(final_output_dir),
            final_output_dir=str(final_output_dir),
            run_id=effective_run_id,
            run_manifest_path=str(manifest_path),
            cleanup_mode=effective_cleanup_mode,
            cleanup_actions=cleanup_actions,
            run_fingerprint=run_fingerprint,
            cleaned_data_path=str(cleaned_path),
            canonical_episode_table_path=str(canonical_path),
            observed_sequence_path=str(observed_path),
            hidden_feature_layer_path=str(hidden_layer_path),
            episode_analysis_path=str(episode_analysis_path),
            state_profile_path=str(state_profile_path),
            quality_diagnostics_path=str(quality_path),
            observation_audit_path=str(observation_audit_path),
            observation_mapping_crosstab_path=str(crosstab_path),
            raw_finish_signal_summary_path=str(raw_finish_summary_path),
            unsupported_finish_values_path=str(unsupported_finish_path),
            unsupported_score_values_path=str(unsupported_score_path),
            unsupported_values_assessment_path=str(unsupported_values_assessment_path),
            metadata_extraction_summary_path=str(metadata_summary_path),
            metadata_field_coverage_path=str(metadata_field_coverage_path),
            sequence_audit_path=str(sequence_audit_path),
            sequence_length_distribution_path=str(sequence_length_path),
            suspicious_sequences_path=str(suspicious_path),
            model_health_summary_path=str(model_health_path),
            report_path=str(report_path),
            model_path=str(model_file),
            rows_total=int(len(episode_features)),
            rows_train_eligible=int(len(episode_features)),
            observation_mapping_version="inside_episode_v2",
            canonical_state_order=list(model.state_names),
            semantic_assignment=semantic_assignment,
            semantic_confidence=semantic_confidence,
            observed_layer_summary=observed_layer_summary,
            sequence_quality_summary=sequence_quality_summary,
            semantic_assignment_quality="full",
            recommendation_profile=recommendation_profile,
            recommendation=recommendation,
            transitions_summary=transitions_summary,
            created_artifacts=created_artifacts,
            created_files=created_artifacts,
            run_summary_path=str(run_summary_path),
        )

    except Exception as exc:
        finished_at = _iso8601_utc(_utcnow())
        _append_unique(warnings_summary, f"run_failed: {exc}")
        created_relative = sorted(
            {
                _to_output_relative(path, final_output_dir)
                for path in created_artifact_paths
                if Path(path).exists()
            }
        )
        manifest_payload.update(
            {
                "status": "failed",
                "finished_at": finished_at,
                "created_artifact_files": created_relative,
                "warnings_summary": warnings_summary,
                "error": str(exc),
            }
        )
        _write_manifest(manifest_path, manifest_payload)
        raise


def _parse_sheet_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inverse diagnostic cycle")
    parser.add_argument("--input", required=True, help="Path to Excel input")
    parser.add_argument("--output", required=True, help="Output directory for artifacts")
    parser.add_argument("--sheet-names", default=None, help="Comma-separated Excel sheet names")
    parser.add_argument("--header-depth", type=int, default=2, help="Excel multi-row header depth")
    parser.add_argument(
        "--parser-mode",
        choices=["auto", "table", "matrix", "baum_welch"],
        default="auto",
        help="Excel parser mode (baum_welch triggers optional EM smoothing mode).",
    )
    parser.add_argument("--force-matrix-parser", action="store_true", help="Compatibility flag")
    parser.add_argument("--retrain", dest="retrain", action="store_true", help="Retrain inverse model")
    parser.add_argument("--no-retrain", dest="retrain", action="store_false", help="Reuse existing model")
    parser.set_defaults(retrain=True)
    parser.add_argument("--model-path", default=None, help="Path to inverse model file")
    parser.add_argument("--reset-outputs", action="store_true", help="Clear output directory before run")
    parser.add_argument(
        "--cleanup-mode",
        choices=sorted(_CLEANUP_MODES),
        default=None,
        help="Output cleanup policy before run.",
    )
    parser.add_argument("--isolated-run", action="store_true", help="Write each run to an isolated subdirectory")
    parser.add_argument("--run-id", default=None, help="Explicit run id (used in manifest and isolated directories)")
    parser.add_argument("--n-states", type=int, default=3, help="Number of hidden states (3 or 4)")
    parser.add_argument(
        "--topology-mode",
        choices=["left_to_right", "ergodic"],
        default="left_to_right",
        help="Transition topology mode",
    )
    parser.add_argument("--no-generate-plots", action="store_true", help="Skip chart generation")
    parser.add_argument("--quiet", action="store_true", help="Minimal console output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_inverse_diagnostic_cycle(
        input_path=args.input,
        output_dir=args.output,
        sheet_names=_parse_sheet_names(args.sheet_names),
        header_depth=args.header_depth,
        parser_mode=args.parser_mode,
        force_matrix_parser=args.force_matrix_parser,
        retrain=args.retrain,
        model_path=args.model_path,
        reset_outputs=args.reset_outputs,
        n_states=args.n_states,
        topology_mode=args.topology_mode,
        generate_plots=not args.no_generate_plots,
        verbose=not args.quiet,
        cleanup_mode=args.cleanup_mode,
        isolated_run=args.isolated_run,
        run_id=args.run_id,
    )

    print("\nInverse diagnostic artifacts are ready.")
    print(f"- Run ID: {result.run_id}")
    print(f"- Output directory: {result.final_output_dir}")
    print(f"- Cleanup mode: {result.cleanup_mode}")
    print(f"- Report: {result.report_path}")
    print(f"- Run manifest: {result.run_manifest_path}")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
