from __future__ import annotations

import argparse
import json
import hashlib
import re
import subprocess
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.diagnostics import (
    build_metadata_extraction_summary,
    build_model_health_summary,
    build_observation_audit,
    build_sequence_audit,
    write_metadata_audit,
    write_model_health_summary,
    write_observation_audit,
    write_sequence_audit,
)
from hidden_patterns_combat.features import build_hidden_state_feature_layer, encode_features
from hidden_patterns_combat.modeling import InverseDiagnosticHMM
from hidden_patterns_combat.preprocessing import (
    build_canonical_episode_table,
    build_observed_zap_classes,
    load_observation_mapping_config,
    run_preprocessing,
)
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
_ARTIFACT_DIRS = ("cleaned", "features", "diagnostics", "plots", "reports")


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


def _cleanup_output_dir(
    output_dir: Path,
    *,
    cleanup_mode: str,
) -> list[str]:
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
        "plots": output_dir / "plots",
        "reports": output_dir / "reports",
        "diagnostics": output_dir / "diagnostics",
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


def _expected_artifact_files(*, generate_plots: bool) -> list[str]:
    expected = [
        "cleaned/raw_combined.csv",
        "cleaned/cleaned_tidy.csv",
        "cleaned/data_dictionary.csv",
        "cleaned/validation.json",
        "cleaned/canonical_episode_table.csv",
        "cleaned/observed_sequence.csv",
        "features/hidden_state_features.csv",
        "diagnostics/episode_analysis.csv",
        "diagnostics/state_profile.csv",
        "diagnostics/quality_diagnostics.json",
        "diagnostics/run_summary.json",
        "diagnostics/observation_audit.json",
        "diagnostics/observation_mapping_crosstab.csv",
        "diagnostics/raw_finish_signal_summary.csv",
        "diagnostics/unsupported_finish_values.csv",
        "diagnostics/unsupported_score_values.csv",
        "diagnostics/metadata_extraction_summary.json",
        "diagnostics/metadata_field_coverage.csv",
        "diagnostics/sequence_audit.json",
        "diagnostics/sequence_length_distribution.csv",
        "diagnostics/suspicious_sequences.csv",
        "diagnostics/model_health_summary.json",
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


def _normalize_semantic_quality_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"full", "full_semantic_assignment"}:
        return "full"
    if text in {"partial", "partial_semantic_assignment"}:
        return "partial"
    if text in {"failed", "failed_semantic_assignment"}:
        return "failed"
    return "failed"


def _state_probability_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("p_state_")],
        key=lambda c: int(c.replace("p_state_", "")) if c.replace("p_state_", "").isdigit() else 10**6,
    )


def _dominant_link_row(row: pd.Series) -> str:
    maneuvering = float(row.get("maneuver_right_code", 0.0) + row.get("maneuver_left_code", 0.0))
    kfv = float(
        row.get("kfv_capture_code", 0.0)
        + row.get("kfv_grip_code", 0.0)
        + row.get("kfv_wrap_code", 0.0)
        + row.get("kfv_hook_code", 0.0)
        + row.get("kfv_post_code", 0.0)
    )
    vup = float(row.get("vup_code", 0.0))

    scores = {"maneuvering": maneuvering, "kfv": kfv, "vup": vup}
    return max(scores, key=scores.get)


def _build_state_profile(analysis_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "maneuver_right_code",
        "maneuver_left_code",
        "kfv_capture_code",
        "kfv_grip_code",
        "kfv_wrap_code",
        "kfv_hook_code",
        "kfv_post_code",
        "vup_code",
        "score",
        "confidence",
    ]
    present = [c for c in group_cols if c in analysis_df.columns]

    profile = analysis_df.groupby("hidden_state", dropna=False)[present].mean(numeric_only=True)
    profile["episodes_count"] = analysis_df.groupby("hidden_state").size()
    profile = profile.reset_index()

    if "hidden_state_name" in analysis_df.columns:
        mapping = (
            analysis_df[["hidden_state", "hidden_state_name"]]
            .drop_duplicates()
            .set_index("hidden_state")["hidden_state_name"]
            .to_dict()
        )
        profile["hidden_state_name"] = profile["hidden_state"].map(
            lambda sid: str(mapping.get(sid, f"state_{int(sid)}"))
        )
    else:
        profile["hidden_state_name"] = profile["hidden_state"].map(lambda sid: f"state_{int(sid)}")

    profile["key_link"] = profile.apply(_dominant_link_row, axis=1)
    return profile


def _observed_layer_summary(analysis_df: pd.DataFrame) -> dict[str, float]:
    n = max(1, len(analysis_df))
    resolution = analysis_df.get("observation_resolution_type", pd.Series(["unknown"] * n))
    confidence = analysis_df.get("observation_confidence_label", pd.Series(["low"] * n))

    summary = {
        "direct_share": float((resolution == "direct_finish_signal").mean()),
        "inferred_from_score_share": float((resolution == "inferred_from_score").mean()),
        "no_score_rule_share": float((resolution == "no_score_rule").mean()),
        "ambiguous_share": float((resolution == "ambiguous").mean()),
        "unknown_share": float((resolution == "unknown").mean()),
        "high_conf_share": float((confidence == "high").mean()),
        "medium_conf_share": float((confidence == "medium").mean()),
        "low_conf_share": float((confidence == "low").mean()),
    }
    return summary


def _sequence_quality_summary(analysis_df: pd.DataFrame) -> dict[str, float]:
    n = max(1, len(analysis_df))
    quality = analysis_df.get("sequence_quality_flag", pd.Series(["low"] * n))
    resolution = analysis_df.get("sequence_resolution_type", pd.Series(["fallback"] * n))

    return {
        "high_quality_share": float((quality == "high").mean()),
        "medium_quality_share": float((quality == "medium").mean()),
        "low_quality_share": float((quality == "low").mean()),
        "explicit_sequence_share": float((resolution == "explicit").mean()),
        "surrogate_sequence_share": float((resolution == "surrogate").mean()),
        "fallback_sequence_share": float((resolution == "fallback").mean()),
    }


def _transition_summary(analysis_df: pd.DataFrame) -> list[dict[str, object]]:
    if analysis_df.empty:
        return []

    seq = (
        analysis_df.get("sequence_id", pd.Series(["sequence_0"] * len(analysis_df), index=analysis_df.index))
        .fillna("sequence_0")
        .astype(str)
        .replace({"": "sequence_0", "nan": "sequence_0", "None": "sequence_0"})
    )
    states = analysis_df["hidden_state"].astype(int).tolist()
    names = analysis_df["hidden_state_name"].astype(str).tolist()

    transitions: dict[tuple[int, int, str, str], int] = {}
    start = 0
    for i in range(1, len(states) + 1):
        if i < len(states) and seq.iloc[i] == seq.iloc[start]:
            continue
        segment_states = states[start:i]
        segment_names = names[start:i]
        for j in range(len(segment_states) - 1):
            key = (
                int(segment_states[j]),
                int(segment_states[j + 1]),
                str(segment_names[j]),
                str(segment_names[j + 1]),
            )
            transitions[key] = transitions.get(key, 0) + 1
        start = i

    total = sum(transitions.values())
    if total <= 0:
        return []

    rows: list[dict[str, object]] = []
    for (src, dst, src_name, dst_name), count in transitions.items():
        rows.append(
            {
                "from_state": src,
                "to_state": dst,
                "from_name": src_name,
                "to_name": dst_name,
                "count": int(count),
                "share": float(count / total),
                "is_self_loop": bool(src == dst),
            }
        )

    rows.sort(key=lambda row: int(row["count"]), reverse=True)
    return rows


def _semantic_label_for_state(state_id: int, canonical_map: dict[str, object]) -> str:
    assignment = canonical_map.get("semantic_assignment", {}) or {}
    s1 = assignment.get("S1")
    s2 = assignment.get("S2")
    s3 = assignment.get("S3")

    def _to_int(value: object) -> int | None:
        if value is None:
            return None
        text = str(value)
        if text.lstrip("-").isdigit():
            return int(text)
        return None

    s1i = _to_int(s1)
    s2i = _to_int(s2)
    s3i = _to_int(s3)

    if s1i is not None and state_id == s1i:
        return "maneuvering"
    if s2i is not None and state_id == s2i:
        return "kfv"
    if s3i is not None and state_id == s3i:
        return "vup"
    return "other"


def _dominant_state_by_coverage(analysis_df: pd.DataFrame) -> tuple[int, float]:
    counts = analysis_df["hidden_state"].value_counts(normalize=True)
    if counts.empty:
        return -1, 0.0
    return int(counts.index[0]), float(counts.iloc[0])


def _dominant_state_high_conf(analysis_df: pd.DataFrame, threshold: float = 0.70) -> tuple[int, float]:
    high = analysis_df[analysis_df["confidence"] >= threshold]
    if high.empty:
        return -1, 0.0
    counts = high["hidden_state"].value_counts(normalize=True)
    if counts.empty:
        return -1, 0.0
    return int(counts.index[0]), float(counts.iloc[0])


def _dominant_state_mean_posterior(analysis_df: pd.DataFrame) -> tuple[int, float]:
    prob_cols = _state_probability_columns(analysis_df)
    if not prob_cols:
        return -1, 0.0
    means = analysis_df[prob_cols].mean(axis=0)
    best_col = str(means.idxmax())
    try:
        state_id = int(best_col.replace("p_state_", ""))
    except Exception:
        return -1, 0.0
    return state_id, float(means.max())


def _recommendation_profile(
    analysis_df: pd.DataFrame,
    canonical_map: dict[str, object],
    observed_summary: dict[str, float],
    sequence_summary: dict[str, float],
    model_health_summary: dict[str, Any],
) -> tuple[str, str]:
    if analysis_df.empty:
        return "low-confidence profile", "Интерпретация ограничена: отсутствуют эпизоды для анализа."

    dominant_cov_state, cov_share = _dominant_state_by_coverage(analysis_df)
    dominant_hc_state, hc_share = _dominant_state_high_conf(analysis_df)
    dominant_mp_state, mp_share = _dominant_state_mean_posterior(analysis_df)

    cov_label = _semantic_label_for_state(dominant_cov_state, canonical_map) if dominant_cov_state >= 0 else "other"
    hc_label = _semantic_label_for_state(dominant_hc_state, canonical_map) if dominant_hc_state >= 0 else "other"
    mp_label = _semantic_label_for_state(dominant_mp_state, canonical_map) if dominant_mp_state >= 0 else "other"

    votes = [cov_label, hc_label, mp_label]
    semantic_votes = [v for v in votes if v in {"maneuvering", "kfv", "vup"}]
    consensus = max(semantic_votes, key=semantic_votes.count) if semantic_votes else "other"
    consensus_count = semantic_votes.count(consensus) if semantic_votes else 0

    data_low_conf = (
        observed_summary.get("direct_share", 0.0) < 0.05
        or observed_summary.get("no_score_rule_share", 0.0) > 0.70
        or observed_summary.get("ambiguous_share", 0.0) + observed_summary.get("unknown_share", 0.0) > 0.20
        or sequence_summary.get("surrogate_sequence_share", 0.0) > 0.80
    )
    self_share = float(model_health_summary.get("self_transition_share", 0.0))
    semantic_quality = _normalize_semantic_quality_label(
        model_health_summary.get("semantic_assignment_quality", "failed")
    )
    confirmed_states = [str(x) for x in (model_health_summary.get("semantic_confirmed_states", []) or [])]
    assigned_states = [str(x) for x in (model_health_summary.get("semantic_assigned_states", []) or [])]
    degenerate_transition_warning = bool(model_health_summary.get("degenerate_transition_warning", False))
    maneuvering_only_warning = bool(model_health_summary.get("maneuvering_only_state_profile_warning", False))

    metrics_line = (
        "Метрики профиля: coverage={cov:.2f}, high_conf={hc:.2f}, mean_posterior={mp:.2f}, "
        "self_transition_share={selfs:.2f}."
    ).format(cov=cov_share, hc=hc_share, mp=mp_share, selfs=self_share)

    if semantic_quality == "failed":
        return (
            "cautious profile",
            "Содержательная привязка скрытых состояний к КФВ/ВУП не стабилизировалась на текущих данных. "
            + metrics_line,
        )

    if semantic_quality == "partial":
        if confirmed_states:
            confirmed_text = ", ".join(sorted(set(confirmed_states)))
            return (
                "cautious profile",
                f"Семантика подтверждена только для части состояний ({confirmed_text}); "
                "для остальных состояний содержательная интерпретация пока неустойчива. "
                + metrics_line,
            )
        if assigned_states:
            assigned_text = ", ".join(sorted(set(assigned_states)))
            return (
                "cautious profile",
                f"Назначения семантики есть ({assigned_text}), но устойчивость подтверждена недостаточно. "
                + metrics_line,
            )
        return (
            "cautious profile",
            "Семантика состояний назначена частично, но содержательная интерпретация ограничена. "
            + metrics_line,
        )

    if maneuvering_only_warning:
        return (
            "cautious profile",
            "Все скрытые состояния имеют maneuvering-like профиль; контраст между КФВ/ВУП не подтвержден. "
            + metrics_line,
        )

    if degenerate_transition_warning:
        return (
            "cautious profile",
            "Скрытая динамика близка к вырожденной (self-loops доминируют или эффективно используется мало состояний). "
            + metrics_line,
        )

    if data_low_conf:
        return (
            "cautious profile",
            "Интерпретация ограничена качеством наблюдаемого слоя/сегментации последовательностей. "
            + metrics_line,
        )

    confident = consensus_count >= 2 and cov_share >= 0.40 and mp_share >= 0.40
    if confident and consensus == "maneuvering":
        return (
            "maneuvering-dominant",
            "Наиболее вероятное ключевое звено: маневрирование (S1). "
            "Рекомендация: усилить перевод маневрирования в устойчивые входы в КФВ. "
            + metrics_line,
        )

    if confident and consensus == "kfv":
        return (
            "kfv-dominant",
            "Наиболее вероятное ключевое звено: КФВ (S2). "
            "Рекомендация: сместить акцент подготовки на КФВ-связки и развитие позиции. "
            + metrics_line,
        )

    if confident and consensus == "vup":
        return (
            "vup-dominant",
            "Наиболее вероятное ключевое звено: ВУП (S3). "
            "Рекомендация: усилить ВУП-компонент и доведение эпизодов до завершения. "
            + metrics_line,
        )

    if cov_share < 0.40 and mp_share < 0.40:
        return (
            "low-confidence profile",
            "Профиль состояния неустойчивый; уверенного доминирования состояния не выявлено. "
            + metrics_line,
        )

    return (
        "mixed profile",
        "Профиль смешанный: нет устойчивого доминирования одного ключевого звена. "
        "Рекомендация: распределить фокус между маневрированием, КФВ и ВУП по контексту эпизодов. "
        + metrics_line,
    )


def _write_inverse_report(
    report_path: Path,
    analysis_df: pd.DataFrame,
    state_profile: pd.DataFrame,
    recommendation_profile: str,
    recommendation: str,
    observed_summary: dict[str, float],
    sequence_summary: dict[str, float],
    transitions: list[dict[str, object]],
    observation_audit_summary: dict[str, Any],
    metadata_summary: dict[str, Any],
    sequence_audit_summary: dict[str, Any],
    model_health_summary: dict[str, Any],
    run_provenance: dict[str, Any] | None = None,
) -> None:
    def _frame_to_markdown(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False)
        except Exception:
            return frame.to_string(index=False)

    profile_cols = [
        "hidden_state",
        "hidden_state_name",
        "episodes_count",
        "confidence",
        "key_link",
    ]
    profile_view = state_profile[[c for c in profile_cols if c in state_profile.columns]].copy()

    sample_rows = analysis_df[
        [
            c
            for c in [
                "episode_id",
                "sequence_id",
                "observed_zap_class",
                "observation_resolution_type",
                "observation_confidence_label",
                "observation_quality_flag",
                "hidden_state_name",
                "confidence",
            ]
            if c in analysis_df.columns
        ]
    ].head(40)

    transition_view = pd.DataFrame(transitions[:15])
    semantic_quality = _normalize_semantic_quality_label(model_health_summary.get("semantic_assignment_quality", "failed"))
    confirmed_states = [str(x) for x in (model_health_summary.get("semantic_confirmed_states", []) or [])]
    assigned_states = [str(x) for x in (model_health_summary.get("semantic_assigned_states", []) or [])]
    unconfirmed_states = [str(x) for x in (model_health_summary.get("semantic_unconfirmed_states", []) or [])]

    unique_observed = (
        sorted(analysis_df["observed_zap_class"].dropna().astype(str).unique().tolist())
        if "observed_zap_class" in analysis_df.columns
        else []
    )
    direct_share = float(observed_summary.get("direct_share", 0.0))
    inferred_share = float(observed_summary.get("inferred_from_score_share", 0.0))
    no_score_share = float(observed_summary.get("no_score_rule_share", 0.0))
    unknown_share = float(observed_summary.get("unknown_share", 0.0))
    ambiguous_share = float(observed_summary.get("ambiguous_share", 0.0))
    surrogate_share = float(sequence_summary.get("surrogate_sequence_share", 0.0))
    explicit_share = float(sequence_summary.get("explicit_sequence_share", 0.0))
    fallback_share = float(sequence_summary.get("fallback_sequence_share", 0.0))

    direct_finish_available = bool(observation_audit_summary.get("direct_finish_observations_available", False))
    unsupported_scores = [int(x) for x in (observation_audit_summary.get("unsupported_score_values", []) or [])]
    unsupported_finish_columns = [
        str(x) for x in (observation_audit_summary.get("unsupported_finish_columns_with_positive_values", []) or [])
    ]
    mapping_gap_detected = bool(observation_audit_summary.get("mapping_gap_detected", False))
    mapped_finish_positive_rows = int(observation_audit_summary.get("mapped_finish_positive_rows", 0))
    unmapped_finish_positive_rows = int(observation_audit_summary.get("unmapped_finish_positive_rows", 0))
    finish_presence = observation_audit_summary.get("finish_signal_presence", {}) or {}

    metadata_informative = [str(x) for x in (metadata_summary.get("informative_fields", []) or [])]
    metadata_non_informative = [str(x) for x in (metadata_summary.get("non_informative_fields", []) or [])]
    segmentation_support = metadata_summary.get("segmentation_support", {}) or {}
    temporal_support = metadata_summary.get("temporal_modeling_support", {}) or {}
    critical_field_quality = metadata_summary.get("critical_field_quality", {}) or {}
    degenerate_transition_warning = bool(model_health_summary.get("degenerate_transition_warning", False))
    maneuvering_only_warning = bool(model_health_summary.get("maneuvering_only_state_profile_warning", False))

    exec_flags: list[str] = []
    if not direct_finish_available:
        exec_flags.append("Direct finish observations are absent or effectively absent in current run.")
    if no_score_share >= 0.70:
        exec_flags.append("Observed layer is mostly no_score.")
    if unsupported_scores:
        exec_flags.append(f"Unsupported score values detected: {unsupported_scores}.")
    if unsupported_finish_columns:
        exec_flags.append("Raw finish/action columns contain unmapped positive signals.")
    if mapping_gap_detected:
        exec_flags.append("Direct finish/action mapping has coverage gaps (unmapped positive signals exist).")
    if explicit_share <= 0.01 and surrogate_share >= 0.90:
        exec_flags.append("Sequence segmentation is surrogate-based.")
    if semantic_quality in {"partial", "failed"}:
        exec_flags.append("State semantics are not fully stabilized; strong KFV/VUP interpretation is unsafe.")
    if degenerate_transition_warning:
        exec_flags.append("Hidden transition dynamics are close to degenerate (self-loops dominate).")
    if maneuvering_only_warning:
        exec_flags.append("All hidden states are maneuvering-like; semantic contrast is weak.")

    limitations: list[str] = []
    for source_key in ("warnings",):
        limitations.extend([str(x) for x in (observation_audit_summary.get(source_key, []) or [])])
        limitations.extend([str(x) for x in (metadata_summary.get(source_key, []) or [])])
        limitations.extend([str(x) for x in (sequence_audit_summary.get(source_key, []) or [])])
        limitations.extend([str(x) for x in (model_health_summary.get(source_key, []) or [])])
    limitations = sorted(set(x for x in limitations if x))

    interpretation_line = "Интерпретация ограничена."
    if semantic_quality == "failed":
        interpretation_line = (
            "Содержательная привязка скрытых состояний к КФВ/ВУП не стабилизировалась на текущих данных."
        )
    elif semantic_quality == "partial":
        if confirmed_states:
            interpretation_line = (
                f"Семантика подтверждается только для {', '.join(sorted(set(confirmed_states)))}; "
                "интерпретация остальных состояний не стабилизировалась."
            )
        else:
            interpretation_line = (
                "Семантика назначена частично, но подтверждение недостаточно; строгая привязка S1/S2/S3 ограничена."
            )
    elif maneuvering_only_warning:
        interpretation_line = (
            "Все скрытые состояния имеют maneuvering-like profile; контрастные профили КФВ/ВУП не подтверждены."
        )
    elif degenerate_transition_warning:
        interpretation_line = (
            "Скрытая динамика близка к вырожденной (self-loops доминируют); содержательные выводы ограничены."
        )
    elif semantic_quality == "full" and not profile_view.empty:
        dominant = profile_view.sort_values("episodes_count", ascending=False).iloc[0]
        interpretation_line = (
            f"Содержательная интерпретация допустима: доминирует {dominant.get('hidden_state_name', 'unknown')} "
            f"(key_link={dominant.get('key_link', 'unknown')}, episodes={int(dominant.get('episodes_count', 0))})."
        )
    if maneuvering_only_warning and semantic_quality in {"partial", "failed"}:
        interpretation_line += " Дополнительно: все состояния имеют maneuvering-like profile."

    metadata_view = pd.DataFrame(
        [
            {
                "field": field,
                "informative": bool(payload.get("informative", False)),
                "source_column": payload.get("source_column"),
                "missing_share": float(payload.get("missing_share", 1.0)),
                "zero_share": payload.get("zero_share"),
            }
            for field, payload in (metadata_summary.get("field_quality", {}) or {}).items()
        ]
    )
    if not metadata_view.empty:
        metadata_view = metadata_view.sort_values(["informative", "field"], ascending=[False, True]).head(20)
    critical_field_view = pd.DataFrame(
        [
            {
                "field": field,
                "found": bool(payload.get("found", False)),
                "source_column": payload.get("source_column"),
                "missing_share": float(payload.get("missing_share", 1.0)),
                "zero_share": payload.get("zero_share"),
                "informative": bool(payload.get("informative", False)),
                "sample_values": ", ".join([str(x) for x in (payload.get("sample_values", []) or [])][:3]),
            }
            for field, payload in critical_field_quality.items()
        ]
    )
    if not critical_field_view.empty:
        critical_field_view = critical_field_view.sort_values(["informative", "field"], ascending=[False, True]).reset_index(drop=True)

    next_actions: list[str] = []
    if not direct_finish_available:
        next_actions.append(
            "Проверить workbook на прямые finish/action поля и зафиксировать их в mapping rules как явные direct signals."
        )
    if unsupported_finish_columns:
        next_actions.append(
            "Расширить mapping rules для unsupported finish/action колонок из diagnostics/unsupported_finish_values.csv."
        )
    elif mapping_gap_detected:
        next_actions.append(
            "Проверить observation mapping: есть unmapped finish/action сигналы при наличии direct finish колонок."
        )
    if unsupported_scores:
        next_actions.append(
            "Добавить правила для unsupported score values или подтвердить, что они должны оставаться unknown."
        )
    if not bool(metadata_summary.get("episode_time_informative", False)):
        next_actions.append(
            "Подтвердить источник episode_time и расширить aliases для времени в canonical extraction."
        )
    if not bool(metadata_summary.get("weight_class_informative", False)):
        next_actions.append(
            "Подтвердить источник weight_class; не использовать эвристическую подмену без явного признака колонки."
        )
    if explicit_share <= 0.01:
        next_actions.append(
            "Добавить явный bout/sequence marker на уровне ingest или metadata export, чтобы уйти от surrogate-only segmentation."
        )
    if bool(model_health_summary.get("degenerate_transition_warning", False)):
        next_actions.append("Перед повторным обучением увеличить информативность observed layer и разнообразие последовательностей.")
    if semantic_quality in {"partial", "failed"}:
        next_actions.append("Повторно оценить semantic assignment после усиления observed/metadata/segmentation слоев.")

    lines = [
        "# Inverse Diagnostic Report",
        "",
        "## 1) Executive summary",
        f"- Rows analyzed: {len(analysis_df)}",
        f"- Unique observed classes: {unique_observed}",
        f"- Mean posterior confidence: {float(analysis_df['confidence'].mean()):.3f}" if "confidence" in analysis_df.columns else "- Mean posterior confidence: n/a",
        f"- semantic_assignment_quality: {semantic_quality}",
    ]
    for flag in exec_flags:
        lines.append(f"- {flag}")

    lines += [
        "",
        "## 2) Observed layer quality",
        f"- direct_finish_signal share: {direct_share:.3f}",
        f"- inferred_from_score share: {inferred_share:.3f}",
        f"- no_score_rule share: {no_score_share:.3f}",
        f"- ambiguous share: {ambiguous_share:.3f}",
        f"- unknown share: {unknown_share:.3f}",
        f"- direct observations available: {direct_finish_available}",
    ]
    if no_score_share >= 0.70:
        lines.append("- observed layer mostly no_score: True")

    lines += [
        "",
        "## 3) Raw finish/action signal audit",
        f"- candidate finish/action columns: {int(observation_audit_summary.get('candidate_finish_columns_count', 0))}",
        f"- supported finish columns: {observation_audit_summary.get('supported_finish_columns', [])}",
        f"- supported finish columns with positive values: {observation_audit_summary.get('supported_finish_columns_with_positive_values', [])}",
        f"- mapped finish positive rows: {mapped_finish_positive_rows}",
        f"- unmapped finish positive rows: {unmapped_finish_positive_rows}",
        f"- mapping gap detected: {mapping_gap_detected}",
        f"- unsupported finish columns with positive values: {unsupported_finish_columns}",
        f"- unsupported finish values rows: {int(observation_audit_summary.get('unsupported_finish_values_rows', 0))}",
        f"- unsupported score values: {unsupported_scores}",
        f"- direct finish match class counts: {observation_audit_summary.get('direct_finish_match_class_counts', {})}",
        f"- hold/arm/leg presence: {finish_presence}",
    ]
    if not direct_finish_available:
        lines.append("- direct finish observations absent in this run: True")

    lines += [
        "",
        "## 4) Metadata and time extraction quality",
        f"- informative metadata fields: {metadata_informative}",
        f"- non-informative metadata fields: {metadata_non_informative}",
        f"- episode_time informative: {bool(metadata_summary.get('episode_time_informative', False))}",
        f"- pause_time informative: {bool(metadata_summary.get('pause_time_informative', False))}",
        f"- weight_class informative: {bool(metadata_summary.get('weight_class_informative', False))}",
        f"- segmentation support: {segmentation_support}",
        f"- temporal modeling support: {temporal_support}",
    ]
    if not critical_field_view.empty:
        lines += [
            "",
            "### Critical Metadata/Time Fields",
            _frame_to_markdown(critical_field_view),
        ]
    if metadata_view.empty:
        lines.append("- metadata field coverage table is unavailable.")
    else:
        lines += [
            "",
            "### Metadata Field Coverage (preview)",
            _frame_to_markdown(metadata_view),
        ]

    lines += [
        "",
        "## 5) Sequence segmentation quality",
        f"- explicit/surrogate/fallback shares: {explicit_share:.3f} / {surrogate_share:.3f} / {fallback_share:.3f}",
        f"- high/medium/low sequence quality shares: {sequence_summary.get('high_quality_share', 0.0):.3f} / {sequence_summary.get('medium_quality_share', 0.0):.3f} / {sequence_summary.get('low_quality_share', 0.0):.3f}",
        f"- explicit sequence fields checked: {sequence_audit_summary.get('explicit_sequence_fields_checked', [])}",
        f"- explicit sequence source column: {sequence_audit_summary.get('explicit_sequence_source_column')}",
        f"- explicit sequence fields missing: {sequence_audit_summary.get('explicit_sequence_fields_missing', [])}",
        f"- surrogate_based_segmentation: {bool(sequence_audit_summary.get('surrogate_based_segmentation', False))}",
        f"- surrogate reason counts: {sequence_audit_summary.get('surrogate_reason_counts', {})}",
        f"- suspicious potential multi-bout sequences: {sequence_audit_summary.get('suspicious_potential_multi_bout_sequences', 0)}",
        f"- potential multi-bout ids preview: {sequence_audit_summary.get('potential_multi_bout_sequence_ids_preview', [])}",
    ]

    lines += [
        "",
        "## 6) Model health",
        f"- self_transition_share: {float(model_health_summary.get('self_transition_share', 0.0)):.3f}",
        f"- top_self_transition_share: {float(model_health_summary.get('top_self_transition_share', 0.0)):.3f}",
        f"- effective_state_usage: {float(model_health_summary.get('effective_state_usage', 0.0)):.3f}",
        f"- degenerate_transition_warning: {bool(model_health_summary.get('degenerate_transition_warning', False))}",
        f"- low_information_observed_layer_warning: {bool(model_health_summary.get('low_information_observed_layer_warning', False))}",
        f"- maneuvering_only_state_profile_warning: {bool(model_health_summary.get('maneuvering_only_state_profile_warning', False))}",
    ]

    lines += [
        "",
        "## 7) State semantics quality",
        f"- semantic_assignment_quality: {semantic_quality}",
        f"- semantic_assignment: {model_health_summary.get('semantic_assignment', {})}",
        f"- semantic_confidence: {model_health_summary.get('semantic_confidence', {})}",
        f"- confirmed states: {confirmed_states}",
        f"- assigned but unconfirmed states: {unconfirmed_states}",
        f"- all assigned states: {assigned_states}",
        f"- {interpretation_line}",
        "",
        _frame_to_markdown(profile_view) if not profile_view.empty else "State profile is unavailable.",
    ]

    lines += [
        "",
        "## 8) Recommendation",
        f"- Profile: {recommendation_profile}",
        f"- {recommendation}",
        "",
        "## 9) Limitations",
    ]
    if limitations:
        for item in limitations:
            lines.append(f"- {item}")
    else:
        lines.append("- No critical caveats detected.")

    lines += [
        "",
        "## 10) Concrete next actions",
    ]
    if next_actions:
        for idx, action in enumerate(next_actions, start=1):
            lines.append(f"{idx}. {action}")
    else:
        lines.append("1. Continue with targeted data collection and rerun diagnostics.")

    lines += [
        "",
        "### Episode Preview",
        _frame_to_markdown(sample_rows) if not sample_rows.empty else "Episode preview is unavailable.",
        "",
        "### Transition Preview",
        _frame_to_markdown(transition_view) if not transition_view.empty else "No transitions available.",
    ]

    provenance = run_provenance or {}
    provenance_warnings = [str(x) for x in (provenance.get("key_warnings", []) or []) if str(x).strip()]
    lines += [
        "",
        "## 11) Run provenance / Reproducibility",
        f"- run_id: {provenance.get('run_id', 'n/a')}",
        f"- input file: {provenance.get('input_file', 'n/a')}",
        f"- output dir: {provenance.get('output_dir', 'n/a')}",
        f"- started_at: {provenance.get('started_at', 'n/a')}",
        f"- finished_at: {provenance.get('finished_at', 'n/a')}",
        f"- topology_mode: {provenance.get('topology_mode', 'n/a')}",
        f"- model_mode: {provenance.get('model_mode', 'n/a')}",
        f"- mapping_version: {provenance.get('mapping_version', 'n/a')}",
    ]
    if provenance_warnings:
        lines.append("- key warnings:")
        for warning in provenance_warnings:
            lines.append(f"  - {warning}")
    else:
        lines.append("- key warnings: none")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    """Run the inverse-diagnostic pipeline with run isolation and artifact provenance.

    Cleanup defaults:
    - `isolated_run=True`  -> `cleanup_mode="none"` (new unique run directory).
    - `isolated_run=False` -> `cleanup_mode="artifacts_only"` (safe cleanup of pipeline outputs only).
    - legacy `reset_outputs=True` upgrades default cleanup to `full_run_dir`.
    """
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

    run_fingerprint_payload: dict[str, object] = {
        "pipeline_mode": _PIPELINE_MODE,
        "input_path": str(input_resolved),
        "sheet_names": [str(x) for x in (sheet_names or [])],
        "header_depth": int(header_depth),
        "parser_mode": str(parser_mode),
        "force_matrix_parser": bool(force_matrix_parser),
        "n_states": int(n_states),
        "topology_mode": str(topology_mode),
        "retrain": bool(retrain),
        "model_path": None if model_path is None else str(model_path),
    }
    run_fingerprint = _stable_json_fingerprint(run_fingerprint_payload)
    expected_artifacts = _expected_artifact_files(generate_plots=generate_plots)

    manifest_warnings: list[str] = []
    if not input_resolved.exists():
        _append_unique(manifest_warnings, f"input_file_missing: {input_resolved}")
    input_hash: str | None = None
    if input_resolved.exists():
        try:
            input_hash = _sha256_file(input_resolved)
        except Exception as exc:
            _append_unique(manifest_warnings, f"input_file_hash_error: {exc}")

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
        "retrain": bool(retrain),
        "model_path_used": None if model_path is None else str(model_path),
        "mapping_version": None,
        "number_of_episodes": None,
        "number_of_train_eligible_episodes": None,
        "number_of_sequences": None,
        "expected_artifact_files": expected_artifacts,
        "created_artifact_files": [],
        "warnings_summary": manifest_warnings.copy(),
        "error": None,
    }
    _write_manifest(manifest_path, manifest_payload)

    if cleanup_actions:
        log(f"[cleanup] mode={effective_cleanup_mode}, removed={len(cleanup_actions)} item(s).")
    elif effective_cleanup_mode != "none":
        log(f"[cleanup] mode={effective_cleanup_mode}, no stale pipeline artifacts found.")
    else:
        log("[cleanup] mode=none, previous artifacts are preserved.")

    warnings_summary: list[str] = manifest_warnings.copy()
    created_artifact_paths: list[Path] = [manifest_path]

    try:
        if not input_resolved.exists():
            raise FileNotFoundError(f"Input Excel file not found: {input_resolved}")

        log("[1/7] Preprocessing input workbook...")
        preprocessing = run_preprocessing(
            excel_path=input_resolved,
            sheet_selector=sheet_names,
            header_depth=header_depth,
            output_dir=dirs["cleaned"],
            parser_mode=parser_mode,
            force_matrix_parser=force_matrix_parser,
        )

        cleaned_data_path = Path(preprocessing.exports["cleaned_csv"])
        cleaned_df = pd.read_csv(cleaned_data_path)
        if cleaned_df.empty:
            raise ValueError("Preprocessing produced empty dataframe.")

        log("[2/7] Building canonical observations and episode table...")
        cfg = PipelineConfig()
        cfg.model.n_hidden_states = n_states
        cfg.model.topology_mode = topology_mode

        encoded = encode_features(cleaned_df, cfg.features)

        observation_cfg = load_observation_mapping_config()
        observation_result = build_observed_zap_classes(cleaned_df, config=observation_cfg)
        observation_audit_result = build_observation_audit(
            cleaned_df=cleaned_df,
            observation_df=observation_result.observations,
            cfg=observation_cfg,
            score_column=observation_result.score_column,
            finish_signal_columns=observation_result.finish_signal_columns,
        )

        canonical_result = build_canonical_episode_table(
            cleaned_df=cleaned_df,
            observation_df=observation_result.observations,
            hidden_features=encoded.features,
        )
        canonical_df = canonical_result.canonical_table
        metadata_audit_result = build_metadata_extraction_summary(
            canonical_df=canonical_df,
            extraction_info=canonical_result.extraction_info,
        )

        hidden_layer = build_hidden_state_feature_layer(canonical_df)
        sequence_ids = (
            canonical_df.get("sequence_id", pd.Series(["sequence_0"] * len(canonical_df), index=canonical_df.index))
            .fillna("sequence_0")
            .astype(str)
            .replace({"": "sequence_0", "nan": "sequence_0", "None": "sequence_0"})
        )

        canonical_path = dirs["cleaned"] / "canonical_episode_table.csv"
        observations_path = dirs["cleaned"] / "observed_sequence.csv"
        hidden_layer_path = dirs["features"] / "hidden_state_features.csv"

        canonical_df.to_csv(canonical_path, index=False)
        observation_result.observations.to_csv(observations_path, index=False)
        hidden_layer.hidden_state_features.to_csv(hidden_layer_path, index=False)

        observation_audit_paths = write_observation_audit(observation_audit_result, diagnostics_dir=dirs["diagnostics"])
        metadata_extraction_summary_path = write_metadata_audit(
            metadata_audit_result,
            diagnostics_dir=dirs["diagnostics"],
        )
        metadata_field_coverage_path = dirs["diagnostics"] / "metadata_field_coverage.csv"

        model_file = Path(model_path) if model_path else (dirs["models"] / "inverse_hmm.pkl")

        train_mask = canonical_df["is_train_eligible"].fillna(False).astype(bool)
        rows_train = int(train_mask.sum())
        if rows_train == 0:
            raise ValueError(
                "No train-eligible episodes for inverse model. "
                "Check observation/sequence quality flags and source data completeness."
            )

        log("[3/7] Training/loading inverse diagnostic HMM...")
        model_retrained = False
        if retrain or not model_file.exists():
            model_retrained = True
            model = InverseDiagnosticHMM(
                cfg=cfg.model,
                observation_classes=[
                    "zap_r",
                    "zap_n",
                    "zap_t",
                    "hold",
                    "arm_submission",
                    "leg_submission",
                    "no_score",
                    "unknown",
                ],
            )
            model.fit(
                observed_sequence=canonical_df.loc[train_mask, "observed_zap_class"].reset_index(drop=True),
                sequence_ids=sequence_ids.loc[train_mask].reset_index(drop=True),
                hidden_state_features=hidden_layer.hidden_state_features.loc[train_mask].reset_index(drop=True),
            )
            model.save(model_file)
        else:
            model = InverseDiagnosticHMM.load(model_file)

        log("[4/7] Viterbi decoding and posterior profile...")
        prediction = model.predict(
            observed_sequence=canonical_df["observed_zap_class"],
            sequence_ids=sequence_ids,
        )

        canonical_map = model.canonical_state_mapping()
        canonical_to_name = {
            int(k): str(v)
            for k, v in (canonical_map.get("canonical_to_name", {}) or {}).items()
        }

        analysis_df = canonical_df.copy()
        analysis_df["hidden_state"] = pd.Series(prediction.states).astype(int)
        analysis_df["hidden_state_name"] = analysis_df["hidden_state"].map(
            lambda sid: canonical_to_name.get(int(sid), model.state_definition.state_name(int(sid)))
        )
        probs_df = pd.DataFrame(prediction.state_probabilities).add_prefix("p_state_")
        analysis_df = pd.concat([analysis_df.reset_index(drop=True), probs_df.reset_index(drop=True)], axis=1)
        analysis_df["confidence"] = probs_df.max(axis=1)
        analysis_df["observed_result"] = analysis_df["score"]

        observed_summary = _observed_layer_summary(analysis_df)
        transitions = _transition_summary(analysis_df)
        sequence_audit_result = build_sequence_audit(
            analysis_df,
            extraction_info=canonical_result.extraction_info,
        )
        sequence_audit_paths = write_sequence_audit(sequence_audit_result, diagnostics_dir=dirs["diagnostics"])
        seq_summary = {
            "high_quality_share": float(sequence_audit_result.summary.get("high_quality_share", 0.0)),
            "medium_quality_share": float(sequence_audit_result.summary.get("medium_quality_share", 0.0)),
            "low_quality_share": float(sequence_audit_result.summary.get("low_quality_share", 0.0)),
            "explicit_sequence_share": float(sequence_audit_result.summary.get("explicit_sequence_share", 0.0)),
            "surrogate_sequence_share": float(sequence_audit_result.summary.get("surrogate_sequence_share", 0.0)),
            "fallback_sequence_share": float(sequence_audit_result.summary.get("fallback_sequence_share", 0.0)),
        }

        state_profile = _build_state_profile(analysis_df)
        model_health_result = build_model_health_summary(
            analysis_df=analysis_df,
            transitions=transitions,
            canonical_map=canonical_map,
            observed_summary=observed_summary,
            state_profile=state_profile,
        )
        model_health_summary_path = write_model_health_summary(
            model_health_result,
            diagnostics_dir=dirs["diagnostics"],
        )

        recommendation_profile, recommendation = _recommendation_profile(
            analysis_df=analysis_df,
            canonical_map=canonical_map,
            observed_summary=observed_summary,
            sequence_summary=seq_summary,
            model_health_summary=model_health_result.summary,
        )
        semantic_assignment: dict[str, int] = {}
        for key, value in (canonical_map.get("semantic_assignment", {}) or {}).items():
            try:
                semantic_assignment[str(key)] = int(value)
            except Exception:
                continue
        semantic_confidence = {
            str(k): float(v) for k, v in (canonical_map.get("semantic_confidence", {}) or {}).items()
        }

        episode_analysis_path = dirs["diagnostics"] / "episode_analysis.csv"
        state_profile_path = dirs["diagnostics"] / "state_profile.csv"
        quality_diagnostics_path = dirs["diagnostics"] / "quality_diagnostics.json"
        run_summary_path = dirs["diagnostics"] / "run_summary.json"
        report_path = dirs["reports"] / "inverse_diagnostic_report.md"

        analysis_df.to_csv(episode_analysis_path, index=False)
        state_profile.to_csv(state_profile_path, index=False)

        run_summary_payload = {
            "run_id": effective_run_id,
            "run_fingerprint": run_fingerprint,
            "input_path": str(input_resolved),
            "output_dir": str(final_output_dir),
            "run_manifest_path": str(manifest_path),
            "rows_total": int(len(canonical_df)),
            "rows_train_eligible": int(rows_train),
            "n_sequences": int(sequence_ids.nunique(dropna=False)),
            "observation_mapping_version": str(observation_cfg.version),
            "semantic_assignment_quality": str(model_health_result.summary.get("semantic_assignment_quality", "failed")),
            "semantic_assignment": semantic_assignment,
            "semantic_confidence": semantic_confidence,
            "recommendation_profile": recommendation_profile,
            "recommendation": recommendation,
            "observed_layer_summary": observed_summary,
            "sequence_quality_summary": seq_summary,
            "direct_finish_observations_available": bool(
                observation_audit_result.summary.get("direct_finish_observations_available", False)
            ),
            "unsupported_score_values": [
                int(x) for x in (observation_audit_result.summary.get("unsupported_score_values", []) or [])
            ],
            "unsupported_finish_columns_with_positive_values": [
                str(x)
                for x in (
                    observation_audit_result.summary.get("unsupported_finish_columns_with_positive_values", []) or []
                )
            ],
            "episode_time_informative": bool(metadata_audit_result.summary.get("episode_time_informative", False)),
            "weight_class_informative": bool(metadata_audit_result.summary.get("weight_class_informative", False)),
            "surrogate_based_segmentation": bool(
                sequence_audit_result.summary.get("surrogate_based_segmentation", False)
            ),
        }
        run_summary_path.write_text(
            json.dumps(run_summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        quality_payload = {
            "run_summary": run_summary_payload,
            "observed_layer_summary": observed_summary,
            "sequence_quality_summary": seq_summary,
            "transitions_summary": transitions,
            "recommendation_profile": recommendation_profile,
            "observation_audit_summary": observation_audit_result.summary,
            "metadata_extraction_summary": metadata_audit_result.summary,
            "sequence_audit_summary": sequence_audit_result.summary,
            "model_health_summary": model_health_result.summary,
        }
        quality_diagnostics_path.write_text(
            json.dumps(quality_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for source in (
            observation_audit_result.summary,
            metadata_audit_result.summary,
            sequence_audit_result.summary,
            model_health_result.summary,
        ):
            for warning in source.get("warnings", []) or []:
                _append_unique(warnings_summary, str(warning))
        if not isolated_run and effective_cleanup_mode == "none":
            _append_unique(
                warnings_summary,
                "fixed_output_without_cleanup: stale artifacts from older runs may remain.",
            )

        log("[5/7] Rendering report and plots...")
        _write_inverse_report(
            report_path=report_path,
            analysis_df=analysis_df,
            state_profile=state_profile,
            recommendation_profile=recommendation_profile,
            recommendation=recommendation,
            observed_summary=observed_summary,
            sequence_summary=seq_summary,
            transitions=transitions,
            observation_audit_summary=observation_audit_result.summary,
            metadata_summary=metadata_audit_result.summary,
            sequence_audit_summary=sequence_audit_result.summary,
            model_health_summary=model_health_result.summary,
            run_provenance={
                "run_id": effective_run_id,
                "input_file": str(input_resolved.name),
                "output_dir": str(final_output_dir),
                "started_at": started_at,
                "finished_at": _iso8601_utc(_utcnow()),
                "topology_mode": str(topology_mode),
                "model_mode": "retrained" if model_retrained else "reused_existing_model",
                "mapping_version": str(observation_cfg.version),
                "key_warnings": warnings_summary,
            },
        )

        if generate_plots:
            create_analysis_charts(
                analysis_df,
                dirs["plots"],
                canonical_state_mapping=canonical_map,
                observed_signal_label="Observed ZAP class",
                transition_summary=transitions,
            )

        artifacts = [
            Path(preprocessing.exports["raw_csv"]),
            cleaned_data_path,
            canonical_path,
            observations_path,
            hidden_layer_path,
            model_file,
            episode_analysis_path,
            state_profile_path,
            quality_diagnostics_path,
            run_summary_path,
            Path(observation_audit_paths["observation_audit_json"]),
            Path(observation_audit_paths["observation_mapping_crosstab_csv"]),
            Path(observation_audit_paths["raw_finish_signal_summary_csv"]),
            Path(observation_audit_paths["unsupported_finish_values_csv"]),
            Path(observation_audit_paths["unsupported_score_values_csv"]),
            Path(metadata_extraction_summary_path),
            Path(metadata_field_coverage_path),
            Path(sequence_audit_paths["sequence_audit_json"]),
            Path(sequence_audit_paths["sequence_length_distribution_csv"]),
            Path(sequence_audit_paths["suspicious_sequences_csv"]),
            Path(model_health_summary_path),
            report_path,
            manifest_path,
        ]
        for export_key in ("mapping_csv", "validation_json", "episodes_tidy_csv", "matrix_label_mapping_csv"):
            export_path = preprocessing.exports.get(export_key)
            if export_path:
                artifacts.append(Path(export_path))
        if generate_plots:
            artifacts.extend(sorted(dirs["plots"].glob("*.png")))
        created_artifact_paths.extend(artifacts)
        created_artifacts: list[str] = []
        for path in created_artifact_paths:
            if not Path(path).exists():
                continue
            value = str(path)
            if value in created_artifacts:
                continue
            created_artifacts.append(value)
        created_relative = sorted(
            {
                _to_output_relative(p, final_output_dir)
                for p in created_artifacts
                if _to_output_relative(p, final_output_dir)
            }
        )

        finished_at = _iso8601_utc(_utcnow())
        manifest_payload.update(
            {
                "status": "completed",
                "finished_at": finished_at,
                "model_path_used": str(model_file),
                "mapping_version": str(observation_cfg.version),
                "number_of_episodes": int(len(canonical_df)),
                "number_of_train_eligible_episodes": int(rows_train),
                "number_of_sequences": int(sequence_ids.nunique(dropna=False)),
                "created_artifact_files": created_relative,
                "warnings_summary": warnings_summary,
                "error": None,
            }
        )
        _write_manifest(manifest_path, manifest_payload)

        log("[6/7] Finalizing outputs...")
        result = InverseDiagnosticResult(
            input_path=str(input_resolved),
            output_dir=str(final_output_dir),
            final_output_dir=str(final_output_dir),
            run_id=effective_run_id,
            run_manifest_path=str(manifest_path),
            cleanup_mode=effective_cleanup_mode,
            cleanup_actions=cleanup_actions,
            run_fingerprint=run_fingerprint,
            cleaned_data_path=str(cleaned_data_path),
            canonical_episode_table_path=str(canonical_path),
            observed_sequence_path=str(observations_path),
            hidden_feature_layer_path=str(hidden_layer_path),
            episode_analysis_path=str(episode_analysis_path),
            state_profile_path=str(state_profile_path),
            quality_diagnostics_path=str(quality_diagnostics_path),
            observation_audit_path=str(observation_audit_paths["observation_audit_json"]),
            observation_mapping_crosstab_path=str(observation_audit_paths["observation_mapping_crosstab_csv"]),
            raw_finish_signal_summary_path=str(observation_audit_paths["raw_finish_signal_summary_csv"]),
            unsupported_finish_values_path=str(observation_audit_paths["unsupported_finish_values_csv"]),
            unsupported_score_values_path=str(observation_audit_paths["unsupported_score_values_csv"]),
            metadata_extraction_summary_path=str(metadata_extraction_summary_path),
            metadata_field_coverage_path=str(metadata_field_coverage_path),
            sequence_audit_path=str(sequence_audit_paths["sequence_audit_json"]),
            sequence_length_distribution_path=str(sequence_audit_paths["sequence_length_distribution_csv"]),
            suspicious_sequences_path=str(sequence_audit_paths["suspicious_sequences_csv"]),
            model_health_summary_path=str(model_health_summary_path),
            report_path=str(report_path),
            model_path=str(model_file),
            rows_total=len(canonical_df),
            rows_train_eligible=rows_train,
            observation_mapping_version=str(observation_cfg.version),
            canonical_state_order=[str(x) for x in (canonical_map.get("canonical_state_names", []) or [])],
            semantic_assignment=semantic_assignment,
            semantic_confidence=semantic_confidence,
            observed_layer_summary=observed_summary,
            sequence_quality_summary=seq_summary,
            semantic_assignment_quality=str(model_health_result.summary.get("semantic_assignment_quality", "failed")),
            recommendation_profile=recommendation_profile,
            recommendation=recommendation,
            transitions_summary=transitions,
            created_artifacts=created_artifacts,
            created_files=created_artifacts,
            run_summary_path=str(run_summary_path),
        )

        log("[7/7] Inverse diagnostic cycle completed.")
        return result
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
        choices=["auto", "table", "matrix"],
        default="auto",
        help="Excel parser mode.",
    )
    parser.add_argument("--force-matrix-parser", action="store_true", help="Force matrix parser")
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
    parser.add_argument("--n-states", type=int, default=3, help="Number of hidden states")
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
