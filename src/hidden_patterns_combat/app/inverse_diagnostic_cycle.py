from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

import pandas as pd

from hidden_patterns_combat.config import PipelineConfig
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
    cleaned_data_path: str
    canonical_episode_table_path: str
    observed_sequence_path: str
    hidden_feature_layer_path: str
    episode_analysis_path: str
    state_profile_path: str
    quality_diagnostics_path: str
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
    recommendation_profile: str
    recommendation: str
    transitions_summary: list[dict[str, object]]
    created_artifacts: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _prepare_output_dirs(output_dir: Path, reset_outputs: bool) -> dict[str, Path]:
    if reset_outputs and output_dir.exists():
        shutil.rmtree(output_dir)

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

    name_map = {int(k): str(v) for k, v in (canonical_map.get("canonical_to_name", {}) or {}).items()}
    state_name = name_map.get(int(state_id), "")
    if state_name == "S1":
        return "maneuvering"
    if state_name == "S2":
        return "kfv"
    if state_name == "S3":
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
    transitions: list[dict[str, object]],
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
        observed_summary.get("ambiguous_share", 0.0) + observed_summary.get("unknown_share", 0.0) > 0.30
        or observed_summary.get("low_conf_share", 0.0) > 0.40
        or sequence_summary.get("low_quality_share", 0.0) > 0.35
    )

    transition_total = sum(int(row.get("count", 0)) for row in transitions)
    self_share = (
        sum(int(row.get("count", 0)) for row in transitions if bool(row.get("is_self_loop", False))) / transition_total
        if transition_total > 0
        else 0.0
    )

    metrics_line = (
        "Метрики профиля: coverage={cov:.2f}, high_conf={hc:.2f}, mean_posterior={mp:.2f}, "
        "self_transition_share={selfs:.2f}."
    ).format(cov=cov_share, hc=hc_share, mp=mp_share, selfs=self_share)

    if data_low_conf:
        return (
            "low-confidence profile",
            "Интерпретация ограничена качеством наблюдаемого слоя. "
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

    transition_view = pd.DataFrame(transitions[:12])

    lines = [
        "# Inverse Diagnostic Report",
        "",
        "## 1) Краткое резюме",
        f"- Всего эпизодов: {len(analysis_df)}",
        f"- Уникальных observed_zap_class: {sorted(analysis_df['observed_zap_class'].dropna().astype(str).unique().tolist())}",
        f"- Средняя confidence: {float(analysis_df['confidence'].mean()):.3f}",
        "",
        "## 2) Качество наблюдаемого слоя",
        f"- direct_finish_signal share: {observed_summary.get('direct_share', 0.0):.3f}",
        f"- inferred_from_score share: {observed_summary.get('inferred_from_score_share', 0.0):.3f}",
        f"- ambiguous+unknown share: {(observed_summary.get('ambiguous_share', 0.0) + observed_summary.get('unknown_share', 0.0)):.3f}",
        f"- high/medium/low confidence shares: {observed_summary.get('high_conf_share', 0.0):.3f} / {observed_summary.get('medium_conf_share', 0.0):.3f} / {observed_summary.get('low_conf_share', 0.0):.3f}",
        "",
        "## 3) Качество сегментации последовательностей",
        f"- explicit/surrogate/fallback shares: {sequence_summary.get('explicit_sequence_share', 0.0):.3f} / {sequence_summary.get('surrogate_sequence_share', 0.0):.3f} / {sequence_summary.get('fallback_sequence_share', 0.0):.3f}",
        f"- high/medium/low sequence quality shares: {sequence_summary.get('high_quality_share', 0.0):.3f} / {sequence_summary.get('medium_quality_share', 0.0):.3f} / {sequence_summary.get('low_quality_share', 0.0):.3f}",
        "",
        "## 4) Наблюдаемая последовательность по эпизодам",
        _frame_to_markdown(sample_rows),
        "",
        "## 5) Профиль скрытых состояний",
        _frame_to_markdown(profile_view),
        "",
        "## 6) Переходы состояний",
        _frame_to_markdown(transition_view) if not transition_view.empty else "Нет переходов для отображения.",
        "",
        "## 7) Интерпретация",
        "- S1: стойки и маневрирование",
        "- S2: КФВ",
        "- S3: ВУП",
        "",
        "## 8) Профиль рекомендации",
        f"- {recommendation_profile}",
        "",
        "## 9) Рекомендация",
        f"- {recommendation}",
    ]

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
) -> InverseDiagnosticResult:
    input_resolved = _resolve_input_path(input_path)
    output_dir = Path(output_dir)

    if not input_resolved.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_resolved}")

    dirs = _prepare_output_dirs(output_dir, reset_outputs=reset_outputs)

    def log(message: str) -> None:
        if verbose:
            print(message)

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

    canonical_result = build_canonical_episode_table(
        cleaned_df=cleaned_df,
        observation_df=observation_result.observations,
        hidden_features=encoded.features,
    )
    canonical_df = canonical_result.canonical_table

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

    model_file = Path(model_path) if model_path else (dirs["models"] / "inverse_hmm.pkl")

    train_mask = canonical_df["is_train_eligible"].fillna(False).astype(bool)
    rows_train = int(train_mask.sum())
    if rows_train == 0:
        raise ValueError(
            "No train-eligible episodes for inverse model. "
            "Check observation/sequence quality flags and source data completeness."
        )

    log("[3/7] Training/loading inverse diagnostic HMM...")
    if retrain or not model_file.exists():
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
    seq_summary = _sequence_quality_summary(analysis_df)
    transitions = _transition_summary(analysis_df)

    recommendation_profile, recommendation = _recommendation_profile(
        analysis_df=analysis_df,
        canonical_map=canonical_map,
        observed_summary=observed_summary,
        sequence_summary=seq_summary,
        transitions=transitions,
    )

    episode_analysis_path = dirs["diagnostics"] / "episode_analysis.csv"
    state_profile_path = dirs["diagnostics"] / "state_profile.csv"
    quality_diagnostics_path = dirs["diagnostics"] / "quality_diagnostics.json"
    report_path = dirs["reports"] / "inverse_diagnostic_report.md"

    analysis_df.to_csv(episode_analysis_path, index=False)

    state_profile = _build_state_profile(analysis_df)
    state_profile.to_csv(state_profile_path, index=False)

    quality_payload = {
        "observed_layer_summary": observed_summary,
        "sequence_quality_summary": seq_summary,
        "transitions_summary": transitions,
        "recommendation_profile": recommendation_profile,
    }
    quality_diagnostics_path.write_text(
        json.dumps(quality_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
    )

    if generate_plots:
        create_analysis_charts(
            analysis_df,
            dirs["plots"],
            canonical_state_mapping=canonical_map,
            observed_signal_label="Observed ZAP class",
            transition_summary=transitions,
        )

    semantic_assignment = {
        str(k): int(v) for k, v in (canonical_map.get("semantic_assignment", {}) or {}).items()
    }
    semantic_confidence = {
        str(k): float(v) for k, v in (canonical_map.get("semantic_confidence", {}) or {}).items()
    }

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
        report_path,
    ]
    if generate_plots:
        artifacts.extend(sorted(dirs["plots"].glob("*.png")))

    log("[6/7] Finalizing outputs...")
    result = InverseDiagnosticResult(
        input_path=str(input_resolved),
        output_dir=str(output_dir),
        cleaned_data_path=str(cleaned_data_path),
        canonical_episode_table_path=str(canonical_path),
        observed_sequence_path=str(observations_path),
        hidden_feature_layer_path=str(hidden_layer_path),
        episode_analysis_path=str(episode_analysis_path),
        state_profile_path=str(state_profile_path),
        quality_diagnostics_path=str(quality_diagnostics_path),
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
        recommendation_profile=recommendation_profile,
        recommendation=recommendation,
        transitions_summary=transitions,
        created_artifacts=[str(p) for p in artifacts if Path(p).exists()],
    )

    log("[7/7] Inverse diagnostic cycle completed.")
    return result


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
    )

    print("\nInverse diagnostic artifacts are ready.")
    print(f"- Output directory: {result.output_dir}")
    print(f"- Report: {result.report_path}")
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
