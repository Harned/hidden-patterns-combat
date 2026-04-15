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
    report_path: str
    model_path: str
    rows_total: int
    rows_train_eligible: int
    observation_mapping_version: str
    canonical_state_order: list[str]
    semantic_assignment: dict[str, int]
    semantic_confidence: dict[str, float]
    recommendation: str
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


def _safe_sequence_ids(canonical: pd.DataFrame, encoded_metadata: pd.DataFrame) -> pd.Series:
    if "sequence_id" in encoded_metadata.columns:
        seq = encoded_metadata["sequence_id"].astype(str)
        return (
            seq.fillna("sequence_0")
            .str.strip()
            .replace({"": "sequence_0", "nan": "sequence_0", "None": "sequence_0"})
        )

    athlete = canonical.get("athlete_id", pd.Series(["ath"] * len(canonical), index=canonical.index)).astype(str)
    sheet = canonical.get("sheet_name", pd.Series(["sheet"] * len(canonical), index=canonical.index)).astype(str)
    return (sheet + "::" + athlete).replace({"": "sequence_0"})


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
    profile["hidden_state_name"] = profile["hidden_state_name"] if "hidden_state_name" in profile.columns else profile["hidden_state"].map(
        lambda sid: str(analysis_df[analysis_df["hidden_state"] == sid]["hidden_state_name"].iloc[0])
        if (analysis_df["hidden_state"] == sid).any()
        else f"state_{sid}"
    )
    profile["key_link"] = profile.apply(_dominant_link_row, axis=1)
    return profile


def _fight_recommendation(analysis_df: pd.DataFrame, canonical_map: dict[str, object]) -> str:
    prob_cols = _state_probability_columns(analysis_df)
    if not prob_cols:
        return "Недостаточно данных для рекомендации (нет вероятностного профиля состояний)."

    mean_probs = analysis_df[prob_cols].mean(axis=0)
    top_col = mean_probs.idxmax()
    top_prob = float(mean_probs.max())
    unknown_share = float((analysis_df["observed_zap_class"] == "unknown").mean()) if "observed_zap_class" in analysis_df.columns else 0.0

    try:
        top_state = int(top_col.replace("p_state_", ""))
    except Exception:
        return "Недостаточно данных для рекомендации (не удалось определить доминирующее состояние)."

    state_name = str((canonical_map.get("canonical_to_name", {}) or {}).get(top_state, f"state_{top_state}"))
    semantic_assignment = canonical_map.get("semantic_assignment", {}) or {}
    s1 = int(semantic_assignment.get("S1", -1)) if str(semantic_assignment.get("S1", "-1")).lstrip("-").isdigit() else -1
    s2 = int(semantic_assignment.get("S2", -1)) if str(semantic_assignment.get("S2", "-1")).lstrip("-").isdigit() else -1
    s3 = int(semantic_assignment.get("S3", -1)) if str(semantic_assignment.get("S3", "-1")).lstrip("-").isdigit() else -1

    if top_prob < 0.55 or unknown_share > 0.35:
        return (
            "Уверенность недостаточна для целевой тактической рекомендации: "
            f"макс. средняя posterior={top_prob:.2f}, доля unknown={unknown_share:.2f}."
        )

    if s2 == top_state or state_name == "S2":
        return "Ключевое звено: КФВ. Рекомендация: сместить акцент подготовки и разбора на КФВ-связки."

    if s3 == top_state or state_name == "S3":
        return "Ключевое звено: ВУП. Рекомендация: усилить подготовку по ВУП и доведению эпизодов до завершения."

    if s1 == top_state or state_name == "S1":
        return "Ключевое звено: стойка и маневрирование. Рекомендация: усилить работу по входу в выгодный КФВ через маневрирование."

    return "Доминирует нейтральное латентное состояние; целевая рекомендация ограничена текущей уверенностью модели."


def _write_inverse_report(
    report_path: Path,
    analysis_df: pd.DataFrame,
    state_profile: pd.DataFrame,
    recommendation: str,
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
                "observed_zap_class",
                "hidden_state_name",
                "confidence",
                "observation_quality_flag",
            ]
            if c in analysis_df.columns
        ]
    ].head(30)

    lines = [
        "# Inverse Diagnostic Report",
        "",
        "## 1) Краткое резюме",
        f"- Всего эпизодов: {len(analysis_df)}",
        f"- Уникальных observed_zap_class: {sorted(analysis_df['observed_zap_class'].dropna().astype(str).unique().tolist())}",
        f"- Средняя confidence: {float(analysis_df['confidence'].mean()):.3f}",
        "",
        "## 2) Наблюдаемая последовательность по эпизодам",
        _frame_to_markdown(sample_rows),
        "",
        "## 3) Профиль скрытых состояний",
        _frame_to_markdown(profile_view),
        "",
        "## 4) Интерпретация",
        "- S1: стойки и маневрирование",
        "- S2: КФВ",
        "- S3: ВУП",
        "",
        "## 5) Рекомендация",
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

    log("[2/7] Building hidden-state feature layer...")
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
    sequence_ids = _safe_sequence_ids(canonical_df, encoded.metadata)

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
        raise ValueError("No train-eligible episodes for inverse model. Check observation mapping quality.")

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

    episode_analysis_path = dirs["diagnostics"] / "episode_analysis.csv"
    state_profile_path = dirs["diagnostics"] / "state_profile.csv"
    report_path = dirs["reports"] / "inverse_diagnostic_report.md"

    analysis_df.to_csv(episode_analysis_path, index=False)

    state_profile = _build_state_profile(analysis_df)
    state_profile.to_csv(state_profile_path, index=False)

    recommendation = _fight_recommendation(analysis_df, canonical_map)

    log("[5/7] Rendering report and plots...")
    _write_inverse_report(
        report_path=report_path,
        analysis_df=analysis_df,
        state_profile=state_profile,
        recommendation=recommendation,
    )

    if generate_plots:
        create_analysis_charts(
            analysis_df,
            dirs["plots"],
            canonical_state_mapping=canonical_map,
            observed_signal_label="Observed ZAP class",
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
        report_path=str(report_path),
        model_path=str(model_file),
        rows_total=len(canonical_df),
        rows_train_eligible=rows_train,
        observation_mapping_version=str(observation_cfg.version),
        canonical_state_order=[str(x) for x in (canonical_map.get("canonical_state_names", []) or [])],
        semantic_assignment=semantic_assignment,
        semantic_confidence=semantic_confidence,
        recommendation=recommendation,
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
