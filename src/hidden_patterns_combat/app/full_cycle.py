from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from hidden_patterns_combat.analysis.interpreter import state_profile_table, text_summary
from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.features.encoder import encode_features, select_hmm_input_features
from hidden_patterns_combat.features.engineering import FeatureEngineeringResult, export_feature_sets
from hidden_patterns_combat.modeling.hmm_pipeline import HMMEngine
from hidden_patterns_combat.modeling.interpretation import interpret_decoded_states
from hidden_patterns_combat.preprocessing import run_preprocessing
from hidden_patterns_combat.visualization import create_analysis_charts


@dataclass
class FullCycleResult:
    input_path: str
    output_dir: str
    cleaned_data_path: str
    features_path: str
    model_path: str | None
    report_path: str
    plots_dir: str
    diagnostics_path: str | None
    n_rows_raw: int
    n_rows_clean: int
    n_sequences: int
    n_features: int
    state_summary: list[dict[str, object]]
    sample_analysis: dict[str, object]
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


def _resolve_model_path(
    model_path: str | Path | None,
    model_dir: Path,
) -> Path:
    if model_path:
        return Path(model_path)
    return model_dir / "hmm_model.pkl"


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


def _build_sample_analysis(analysis_df: pd.DataFrame) -> dict[str, object]:
    if analysis_df.empty:
        return {"message": "No rows available for sample analysis."}

    row = analysis_df.iloc[0]
    prob_cols = sorted([c for c in analysis_df.columns if c.startswith("p_state_")])
    probabilities = {c: float(row[c]) for c in prob_cols}

    return {
        "episode_id": str(row.get("episode_id", "0")),
        "sequence_id": str(row.get("sequence_id", "sequence_0")),
        "hidden_state": int(row.get("hidden_state", -1)),
        "hidden_state_name": str(row.get("hidden_state_name", "unknown")),
        "observed_result": float(row.get("observed_result", 0.0)),
        "probabilities": probabilities,
    }


def _write_full_cycle_report(
    report_path: Path,
    result: FullCycleResult,
    n_states: int,
) -> None:
    lines = [
        "# Full Cycle Report",
        "",
        f"- Input file: `{result.input_path}`",
        f"- Output dir: `{result.output_dir}`",
        f"- Rows raw/clean: {result.n_rows_raw}/{result.n_rows_clean}",
        f"- Engineered features: {result.n_features}",
        f"- Sequences: {result.n_sequences}",
        f"- Model states: {n_states}",
        "",
        "## State Summary",
    ]

    if result.state_summary:
        for item in result.state_summary:
            lines.append(
                "- state={state}, name={name}, episodes={episodes}, result_mean={result_mean:.4f}".format(
                    state=item.get("hidden_state", "?"),
                    name=item.get("state_name", "unknown"),
                    episodes=item.get("episodes_count", 0),
                    result_mean=float(item.get("observed_result", 0.0)),
                )
            )
    else:
        lines.append("- No state summary available.")

    lines.extend(
        [
            "",
            "## Sample Analysis",
            f"- {json.dumps(result.sample_analysis, ensure_ascii=False)}",
            "",
            "## Artifacts",
        ]
    )
    for path in result.created_artifacts:
        lines.append(f"- `{path}`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_cycle(
    input_path: str | Path,
    output_dir: str | Path,
    mode: str | None = None,
    sheet_names: list[str] | None = None,
    parser_mode: str = "auto",
    force_matrix_parser: bool | None = None,
    retrain: bool = True,
    load_existing_model: bool = False,
    model_path: str | Path | None = None,
    save_model: bool = True,
    generate_plots: bool = True,
    reset_outputs: bool = False,
    random_state: int = 42,
    n_states: int = 3,
    topology_mode: str = "left_to_right",
    verbose: bool = True,
) -> FullCycleResult:
    input_path = _resolve_input_path(input_path)
    output_dir = Path(output_dir)

    if mode is not None and mode not in {"full", "fast"}:
        raise ValueError(f"Unsupported mode={mode!r}. Use one of: full, fast.")

    if mode == "full":
        retrain = True
        load_existing_model = False
        reset_outputs = True
        save_model = True
    elif mode == "fast":
        retrain = False
        load_existing_model = True
        reset_outputs = False

    if not input_path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_path}")

    if not retrain and not load_existing_model:
        raise ValueError("When retrain=False, set load_existing_model=True to run analysis.")

    dirs = _prepare_output_dirs(output_dir, reset_outputs=reset_outputs)

    def log(message: str) -> None:
        if verbose:
            print(message)

    log("[1/8] Preprocessing Excel input...")
    preprocessing = run_preprocessing(
        excel_path=input_path,
        sheet_selector=sheet_names,
        output_dir=dirs["cleaned"],
        parser_mode=parser_mode,
        force_matrix_parser=force_matrix_parser,
    )

    cleaned_data_path = Path(preprocessing.exports["cleaned_csv"])
    cleaned_df = pd.read_csv(cleaned_data_path)
    if cleaned_df.empty:
        raise ValueError(
            "Preprocessing produced 0 rows. Check input workbook/sheet selection and cleaning rules."
        )

    log("[2/8] Feature engineering...")
    cfg = PipelineConfig()
    cfg.model.n_hidden_states = n_states
    cfg.model.random_state = random_state
    cfg.model.topology_mode = topology_mode

    encoded = encode_features(cleaned_df, cfg.features)
    hmm_features = select_hmm_input_features(encoded.features)
    feature_exports = export_feature_sets(
        result=FeatureEngineeringResult(
            raw_feature_set=encoded.raw,
            engineered_feature_set=encoded.features,
            metadata=encoded.metadata,
            traceability=encoded.traceability,
            validation=encoded.validation,
        ),
        output_dir=dirs["features"],
    )

    log("[3/8] Preparing HMM sequences...")
    if encoded.features.empty:
        raise ValueError("Feature engineering produced 0 rows; HMM training cannot proceed.")

    sequence_ids = (
        encoded.metadata["sequence_id"]
        if "sequence_id" in encoded.metadata.columns
        else pd.Series(["sequence_0"] * len(hmm_features), index=hmm_features.index)
    )
    sequence_ids = (
        sequence_ids.fillna("sequence_0")
        .astype(str)
        .str.strip()
        .replace({"": "sequence_0", "nan": "sequence_0", "None": "sequence_0"})
    )

    if len(sequence_ids) != len(hmm_features):
        raise ValueError(
            f"Sequence id count mismatch: sequence_ids={len(sequence_ids)} vs features={len(hmm_features)}."
        )

    if len(hmm_features) < n_states:
        raise ValueError(
            f"Not enough rows for n_states={n_states}: got {len(hmm_features)} rows after preprocessing. "
            "Use fewer states or provide more data."
        )
    n_sequences = int(sequence_ids.nunique(dropna=False))
    n_features = int(hmm_features.shape[1])

    model_file = _resolve_model_path(model_path, dirs["models"])

    log("[4/8] Training/loading model...")
    if retrain:
        engine = HMMEngine(cfg.model)
        engine.fit(hmm_features, sequence_ids=sequence_ids)
        if save_model:
            engine.save(model_file)
    else:
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found for reuse mode: {model_file}. "
                "Either provide --model-path to an existing model or run with retrain=True."
            )
        engine = HMMEngine.load(model_file)

    log("[5/8] Decoding hidden states and running analysis...")
    prediction = engine.predict(hmm_features, sequence_ids=sequence_ids)
    analysis_df = pd.concat(
        [
            encoded.metadata.reset_index(drop=True),
            encoded.features.reset_index(drop=True),
            pd.DataFrame(
                {
                    "hidden_state": prediction.states,
                    "hidden_state_name": prediction.state_names,
                }
            ),
            pd.DataFrame(prediction.state_probabilities).add_prefix("p_state_"),
        ],
        axis=1,
    )

    episode_analysis_path = dirs["diagnostics"] / "episode_analysis.csv"
    state_profile_path = dirs["diagnostics"] / "state_profile.csv"
    diagnostics_path = dirs["diagnostics"] / "hmm_state_interpretation.csv"
    interpretation_text_path = dirs["diagnostics"] / "interpretation.txt"

    analysis_df.to_csv(episode_analysis_path, index=False)

    state_series = pd.Series(prediction.states, name="hidden_state")
    profile = state_profile_table(encoded.features, state_series, state_definition=engine.state_definition)
    profile.to_csv(state_profile_path, index=False)

    diagnostics_df = interpret_decoded_states(encoded.features, state_series, engine.state_definition)
    diagnostics_df.to_csv(diagnostics_path, index=False)

    interpretation_text_path.write_text(text_summary(profile), encoding="utf-8")

    log("[6/8] Generating visualizations...")
    if generate_plots:
        create_analysis_charts(analysis_df, dirs["plots"])

    state_summary = diagnostics_df.to_dict(orient="records")
    sample_analysis = _build_sample_analysis(analysis_df)

    artifacts = [
        Path(preprocessing.exports["raw_csv"]),
        cleaned_data_path,
        Path(preprocessing.exports["mapping_csv"]),
        Path(preprocessing.exports["validation_json"]),
        Path(feature_exports["raw_feature_set_csv"]),
        Path(feature_exports["engineered_feature_set_csv"]),
        Path(feature_exports["traceability_csv"]),
        Path(feature_exports["validation_json"]),
        episode_analysis_path,
        state_profile_path,
        diagnostics_path,
        interpretation_text_path,
    ]

    if save_model and model_file.exists():
        artifacts.append(model_file)

    if generate_plots:
        artifacts.extend(sorted(dirs["plots"].glob("*.png")))

    result = FullCycleResult(
        input_path=str(input_path),
        output_dir=str(output_dir),
        cleaned_data_path=str(cleaned_data_path),
        features_path=str(feature_exports["engineered_feature_set_csv"]),
        model_path=str(model_file) if model_file.exists() else None,
        report_path=str(dirs["reports"] / "full_cycle_report.md"),
        plots_dir=str(dirs["plots"]),
        diagnostics_path=str(diagnostics_path),
        n_rows_raw=preprocessing.rows_raw,
        n_rows_clean=preprocessing.rows_cleaned,
        n_sequences=n_sequences,
        n_features=n_features,
        state_summary=state_summary,
        sample_analysis=sample_analysis,
        created_artifacts=[str(p) for p in artifacts if Path(p).exists()],
    )

    log("[7/8] Writing summary/report...")
    _write_full_cycle_report(Path(result.report_path), result=result, n_states=n_states)
    if Path(result.report_path).exists():
        result.created_artifacts.append(result.report_path)

    log("[8/8] Full cycle completed.")
    return result


def _parse_sheet_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full end-to-end HMM cycle")
    parser.add_argument("--input", required=True, help="Path to Excel input")
    parser.add_argument("--output", required=True, help="Output directory for artifacts")
    parser.add_argument("--sheet-names", default=None, help="Comma-separated Excel sheet names")
    parser.add_argument(
        "--parser-mode",
        choices=["auto", "table", "matrix"],
        default="auto",
        help="Excel parser mode.",
    )
    parser.add_argument("--force-matrix-parser", action="store_true", help="Backward-compatible alias for parser-mode=matrix.")

    parser.add_argument("--retrain", dest="retrain", action="store_true", help="Train model before analysis")
    parser.add_argument("--no-retrain", dest="retrain", action="store_false", help="Skip training and reuse model")
    parser.set_defaults(retrain=True)

    parser.add_argument(
        "--load-existing-model",
        action="store_true",
        help="Allow loading model from --model-path or output/models/hmm_model.pkl",
    )
    parser.add_argument("--model-path", default=None, help="Path to model file (.pkl)")

    parser.add_argument("--save-model", dest="save_model", action="store_true", help="Save trained model")
    parser.add_argument("--no-save-model", dest="save_model", action="store_false", help="Do not save trained model")
    parser.set_defaults(save_model=True)

    parser.add_argument("--generate-plots", dest="generate_plots", action="store_true", help="Generate plots")
    parser.add_argument("--no-generate-plots", dest="generate_plots", action="store_false", help="Skip plots")
    parser.set_defaults(generate_plots=True)

    parser.add_argument("--reset-outputs", action="store_true", help="Clear output directory before run")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for HMM")
    parser.add_argument("--n-states", type=int, default=3, help="Number of hidden HMM states")
    parser.add_argument(
        "--topology-mode",
        choices=["left_to_right", "ergodic"],
        default="left_to_right",
        help="HMM transition topology mode.",
    )

    parser.add_argument(
        "--mode",
        choices=["full", "fast"],
        default=None,
        help="Preset: full=reset+retrain+save_model, fast=no_reset+reuse_model",
    )

    parser.add_argument("--quiet", action="store_true", help="Minimal console output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    retrain = args.retrain
    load_existing_model = args.load_existing_model
    reset_outputs = args.reset_outputs
    save_model = args.save_model

    if args.mode == "full":
        retrain = True
        load_existing_model = False
        reset_outputs = True
        save_model = True
    elif args.mode == "fast":
        retrain = False
        load_existing_model = True
        reset_outputs = False

    result = run_full_cycle(
        input_path=args.input,
        output_dir=args.output,
        sheet_names=_parse_sheet_names(args.sheet_names),
        parser_mode=args.parser_mode,
        force_matrix_parser=args.force_matrix_parser,
        retrain=retrain,
        load_existing_model=load_existing_model,
        model_path=args.model_path,
        save_model=save_model,
        generate_plots=args.generate_plots,
        reset_outputs=reset_outputs,
        random_state=args.random_state,
        n_states=args.n_states,
        topology_mode=args.topology_mode,
        verbose=not args.quiet,
    )

    payload = result.as_dict()
    print("\nFull cycle artifacts are ready.")
    print(f"- Output directory: {payload['output_dir']}")
    print(f"- Report: {payload['report_path']}")
    print(f"- Model: {payload['model_path']}")
    print("\nResult summary:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
