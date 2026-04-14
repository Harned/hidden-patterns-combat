from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hidden_patterns_combat.analysis.interpreter import state_profile_table, text_summary
from hidden_patterns_combat.config import PipelineConfig
from hidden_patterns_combat.features.encoder import encode_features, select_hmm_input_features
from hidden_patterns_combat.features.engineering import FeatureEngineeringResult, export_feature_sets
from hidden_patterns_combat.modeling.hmm_pipeline import HMMEngine
from hidden_patterns_combat.modeling.interpretation import interpret_decoded_states
from hidden_patterns_combat.preprocessing import run_preprocessing
from hidden_patterns_combat.visualization import create_analysis_charts

logger = logging.getLogger(__name__)


@dataclass
class FullCycleResult:
    input_path: str
    output_dir: str
    requested_sheets: list[str]
    loaded_sheets: list[str]
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
    hmm_input_features: list[str]
    dropped_constant_hmm_features: list[str]
    hmm_converged: bool | None
    hmm_n_iterations: int | None
    hmm_last_delta: float | None
    canonical_state_order: list[str]
    canonical_state_indices: list[int]
    canonical_state_mapping: dict[str, object]
    semantic_assignment: dict[str, int]
    semantic_confidence: dict[str, float]
    semantic_warnings: list[str]
    consistency_warnings: list[str]
    semantic_order_matches_topology_before_reorder: bool | None
    transition_summary: list[dict[str, object]]
    transition_alignment: dict[str, float]
    observed_signal: dict[str, object]
    observed_result_source_columns: list[str]
    observed_result_warning: str | None
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


def _is_present_text(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "<na>"}


def _build_sample_analysis(
    analysis_df: pd.DataFrame,
    canonical_state_mapping: dict[str, object] | None = None,
) -> dict[str, object]:
    if analysis_df.empty:
        return {"message": "No rows available for sample analysis."}

    key_signal_cols = [
        c
        for c in (
            "maneuver_right_code",
            "maneuver_left_code",
            "kfv_code",
            "vup_code",
            "outcome_actions_code",
            "duration",
            "pause",
            "observed_result",
        )
        if c in analysis_df.columns
    ]
    signal = (
        analysis_df[key_signal_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs().sum(axis=1)
        if key_signal_cols
        else pd.Series([0.0] * len(analysis_df), index=analysis_df.index)
    )
    episode_valid = (
        analysis_df["episode_id"].apply(_is_present_text)
        if "episode_id" in analysis_df.columns
        else pd.Series([False] * len(analysis_df), index=analysis_df.index)
    )
    sequence_valid = (
        analysis_df["sequence_id"].apply(_is_present_text)
        if "sequence_id" in analysis_df.columns
        else pd.Series([False] * len(analysis_df), index=analysis_df.index)
    )

    strict_mask = episode_valid & sequence_valid & signal.gt(0.0)
    soft_mask = episode_valid & sequence_valid
    if strict_mask.any():
        selected = analysis_df.loc[strict_mask].iloc[0]
        warning: str | None = None
    elif soft_mask.any():
        selected = analysis_df.loc[soft_mask].iloc[0]
        warning = "Sample selected without feature signal > 0; check sparsity/coverage."
    else:
        selected = analysis_df.iloc[0]
        warning = "No fully valid sample row found (episode_id/sequence_id missing). Showing first row."

    row = selected
    prob_cols = sorted([c for c in analysis_df.columns if c.startswith("p_state_")])
    canonical_state_mapping = canonical_state_mapping or {}
    canonical_to_name = {
        int(k): str(v) for k, v in (canonical_state_mapping.get("canonical_to_name", {}) or {}).items()
    }
    probabilities: dict[str, float] = {}
    for c in prob_cols:
        suffix = c.replace("p_state_", "", 1)
        if suffix.isdigit() and int(suffix) in canonical_to_name:
            probabilities[canonical_to_name[int(suffix)]] = float(row[c])
        else:
            probabilities[c] = float(row[c])

    payload = {
        "episode_id": str(row.get("episode_id", "0")),
        "sequence_id": str(row.get("sequence_id", "sequence_0")),
        "hidden_state": int(row.get("hidden_state", -1)),
        "hidden_state_name": str(row.get("hidden_state_name", "unknown")),
        "observed_result": float(row.get("observed_result", 0.0)),
        "probabilities": probabilities,
    }
    if warning:
        payload["warning"] = warning
    return payload


def _parse_json_list(raw: object) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v).strip()]


def _build_observed_signal_info(
    traceability: pd.DataFrame,
    raw_features: pd.DataFrame,
) -> dict[str, object]:
    if traceability.empty or "engineered_feature" not in traceability.columns:
        return {
            "signal_column": "observed_result",
            "signal_label": "Observed proxy score",
            "source_columns": [],
            "available_outcome_columns": [],
            "direct_zap": False,
            "classification": "proxy",
            "is_direct_observed_zap": False,
            "caveat": "Traceability is unavailable: observed signal source cannot be verified.",
        }

    rows = traceability[traceability["engineered_feature"] == "observed_result"]
    if rows.empty:
        return {
            "signal_column": "observed_result",
            "signal_label": "Observed proxy score",
            "source_columns": [],
            "available_outcome_columns": [],
            "direct_zap": False,
            "classification": "proxy",
            "is_direct_observed_zap": False,
            "caveat": "observed_result traceability row is missing.",
        }

    source_columns = _parse_json_list(rows.iloc[0].get("source_columns", "[]"))
    available_outcomes = sorted(
        [c for c in raw_features.columns if c.lower().startswith("outcomes__")]
    )
    direct_candidates = [
        c for c in available_outcomes
        if "zap" in c.lower() or c.lower() in {"outcomes__observed_zap", "outcomes__zap"}
    ]

    if direct_candidates:
        return {
            "signal_column": direct_candidates[0],
            "signal_label": "Observed ZAP",
            "source_columns": direct_candidates,
            "available_outcome_columns": available_outcomes,
            "direct_zap": True,
            "classification": "direct_zap",
            "is_direct_observed_zap": True,
            "caveat": None,
        }

    caveat = (
        "Current observed signal is a proxy score (numeric passthrough), not a direct observed ZAP variable. "
        "Dataset contains outcome score/actions, but no explicit observed ZAP column and no validated "
        "transformation rule from outcomes to direct ZAP."
    )
    return {
        "signal_column": "observed_result",
        "signal_label": "Observed proxy score",
        "source_columns": source_columns,
        "available_outcome_columns": available_outcomes,
        "direct_zap": False,
        "classification": "proxy",
        "is_direct_observed_zap": False,
        "caveat": caveat,
    }


def _observed_result_diagnostics(traceability: pd.DataFrame, raw_features: pd.DataFrame) -> tuple[list[str], str | None]:
    observed_signal = _build_observed_signal_info(traceability=traceability, raw_features=raw_features)
    return (
        [str(x) for x in observed_signal.get("source_columns", [])],
        str(observed_signal.get("caveat")) if observed_signal.get("caveat") else None,
    )


def _build_transition_summary(
    states: np.ndarray,
    sequence_ids: pd.Series,
    state_name_lookup: dict[int, str],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    if len(states) == 0:
        return [], {"forward_share": 0.0, "backward_share": 0.0, "self_share": 0.0}

    seq = (
        sequence_ids.fillna("sequence_0")
        .astype(str)
        .str.strip()
        .replace({"": "sequence_0", "nan": "sequence_0", "None": "sequence_0"})
    )

    transition_counts: dict[tuple[int, int], int] = {}
    forward = 0
    backward = 0
    self_loops = 0
    total = 0

    start = 0
    values = states.tolist()
    for i in range(1, len(values) + 1):
        if i < len(values) and seq.iloc[i] == seq.iloc[start]:
            continue

        segment = values[start:i]
        for j in range(len(segment) - 1):
            src = int(segment[j])
            dst = int(segment[j + 1])
            transition_counts[(src, dst)] = transition_counts.get((src, dst), 0) + 1
            total += 1
            if dst > src:
                forward += 1
            elif dst < src:
                backward += 1
            else:
                self_loops += 1
        start = i

    if total == 0:
        alignment = {"forward_share": 0.0, "backward_share": 0.0, "self_share": 1.0}
        return [], alignment

    summary = [
        {
            "from_state": src,
            "to_state": dst,
            "from_name": state_name_lookup.get(src, f"state_{src}"),
            "to_name": state_name_lookup.get(dst, f"state_{dst}"),
            "count": count,
            "share": float(count / total),
        }
        for (src, dst), count in transition_counts.items()
    ]
    summary = sorted(summary, key=lambda row: (int(row["count"]), float(row["share"])), reverse=True)

    alignment = {
        "forward_share": float(forward / total),
        "backward_share": float(backward / total),
        "self_share": float(self_loops / total),
    }
    return summary, alignment


def _canonicalize_analysis_states(
    analysis_df: pd.DataFrame,
    canonical_state_mapping: dict[str, object],
) -> pd.DataFrame:
    out = analysis_df.copy()
    canonical_to_name = {
        int(k): str(v) for k, v in (canonical_state_mapping.get("canonical_to_name", {}) or {}).items()
    }
    if not canonical_to_name:
        return out

    canonical_state_id = out["hidden_state"].astype(int)
    canonical_state_name = canonical_state_id.map(
        lambda sid: canonical_to_name.get(int(sid), f"state_{int(sid)}")
    )

    out["hidden_state"] = canonical_state_id
    out["hidden_state_name"] = canonical_state_name
    out["canonical_state_id"] = canonical_state_id
    out["canonical_state_name"] = canonical_state_name
    return out


def _transition_summary_to_count_map(transition_summary: list[dict[str, object]]) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for row in transition_summary:
        src = int(row.get("from_state", -1))
        dst = int(row.get("to_state", -1))
        if src < 0 or dst < 0:
            continue
        out[(src, dst)] = int(row.get("count", 0))
    return out


def _downstream_consistency_warnings(
    analysis_df: pd.DataFrame,
    canonical_state_mapping: dict[str, object],
    transition_summary: list[dict[str, object]],
    sequence_ids: pd.Series,
) -> list[str]:
    warnings: list[str] = []
    canonical_names = {str(x) for x in (canonical_state_mapping.get("canonical_state_names", []) or [])}
    if "canonical_state_name" in analysis_df.columns:
        observed_names = set(analysis_df["canonical_state_name"].dropna().astype(str).unique().tolist())
    elif "hidden_state_name" in analysis_df.columns:
        observed_names = set(analysis_df["hidden_state_name"].dropna().astype(str).unique().tolist())
    else:
        observed_names = set()

    unexpected_names = sorted([x for x in observed_names if x not in canonical_names])
    if unexpected_names:
        warnings.append(
            "Unexpected state labels in analysis/plots source (not in canonical mapping): "
            f"{unexpected_names}"
        )

    transition_names = {
        str(row.get("from_name", ""))
        for row in transition_summary
    } | {str(row.get("to_name", "")) for row in transition_summary}
    unexpected_transition_names = sorted(
        [x for x in transition_names if x and x not in canonical_names]
    )
    if unexpected_transition_names:
        warnings.append(
            "Transition summary contains non-canonical state labels: "
            f"{unexpected_transition_names}"
        )

    state_name_lookup = {
        int(k): str(v)
        for k, v in (canonical_state_mapping.get("canonical_to_name", {}) or {}).items()
    }
    states = analysis_df["hidden_state"].astype(int).to_numpy()
    recomputed, _ = _build_transition_summary(
        states=states,
        sequence_ids=sequence_ids,
        state_name_lookup=state_name_lookup,
    )
    if _transition_summary_to_count_map(recomputed) != _transition_summary_to_count_map(transition_summary):
        warnings.append(
            "Transition summary differs from canonical recomputation on analysis dataframe."
        )

    return warnings


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
        f"- Requested sheets: `{result.requested_sheets}`",
        f"- Loaded sheets: `{result.loaded_sheets}`",
        f"- Rows raw/clean: {result.n_rows_raw}/{result.n_rows_clean}",
        f"- Engineered features: {result.n_features}",
        f"- HMM input features: `{result.hmm_input_features}`",
        f"- Dropped constant HMM features: `{result.dropped_constant_hmm_features}`",
        f"- HMM converged: {result.hmm_converged}",
        f"- HMM iterations: {result.hmm_n_iterations}",
        f"- HMM last delta: {result.hmm_last_delta}",
        f"- Sequences: {result.n_sequences}",
        f"- Model states: {n_states}",
        f"- Canonical state order: {result.canonical_state_order}",
        f"- Semantic confidence: {result.semantic_confidence}",
        f"- Semantic assignment (canonical indices): {result.semantic_assignment}",
        (
            "- Semantic order matched topology before reorder: "
            f"{result.semantic_order_matches_topology_before_reorder}"
        ),
        f"- Transition alignment: {result.transition_alignment}",
        f"- observed_result source columns: {result.observed_result_source_columns}",
        f"- observed_result note: {result.observed_result_warning}",
        "",
        "## Observed Signal",
        f"- Signal column: {result.observed_signal.get('signal_column')}",
        f"- Signal label: {result.observed_signal.get('signal_label')}",
        f"- Source columns: {result.observed_signal.get('source_columns')}",
        f"- Is direct observed ZAP: {result.observed_signal.get('is_direct_observed_zap')}",
        f"- Classification: {result.observed_signal.get('classification')}",
        f"- Caveat: {result.observed_signal.get('caveat')}",
        "",
        "## State Summary",
    ]

    if result.state_summary:
        for item in result.state_summary:
            lines.append(
                "- state={state}, name={name}, episodes={episodes}, result_mean={result_mean:.4f}".format(
                    state=item.get("hidden_state", item.get("state_id", "?")),
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
            "## Transition Summary",
        ]
    )
    if result.transition_summary:
        for row in result.transition_summary[:10]:
            lines.append(
                "- {src} -> {dst}: count={count}, share={share:.3f}".format(
                    src=row.get("from_name", f"state_{row.get('from_state', '?')}"),
                    dst=row.get("to_name", f"state_{row.get('to_state', '?')}"),
                    count=int(row.get("count", 0)),
                    share=float(row.get("share", 0.0)),
                )
            )
    else:
        lines.append("- No transitions available.")

    lines.extend(["", "## Semantic Warnings"])
    if result.semantic_warnings:
        for warning in result.semantic_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Consistency Warnings"])
    if result.consistency_warnings:
        for warning in result.consistency_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None.")

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
    header_depth: int = 2,
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
    requested_sheets = list(sheet_names) if sheet_names else []

    def log(message: str) -> None:
        if verbose:
            print(message)

    log(f"[1/8] Preprocessing Excel input... requested_sheets={requested_sheets or 'ALL'}")
    preprocessing = run_preprocessing(
        excel_path=input_path,
        sheet_selector=sheet_names,
        header_depth=header_depth,
        output_dir=dirs["cleaned"],
        parser_mode=parser_mode,
        force_matrix_parser=force_matrix_parser,
    )
    log(f"      loaded_sheets={preprocessing.sheets_loaded}")

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
    hmm_features, hmm_feature_info = select_hmm_input_features(encoded.features, return_info=True)
    dropped_constant = hmm_feature_info.get("dropped_constant_features", [])
    if dropped_constant:
        log(f"      dropped_constant_hmm_features={dropped_constant}")
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
    training_diag: dict[str, object] = {
        "converged": None,
        "n_iterations": None,
        "last_delta": None,
    }
    if retrain:
        engine = HMMEngine(cfg.model)
        engine.fit(hmm_features, sequence_ids=sequence_ids)
        if engine.last_training_result is not None:
            training_diag = {
                "converged": bool(engine.last_training_result.converged),
                "n_iterations": int(engine.last_training_result.n_iterations),
                "last_delta": (
                    float(engine.last_training_result.last_delta)
                    if engine.last_training_result.last_delta is not None
                    else None
                ),
            }
            if training_diag["last_delta"] is not None and float(training_diag["last_delta"]) < 0:
                log(
                    "      convergence_note=negative final monitor delta; "
                    "model may be in a local unstable optimum."
                )
        if save_model:
            engine.save(model_file)
    else:
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found for reuse mode: {model_file}. "
                "Either provide --model-path to an existing model or run with retrain=True."
            )
        engine = HMMEngine.load(model_file)
    resolved_n_states = int(getattr(engine.model, "n_components", n_states))

    log("[5/8] Decoding hidden states and running analysis...")
    prediction = engine.predict(hmm_features, sequence_ids=sequence_ids)
    raw_analysis_df = pd.concat(
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

    semantic_diag = engine.last_semantic_diagnostics or {}
    canonical_state_mapping = engine.canonical_state_mapping()
    analysis_df = _canonicalize_analysis_states(
        raw_analysis_df,
        canonical_state_mapping=canonical_state_mapping,
    )
    canonical_state_indices = [
        int(i) for i in (canonical_state_mapping.get("canonical_state_ids", list(range(resolved_n_states))) or [])
    ]
    canonical_state_order = [str(x) for x in (canonical_state_mapping.get("canonical_state_names", []) or [])]
    if not canonical_state_order:
        canonical_state_order = [engine.state_definition.state_name(i) for i in canonical_state_indices]
    semantic_assignment = {
        str(k): int(v)
        for k, v in (
            canonical_state_mapping.get("semantic_assignment")
            or semantic_diag.get("semantic_to_state", {})
            or {}
        ).items()
    }
    semantic_confidence = {
        str(k): float(v)
        for k, v in (
            canonical_state_mapping.get("semantic_confidence")
            or semantic_diag.get("semantic_confidence", {})
            or {}
        ).items()
    }
    semantic_warnings = [str(w) for w in (semantic_diag.get("warnings", []) or [])]
    semantic_order_matches_topology_before_reorder = (
        canonical_state_mapping.get("semantic_order_matches_topology_before_reorder")
        if canonical_state_mapping
        else semantic_diag.get("semantic_order_matches_topology_before_reorder")
    )
    state_name_lookup = {
        int(k): str(v)
        for k, v in (
            canonical_state_mapping.get("canonical_to_name")
            or {idx: engine.state_definition.state_name(idx) for idx in range(resolved_n_states)}
        ).items()
    }
    transition_summary, transition_alignment = _build_transition_summary(
        states=prediction.states,
        sequence_ids=sequence_ids,
        state_name_lookup=state_name_lookup,
    )
    consistency_warnings = _downstream_consistency_warnings(
        analysis_df=analysis_df,
        canonical_state_mapping=canonical_state_mapping,
        transition_summary=transition_summary,
        sequence_ids=sequence_ids,
    )
    for warning in consistency_warnings:
        logger.warning("Downstream consistency warning: %s", warning)
    observed_signal = _build_observed_signal_info(
        traceability=encoded.traceability,
        raw_features=encoded.raw,
    )
    observed_result_source_columns, observed_result_warning = _observed_result_diagnostics(
        encoded.traceability,
        encoded.raw,
    )

    episode_analysis_path = dirs["diagnostics"] / "episode_analysis.csv"
    state_profile_path = dirs["diagnostics"] / "state_profile.csv"
    diagnostics_path = dirs["diagnostics"] / "hmm_state_interpretation.csv"
    interpretation_text_path = dirs["diagnostics"] / "interpretation.txt"
    semantic_diagnostics_path = dirs["diagnostics"] / "semantic_diagnostics.json"

    analysis_df.to_csv(episode_analysis_path, index=False)

    state_series = pd.Series(prediction.states, name="hidden_state")
    profile = state_profile_table(encoded.features, state_series, state_definition=engine.state_definition)
    profile.to_csv(state_profile_path, index=False)

    diagnostics_df = interpret_decoded_states(
        encoded.features,
        state_series,
        engine.state_definition,
        semantic_diagnostics=engine.last_semantic_diagnostics,
    )
    diagnostics_df.to_csv(diagnostics_path, index=False)

    interpretation_text_path.write_text(text_summary(profile), encoding="utf-8")
    semantic_diagnostics_path.write_text(
        json.dumps(
            {
                "canonical_state_order": canonical_state_order,
                "canonical_state_indices": canonical_state_indices,
                "semantic_assignment": semantic_assignment,
                "semantic_confidence": semantic_confidence,
                "semantic_warnings": semantic_warnings,
                "semantic_order_matches_topology_before_reorder": (
                    semantic_order_matches_topology_before_reorder
                ),
                "transition_alignment": transition_alignment,
                "top_transitions": transition_summary[:10],
                "canonical_transition_summary": transition_summary,
                "canonical_state_mapping": canonical_state_mapping,
                "observed_signal": observed_signal,
                "consistency_warnings": consistency_warnings,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log("[6/8] Generating visualizations...")
    if generate_plots:
        create_analysis_charts(
            analysis_df,
            dirs["plots"],
            canonical_state_mapping=canonical_state_mapping,
            observed_signal_label=str(observed_signal.get("signal_label", "Observed signal")),
            transition_summary=transition_summary,
        )

    state_summary = diagnostics_df.to_dict(orient="records")
    sample_analysis = _build_sample_analysis(
        analysis_df,
        canonical_state_mapping=canonical_state_mapping,
    )

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
        semantic_diagnostics_path,
    ]

    if save_model and model_file.exists():
        artifacts.append(model_file)

    if generate_plots:
        artifacts.extend(sorted(dirs["plots"].glob("*.png")))

    result = FullCycleResult(
        input_path=str(input_path),
        output_dir=str(output_dir),
        requested_sheets=requested_sheets,
        loaded_sheets=preprocessing.sheets_loaded,
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
        hmm_input_features=list(hmm_features.columns),
        dropped_constant_hmm_features=dropped_constant,
        hmm_converged=training_diag["converged"],
        hmm_n_iterations=training_diag["n_iterations"],
        hmm_last_delta=training_diag["last_delta"],
        canonical_state_order=canonical_state_order,
        canonical_state_indices=canonical_state_indices,
        canonical_state_mapping=canonical_state_mapping,
        semantic_assignment=semantic_assignment,
        semantic_confidence=semantic_confidence,
        semantic_warnings=semantic_warnings,
        consistency_warnings=consistency_warnings,
        semantic_order_matches_topology_before_reorder=(
            bool(semantic_order_matches_topology_before_reorder)
            if semantic_order_matches_topology_before_reorder is not None
            else None
        ),
        transition_summary=transition_summary,
        transition_alignment=transition_alignment,
        observed_signal=observed_signal,
        observed_result_source_columns=observed_result_source_columns,
        observed_result_warning=observed_result_warning,
        state_summary=state_summary,
        sample_analysis=sample_analysis,
        created_artifacts=[str(p) for p in artifacts if Path(p).exists()],
    )

    log("[7/8] Writing summary/report...")
    _write_full_cycle_report(Path(result.report_path), result=result, n_states=resolved_n_states)
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
    parser.add_argument("--header-depth", type=int, default=2, help="Excel multi-row header depth for table parser.")
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
        header_depth=args.header_depth,
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
