from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import logging
import re

import numpy as np
import pandas as pd

from hidden_patterns_combat.config import FeatureConfig
from hidden_patterns_combat.preprocessing.data_dictionary import DataDictionary

logger = logging.getLogger(__name__)


EncodingMode = str  # bitpack | count | sum | any


@dataclass
class FeatureEngineeringConfig:
    """Configurable rules for compact feature engineering."""

    maneuver_split_strategy: str = "half"  # half | explicit_tokens
    required_groups: tuple[str, ...] = ("maneuvering", "kfv", "vup", "outcomes")

    group_encoders: dict[str, EncodingMode] = field(
        default_factory=lambda: {
            "maneuver_right": "bitpack",
            "maneuver_left": "bitpack",
            "grips": "bitpack",
            "holds": "bitpack",
            "bodylocks": "bitpack",
            "underhooks": "bitpack",
            "posts": "bitpack",
            "kfv_all": "bitpack",
            "vup": "bitpack",
            "outcome_actions": "bitpack",
            "observed_result": "sum",
            "duration": "sum",
            "pause": "sum",
        }
    )

    # Index ranges are 1-based and inclusive over ordered KFV indicators.
    # This is a research-MVP hypothesis and can be replaced without changing API.
    kfv_subgroup_ranges: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "grips": (1, 6),
            "holds": (7, 12),
            "bodylocks": (13, 18),
            "underhooks": (19, 24),
            "posts": (25, 29),
        }
    )


@dataclass
class FeatureValidationReport:
    is_valid: bool
    missing_groups: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class FeatureEngineeringResult:
    raw_feature_set: pd.DataFrame
    engineered_feature_set: pd.DataFrame
    metadata: pd.DataFrame
    traceability: pd.DataFrame
    validation: FeatureValidationReport


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    as_str = series.astype(str).str.strip().str.lower()
    mapper = {
        "1": 1,
        "0": 0,
        "да": 1,
        "нет": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
    }
    mapped = as_str.map(mapper)
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = mapped.where(~mapped.isna(), (numeric > 0).astype(float))
    return mapped.fillna(0).astype(int)


def _compact_bitpack(binary_frame: pd.DataFrame) -> pd.Series:
    if binary_frame.empty:
        return pd.Series(np.zeros(len(binary_frame), dtype=int), index=binary_frame.index)
    bits = np.array([1 << i for i in range(binary_frame.shape[1])], dtype=np.int64)
    data = binary_frame.to_numpy(dtype=np.int64)
    return pd.Series((data * bits).sum(axis=1), index=binary_frame.index)


def _encode(binary_frame: pd.DataFrame, mode: EncodingMode) -> pd.Series:
    if binary_frame.empty:
        return pd.Series(np.zeros(len(binary_frame), dtype=float), index=binary_frame.index)

    if mode == "bitpack":
        return _compact_bitpack(binary_frame)
    if mode == "count":
        return binary_frame.sum(axis=1)
    if mode == "sum":
        return binary_frame.sum(axis=1)
    if mode == "any":
        return (binary_frame.sum(axis=1) > 0).astype(int)
    raise ValueError(f"Unsupported encoding mode: {mode}")


def _parse_indicator_idx(col: str) -> int | None:
    m = re.search(r"indicator_(\d+)$", col)
    if m:
        return int(m.group(1))
    m2 = re.search(r"_(\d+)$", col)
    if m2:
        return int(m2.group(1))
    return None


def _find_columns(df: pd.DataFrame, tokens: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    low_tokens = tuple(t.lower() for t in tokens)
    for col in df.columns:
        low = col.lower()
        if any(t in low for t in low_tokens):
            result.append(col)
    return result


class FeatureEngineer:
    def __init__(
        self,
        feature_cfg: FeatureConfig,
        engineering_cfg: FeatureEngineeringConfig | None = None,
        data_dictionary: DataDictionary | None = None,
    ):
        self.feature_cfg = feature_cfg
        self.engineering_cfg = engineering_cfg or FeatureEngineeringConfig()
        self.dictionary = data_dictionary or DataDictionary.default()

    def _candidate_columns_for_group(self, df: pd.DataFrame, group: str) -> list[str]:
        cols = self.dictionary.columns_for_group(list(df.columns), group)
        if cols:
            return cols
        if group == "kfv":
            return _find_columns(df, self.feature_cfg.kfv_tokens)
        if group == "vup":
            return _find_columns(df, self.feature_cfg.vup_tokens)
        if group == "maneuvering":
            return _find_columns(df, self.feature_cfg.maneuver_group_tokens)
        return []

    def _split_maneuver(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        right = _find_columns(df, self.feature_cfg.maneuver_right_tokens)
        left = _find_columns(df, self.feature_cfg.maneuver_left_tokens)
        if right or left:
            return right, left

        group_cols = self._candidate_columns_for_group(df, "maneuvering")
        if not group_cols:
            return [], []

        midpoint = len(group_cols) // 2
        if self.engineering_cfg.maneuver_split_strategy == "half" and midpoint > 0:
            return group_cols[:midpoint], group_cols[midpoint:]

        if midpoint == 0:
            return group_cols, []
        return group_cols[:midpoint], group_cols[midpoint:]

    @staticmethod
    def _to_binary_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(index=df.index)
        return df[cols].apply(_to_binary)

    def _split_kfv_subgroups(self, kfv_cols: list[str]) -> dict[str, list[str]]:
        ordered = sorted(kfv_cols, key=lambda c: (_parse_indicator_idx(c) or 10**6, c))
        if not ordered:
            return {k: [] for k in self.engineering_cfg.kfv_subgroup_ranges}

        by_idx: list[tuple[int, str]] = []
        for i, c in enumerate(ordered, start=1):
            by_idx.append((_parse_indicator_idx(c) or i, c))

        result: dict[str, list[str]] = {}
        for subgroup, (start, end) in self.engineering_cfg.kfv_subgroup_ranges.items():
            result[subgroup] = [c for idx, c in by_idx if start <= idx <= end]
        return result

    def _metadata_column(self, df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        lowered = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
            for col in df.columns:
                if candidate in col.lower():
                    return col
        return None

    def _validate(self, group_columns: dict[str, list[str]]) -> FeatureValidationReport:
        missing = [g for g in self.engineering_cfg.required_groups if not group_columns.get(g)]
        warnings: list[str] = []
        if not group_columns.get("maneuvering"):
            warnings.append("No maneuvering columns found. Right/left features will be zeroed.")
        if not group_columns.get("kfv"):
            warnings.append("No KFV columns found. KFV subgroup features will be zeroed.")
        if not group_columns.get("vup"):
            warnings.append("No VUP columns found. VUP feature will be zeroed.")
        if not group_columns.get("outcomes"):
            warnings.append("No outcomes columns found. Outcome-action features will be zeroed.")

        return FeatureValidationReport(
            is_valid=len(missing) == 0,
            missing_groups=missing,
            warnings=warnings,
        )

    def transform(self, cleaned_df: pd.DataFrame) -> FeatureEngineeringResult:
        df = cleaned_df.copy()

        right_cols, left_cols = self._split_maneuver(df)
        kfv_cols = self._candidate_columns_for_group(df, "kfv")
        vup_cols = self._candidate_columns_for_group(df, "vup")
        outcomes_cols = self._candidate_columns_for_group(df, "outcomes")
        maneuver_all = sorted(set(right_cols + left_cols))

        group_columns = {
            "maneuvering": maneuver_all,
            "kfv": kfv_cols,
            "vup": vup_cols,
            "outcomes": outcomes_cols,
        }
        validation = self._validate(group_columns)

        kfv_subgroups = self._split_kfv_subgroups(kfv_cols)

        engineered = pd.DataFrame(index=df.index)
        trace_rows: list[dict[str, object]] = []

        def add_feature(name: str, cols: list[str], mode_key: str) -> None:
            mode = self.engineering_cfg.group_encoders.get(mode_key, "bitpack")
            bin_df = self._to_binary_frame(df, cols)
            engineered[name] = _encode(bin_df, mode)
            trace_rows.append(
                {
                    "engineered_feature": name,
                    "encoding_mode": mode,
                    "source_columns": json.dumps(cols, ensure_ascii=False),
                    "source_group": mode_key,
                    "source_count": len(cols),
                }
            )

        add_feature("maneuver_right_code", right_cols, "maneuver_right")
        add_feature("maneuver_left_code", left_cols, "maneuver_left")

        add_feature("grips_code", kfv_subgroups.get("grips", []), "grips")
        add_feature("holds_code", kfv_subgroups.get("holds", []), "holds")
        add_feature("bodylocks_code", kfv_subgroups.get("bodylocks", []), "bodylocks")
        add_feature("underhooks_code", kfv_subgroups.get("underhooks", []), "underhooks")
        add_feature("posts_code", kfv_subgroups.get("posts", []), "posts")
        add_feature("kfv_code", kfv_cols, "kfv_all")

        add_feature("vup_code", vup_cols, "vup")

        outcome_action_cols = [c for c in outcomes_cols if "score" not in c.lower() and "балл" not in c.lower()]
        add_feature("outcome_actions_code", outcome_action_cols, "outcome_actions")

        duration_col = self._metadata_column(df, self.feature_cfg.duration_column_candidates)
        pause_col = self._metadata_column(df, self.feature_cfg.pause_column_candidates)
        result_col = self._metadata_column(df, self.feature_cfg.result_column_candidates)
        episode_col = self._metadata_column(df, self.feature_cfg.episode_id_column_candidates)
        athlete_col = self._metadata_column(df, ("metadata__athlete_name", "фио борца", "athlete_name"))
        sheet_col = self._metadata_column(df, ("metadata__sheet", "_sheet", "sheet"))

        engineered["duration"] = pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0) if duration_col else 0.0
        trace_rows.append(
            {
                "engineered_feature": "duration",
                "encoding_mode": "numeric_passthrough",
                "source_columns": json.dumps([duration_col] if duration_col else [], ensure_ascii=False),
                "source_group": "metadata",
                "source_count": 1 if duration_col else 0,
            }
        )

        engineered["pause"] = pd.to_numeric(df[pause_col], errors="coerce").fillna(0.0) if pause_col else 0.0
        trace_rows.append(
            {
                "engineered_feature": "pause",
                "encoding_mode": "numeric_passthrough",
                "source_columns": json.dumps([pause_col] if pause_col else [], ensure_ascii=False),
                "source_group": "metadata",
                "source_count": 1 if pause_col else 0,
            }
        )

        engineered["observed_result"] = pd.to_numeric(df[result_col], errors="coerce").fillna(0.0) if result_col else 0.0
        trace_rows.append(
            {
                "engineered_feature": "observed_result",
                "encoding_mode": "numeric_passthrough",
                "source_columns": json.dumps([result_col] if result_col else [], ensure_ascii=False),
                "source_group": "outcomes",
                "source_count": 1 if result_col else 0,
            }
        )

        metadata = pd.DataFrame(index=df.index)
        metadata["episode_id"] = (
            df[episode_col].astype(str)
            if episode_col
            else pd.Series(df.index.astype(str), index=df.index)
        )
        if athlete_col:
            metadata["athlete_name"] = df[athlete_col].astype(str)
        if sheet_col:
            metadata["source_sheet"] = df[sheet_col].astype(str)
            metadata["sequence_id"] = df[sheet_col].astype(str)
        elif "athlete_name" in metadata.columns:
            metadata["sequence_id"] = metadata["athlete_name"].astype(str)
        else:
            metadata["sequence_id"] = pd.Series(["sequence_0"] * len(df), index=df.index)

        raw_cols = sorted(set(
            maneuver_all +
            kfv_cols +
            vup_cols +
            outcomes_cols +
            ([duration_col] if duration_col else []) +
            ([pause_col] if pause_col else []) +
            ([result_col] if result_col else []) +
            ([episode_col] if episode_col else []) +
            ([athlete_col] if athlete_col else []) +
            ([sheet_col] if sheet_col else [])
        ))
        raw_feature_set = df[raw_cols].copy() if raw_cols else pd.DataFrame(index=df.index)

        traceability = pd.DataFrame(trace_rows)

        logger.info(
            "Feature engineering completed: raw_features=%d, engineered_features=%d",
            raw_feature_set.shape[1], engineered.shape[1],
        )

        return FeatureEngineeringResult(
            raw_feature_set=raw_feature_set,
            engineered_feature_set=engineered,
            metadata=metadata,
            traceability=traceability,
            validation=validation,
        )


def export_feature_sets(
    result: FeatureEngineeringResult,
    output_dir: str | Path,
    save_parquet: bool = False,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_path = out / "raw_feature_set.csv"
    eng_path = out / "engineered_feature_set.csv"
    trace_path = out / "feature_traceability.csv"
    val_path = out / "feature_validation.json"

    result.raw_feature_set.to_csv(raw_path, index=False)
    result.engineered_feature_set.to_csv(eng_path, index=False)
    result.traceability.to_csv(trace_path, index=False)
    val_path.write_text(json.dumps(result.validation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    outputs = {
        "raw_feature_set_csv": str(raw_path),
        "engineered_feature_set_csv": str(eng_path),
        "traceability_csv": str(trace_path),
        "validation_json": str(val_path),
    }

    if save_parquet:
        try:
            raw_parquet = out / "raw_feature_set.parquet"
            eng_parquet = out / "engineered_feature_set.parquet"
            result.raw_feature_set.to_parquet(raw_parquet, index=False)
            result.engineered_feature_set.to_parquet(eng_parquet, index=False)
            outputs["raw_feature_set_parquet"] = str(raw_parquet)
            outputs["engineered_feature_set_parquet"] = str(eng_parquet)
        except Exception:
            pass

    return outputs
