from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


_EPISODE_MARKER_HINTS = (
    "№ эпизода",
    "номер эпизода",
    "эпизод №",
    "episode id",
    "episode number",
)


def normalize_label_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower().replace("ё", "е")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^0-9a-zа-я№#\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_empty(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none"}


def _is_episode_marker(text: str) -> bool:
    if not text:
        return False
    if any(hint in text for hint in _EPISODE_MARKER_HINTS):
        return True
    return "эпизод" in text and ("№" in text or "номер" in text or "#" in text)


def _is_episode_id_like(value: object) -> bool:
    if _is_empty(value):
        return False
    text = str(value).strip()
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return True
    return bool(re.fullmatch(r"[a-zA-Z]*\d+[a-zA-Z]*", text))


def detect_matrix_episode_sheet(raw_df: pd.DataFrame) -> bool:
    if raw_df.empty:
        return False

    scan_rows = min(20, raw_df.shape[0])
    scan_cols = min(8, raw_df.shape[1])

    for r in range(scan_rows):
        for c in range(scan_cols):
            text = normalize_label_text(raw_df.iat[r, c])
            if not _is_episode_marker(text):
                continue

            episode_cols = [j for j in range(c + 1, raw_df.shape[1]) if _is_episode_id_like(raw_df.iat[r, j])]
            if len(episode_cols) < 2:
                continue

            informative_rows = 0
            for rr in range(r + 1, min(raw_df.shape[0], r + 150)):
                has_label = any(not _is_empty(raw_df.iat[rr, cc]) for cc in range(c + 1))
                has_data = any(not _is_empty(raw_df.iat[rr, ec]) for ec in episode_cols)
                if has_label and has_data:
                    informative_rows += 1
            if informative_rows >= 3:
                return True

    return False


def normalize_matrix_feature_label(raw_label: str) -> str:
    text = normalize_label_text(raw_label)

    if _is_episode_marker(text):
        return "episode_id"
    if "время" in text and "пауз" in text:
        return "pause_duration"
    if ("время" in text and "эпизод" in text) or "длительность" in text:
        return "episode_duration"
    if any(token in text for token in ("балл", "очк", "результ", "score", "points")):
        return "observed_result"

    if any(token in text for token in ("правосторон", "правой стойк", "стойка прав", "right")):
        return "maneuver_right_indicator"
    if any(token in text for token in ("левосторон", "левой стойк", "стойка лев", "left")):
        return "maneuver_left_indicator"

    if "обхват" in text:
        return "kfv_bodylocks_indicator"
    if "прихват" in text:
        return "kfv_underhooks_indicator"
    if "захват" in text:
        return "kfv_grips_indicator"
    if "упор" in text:
        return "kfv_posts_indicator"
    if "хват" in text:
        return "kfv_holds_indicator"

    if any(token in text for token in ("вуп", "выведен", "vup")):
        return "vup_indicator"

    if any(token in text for token in ("заверша", "атак", "прием", "finish", "outcome")):
        return "outcome_action_indicator"

    return "unknown_indicator"


def _to_binary(value: object) -> int:
    if _is_empty(value):
        return 0

    text = normalize_label_text(value)
    if text in {"да", "yes", "true", "истина", "1"}:
        return 1
    if text in {"нет", "no", "false", "0"}:
        return 0

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return int(float(numeric) > 0)
    return 0


def _to_float(value: object) -> float:
    if _is_empty(value):
        return 0.0
    text = str(value).strip().replace(",", ".")
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else 0.0


def _compose_label(cells: list[object]) -> str:
    parts: list[str] = []
    prev = ""
    for value in cells:
        text = normalize_label_text(value)
        if not text:
            continue
        if text == prev:
            continue
        parts.append(text)
        prev = text
    return " | ".join(parts)


def _find_episode_axis(raw_df: pd.DataFrame) -> tuple[int, int, list[int]]:
    for r in range(raw_df.shape[0]):
        for c in range(raw_df.shape[1]):
            text = normalize_label_text(raw_df.iat[r, c])
            if not _is_episode_marker(text):
                continue
            episode_cols = [j for j in range(c + 1, raw_df.shape[1]) if _is_episode_id_like(raw_df.iat[r, j])]
            if len(episode_cols) >= 1:
                return r, c, episode_cols
    raise ValueError("Could not locate episode axis marker (e.g., '№ эпизода').")


def _extract_athlete_name(raw_df: pd.DataFrame, marker_col: int, marker_row: int) -> str | None:
    search_rows = min(raw_df.shape[0], max(marker_row + 1, 12))
    search_cols = min(raw_df.shape[1], max(marker_col + 3, 4))

    for r in range(search_rows):
        for c in range(search_cols):
            text = normalize_label_text(raw_df.iat[r, c])
            if not any(token in text for token in ("фио", "athlete", "fighter", "борца")):
                continue
            for cc in range(c + 1, min(raw_df.shape[1], c + 6)):
                val = raw_df.iat[r, cc]
                if _is_empty(val):
                    continue
                val_text = str(val).strip()
                if normalize_label_text(val_text) == text:
                    continue
                return val_text
    return None


@dataclass
class MatrixParseResult:
    tidy: pd.DataFrame
    label_mapping: pd.DataFrame
    assumptions: list[str]


def load_matrix_episode_sheet(raw_df: pd.DataFrame, sheet_name: str) -> MatrixParseResult:
    if raw_df.empty:
        return MatrixParseResult(tidy=pd.DataFrame(), label_mapping=pd.DataFrame(), assumptions=["empty_sheet"])

    work = raw_df.copy().dropna(axis=0, how="all").dropna(axis=1, how="all")
    marker_row, marker_col, episode_cols = _find_episode_axis(work)

    label_view = work.copy()
    label_view.iloc[:, : marker_col + 1] = label_view.iloc[:, : marker_col + 1].ffill(axis=0)

    episode_ids_raw = [work.iat[marker_row, c] for c in episode_cols]
    episode_ids: list[str] = []
    for idx, value in enumerate(episode_ids_raw, start=1):
        if _is_empty(value):
            episode_ids.append(str(idx))
        else:
            episode_ids.append(str(value).strip())

    athlete_name = _extract_athlete_name(work, marker_col=marker_col, marker_row=marker_row)

    rows: list[dict[str, object]] = []
    for i, episode_id in enumerate(episode_ids, start=1):
        row = {
            "source_sheet": sheet_name,
            "episode_order": i,
            "episode_id": episode_id,
        }
        if athlete_name:
            row["athlete_name"] = athlete_name
        rows.append(row)

    label_mapping_rows: list[dict[str, object]] = []
    counters: dict[str, int] = {}

    for r in range(marker_row + 1, work.shape[0]):
        source_label = _compose_label(label_view.iloc[r, : marker_col + 1].tolist())
        if not source_label:
            continue

        base_name = normalize_matrix_feature_label(source_label)
        if base_name == "episode_id":
            continue

        values = [work.iat[r, c] for c in episode_cols]
        if all(_is_empty(v) for v in values):
            continue

        if base_name in {"episode_duration", "pause_duration", "observed_result"}:
            col_name = base_name
            parser_kind = "numeric"
        else:
            counters[base_name] = counters.get(base_name, 0) + 1
            col_name = f"{base_name}_{counters[base_name]:02d}"
            parser_kind = "binary"

        label_mapping_rows.append(
            {
                "source_sheet": sheet_name,
                "source_label": source_label,
                "normalized_label": normalize_label_text(source_label),
                "normalized_column": col_name,
                "parser_kind": parser_kind,
            }
        )

        for i, value in enumerate(values):
            rows[i][col_name] = _to_float(value) if parser_kind == "numeric" else _to_binary(value)

    tidy = pd.DataFrame(rows)

    indicator_cols = [
        c
        for c in tidy.columns
        if c.endswith("_indicator_01") or "_indicator_" in c or c.startswith("unknown_indicator_")
    ]
    for col in indicator_cols:
        tidy[col] = pd.to_numeric(tidy[col], errors="coerce").fillna(0).astype(int)

    for col in ("episode_duration", "pause_duration", "observed_result"):
        if col in tidy.columns:
            tidy[col] = pd.to_numeric(tidy[col], errors="coerce").fillna(0.0)

    preferred = ["source_sheet", "athlete_name", "episode_id", "episode_order", "episode_duration", "pause_duration", "observed_result"]
    dynamic = [c for c in tidy.columns if c not in preferred]
    tidy = tidy[[c for c in preferred if c in tidy.columns] + sorted(dynamic)]

    mapping = pd.DataFrame(label_mapping_rows)
    assumptions = [
        "episode_columns_detected_from_marker_row",
        "binary_indicators_use_nonzero_or_yes_no_coercion",
    ]
    return MatrixParseResult(tidy=tidy, label_mapping=mapping, assumptions=assumptions)
