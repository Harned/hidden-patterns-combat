"""Column mapping: flatten multi-row headers и preflight-эвристика.

Модуль принципиально не решает, какая колонка относится к какой роли.
Он предлагает кандидатов и помогает пользователю сформировать
:class:`hpc_algo.schema.ColumnMappingConfig`. Подтверждение — за человеком.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from hpc_algo.schema import (
    ColumnMappingConfig,
    HiddenGroup,
    SheetMapping,
)
from hpc_algo.text_utils import normalize_header

_FLATTEN_SEP = " | "


def _clean_level(value: object) -> str:
    """Очистить один уровень заголовка от 'Unnamed:' / NaN / пустоты."""

    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.startswith("Unnamed:"):
        return ""
    return s


def column_levels(column: object, idx: int) -> list[str]:
    """Вернуть очищенные уровни заголовка одной колонки.

    Для одиночного заголовка — список из одного элемента. Для MultiIndex —
    список непустых уровней (с устранением подряд идущих дубликатов,
    возникающих из merged-ячеек).
    """

    if isinstance(column, tuple):
        parts = [_clean_level(x) for x in column]
    else:
        parts = [_clean_level(column)]
    non_empty = [p for p in parts if p]
    if not non_empty:
        return [f"column_{idx}"]
    dedup: list[str] = []
    for p in non_empty:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    return dedup


def flatten_columns(columns: Iterable[object]) -> list[str]:
    """Превратить (возможно) MultiIndex-колонки в плоский ``list[str]``."""

    result: list[str] = []
    for idx, col in enumerate(columns):
        levels = column_levels(col, idx)
        result.append(_FLATTEN_SEP.join(levels))
    return result


# ---------------------------------------------------------------------------
# Эвристическое распределение колонок по ролям (preflight)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    group: HiddenGroup
    markers: tuple[str, ...]


# Служебные роли: проверяются только по *атомарному* (последнему непустому)
# уровню заголовка. Это защищает от ложных срабатываний, когда super-заголовок
# (например, «Технико-тактический эпизод») через merged-ячейки наследуется
# на другие колонки.
_SERVICE_RULES: tuple[_Rule, ...] = (
    _Rule(HiddenGroup.ATHLETE, ("фио", "спортсмен", "борец", "атлет", "боец", "борца")),
    # TIME идёт раньше EPISODE: маркер "время эпизода" перехватывает
    # колонку «Время эпизода, с.», пока подстрока "эпизода" не успела
    # сработать в правиле EPISODE.
    _Rule(HiddenGroup.TIME, ("время эпизода", "время паузы", "длительн", "секунд")),
    # `№` после NFKC нормализации становится `no`. Добавлены оба варианта.
    _Rule(
        HiddenGroup.EPISODE,
        ("no эпизода", "номер эпизода", "эпизода", "№ эпизода"),
    ),
    _Rule(
        HiddenGroup.BOUT,
        ("no схватки", "номер схватки", "схватки", "схватка", "поединок"),
    ),
    _Rule(HiddenGroup.WEIGHT, ("весовая категория", "вес кг", "категория")),
)


# Предметные роли: проверяются по всем уровням заголовка. Порядок перечисления
# задаёт приоритет (сильнее — раньше).
_DOMAIN_RULES: tuple[_Rule, ...] = (
    _Rule(HiddenGroup.ZAP, ("зап", "удержание", "болев")),
    _Rule(HiddenGroup.VUP, ("вуп", "выведение")),
    _Rule(HiddenGroup.KFV, ("кфв", "захват", "обхват", "прихват", "упор", " хват")),
    _Rule(HiddenGroup.MANEUVERING, ("маневр", "стойк")),
)


def _assign_role(levels: list[str]) -> HiddenGroup | None:
    """Определить предполагаемую роль на основе уровней заголовка."""

    if not levels:
        return None

    atomic_norm = normalize_header(levels[-1])
    if atomic_norm:
        for rule in _SERVICE_RULES:
            if any(m in atomic_norm for m in rule.markers):
                return rule.group

    full_norm = " | ".join(normalize_header(lvl) for lvl in levels if lvl)
    for rule in _DOMAIN_RULES:
        if any(m in full_norm for m in rule.markers):
            return rule.group

    return None


# ---------------------------------------------------------------------------
# Preflight over a single sheet
# ---------------------------------------------------------------------------


def guess_header_rows(df_raw: pd.DataFrame, max_rows: int = 4) -> list[int]:
    """Подобрать строки-заголовки эвристически.

    Идея: заголовки — это строки, где преобладают строковые значения и
    много пропусков (из-за merged-cells), а данные — строки с числами.
    Возвращаем первые подряд идущие «headerish» строки.
    """

    header_rows: list[int] = []
    for i in range(min(max_rows, len(df_raw))):
        row = df_raw.iloc[i]
        non_null = row.dropna()
        if non_null.empty:
            continue
        strings = sum(1 for v in non_null if isinstance(v, str))
        numerics = sum(
            1
            for v in non_null
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        if strings >= numerics and strings >= 1:
            header_rows.append(i)
        else:
            break
    return header_rows or [0]


def read_sheet_with_header_rows(
    path,
    sheet_name: str,
    header_rows: list[int],
) -> pd.DataFrame:
    """Прочитать лист с заданными header-строками и применить flatten."""

    header = header_rows if len(header_rows) > 1 else (header_rows[0] if header_rows else 0)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header, engine="openpyxl")

    # Сохраняем уровни через _flatten_levels_cache для будущего reuse — но
    # pandas после assignment теряет MultiIndex, поэтому уровни восстановим
    # из raw-структуры на стороне preflight.
    df.columns = flatten_columns(df.columns)
    return df


def _extract_levels_from_multiindex(columns) -> list[list[str]]:
    """Получить уровни заголовка для каждой колонки из pandas Index/MultiIndex."""

    levels_per_col: list[list[str]] = []
    for idx, col in enumerate(columns):
        levels_per_col.append(column_levels(col, idx))
    return levels_per_col


def preflight_sheet(
    path,
    sheet_name: str,
    header_rows: list[int] | None = None,
) -> tuple[list[int], pd.DataFrame, SheetMapping]:
    """Выполнить preflight одного листа.

    Возвращает:
    * ``header_rows`` — решение о заголовках (из аргумента или угаданное);
    * ``df`` — DataFrame с плоскими именами колонок;
    * ``SheetMapping`` — предлагаемое распределение ролей.
    """

    if header_rows is None:
        raw = pd.read_excel(
            path, sheet_name=sheet_name, header=None, engine="openpyxl"
        )
        header_rows = guess_header_rows(raw)

    # Читаем лист с MultiIndex, чтобы сохранить уровни для эвристики ролей.
    header_arg = (
        header_rows if len(header_rows) > 1 else (header_rows[0] if header_rows else 0)
    )
    raw_df = pd.read_excel(
        path, sheet_name=sheet_name, header=header_arg, engine="openpyxl"
    )
    levels_per_col = _extract_levels_from_multiindex(raw_df.columns)
    flat_names = [" | ".join(lv) for lv in levels_per_col]
    raw_df.columns = flat_names

    roles: dict[HiddenGroup, list[str]] = {}
    for flat_name, levels in zip(flat_names, levels_per_col, strict=True):
        role = _assign_role(levels)
        if role is None:
            continue
        roles.setdefault(role, []).append(flat_name)

    mapping = SheetMapping(
        header_rows=list(header_rows),
        data_start_row=None,
        roles=roles,
    )
    return header_rows, raw_df, mapping


def preflight(path, sheet_names: list[str] | None = None) -> ColumnMappingConfig:
    """Полный preflight по всем листам."""

    xl = pd.ExcelFile(path, engine="openpyxl")
    names = sheet_names if sheet_names is not None else xl.sheet_names

    sheets: dict[str, SheetMapping] = {}
    for name in names:
        try:
            _, _, sm = preflight_sheet(path, name)
        except Exception:  # noqa: BLE001 — отдельный лист не должен ронять preflight
            sheets[name] = SheetMapping(header_rows=[0], roles={})
            continue
        sheets[name] = sm
    return ColumnMappingConfig(version="1", sheets=sheets)


# ---------------------------------------------------------------------------
# Применение mapping к загруженному Excel
# ---------------------------------------------------------------------------


def describe_sheet_columns(
    path,
    sheet_name: str,
    header_rows: list[int] | None = None,
    sample_size: int = 5,
) -> tuple[list[int], list[dict[str, object]]]:
    """Вернуть полный список колонок листа с flatten-именами и примерами.

    * ``header_rows`` — если ``None``, подбирается эвристически.
    * Для каждой колонки возвращается ``name`` (flatten),
      ``levels`` (список уровней), ``dtype``, ``non_null_count``,
      ``sample_values`` (до ``sample_size`` непустых уникальных значений),
      ``role_hint`` — предполагаемая роль на основе эвристики
      :func:`_assign_role` (``None``, если эвристика не сработала).

    Возвращает ``(header_rows_used, columns)``.
    """

    if header_rows is None:
        raw = pd.read_excel(
            path, sheet_name=sheet_name, header=None, engine="openpyxl"
        )
        header_rows = guess_header_rows(raw)

    header_arg = (
        header_rows if len(header_rows) > 1 else (header_rows[0] if header_rows else 0)
    )
    df = pd.read_excel(
        path, sheet_name=sheet_name, header=header_arg, engine="openpyxl"
    )
    levels_per_col = [
        column_levels(col, idx) for idx, col in enumerate(df.columns)
    ]
    flat_names = [" | ".join(lv) for lv in levels_per_col]
    df.columns = flat_names

    result: list[dict[str, object]] = []
    for flat_name, levels in zip(flat_names, levels_per_col, strict=True):
        series = df[flat_name]
        role = _assign_role(levels)
        non_null = series.dropna()
        # Не тащим не-JSON-совместимые значения (pd.Timestamp и т.п.).
        samples: list[object] = []
        for v in non_null.drop_duplicates().head(sample_size).tolist():
            if isinstance(v, (pd.Timestamp,)):
                samples.append(v.isoformat())
            else:
                samples.append(v if isinstance(v, (str, int, float, bool)) else str(v))
        result.append(
            {
                "name": flat_name,
                "levels": levels,
                "dtype": str(series.dtype),
                "non_null_count": int(non_null.count()),
                "null_count": int(series.isna().sum()),
                "sample_values": samples,
                "role_hint": role.value if role is not None else None,
            }
        )
    return list(header_rows), result


def load_mapped_sheets(
    path,
    config: ColumnMappingConfig,
) -> dict[str, pd.DataFrame]:
    """Прочитать все листы из config с их header_rows и flatten-именами."""

    frames: dict[str, pd.DataFrame] = {}
    for sheet_name, sm in config.sheets.items():
        try:
            df = read_sheet_with_header_rows(path, sheet_name, sm.header_rows)
        except Exception:  # noqa: BLE001
            continue
        if sm.data_start_row is not None:
            # pd.read_excel уже использует header, но если пользователь настоял
            # на data_start_row — срежем дополнительно (относительно позиции после header).
            first_header = max(sm.header_rows) if sm.header_rows else 0
            offset = sm.data_start_row - first_header - 1
            if offset > 0:
                df = df.iloc[offset:].reset_index(drop=True)
        frames[sheet_name] = df
    return frames
