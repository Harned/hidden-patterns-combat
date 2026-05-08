"""Разбиение листа `Общее` на поединки (bout) и эпизоды (TASK_SPEC_011).

Контракт:

* читаем лист с 3-уровневой шапкой и flatten-колонками
  (см. :func:`hpc_algo.mapping.flatten_columns`);
* идём по строкам сверху вниз, выделяем bout'ы по двум маркерам:
  полностью пустая строка-разделитель и сброс ``№ эпизода`` к меньшему
  значению;
* нормализуем значения признаковых колонок: ``1`` / ``2`` оставляем,
  всё остальное (>2, < 0, дробное, нечисловое, пустое) приводим к ``0``
  и фиксируем :class:`MarkovWarning` ``data.value_out_of_range`` /
  ``data.non_numeric_feature``.

Модуль не определяет состояние эпизода — этим занимается
:mod:`hpc_algo.markov_individual`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from hpc_algo.mapping import flatten_columns, guess_header_rows
from hpc_algo.schema import MarkovWarning

_ATHLETE_HINTS: tuple[str, ...] = ("фио", "борц", "спортсм", "атлет", "боец")
_EPISODE_NUM_HINTS: tuple[str, ...] = ("№ эпизод", "номер эпизод", "no эпизод", "n эпизод")
_EPISODE_TIME_HINTS: tuple[str, ...] = ("время эпизод",)
_PAUSE_TIME_HINTS: tuple[str, ...] = ("время паузы",)
_SCORE_HINTS: tuple[str, ...] = ("баллы",)


@dataclass(frozen=True)
class BaseColumns:
    """Имена «служебных» колонок после flatten."""

    athlete: str | None
    episode_num: str | None
    episode_time: str | None
    pause_time: str | None
    score: str | None


@dataclass
class RawEpisode:
    """Одна строка листа после нормализации.

    «Сырая» в смысле «состояние ещё не назначено». Дальше передаётся в
    :func:`hpc_algo.markov_individual.build_episode_sequence`.
    """

    row_index: int
    bout_id: str
    episode_idx_in_bout: int
    athlete: str
    episode_num_raw: float | None
    episode_time: float | None
    pause_time: float | None
    score: float | None
    feature_values: dict[str, float]
    warnings: list[MarkovWarning] = field(default_factory=list)


def _find_column(flat_columns: Iterable[str], hints: tuple[str, ...]) -> str | None:
    """Найти первую flatten-колонку, чьё имя содержит один из ``hints`` (lower)."""

    columns = list(flat_columns)
    lowered = [(c, c.lower()) for c in columns]
    for hint in hints:
        h = hint.lower()
        for original, low in lowered:
            if h in low:
                return original
    return None


def detect_base_columns(flat_columns: Iterable[str]) -> BaseColumns:
    """Эвристически найти служебные колонки (athlete / episode num / times / score)."""

    cols = list(flat_columns)
    return BaseColumns(
        athlete=_find_column(cols, _ATHLETE_HINTS),
        episode_num=_find_column(cols, _EPISODE_NUM_HINTS),
        episode_time=_find_column(cols, _EPISODE_TIME_HINTS),
        pause_time=_find_column(cols, _PAUSE_TIME_HINTS),
        score=_find_column(cols, _SCORE_HINTS),
    )


def read_episodes_sheet(
    path: str | Path,
    sheet: str = "Общее",
    header_rows: tuple[int, ...] | None = (0, 1, 2),
) -> pd.DataFrame:
    """Прочитать лист с многострочной шапкой и привести колонки к flatten-виду.

    Если ``header_rows`` равен ``None``, заголовки определяются эвристически
    через :func:`hpc_algo.mapping.guess_header_rows` (учитывает возможный
    пустой первый ряд и плавающее количество уровней).
    """

    if header_rows is None:
        raw = pd.read_excel(
            path,
            sheet_name=sheet,
            header=None,
            engine="openpyxl",
            nrows=8,
        )
        header_rows = tuple(guess_header_rows(raw))

    df = pd.read_excel(
        path,
        sheet_name=sheet,
        header=list(header_rows),
        engine="openpyxl",
    )
    df.columns = flatten_columns(df.columns)
    return df


def _is_blank_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return bool(isinstance(value, str) and not value.strip())


def _is_blank_row(row: pd.Series) -> bool:
    return all(_is_blank_value(v) for v in row.tolist())


def _coerce_optional_float(value: object) -> float | None:
    if _is_blank_value(value):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_feature_value(value: object) -> tuple[float, MarkovWarning | None]:
    """Привести значение признака к ``{0.0, 1.0, 2.0}`` с защитой ``>2 → log&skip``.

    Возвращает (normalized_value, warning_or_None). Любое отклонение от
    ожидаемого алфавита превращается в 0.0 + warning.
    """

    if _is_blank_value(value):
        return 0.0, None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0, MarkovWarning(
            code="data.non_numeric_feature",
            message=f"Non-numeric value in feature column: {value!r}",
            context={"value": str(value)},
        )

    if f in (0.0, 1.0, 2.0):
        return f, None

    return 0.0, MarkovWarning(
        code="data.value_out_of_range",
        message=f"Feature value out of {{0,1,2}}: {f}",
        context={"value": f},
    )


def split_into_bouts_and_episodes(
    df: pd.DataFrame,
    base: BaseColumns,
    feature_columns: Iterable[str],
) -> tuple[list[RawEpisode], list[MarkovWarning]]:
    """Пройти по DataFrame и собрать список нормализованных эпизодов.

    Правила границы bout'а:

    * полностью пустая строка между поединками закрывает текущий bout;
    * сброс ``№ эпизода`` к строго меньшему значению
      (``ep_num < last_ep_num``) открывает новый bout. Используется
      строгое неравенство, т.к. при интерливинге двух участников
      ``ep_num`` повторяется (``1, 1, 2, 2, …``) — это один bout, а не
      несколько;
    * пустое ``Время паузы`` у эпизода — индикативный признак «последний
      в bout», но сам по себе bout не закрывает (закрытие происходит на
      ближайшей пустой строке или при сбросе ``№ эпизода``).

    Замечание о layout. Если файл хранит «все строки одного спортсмена
    подряд, затем все строки другого спортсмена того же поединка»
    (``А-ep1, А-ep2, А-ep3, Б-ep1, Б-ep2, Б-ep3``), то правило строгого
    сброса само по себе не отделит участников одного bout'а от
    следующего: правильную границу должна задать пустая строка.

    Транзиции внутри bout считаются вызывающим кодом
    (:mod:`hpc_algo.markov_individual`); этот модуль гарантирует только
    корректное наклеивание ``bout_id``.
    """

    feature_cols = list(feature_columns)
    global_warnings: list[MarkovWarning] = []
    if base.episode_num is None:
        global_warnings.append(
            MarkovWarning(
                code="episode_split.missing_episode_column",
                message=(
                    "Не найдена колонка с № эпизода; разбиение на bout'ы по"
                    " правилу сброса работать не будет."
                ),
            )
        )

    episodes: list[RawEpisode] = []
    bout_counter = 0
    current_bout_id: str | None = None
    last_episode_num: float | None = None
    current_bout_episode_idx = 0
    current_athlete: str = ""

    def open_new_bout() -> str:
        nonlocal bout_counter, current_bout_id, last_episode_num, current_bout_episode_idx
        bout_counter += 1
        current_bout_id = f"bout_{bout_counter}"
        last_episode_num = None
        current_bout_episode_idx = 0
        return current_bout_id

    for row_idx, row in df.iterrows():
        if _is_blank_row(row):
            current_bout_id = None
            last_episode_num = None
            current_athlete = ""
            continue

        ep_num_raw = (
            row.get(base.episode_num) if base.episode_num is not None else None
        )
        ep_num: float | None = _coerce_optional_float(ep_num_raw)

        if current_bout_id is None or (
            ep_num is not None
            and last_episode_num is not None
            and ep_num < last_episode_num
        ):
            open_new_bout()

        current_bout_episode_idx += 1

        # ФИО хранится в merged-cell: непустое значение появляется только на
        # первой строке блока спортсмена. Внутри bout'а forward-fill'им
        # последнее увиденное имя; на blank-row сбрасываем (см. continue выше).
        if base.athlete is not None:
            v = row.get(base.athlete)
            if not _is_blank_value(v):
                current_athlete = str(v).strip()
        athlete = current_athlete

        ep_time = (
            _coerce_optional_float(row.get(base.episode_time))
            if base.episode_time is not None
            else None
        )
        pause_time = (
            _coerce_optional_float(row.get(base.pause_time))
            if base.pause_time is not None
            else None
        )
        score = (
            _coerce_optional_float(row.get(base.score))
            if base.score is not None
            else None
        )

        feature_values: dict[str, float] = {}
        per_row_warnings: list[MarkovWarning] = []
        for col in feature_cols:
            normalized, w = _normalize_feature_value(row.get(col))
            if w is not None:
                w.context.setdefault("row_index", int(row_idx))
                w.context.setdefault("column", col)
                per_row_warnings.append(w)
            feature_values[col] = normalized

        episodes.append(
            RawEpisode(
                row_index=int(row_idx),
                bout_id=current_bout_id or open_new_bout(),
                episode_idx_in_bout=current_bout_episode_idx,
                athlete=athlete,
                episode_num_raw=ep_num,
                episode_time=ep_time,
                pause_time=pause_time,
                score=score,
                feature_values=feature_values,
                warnings=per_row_warnings,
            )
        )

        last_episode_num = ep_num

    flat_warnings: list[MarkovWarning] = list(global_warnings)
    for ep in episodes:
        flat_warnings.extend(ep.warnings)

    return episodes, flat_warnings
