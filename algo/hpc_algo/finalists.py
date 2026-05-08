"""Список призёров 1–3 каждой весовой категории (TASK_SPEC_012).

Источник истины — внешний YAML (по умолчанию ``config/finalists.yaml``):

    version: 1
    weight_classes:
      "48":
        1: "Иванов И. И."
        2: "Петров П. П."
        3: "Сидоров С. С."
      "52":
        1: "..."
        2: "..."
        3: "..."

Если YAML отсутствует, в качестве fallback используется эвристическая
детекция колонок ``Весовая категория`` и ``Место`` в Excel
(``finalists_from_episodes``). Это работает только если такие колонки
действительно есть в листе. На практике чище держать призёров в
явном YAML — он короткий и не зависит от неоднозначностей реального
файла.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from hpc_algo.episode_split import RawEpisode
from hpc_algo.schema import FinalistEntry, MarkovWarning


def load_finalists_yaml(
    path: str | Path,
) -> tuple[list[FinalistEntry], list[MarkovWarning]]:
    """Толерантный загрузчик списка призёров.

    Контракт:

    * формат — ``{weight_classes: {wc: {place: athlete}}}``;
      допускается также плоский ``{wc: {place: athlete}}``;
    * ``place`` приводится к ``int``; неизвестное / нечисловое →
      warning ``finalists.invalid_place``, запись игнорируется;
    * пустое имя спортсмена → warning ``finalists.empty_athlete``,
      пропуск;
    * любые проблемы — warning, не исключение, чтобы оркестратор мог
      честно отрапортовать пустую категорию.
    """

    p = Path(path)
    raw_text = p.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text)

    warnings: list[MarkovWarning] = []
    if raw is None:
        return [], warnings
    if not isinstance(raw, dict):
        raise ValueError(
            f"finalists.yaml must be a YAML mapping, got: {type(raw).__name__}"
        )

    body: Any = (
        raw["weight_classes"]
        if "weight_classes" in raw and isinstance(raw["weight_classes"], dict)
        else raw
    )

    if not isinstance(body, dict):
        warnings.append(
            MarkovWarning(
                code="finalists.invalid_root",
                message="weight_classes must be a YAML mapping; configuration ignored.",
            )
        )
        return [], warnings

    out: list[FinalistEntry] = []
    for wc_raw, athletes_block in body.items():
        wc = str(wc_raw)
        if not isinstance(athletes_block, dict):
            warnings.append(
                MarkovWarning(
                    code="finalists.invalid_class_block",
                    message=(
                        f"Класс {wc!r} должен содержать mapping place->athlete; "
                        "пропуск."
                    ),
                    context={"weight_class": wc},
                )
            )
            continue
        for place_raw, athlete_raw in athletes_block.items():
            try:
                place = int(place_raw)
            except (TypeError, ValueError):
                warnings.append(
                    MarkovWarning(
                        code="finalists.invalid_place",
                        message=(
                            f"Неверное место {place_raw!r} в категории {wc!r}; пропуск."
                        ),
                        context={"weight_class": wc, "place": str(place_raw)},
                    )
                )
                continue
            athlete = str(athlete_raw or "").strip()
            if not athlete:
                warnings.append(
                    MarkovWarning(
                        code="finalists.empty_athlete",
                        message=(
                            f"Пустое имя для места {place} в категории {wc!r}; пропуск."
                        ),
                        context={"weight_class": wc, "place": place},
                    )
                )
                continue
            out.append(
                FinalistEntry(athlete=athlete, place=place, weight_class=wc)
            )
    return out, warnings


_WEIGHT_HINTS: tuple[str, ...] = ("весов", "категори", "weight")
_PLACE_HINTS: tuple[str, ...] = ("место", "place", "медал", "призов")


def detect_meta_columns(
    flat_columns: Iterable[str],
) -> tuple[str | None, str | None]:
    """Эвристически найти колонки ``Весовая категория`` и ``Место``.

    Возвращает кортеж ``(weight_class_col, place_col)``; ``None`` если
    подходящая колонка не найдена. Для ``place`` намеренно ищем
    разные варианты подсказок (`Место`, `Медаль`, `Призовое`), потому
    что в реальных файлах оформление непредсказуемое.
    """

    columns = list(flat_columns)
    lowered = [(c, c.lower()) for c in columns]

    def _match(hints: tuple[str, ...]) -> str | None:
        for h in hints:
            for original, low in lowered:
                if h in low:
                    return original
        return None

    return _match(_WEIGHT_HINTS), _match(_PLACE_HINTS)


def finalists_from_episodes(
    raw_episodes: list[RawEpisode],
    df_columns: Iterable[str],
    df_rows: list[dict[str, Any]],
) -> tuple[list[FinalistEntry], list[MarkovWarning]]:
    """Извлечь финалистов прямо из Excel, если есть нужные колонки.

    Сейчас реализован самый осторожный вариант: ищем колонки
    ``Весовая категория`` и ``Место``, и для каждого спортсмена
    берём первое непустое значение каждой колонки. Если колонки нет —
    возвращаем пустой список + warning, чтобы оркестратор пошёл по
    YAML-пути.

    ``df_rows`` — список dict-row из исходного DataFrame, в той же
    последовательности, что и ``raw_episodes``. Это нужно, чтобы
    избежать пере-импорта pandas в эту функцию.
    """

    weight_col, place_col = detect_meta_columns(df_columns)
    warnings: list[MarkovWarning] = []
    if weight_col is None or place_col is None:
        warnings.append(
            MarkovWarning(
                code="finalists.columns_missing",
                message=(
                    "В листе не найдены колонки 'Весовая категория' и/или 'Место'."
                    " Используйте config/finalists.yaml."
                ),
                context={
                    "weight_col_found": weight_col is not None,
                    "place_col_found": place_col is not None,
                },
            )
        )
        return [], warnings

    by_athlete: dict[str, dict[str, Any]] = {}
    for ep, row in zip(raw_episodes, df_rows, strict=False):
        if not ep.athlete:
            continue
        slot = by_athlete.setdefault(
            ep.athlete, {"weight_class": None, "place": None}
        )
        if slot["weight_class"] is None:
            v = row.get(weight_col)
            if v is not None and str(v).strip():
                slot["weight_class"] = str(v).strip()
        if slot["place"] is None:
            v = row.get(place_col)
            if v is not None and str(v).strip():
                with contextlib.suppress(TypeError, ValueError):
                    slot["place"] = int(float(v))

    out: list[FinalistEntry] = []
    for athlete, slot in by_athlete.items():
        if slot["weight_class"] is None or slot["place"] is None:
            continue
        if slot["place"] < 1:
            continue
        out.append(
            FinalistEntry(
                athlete=athlete,
                place=int(slot["place"]),
                weight_class=str(slot["weight_class"]),
            )
        )
    if not out:
        warnings.append(
            MarkovWarning(
                code="finalists.empty_after_extraction",
                message=(
                    "Колонки 'Весовая категория'/'Место' нашлись, но непустых пар"
                    " (athlete, place, weight_class) собрать не удалось."
                ),
            )
        )
    return out, warnings
