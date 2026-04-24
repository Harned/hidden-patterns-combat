"""Эвристическая детекция колонок-кандидатов для предметных групп.

Важно: это *только* подсказки, а не решение. Любой обнаруженный кандидат
должен пройти подтверждение (ручное сопоставление / config) до того, как
на нём строятся содержательные выводы.

Эвристика:
    * смотрим на нормализованное имя колонки и на небольшой семпл значений;
    * каждому кандидату присваиваем score ∈ [0, 1];
    * кандидат «группа распознана уверенно» только если score ≥ STRONG_THRESHOLD
      и подтверждён содержимым (для ЗАП — наличие ожидаемых меток).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hpc_algo.loading import LoadedExcel
from hpc_algo.schema import ColumnDetectionReport, HiddenGroup, HiddenGroupCandidate
from hpc_algo.text_utils import normalize_header

# Пороги честности. Ниже STRONG_THRESHOLD мы НЕ объявляем группу распознанной.
_STRONG_THRESHOLD = 0.7
_WEAK_THRESHOLD = 0.4


@dataclass(frozen=True)
class _Rule:
    group: HiddenGroup
    header_markers: tuple[str, ...]
    base_score: float
    rationale: str


# Предметные эвристики. Важно: observations = ЗАП, поэтому ЗАП-колонки
# выделяются отдельно и имеют самые высокие базовые веса.
_HEADER_RULES: tuple[_Rule, ...] = (
    # --- ЗАП (наблюдения) ---
    _Rule(
        group=HiddenGroup.ZAP,
        header_markers=("зап",),
        base_score=0.85,
        rationale="Заголовок содержит маркер 'ЗАП'.",
    ),
    _Rule(
        group=HiddenGroup.ZAP,
        header_markers=("удержание", "болевой"),
        base_score=0.7,
        rationale="Заголовок содержит маркер наблюдаемого судейского результата (удержание/болевой).",
    ),
    # --- Маневрирование ---
    _Rule(
        group=HiddenGroup.MANEUVERING,
        header_markers=("маневр", "стойк"),
        base_score=0.75,
        rationale="Заголовок содержит маркер маневрирования/стойки.",
    ),
    # --- КФВ и его подгруппы ---
    _Rule(
        group=HiddenGroup.KFV,
        header_markers=("кфв",),
        base_score=0.8,
        rationale="Заголовок содержит маркер 'КФВ'.",
    ),
    _Rule(
        group=HiddenGroup.KFV,
        header_markers=("захват", "хват", "обхват", "прихват", "упор"),
        base_score=0.6,
        rationale="Заголовок относится к подгруппе КФВ (захват/хват/обхват/прихват/упор).",
    ),
    # --- ВУП ---
    _Rule(
        group=HiddenGroup.VUP,
        header_markers=("вуп", "выведение"),
        base_score=0.8,
        rationale="Заголовок содержит маркер ВУП / 'выведение'.",
    ),
    # --- Служебные ---
    _Rule(
        group=HiddenGroup.TIME,
        header_markers=("время", "длительн", "секунд"),
        base_score=0.7,
        rationale="Заголовок похож на временной атрибут эпизода.",
    ),
    _Rule(
        group=HiddenGroup.ATHLETE,
        header_markers=("спортсмен", "фамил", "атлет", "боец"),
        base_score=0.7,
        rationale="Заголовок похож на идентификатор спортсмена.",
    ),
    _Rule(
        group=HiddenGroup.EPISODE,
        header_markers=("эпизод",),
        base_score=0.7,
        rationale="Заголовок похож на идентификатор эпизода.",
    ),
    _Rule(
        group=HiddenGroup.BOUT,
        header_markers=("схватк", "поединок", "бой"),
        base_score=0.7,
        rationale="Заголовок похож на идентификатор схватки.",
    ),
    _Rule(
        group=HiddenGroup.WEIGHT,
        header_markers=("вес", "кг", "категори"),
        base_score=0.6,
        rationale="Заголовок похож на весовую категорию.",
    ),
)


# Ожидаемые значения ЗАП. Используются для подтверждения кандидатов.
_ZAP_EXPECTED_TOKENS = (
    "зап-р",
    "зап-н",
    "зап-т",
    "зап р",
    "зап н",
    "зап т",
    "удержан",
    "болев",
)


def _zap_content_support(values: Iterable[Any]) -> float:
    """Во сколько раз содержимое колонки похоже на ЗАП-метки."""

    normalized = [normalize_header(v) for v in values if v is not None]
    normalized = [v for v in normalized if v]
    if not normalized:
        return 0.0
    hits = sum(1 for v in normalized if any(tok in v for tok in _ZAP_EXPECTED_TOKENS))
    return hits / len(normalized)


def _time_content_support(series: pd.Series) -> float:
    """Насколько содержимое колонки похоже на время/длительность."""

    dropped = series.dropna()
    if dropped.empty:
        return 0.0
    numeric = pd.to_numeric(dropped, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    datetime_ratio = 1.0 if np.issubdtype(series.dtype, np.datetime64) else 0.0
    return max(numeric_ratio, datetime_ratio)


def _detect_sheet(sheet_name: str, df: pd.DataFrame) -> list[HiddenGroupCandidate]:
    candidates: list[HiddenGroupCandidate] = []

    for raw_col in df.columns:
        column = str(raw_col)
        header = normalize_header(raw_col)
        if not header:
            continue

        series = df[raw_col]
        sample = series.dropna().head(20).tolist()

        for rule in _HEADER_RULES:
            matched_marker = next(
                (m for m in rule.header_markers if m in header), None
            )
            if matched_marker is None:
                continue

            score = rule.base_score

            # Контент-подтверждение для критичных групп.
            if rule.group == HiddenGroup.ZAP:
                support = _zap_content_support(sample)
                # Сильный boost если в значениях встречаются ожидаемые метки.
                score = min(1.0, score + 0.1 * support)
                if support == 0.0 and matched_marker != "зап":
                    # Маркеры "удержание"/"болевой" без ЗАП-подобного контента — слабее.
                    score *= 0.8
            elif rule.group == HiddenGroup.TIME:
                score = min(1.0, score * (0.5 + 0.5 * _time_content_support(series)))

            sample_repr = [str(v) for v in sample[:5]]
            candidates.append(
                HiddenGroupCandidate(
                    group=rule.group,
                    sheet=sheet_name,
                    column=column,
                    score=round(score, 3),
                    rationale=(
                        f"{rule.rationale} Совпавший маркер: '{matched_marker}'. "
                        f"Примеры значений: {sample_repr}."
                    ),
                    sample_values=sample_repr,
                )
            )

    return candidates


def detect_columns(loaded: LoadedExcel) -> ColumnDetectionReport:
    """Собрать :class:`ColumnDetectionReport` по всем листам."""

    all_candidates: list[HiddenGroupCandidate] = []
    for sheet_name, df in loaded.sheets.items():
        all_candidates.extend(_detect_sheet(sheet_name, df))

    # Какие группы мы считаем уверенно распознанными.
    detected: set[HiddenGroup] = set()
    for c in all_candidates:
        if c.score >= _STRONG_THRESHOLD:
            detected.add(c.group)

    required = {
        HiddenGroup.ZAP,
        HiddenGroup.MANEUVERING,
        HiddenGroup.KFV,
        HiddenGroup.VUP,
        HiddenGroup.TIME,
    }
    missing = sorted(required - detected, key=lambda g: g.value)

    assumptions: list[str] = [
        "Детекция колонок эвристическая: опирается на нормализованные заголовки "
        "и на небольшой семпл значений.",
        f"Порог уверенной детекции: score ≥ {_STRONG_THRESHOLD}.",
        "Любой распознанный кандидат требует подтверждения через ручное сопоставление "
        "или config column_mapping перед содержательным анализом.",
    ]

    # Сортируем кандидатов для стабильного UI: сначала сильные, потом по группе и колонке.
    all_candidates.sort(
        key=lambda c: (-c.score, c.group.value, c.sheet, c.column)
    )

    return ColumnDetectionReport(
        candidates=all_candidates,
        detected_groups=sorted(detected, key=lambda g: g.value),
        missing_groups=list(missing),
        assumptions=assumptions,
    )


def strong_zap_candidates(report: ColumnDetectionReport) -> list[HiddenGroupCandidate]:
    """Только уверенные ЗАП-кандидаты, пригодные для baseline-распределений."""

    return [
        c for c in report.candidates
        if c.group == HiddenGroup.ZAP and c.score >= _STRONG_THRESHOLD
    ]


def weak_candidates(report: ColumnDetectionReport) -> list[HiddenGroupCandidate]:
    """Кандидаты в серой зоне — пригодны для ``warnings``, не для выводов."""

    return [
        c for c in report.candidates
        if _WEAK_THRESHOLD <= c.score < _STRONG_THRESHOLD
    ]
