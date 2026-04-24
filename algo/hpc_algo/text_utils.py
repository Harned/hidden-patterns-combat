"""Небольшие утилиты работы с кириллическими заголовками.

Назначение ограничено: normalization названий колонок для эвристик детекции.
Любая эвристика, построенная на этих функциях, должна трактоваться как
*подсказка*, а не как решение — подтверждение column mapping всегда за
человеком.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+", re.UNICODE)
_PUNCT_RE = re.compile(r"[\\/_.,;:()\[\]{}\-\+\*\"'`]", re.UNICODE)


def normalize_header(value: object) -> str:
    """Нормализовать заголовок колонки для поиска по ключевым словам.

    * приведение к строке;
    * Unicode NFKC;
    * нижний регистр;
    * замена `ё` -> `е`;
    * удаление пунктуации/подчёркиваний;
    * схлопывание пробелов.

    Возвращает пустую строку для ``None`` / ``NaN``.
    """

    if value is None:
        return ""
    s = str(value)
    if s.lower() == "nan":
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("ё", "е")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def contains_any(haystack: str, needles: list[str]) -> bool:
    """Вернуть ``True``, если в нормализованной строке встречается любой маркер."""

    return any(n in haystack for n in needles)


def first_match(haystack: str, needles: list[str]) -> str | None:
    for n in needles:
        if n in haystack:
            return n
    return None
