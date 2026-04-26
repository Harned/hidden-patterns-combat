"""Фикстуры для тестов processing module.

Все фикстуры — маленькие synthetic Excel-файлы, собираемые в tmp_path.
Реальный Excel ``docs/Оценка СД содержание.xlsx`` в unit-тестах не
используется намеренно: это предметные данные, тестируем на них вручную
через CLI, а не в CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def zap_only_excel(tmp_path: Path) -> Path:
    """Файл с явной ЗАП-колонкой и несколькими строками."""

    path = tmp_path / "zap_only.xlsx"
    df = pd.DataFrame(
        {
            "Спортсмен": ["Иванов", "Петров", "Иванов", "Сидоров"],
            "Эпизод": [1, 1, 2, 1],
            "Время, сек": [12.3, 7.5, 20.1, 9.0],
            "ЗАП": ["ЗАП-Р", "ЗАП-Т", "удержание", "ЗАП-Н"],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Общее", index=False)
    return path


@pytest.fixture
def full_structure_excel(tmp_path: Path) -> Path:
    """Файл, где присутствуют маркеры всех обязательных групп."""

    path = tmp_path / "full_structure.xlsx"
    df = pd.DataFrame(
        {
            "Спортсмен": ["A", "B", "A"],
            "Схватка": [1, 1, 2],
            "Эпизод": [1, 2, 1],
            "Время, сек": [10, 15, 12],
            "Маневрирование: стойка": ["правая", "левая", "правая"],
            "КФВ захват": ["двусторонний", "односторонний", "двусторонний"],
            "ВУП": ["передний", "задний", "боковой"],
            "ЗАП": ["ЗАП-Р", "ЗАП-Н", "удержание"],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Эпизоды", index=False)
    return path


@pytest.fixture
def opaque_excel(tmp_path: Path) -> Path:
    """Файл без осмысленных предметных заголовков — детекция не должна
    объявить ничего уверенным."""

    path = tmp_path / "opaque.xlsx"
    df = pd.DataFrame(
        {
            "col_1": [1, 2, 3],
            "col_2": ["x", "y", "z"],
            "col_3": [0.1, 0.2, 0.3],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    return path


@pytest.fixture
def empty_excel(tmp_path: Path) -> Path:
    path = tmp_path / "empty.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame().to_excel(writer, sheet_name="Sheet1", index=False)
    return path


@pytest.fixture
def multirow_header_excel(tmp_path: Path) -> Path:
    """Эмитируем реальный формат `Оценка СД содержание.xlsx`.

    Три строки заголовка:
      row 0: группировочные супер-заголовки ("Баллы", "Стойка и маневрирование самбиста...", "ВУП")
      row 1: подгруппы ("Правосторонняя стойка (ПС)", "КФВ", ...)
      row 2: атомарные заголовки ("№ эпизода", "Время эпизода", "Вперед-влево", ...)
      row 3+: данные.
    """

    import openpyxl

    path = tmp_path / "multirow.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "48"

    # row 1 (index 0): супер-заголовки
    ws.append(
        [
            "ФИО борца",
            "Технико-тактический эпизод",
            None,
            None,
            "Баллы",
            "Стойка и маневрирование самбиста (основные в эпизоде)",
            None,
            "КФВ",
            None,
            "ВУП",
        ]
    )
    # row 2 (index 1): подгруппы
    ws.append(
        [
            None,
            None,
            None,
            None,
            None,
            "Правосторонняя стойка (ПС)",
            None,
            "Захваты",
            None,
            None,
        ]
    )
    # row 3 (index 2): атомарные заголовки
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            "Время паузы, с.",
            "ЗАП",
            "Вперед-влево",
            "Назад",
            "Двусторонний захват",
            "Односторонний захват",
            "Передний",
        ]
    )
    # data rows
    rows = [
        ["Иванов", 1, 34, 7, "ЗАП-Р", 1, 0, 1, 0, 0],
        ["Иванов", 2, 29, 8, "ЗАП-Т", 0, 1, 0, 1, 0],
        ["Петров", 1, 15, 5, "удержание", 1, 0, 1, 0, 1],
        ["Петров", 2, 22, 6, "ЗАП-Н", 0, 0, 0, 1, 1],
        ["Сидоров", 1, 18, 4, "ЗАП-Р", 1, 1, 0, 0, 0],
    ]
    for r in rows:
        ws.append(r)

    wb.save(path)
    return path


@pytest.fixture
def multirow_binary_zap_excel(tmp_path: Path) -> Path:
    """Excel с multi-row header и **бинарной** кодировкой ЗАП.

    В реальном файле ЗАП-события разнесены по отдельным колонкам
    (удержание / болевой приём / …) с целочисленными значениями 0/1.
    Фикстура воспроизводит эту структуру минимальным набором каналов.
    """

    import openpyxl

    path = tmp_path / "multirow_binary.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "48"

    # row 0
    ws.append(
        [
            "ФИО борца",
            "Технико-тактический эпизод",
            None,
            "Стойка и маневрирование самбиста",
            None,
            "КФВ",
            "ВУП",
            "Завершающие атаку приемы (n)",
            None,
            None,
        ]
    )
    # row 1
    ws.append(
        [
            None,
            None,
            None,
            "Правосторонняя стойка (ПС)",
            None,
            "Захваты",
            None,
            None,
            "Болевой прием",
            None,
        ]
    )
    # row 2
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            "Вперед",
            "Назад",
            "Двусторонний",
            "Передний",
            "Удержание",
            "На руку",
            "На ногу",
        ]
    )
    # data
    rows = [
        ["Иванов", 1, 34, 1, 0, 1, 0, 1, 0, 0],
        ["Иванов", 2, 29, 0, 1, 0, 1, 0, 1, 0],
        ["Петров", 1, 15, 1, 0, 1, 1, 0, 0, 1],
        ["Петров", 2, 22, 0, 0, 0, 0, 2, 0, 0],  # 2 удержания в одном эпизоде
        ["Сидоров", 1, 18, 1, 1, 0, 0, 0, 0, 0],
    ]
    for r in rows:
        ws.append(r)

    wb.save(path)
    return path


@pytest.fixture
def dense_hmm_ready_excel(tmp_path: Path) -> Path:
    """Плотная synthetic-фикстура для успешного прохождения HMM-guards.

    40 эпизодов, ЗАП-события разнесены по 4 каналам.
    """

    import random

    import openpyxl

    path = tmp_path / "dense_hmm.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "48"

    ws.append(
        [
            "ФИО борца",
            "Технико-тактический эпизод",
            None,
            "Завершающие атаку приемы (n)",
            None,
            None,
            None,
        ]
    )
    ws.append([None, None, None, None, "Болевой прием", None, None])
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            "Удержание",
            "На руку",
            "На ногу",
            "ЗАП-Р",
        ]
    )

    rng = random.Random(0)
    names = ["Иванов", "Петров", "Сидоров", "Кузнецов"]
    for i in range(1, 41):
        name = rng.choice(names)
        time_s = rng.randint(10, 40)
        udrzh = 1 if rng.random() < 0.3 else 0
        ruka = 1 if rng.random() < 0.2 else 0
        noga = 1 if rng.random() < 0.2 else 0
        zap_r = 1 if rng.random() < 0.25 else 0
        ws.append([name, i, time_s, udrzh, ruka, noga, zap_r])

    wb.save(path)
    return path


@pytest.fixture
def thin_excel(tmp_path: Path) -> Path:
    """Очень маленький файл — guard'ы обязаны заблокировать HMM."""

    import openpyxl

    path = tmp_path / "thin.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "48"
    ws.append(["ФИО борца", "Технико-тактический эпизод", None, "Завершающие атаку приемы (n)"])
    ws.append([None, None, None, None])
    ws.append([None, "№ эпизода", "Время эпизода, с.", "Удержание"])
    ws.append(["Иванов", 1, 12, 1])
    ws.append(["Петров", 1, 10, 0])
    wb.save(path)
    return path


@pytest.fixture
def balls_zap_excel(tmp_path: Path) -> Path:
    """Multi-row header с супер-заголовком «Баллы» и числовой судейской шкалой.

    Цель — проверить, что:
      * детекция распознаёт колонку под «Баллы» как ZAP-кандидата по
        маркеру `балл` + числовой контент-поддержке;
      * после явного включения «Баллы» в роль ZAP плотность ZAP-сигнала
        достаточна для прохождения guard ``low_zap_density``.
    """

    import openpyxl

    path = tmp_path / "balls_zap.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Общее"

    # row 0: супер-заголовки.
    ws.append(
        [
            "ФИО борца",
            "Технико-тактический эпизод",
            None,
            "Баллы",
            "Завершающие атаку приемы (n)",
            None,
        ]
    )
    # row 1: подгруппы (для «Баллы» под-заголовка нет — атомарно, как в
    # реальном файле).
    ws.append([None, None, None, None, None, None])
    # row 2: атомарные заголовки.
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            None,
            "Удержание",
            "На руку",
        ]
    )

    import random

    rng = random.Random(7)
    # Алфавит судейских баллов: 1/2/4/6/8 (малый, max=8, ≤ 16).
    score_alphabet = [1, 2, 4, 6, 8]
    names = ["Иванов", "Петров", "Сидоров", "Кузнецов"]
    for i in range(1, 81):  # 80 эпизодов.
        name = rng.choice(names)
        time_s = rng.randint(10, 40)
        # ~35% эпизодов оценены ненулевым баллом → плотность ZAP > порога.
        score: int | None
        if rng.random() < 0.35:
            score = rng.choice(score_alphabet)
        else:
            score = 0
        # Удержание/«На руку» — редкие, чтобы без «Баллы» сигнал был слабым
        # (как в реальном файле).
        udrzh = 1 if rng.random() < 0.04 else 0
        ruka = 1 if rng.random() < 0.03 else 0
        ws.append([name, i, time_s, score, udrzh, ruka])

    wb.save(path)
    return path


@pytest.fixture
def very_dense_excel(tmp_path: Path) -> Path:
    """Плотная фикстура (≥150 эпизодов) для 7-state HMM.

    Алфавит специально богатый, чтобы guards detailed проходили.
    """

    import random

    import openpyxl

    path = tmp_path / "very_dense.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "all"

    ws.append(
        [
            "ФИО борца",
            "Технико-тактический эпизод",
            None,
            "Завершающие атаку приемы (n)",
            None,
            None,
            None,
            None,
        ]
    )
    ws.append([None, None, None, None, "Болевой прием", None, None, None])
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            "Удержание",
            "На руку",
            "На ногу",
            "ЗАП-Р",
            "ЗАП-Т",
        ]
    )

    rng = random.Random(0)
    names = ["Иванов", "Петров", "Сидоров", "Кузнецов", "Смирнов", "Попов"]
    for i in range(1, 181):
        name = rng.choice(names)
        ws.append(
            [
                name,
                i,
                rng.randint(10, 40),
                1 if rng.random() < 0.35 else 0,
                1 if rng.random() < 0.25 else 0,
                1 if rng.random() < 0.2 else 0,
                1 if rng.random() < 0.25 else 0,
                1 if rng.random() < 0.2 else 0,
            ]
        )

    wb.save(path)
    return path
