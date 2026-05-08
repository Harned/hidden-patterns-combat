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
def bout_resets_excel(tmp_path: Path) -> Path:
    """Excel без явной колонки «Схватка», но с эпизодами, перезапускающимися
    в каждой схватке.

    На каждого из четырёх борцов приходится 2 виртуальных схватки по 3
    эпизода (нумерация эпизодов сбрасывается с 3 → 1 при переходе к
    следующей схватке). Без виртуального bout группировка по одному
    борцу даёт 1 длинную серию на спортсмена; с виртуальным bout —
    8 серий длины 3, что соответствует реальной структуре боёв.
    """

    import openpyxl

    path = tmp_path / "bout_resets.xlsx"
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
        ]
    )
    ws.append([None, None, None, "Болевой прием", None])
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            "Удержание",
            "На руку",
        ]
    )

    athletes = ["Иванов", "Петров", "Сидоров", "Кузнецов"]
    pattern = [
        # bout 1: episodes 1..3, then bout 2: episodes 1..3
        (1, 1, 0),
        (2, 0, 1),
        (3, 1, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
    ]
    for athlete in athletes:
        for ep, udrzh, ruka in pattern:
            ws.append([athlete, ep, 15 + ep, udrzh, ruka])

    wb.save(path)
    return path


@pytest.fixture
def totals_row_excel(tmp_path: Path) -> Path:
    """Excel с одной строкой-итогом «Итого» в хвосте листа.

    Реальные файлы вида ``docs/Оценка СД содержание.xlsx`` иногда
    содержат хвостовую строку с агрегатами, структурно неотличимую от
    обычного эпизода. Без фильтра она попадает в baseline/HMM и
    исказит распределения. Фикстура минимальна (3 эпизода + 1 итог)
    и предназначена только для проверки детектора.
    """

    import openpyxl

    path = tmp_path / "totals.xlsx"
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
        ]
    )
    ws.append([None, None, None, "Болевой прием", None])
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            "Удержание",
            "На руку",
        ]
    )
    ws.append(["Иванов", 1, 12, 1, 0])
    ws.append(["Иванов", 2, 18, 0, 1])
    ws.append(["Иванов", 3, 14, 1, 0])
    ws.append(["Итого", None, 44, 2, 1])
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


@pytest.fixture
def markov_two_bouts_excel(tmp_path: Path) -> Path:
    """Фикстура для TASK_SPEC_011: 3-уровневая шапка, 2 поединка, ep_num reset.

    Структура листа `Общее` повторяет реальный формат
    ``docs/Оценка СД содержание.xlsx`` в редуцированном виде:

    * 12 колонок: 5 служебных (ФИО, № эпизода, время эпизода / паузы) +
      по одной колонке-представителю каждой из 4 признаковых групп
      (manoeuvring / grip / off_balance / technical_action), плюс
      несколько вспомогательных под завершающие приёмы.
    * Поединок 1: 3 эпизода × 2 спортсмена (Иванов / Петров), пустое
      ``Время паузы`` у последнего эпизода каждого, дальше — пустая
      строка-разделитель.
    * Поединок 2: 2 эпизода × 2 спортсмена; ``№ эпизода`` сбрасывается
      к 1, что является вторым маркером границы bout'а.
    * В одной строке стоит значение ``8`` в признаковой колонке —
      проверка защиты ``>2 → log&skip``.
    """

    import openpyxl

    path = tmp_path / "markov_two_bouts.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Общее"

    # row 0: super-headers
    ws.append(
        [
            "ФИО борца",
            "Технико-тактический эпизод",
            None,
            None,
            "Стойка и маневрирование",
            None,
            "КФВ",
            None,
            "ВУП",
            "Завершающие атаку приемы (n)",
            None,
            None,
        ]
    )
    # row 1: subgroups
    ws.append(
        [
            None,
            None,
            None,
            None,
            "Правосторонняя стойка (ПС)",
            "Левосторонняя стойка (ЛС)",
            "Захваты",
            "Хваты",
            None,
            None,
            "Болевой прием",
            "Болевой прием",
        ]
    )
    # row 2: atomic headers
    ws.append(
        [
            None,
            "№ эпизода",
            "Время эпизода, с.",
            "Время паузы, с.",
            "Вперед-влево",
            "Назад",
            "Двусторонний",
            "ХвШ",
            "ВУП-Р",
            "Удержание",
            "На руку",
            "На ногу",
        ]
    )

    # Schema reminder for data rows:
    #   ФИО, № эпизода, время эп., время паузы,
    #   ПС/Вперед-влево, ЛС/Назад, КФВ/Двусторонний, КФВ/Хваты/ХвШ, ВУП-Р,
    #   Удержание, Болевой/На руку, Болевой/На ногу
    rows = [
        # Bout 1 (2 athletes × 3 episodes)
        ["Иванов", 1, 30, 5, 1, 0, 0, 0, 0, 0, 0, 0],  # manoeuvring
        ["Петров", 1, 30, 5, 0, 1, 1, 0, 0, 0, 0, 0],  # grip (mano+grip → grip wins)
        ["Иванов", 2, 25, 4, 0, 0, 1, 0, 1, 0, 0, 0],  # off_balance (grip+vup → off_balance)
        ["Петров", 2, 25, 4, 0, 0, 0, 0, 0, 1, 0, 0],  # technical_action
        ["Иванов", 3, 22, None, 0, 0, 1, 0, 0, 0, 0, 0],  # grip; pause empty
        ["Петров", 3, 22, None, 0, 0, 0, 0, 0, 0, 1, 0],  # technical_action; pause empty
        # blank separator between bouts
        [None, None, None, None, None, None, None, None, None, None, None, None],
        # Bout 2 (ep_num resets to 1)
        ["Иванов", 1, 18, 6, 1, 1, 0, 0, 0, 0, 0, 0],  # manoeuvring
        ["Петров", 1, 18, 6, 0, 0, 1, 1, 0, 0, 0, 0],  # grip
        ["Иванов", 2, 20, None, 0, 0, 0, 0, 0, 1, 0, 1],  # technical_action
        # value=8 in feature → out-of-range warning, state collapses to pause
        ["Петров", 2, 20, None, 0, 0, 8, 0, 0, 0, 0, 0],  # pause + warning
    ]
    for r in rows:
        ws.append(r)

    wb.save(path)
    return path
