import pandas as pd

from hidden_patterns_combat.preprocessing.transform import transform_raw_to_tidy


def test_transform_assigns_blocks_and_normalized_names():
    raw = pd.DataFrame(
        {
            "фио борца": ["A"],
            "баллы": [2],
            "стойка и маневрирование самбиста": [1],
            "контакты физического взаимодействия": [0],
            "выведение соперника": [1],
            "завершающие атаку приемы": [0],
            "_sheet": ["Общее"],
        }
    )
    tr = transform_raw_to_tidy(raw)
    cols = set(tr.cleaned.columns)
    assert "metadata__athlete_name" in cols
    assert "outcomes__score" in cols
    assert any(c.startswith("maneuvering__") for c in cols)
    assert any(c.startswith("kfv__") for c in cols)
    assert any(c.startswith("vup__") for c in cols)


def test_transform_keeps_unrecognized_columns_in_other_block():
    raw = pd.DataFrame({"кастомная колонка": [1, 2], "баллы": [1, 0]})
    tr = transform_raw_to_tidy(raw)
    assert any(c.startswith("other__") for c in tr.cleaned.columns)
    assert "outcomes__score" in tr.cleaned.columns
