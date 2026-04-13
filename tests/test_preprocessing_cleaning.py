import pandas as pd

from hidden_patterns_combat.preprocessing.cleaning import clean_episode_table


def test_cleaning_drops_aggregate_rows():
    df = pd.DataFrame(
        {
            "фио борца": ["Иванов", "Итог"],
            "баллы": [1, 10],
            "стойка и маневрирование": [1, 0],
        }
    )
    out = clean_episode_table(df)
    assert len(out) == 1
    assert out["фио борца"].iloc[0] == "Иванов"


def test_cleaning_keeps_valid_episode_row():
    df = pd.DataFrame(
        {
            "фио борца": ["Петров"],
            "номер эпизода": ["1"],
            "баллы": [0],
            "кфв": [1],
        }
    )
    out = clean_episode_table(df)
    assert len(out) == 1


def test_cleaning_drops_low_information_row():
    df = pd.DataFrame(
        {
            "фио борца": ["", "Иванов"],
            "номер эпизода": ["", "2"],
            "баллы": [0, 0],
            "кфв": [0, 1],
        }
    )
    out = clean_episode_table(df)
    assert len(out) == 1
    assert out["фио борца"].iloc[0] == "Иванов"
