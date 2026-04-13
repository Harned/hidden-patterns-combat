from hidden_patterns_combat.preprocessing.data_dictionary import DataDictionary


def test_data_dictionary_lookup_and_group_selection():
    dd = DataDictionary.default()
    entry = dd.lookup("фио борца")
    assert entry is not None
    assert entry.normalized_field == "metadata__athlete_name"
    assert entry.logical_group == "metadata"

    cols = [
        "фио борца",
        "стойка и маневрирование самбиста (основные в эпизоде)",
        "контакты физического взаимодействия (захваты, обхваты, прихваты, хваты, упоры)",
    ]
    maneuver = dd.columns_for_group(cols, "maneuvering")
    assert maneuver == ["стойка и маневрирование самбиста (основные в эпизоде)"]


def test_data_dictionary_group_selection_fallback_by_tokens():
    dd = DataDictionary.default()
    cols = ["unknown", "контакты физического взаимодействия x", "выведение y"]
    kfv = dd.columns_for_group(cols, "kfv")
    vup = dd.columns_for_group(cols, "vup")
    assert kfv == ["контакты физического взаимодействия x"]
    assert vup == ["выведение y"]
