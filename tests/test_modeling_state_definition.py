from hidden_patterns_combat.modeling.state_definition import StateDefinition


def test_research_default_names_for_three_states():
    st = StateDefinition.research_default(3)
    assert st.state_names() == ["S1", "S2", "S3"]


def test_state_definition_serialization_roundtrip():
    st = StateDefinition.research_default(4)
    payload = st.to_dict()
    loaded = StateDefinition.from_dict(payload)
    assert loaded.state_name(0) == "S1"
    assert loaded.state_name(3) == "S4"
