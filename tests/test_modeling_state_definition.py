from hidden_patterns_combat.modeling.state_definition import StateDefinition


def test_research_default_names_for_three_states():
    st = StateDefinition.research_default(3)
    assert st.state_names() == ["state_0", "state_1", "state_2"]


def test_state_definition_serialization_roundtrip():
    st = StateDefinition.research_default(4)
    payload = st.to_dict()
    loaded = StateDefinition.from_dict(payload)
    assert loaded.state_name(0) == "state_0"
    assert loaded.state_name(3) == "state_3"
