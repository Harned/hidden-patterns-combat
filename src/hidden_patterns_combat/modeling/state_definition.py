from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HiddenState:
    state_id: int
    name: str
    description: str


@dataclass
class StateDefinition:
    """Declarative hidden-state layer.

    Important: hidden states are latent model variables and are not equal to engineered features.
    """

    states: list[HiddenState]

    @classmethod
    def from_count(cls, n_states: int) -> "StateDefinition":
        return cls(
            states=[
                HiddenState(
                    state_id=i,
                    name=f"state_{i}",
                    description="Latent tactical state inferred by HMM.",
                )
                for i in range(n_states)
            ]
        )

    @classmethod
    def research_default(cls, n_states: int = 3) -> "StateDefinition":
        # Safe default for MVP:
        # keep state labels latent and neutral at model output layer.
        return cls.from_count(n_states)

    def to_dict(self) -> dict[str, object]:
        return {"states": [asdict(s) for s in self.states]}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StateDefinition":
        states = [HiddenState(**row) for row in payload.get("states", [])]
        return cls(states=states)

    def state_names(self) -> list[str]:
        return [s.name for s in self.states]

    def state_name(self, state_id: int) -> str:
        for st in self.states:
            if st.state_id == state_id:
                return st.name
        return f"state_{state_id}"
