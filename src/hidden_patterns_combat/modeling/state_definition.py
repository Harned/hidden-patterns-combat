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
                    description="Latent tactical state inferred by HMM from observations.",
                )
                for i in range(n_states)
            ]
        )

    @classmethod
    def research_default(cls, n_states: int = 3) -> "StateDefinition":
        defaults = [
            HiddenState(0, "S1", "Stance and maneuvering phase"),
            HiddenState(1, "S2", "Physical interaction contacts (KFV)"),
            HiddenState(2, "S3", "Destabilization / VUP phase"),
        ]
        if n_states <= 3:
            return cls(states=defaults[:n_states])

        extra = [
            HiddenState(i, f"S{i+1}", "Additional latent state (configurable hypothesis)")
            for i in range(3, n_states)
        ]
        return cls(states=defaults + extra)

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
