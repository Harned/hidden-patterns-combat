from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


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

    @classmethod
    def from_mapping(
        cls,
        n_states: int,
        name_mapping: dict[int, str],
        description_mapping: dict[int, str] | None = None,
    ) -> "StateDefinition":
        description_mapping = description_mapping or {}
        states = []
        for state_id in range(n_states):
            name = name_mapping.get(state_id, f"state_{state_id}")
            description = description_mapping.get(
                state_id,
                "Latent tactical state inferred by HMM.",
            )
            states.append(HiddenState(state_id=state_id, name=name, description=description))
        return cls(states=states)


def build_semantic_state_definition(
    features: pd.DataFrame,
    decoded_states: np.ndarray,
    n_states: int,
) -> StateDefinition:
    """Derive stable semantic labels S1/S2/S3 from state feature profiles.

    Semantic targets:
    - S1: maneuvering-dominant
    - S2: KFV-dominant
    - S3: VUP-dominant
    """
    frame = features.copy().reset_index(drop=True)
    frame["_state"] = decoded_states

    block_columns = {
        "maneuvering": ["maneuver_right_code", "maneuver_left_code"],
        "kfv": ["kfv_code", "grips_code", "holds_code", "bodylocks_code", "underhooks_code", "posts_code"],
        "vup": ["vup_code"],
    }

    def _block_score(row: pd.Series, cols: list[str]) -> float:
        present = [c for c in cols if c in row.index]
        if not present:
            return 0.0
        return float(np.mean([abs(float(row[c])) for c in present]))

    profile = frame.groupby("_state", dropna=False).mean(numeric_only=True)
    scores: dict[int, dict[str, float]] = {}
    for state_id in range(n_states):
        if state_id in profile.index:
            row = profile.loc[state_id]
            block_scores = {block: _block_score(row, cols) for block, cols in block_columns.items()}
        else:
            block_scores = {block: 0.0 for block in block_columns}
        total = sum(block_scores.values())
        shares = {
            block: (value / total if total > 0 else 0.0)
            for block, value in block_scores.items()
        }
        scores[state_id] = {f"{k}_score": v for k, v in block_scores.items()} | {
            f"{k}_share": v for k, v in shares.items()
        }

    semantic_targets = [
        ("maneuvering", "S1", "Маневрирование-доминантное латентное состояние."),
        ("kfv", "S2", "КФВ-доминантное латентное состояние."),
        ("vup", "S3", "ВУП-доминантное латентное состояние."),
    ]

    remaining = set(range(n_states))
    name_mapping: dict[int, str] = {}
    description_mapping: dict[int, str] = {}

    for block, semantic_name, semantic_description in semantic_targets:
        if not remaining:
            break
        ordered = sorted(
            remaining,
            key=lambda sid: (
                scores[sid].get(f"{block}_share", 0.0),
                scores[sid].get(f"{block}_score", 0.0),
                -sid,
            ),
            reverse=True,
        )
        selected = ordered[0]
        name_mapping[selected] = semantic_name
        description_mapping[selected] = semantic_description
        remaining.remove(selected)

    for state_id in sorted(remaining):
        fallback_name = f"state_{state_id}"
        name_mapping[state_id] = fallback_name
        description_mapping[state_id] = "Дополнительное латентное состояние."

    return StateDefinition.from_mapping(
        n_states=n_states,
        name_mapping=name_mapping,
        description_mapping=description_mapping,
    )
