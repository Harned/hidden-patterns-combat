from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


SEMANTIC_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("maneuvering", "S1", "Маневрирование-доминантное латентное состояние."),
    ("kfv", "S2", "КФВ-доминантное латентное состояние."),
    ("vup", "S3", "ВУП-доминантное латентное состояние."),
)

BLOCK_COLUMNS: dict[str, list[str]] = {
    "maneuvering": ["maneuver_right_code", "maneuver_left_code"],
    "kfv": ["kfv_code", "grips_code", "holds_code", "bodylocks_code", "underhooks_code", "posts_code"],
    "vup": ["vup_code"],
}


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


@dataclass
class SemanticOrderingDiagnostics:
    """Explicit semantic layer that separates discovery, naming and ordering."""

    original_name_mapping: dict[int, str]
    original_description_mapping: dict[int, str]
    canonical_order: list[int]
    semantic_to_original_state: dict[str, int]
    semantic_confidence: dict[str, float]
    state_profiles: list[dict[str, object]]
    semantic_order_matches_topology: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "original_name_mapping": {int(k): v for k, v in self.original_name_mapping.items()},
            "original_description_mapping": {
                int(k): v for k, v in self.original_description_mapping.items()
            },
            "canonical_order": [int(x) for x in self.canonical_order],
            "semantic_to_original_state": {
                k: int(v) for k, v in self.semantic_to_original_state.items()
            },
            "semantic_confidence": {k: float(v) for k, v in self.semantic_confidence.items()},
            "state_profiles": self.state_profiles,
            "semantic_order_matches_topology": bool(self.semantic_order_matches_topology),
            "warnings": list(self.warnings),
        }

    def original_state_definition(self, n_states: int) -> StateDefinition:
        return StateDefinition.from_mapping(
            n_states=n_states,
            name_mapping=self.original_name_mapping,
            description_mapping=self.original_description_mapping,
        )

    def canonical_state_definition(self, n_states: int) -> StateDefinition:
        canonical_names = {
            new_idx: self.original_name_mapping.get(old_idx, f"state_{new_idx}")
            for new_idx, old_idx in enumerate(self.canonical_order)
        }
        canonical_desc = {
            new_idx: self.original_description_mapping.get(old_idx, "Latent tactical state inferred by HMM.")
            for new_idx, old_idx in enumerate(self.canonical_order)
        }
        return StateDefinition.from_mapping(
            n_states=n_states,
            name_mapping=canonical_names,
            description_mapping=canonical_desc,
        )


def _block_score(row: pd.Series, cols: list[str]) -> float:
    present = [c for c in cols if c in row.index]
    if not present:
        return 0.0
    return float(np.mean([abs(float(row[c])) for c in present]))


def _state_profile(
    features: pd.DataFrame,
    decoded_states: np.ndarray,
    n_states: int,
) -> dict[int, dict[str, float | str]]:
    frame = features.copy().reset_index(drop=True)
    frame["_state"] = decoded_states
    profile = frame.groupby("_state", dropna=False).mean(numeric_only=True)

    scores: dict[int, dict[str, float | str]] = {}
    for state_id in range(n_states):
        if state_id in profile.index:
            row = profile.loc[state_id]
            block_scores = {block: _block_score(row, cols) for block, cols in BLOCK_COLUMNS.items()}
        else:
            block_scores = {block: 0.0 for block in BLOCK_COLUMNS}

        total = sum(block_scores.values())
        block_shares = {k: (v / total if total > 0 else 0.0) for k, v in block_scores.items()}
        ordered = sorted(block_shares.items(), key=lambda it: it[1], reverse=True)
        dominant_block = ordered[0][0] if ordered else "unknown"
        dominant_share = float(ordered[0][1]) if ordered else 0.0
        second_share = float(ordered[1][1]) if len(ordered) > 1 else 0.0
        dominance_margin = max(0.0, dominant_share - second_share)

        scores[state_id] = {
            **{f"{k}_score": float(v) for k, v in block_scores.items()},
            **{f"{k}_share": float(v) for k, v in block_shares.items()},
            "dominant_block": dominant_block,
            "dominant_share": dominant_share,
            "dominance_margin": dominance_margin,
        }

    return scores


def derive_semantic_ordering(
    features: pd.DataFrame,
    decoded_states: np.ndarray,
    n_states: int,
    min_dominance_share: float = 0.45,
    min_dominance_margin: float = 0.10,
    min_block_score: float = 0.15,
) -> SemanticOrderingDiagnostics:
    """Discover semantic states and derive canonical S1->S2->S3 ordering.

    Important: this function only *derives* semantic assignment and canonical ordering.
    Reordering model parameters should be done in the training layer.
    """
    scores = _state_profile(features=features, decoded_states=decoded_states, n_states=n_states)

    remaining = set(range(n_states))
    name_mapping: dict[int, str] = {}
    description_mapping: dict[int, str] = {}
    semantic_to_state: dict[str, int] = {}
    semantic_confidence: dict[str, float] = {name: 0.0 for _, name, _ in SEMANTIC_TARGETS}

    for block, semantic_name, semantic_description in SEMANTIC_TARGETS:
        if not remaining:
            break

        ranked = sorted(
            remaining,
            key=lambda sid: (
                float(scores[sid].get(f"{block}_share", 0.0)),
                float(scores[sid].get("dominance_margin", 0.0)),
                float(scores[sid].get(f"{block}_score", 0.0)),
                -sid,
            ),
            reverse=True,
        )

        if not ranked:
            continue

        selected = ranked[0]
        target_share = float(scores[selected].get(f"{block}_share", 0.0))
        target_score = float(scores[selected].get(f"{block}_score", 0.0))
        margin = float(scores[selected].get("dominance_margin", 0.0))
        dominant_block = str(scores[selected].get("dominant_block", "unknown"))

        is_confident = (
            target_share >= min_dominance_share
            and margin >= min_dominance_margin
            and target_score >= min_block_score
            and dominant_block == block
        )

        if is_confident:
            name_mapping[selected] = semantic_name
            description_mapping[selected] = semantic_description
            semantic_to_state[semantic_name] = selected
            semantic_confidence[semantic_name] = float(np.clip(0.5 * target_share + 0.5 * margin, 0.0, 1.0))
            remaining.remove(selected)

    for state_id in sorted(remaining):
        fallback_name = f"state_{state_id}"
        dom_block = str(scores[state_id].get("dominant_block", "unknown"))
        dom_share = float(scores[state_id].get("dominant_share", 0.0))
        name_mapping[state_id] = fallback_name
        description_mapping[state_id] = (
            "Нейтральное латентное состояние: уверенного соответствия S1/S2/S3 нет "
            f"(dominant={dom_block}, share={dom_share:.3f})."
        )

    semantic_prefix = [semantic_to_state[name] for _, name, _ in SEMANTIC_TARGETS if name in semantic_to_state]
    trailing = [sid for sid in sorted(range(n_states)) if sid not in semantic_prefix]
    canonical_order = semantic_prefix + trailing

    state_profiles: list[dict[str, object]] = []
    for sid in range(n_states):
        state_profiles.append(
            {
                "state_id": int(sid),
                "assigned_name": name_mapping.get(sid, f"state_{sid}"),
                "dominant_block": str(scores[sid].get("dominant_block", "unknown")),
                "dominant_share": float(scores[sid].get("dominant_share", 0.0)),
                "dominance_margin": float(scores[sid].get("dominance_margin", 0.0)),
                "maneuvering_score": float(scores[sid].get("maneuvering_score", 0.0)),
                "kfv_score": float(scores[sid].get("kfv_score", 0.0)),
                "vup_score": float(scores[sid].get("vup_score", 0.0)),
                "maneuvering_share": float(scores[sid].get("maneuvering_share", 0.0)),
                "kfv_share": float(scores[sid].get("kfv_share", 0.0)),
                "vup_share": float(scores[sid].get("vup_share", 0.0)),
            }
        )

    warnings: list[str] = []
    for _, semantic_name, _ in SEMANTIC_TARGETS:
        if semantic_name not in semantic_to_state:
            warnings.append(
                f"Semantic state {semantic_name} not confidently discovered from current data/profiles."
            )

    semantic_order_matches_topology = canonical_order == list(range(n_states))
    if not semantic_order_matches_topology:
        warnings.append(
            "Semantic order differs from internal topology order before canonical reordering."
        )

    return SemanticOrderingDiagnostics(
        original_name_mapping=name_mapping,
        original_description_mapping=description_mapping,
        canonical_order=canonical_order,
        semantic_to_original_state=semantic_to_state,
        semantic_confidence=semantic_confidence,
        state_profiles=state_profiles,
        semantic_order_matches_topology=semantic_order_matches_topology,
        warnings=warnings,
    )


def build_semantic_state_definition(
    features: pd.DataFrame,
    decoded_states: np.ndarray,
    n_states: int,
    min_dominance_share: float = 0.45,
    min_dominance_margin: float = 0.10,
    min_block_score: float = 0.15,
) -> StateDefinition:
    """Backward-compatible facade returning only names for original state ids."""
    diagnostics = derive_semantic_ordering(
        features=features,
        decoded_states=decoded_states,
        n_states=n_states,
        min_dominance_share=min_dominance_share,
        min_dominance_margin=min_dominance_margin,
        min_block_score=min_block_score,
    )
    return diagnostics.original_state_definition(n_states=n_states)
