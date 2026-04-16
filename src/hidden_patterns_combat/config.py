from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HeaderConfig:
    multirow_header_depth: int = 2
    drop_unnamed_levels: bool = True


@dataclass
class FeatureConfig:
    episode_id_column_candidates: tuple[str, ...] = (
        "metadata__episode_id",
        "номер эпизода",
        "episode",
        "episode_id",
    )
    duration_column_candidates: tuple[str, ...] = (
        "metadata__episode_duration",
        "время эпизода",
        "episode_duration",
        "duration",
    )
    pause_column_candidates: tuple[str, ...] = (
        "metadata__pause_duration",
        "pause_duration",
        "время паузы",
        "pause",
    )
    result_column_candidates: tuple[str, ...] = (
        "outcomes__score",
        "observed_result",
        "баллы",
        "результат",
        "оценка",
        "result",
    )

    maneuver_right_tokens: tuple[str, ...] = ("стойка прав", "maneuver_right", "mr_")
    maneuver_left_tokens: tuple[str, ...] = ("стойка лев", "maneuver_left", "ml_")
    maneuver_group_tokens: tuple[str, ...] = (
        "стойка и маневрирование",
        "maneuver",
    )
    kfv_tokens: tuple[str, ...] = (
        "кфв",
        "захват",
        "хват",
        "обхват",
        "прихват",
        "упор",
        "kfv",
    )
    vup_tokens: tuple[str, ...] = ("вуп", "выведение", "vup")


@dataclass
class ModelConfig:
    n_hidden_states: int = 3
    covariance_type: str = "diag"
    n_iter: int = 300
    random_state: int = 42
    topology_mode: str = "left_to_right"  # left_to_right | ergodic
    semantic_init_enabled: bool = True
    canonical_reorder_enabled: bool = True
    min_forward_transition: float = 0.05
    min_self_transition: float = 0.55
    max_self_transition: float = 0.94
    inverse_initial_self_transition: float = 0.88
    inverse_anchor_blend: float = 0.50
    inverse_emission_prior_blend: float = 0.08
    inverse_anchor_power: float = 2.0
    inverse_stage_prior_blend: float = 0.30
    block_weights: dict[str, float] = field(
        default_factory=lambda: {
            "maneuvering": 1.0,
            "kfv": 0.85,
            "vup": 1.35,
            "temporal": 0.90,
            "other": 1.0,
        }
    )


@dataclass
class PipelineConfig:
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts")
    header: HeaderConfig = field(default_factory=HeaderConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
