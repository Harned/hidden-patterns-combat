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
        "номер эпизода",
        "episode",
        "episode_id",
    )
    duration_column_candidates: tuple[str, ...] = (
        "время эпизода",
        "episode_duration",
        "duration",
    )
    pause_column_candidates: tuple[str, ...] = (
        "время паузы",
        "pause",
    )
    result_column_candidates: tuple[str, ...] = (
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


@dataclass
class PipelineConfig:
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts")
    header: HeaderConfig = field(default_factory=HeaderConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
