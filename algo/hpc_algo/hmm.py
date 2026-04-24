"""HMM-ветка processing module (TASK_SPEC_004).

Ключевые предметные инварианты (не переговариваются):

* observations = ЗАП (токены строятся из
  :attr:`BaselineReport.zap_events_by_channel` и категориальных
  ЗАП-колонок);
* hidden states = ``маневрирование`` / ``КФВ`` / ``ВУП``;
* ``fighter_style`` не используется.

HMM запускается **только** если все data-quality guard'ы пройдены.
Иначе возвращается ``None`` и пользователь видит warning
``hmm.guards_failed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hpc_algo.baseline import (
    ZAP_KIND_BINARY,
    ZAP_KIND_CATEGORICAL,
    ZAP_KIND_COUNT,
    classify_zap_column,
)
from hpc_algo.schema import (
    BaselineReport,
    ColumnMappingConfig,
    HiddenGroup,
    HMMParameters,
    HMMResult,
    HMMTrajectory,
    WarningItem,
    WarningSeverity,
)

# --- Предметные константы модели ---

STATE_LABELS: tuple[str, ...] = (
    HiddenGroup.MANEUVERING.value,
    HiddenGroup.KFV.value,
    HiddenGroup.VUP.value,
)

NOOP_TOKEN = "_noop_"

# --- Параметры guard'ов по умолчанию ---

DEFAULT_MIN_EPISODES = 30
DEFAULT_MIN_ZAP_EVENTS = 20
DEFAULT_MIN_ALPHABET = 2
DEFAULT_N_ITER = 50
DEFAULT_SEED = 42


@dataclass
class HMMRunConfig:
    """Параметры запуска HMM-ветки."""

    enable_hmm: bool = True
    min_episodes: int = DEFAULT_MIN_EPISODES
    min_zap_events: int = DEFAULT_MIN_ZAP_EVENTS
    min_alphabet: int = DEFAULT_MIN_ALPHABET
    n_iter: int = DEFAULT_N_ITER
    random_seed: int = DEFAULT_SEED


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSequence:
    sheet: str
    episode_key: str
    episode_index: int
    tokens: list[str]


def _channel_from_flat_name(name: str) -> str:
    parts = [p.strip() for p in name.split(" | ") if p.strip()]
    return parts[-1] if parts else name


def _episode_key(row: pd.Series, episode_columns: list[str]) -> str:
    pieces: list[str] = []
    for c in episode_columns:
        v = row.get(c)
        if pd.isna(v):
            continue
        pieces.append(str(v))
    return "|".join(pieces) if pieces else ""


def build_observation_sequences(
    frames: dict[str, pd.DataFrame],
    config: ColumnMappingConfig,
) -> tuple[list[EpisodeSequence], list[str]]:
    """Построить последовательности ЗАП-событий по эпизодам.

    Возвращает ``(sequences, alphabet)``. Порядок токенов в эпизоде
    сохраняет порядок строк в df (каждая строка — отдельный
    суб-интервал эпизода; в реальных данных строки, относящиеся к
    одному ``№ эпизода``, могут повторяться).

    Алфавит сортируется детерминированно.
    """

    sequences: list[EpisodeSequence] = []
    alphabet_set: set[str] = set()

    for sheet_name, sheet_mapping in config.sheets.items():
        df = frames.get(sheet_name)
        if df is None or df.empty:
            continue

        zap_cols = [
            c for c in sheet_mapping.roles.get(HiddenGroup.ZAP, []) if c in df.columns
        ]
        episode_cols = [
            c
            for c in sheet_mapping.roles.get(HiddenGroup.EPISODE, [])
            if c in df.columns
        ]
        if not zap_cols:
            continue

        # Предвычислим kind для каждой колонки один раз.
        kinds: dict[str, str] = {}
        for col in zap_cols:
            kind, _ = classify_zap_column(df[col])
            kinds[col] = kind

        # Группируем строки по episode_key; если episode columns не
        # указаны — каждая строка = отдельный эпизод.
        if episode_cols:
            grouper_keys = [
                _episode_key(row, episode_cols) for _, row in df.iterrows()
            ]
        else:
            grouper_keys = [str(i) for i in range(len(df))]

        # Накопим токены per episode_key, сохраняя порядок появления.
        order: list[str] = []
        buckets: dict[str, list[str]] = {}
        for key, (_, row) in zip(grouper_keys, df.iterrows(), strict=True):
            if not key:
                continue
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            tokens_row: list[str] = []
            for col in zap_cols:
                val = row.get(col)
                if pd.isna(val):
                    continue
                kind = kinds[col]
                channel = _channel_from_flat_name(col)
                if kind in {ZAP_KIND_BINARY, ZAP_KIND_COUNT}:
                    try:
                        numeric = float(val)
                    except (TypeError, ValueError):
                        continue
                    if numeric <= 0:
                        continue
                    # count-колонка value=2 — два события одного типа подряд.
                    repeats = int(numeric) if numeric >= 1 else 0
                    tokens_row.extend([channel] * repeats)
                elif kind == ZAP_KIND_CATEGORICAL:
                    token = str(val).strip()
                    if token:
                        tokens_row.append(token)
            buckets[key].extend(tokens_row)

        for idx, key in enumerate(order):
            tokens = buckets[key]
            if not tokens:
                tokens = [NOOP_TOKEN]
            alphabet_set.update(tokens)
            sequences.append(
                EpisodeSequence(
                    sheet=sheet_name,
                    episode_key=key,
                    episode_index=idx,
                    tokens=tokens,
                )
            )

    alphabet = sorted(alphabet_set)
    return sequences, alphabet


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def evaluate_guards(
    baseline: BaselineReport,
    config: ColumnMappingConfig,
    sequences: list[EpisodeSequence],
    alphabet: list[str],
    run_config: HMMRunConfig,
) -> list[WarningItem]:
    """Вернуть список сработавших guard'ов (пустой => HMM разрешена)."""

    failed: list[WarningItem] = []

    if not run_config.enable_hmm:
        failed.append(
            WarningItem(
                code="hmm.disabled",
                message="HMM-ветка отключена в конфигурации анализа.",
                severity=WarningSeverity.INFO,
            )
        )
        return failed

    if config.is_empty():
        failed.append(
            WarningItem(
                code="hmm.guards_failed",
                message="HMM не запущена: не задан column mapping.",
                severity=WarningSeverity.WARNING,
                context={"guard": "mapping_required"},
            )
        )
        return failed

    any_zap = any(
        HiddenGroup.ZAP in sm.roles and sm.roles[HiddenGroup.ZAP]
        for sm in config.sheets.values()
    )
    if not any_zap:
        failed.append(
            WarningItem(
                code="hmm.guards_failed",
                message="HMM не запущена: в mapping нет ни одной ЗАП-колонки.",
                severity=WarningSeverity.WARNING,
                context={"guard": "zap_role_required"},
            )
        )

    total_episodes = sum(baseline.episodes_per_sheet.values())
    if total_episodes < run_config.min_episodes:
        failed.append(
            WarningItem(
                code="hmm.guards_failed",
                message=(
                    f"HMM не запущена: эпизодов {total_episodes} < "
                    f"минимума {run_config.min_episodes}."
                ),
                severity=WarningSeverity.WARNING,
                context={
                    "guard": "min_episodes",
                    "actual": total_episodes,
                    "required": run_config.min_episodes,
                },
            )
        )

    total_events = sum(baseline.zap_events_by_channel.values()) + sum(
        sum(v.values()) for v in baseline.zap_value_counts.values()
    )
    if total_events < run_config.min_zap_events:
        failed.append(
            WarningItem(
                code="hmm.guards_failed",
                message=(
                    f"HMM не запущена: ЗАП-событий {total_events} < "
                    f"минимума {run_config.min_zap_events}."
                ),
                severity=WarningSeverity.WARNING,
                context={
                    "guard": "min_zap_events",
                    "actual": total_events,
                    "required": run_config.min_zap_events,
                },
            )
        )

    non_noop_alphabet = [t for t in alphabet if t != NOOP_TOKEN]
    if len(non_noop_alphabet) < run_config.min_alphabet:
        failed.append(
            WarningItem(
                code="hmm.guards_failed",
                message=(
                    f"HMM не запущена: алфавит наблюдений {len(non_noop_alphabet)} "
                    f"меньше минимального {run_config.min_alphabet}."
                ),
                severity=WarningSeverity.WARNING,
                context={
                    "guard": "min_alphabet",
                    "actual": len(non_noop_alphabet),
                    "required": run_config.min_alphabet,
                },
            )
        )

    if not sequences:
        failed.append(
            WarningItem(
                code="hmm.guards_failed",
                message="HMM не запущена: не удалось построить ни одной последовательности.",
                severity=WarningSeverity.WARNING,
                context={"guard": "no_sequences"},
            )
        )

    return failed


# ---------------------------------------------------------------------------
# Domain-informed initialization
# ---------------------------------------------------------------------------


def _domain_initial_distribution(n_states: int = 3) -> np.ndarray:
    """π: старт с высокой вероятностью в маневрировании."""

    return np.array([0.8, 0.15, 0.05], dtype=float)[:n_states]


def _domain_transition_matrix(n_states: int = 3) -> np.ndarray:
    """A: трёхдиагональная движение по цепочке вперёд + небольшое самопоглощение."""

    a = np.array(
        [
            [0.6, 0.35, 0.05],  # маневрирование -> КФВ
            [0.1, 0.5, 0.4],    # КФВ -> ВУП
            [0.05, 0.1, 0.85],  # ВУП — поглощающее
        ],
        dtype=float,
    )
    return a[:n_states, :n_states]


def _domain_emission_matrix(
    n_states: int, n_observations: int, seed: int
) -> np.ndarray:
    """Ненулевая эмиссия для всех пар state/observation (numerical stability)."""

    rng = np.random.default_rng(seed)
    raw = rng.random((n_states, n_observations)) + 0.1
    return raw / raw.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Fit + evaluate
# ---------------------------------------------------------------------------


def _tokens_to_indices(
    tokens: list[str], vocab: dict[str, int]
) -> list[int]:
    return [vocab[t] for t in tokens]


def fit_hmm(
    sequences: list[EpisodeSequence],
    alphabet: list[str],
    run_config: HMMRunConfig,
) -> tuple[HMMResult, dict[str, Any]] | None:
    """Обучить HMM и собрать :class:`HMMResult`.

    Возвращает ``None`` если `hmmlearn` недоступен или модель не сошлась
    (и при этом guard-check, ведомый ``evaluate_guards``, всё равно
    должен быть пройдён вне этой функции).
    """

    try:
        from hmmlearn.hmm import CategoricalHMM
    except Exception:
        return None

    vocab = {tok: i for i, tok in enumerate(alphabet)}
    n_states = len(STATE_LABELS)
    n_obs = len(alphabet)

    X_list: list[np.ndarray] = []
    lengths: list[int] = []
    for seq in sequences:
        indices = _tokens_to_indices(seq.tokens, vocab)
        if not indices:
            continue
        X_list.append(np.array(indices).reshape(-1, 1))
        lengths.append(len(indices))

    if not X_list:
        return None

    X = np.concatenate(X_list)

    model = CategoricalHMM(
        n_components=n_states,
        n_iter=run_config.n_iter,
        random_state=run_config.random_seed,
        init_params="",
        params="ste",
        tol=1e-4,
    )
    model.startprob_ = _domain_initial_distribution(n_states)
    model.transmat_ = _domain_transition_matrix(n_states)
    model.emissionprob_ = _domain_emission_matrix(
        n_states, n_obs, run_config.random_seed
    )
    model.n_features = n_obs

    try:
        model.fit(X, lengths)
        log_likelihood = float(model.score(X, lengths))
    except Exception:  # noqa: BLE001
        return None

    # per-episode Viterbi
    trajectories: list[HMMTrajectory] = []
    for seq, length in zip(sequences, lengths, strict=True):
        indices = np.array(
            _tokens_to_indices(seq.tokens, vocab)
        ).reshape(-1, 1)
        if indices.size == 0:
            continue
        log_prob, states = model.decode(indices, algorithm="viterbi")
        trajectories.append(
            HMMTrajectory(
                sheet=seq.sheet,
                episode_index=seq.episode_index,
                length=length,
                observation_tokens=seq.tokens,
                state_path=[STATE_LABELS[s] for s in states],
                log_likelihood=float(log_prob),
            )
        )

    # state distribution
    total_state_steps = np.zeros(n_states, dtype=float)
    for tr in trajectories:
        for s in tr.state_path:
            total_state_steps[STATE_LABELS.index(s)] += 1
    denom = total_state_steps.sum() or 1.0
    state_distribution = {
        STATE_LABELS[i]: float(round(total_state_steps[i] / denom, 6))
        for i in range(n_states)
    }

    # sanity check матрицы переходов: преобладание диагонали/поддиагонали
    A = np.array(model.transmat_)
    diag_mass = float(np.diag(A).sum())
    forward_mass = float(np.sum(np.diag(A, k=1))) if n_states > 1 else 0.0
    sanity_ok = (diag_mass + forward_mass) / n_states >= 0.5

    parameters = HMMParameters(
        n_states=n_states,
        n_observations=n_obs,
        state_labels=list(STATE_LABELS),
        observation_labels=list(alphabet),
        initial_distribution=[float(x) for x in model.startprob_],
        transition_matrix=[[float(x) for x in row] for row in model.transmat_],
        emission_matrix=[[float(x) for x in row] for row in model.emissionprob_],
        random_seed=run_config.random_seed,
        n_iter=int(model.monitor_.iter),
        converged=bool(model.monitor_.converged),
        log_likelihood=log_likelihood,
    )

    interpretation = _build_interpretation(
        state_distribution, A, alphabet, model.emissionprob_
    )

    sanity = {
        "transition_diagonal_mass": diag_mass,
        "transition_forward_mass": forward_mass,
        "transition_dominance_ok": bool(sanity_ok),
        "episodes_used": len(trajectories),
    }

    result = HMMResult(
        parameters=parameters,
        trajectories=trajectories,
        state_distribution=state_distribution,
        sanity=sanity,
        interpretation=interpretation,
    )
    return result, sanity


def _build_interpretation(
    state_distribution: dict[str, float],
    A: np.ndarray,
    alphabet: list[str],
    emissions: np.ndarray,
) -> str:
    lines: list[str] = []
    top_state = max(state_distribution.items(), key=lambda kv: kv[1])[0]
    lines.append(
        f"Чаще всего модель находится в состоянии '{top_state}' "
        f"({state_distribution[top_state] * 100:.1f}% времени)."
    )

    # самые вероятные переходы
    for i, from_state in enumerate(STATE_LABELS):
        j = int(np.argmax(A[i]))
        to_state = STATE_LABELS[j]
        lines.append(
            f"Из '{from_state}' модель чаще всего переходит в "
            f"'{to_state}' (p={A[i][j]:.2f})."
        )

    # ключевые эмиссии per state
    for i, state in enumerate(STATE_LABELS):
        top_idx = int(np.argmax(emissions[i]))
        lines.append(
            f"Состояние '{state}' чаще всего порождает наблюдение "
            f"'{alphabet[top_idx]}' (p={emissions[i][top_idx]:.2f})."
        )

    lines.append(
        "Все формулировки — вероятностные; при недостаточности данных "
        "HMM не запускается и результат остаётся baseline_only."
    )
    return "\n".join(lines)


__all__ = [
    "HMMRunConfig",
    "EpisodeSequence",
    "STATE_LABELS",
    "NOOP_TOKEN",
    "build_observation_sequences",
    "evaluate_guards",
    "fit_hmm",
]
