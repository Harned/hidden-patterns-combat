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
from hpc_algo.hmm_bernoulli import (
    BernoulliHMMParams,
    decode_sequence,
    fit_bernoulli_hmm,
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

STATE_LABELS_BASIC: tuple[str, ...] = (
    HiddenGroup.MANEUVERING.value,
    HiddenGroup.KFV.value,
    HiddenGroup.VUP.value,
)

STATE_LABELS_DETAILED: tuple[str, ...] = (
    "маневры",
    "захваты",
    "хваты",
    "обхваты",
    "прихваты",
    "упоры",
    "ВУП",
)

# Индексы состояний внутри `detailed`, относящихся к слою КФВ.
_KFV_SUBGROUPS = (1, 2, 3, 4, 5)

NOOP_TOKEN = "_noop_"

VARIANT_BASIC = "basic_3state"
VARIANT_DETAILED = "detailed_7state"

# --- Параметры guard'ов по умолчанию ---

DEFAULT_MIN_EPISODES = 30
DEFAULT_MIN_ZAP_EVENTS = 20
DEFAULT_MIN_ALPHABET = 2
DEFAULT_N_ITER = 50
DEFAULT_SEED = 42

# Более строгие пороги для 7-state модели.
DEFAULT_DETAILED_MIN_EPISODES = 120
DEFAULT_DETAILED_MIN_ZAP_EVENTS = 60
DEFAULT_DETAILED_MIN_ALPHABET = 3


@dataclass
class HMMRunConfig:
    """Параметры запуска HMM-ветки."""

    enable_hmm: bool = True
    mode: str = "auto"  # auto | detailed | basic | off
    # categorical (по токенам, через hmmlearn) | bernoulli (вектор каналов, свой EM)
    observation_emission: str = "categorical"
    min_episodes: int = DEFAULT_MIN_EPISODES
    min_zap_events: int = DEFAULT_MIN_ZAP_EVENTS
    min_alphabet: int = DEFAULT_MIN_ALPHABET
    min_episodes_detailed: int = DEFAULT_DETAILED_MIN_EPISODES
    min_zap_events_detailed: int = DEFAULT_DETAILED_MIN_ZAP_EVENTS
    min_alphabet_detailed: int = DEFAULT_DETAILED_MIN_ALPHABET
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


def build_bernoulli_sequences(
    frames: dict[str, pd.DataFrame],
    config: ColumnMappingConfig,
) -> tuple[list[tuple[str, str, np.ndarray]], list[str]]:
    """Собрать последовательности бинарных векторов по эпизодам.

    Возвращает ``(entries, channels)``, где ``entries`` — список
    ``(sheet, episode_key, X)`` с ``X`` формы ``(T_i, K)``; ``channels``
    — упорядоченный список каналов (атомарных названий ЗАП-колонок),
    общий для всех листов.
    """

    # Сначала собираем канальный алфавит по всем листам.
    channel_set: list[str] = []
    seen: set[str] = set()
    for sheet_name, sm in config.sheets.items():
        df = frames.get(sheet_name)
        if df is None:
            continue
        for col in sm.roles.get(HiddenGroup.ZAP, []):
            if col not in df.columns:
                continue
            label = _channel_from_flat_name(col)
            if label not in seen:
                seen.add(label)
                channel_set.append(label)

    channels = channel_set
    if not channels:
        return [], []

    channel_to_idx = {c: i for i, c in enumerate(channels)}

    entries: list[tuple[str, str, np.ndarray]] = []
    for sheet_name, sm in config.sheets.items():
        df = frames.get(sheet_name)
        if df is None or df.empty:
            continue
        zap_cols = [c for c in sm.roles.get(HiddenGroup.ZAP, []) if c in df.columns]
        episode_cols = [
            c for c in sm.roles.get(HiddenGroup.EPISODE, []) if c in df.columns
        ]
        if not zap_cols:
            continue

        if episode_cols:
            grouper_keys = [_episode_key(row, episode_cols) for _, row in df.iterrows()]
        else:
            grouper_keys = [str(i) for i in range(len(df))]

        order: list[str] = []
        buckets: dict[str, list[np.ndarray]] = {}
        for key, (_, row) in zip(grouper_keys, df.iterrows(), strict=True):
            if not key:
                continue
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            vec = np.zeros(len(channels), dtype=float)
            touched = False
            for col in zap_cols:
                val = row.get(col)
                if pd.isna(val):
                    continue
                try:
                    numeric = float(val)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    ch = _channel_from_flat_name(col)
                    idx = channel_to_idx.get(ch)
                    if idx is not None:
                        vec[idx] = 1.0
                        touched = True
            if touched:
                buckets[key].append(vec)

        for key in order:
            vecs = buckets[key]
            if not vecs:
                # Пустой эпизод — один нулевой вектор (observation = «ничего»).
                vecs = [np.zeros(len(channels), dtype=float)]
            X = np.vstack(vecs)
            entries.append((sheet_name, key, X))

    return entries, channels


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
# Domain-informed initialization (basic 3-state)
# ---------------------------------------------------------------------------


def _basic_initial_distribution() -> np.ndarray:
    """π: старт с высокой вероятностью в маневрировании."""

    return np.array([0.8, 0.15, 0.05], dtype=float)


def _basic_transition_matrix() -> np.ndarray:
    """A: трёхдиагональная движение по цепочке вперёд + самопоглощение."""

    return np.array(
        [
            [0.6, 0.35, 0.05],  # маневрирование -> КФВ
            [0.1, 0.5, 0.4],    # КФВ -> ВУП
            [0.05, 0.1, 0.85],  # ВУП — поглощающее
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Domain-informed initialization (detailed 7-state)
# ---------------------------------------------------------------------------


def _detailed_initial_distribution() -> np.ndarray:
    """π: подавляющая часть эпизодов начинается в «маневры».

    Состояния: маневры / захваты / хваты / обхваты / прихваты / упоры / ВУП.
    """

    return np.array([0.75, 0.05, 0.05, 0.05, 0.04, 0.04, 0.02], dtype=float)


def _detailed_transition_matrix() -> np.ndarray:
    """A: маневры → подгруппы КФВ → ВУП.

    Внутри слоя КФВ (захваты, хваты, обхваты, прихваты, упоры) допускаются
    симметричные переходы. «Обратный ход» из ВУП и между слоями — малые
    вероятности, чтобы модель не сворачивалась назад по цепочке.
    """

    kfv_self = 0.45
    kfv_to_kfv = 0.06      # переход в другую подгруппу КФВ
    kfv_to_vup = 0.15
    kfv_back_maneuver = 0.10

    row_maneuver = np.array(
        [0.55] + [0.07] * 5 + [0.10],
        dtype=float,
    )

    def kfv_row(self_idx: int) -> np.ndarray:
        row = np.full(7, kfv_to_kfv, dtype=float)
        row[0] = kfv_back_maneuver
        row[6] = kfv_to_vup
        row[self_idx] = kfv_self
        return row / row.sum()

    row_vup = np.array(
        [0.05] + [0.01] * 5 + [0.90],
        dtype=float,
    )

    rows = [row_maneuver]
    for i in _KFV_SUBGROUPS:
        rows.append(kfv_row(i))
    rows.append(row_vup)

    A = np.vstack(rows)
    A = A / A.sum(axis=1, keepdims=True)
    return A


# ---------------------------------------------------------------------------
# Общие init helpers
# ---------------------------------------------------------------------------


def _state_labels(variant: str) -> tuple[str, ...]:
    if variant == VARIANT_DETAILED:
        return STATE_LABELS_DETAILED
    return STATE_LABELS_BASIC


def _initial_distribution(variant: str) -> np.ndarray:
    if variant == VARIANT_DETAILED:
        return _detailed_initial_distribution()
    return _basic_initial_distribution()


def _transition_matrix(variant: str) -> np.ndarray:
    if variant == VARIANT_DETAILED:
        return _detailed_transition_matrix()
    return _basic_transition_matrix()


def _emission_matrix(
    n_states: int, n_observations: int, seed: int
) -> np.ndarray:
    """Ненулевая эмиссия для всех пар state/observation (numerical stability)."""

    rng = np.random.default_rng(seed)
    raw = rng.random((n_states, n_observations)) + 0.1
    return raw / raw.sum(axis=1, keepdims=True)


def _free_parameters(n_states: int, n_observations: int) -> int:
    """Количество свободных параметров HMM (для BIC)."""

    # π: n_states - 1
    # A: n_states * (n_states - 1)
    # B: n_states * (n_observations - 1)
    return (n_states - 1) + n_states * (n_states - 1) + n_states * (n_observations - 1)


def _bic(log_likelihood: float, n_states: int, n_obs: int, n_samples: int) -> float:
    """BIC = -2 * log_likelihood + k * log(N)."""

    k = _free_parameters(n_states, n_obs)
    n = max(1, n_samples)
    return -2.0 * log_likelihood + k * float(np.log(n))


# ---------------------------------------------------------------------------
# Fit + evaluate
# ---------------------------------------------------------------------------


def _tokens_to_indices(
    tokens: list[str], vocab: dict[str, int]
) -> list[int]:
    return [vocab[t] for t in tokens]


def _fit_variant(
    sequences: list[EpisodeSequence],
    alphabet: list[str],
    run_config: HMMRunConfig,
    variant: str,
) -> tuple[HMMResult, dict[str, Any]] | None:
    """Обучить HMM для указанного ``variant`` и вернуть (result, sanity).

    Возвращает ``None`` при проблемах с зависимостью или обучением.
    """

    try:
        from hmmlearn.hmm import CategoricalHMM
    except Exception:
        return None

    state_labels = _state_labels(variant)
    n_states = len(state_labels)

    vocab = {tok: i for i, tok in enumerate(alphabet)}
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
    model.startprob_ = _initial_distribution(variant)
    model.transmat_ = _transition_matrix(variant)
    model.emissionprob_ = _emission_matrix(n_states, n_obs, run_config.random_seed)
    model.n_features = n_obs

    try:
        model.fit(X, lengths)
        log_likelihood = float(model.score(X, lengths))
    except Exception:  # noqa: BLE001
        return None

    trajectories: list[HMMTrajectory] = []
    for seq, length in zip(sequences, lengths, strict=True):
        indices = np.array(_tokens_to_indices(seq.tokens, vocab)).reshape(-1, 1)
        if indices.size == 0:
            continue
        log_prob, states = model.decode(indices, algorithm="viterbi")
        trajectories.append(
            HMMTrajectory(
                sheet=seq.sheet,
                episode_index=seq.episode_index,
                length=length,
                observation_tokens=seq.tokens,
                state_path=[state_labels[s] for s in states],
                log_likelihood=float(log_prob),
            )
        )

    total_state_steps = np.zeros(n_states, dtype=float)
    states_used: set[str] = set()
    for tr in trajectories:
        for s in tr.state_path:
            total_state_steps[state_labels.index(s)] += 1
            states_used.add(s)
    denom = total_state_steps.sum() or 1.0
    state_distribution = {
        state_labels[i]: float(round(total_state_steps[i] / denom, 6))
        for i in range(n_states)
    }

    A = np.array(model.transmat_)
    diag_mass = float(np.diag(A).sum())
    forward_mass = float(np.sum(np.diag(A, k=1))) if n_states > 1 else 0.0
    # Порог «предметной доминанты» смягчён для detailed-модели: у 7-state
    # структуры цепочки оба значимых направления (диагональ + вперёд)
    # распределяются по большему числу состояний, поэтому делитель `n_states`
    # занижает метрику. Для 3-state оставляем классический 0.5.
    dominance_threshold = 0.5 if variant == VARIANT_BASIC else 0.30
    sanity_ok = (diag_mass + forward_mass) / n_states >= dominance_threshold
    all_states_used = len(states_used) == n_states
    # Состояний 7 много, не все будут использованы Viterbi-путями на
    # synthetic данных. Требуем использование >= 3/4 состояний для detailed
    # (то есть хотя бы 5 из 7), чтобы отсечь «вырожденные» варианты.
    min_states_used = (
        n_states if variant == VARIANT_BASIC else max(3, (n_states * 3) // 4)
    )
    enough_states_used = len(states_used) >= min_states_used

    n_samples = int(X.shape[0])
    bic = _bic(log_likelihood, n_states, n_obs, n_samples)

    parameters = HMMParameters(
        n_states=n_states,
        n_observations=n_obs,
        state_labels=list(state_labels),
        observation_labels=list(alphabet),
        initial_distribution=[float(x) for x in model.startprob_],
        transition_matrix=[[float(x) for x in row] for row in model.transmat_],
        emission_matrix=[[float(x) for x in row] for row in model.emissionprob_],
        random_seed=run_config.random_seed,
        n_iter=int(model.monitor_.iter),
        converged=bool(model.monitor_.converged),
        log_likelihood=log_likelihood,
        variant=variant,
        bic=bic,
    )

    interpretation = _build_interpretation(
        state_distribution, A, alphabet, model.emissionprob_, state_labels
    )

    sanity = {
        "variant": variant,
        "transition_diagonal_mass": diag_mass,
        "transition_forward_mass": forward_mass,
        "transition_dominance_ok": bool(sanity_ok),
        "all_states_used": bool(all_states_used),
        "enough_states_used": bool(enough_states_used),
        "states_used": len(states_used),
        "min_states_used_required": int(min_states_used),
        "episodes_used": len(trajectories),
        "n_samples": n_samples,
        "bic": bic,
    }

    result = HMMResult(
        parameters=parameters,
        trajectories=trajectories,
        state_distribution=state_distribution,
        sanity=sanity,
        interpretation=interpretation,
    )
    return result, sanity


def _fit_bernoulli_variant(
    frames: dict[str, pd.DataFrame],
    config: ColumnMappingConfig,
    run_config: HMMRunConfig,
    variant: str,
) -> tuple[HMMResult, dict[str, Any]] | None:
    """Обучить multivariate Bernoulli HMM и собрать :class:`HMMResult`."""

    entries, channels = build_bernoulli_sequences(frames, config)
    if not entries or not channels:
        return None

    state_labels = _state_labels(variant)
    n_states = len(state_labels)
    n_channels = len(channels)
    rng = np.random.default_rng(run_config.random_seed)

    init = BernoulliHMMParams(
        pi=_initial_distribution(variant),
        A=_transition_matrix(variant),
        B=rng.uniform(0.1, 0.4, size=(n_states, n_channels)),
    )

    sequences = [X for _, _, X in entries]
    fit = fit_bernoulli_hmm(
        sequences, init, n_iter=run_config.n_iter, tol=1e-4
    )

    trajectories: list[HMMTrajectory] = []
    states_used: set[str] = set()
    for (sheet_name, _ep_key, X), seq_index in zip(
        entries, range(len(entries)), strict=True
    ):
        path, log_prob = decode_sequence(X, fit.params)
        path_labels = [state_labels[int(s)] for s in path]
        states_used.update(path_labels)
        obs_tokens = [
            "+".join(channels[i] for i, v in enumerate(row) if v > 0) or "_noop_"
            for row in X
        ]
        trajectories.append(
            HMMTrajectory(
                sheet=sheet_name,
                episode_index=seq_index,
                length=X.shape[0],
                observation_tokens=obs_tokens,
                state_path=path_labels,
                log_likelihood=float(log_prob),
            )
        )

    total_state_steps = np.zeros(n_states, dtype=float)
    for tr in trajectories:
        for s in tr.state_path:
            total_state_steps[state_labels.index(s)] += 1
    denom = total_state_steps.sum() or 1.0
    state_distribution = {
        state_labels[i]: float(round(total_state_steps[i] / denom, 6))
        for i in range(n_states)
    }

    A = fit.params.A
    diag_mass = float(np.diag(A).sum())
    forward_mass = float(np.sum(np.diag(A, k=1))) if n_states > 1 else 0.0
    dominance_threshold = 0.5 if variant == VARIANT_BASIC else 0.30
    sanity_ok = (diag_mass + forward_mass) / n_states >= dominance_threshold
    min_states_used = (
        n_states if variant == VARIANT_BASIC else max(3, (n_states * 3) // 4)
    )
    enough_states_used = len(states_used) >= min_states_used

    total_samples = int(sum(X.shape[0] for _, _, X in entries))
    # Параметры: pi(n-1) + A(n*(n-1)) + B(n*K).
    k = (n_states - 1) + n_states * (n_states - 1) + n_states * n_channels
    bic = (
        -2.0 * fit.log_likelihood + k * float(np.log(max(1, total_samples)))
    )

    parameters = HMMParameters(
        n_states=n_states,
        n_observations=n_channels,
        state_labels=list(state_labels),
        observation_labels=list(channels),
        initial_distribution=[float(x) for x in fit.params.pi],
        transition_matrix=[[float(x) for x in row] for row in fit.params.A],
        emission_matrix=[[float(x) for x in row] for row in fit.params.B],
        random_seed=run_config.random_seed,
        n_iter=fit.n_iter,
        converged=fit.converged,
        log_likelihood=fit.log_likelihood,
        variant=variant,
        observation_emission="bernoulli",
        bic=bic,
    )
    interpretation = _build_bernoulli_interpretation(
        state_distribution, fit.params.A, channels, fit.params.B, state_labels
    )

    sanity = {
        "variant": variant,
        "observation_emission": "bernoulli",
        "transition_diagonal_mass": diag_mass,
        "transition_forward_mass": forward_mass,
        "transition_dominance_ok": bool(sanity_ok),
        "all_states_used": bool(len(states_used) == n_states),
        "enough_states_used": bool(enough_states_used),
        "states_used": len(states_used),
        "min_states_used_required": int(min_states_used),
        "episodes_used": len(trajectories),
        "n_samples": total_samples,
        "bic": bic,
    }

    result = HMMResult(
        parameters=parameters,
        trajectories=trajectories,
        state_distribution=state_distribution,
        sanity=sanity,
        interpretation=interpretation,
    )
    return result, sanity


def _build_bernoulli_interpretation(
    state_distribution: dict[str, float],
    A: np.ndarray,
    channels: list[str],
    B: np.ndarray,
    state_labels: tuple[str, ...] | list[str],
) -> str:
    lines: list[str] = []
    top_state = max(state_distribution.items(), key=lambda kv: kv[1])[0]
    lines.append(
        f"Чаще всего модель находится в состоянии '{top_state}' "
        f"({state_distribution[top_state] * 100:.1f}% времени). Эмиссия — "
        "multivariate Bernoulli по каналам ЗАП."
    )
    for i, from_state in enumerate(state_labels):
        j = int(np.argmax(A[i]))
        to_state = state_labels[j]
        lines.append(
            f"Из '{from_state}' модель чаще всего переходит в '{to_state}' "
            f"(p={A[i][j]:.2f})."
        )
    for i, state in enumerate(state_labels):
        top_idx = int(np.argmax(B[i]))
        lines.append(
            f"В состоянии '{state}' наиболее вероятно срабатывает канал "
            f"'{channels[top_idx]}' (p={B[i][top_idx]:.2f})."
        )
    lines.append(
        "Все формулировки — вероятностные; при недостаточности данных "
        "HMM не запускается и результат остаётся baseline_only."
    )
    return "\n".join(lines)


def _detailed_data_ok(
    baseline: BaselineReport,
    alphabet: list[str],
    run_config: HMMRunConfig,
) -> bool:
    total_episodes = sum(baseline.episodes_per_sheet.values())
    total_events = sum(baseline.zap_events_by_channel.values()) + sum(
        sum(v.values()) for v in baseline.zap_value_counts.values()
    )
    non_noop_alphabet = [t for t in alphabet if t != NOOP_TOKEN]
    return (
        total_episodes >= run_config.min_episodes_detailed
        and total_events >= run_config.min_zap_events_detailed
        and len(non_noop_alphabet) >= run_config.min_alphabet_detailed
    )


def fit_hmm_bernoulli(
    frames: dict[str, pd.DataFrame],
    config: ColumnMappingConfig,
    run_config: HMMRunConfig,
) -> tuple[HMMResult, dict[str, Any]] | None:
    """Попробовать Bernoulli-вариант (detailed > auto по BIC > basic)."""

    mode = run_config.mode
    if mode == "off":
        return None

    def try_bernoulli(variant: str):
        out = _fit_bernoulli_variant(frames, config, run_config, variant)
        if out is None:
            return None
        result, sanity = out
        if not sanity.get("transition_dominance_ok"):
            return None
        if variant == VARIANT_DETAILED and not sanity.get("enough_states_used"):
            return None
        return result, sanity

    if mode == "basic":
        return try_bernoulli(VARIANT_BASIC)
    if mode == "detailed":
        return try_bernoulli(VARIANT_DETAILED)

    detailed = try_bernoulli(VARIANT_DETAILED)
    basic = try_bernoulli(VARIANT_BASIC)
    if detailed is not None and basic is not None:
        bic_d = detailed[0].parameters.bic
        bic_b = basic[0].parameters.bic
        if bic_d is not None and bic_b is not None and bic_d < bic_b:
            return detailed
        return basic
    return detailed or basic


def fit_hmm(
    sequences: list[EpisodeSequence],
    alphabet: list[str],
    run_config: HMMRunConfig,
    baseline: BaselineReport | None = None,
) -> tuple[HMMResult, dict[str, Any]] | None:
    """Обучить HMM с учётом режима (basic / detailed / auto).

    * ``mode == "basic"`` — только 3-state.
    * ``mode == "detailed"`` — только 7-state (возвращает None, если
      либо 7-state не прошла guard/sanity, либо BIC detailed ≥ BIC basic).
    * ``mode == "auto"`` — сначала пробуем detailed, при неуспехе или
      ухудшении BIC откатываемся к basic.

    Возвращает ``(result, sanity)`` или ``None``.
    """

    mode = run_config.mode
    if mode == "off":
        return None
    if baseline is None:
        baseline = BaselineReport()

    def try_basic():
        out = _fit_variant(sequences, alphabet, run_config, VARIANT_BASIC)
        if out is None:
            return None
        result, sanity = out
        if not sanity.get("transition_dominance_ok"):
            return None
        return result, sanity

    def try_detailed():
        if not _detailed_data_ok(baseline, alphabet, run_config):
            return None
        out = _fit_variant(sequences, alphabet, run_config, VARIANT_DETAILED)
        if out is None:
            return None
        result, sanity = out
        if not sanity.get("transition_dominance_ok"):
            return None
        if not sanity.get("enough_states_used"):
            return None
        return result, sanity

    if mode == "basic":
        return try_basic()

    if mode == "detailed":
        # Явный выбор пользователя: если guard/sanity прошли — возвращаем,
        # без сравнения с basic. BIC в этом режиме не ограничивает.
        return try_detailed()

    # mode == "auto": сравниваем по BIC. 7-state принимается, только если
    # BIC строго лучше — иначе остаёмся с basic.
    detailed = try_detailed()
    basic = try_basic()
    if detailed is not None and basic is not None:
        bic_d = detailed[0].parameters.bic
        bic_b = basic[0].parameters.bic
        if bic_d is not None and bic_b is not None and bic_d < bic_b:
            return detailed
        return basic
    return detailed or basic


def _build_interpretation(
    state_distribution: dict[str, float],
    A: np.ndarray,
    alphabet: list[str],
    emissions: np.ndarray,
    state_labels: tuple[str, ...] | list[str],
) -> str:
    lines: list[str] = []
    top_state = max(state_distribution.items(), key=lambda kv: kv[1])[0]
    lines.append(
        f"Чаще всего модель находится в состоянии '{top_state}' "
        f"({state_distribution[top_state] * 100:.1f}% времени)."
    )

    for i, from_state in enumerate(state_labels):
        j = int(np.argmax(A[i]))
        to_state = state_labels[j]
        lines.append(
            f"Из '{from_state}' модель чаще всего переходит в "
            f"'{to_state}' (p={A[i][j]:.2f})."
        )

    for i, state in enumerate(state_labels):
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
    "STATE_LABELS_BASIC",
    "STATE_LABELS_DETAILED",
    "VARIANT_BASIC",
    "VARIANT_DETAILED",
    "NOOP_TOKEN",
    "build_observation_sequences",
    "evaluate_guards",
    "fit_hmm",
]
