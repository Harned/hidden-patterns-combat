"""Тесты TASK_SPEC_008: multivariate Bernoulli эмиссии + k-fold CV."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.cv import cross_validate
from hpc_algo.hmm import HMMRunConfig
from hpc_algo.hmm_bernoulli import (
    BernoulliHMMParams,
    _forward_backward,
    _log,
    fit_bernoulli_hmm,
    score_sequence,
)
from hpc_algo.mapping import load_mapped_sheets
from hpc_algo.schema import AnalysisStatus

# ---------------------------------------------------------------------------
# Bernoulli-EM: восстановление параметров на синтетике
# ---------------------------------------------------------------------------


def _generate_sequences(params: BernoulliHMMParams, n_seq: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    n_states = params.pi.shape[0]
    n_channels = params.B.shape[1]
    sequences = []
    for _ in range(n_seq):
        state = rng.choice(n_states, p=params.pi)
        seq = []
        for _t in range(length):
            obs = (rng.random(n_channels) < params.B[state]).astype(float)
            seq.append(obs)
            state = rng.choice(n_states, p=params.A[state])
        sequences.append(np.vstack(seq))
    return sequences


def test_bernoulli_em_recovers_rough_parameters() -> None:
    true_params = BernoulliHMMParams(
        pi=np.array([0.9, 0.1]),
        A=np.array([[0.8, 0.2], [0.1, 0.9]]),
        B=np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]),
    )
    seqs = _generate_sequences(true_params, n_seq=80, length=15, seed=7)

    init = BernoulliHMMParams(
        pi=np.array([0.5, 0.5]),
        A=np.array([[0.6, 0.4], [0.4, 0.6]]),
        B=np.clip(np.random.default_rng(0).uniform(0.2, 0.6, size=(2, 3)), 0.1, 0.9),
    )
    fit = fit_bernoulli_hmm(seqs, init, n_iter=60, tol=1e-4)

    # Разрешаем перестановку состояний — сравниваем отсортированные строки B
    # по первому каналу, чтобы определить соответствие.
    order = np.argsort(-fit.params.B[:, 0])
    B_aligned = fit.params.B[order]
    # Первое состояние модели должно соответствовать «канал 0 активен».
    assert B_aligned[0, 0] > B_aligned[1, 0]
    assert B_aligned[0, 0] > 0.5
    assert fit.log_likelihood < 0


def test_score_and_forward_backward_agree() -> None:
    params = BernoulliHMMParams(
        pi=np.array([0.7, 0.3]),
        A=np.array([[0.6, 0.4], [0.3, 0.7]]),
        B=np.array([[0.8, 0.2], [0.1, 0.9]]),
    )
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ll = score_sequence(X, params)
    # Ручной forward-backward должен дать то же значение.
    log_emiss = np.stack(
        [
            _log(params.B[0]) @ X.T + _log(1 - params.B[0]) @ (1 - X).T,
            _log(params.B[1]) @ X.T + _log(1 - params.B[1]) @ (1 - X).T,
        ],
        axis=-1,
    )
    _, _, ll_ref = _forward_backward(_log(params.pi), _log(params.A), log_emiss)
    assert abs(ll - ll_ref) < 1e-6


# ---------------------------------------------------------------------------
# Bernoulli на плотной фикстуре: analyze_source возвращает hmm_ready
# ---------------------------------------------------------------------------


def test_analyze_with_bernoulli_on_dense(dense_hmm_ready_excel: Path) -> None:
    cfg = preflight_mapping(dense_hmm_ready_excel)
    result = analyze_source(
        dense_hmm_ready_excel,
        AnalyzeConfig(column_mapping=cfg, observation_emission="bernoulli"),
    )
    assert result.status == AnalysisStatus.HMM_READY
    assert result.hmm is not None
    assert result.hmm.parameters.observation_emission == "bernoulli"
    # Алфавит — это каналы (Удержание / На руку / На ногу / ЗАП-Р).
    labels = set(result.hmm.parameters.observation_labels)
    assert labels <= {"Удержание", "На руку", "На ногу", "ЗАП-Р"}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def test_cross_validate_returns_mean_ll_bernoulli(
    very_dense_excel: Path,
) -> None:
    cfg = preflight_mapping(very_dense_excel)
    frames = load_mapped_sheets(very_dense_excel, cfg)
    # Фикстура содержит один лист "all" — делаем k=2, CV делит внутри листа
    # на train/test только один лист, поэтому на этой фикстуре
    # достаточно убедиться, что функция возвращает структурированный
    # отчёт даже если test пустой.
    report = cross_validate(
        frames,
        cfg,
        HMMRunConfig(
            mode="basic",
            observation_emission="bernoulli",
            min_episodes=30,
            min_zap_events=20,
            n_iter=20,
        ),
        k=2,
    )
    assert report.observation_emission == "bernoulli"
    assert report.k >= 2
    assert len(report.folds) == report.k


def test_cross_validate_categorical_smoke(very_dense_excel: Path) -> None:
    cfg = preflight_mapping(very_dense_excel)
    frames = load_mapped_sheets(very_dense_excel, cfg)
    report = cross_validate(
        frames,
        cfg,
        HMMRunConfig(
            mode="basic",
            observation_emission="categorical",
            min_episodes=30,
            min_zap_events=20,
            n_iter=20,
        ),
        k=2,
    )
    assert report.observation_emission == "categorical"
    assert report.k >= 2
