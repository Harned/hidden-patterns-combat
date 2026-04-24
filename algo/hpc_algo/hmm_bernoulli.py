"""Multivariate Bernoulli HMM и лёгкий EM поверх NumPy.

Реализация лаконичная и самодостаточная: hmmlearn не предоставляет
multivariate Bernoulli в общем виде, а таскать scikit-learn-specific
нетривиальные зависимости не хотим. Нам достаточно:

* forward-backward на log-space,
* decode/Viterbi на log-space,
* EM-обновление ``π``, ``A``, ``B``.

Наблюдение на шаге — бинарный вектор длины ``K`` (число каналов ЗАП),
элементы ``∈ {0, 1}``. ``B[i, k]`` — ``p(channel_k = 1 | state_i)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOG_EPS = 1e-12


def _log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _LOG_EPS, 1.0))


def _log_bernoulli_likelihood(
    X: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Матрица log p(x_t | state) размера (T, n_states)."""

    log_B = _log(B)
    log_one_minus_B = _log(1.0 - B)
    # X: (T, K); B: (n_states, K)
    # result[t, i] = sum_k x_t_k * log B[i, k] + (1-x_t_k) * log(1-B[i, k])
    return X @ log_B.T + (1.0 - X) @ log_one_minus_B.T


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if axis is None:
        return res.squeeze()
    return res.squeeze(axis=axis)


@dataclass
class BernoulliHMMParams:
    pi: np.ndarray          # (n_states,)
    A: np.ndarray           # (n_states, n_states)
    B: np.ndarray           # (n_states, n_channels), p(channel=1|state)


@dataclass
class BernoulliHMMFitResult:
    params: BernoulliHMMParams
    log_likelihood: float
    n_iter: int
    converged: bool


def _forward_backward(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_emiss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Forward-backward в log-space для одной последовательности."""

    T, n_states = log_emiss.shape
    log_alpha = np.full((T, n_states), -np.inf)
    log_alpha[0] = log_pi + log_emiss[0]
    for t in range(1, T):
        # log_alpha[t, j] = log_emiss[t, j] + logsumexp_i(log_alpha[t-1, i] + log_A[i, j])
        stacked = log_alpha[t - 1, :, None] + log_A
        log_alpha[t] = log_emiss[t] + _logsumexp(stacked, axis=0)

    log_beta = np.full((T, n_states), -np.inf)
    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        # log_beta[t, i] = logsumexp_j(log_A[i, j] + log_emiss[t+1, j] + log_beta[t+1, j])
        stacked = log_A + (log_emiss[t + 1] + log_beta[t + 1])
        log_beta[t] = _logsumexp(stacked, axis=1)

    log_seq = _logsumexp(log_alpha[-1])
    return log_alpha, log_beta, float(log_seq)


def _viterbi(
    log_pi: np.ndarray,
    log_A: np.ndarray,
    log_emiss: np.ndarray,
) -> tuple[np.ndarray, float]:
    T, n_states = log_emiss.shape
    dp = np.full((T, n_states), -np.inf)
    back = np.zeros((T, n_states), dtype=int)
    dp[0] = log_pi + log_emiss[0]
    for t in range(1, T):
        scores = dp[t - 1, :, None] + log_A
        back[t] = np.argmax(scores, axis=0)
        dp[t] = log_emiss[t] + np.max(scores, axis=0)
    final = int(np.argmax(dp[-1]))
    path = np.zeros(T, dtype=int)
    path[-1] = final
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(np.max(dp[-1]))


def fit_bernoulli_hmm(
    sequences: list[np.ndarray],
    init: BernoulliHMMParams,
    n_iter: int = 50,
    tol: float = 1e-4,
) -> BernoulliHMMFitResult:
    """EM для multivariate Bernoulli HMM по нескольким последовательностям.

    Каждая последовательность — (T_i, K)-массив из 0/1 (можно целые).
    """

    pi = init.pi.astype(float).copy()
    A = init.A.astype(float).copy()
    B = np.clip(init.B.astype(float).copy(), 1e-4, 1.0 - 1e-4)
    n_states = pi.shape[0]

    last_ll = -np.inf
    converged = False
    iter_used = 0

    for it in range(1, n_iter + 1):
        iter_used = it
        log_pi = _log(pi)
        log_A = _log(A)

        # Accumulators.
        expected_start = np.zeros(n_states)
        expected_trans = np.zeros((n_states, n_states))
        expected_channel_on = np.zeros(B.shape)
        expected_state_occ = np.zeros(n_states)

        total_ll = 0.0
        for X in sequences:
            if X.shape[0] == 0:
                continue
            log_emiss = _log_bernoulli_likelihood(X, B)
            log_alpha, log_beta, log_seq = _forward_backward(
                log_pi, log_A, log_emiss
            )
            if not np.isfinite(log_seq):
                # На дегенерированной последовательности пропускаем вклад.
                continue
            total_ll += log_seq

            # gamma_t,i = exp(log_alpha + log_beta - log_seq)
            gamma = np.exp(log_alpha + log_beta - log_seq)
            # xi_t,i,j = exp(log_alpha_t,i + log_A_ij + log_emiss_{t+1,j} + log_beta_{t+1,j} - log_seq)
            T = X.shape[0]
            expected_start += gamma[0]
            if T > 1:
                log_xi = (
                    log_alpha[:-1, :, None]
                    + log_A[None, :, :]
                    + (log_emiss[1:, None, :] + log_beta[1:, None, :])
                    - log_seq
                )
                xi = np.exp(log_xi)
                expected_trans += xi.sum(axis=0)
            expected_state_occ += gamma.sum(axis=0)
            expected_channel_on += gamma.T @ X  # (n_states, K)

        # M-step.
        pi_new = expected_start / max(expected_start.sum(), _LOG_EPS)
        row_sums = expected_trans.sum(axis=1, keepdims=True)
        A_new = np.where(row_sums > 0, expected_trans / np.maximum(row_sums, _LOG_EPS), A)
        B_new = np.clip(
            expected_channel_on
            / np.maximum(expected_state_occ[:, None], _LOG_EPS),
            1e-4,
            1.0 - 1e-4,
        )

        pi, A, B = pi_new, A_new, B_new

        if (
            np.isfinite(total_ll)
            and np.isfinite(last_ll)
            and abs(total_ll - last_ll) < tol
        ):
            converged = True
            last_ll = total_ll
            break
        last_ll = total_ll

    return BernoulliHMMFitResult(
        params=BernoulliHMMParams(pi=pi, A=A, B=B),
        log_likelihood=float(last_ll),
        n_iter=iter_used,
        converged=converged,
    )


def score_sequence(
    X: np.ndarray, params: BernoulliHMMParams
) -> float:
    """Log-likelihood одной последовательности (для CV)."""

    if X.shape[0] == 0:
        return 0.0
    log_pi = _log(params.pi)
    log_A = _log(params.A)
    log_emiss = _log_bernoulli_likelihood(X, params.B)
    _, _, log_seq = _forward_backward(log_pi, log_A, log_emiss)
    return float(log_seq)


def decode_sequence(
    X: np.ndarray, params: BernoulliHMMParams
) -> tuple[np.ndarray, float]:
    """Viterbi-путь и log-prob для одной последовательности."""

    log_pi = _log(params.pi)
    log_A = _log(params.A)
    log_emiss = _log_bernoulli_likelihood(X, params.B)
    return _viterbi(log_pi, log_A, log_emiss)
