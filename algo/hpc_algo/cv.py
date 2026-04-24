"""Cross-validation для HMM: k-fold по листам / весовым категориям.

Задача — честно сравнить варианты модели (basic vs detailed, categorical
vs bernoulli) без оверфита. Мы делим листы на ``k`` частей: на каждой
итерации обучаем HMM на ``k-1`` частях и считаем
held-out log-likelihood на оставшейся.

Функция работает на уже загруженных фреймах, чтобы не держать
зависимость на Excel/IO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hpc_algo import hmm as hmm_mod
from hpc_algo.hmm import HMMRunConfig
from hpc_algo.hmm_bernoulli import score_sequence
from hpc_algo.schema import ColumnMappingConfig


@dataclass
class CVFoldResult:
    fold: int
    train_sheets: list[str]
    test_sheets: list[str]
    train_log_likelihood: float | None
    test_log_likelihood: float | None
    test_samples: int
    status: str


@dataclass
class CVReport:
    variant: str
    observation_emission: str
    k: int
    folds: list[CVFoldResult]
    mean_test_ll: float | None
    std_test_ll: float | None
    mean_test_ll_per_sample: float | None


def _split_k_folds(items: list[str], k: int) -> list[tuple[list[str], list[str]]]:
    if k < 2:
        raise ValueError("k должно быть >= 2 для k-fold CV.")
    if k > len(items):
        k = max(2, len(items))
    chunks: list[list[str]] = [[] for _ in range(k)]
    for i, name in enumerate(items):
        chunks[i % k].append(name)
    folds: list[tuple[list[str], list[str]]] = []
    for i in range(k):
        test = chunks[i]
        train = [x for j, chunk in enumerate(chunks) if j != i for x in chunk]
        folds.append((train, test))
    return folds


def _subset(frames: dict[str, pd.DataFrame], names: list[str]) -> dict[str, pd.DataFrame]:
    return {n: frames[n] for n in names if n in frames}


def _subset_mapping(config: ColumnMappingConfig, names: list[str]) -> ColumnMappingConfig:
    return ColumnMappingConfig(
        version=config.version,
        sheets={n: sm for n, sm in config.sheets.items() if n in names},
    )


def cross_validate(
    frames: dict[str, pd.DataFrame],
    config: ColumnMappingConfig,
    run_config: HMMRunConfig,
    k: int = 5,
) -> CVReport:
    """K-fold CV по листам для текущей конфигурации HMM."""

    sheet_names = list(config.sheets.keys())
    sheet_names = [n for n in sheet_names if n in frames]
    folds = _split_k_folds(sheet_names, k)

    fold_results: list[CVFoldResult] = []
    variant_used = "unknown"

    for i, (train_names, test_names) in enumerate(folds):
        train_frames = _subset(frames, train_names)
        train_cfg = _subset_mapping(config, train_names)
        test_frames = _subset(frames, test_names)
        test_cfg = _subset_mapping(config, test_names)

        if run_config.observation_emission == "bernoulli":
            fit = hmm_mod.fit_hmm_bernoulli(train_frames, train_cfg, run_config)
            if fit is None:
                fold_results.append(
                    CVFoldResult(
                        fold=i,
                        train_sheets=train_names,
                        test_sheets=test_names,
                        train_log_likelihood=None,
                        test_log_likelihood=None,
                        test_samples=0,
                        status="train_failed",
                    )
                )
                continue
            result, _sanity = fit
            variant_used = result.parameters.variant

            # Score test sequences.
            test_entries, _ = hmm_mod.build_bernoulli_sequences(test_frames, test_cfg)
            params = _bernoulli_params_from_result(result)
            total_ll = 0.0
            total_samples = 0
            for _, _, X in test_entries:
                total_ll += score_sequence(X, params)
                total_samples += int(X.shape[0])
            fold_results.append(
                CVFoldResult(
                    fold=i,
                    train_sheets=train_names,
                    test_sheets=test_names,
                    train_log_likelihood=result.parameters.log_likelihood,
                    test_log_likelihood=total_ll if total_samples else None,
                    test_samples=total_samples,
                    status="done" if total_samples else "no_test_data",
                )
            )
        else:
            sequences, alphabet = hmm_mod.build_observation_sequences(
                train_frames, train_cfg
            )
            fit = hmm_mod.fit_hmm(sequences, alphabet, run_config)
            if fit is None:
                fold_results.append(
                    CVFoldResult(
                        fold=i,
                        train_sheets=train_names,
                        test_sheets=test_names,
                        train_log_likelihood=None,
                        test_log_likelihood=None,
                        test_samples=0,
                        status="train_failed",
                    )
                )
                continue
            result, _sanity = fit
            variant_used = result.parameters.variant

            # Test: build sequences using the SAME alphabet, ignoring tokens
            # that were not seen during training (они отсутствуют в vocab).
            vocab = {tok: idx for idx, tok in enumerate(alphabet)}
            test_seq, _ = hmm_mod.build_observation_sequences(test_frames, test_cfg)
            total_ll, total_samples = _score_categorical_sequences(
                test_seq, vocab, result
            )
            fold_results.append(
                CVFoldResult(
                    fold=i,
                    train_sheets=train_names,
                    test_sheets=test_names,
                    train_log_likelihood=result.parameters.log_likelihood,
                    test_log_likelihood=total_ll if total_samples else None,
                    test_samples=total_samples,
                    status="done" if total_samples else "no_test_data",
                )
            )

    lls = [f.test_log_likelihood for f in fold_results if f.test_log_likelihood is not None]
    samples = sum(f.test_samples for f in fold_results)
    if lls:
        mean_ll = float(np.mean(lls))
        std_ll = float(np.std(lls))
        mean_per_sample = float(sum(lls) / max(1, samples))
    else:
        mean_ll = None
        std_ll = None
        mean_per_sample = None

    return CVReport(
        variant=variant_used,
        observation_emission=run_config.observation_emission,
        k=k,
        folds=fold_results,
        mean_test_ll=mean_ll,
        std_test_ll=std_ll,
        mean_test_ll_per_sample=mean_per_sample,
    )


def _bernoulli_params_from_result(result) -> Any:  # HMMResult → BernoulliHMMParams
    """Вытянуть ``BernoulliHMMParams`` из уже собранного ``HMMResult``."""

    from hpc_algo.hmm_bernoulli import BernoulliHMMParams

    return BernoulliHMMParams(
        pi=np.asarray(result.parameters.initial_distribution, dtype=float),
        A=np.asarray(result.parameters.transition_matrix, dtype=float),
        B=np.asarray(result.parameters.emission_matrix, dtype=float),
    )


def _score_categorical_sequences(
    test_seq, vocab: dict[str, int], result
) -> tuple[float, int]:
    """Log-likelihood категориальной HMM на тестовой выборке.

    Пишем скоринг сами, без `hmmlearn`, чтобы не зависеть от его
    внутреннего состояния после fit.
    """

    import numpy as np

    pi = np.asarray(result.parameters.initial_distribution, dtype=float)
    A = np.asarray(result.parameters.transition_matrix, dtype=float)
    B = np.asarray(result.parameters.emission_matrix, dtype=float)

    from hpc_algo.hmm_bernoulli import _forward_backward, _log  # переиспользуем

    log_pi = _log(pi)
    log_A = _log(A)
    log_B = _log(B)

    total_ll = 0.0
    total_samples = 0
    for seq in test_seq:
        if not seq.tokens:
            continue
        indices = [vocab[t] for t in seq.tokens if t in vocab]
        if not indices:
            continue
        log_emiss = log_B[:, indices].T  # (T, n_states)
        _, _, ll = _forward_backward(log_pi, log_A, log_emiss)
        total_ll += ll
        total_samples += len(indices)
    return total_ll, total_samples


__all__ = [
    "cross_validate",
    "CVReport",
    "CVFoldResult",
]
