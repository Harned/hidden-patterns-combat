from __future__ import annotations

import numpy as np

from hidden_patterns_combat.modeling.training import HMMTrainer


class _DummyModel:
    def __init__(self):
        self.transmat_ = np.array(
            [
                [0.2, 0.4, 0.4],
                [0.6, 0.2, 0.2],
                [0.3, 0.3, 0.4],
            ],
            dtype=float,
        )
        self.startprob_ = np.array([0.34, 0.33, 0.33], dtype=float)

    def fit(self, x, lengths=None):
        return self

    def score(self, x, lengths=None):
        return 0.0


def test_left_to_right_topology_blocks_backward_transitions():
    model = _DummyModel()
    trainer = HMMTrainer(model, topology_mode="left_to_right")
    trainer.fit(np.array([[0.0], [1.0], [2.0]]), lengths=[3])

    assert model.transmat_[1, 0] == 0.0
    assert model.transmat_[2, 0] == 0.0
    assert model.transmat_[2, 1] == 0.0
    assert model.startprob_.tolist() == [1.0, 0.0, 0.0]


def test_ergodic_mode_keeps_transitions_unconstrained():
    model = _DummyModel()
    trainer = HMMTrainer(model, topology_mode="ergodic")
    trainer.fit(np.array([[0.0], [1.0], [2.0]]), lengths=[3])

    assert model.transmat_[1, 0] > 0
    assert model.transmat_[2, 0] > 0
    assert model.transmat_[2, 1] > 0
    assert np.isclose(model.transmat_.sum(axis=1), 1.0).all()


def test_left_to_right_can_enforce_min_forward_transition():
    model = _DummyModel()
    model.transmat_ = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    trainer = HMMTrainer(model, topology_mode="left_to_right", min_forward_transition=0.1)
    trainer.fit(np.array([[0.0], [1.0], [2.0]]), lengths=[3])

    assert model.transmat_[0, 1] > 0.0
    assert model.transmat_[1, 2] > 0.0
