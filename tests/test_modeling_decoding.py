import numpy as np

from hidden_patterns_combat.modeling.decoding import HMMDecoder


class _DummyModel:
    def predict(self, x, lengths=None):
        return np.zeros(len(x), dtype=int)

    def predict_proba(self, x, lengths=None):
        n = len(x)
        return np.tile(np.array([[0.7, 0.3]]), (n, 1))

    def score(self, x, lengths=None):
        return -12.5


def test_decoder_basic_inference_with_dummy_model():
    decoder = HMMDecoder(_DummyModel())
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    res = decoder.decode(x, lengths=[2])
    assert res.states.tolist() == [0, 0]
    assert res.state_probabilities.shape == (2, 2)
    assert res.log_likelihood == -12.5
