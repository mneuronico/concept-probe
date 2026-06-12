"""Model-free tests for the steering hooks, including the contamination regression:
after the steering context exits, forward passes must be unmodified."""
import numpy as np
import torch

from concept_probe.modeling import get_transformer_layers
from concept_probe.steering import MultiLayerSteererLayerwise


class _Block(torch.nn.Module):
    def __init__(self, return_tuple: bool = False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))  # hook needs a parameter for device
        self.return_tuple = return_tuple

    def forward(self, x):
        return (x, "aux") if self.return_tuple else x


class _FakeModel(torch.nn.Module):
    def __init__(self, n_layers: int = 3, return_tuple: bool = False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_Block(return_tuple=return_tuple) for _ in range(n_layers)]
        )


VECS = np.stack([np.eye(4, dtype=np.float32)[i % 4] for i in range(3)])  # v[l] = e_{l}
X = torch.zeros(1, 2, 4)


def _run_layer(model, idx):
    return get_transformer_layers(model)[idx](X)


def test_steering_adds_alpha_v_distributed():
    model = _FakeModel()
    with MultiLayerSteererLayerwise(model, [0, 1], VECS, alpha=2.0, distribute=True):
        out0 = _run_layer(model, 0)
        out1 = _run_layer(model, 1)
        out2 = _run_layer(model, 2)  # not steered
    np.testing.assert_allclose(out0.numpy()[0, 0], [1, 0, 0, 0])  # alpha/2 * e0
    np.testing.assert_allclose(out1.numpy()[0, 0], [0, 1, 0, 0])
    np.testing.assert_allclose(out2.numpy()[0, 0], [0, 0, 0, 0])
    # offset applied at every position
    np.testing.assert_allclose(out0.numpy()[0, 1], [1, 0, 0, 0])


def test_steering_not_distributed_uses_full_alpha():
    model = _FakeModel()
    with MultiLayerSteererLayerwise(model, [0, 1], VECS, alpha=2.0, distribute=False):
        out0 = _run_layer(model, 0)
    np.testing.assert_allclose(out0.numpy()[0, 0], [2, 0, 0, 0])


def test_hooks_removed_after_exit():
    # Regression for the scoring-contamination bug: outside the context the forward
    # pass must be unmodified, so post-hoc scoring reads the text, not the vector.
    model = _FakeModel()
    with MultiLayerSteererLayerwise(model, [0, 1, 2], VECS, alpha=5.0, distribute=False):
        steered = _run_layer(model, 0)
    clean = _run_layer(model, 0)
    assert float(steered.abs().sum()) > 0
    assert float(clean.abs().sum()) == 0


def test_alpha_zero_is_noop():
    model = _FakeModel()
    with MultiLayerSteererLayerwise(model, [0, 1], VECS, alpha=0.0, distribute=True) as s:
        assert s.steerers == []
        out0 = _run_layer(model, 0)
    assert float(out0.abs().sum()) == 0


def test_tuple_outputs_preserved():
    model = _FakeModel(return_tuple=True)
    with MultiLayerSteererLayerwise(model, [1], VECS, alpha=3.0, distribute=False):
        out = _run_layer(model, 1)
    assert isinstance(out, tuple) and out[1] == "aux"
    np.testing.assert_allclose(out[0].numpy()[0, 0], [0, 3, 0, 0])
