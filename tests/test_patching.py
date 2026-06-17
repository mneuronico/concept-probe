"""Model-free tests for concept_probe.patching.

A tiny chained model (each block adds a fixed vector) makes the residual stream
fully predictable, so patches can be checked against closed-form expectations.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from concept_probe.modeling import get_transformer_layers
from concept_probe.patching import (
    ActivationPatch,
    ResidualStreamEditor,
    build_patch_edit_fn,
    patched_forward_hidden_states,
)

DIM = 4


class _AddBlock(torch.nn.Module):
    """resid_post = resid_pre + (i+1); returns a tuple to mimic HF decoder layers."""

    def __init__(self, i: int, return_tuple: bool = True):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))  # gives the module a device
        self.bias = float(i + 1)
        self.return_tuple = return_tuple

    def forward(self, x):
        y = x + self.bias
        return (y, "aux") if self.return_tuple else y


class _ChainModel(torch.nn.Module):
    """Embeds input_ids -> (1, seq, DIM) and runs the blocks in sequence."""

    def __init__(self, n_layers: int = 4, return_tuple: bool = True):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_AddBlock(i, return_tuple=return_tuple) for i in range(n_layers)]
        )
        self.device = torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, return_dict=True):
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, DIM)  # (1, seq, DIM)
        for layer in self.model.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return SimpleNamespace(logits=x)


# ----------------------------- ActivationPatch -------------------------------
def test_activation_patch_validates_and_broadcasts_1d():
    p = ActivationPatch(layer=2, positions=5, values=torch.ones(DIM))
    assert p.layer == 2 and p.positions == [5] and p.values.shape == (1, DIM)


def test_activation_patch_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        ActivationPatch(layer=0, positions=[0, 1], values=torch.ones(1, DIM))


# ----------------------------- build_patch_edit_fn ---------------------------
def test_edit_fn_overwrites_positions_out_of_place():
    edit = build_patch_edit_fn([ActivationPatch(layer=1, positions=[1], values=torch.arange(DIM).float())])
    hidden = torch.zeros(1, 3, DIM)
    out = edit(1, hidden)
    np.testing.assert_allclose(out[0, 1].numpy(), np.arange(DIM))
    assert float(out[0, 0].abs().sum()) == 0 and float(out[0, 2].abs().sum()) == 0
    # input was not mutated (out-of-place)
    assert float(hidden.abs().sum()) == 0
    # a layer with no patch is returned untouched
    assert edit(0, hidden) is hidden


def test_edit_fn_out_of_range_position_raises():
    edit = build_patch_edit_fn([ActivationPatch(layer=0, positions=[9], values=torch.ones(DIM))])
    with pytest.raises(IndexError):
        edit(0, torch.zeros(1, 3, DIM))


# ----------------------------- ResidualStreamEditor --------------------------
def test_editor_applies_and_removes_hooks():
    model = _ChainModel()
    X = torch.zeros(1, 2, DIM)

    def edit(li, h):
        return h + 10.0 if li == 0 else h

    layer0 = get_transformer_layers(model)[0]
    with ResidualStreamEditor(model, edit):
        edited = layer0(X)[0]
    clean = layer0(X)[0]
    assert float((edited - clean).abs().max()) == 10.0  # +1 bias +10 vs +1 bias
    # hooks removed after exit
    assert float((clean - (X + 1.0)).abs().max()) == 0.0


def test_editor_preserves_tuple_aux():
    model = _ChainModel(return_tuple=True)
    with ResidualStreamEditor(model, lambda li, h: h):
        out = get_transformer_layers(model)[1](torch.zeros(1, 1, DIM))
    assert isinstance(out, tuple) and out[1] == "aux"


def test_editor_negative_layer_index():
    model = _ChainModel(n_layers=4)
    ed = ResidualStreamEditor(model, lambda li, h: h, layers=[-1])
    assert ed.layers == [3]


# ----------------------- patched_forward_hidden_states -----------------------
def _ids(seq):
    return torch.tensor([seq], dtype=torch.long)


def test_noop_patch_reproduces_clean_stream():
    model = _ChainModel(n_layers=4)
    ids = _ids([1, 2, 3])
    hs = patched_forward_hidden_states(model, ids, [])
    # resid_post[l] = emb + sum_{k<=l}(k+1); emb[token] = token value broadcast
    emb = ids.float().unsqueeze(-1).repeat(1, 1, DIM)[0]
    cum = 0.0
    for l in range(4):
        cum += (l + 1)
        np.testing.assert_allclose(hs[l].numpy(), (emb + cum).numpy(), rtol=0, atol=1e-5)


def test_patch_reflected_at_layer_and_downstream_not_before():
    model = _ChainModel(n_layers=4)
    ids = _ids([1, 2, 3])
    val = torch.full((DIM,), 99.0)
    # patch position 1 at layer 2
    hs = patched_forward_hidden_states(model, ids, [ActivationPatch(layer=2, positions=[1], values=val)])
    emb = ids.float().unsqueeze(-1).repeat(1, 1, DIM)[0]
    # layers before 2: unchanged
    np.testing.assert_allclose(hs[0][1].numpy(), (emb[1] + 1).numpy())
    np.testing.assert_allclose(hs[1][1].numpy(), (emb[1] + 1 + 2).numpy())
    # layer 2 at the patched position: exactly the patch value
    np.testing.assert_allclose(hs[2][1].numpy(), val.numpy())
    # layer 3 downstream: patch value + block-3 bias (4)
    np.testing.assert_allclose(hs[3][1].numpy(), (val + 4).numpy())
    # an unpatched position at layer 2 is the clean value
    np.testing.assert_allclose(hs[2][0].numpy(), (emb[0] + 1 + 2 + 3).numpy())
