"""NaN or inf hidden states stop the run where they appear, with the model's dtype in the message.

Gemma 3 overflows in float16; before this check the NaN layers went on to produce NaN directions
and scores, and the layer sweep skipped them in silence.
"""
import numpy as np
import pytest
import torch

from concept_probe import Encoded, NonFiniteActivationsError, Patch, intervene, read_encoded, read_texts
from concept_probe.modeling import get_transformer_layers
from fakes import CharChatTok
from tiny_model import TinyCausalLM

IDS = torch.tensor([ord(c) for c in "^pp nx"])


def _every_token(model):
    # (tokens, layers, dim): every token of the input at every layer
    return read_encoded(model, CharChatTok(), [Encoded(IDS)], "sequence_tokens", cache=False).data


def _overflow_from(model, layer):
    def hook(module, inputs, output):
        return output * float("inf")

    return get_transformer_layers(model)[layer].register_forward_hook(hook)


def test_finite_hidden_states_pass():
    hs = _every_token(TinyCausalLM())
    assert hs.shape[:2] == (6, 4) and np.all(np.isfinite(hs))


def test_non_finite_hidden_states_raise_naming_layers_and_dtype():
    model = TinyCausalLM()
    handle = _overflow_from(model, 2)
    try:
        with pytest.raises(NonFiniteActivationsError, match=r"layers \[2, 3\] of 4.*torch\.float64"):
            _every_token(model)
    finally:
        handle.remove()


def test_reading_texts_stops_at_the_overflow():
    model = TinyCausalLM()
    handle = _overflow_from(model, 3)
    try:
        with pytest.raises(NonFiniteActivationsError, match="bfloat16"):
            read_texts(model, CharChatTok(), ["pp x"], cache=False)
    finally:
        handle.remove()


def test_a_patch_that_writes_nan_is_caught():
    model = TinyCausalLM()
    with intervene(model, Patch(1, [2], torch.full((16,), float("nan")))):
        with pytest.raises(NonFiniteActivationsError, match=r"layers \[1, 2, 3\]"):
            _every_token(model)
    # a finite patch still runs, and reading reports the patched value at the patched layer
    with intervene(model, Patch(1, [2], torch.zeros(16))):
        hs = _every_token(model)
    np.testing.assert_allclose(hs[2, 1], 0.0)


def test_finite_layers_can_still_be_read_when_later_ones_overflow():
    model = TinyCausalLM()
    handle = _overflow_from(model, 2)
    try:
        early = read_texts(model, CharChatTok(), ["pp x"], layers=[0, 1], cache=False)
        assert np.isfinite(early.data).all()
        with pytest.raises(NonFiniteActivationsError, match=r"layers \[2\]"):
            read_texts(model, CharChatTok(), ["pp x"], layers=[1, 2], cache=False)
    finally:
        handle.remove()
