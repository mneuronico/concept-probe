"""NaN or inf hidden states stop the run where they appear, with the model's dtype in the message.

Gemma 3 overflows in float16; before this check the NaN layers went on to produce NaN directions
and scores, and the layer sweep skipped them in silence.
"""
import numpy as np
import pytest
import torch

from concept_probe import NonFiniteActivationsError, extract_text_reps
from concept_probe.modeling import get_transformer_layers
from concept_probe.patching import ActivationPatch, patched_forward_hidden_states
from concept_probe.probe import forward_hidden_states_all_layers
from fakes import CharChatTok
from tiny_model import TinyCausalLM

IDS = torch.tensor([[ord(c) for c in "^pp nx"]])


def _overflow_from(model, layer):
    def hook(module, inputs, output):
        return output * float("inf")

    return get_transformer_layers(model)[layer].register_forward_hook(hook)


def test_finite_hidden_states_pass():
    hs = forward_hidden_states_all_layers(TinyCausalLM(), IDS)
    assert len(hs) == 4 and all(bool(torch.isfinite(h).all()) for h in hs)


def test_non_finite_hidden_states_raise_naming_layers_and_dtype():
    model = TinyCausalLM()
    handle = _overflow_from(model, 2)
    try:
        with pytest.raises(NonFiniteActivationsError, match=r"layers \[2, 3\] of 4.*torch\.float64"):
            forward_hidden_states_all_layers(model, IDS)
    finally:
        handle.remove()


def test_reading_texts_stops_at_the_overflow():
    model = TinyCausalLM()
    handle = _overflow_from(model, 3)
    try:
        with pytest.raises(NonFiniteActivationsError, match="bfloat16"):
            extract_text_reps(model, CharChatTok(), ["pp x"])
    finally:
        handle.remove()


def test_a_patch_that_writes_nan_is_caught():
    patch = ActivationPatch(layer=1, positions=[2], values=torch.full((16,), float("nan")))
    with pytest.raises(NonFiniteActivationsError, match=r"layers \[1, 2, 3\]"):
        patched_forward_hidden_states(TinyCausalLM(), IDS, [patch])
    # a finite patch still runs
    ok = ActivationPatch(layer=1, positions=[2], values=torch.zeros(16))
    hs = patched_forward_hidden_states(TinyCausalLM(), IDS, [ok])
    np.testing.assert_allclose(hs[1][2].numpy(), 0.0)
