"""Model-free tests for pooling (reps_from_hs_layers) and token scoring."""
import numpy as np
import torch
import pytest

from concept_probe.generation import TokenScorer
from concept_probe.probe import EmptySpanError, reps_from_hs_layers

# 2 layers, 4 tokens, dim 2; token t has value (t+1) in both dims, layer 1 doubled.
HS = [
    torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
    torch.tensor([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [8.0, 8.0]]),
]


def test_sequence_modes():
    reps = reps_from_hs_layers(HS, prompt_len=None, mode="sequence_last", last_k=2)
    np.testing.assert_allclose(reps, [[4, 4], [8, 8]])

    reps = reps_from_hs_layers(HS, prompt_len=None, mode="sequence_all_mean", last_k=2)
    np.testing.assert_allclose(reps, [[2.5, 2.5], [5, 5]])

    reps = reps_from_hs_layers(HS, prompt_len=None, mode="sequence_last_k_mean", last_k=2)
    np.testing.assert_allclose(reps, [[3.5, 3.5], [7, 7]])


def test_assistant_modes_use_completion_span():
    reps = reps_from_hs_layers(HS, prompt_len=2, mode="assistant_all_mean", last_k=2)
    np.testing.assert_allclose(reps, [[3.5, 3.5], [7, 7]])  # mean of tokens 2,3

    reps = reps_from_hs_layers(HS, prompt_len=2, mode="assistant_last_k_mean", last_k=1)
    np.testing.assert_allclose(reps, [[4, 4], [8, 8]])


def test_an_empty_span_raises_instead_of_pooling_the_prompt():
    # pooling the whole sequence instead would put the prompt (and its system prompt) in its place
    for mode in ("assistant_all_mean", "assistant_last", "assistant_last_k_mean"):
        with pytest.raises(EmptySpanError):
            reps_from_hs_layers(HS, prompt_len=4, mode=mode, last_k=2)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        reps_from_hs_layers(HS, prompt_len=None, mode="bogus", last_k=2)


def test_token_scores_per_layer_and_aggregates():
    per_layer = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]], dtype=np.float32)
    mean = TokenScorer("c", [0, 1], np.eye(2, dtype=np.float32), "mean")
    np.testing.assert_allclose(mean.combine(per_layer), [1.5, 3, 4.5, 6])
    total = TokenScorer("c", [0, 1], np.eye(2, dtype=np.float32), "sum")
    np.testing.assert_allclose(total.combine(per_layer), [3, 6, 9, 12])
    assert mean.token_z(per_layer) is None  # no calibration, no z
