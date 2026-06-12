"""Model-free tests for pooling (reps_from_hs_layers) and token scoring."""
import numpy as np
import torch
import pytest

from concept_probe.probe import reps_from_hs_layers, token_scores_from_hs_layers

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


def test_assistant_fallback_warns_once():
    messages = []
    reps = reps_from_hs_layers(
        HS, prompt_len=4, mode="assistant_all_mean", last_k=2, warn=messages.append
    )
    np.testing.assert_allclose(reps, [[2.5, 2.5], [5, 5]])  # full-sequence fallback
    assert len(messages) == 1
    assert "no completion tokens" in messages[0]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        reps_from_hs_layers(HS, prompt_len=None, mode="bogus", last_k=2)


def test_token_scores_per_layer_and_aggregates():
    cv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    per_layer, agg = token_scores_from_hs_layers(HS, cv, [0, 1], aggregate="mean")
    np.testing.assert_allclose(per_layer[:, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(per_layer[:, 1], [2, 4, 6, 8])
    np.testing.assert_allclose(agg, [1.5, 3, 4.5, 6])

    _, agg_sum = token_scores_from_hs_layers(HS, cv, [0, 1], aggregate="sum")
    np.testing.assert_allclose(agg_sum, [3, 6, 9, 12])


def test_token_scores_reference_layer():
    cv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    per_layer, _ = token_scores_from_hs_layers(HS, cv, [0, 1], aggregate="mean", reference_layer=0)
    # both layers projected on cv[0] = e0
    np.testing.assert_allclose(per_layer[:, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(per_layer[:, 1], [2, 4, 6, 8])

    with pytest.raises(ValueError):
        token_scores_from_hs_layers(HS, cv, [0], aggregate="mean", reference_layer=5)
