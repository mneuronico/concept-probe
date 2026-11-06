"""Model-free tests for the math under a probe: effect sizes, unit vectors, layer indices, token summaries."""
import numpy as np
import pytest

from concept_probe import summarize_tokens
from concept_probe.directions import normed_np
from concept_probe.stats import cohen_d
from concept_probe.utils import resolve_layer_index


def test_cohen_d_known_value():
    x = np.array([2.0, 4.0, 6.0])
    y = np.array([1.0, 2.0, 3.0])
    # pooled var = (2*4 + 2*1) / 4 = 2.5 ; d = (4 - 2) / sqrt(2.5)
    assert abs(cohen_d(x, y) - 2.0 / np.sqrt(2.5)) < 1e-6


def test_cohen_d_small_n_is_nan():
    assert np.isnan(cohen_d(np.array([1.0]), np.array([1.0, 2.0])))
    assert np.isnan(cohen_d(np.array([1.0, 2.0]), np.array([3.0])))


def test_normed_np_unit_norm():
    v = normed_np(np.array([3.0, 4.0]))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_layer_index_ints_fractions_and_errors():
    assert resolve_layer_index(1, 29) == 1
    assert resolve_layer_index(-1, 29) == 28
    assert resolve_layer_index({"frac": 0.5}, 29) == 14
    assert resolve_layer_index({"frac": 1.0}, 29) == 28
    for bad in (1.0, 0.5, True, "3", {"fraction": 0.5}, {"frac": 2.0}):
        with pytest.raises((TypeError, ValueError)):
            resolve_layer_index(bad, 29)
    with pytest.raises(ValueError, match="out of range"):
        resolve_layer_index(29, 29)


def test_token_scores_are_summarized_per_text():
    values = np.array([1.0, -3.0, 2.0, 0.5, 4.0])
    items = np.array([0, 0, 0, 1, 1])
    for how, expected in (("mean", [0.0, 2.25]), ("max", [2.0, 4.0]), ("min", [-3.0, 0.5]),
                          ("median", [1.0, 2.25]), ("extreme", [-3.0, 4.0]), ("p50", [1.0, 2.25])):
        out, order = summarize_tokens(values, items, how)
        np.testing.assert_allclose(out, expected)
        assert order == [0, 1]
    assert summarize_tokens(values, items, "extreme", center=1.0)[0].tolist() == [-3.0, 4.0]
    with pytest.raises(ValueError, match="unknown summary"):
        summarize_tokens(values, items, "loudest")
