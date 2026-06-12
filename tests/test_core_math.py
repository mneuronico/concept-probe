"""Model-free tests for the training math: effect sizes, layer arithmetic, selection."""
import numpy as np
import pytest

from concept_probe.probe import (
    _layer_index_from_value,
    _normalize_best_layer_search,
    cohen_d_np,
    normed_np,
    pick_layer,
    select_layers,
)


def test_cohen_d_known_value():
    x = np.array([2.0, 4.0, 6.0])
    y = np.array([1.0, 2.0, 3.0])
    # pooled var = (2*4 + 2*1) / 4 = 2.5 ; d = (4 - 2) / sqrt(2.5)
    assert abs(cohen_d_np(x, y) - 2.0 / np.sqrt(2.5)) < 1e-6


def test_cohen_d_small_n_is_nan():
    assert np.isnan(cohen_d_np(np.array([1.0]), np.array([1.0, 2.0])))
    assert np.isnan(cohen_d_np(np.array([1.0, 2.0]), np.array([3.0])))


def test_normed_np_unit_norm():
    v = normed_np(np.array([3.0, 4.0]))
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_pick_layer_fraction_and_explicit():
    assert pick_layer(29, None, 0.75) == 21
    assert pick_layer(29, 5, 0.75) == 5


def test_pick_layer_rejects_out_of_range():
    with pytest.raises(ValueError):
        pick_layer(29, -1, 0.75)
    with pytest.raises(ValueError):
        pick_layer(29, 29, 0.75)


def test_layer_index_from_value_fraction_vs_int():
    assert _layer_index_from_value(1, 29) == 1
    assert _layer_index_from_value(0.5, 29) == 14
    assert _layer_index_from_value(1.0, 29) == 28  # fraction: last layer


def test_layer_index_from_value_rejects_ambiguous_float():
    with pytest.raises(ValueError):
        _layer_index_from_value(2.0, 29)


def test_layer_index_from_value_rejects_bool():
    with pytest.raises(TypeError):
        _layer_index_from_value(True, 29)


def test_best_layer_search_default_and_merge():
    info = _normalize_best_layer_search(None, 10)
    assert info["is_default"] and info["layers"] == list(range(10))

    info = _normalize_best_layer_search([7, 3], 10)  # swapped bounds
    assert info["intervals"] == [(3, 7)]

    info = _normalize_best_layer_search([[0, 2], [2, 5]], 10)  # overlapping merge
    assert info["intervals"] == [(0, 5)]

    info = _normalize_best_layer_search([0.0, 1.0], 10)  # fractions span all layers
    assert info["layers"] == list(range(10))


def test_select_layers_modes():
    assert select_layers({"mode": "best"}, 16, 29) == [16]
    assert select_layers({"mode": "window", "window_radius": 2}, 1, 29) == [0, 1, 2, 3]
    assert select_layers({"mode": "explicit", "layers": [3, 7]}, 16, 29) == [3, 7]
    assert select_layers({"mode": "all"}, 16, 3) == [0, 1, 2]


def test_select_layers_rejects_negative_and_out_of_range():
    with pytest.raises(ValueError):
        select_layers({"mode": "explicit", "layers": [-1]}, 16, 29)
    with pytest.raises(ValueError):
        select_layers({"mode": "explicit", "layers": [29]}, 16, 29)
    with pytest.raises(ValueError):
        select_layers({"mode": "bogus"}, 16, 29)
