"""Model-free tests for the training math: effect sizes, layer arithmetic, selection."""
import numpy as np
import pytest

from concept_probe.probe import (
    _normalize_best_layer_search,
    cohen_d_np,
    normed_np,
    pick_layer,
    select_layers,
)
from concept_probe.utils import resolve_layer_index


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
    assert pick_layer(29, -1, 0.75) == 28  # negative indices count from the end


def test_pick_layer_rejects_out_of_range():
    with pytest.raises(ValueError):
        pick_layer(29, 29, 0.75)
    with pytest.raises(ValueError):
        pick_layer(29, -30, 0.75)
    with pytest.raises(ValueError):
        pick_layer(29, None, 1.5)


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


def test_best_layer_search_default_and_merge():
    info = _normalize_best_layer_search(None, 10)
    assert info["is_default"] and info["layers"] == list(range(10))

    info = _normalize_best_layer_search([7, 3], 10)  # swapped bounds
    assert info["intervals"] == [(3, 7)]

    info = _normalize_best_layer_search([[0, 2], [2, 5]], 10)  # overlapping merge
    assert info["intervals"] == [(0, 5)]

    info = _normalize_best_layer_search({"frac": [0.0, 1.0]}, 10)  # fractions span all layers
    assert info["layers"] == list(range(10))

    info = _normalize_best_layer_search([-3, -1], 10)  # negative indices count from the end
    assert info["intervals"] == [(7, 9)]


def test_best_layer_search_rejects_out_of_range_and_bare_floats():
    with pytest.raises(ValueError, match="out of range"):
        _normalize_best_layer_search([5, 40], 10)  # used to be clamped silently
    with pytest.raises(TypeError):
        _normalize_best_layer_search([0.2, 0.8], 10)  # fractions must be written {"frac": [...]}


def test_select_layers_modes():
    assert select_layers({"mode": "best"}, 16, 29) == [16]
    assert select_layers({"mode": "window", "window_radius": 2}, 1, 29) == [0, 1, 2, 3]
    assert select_layers({"mode": "explicit", "layers": [3, 7]}, 16, 29) == [3, 7]
    assert select_layers({"mode": "explicit", "layers": [-1]}, 16, 29) == [28]
    assert select_layers({"mode": "all"}, 16, 3) == [0, 1, 2]


def test_select_layers_rejects_out_of_range_and_unknown_modes():
    with pytest.raises(ValueError):
        select_layers({"mode": "explicit", "layers": [29]}, 16, 29)
    with pytest.raises(ValueError):
        select_layers({"mode": "explicit", "layers": []}, 16, 29)
    with pytest.raises(ValueError):
        select_layers({"mode": "window", "window_radius": -1}, 16, 29)
    with pytest.raises(ValueError):
        select_layers({"mode": "bogus"}, 16, 29)
