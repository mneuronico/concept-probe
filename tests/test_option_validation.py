"""Model-free tests that misspelled or out-of-range options raise instead of doing something else."""
from types import SimpleNamespace

import numpy as np
import pytest

from concept_probe.config import ConceptSpec, resolve_config
from concept_probe.logger import JsonlLogger
from concept_probe.probe import ConceptProbe, _resolve_steer_layers, _select_best_layer, _validate_alpha_unit


# ----------------------------------------------------------------- steer_layers
def test_steer_layer_keywords():
    assert _resolve_steer_layers("probe", 5, 2, 10) == [5]
    assert _resolve_steer_layers("window", 5, 2, 10) == [3, 4, 5, 6, 7]
    assert _resolve_steer_layers("window", 9, 2, 10) == [7, 8, 9]


def test_explicit_steer_layers():
    assert _resolve_steer_layers([1, 3], 5, 2, 10) == [1, 3]
    assert _resolve_steer_layers(4, 5, 2, 10) == [4]
    assert _resolve_steer_layers([-1], 5, 2, 10) == [9]  # negative indices count from the end


@pytest.mark.parametrize("bad", ["12", "best", [], [1, 1], [10], [-11], [9, -1], [1.0], [True], None])
def test_bad_steer_layers_raise(bad):
    with pytest.raises((ValueError, TypeError)):
        _resolve_steer_layers(bad, 5, 2, 10)


def test_negative_window_radius_raises():
    with pytest.raises(ValueError):
        _resolve_steer_layers("window", 5, -1, 10)


# ------------------------------------------------------------------- alpha_unit
def test_alpha_units():
    assert _validate_alpha_unit("raw") == "raw" and _validate_alpha_unit("sigma") == "sigma"
    assert _validate_alpha_unit("resid_norm") == "resid_norm"
    for bad in ("Sigma", "sd", None):
        with pytest.raises(ValueError):
            _validate_alpha_unit(bad)


# ----------------------------------------------------------------- sweep_select
def test_p_value_selection_skips_layers_with_reversed_separation():
    # layer 0 has the smallest p but separates the eval sets the wrong way (d < 0)
    assert _select_best_layer([-3.0, 0.5, 1.0], [1e-9, 0.2, 0.01], [0, 1, 2], "p_value") == 2


def test_p_value_ties_go_to_the_larger_effect():
    assert _select_best_layer([2.0, 5.0, 3.0], [0.0, 0.0, 0.0], [0, 1, 2], "p_value") == 1


def test_selection_respects_the_search_range():
    assert _select_best_layer([5.0, 1.0, 2.0], [0.1, 0.1, 0.1], [1, 2], "effect_size") == 2


def test_p_value_selection_without_a_positive_layer_raises():
    with pytest.raises(ValueError, match="d > 0"):
        _select_best_layer([-1.0, -2.0], [0.01, 0.01], [0, 1], "p_value")


@pytest.mark.parametrize("bad", ["pvalue", "p", "Effect_size"])
def test_unknown_sweep_select_raises(bad):
    with pytest.raises(ValueError, match="sweep_select"):
        _select_best_layer([1.0], [0.1], [0], bad)


# ------------------------------------------- score_prompts validates before writing
def _trained_probe(tmp_path):
    bundle = SimpleNamespace(model_id="fake", tokenizer=None, model=None, num_layers=4)
    probe = ConceptProbe(
        model_bundle=bundle,
        config=resolve_config({"output": {"console": False}}),
        concept=ConceptSpec(name="c", pos_label="p", neg_label="n", pos_system="a", neg_system="b"),
        run_dir=str(tmp_path),
        logger=JsonlLogger(str(tmp_path / "log.jsonl")),
    )
    probe.concept_vectors = np.eye(4, dtype=np.float32)
    probe.best_layer = 2
    probe.sigma_best_layer = 0.5
    probe.resid_norm = np.ones(4)
    return probe


@pytest.mark.parametrize("kwargs", [{"alpha_unit": "Sigma"}, {"steer_layers": "12"}, {"steer_layers": [9]}])
def test_score_prompts_rejects_bad_options_before_writing_outputs(tmp_path, kwargs):
    probe = _trained_probe(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        probe.score_prompts(["hi"], alphas=[1.0], **kwargs)
    assert not (tmp_path / "scores").exists()


@pytest.mark.parametrize("unit, missing", [("sigma", "sigma_best_layer"), ("resid_norm", "resid_norm")])
def test_a_unit_without_its_reference_raises_before_writing_outputs(tmp_path, unit, missing):
    probe = _trained_probe(tmp_path)
    setattr(probe, missing, None)
    with pytest.raises(ValueError, match=missing):
        probe.score_prompts(["hi"], alphas=[2.0], alpha_unit=unit)
    assert not (tmp_path / "scores").exists()
