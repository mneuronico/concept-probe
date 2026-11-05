"""Model-free tests of what steered scoring does by default and of what each record says it did."""
import inspect
import json
import os

import numpy as np
import pytest
import torch

from concept_probe import concept_runner
from concept_probe.config import load_defaults, resolve_config
from concept_probe.multi_probe import multi_probe_score_prompts
from concept_probe import Encoded, intervene, read_encoded
from concept_probe.probe import reps_from_hs_layers
from concept_probe.visuals import _token_for_display, plot_sweep
from fakes import CharModel
from test_inputs import _probe


# ------------------------------------------------------------- which layers, which vectors
def test_by_default_only_the_probe_layer_is_steered(tmp_path, monkeypatch):
    assert resolve_config()["steering"]["steer_layers"] == "probe"
    scores = _probe(tmp_path, monkeypatch).score_prompts(["hi"], alphas=[1.5], save_html=False)
    assert scores.settings["steer_layers"] == [0]
    assert scores[0].info["steer_alpha_per_layer"] == {0: 1.5}
    assert scores[0].info["intervention"]["type"] == "add"


def test_a_window_divides_alpha_by_the_layers_actually_steered(tmp_path, monkeypatch):
    # best layer 0 with radius 2 in a 2-layer model: layers 0 and 1, not 5
    scores = _probe(tmp_path, monkeypatch).score_prompts(["hi"], alphas=[1.0], steer_layers="window", save_html=False)
    assert scores.settings["steer_layers"] == [0, 1]
    assert scores[0].info["steer_alpha_per_layer"] == {0: 0.5, 1: 0.5}


def test_each_steered_layer_adds_its_own_vector(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch, vectors=[[1.0, 0.0], [0.0, 1.0]])
    probe.resid_norm = np.array([1.0, 3.0])
    steer = probe.steering(2.0, "resid_norm", layers=[0, 1], distribute=True)
    assert steer.alpha_per_layer == {0: 1.0, 1: 3.0}
    model = probe.model_bundle.model
    tok = probe.model_bundle.tokenizer

    def read():  # every layer's residual stream at the one token of "x"
        return read_encoded(model, tok, [Encoded(torch.tensor([ord("x")]))], "sequence_tokens", cache=False).data[0]

    clean = read()
    with intervene(model, steer):
        steered = read()
    added = steered - clean
    np.testing.assert_allclose(added[0], [1.0, 0.0])  # 1.0 * v_0 at layer 0
    np.testing.assert_allclose(added[1] - 2.0 * added[0], [0.0, 3.0])  # block 1 doubles v_0, then adds 3.0 * v_1


def test_unsteered_records_have_no_per_layer_alpha(tmp_path, monkeypatch):
    it = _probe(tmp_path, monkeypatch).score_prompts(["hi"], alphas=[0.0], save_html=False)[0]
    assert it.info["steer_alpha_per_layer"] == {} and it.info["intervention"] is None


# ------------------------------------------------------------- generation settings
def test_generation_is_greedy_by_default_and_recorded(tmp_path, monkeypatch):
    generation = _probe(tmp_path, monkeypatch).score_prompts(["hi"], save_html=False).settings["generation"]
    assert generation["greedy"] is True
    assert generation["seed"] is None and generation["temperature"] is None


def test_sampling_is_seeded_from_the_config_unless_a_seed_is_given(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    generation = probe.score_prompts(["hi"], greedy=False, temperature=0.3, save_html=False).settings["generation"]
    assert generation == {"greedy": False, "max_new_tokens": generation["max_new_tokens"], "temperature": 0.3,
                          "top_p": probe.config["steering"]["steer_top_p"], "seed": 0,
                          "batch_size": generation["batch_size"]}
    assert probe.score_prompts(["hi"], greedy=False, seed=7, save_html=False).settings["generation"]["seed"] == 7


def test_multi_probe_records_generation_and_per_layer_alpha(tmp_path, monkeypatch):
    a = _probe(tmp_path, monkeypatch, "a")
    out = multi_probe_score_prompts([a], ["hi"], alphas=[2.0], steer_probe=0, steer_layers="window",
                                    save_html=False, output_root=str(tmp_path / "m"))
    assert out[0].info["steer_alpha_per_layer"] == {0: 1.0, 1: 1.0}
    assert out.settings["generation"]["greedy"] is True


# ------------------------------------------------------------- alphas
def test_empty_alphas_raise(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="alphas is empty"):
        probe.score_prompts(["hi"], alphas=[])
    with pytest.raises(ValueError, match="alphas is empty"):
        multi_probe_score_prompts([probe], ["hi"], alphas=[], output_root=str(tmp_path / "m"))


def test_multi_probe_nonzero_alphas_need_a_steering_probe(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="steer_probe"):
        multi_probe_score_prompts([probe], ["hi"], alphas=[1.0], output_root=str(tmp_path / "m"))


def test_every_entry_point_defaults_to_raw_alphas():
    assert inspect.signature(concept_runner.train_concept_from_json).parameters["alpha_unit"].default == "raw"


# ------------------------------------------------------------- multi-probe outputs
def test_multi_probe_saves_each_probes_per_layer_scores(tmp_path, monkeypatch):
    a = _probe(tmp_path, monkeypatch, "a")
    b = _probe(tmp_path, monkeypatch, "b", vectors=[[-1.0, 0.0], [-1.0, 0.0]])
    out = multi_probe_score_prompts([a, b], ["hi", "yo"], save_html=False, output_root=str(tmp_path / "m"), save=True)
    z = np.load(os.path.join(out.dir, "scores.npz"))
    assert list(z["probe_names"]) == ["a", "b"]
    assert z["scores_0"].shape == z["scores_1"].shape and z["scores_0"].shape[0] == z["offsets"][-1]
    np.testing.assert_array_equal(z["layers_0"], a._layer_selection(None))
    np.testing.assert_allclose(z["scores_1"], -z["scores_0"])
    lines = [json.loads(line) for line in open(os.path.join(out.dir, "scores.jsonl"), encoding="utf-8")]
    assert [l["text"] for l in lines] == ["pp", "pp"] and lines[0]["score_mean"]["b"] == pytest.approx(-1.0)


def test_multi_probe_warnings_reach_the_run_log(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    own = [{"role": "system", "content": "mine"}, {"role": "user", "content": "x"}]
    out = multi_probe_score_prompts([probe], [own], system_prompt="S", save_html=False,
                                    output_root=str(tmp_path / "m"), save=True)
    events = [json.loads(line) for line in open(os.path.join(out.settings["run_dir"], "log.jsonl"), encoding="utf-8")]
    assert any(e["event"] == "warning" for e in events)


def test_segments_carry_the_heatmap_scores_and_say_which(tmp_path, monkeypatch):
    out = _probe(tmp_path, monkeypatch).score_prompts(["hi"], save_html=False, save_segments=True, save=True)
    line = json.loads(open(os.path.join(out.dir, "scores.jsonl"), encoding="utf-8").readline())
    first = line["segments"]["c"][0]
    np.testing.assert_allclose(first["scores"], out[0].token_z["c"][first["token_indices"]])  # token z, as the heatmap


# ------------------------------------------------------------- readouts and files
def test_sequence_modes_stop_where_the_item_ends():
    hs = [torch.tensor([[1.0], [2.0], [3.0], [9.0]])]  # the last position is a trailing EOS
    assert reps_from_hs_layers(hs, 1, "sequence_last", 1, span_end=3)[0, 0] == 3.0
    assert reps_from_hs_layers(hs, 1, "sequence_all_mean", 1, span_end=3)[0, 0] == pytest.approx(2.0)
    assert reps_from_hs_layers(hs, 1, "sequence_last_k_mean", 2, span_end=3)[0, 0] == pytest.approx(2.5)


def test_a_partial_defaults_file_keeps_the_other_library_defaults(tmp_path):
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps({"steering": {"steer_max_new_tokens": 5}}))
    merged = load_defaults(str(path))
    assert merged["steering"]["steer_max_new_tokens"] == 5
    assert merged["steering"]["steer_layers"] == load_defaults()["steering"]["steer_layers"]
    assert merged["readout"] == load_defaults()["readout"]


def test_unknown_keys_in_a_concept_json_raise(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"concept": {}, "prompts": {}, "config_overides": {}}))
    with pytest.raises(ValueError, match="config_overides"):
        concept_runner._load_concept_json(str(path))


@pytest.mark.parametrize("token", ["<bos>", "<eos>", "<pad>", "<|endoftext|>"])
def test_special_tokens_are_shown_as_they_are(token):
    assert _token_for_display(token) == token


def test_the_sweep_plot_can_show_the_selection_half(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "sweep.png"
    assert plot_sweep([0.1, 0.5], [0.5, 0.01], str(out), best_layer=1, selection_d=[0.2, 0.9])
    assert out.exists()
