"""Model-free tests of what the library feeds the model, and of what score_texts/score_prompts return."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from concept_probe import Calibration, ReadContext, preview_input
from concept_probe.config import ConceptSpec, resolve_config
from concept_probe.logger import JsonlLogger
from concept_probe.multi_probe import multi_probe_score_prompts
from concept_probe.probe import ConceptProbe, _encode_text_for_read, _normalize_messages
from fakes import CharChatTok, CharModel


# ------------------------------------------------------------- system prompts
@pytest.mark.parametrize("system", [None, ""])
def test_no_system_message_unless_one_is_set(system):
    assert _normalize_messages("hi", system)[0] == [{"role": "user", "content": "hi"}]
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert _normalize_messages(msgs, system)[0] == msgs  # a message list is used as given


def test_a_set_system_prompt_is_prepended_unless_the_list_has_its_own():
    assert _normalize_messages("hi", "S")[0][0] == {"role": "system", "content": "S"}
    own = [{"role": "system", "content": "mine"}, {"role": "user", "content": "x"}]
    seen = []
    messages, used_own = _normalize_messages(own, "S", warn=seen.append)
    assert messages == own and used_own and seen


# ------------------------------------------------------------- preview_input
def test_preview_of_a_raw_text_is_bos_and_the_text():
    out = preview_input(CharChatTok(), "ppx")
    assert out["input_text"] == "^ppx"
    assert out["text_span"] == (1, 4) and out["span_text"] == "ppx"


def test_preview_of_a_user_turn_and_of_a_generation_prompt():
    tok = CharChatTok()
    assert preview_input(tok, "ppx", role="user")["input_text"] == "^<u>ppx|"
    assert preview_input(tok, "ppx", role="user", system="S")["input_text"] == "^<s>S|<u>ppx|"
    gen = preview_input(tok, "ppx", generation=True)
    assert gen["input_text"] == "^<u>ppx|<a>" and gen["span_text"] == "ppx"  # the text, inside the prompt


def test_a_template_that_strips_the_text_warns_once():
    tok, seen = CharChatTok(strip_content=True), []
    for _ in range(2):
        _encode_text_for_read(tok, " ppx ", None, role="user", warn=seen.append)
    assert len([w for w in seen if "strips" in w]) == 1


# ------------------------------------------------------------- probes on fakes
def _probe(tmp_path, monkeypatch, name="c", vectors=None, config=None):
    # CharModel's hidden states are those of fakes.fake_forward: 'p' is +1, 'n' is -1 at layer 0
    bundle = SimpleNamespace(model_id="fake", tokenizer=CharChatTok(), model=CharModel(), num_layers=2)
    probe = ConceptProbe(
        model_bundle=bundle,
        config=resolve_config(config or {"output": {"console": False}, "plots": {"heatmap_html": False}}),
        concept=ConceptSpec(name=name, pos_label="p", neg_label="n", pos_system="a", neg_system="b"),
        run_dir=str(tmp_path / name),
        logger=JsonlLogger(str(tmp_path / name / "log.jsonl")),
    )
    probe.directions = np.array([[1.0, 0.0], [1.0, 0.0]] if vectors is None else vectors, dtype=np.float32)
    probe.steer_directions = probe.directions
    probe.layer = 0
    # per-item references (an item's mean) and per-token references (single tokens), measured on
    # generated completions, and the same numbers for raw texts (the transfer check's)
    refs = dict(center=np.array([0.0, 0.0]), scale=np.array([0.5, 1.0]),
                token_center=np.array([0.5, 0.0]), token_scale=np.array([0.25, 1.0]))
    probe.calibrations = [Calibration(ReadContext.generated(), source="evaluation", **refs),
                          Calibration(ReadContext(role="raw"), source="transfer check", **refs)]
    probe.steer_sigma = 0.5
    probe.resid_norm = np.array([2.0, 4.0])
    return probe


def test_score_texts_reads_the_raw_text_by_default(tmp_path, monkeypatch):
    scores = _probe(tmp_path, monkeypatch).score_texts(["ppx"], save=True)
    it = scores[0]
    assert it.info["role"] == "raw" and it.info["system_prompt"] is None
    assert it.info["input_text"] == "^ppx" and it.span == (1, 4)
    np.testing.assert_allclose(it.token_scores["c"], [0, 1, 1, 0])
    # single tokens use the token references: (score - 0.5) / 0.25
    np.testing.assert_allclose(it.token_z["c"], [-2, 2, 2, -2])
    # the text's mean over its own tokens uses the item references: (2/3 - 0) / 0.5
    assert scores.means()[0] == pytest.approx(2 / 3)
    assert scores.z()[0] == pytest.approx(4 / 3)
    # and the same numbers are on disk, one set of files for the call
    saved = type(scores).load(scores.dir)
    np.testing.assert_allclose(saved[0].token_z["c"], it.token_z["c"])
    assert sorted(os.listdir(scores.dir)) == ["scores.jsonl", "scores.npz", "settings.json"]


def test_score_texts_rejects_settings_its_role_would_ignore(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="raw"):
        probe.score_texts(["ppx"], system_prompt="S")
    with pytest.raises(ValueError, match="assistant"):
        probe.score_texts(["ppx"], role="user", prompt="q")
    it = probe.score_texts(["ppx"], role="user", system_prompt="S")[0]
    assert it.info["input_text"] == "^<s>S|<u>ppx|"


def test_score_prompts_adds_no_system_and_excludes_the_end_of_sequence_token(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    it = probe.score_prompts(["hi"], alphas=[0.0], save_html=False)[0]
    assert "".join(it.tokens[:it.span[0]]) == "^<u>hi|<a>"
    assert it.text == "pp"
    assert it.span[1] == it.span[0] + 2 and it.tokens[it.span[1]] == "$"
    assert it.score_mean["c"] == pytest.approx(1.0)  # 'pp' only; with the '$' it would be 2/3
    assert it.score_mean_z["c"] == pytest.approx(2.0)  # (1 - 0) / 0.5, item references


def test_resid_norm_steering_scales_each_layer(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    it = probe.score_prompts(["hi"], alphas=[0.1], alpha_unit="resid_norm", steer_layers=[0, 1], save_html=False)[0]
    assert it.info["alpha_value"] == pytest.approx(0.1)
    assert it.info["steer_layer_scales"] == {0: 2.0, 1: 4.0}
    assert it.info["steer_alpha_per_layer"] == {0: pytest.approx(0.1), 1: pytest.approx(0.2)}  # divided, then scaled


def test_random_controls_come_in_the_same_call(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    probe.resid_norm = np.array([2.0, 4.0])
    scores = probe.score_prompts(["hi", "yo"], alphas=[0.0, 1.0, -1.0], n_random=3, alpha_unit="resid_norm",
                                 steer_layers=[0, 1], save_html=False)
    # each prompt: alpha 0, alpha 1 and its 3 controls, alpha -1 and its 3 controls
    assert len(scores) == 2 * 9
    assert [it.info["control"] for it in scores][:9] == [None, None, 0, 1, 2, None, 0, 1, 2]
    assert scores.settings["n_random"] == 3 and scores.settings["random_seed"] == 0
    first = [it.info["intervention"] for it in scores][:9]
    steer, control = first[1], first[2]
    assert control["control_seed"] == [0, 0] and first[6]["control_seed"] == [0, 0]  # the same direction at -1
    assert control["alpha"] == steer["alpha"]  # the same doses at the same layers
    assert control["vector_norm"] == pytest.approx(steer["vector_norm"])  # and the same length at each layer
    multi = multi_probe_score_prompts([probe], ["hi"], alphas=[1.0], steer_probe=0, n_random=2, random_seed=5,
                                      save_html=False, output_root=str(tmp_path / "m"))
    assert [it.info["intervention"].get("control_seed") for it in multi] == [None, [5, 0], [5, 1]]


def test_multi_probe_score_means_exclude_the_end_of_sequence_token(tmp_path, monkeypatch):
    a = _probe(tmp_path, monkeypatch, "a")
    b = _probe(tmp_path, monkeypatch, "b", vectors=[[-1.0, 0.0], [-1.0, 0.0]])
    it = multi_probe_score_prompts([a, b], ["hi"], save_html=False, output_root=str(tmp_path / "m"))[0]
    assert it.score_mean == {"a": pytest.approx(1.0), "b": pytest.approx(-1.0)}
    assert it.score_mean_z["a"] == pytest.approx(2.0)


def test_a_repeated_run_name_does_not_mix_runs(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    kwargs = dict(save_html=False, output_root=str(tmp_path / "m"), run_name="same", save=True)
    multi_probe_score_prompts([probe], ["hi"], **kwargs)
    with pytest.raises(FileExistsError):
        multi_probe_score_prompts([probe], ["hi"], **kwargs)
    multi_probe_score_prompts([probe], ["hi"], overwrite=True, **kwargs)


# ------------------------------------------------------------- multi-probe settings
def _cfg(**sections):
    base = {"output": {"console": False}, "plots": {"heatmap_html": False}}
    base.update(sections)
    return base


def test_multi_probe_refuses_probes_from_different_model_settings(tmp_path, monkeypatch):
    # explicit dtypes: with the default "auto", the config alone does not say which dtype was used
    a = _probe(tmp_path, monkeypatch, "a", config=_cfg(model={"dtype": "bfloat16"}))
    b = _probe(tmp_path, monkeypatch, "b", config=_cfg(model={"dtype": "float32"}))
    with pytest.raises(ValueError, match="dtype"):
        multi_probe_score_prompts([a, b], ["hi"], output_root=str(tmp_path / "m"))


def test_multi_probe_generation_settings_must_agree_or_be_passed(tmp_path, monkeypatch):
    a = _probe(tmp_path, monkeypatch, "a")
    b = _probe(tmp_path, monkeypatch, "b", config=_cfg(steering={"steer_temperature": 0.2}))
    # greedy (the default) does not use the temperature, so the disagreement does not matter
    out = multi_probe_score_prompts([a, b], ["hi"], output_root=str(tmp_path / "g"))
    assert out.settings["generation"]["temperature"] is None
    with pytest.raises(ValueError, match="steer_temperature"):
        multi_probe_score_prompts([a, b], ["hi"], greedy=False, output_root=str(tmp_path / "m"))
    out = multi_probe_score_prompts([a, b], ["hi"], greedy=False, temperature=0.5, output_root=str(tmp_path / "m"))
    assert out[0].text == "pp"
    assert out.settings["generation"]["temperature"] == 0.5


def test_multi_probe_steers_with_the_steering_probe_settings(tmp_path, monkeypatch):
    import json

    a = _probe(tmp_path, monkeypatch, "a")
    b = _probe(tmp_path, monkeypatch, "b", config=_cfg(steering={"steer_distribute": False}))
    out = multi_probe_score_prompts([a, b], ["hi"], alphas=[1.0], steer_probe=1, steer_layers=[0],
                                    output_root=str(tmp_path / "m"), save=True)
    config = json.load(open(os.path.join(out.settings["run_dir"], "config.json")))
    assert config["steer_distribute"] is False and config["steer_probe"] == "b"
    assert out[0].info["steer_alpha_per_layer"] == {0: 1.0}
