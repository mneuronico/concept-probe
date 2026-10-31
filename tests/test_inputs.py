"""Model-free tests of what the library feeds the model, and of what score_texts/score_prompts return."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from concept_probe import multi_probe as multi_mod
from concept_probe import preview_input
from concept_probe import probe as probe_mod
from concept_probe.config import ConceptSpec, resolve_config
from concept_probe.logger import JsonlLogger
from concept_probe.multi_probe import multi_probe_score_prompts
from concept_probe.probe import ConceptProbe, _encode_text_for_read, _normalize_messages
from fakes import CharChatTok, CharModel, fake_forward


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
    assert gen["input_text"] == "^<u>ppx|<a>" and gen["text_span"] is None


def test_a_template_that_strips_the_text_warns_once():
    tok, seen = CharChatTok(strip_content=True), []
    for _ in range(2):
        _encode_text_for_read(tok, " ppx ", None, role="user", warn=seen.append)
    assert len([w for w in seen if "strips" in w]) == 1


# ------------------------------------------------------------- probes on fakes
def _probe(tmp_path, monkeypatch, name="c", vectors=None, config=None):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", fake_forward)
    monkeypatch.setattr(multi_mod, "forward_hidden_states_all_layers", fake_forward)
    bundle = SimpleNamespace(model_id="fake", tokenizer=CharChatTok(), model=CharModel(), num_layers=2)
    probe = ConceptProbe(
        model_bundle=bundle,
        config=resolve_config(config or {"output": {"console": False}, "plots": {"heatmap_html": False}}),
        concept=ConceptSpec(name=name, pos_label="p", neg_label="n", pos_system="a", neg_system="b"),
        run_dir=str(tmp_path / name),
        logger=JsonlLogger(str(tmp_path / name / "log.jsonl")),
    )
    probe.concept_vectors = np.array([[1.0, 0.0], [1.0, 0.0]] if vectors is None else vectors, dtype=np.float32)
    probe.best_layer = 0
    # per-item references (an item's mean) and per-token references (single tokens), measured on
    # generated completions, and the same numbers as the transfer check's references for raw texts
    probe.score_center = np.array([0.0, 0.0])
    probe.score_scale = np.array([0.5, 1.0])
    probe.token_center = np.array([0.5, 0.0])
    probe.token_scale = np.array([0.25, 1.0])
    probe.reference_context = {"kind": "generated", "mode": "assistant_all_mean"}
    probe.transfer_references = {
        "context": probe_mod._text_context("raw", None, None, "assistant_all_mean"),
        "score_center": probe.score_center, "score_scale": probe.score_scale,
        "token_center": probe.token_center, "token_scale": probe.token_scale,
    }
    probe.sigma_best_layer = 0.5
    probe.resid_norm = np.array([2.0, 4.0])
    return probe


def test_score_texts_reads_the_raw_text_by_default(tmp_path, monkeypatch):
    rec = _probe(tmp_path, monkeypatch).score_texts(["ppx"])[0]
    assert rec["role"] == "raw" and rec["system_prompt"] is None
    assert rec["input_text"] == "^ppx" and rec["text_span"] == [1, 4]
    z = np.load(rec["npz_path"])
    np.testing.assert_allclose(z["scores_agg"], [0, 1, 1, 0])
    # single tokens use the token references: (score - 0.5) / 0.25
    np.testing.assert_allclose(z["scores_agg_z"], [-2, 2, 2, -2])
    assert rec["scores_standardized"]
    # the text's mean over its own tokens uses the item references: (2/3 - 0) / 0.5
    assert rec["score_mean"] == pytest.approx(2 / 3)
    assert rec["score_mean_z"] == pytest.approx(4 / 3)


def test_score_texts_rejects_settings_its_role_would_ignore(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="raw"):
        probe.score_texts(["ppx"], system_prompt="S")
    with pytest.raises(ValueError, match="assistant"):
        probe.score_texts(["ppx"], role="user", prompt="q")
    rec = probe.score_texts(["ppx"], role="user", system_prompt="S")[0]
    assert rec["input_text"] == "^<s>S|<u>ppx|"


def test_score_prompts_adds_no_system_and_excludes_the_end_of_sequence_token(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    rec = probe.score_prompts(["hi"], alphas=[0.0], save_html=False)[0]
    assert rec["prompt_text"] == "^<u>hi|<a>"
    assert rec["completion"] == "pp"
    assert rec["completion_end"] == rec["prompt_len"] + 2
    assert rec["score_mean"] == pytest.approx(1.0)  # 'pp' only; with the '$' it would be 2/3
    assert rec["score_mean_z"] == pytest.approx(2.0)  # (1 - 0) / 0.5, item references


def test_resid_norm_steering_scales_each_layer(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    rec = probe.score_prompts(["hi"], alphas=[0.1], alpha_unit="resid_norm", steer_layers=[0, 1], save_html=False)[0]
    assert rec["alpha_value"] == pytest.approx(0.1)
    assert rec["steer_layer_scales"] == {0: 2.0, 1: 4.0}


def test_the_random_control_vector_of_a_probe_is_the_same_everywhere(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    text_rec = probe.score_texts(["ppx"], random_scoring_vector=True)[0]
    prompt_rec = probe.score_prompts(["hi"], alphas=[1.0], random_scoring_vector=True,
                                     random_steering_vector=True, save_html=False)[0]
    multi = multi_probe_score_prompts([probe], ["hi"], random_scoring_vector=True, save_html=False,
                                      output_root=str(tmp_path / "multi"))
    seeds = {
        text_rec["random_scoring_vector_seed"],
        prompt_rec["random_scoring_vector_seed"],
        prompt_rec["random_steering_vector_seed"],
        multi["results"][0]["random_scoring_vector_seed_by_probe"]["c"],
    }
    assert len(seeds) == 1


def test_multi_probe_score_means_exclude_the_end_of_sequence_token(tmp_path, monkeypatch):
    a = _probe(tmp_path, monkeypatch, "a")
    b = _probe(tmp_path, monkeypatch, "b", vectors=[[-1.0, 0.0], [-1.0, 0.0]])
    rec = multi_probe_score_prompts([a, b], ["hi"], save_html=False, output_root=str(tmp_path / "m"))["results"][0]
    assert rec["score_mean_by_probe"] == {"a": pytest.approx(1.0), "b": pytest.approx(-1.0)}
    assert rec["score_mean_z_by_probe"]["a"] == pytest.approx(2.0)


def test_a_repeated_run_name_does_not_mix_runs(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    kwargs = dict(save_html=False, output_root=str(tmp_path / "m"), run_name="same")
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
    a = _probe(tmp_path, monkeypatch, "a")
    b = _probe(tmp_path, monkeypatch, "b", config=_cfg(model={"dtype": "float32"}))
    with pytest.raises(ValueError, match="dtype"):
        multi_probe_score_prompts([a, b], ["hi"], output_root=str(tmp_path / "m"))


def test_multi_probe_generation_settings_must_agree_or_be_passed(tmp_path, monkeypatch):
    a = _probe(tmp_path, monkeypatch, "a")
    b = _probe(tmp_path, monkeypatch, "b", config=_cfg(steering={"steer_temperature": 0.2}))
    with pytest.raises(ValueError, match="steer_temperature"):
        multi_probe_score_prompts([a, b], ["hi"], output_root=str(tmp_path / "m"))
    out = multi_probe_score_prompts([a, b], ["hi"], temperature=0.5, output_root=str(tmp_path / "m"))
    assert out["results"][0]["completion"] == "pp"


def test_multi_probe_steers_with_the_steering_probe_settings(tmp_path, monkeypatch):
    import json

    a = _probe(tmp_path, monkeypatch, "a")
    b = _probe(tmp_path, monkeypatch, "b", config=_cfg(steering={"steer_distribute": False}))
    out = multi_probe_score_prompts([a, b], ["hi"], alphas=[1.0], steer_probe=1, steer_layers=[0],
                                    output_root=str(tmp_path / "m"))
    config = json.load(open(os.path.join(out["run_dir"], "config.json")))
    assert config["steering"]["steer_distribute"] is False


# ------------------------------------------------------------- marker
def test_the_multi_probe_eval_applies_the_marker(tmp_path, monkeypatch):
    from concept_probe import run_multi_scored_eval

    probe = _probe(tmp_path, monkeypatch, "a")
    items = [{"prompt": "hi", "expected": "pp"}]
    kwargs = dict(analyze=False, output_root=str(tmp_path / "m"))
    assert run_multi_scored_eval([probe], items, **kwargs).per_sample[0]["correct"] is True
    # the completion 'pp' has no ':', so with the marker it cannot match
    assert run_multi_scored_eval([probe], items, marker=":", **kwargs).per_sample[0]["correct"] is False


def test_a_marker_with_a_custom_evaluator_raises():
    from concept_probe import run_multi_scored_eval, run_scored_eval

    fake = SimpleNamespace(console=None, concept=SimpleNamespace(name="x"))
    custom = lambda completion, item: {"correct": True}  # noqa: E731
    with pytest.raises(ValueError, match="marker"):
        run_scored_eval(fake, [{"prompt": "hi"}], evaluator=custom, marker=":")
    with pytest.raises(ValueError, match="marker"):
        run_multi_scored_eval([fake], [{"prompt": "hi"}], evaluator=custom, marker=":")

