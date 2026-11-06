"""Model-free tests of what the library feeds the model, and of what score_texts/score_prompts return."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from concept_probe import ReadContext, preview_input
from concept_probe.linear_probe import Calibration
from concept_probe import ConceptProbe, evaluate_steering
from concept_probe.inputs import _encode_text_for_read, _normalize_messages
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
def _probe(tmp_path, name="c", vectors=None):
    # CharModel's hidden states are those of fakes.fake_forward: 'p' is +1, 'n' is -1 at layer 0
    probe = ConceptProbe(name=name, pos_label="p", neg_label="n", run_dir=str(tmp_path / name),
                         identity={"model_id": "fake"})
    probe.model_bundle = SimpleNamespace(model_id="fake", tokenizer=CharChatTok(), model=CharModel(), num_layers=2,
                                         identity={"model_id": "fake"})
    probe.directions = np.array([[1.0, 0.0], [1.0, 0.0]] if vectors is None else vectors, dtype=np.float32)
    probe.steer_directions = probe.directions
    probe.layer = 0
    # per-item references (an item's mean) and per-token references (single tokens), measured on
    # generated completions, and the same numbers for raw texts
    refs = dict(center=np.array([0.0, 0.0]), scale=np.array([0.5, 1.0]),
                token_center=np.array([0.5, 0.0]), token_scale=np.array([0.25, 1.0]))
    probe.calibrations = [Calibration(ReadContext.generated(), source="evaluation", **refs),
                          Calibration(ReadContext(role="raw"), source="texts", **refs)]
    probe.steer_sigma = 0.5
    probe.resid_norm = np.array([2.0, 4.0])
    return probe


def test_score_texts_reads_the_raw_text_by_default(tmp_path):
    scores = _probe(tmp_path).score_texts(["ppx"], save=True, html=False)
    it = scores[0]
    assert it.info["role"] == "raw" and it.info["system"] is None
    assert it.info["input_text"] == "^ppx" and it.span == (1, 4)
    np.testing.assert_allclose(it.token_scores["c"], [0, 1, 1, 0])
    # single tokens use the token references: (score - 0.5) / 0.25
    np.testing.assert_allclose(it.token_z["c"], [-2, 2, 2, -2])
    # the text's mean over its own tokens uses the item references: (2/3 - 0) / 0.5
    assert scores.means()[0] == pytest.approx(2 / 3)
    assert scores.z()[0] == pytest.approx(4 / 3)
    # other summaries of the text's tokens
    assert scores.per_text("max")[0] == pytest.approx(1.0) and scores.per_text("min")[0] == pytest.approx(0.0)
    assert scores.per_text("max", z=True)[0] == pytest.approx(2.0)
    # and the same numbers are on disk, one set of files for the call
    saved = type(scores).load(scores.dir)
    np.testing.assert_allclose(saved[0].token_z["c"], it.token_z["c"])
    assert sorted(os.listdir(scores.dir)) == ["scores.jsonl", "scores.npz", "settings.json"]


def test_score_texts_warns_about_settings_its_role_would_ignore(tmp_path):
    probe = _probe(tmp_path)
    with pytest.warns(UserWarning, match="ignored"):
        probe.score_texts(["ppx"], system="S", save=False)
    with pytest.warns(UserWarning, match="prompt.*ignored"):
        probe.score_texts(["ppx"], role="user", prompt="q", save=False)
    it = probe.score_texts(["ppx"], role="user", system="S", save=False)[0]
    assert it.info["input_text"] == "^<s>S|<u>ppx|"


def test_completions_are_scored_without_the_end_of_sequence_token(tmp_path):
    probe = _probe(tmp_path)
    ev = evaluate_steering(probe, ["hi"], [1.0], unit="raw", n_random=0, n_bootstrap=10, save=False)
    it = next(it for it in ev.scores if it.info["dose"] == 0.0)
    assert "".join(it.tokens[:it.span[0]]) == "^<u>hi|<a>"
    assert it.text == "pp"
    assert it.span[1] == it.span[0] + 2 and it.tokens[it.span[1]] == "$"
    assert it.score_mean["c"] == pytest.approx(1.0)  # 'pp' only; with the '$' it would be 2/3
    assert it.info["c (eval SD)"] == pytest.approx(2.0)  # 1 / 0.5, the SD of the probe's eval items


def test_resid_norm_steering_scales_each_layer_and_divides_only_when_asked(tmp_path):
    probe = _probe(tmp_path)
    add = probe.steering(0.1, "resid_norm", layers=[0, 1])
    add.prepare(2)
    assert add.alpha_per_layer == {0: pytest.approx(0.2), 1: pytest.approx(0.4)}
    shared = probe.steering(0.1, "resid_norm", layers=[0, 1], distribute=True)
    shared.prepare(2)
    assert shared.alpha_per_layer == {0: pytest.approx(0.1), 1: pytest.approx(0.2)}


def test_random_controls_come_in_the_same_call(tmp_path):
    probe = _probe(tmp_path)
    ev = evaluate_steering(probe, ["hi", "yo"], [1.0, -1.0], n_random=3, layers=[0, 1], n_bootstrap=10, save=False)
    # each dose: the probe and its 3 controls, for each prompt; and dose 0, added
    assert len(ev.scores) == 2 * (1 + 2 * 4)
    assert ev.settings["n_random"] == 3 and ev.settings["random_seed"] == 0
    seeds = {(it.info["dose"], it.info["control"]): (it.info["intervention"] or {}).get("control_seed")
             for it in ev.scores}
    assert seeds[(1.0, 0)] == [0, 0] and seeds[(-1.0, 0)] == [0, 0]  # the same random direction at every dose
    assert "dose 0 added" in [c["chosen"] for c in ev.settings["choices"]]


def test_probes_of_another_model_are_not_read_without_saying_so(tmp_path):
    probe, other = _probe(tmp_path), _probe(tmp_path, name="other")
    other.identity = {"model_id": "another"}
    with pytest.raises(ValueError, match="another model"):
        evaluate_steering(probe, ["hi"], [1.0], unit="raw", probes=[other], n_random=0, save=False)
    ev = evaluate_steering(probe, ["hi"], [1.0], unit="raw", probes=[other], n_random=0, n_bootstrap=10, save=False,
                           allow_model_mismatch=True)
    assert "other" in ev.measures
