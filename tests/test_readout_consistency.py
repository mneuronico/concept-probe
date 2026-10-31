"""Model-free tests that labelled texts are read like the vector's training data.

The character-level fakes (tests/fakes.py) make the pooled spans exact: a token is worth +1
for 'p', -1 for 'n' and 0 otherwise, so the mean over a text's own tokens is known in closed
form, and any template token in the pool would dilute it.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from concept_probe import probe as probe_mod
from concept_probe.config import ConceptSpec, resolve_config
from concept_probe.logger import JsonlLogger
from concept_probe.probe import ConceptProbe, _encode_text_for_read, reps_from_hs_layers
from concept_probe.utils import deep_merge
from fakes import CharChatTok, fake_forward


_CharChatTok = CharChatTok
_fake_forward = fake_forward


# ----------------------------------------------------------------- text spans
def test_user_role_span_is_the_text_itself():
    tok = _CharChatTok()
    ids, start, end = _encode_text_for_read(tok, "ppx", "sys", role="user")
    assert tok.decode(ids[0]) == "^<s>sys|<u>ppx|"
    assert tok.decode(ids[0, start:end]) == "ppx"


def test_user_role_span_when_the_template_strips_the_text():
    tok = _CharChatTok(strip_content=True)
    ids, start, end = _encode_text_for_read(tok, "  ppx ", "sys", role="user")
    assert tok.decode(ids[0, start:end]) == "ppx"


def test_user_role_span_of_a_message_list_is_its_last_message():
    tok = _CharChatTok()
    msgs = [{"role": "user", "content": "nn"}, {"role": "assistant", "content": "pp"}, {"role": "user", "content": "px"}]
    ids, start, end = _encode_text_for_read(tok, msgs, "sys", role="user")
    assert tok.decode(ids[0, start:end]) == "px"


def test_a_text_that_also_occurs_in_the_template_is_found_in_its_own_turn():
    tok = _CharChatTok()
    ids, start, end = _encode_text_for_read(tok, "s", "s", role="user")  # "s" is also the system text
    assert (start, end) == (len("^<s>s|<u>"), len("^<s>s|<u>s"))


def test_unknown_span_makes_assistant_modes_refuse_instead_of_reading_the_template():
    ids, start, end = _encode_text_for_read(_CharChatTok(offsets=False), "ppx", "sys", role="user")
    assert start is None and end is None
    hs = _fake_forward(None, ids)
    with pytest.raises(ValueError, match="sequence_"):
        reps_from_hs_layers(hs, start, "assistant_all_mean", 4, span_end=end)
    reps_from_hs_layers(hs, start, "sequence_all_mean", 4)  # whole-sequence modes still work


def test_assistant_modes_pool_over_the_span_only():
    ids, start, end = _encode_text_for_read(_CharChatTok(), "ppx", "sys", role="user")
    hs = _fake_forward(None, ids)
    np.testing.assert_allclose(reps_from_hs_layers(hs, start, "assistant_all_mean", 4, span_end=end)[0], [2 / 3, 1])
    # the last text token is 'x', not the end-of-turn '|'
    np.testing.assert_allclose(reps_from_hs_layers(hs, start, "assistant_last", 4, span_end=end)[0], [0, 1])
    np.testing.assert_allclose(reps_from_hs_layers(hs, start, "assistant_last_k_mean", 2, span_end=end)[0], [0.5, 1])


def test_sequence_pooling_depends_on_length_and_text_pooling_does_not():
    # The template prefix is identical in every input, so its weight in a sequence mean is 1/length.
    tok = _CharChatTok()
    seq, own = [], []
    for text in ("pp", "pppppp"):
        ids, start, end = _encode_text_for_read(tok, text, "sys", role="user")
        hs = _fake_forward(None, ids)
        seq.append(reps_from_hs_layers(hs, None, "sequence_all_mean", 4)[0, 0])
        own.append(reps_from_hs_layers(hs, start, "assistant_all_mean", 4, span_end=end)[0, 0])
    assert seq[0] < seq[1]
    np.testing.assert_allclose(own, [1.0, 1.0])


# ------------------------------------------------------------ train() end to end
def _fake_generate_batch(model, tokenizer, system, prompts, max_new_tokens, greedy, temperature, top_p,
                         batch_size=16, warn=None, warn_prefix="prompt"):
    # 'pp' (or 'nn') for each question, plus 'x' when it ends with '+', then the end-of-sequence
    # token '$', as a real generate returns it
    out = []
    for p in prompts:
        text = ("pp" if system == "POS" else "nn") + ("x" if p.endswith("+") else "") + "$"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": p}]
        ids = probe_mod.apply_chat(tokenizer, messages, add_generation_prompt=True)[0]
        out.append((torch.cat([ids, torch.tensor([ord(c) for c in text])]), int(ids.shape[-1])))
    return out


EVAL_POS = ["ppp", "ppx"]
EVAL_NEG = ["nnnnnn", "nnx"]


def _concept(**kwargs):
    fields = dict(
        name="c", pos_label="p", neg_label="n", pos_system="POS", neg_system="NEG",
        eval_pos_texts=EVAL_POS, eval_neg_texts=EVAL_NEG,
        train_pos_texts=["pp", "px"], train_neg_texts=["nnnn", "nx"],
    )
    fields.update(kwargs)
    return ConceptSpec(**fields)


def _train(tmp_path, monkeypatch, mode, overrides=None, concept=None):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    monkeypatch.setattr(probe_mod, "generate_batch", _fake_generate_batch)
    tok = _CharChatTok()
    cfg = resolve_config(deep_merge({
        "prompts": {"train_questions": ["q1", "q2"], "eval_questions": ["e1", "e2+"]},
        "training": {"train_prompt_mode": mode},
        "evaluation": {"n_permutations": 50, "n_bootstrap": 50},
        "plots": {"enabled": False},
        "output": {"console": False, "pretty_json_log": False},
    }, overrides or {}))
    bundle = SimpleNamespace(model_id="fake", tokenizer=tok, model=object(), num_layers=2)
    ConceptProbe(model_bundle=bundle, config=cfg, concept=concept or _concept(), run_dir=str(tmp_path),
                 logger=JsonlLogger(str(tmp_path / "log.jsonl"))).train()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    return tok, metrics, np.load(tmp_path / "tensors.npz"), events


def _warnings(events):
    return [e.get("message", "") for e in events if e["event"] == "warning"]


def test_shared_mode_evaluates_on_completions_generated_like_the_training_ones(tmp_path, monkeypatch):
    _, metrics, t, events = _train(tmp_path, monkeypatch, "shared")
    assert metrics["eval_mode_used"] == "generate"
    assert metrics["eval_readout"]["mode"] == "assistant_all_mean"
    assert [e["question"] for e in events if e["event"] == "eval_gen_pos"] == ["e1", "e2+"]
    # direction = e0; the completions 'pp' / 'ppx' (EOS excluded) score 1 and 2/3
    np.testing.assert_allclose(t["eval_pos_scores_mat"][:, 0], [1.0, 2 / 3], atol=1e-6)
    np.testing.assert_allclose(t["eval_neg_scores_mat"][:, 0], [-1.0, -2 / 3], atol=1e-6)


def test_eval_texts_of_a_shared_concept_are_a_transfer_check(tmp_path, monkeypatch):
    _, metrics, t, events = _train(tmp_path, monkeypatch, "shared")
    transfer = metrics["transfer_eval"]
    assert (transfer["readout"]["role"], transfer["readout"]["system"]) == ("raw", None)
    assert (transfer["n_pos"], transfer["n_neg"]) == (2, 2)
    assert [e["text"] for e in events if e["event"] == "transfer_read_pos"] == ["^ppp", "^ppx"]
    np.testing.assert_allclose(t["transfer_pos_scores_mat"][:, 0], [1.0, 2 / 3], atol=1e-6)
    np.testing.assert_allclose(t["transfer_neg_scores_mat"][:, 0], [-1.0, -2 / 3], atol=1e-6)
    assert transfer["d_at_best_layer"] == pytest.approx(float(t["transfer_sweep_d"][metrics["best_layer"]]), rel=1e-5)


def test_choosing_the_layer_on_eval_texts_in_shared_mode_is_explicit_and_warned(tmp_path, monkeypatch):
    _, metrics, _, events = _train(tmp_path, monkeypatch, "shared", {"evaluation": {"eval_mode": "read"}})
    assert metrics["eval_mode_used"] == "read" and metrics["eval_readout"]["role"] == "raw"
    assert metrics["transfer_eval"] is None
    assert any("eval_mode=read with train_prompt_mode=shared" in w for w in _warnings(events))


def test_eval_questions_that_repeat_training_questions_warn(tmp_path, monkeypatch):
    _, _, _, events = _train(tmp_path, monkeypatch, "shared", {"prompts": {"eval_questions": ["q1", "e2+"]}})
    assert any("not held out" in w for w in _warnings(events))


def test_generated_completions_are_pooled_without_their_end_of_sequence_token(tmp_path, monkeypatch):
    _, _, t, _ = _train(tmp_path, monkeypatch, "shared")
    # 'pp' then '$': the mean over 'pp' is exactly 1; with the '$' (value 0) it would be 2/3
    np.testing.assert_allclose(t["train_pos_reps"][:, 0, 0], [1.0, 1.0])
    np.testing.assert_allclose(t["train_neg_reps"][:, 0, 0], [-1.0, -1.0])


def test_texts_mode_reads_raw_texts_by_default(tmp_path, monkeypatch):
    _, metrics, t, events = _train(tmp_path, monkeypatch, "texts")
    assert (metrics["eval_readout"]["role"], metrics["eval_readout"]["system"]) == ("raw", None)
    assert metrics["transfer_eval"] is None
    assert [e["text"] for e in events if e["event"] == "train_read_pos"] == ["^pp", "^px"]
    assert [e["text"] for e in events if e["event"] == "eval_read_pos"] == ["^ppp", "^ppx"]
    np.testing.assert_allclose(t["train_pos_reps"][:, 0, 0], [1.0, 0.5])
    np.testing.assert_allclose(t["eval_pos_scores_mat"][:, 0], [1.0, 2 / 3], atol=1e-6)


def test_user_role_reads_texts_after_the_system_prompt(tmp_path, monkeypatch):
    _, metrics, t, events = _train(tmp_path, monkeypatch, "shared", {
        "prompts": {"neutral_system": "neutral"}, "training": {"train_text_role": "user"},
    })
    readout = metrics["transfer_eval"]["readout"]
    assert (readout["role"], readout["system"]) == ("user", "neutral")
    assert [e["text"] for e in events if e["event"] == "transfer_read_pos"][0] == "^<s>neutral|<u>ppp|"
    np.testing.assert_allclose(t["transfer_pos_scores_mat"][:, 0], [1.0, 2 / 3], atol=1e-6)  # text tokens only


def test_assistant_role_reads_texts_as_the_reply(tmp_path, monkeypatch):
    tok, metrics, _, _ = _train(tmp_path, monkeypatch, "shared", {
        "prompts": {"neutral_system": "neutral"}, "training": {"train_text_role": "assistant"},
    })
    assert metrics["transfer_eval"]["readout"]["role"] == "assistant"
    assert "^<s>neutral|<a>" in tok.rendered


def test_eval_system_follows_the_training_texts_system(tmp_path, monkeypatch):
    _, metrics, _, _ = _train(tmp_path, monkeypatch, "texts", {
        "training": {"train_text_role": "user", "train_text_system": "custom"},
    })
    assert metrics["eval_readout"]["system"] == "custom"


def test_a_different_eval_system_warns(tmp_path, monkeypatch):
    _, _, _, events = _train(tmp_path, monkeypatch, "texts", {
        "training": {"train_text_role": "user"}, "evaluation": {"eval_system": "other"},
    })
    assert any("eval_system differs" in w for w in _warnings(events))


def test_sequence_readouts_warn_about_length(tmp_path, monkeypatch):
    _, _, _, events = _train(tmp_path, monkeypatch, "texts", {"readout": {"read_mode": "sequence_all_mean"}})
    assert any("1/length" in w for w in _warnings(events))


@pytest.mark.parametrize("overrides", [
    {"training": {"train_text_system": "sys"}},                            # raw ignores a system prompt
    {"training": {"train_text_prompt": "q?"}},                             # raw ignores a prompt
    {"evaluation": {"eval_system": "sys"}},                                # raw ignores a system prompt
    {"training": {"train_text_role": "user", "train_text_prompt": "q?"}},  # user ignores a prompt
    {"training": {"train_text_role": "assistant"}},                        # nothing before the reply
    {"training": {"sweep_select": "pvalue"}},
    {"evaluation": {"selection_split": 1.5}},
])
def test_settings_that_would_be_ignored_or_misread_raise_before_anything_runs(tmp_path, monkeypatch, overrides):
    def must_not_generate(*args, **kwargs):
        raise AssertionError("generation ran before the settings were checked")

    monkeypatch.setattr(probe_mod, "generate_batch", must_not_generate)
    cfg = resolve_config(deep_merge({"output": {"console": False}}, overrides))
    bundle = SimpleNamespace(model_id="fake", tokenizer=_CharChatTok(), model=object(), num_layers=2)
    probe = ConceptProbe(model_bundle=bundle, config=cfg, concept=_concept(), run_dir=str(tmp_path),
                         logger=JsonlLogger(str(tmp_path / "log.jsonl")))
    with pytest.raises(ValueError):
        probe.train()
    assert not (tmp_path / "config.json").exists()


def test_identical_system_prompts_raise_in_contrastive_modes(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="pos_system == concept.neg_system"):
        _train(tmp_path, monkeypatch, "shared", concept=_concept(pos_system="", neg_system=""))
    assert not (tmp_path / "config.json").exists()


def test_defaults_read_raw_text_with_no_system_prompt():
    cfg = resolve_config()
    assert cfg["readout"]["read_mode"] == "assistant_all_mean"
    assert cfg["training"]["train_text_role"] == "raw"
    assert cfg["prompts"]["neutral_system"] is None
    assert cfg["evaluation"]["eval_system"] is None
    assert cfg["concept"]["pos_system"] == cfg["concept"]["neg_system"] == ""


# ------------------------------------------------------------ sweep statistics
MANY_POS = ["ppp", "ppx", "pxx", "ppppx", "pppx", "ppxx", "pp", "pxpx"]
MANY_NEG = ["nnn", "nnx", "nxx", "nnnnx", "nnnx", "nnxx", "nn", "nxnx"]


def test_layer_is_chosen_on_one_part_of_the_eval_items_and_d_measured_on_the_rest(tmp_path, monkeypatch):
    _, metrics, t, _ = _train(tmp_path, monkeypatch, "texts",
                              concept=_concept(eval_pos_texts=MANY_POS, eval_neg_texts=MANY_NEG))
    assert metrics["selection"] == "split"
    assert metrics["n_eval_selection"] == {"pos": 4, "neg": 4}
    held_pos = np.setdiff1d(np.arange(8), t["eval_selection_pos_idx"])
    held_neg = np.setdiff1d(np.arange(8), t["eval_selection_neg_idx"])
    layer = metrics["best_layer"]
    expected = probe_mod.cohen_d_np(t["eval_pos_scores_mat"][held_pos, layer].astype(np.float64),
                                    t["eval_neg_scores_mat"][held_neg, layer].astype(np.float64))
    assert metrics["best_d"] == pytest.approx(expected, rel=1e-5)
    lo, hi = metrics["best_d_ci95"]
    assert lo <= hi
    assert 0 < metrics["sweep_max_d_perm_p"] <= 1


def test_without_a_split_best_d_is_flagged_as_post_selection(tmp_path, monkeypatch):
    _, metrics, _, events = _train(tmp_path, monkeypatch, "texts", {"evaluation": {"selection_split": None}},
                                   concept=_concept(eval_pos_texts=MANY_POS, eval_neg_texts=MANY_NEG))
    assert metrics["selection"] == "all_eval"
    assert metrics["best_d"] == pytest.approx(metrics["best_d_all_eval"], rel=1e-5)
    assert any("optimistic" in w for w in _warnings(events))


def test_too_few_eval_items_to_split_fall_back_with_a_warning(tmp_path, monkeypatch):
    _, metrics, _, events = _train(tmp_path, monkeypatch, "shared")  # 2 eval questions per pole
    assert metrics["selection"] == "all_eval"
    assert any("selection_split needs" in w for w in _warnings(events))


def test_item_and_token_references(tmp_path, monkeypatch):
    _, metrics, t, _ = _train(tmp_path, monkeypatch, "shared")
    # per item (a completion's mean): pos 1, 2/3; neg -1, -2/3
    pos, neg = np.array([1.0, 2 / 3]), np.array([-1.0, -2 / 3])
    assert t["score_center"][0] == pytest.approx(0.0, abs=1e-6)
    assert t["score_scale"][0] == pytest.approx(np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2), rel=1e-5)
    # per token: pos tokens p p | p p x = 1 1 1 1 0; neg mirrored. Single tokens spread more than means.
    pos_tok = np.array([1.0, 1, 1, 1, 0])
    assert t["token_center"][0] == pytest.approx(0.0, abs=1e-6)
    assert t["token_scale"][0] == pytest.approx(pos_tok.std(ddof=1), rel=1e-5)
    assert t["token_scale"][0] > t["score_scale"][0]
    assert metrics["sigma_best_layer"] == pytest.approx(t["score_scale"][metrics["best_layer"]])
    assert t["resid_norm"].shape == (2,) and np.all(t["resid_norm"] > 0)


def test_without_eval_the_metrics_are_null_not_zero(tmp_path, monkeypatch):
    _, metrics, t, _ = _train(tmp_path, monkeypatch, "texts", concept=_concept(eval_pos_texts=[], eval_neg_texts=[]))
    assert metrics["best_d"] is None and metrics["best_p"] is None and metrics["selection"] is None
    assert np.all(np.isnan(t["sweep_d"])) and np.all(np.isnan(t["token_scale"]))


def test_few_eval_items_warn(tmp_path, monkeypatch):
    _, _, _, events = _train(tmp_path, monkeypatch, "shared")  # 2 eval questions per pole
    assert any("Only 2 positive and 2 negative eval items" in w for w in _warnings(events))


def test_default_eval_questions_are_many_and_held_out():
    prompts = resolve_config()["prompts"]
    assert len(prompts["eval_questions"]) >= probe_mod.MIN_EVAL_ITEMS_PER_POLE
    assert not set(prompts["eval_questions"]) & set(prompts["train_questions"])


def _generate_with_an_empty_reply(empty_question):
    def generate(model, tokenizer, system, prompts, *args, **kwargs):
        out = _fake_generate_batch(model, tokenizer, system, prompts, *args, **kwargs)
        for i, p in enumerate(prompts):
            if p == empty_question and system == "POS":
                ids, plen = out[i]
                out[i] = (torch.cat([ids[:plen], torch.tensor([ord("$")])]), plen)  # EOS right away
        return out
    return generate


def test_an_empty_completion_is_left_out_instead_of_pooling_the_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    monkeypatch.setattr(probe_mod, "generate_batch", _generate_with_an_empty_reply("q2"))
    cfg = resolve_config({
        "prompts": {"train_questions": ["q1", "q2", "q3"], "eval_questions": ["e1", "e2+"]},
        "evaluation": {"n_permutations": 10, "n_bootstrap": 10},
        "plots": {"enabled": False}, "output": {"console": False, "pretty_json_log": False},
    })
    bundle = SimpleNamespace(model_id="fake", tokenizer=_CharChatTok(), model=object(), num_layers=2)
    ConceptProbe(model_bundle=bundle, config=cfg, concept=_concept(), run_dir=str(tmp_path),
                 logger=JsonlLogger(str(tmp_path / "log.jsonl"))).train()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    t = np.load(tmp_path / "tensors.npz")
    assert metrics["dropped_items"] == {"train_pos": [1]}
    assert (metrics["n_train_pos"], metrics["n_train_neg"]) == (3, 3)  # generated; one left out of pooling
    np.testing.assert_allclose(t["train_pos_reps"][:, 0, 0], [1.0, 1.0])  # only real completions


def test_a_side_with_no_tokens_at_all_raises(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no item has tokens"):
        _train(tmp_path, monkeypatch, "texts", concept=_concept(train_pos_texts=["", ""]))


def test_extract_text_reps_names_an_empty_text(monkeypatch):
    monkeypatch.setattr(probe_mod, "forward_hidden_states_all_layers", _fake_forward)
    with pytest.raises(probe_mod.EmptySpanError, match=r"texts\[1\]"):
        probe_mod.extract_text_reps(object(), _CharChatTok(), ["pp", ""])
