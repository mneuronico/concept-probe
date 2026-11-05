"""Generating in batches, reading while generating, answer probabilities and the files of scores
(layer 5), on the tiny deterministic model."""
import json
import os

import numpy as np
import pytest
import torch

from concept_probe import Add, Scores, answer_logprobs, generate, intervene
from concept_probe.generation import TokenScorer, _project_tokens
from fakes import CharChatTok
from tiny_model import EOS, TinyCausalLM

PROMPTS = ["tell me", "z", "a story xy", "what now?", "x or y"]
RNG = np.random.default_rng(0)
V = RNG.normal(size=16)


def _gen(model, prompts=PROMPTS, **kw):
    return generate(model, CharChatTok(), prompts, max_new_tokens=6, save=False, **kw)


# --------------------------------------------------------------------------------------- batches
@pytest.mark.parametrize("intervention", [None, Add({1: V}, 3.0), Add({1: V}, 3.0, positions="generated"),
                                          Add({0: V}, 4.0, positions=[-1])])
def test_a_batch_padded_on_the_left_generates_what_each_prompt_generates_alone(intervention):
    model = TinyCausalLM()
    batched = _gen(model, batch_size=5, intervention=intervention)
    alone = _gen(model, batch_size=1, intervention=intervention)
    for b, a in zip(batched, alone):
        assert b.text == a.text and b.prompt_len == a.prompt_len and b.completion_end == a.completion_end
        np.testing.assert_array_equal(b.token_ids, a.token_ids)


def test_a_generation_keeps_its_end_of_sequence_token_and_drops_the_padding():
    model = TinyCausalLM()
    gens = _gen(model)
    tok = CharChatTok()
    for i, g in enumerate(gens):
        assert g.index == i and g.prompt == PROMPTS[i]
        assert tok.decode(g.token_ids[:g.prompt_len].tolist()) == f"^<u>{PROMPTS[i]}|<a>"
        assert g.text == tok.decode(g.completion_ids.tolist())
        if g.completion_end < len(g.token_ids):
            assert g.token_ids[g.completion_end] == EOS and len(g.token_ids) == g.completion_end + 1


def test_steering_changes_the_completions_and_is_recorded():
    model = TinyCausalLM()
    clean = _gen(model)
    steered = _gen(model, intervention=Add({1: V}, 8.0))
    assert any(c.text != s.text for c, s in zip(clean, steered))
    assert steered[0].intervention["alpha"] == {1: 8.0} and clean[0].intervention is None


def test_generations_are_saved_when_asked(tmp_path):
    gens = generate(TinyCausalLM(), CharChatTok(), PROMPTS[:2], max_new_tokens=4, save=True, out_dir=str(tmp_path))
    assert gens.dir == str(tmp_path)
    lines = [json.loads(l) for l in open(tmp_path / "generations.jsonl", encoding="utf-8")]
    assert lines[0]["settings"]["greedy"] is True and [l["text"] for l in lines[1:]] == [g.text for g in gens]
    data = np.load(tmp_path / "generations.npz")
    np.testing.assert_array_equal(data["token_ids"][data["offsets"][1]:data["offsets"][2]], gens[1].token_ids)


# ----------------------------------------------------------------------------- reading as it goes
def _scorer():
    directions = np.random.default_rng(1).normal(size=(4, 16)).astype(np.float32)
    return TokenScorer("c", [2], directions[[2]])


def test_on_token_gets_the_score_of_each_token_as_it_is_read():
    model, tok, scorer = TinyCausalLM(), CharChatTok(), _scorer()
    steps = []
    gens = generate(model, tok, PROMPTS, max_new_tokens=6, save=False, batch_size=5, on_token=steps.append,
                    watch=[scorer])
    reread = _project_tokens(model, tok, [torch.as_tensor(g.token_ids) for g in gens], [scorer])
    assert steps[0].step == 0 and steps[0].prompts == list(range(5))
    for row, g in enumerate(gens):
        # step k reads the token at position prompt_len - 1 + k
        for k, step in enumerate(steps):
            pos = g.prompt_len - 1 + k
            if pos >= len(g.token_ids):
                assert not step.active[row]
                continue
            assert step.active[row] and step.token_ids[row] == g.token_ids[pos]
            np.testing.assert_allclose(step.scores["c"][row], reread[row]["c"][pos, 0], rtol=1e-5, atol=1e-5)
        assert step.z["c"] is None  # no calibration


def test_reading_while_steering_includes_the_added_vector():
    model, tok, scorer = TinyCausalLM(), CharChatTok(), _scorer()
    steer = Add({2: scorer.vectors[0]}, 2.0)
    steps = []
    gens = generate(model, tok, PROMPTS[:1], max_new_tokens=3, save=False, intervention=steer, on_token=steps.append,
                    watch=[scorer])
    clean = _project_tokens(model, tok, [torch.as_tensor(gens[0].token_ids)], [scorer])[0]["c"][:, 0]
    with intervene(model, steer):
        steered = _project_tokens(model, tok, [torch.as_tensor(gens[0].token_ids)], [scorer])[0]["c"][:, 0]
    first = gens[0].prompt_len - 1
    np.testing.assert_allclose(steps[0].scores["c"][0], steered[first], rtol=1e-5)
    assert steered[first] - clean[first] == pytest.approx(2.0 * float(scorer.vectors[0] @ scorer.vectors[0]), rel=1e-4)


def test_kept_logits_are_the_steered_model_s_at_each_generated_token():
    model, tok = TinyCausalLM(), CharChatTok()
    steer = Add({1: V}, 3.0)
    gens = generate(model, tok, PROMPTS[:3], max_new_tokens=5, save=False, intervention=steer, keep_logits=3)
    full = generate(model, tok, PROMPTS[:3], max_new_tokens=5, save=False, intervention=steer, keep_logits="all")
    for g, f in zip(gens, full):
        n = len(g.token_ids) - g.prompt_len
        assert g.logits["values"].shape == (n, 3) and f.logits["values"].shape == (n, 128)
        with intervene(model, steer), torch.no_grad():
            logits = model(torch.as_tensor(g.token_ids)[None]).logits[0]
        expected = logits[g.prompt_len - 1:len(g.token_ids) - 1]
        np.testing.assert_allclose(f.logits["values"], expected.numpy(), rtol=1e-5)
        top = torch.topk(expected, 3, dim=-1)
        np.testing.assert_array_equal(g.logits["indices"], top.indices.numpy())


# ---------------------------------------------------------------------------- answer probabilities
def _manual_logprob(model, tok, prompt, answer):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt")[0]
    ans = torch.tensor([ord(c) for c in answer])
    logp = torch.log_softmax(model(torch.cat([ids, ans])[None]).logits[0].float(), dim=-1)
    return float(sum(logp[len(ids) - 1 + j, a] for j, a in enumerate(ans)))


def test_answer_logprobs_match_the_model_and_do_not_depend_on_the_batch():
    model, tok = TinyCausalLM(), CharChatTok()
    out = answer_logprobs(model, tok, PROMPTS, ["p", "nn", "x y"], batch_size=4, save=False)
    one = answer_logprobs(model, tok, PROMPTS, ["p", "nn", "x y"], batch_size=1, save=False)
    np.testing.assert_allclose(out.logprobs, one.logprobs, rtol=1e-9)
    for i, prompt in enumerate(PROMPTS):
        for k, answer in enumerate(["p", "nn", "x y"]):
            assert out.logprobs[i, k] == pytest.approx(_manual_logprob(model, tok, prompt, answer), rel=1e-9)
    assert out.n_tokens[0].tolist() == [1, 2, 3] and out.answer_tokens[0][1] == ["n", "n"]
    np.testing.assert_allclose(out.probs().sum(axis=1), 1.0)
    np.testing.assert_allclose(out.log_odds(0, 1), out.logprobs[:, 0] - out.logprobs[:, 1])


def test_answer_logprobs_under_an_intervention_and_with_answers_per_prompt(tmp_path):
    model, tok = TinyCausalLM(), CharChatTok()
    clean = answer_logprobs(model, tok, PROMPTS[:2], [["p", "n"], ["x", "z"]], save=False)
    steer = Add({1: V}, 5.0, positions=[-1])  # only where the first answer token is predicted
    steered = answer_logprobs(model, tok, PROMPTS[:2], [["p", "n"], ["x", "z"]], intervention=steer, save=True,
                              out_dir=str(tmp_path))
    assert not np.allclose(clean.logprobs, steered.logprobs)
    assert steered.answers == [["p", "n"], ["x", "z"]] and steered.intervention["positions"] == [-1]
    assert json.load(open(tmp_path / "answer_logprobs.json"))["logprobs"] == steered.logprobs.tolist()
    with pytest.raises(ValueError, match="same, non-zero number"):
        answer_logprobs(model, tok, PROMPTS[:2], [["p"], ["x", "z"]], save=False)


# ------------------------------------------------------------------------------------------ scores
def test_scores_saved_and_loaded_are_the_same(tmp_path):
    from concept_probe.scoring import ScoreInput, score_inputs

    model, tok, scorer = TinyCausalLM(), CharChatTok(), _scorer()
    gens = _gen(model, PROMPTS[:3], keep_logits=2)
    inputs = [ScoreInput(torch.as_tensor(g.token_ids), (g.prompt_len, g.completion_end), g.text, g.index,
                         {"prompt": g.prompt, "alpha": 0.0}, logits=g.logits) for g in gens]
    scores = Scores("generations", ["c"], score_inputs(model, tok, inputs, [scorer]), {"note": "x"}, {"c": "none"})
    scores.save(str(tmp_path), html=True)
    assert sorted(os.listdir(tmp_path)) == ["heatmaps.html", "scores.jsonl", "scores.npz", "settings.json"]
    back = Scores.load(str(tmp_path))
    assert back.probes == ["c"] and back.settings == {"note": "x"} and back.z_missing == {"c": "none"}
    for a, b in zip(scores, back):
        assert (a.index, a.text, a.span, a.tokens, a.info) == (b.index, b.text, b.span, b.tokens, b.info)
        np.testing.assert_array_equal(a.token_ids, b.token_ids)
        np.testing.assert_allclose(a.scores["c"], b.scores["c"])
        assert a.score_mean == b.score_mean and b.token_z["c"] is None
        np.testing.assert_allclose(a.logits["values"], b.logits["values"])
    np.testing.assert_allclose(back.means(), scores.means())
    assert scores.column("alpha") == [0.0, 0.0, 0.0]
