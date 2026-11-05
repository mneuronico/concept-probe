"""Evaluating steering (layer 6c): the dose sweep, its random controls, the measures of effect and
coherence, and the largest coherent dose."""
import json
import os

import numpy as np
import pytest

from concept_probe import (Add, CoherenceRule, ConceptProbe, evaluate_steering, read_texts, repetition_rate)
from concept_probe.steering_eval import _largest_coherent
from fakes import CharChatTok
from test_reference_outputs import QUESTIONS, TRAIN_NEG, TRAIN_POS
from tiny_model import TinyCausalLM


@pytest.fixture(scope="module")
def probe():
    model, tok = TinyCausalLM(), CharChatTok()
    p = ConceptProbe.fit(read_texts(model, tok, TRAIN_POS, cache=False), read_texts(model, tok, TRAIN_NEG, cache=False),
                         layer=1, model=model, tokenizer=tok, save=False)
    return p.attach(model, tok)


def _p_share(texts):
    return [t.count("p") / max(1, len(t)) for t in texts]


def test_a_sweep_measures_each_dose_against_dose_zero_with_random_controls(probe):
    ev = evaluate_steering(probe, QUESTIONS, [3.0, -3.0], n_random=4, max_new_tokens=8, measures={"p_share": _p_share})
    assert ev.doses == [-3.0, 0.0, 3.0]  # dose 0 is added: the baseline
    assert ev.measures == {"concept": "probe raw score", "p_share": "function of the texts"}
    # every prompt at each dose, and under 4 controls at each non-zero dose
    assert len(ev.scores) == len(QUESTIONS) * (3 + 2 * 4)
    zero, up = ev.row(0.0), ev.row(3.0)
    assert zero["effects"]["concept"]["change"] == 0.0 and zero["coherence"]["controls"]["coherent"] == []
    effect = up["effects"]["concept"]
    assert effect["change"] > 0 and effect["ci95"][0] > 0 and len(effect["controls"]) == 4
    assert effect["control_p95_abs"] == pytest.approx(np.percentile(np.abs(effect["controls"]), 95))
    assert effect["beyond_controls"] is (abs(effect["change"]) > effect["control_p95_abs"])
    assert up["effects"]["p_share"]["change"] > 0 and ev.row(-3.0)["effects"]["concept"]["change"] < 0
    # each completion carries its measures
    it = ev.scores[0]
    assert {"dose", "control", "perplexity", "repetition", "n_tokens", "concept", "p_share"} <= set(it.info)


def test_a_control_is_the_same_random_direction_at_every_dose(probe):
    ev = evaluate_steering(probe, QUESTIONS[:2], [2.0, 4.0], n_random=2, random_seed=7, max_new_tokens=4)
    controls = [it.info["intervention"] for it in ev.scores if it.info["control"] == 1]
    assert {tuple(c["control_seed"]) for c in controls} == {(7, 1)}
    by_dose = {it.info["dose"]: it.info["intervention"] for it in ev.scores if it.info["control"] == 1}
    assert by_dose[4.0]["alpha"][1] == pytest.approx(2 * by_dose[2.0]["alpha"][1])


def test_an_addition_or_a_function_of_the_dose_can_be_evaluated(probe):
    model, tok = probe.model_bundle.model, probe.model_bundle.tokenizer
    v = np.asarray(probe.steer_directions[1])
    by_add = evaluate_steering(Add({1: v}), QUESTIONS[:3], [3.0], model=model, tokenizer=tok, n_random=2,
                               max_new_tokens=6, probes=[probe])
    by_probe = evaluate_steering(probe, QUESTIONS[:3], [3.0], n_random=2, max_new_tokens=6)
    assert [it.text for it in by_add.scores] == [it.text for it in by_probe.scores]
    by_function = evaluate_steering(lambda d: Add({1: v}, d), QUESTIONS[:3], [3.0], model=model, tokenizer=tok,
                                    n_random=0, max_new_tokens=6, probes=[probe])
    assert by_function.row(3.0)["effects"]["concept"]["change"] == pytest.approx(
        by_probe.row(3.0)["effects"]["concept"]["change"])
    with pytest.raises(ValueError, match="needs the model"):
        evaluate_steering(Add({1: v}), QUESTIONS[:3], [3.0])


def test_coherence_rule_and_the_largest_coherent_dose():
    rule = CoherenceRule()
    zero = {"perplexity": 10.0, "repetition": 0.05}
    assert rule({"perplexity": 19.0, "repetition": 0.2}, zero)
    assert not rule({"perplexity": 21.0, "repetition": 0.0}, zero)  # more than twice as surprising
    assert not rule({"perplexity": 10.0, "repetition": 0.3}, zero)  # repeats itself
    assert not rule({"perplexity": float("nan"), "repetition": 0.0}, zero)  # no completions to judge
    table = [{"dose": d, "coherence": {"coherent": c}} for d, c in
             [(-2.0, False), (-1.0, True), (0.0, True), (1.0, True), (2.0, False), (3.0, True)]]
    assert _largest_coherent(table) == {"positive": 1.0, "negative": -1.0}  # 3.0 is past a break
    assert _largest_coherent([{"dose": 0.0, "coherence": {"coherent": True}},
                              {"dose": 1.0, "coherence": {"coherent": False}}]) == {"positive": None, "negative": None}


def test_a_custom_rule_decides_coherence(probe):
    ev = evaluate_steering(probe, QUESTIONS[:2], [3.0], n_random=1, max_new_tokens=4,
                           coherence=lambda at_dose, at_zero: at_dose["n_tokens"] > 100)
    assert ev.row(3.0)["coherence"]["coherent"] is False and ev.largest_coherent_dose["positive"] is None


def test_repetition_rate():
    assert repetition_rate("the cat sat on the mat") == 0.0
    assert repetition_rate("go go go go go go") == pytest.approx(1 - 1 / 4)
    assert repetition_rate("two words") == 0.0


def test_measure_names_must_not_clash(probe):
    with pytest.raises(ValueError, match="clash"):
        evaluate_steering(probe, QUESTIONS[:1], [1.0], measures={"perplexity": _p_share}, max_new_tokens=2)
    with pytest.raises(ValueError, match="clash"):
        evaluate_steering(probe, QUESTIONS[:1], [1.0], measures={"concept": _p_share}, max_new_tokens=2)


def test_the_evaluation_is_saved_with_its_plot(probe, tmp_path):
    ev = evaluate_steering(probe, QUESTIONS[:2], [2.0, -2.0], n_random=2, max_new_tokens=4, save=True,
                           out_dir=str(tmp_path), html=False)
    assert ev.dir == str(tmp_path)
    assert sorted(os.listdir(tmp_path)) == ["dose_response.png", "scores.jsonl", "scores.npz", "settings.json",
                                            "steering.json"]
    saved = json.load(open(tmp_path / "steering.json", encoding="utf-8"))
    assert saved["doses"] == [-2.0, 0.0, 2.0] and saved["largest_coherent_dose"] == ev.largest_coherent_dose
    assert saved["settings"]["coherence_rule"]["perplexity_ratio"] == [0.5, 2.0]


def test_the_external_judge_is_one_more_measure(probe, monkeypatch):
    from concept_probe import coherence, coherence_judge

    sent = []

    def fake_call(api_key, model, messages):
        batch = json.loads(messages[1]["content"])
        sent.append(len(batch))
        labels = ["COHERENT", "PARTIALLY_COHERENT", "NONSENSE"]
        return json.dumps([{"id": x["id"], "rating": labels[x["id"] % 3]} for x in batch])

    monkeypatch.setattr(coherence, "_call_groq", fake_call)
    ev = evaluate_steering(probe, QUESTIONS[:3], [2.0], n_random=1, max_new_tokens=4,
                           measures={"judge": coherence_judge(api_key="key", max_elements_per_request=4)})
    assert sum(sent) == len(ev.scores) == 3 * 3  # every completion, in requests of at most 4
    assert [it.info["judge"] for it in ev.scores][:3] == [1.0, 0.5, 0.0]
    assert ev.measures["judge"] == "function of the texts"
