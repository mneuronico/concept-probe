"""Reference outputs, on a tiny deterministic model (tests/tiny_model.py).

``reference/reference_outputs.json`` holds what an earlier version of the library returned:
the training of a probe in texts mode, with a user role, with variance-normalized
directions, and on completions generated under a system prompt per pole (shared and opposed
questions); scoring texts token by token; steering and scoring the completions; reading texts and
completions. These tests compute the same things with today's layers (``read_texts``,
``generate``, ``ConceptProbe.fit`` with eval sets, ``evaluate_steering``) and check that the numbers
are the same, up to rounding (reading in batches changes the last digits in half precision).

When a change is meant to alter these outputs, regenerate the reference and commit it together with
a note of what changed and why:

    CP_UPDATE_REFERENCE=1 python -m pytest tests/test_reference_outputs.py
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from concept_probe import (ConceptProbe, EvalSet, ReadContext, Readout, evaluate_steering, generate, read_encoded,
                           read_texts)
from fakes import CharChatTok
from tiny_model import TinyCausalLM

REFERENCE = Path(__file__).parent / "reference" / "reference_outputs.json"
UPDATE = os.environ.get("CP_UPDATE_REFERENCE") == "1"
RTOL, ATOL = 1e-4, 1e-6


def _flip(texts):
    return [t.replace("p", "n") for t in texts]


# Sixteen eval items per pole: with the default split, eight are held out, so a bootstrap resample
# almost never repeats one item eight times (which makes the within-pole SD 0 and d explode).
TRAIN_POS = ["pxp yz", "ppy x", "zp pp", "yxpp", "p zp x", "xpyp", "pp z", "ypxp z", "zzp p", "pyx p"]
EVAL_POS = ["py p", "xp zp", "pp yx", "zpx", "p p y", "yp z", "xxpp", "pzy p",
            "pyz", "zzp x", "xpp", "ppz y", "yxp", "p xy", "zpp z", "ypy"]
TRAIN_NEG, EVAL_NEG = _flip(TRAIN_POS), _flip(EVAL_POS)
QUESTIONS = ["tell me", "a story xy", "z", "what now?", "say yz", "x or y"]
EVAL_QUESTIONS = ["go on", "why z", "pick x", "more y", "and then", "ok zz", "yes?", "a b c",
                  "say x", "tell z", "only y", "then what", "no", "x y", "why not", "go"]
TEXTS_TO_SCORE = ["pp px", "nn xn", "xyz", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "pz"},
                                           {"role": "user", "content": "np"}]]
POS_SYSTEM, NEG_SYSTEM = "pp pp pp", "nn nn nn"
MODEL, TOK = TinyCausalLM(), CharChatTok()
FIT = dict(n_permutations=200, n_bootstrap=200, seed=0, identity="tiny-test", save=False)


# --------------------------------------------------------------------------- helpers
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def _assert_close(actual, expected, path="$"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected a dict, got {actual!r}"
        assert set(actual) == set(expected), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {actual!r}"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) or isinstance(actual, float):
        assert actual is not None and expected is not None, f"{path}: {actual!r} != {expected!r}"
        a, e = float(actual), float(expected)
        if math.isnan(e) or math.isnan(a):
            assert math.isnan(a) and math.isnan(e), f"{path}: {a} != {e}"
        else:
            assert math.isclose(a, e, rel_tol=RTOL, abs_tol=ATOL), f"{path}: {a} != {e}"
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _stored(name):
    assert REFERENCE.exists(), "No reference; run with CP_UPDATE_REFERENCE=1 to create it."
    stored = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert name in stored, f"No reference for {name!r}; run with CP_UPDATE_REFERENCE=1 to create it."
    return stored[name]


def _check(name, value):
    """Compare ``value`` with the stored reference's entries of the same keys."""
    value = json.loads(json.dumps(_jsonable(value)))
    if UPDATE:
        stored = json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.exists() else {}
        stored[name] = {**stored.get(name, {}), **value} if isinstance(value, dict) else value
        REFERENCE.parent.mkdir(exist_ok=True)
        REFERENCE.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    expected = _stored(name)
    if isinstance(value, dict):
        expected = {k: expected[k] for k in value}
    _assert_close(value, expected, name)


def _text_set(name, pos, neg, **read):
    """Eval texts read like the training texts, pooled and token by token."""
    p = read_texts(MODEL, TOK, pos, ["text_mean", "text_tokens"], cache=False, **read)
    n = read_texts(MODEL, TOK, neg, ["text_mean", "text_tokens"], cache=False, **read)
    return EvalSet(name, p["text_mean"], n["text_mean"], tokens=(p["text_tokens"], n["text_tokens"]))


def _train_on_texts(name, estimator="mean_diff", **read):
    P = read_texts(MODEL, TOK, TRAIN_POS, cache=False, **read)
    N = read_texts(MODEL, TOK, TRAIN_NEG, cache=False, **read)
    return ConceptProbe.fit(P, N, eval_sets=_text_set("eval", EVAL_POS, EVAL_NEG, **read), estimator=estimator,
                            name=name, **FIT), (P, N), None


def _generated_set(name, gens_pos, gens_neg, system):
    """Completions as an eval set paired by question: read after their question alone (``system=None``)
    or after the system prompt they were generated with; a question with an empty completion on
    either pole is left out."""
    reads = []
    for gens in (gens_pos, gens_neg):
        encodings = gens.encodings() if system == "as generated" else gens.encodings(system=None)
        reads.append(read_encoded(MODEL, TOK, encodings, ["text_mean", "text_tokens"], skip_empty=True, cache=False))
    both = sorted(set(reads[0]["text_mean"].read_items) & set(reads[1]["text_mean"].read_items))
    sides, tokens = [], []
    for r in reads:
        kept = r["text_mean"].read_items
        sides.append(r["text_mean"].data[[kept.index(q) for q in both]])
        tok = r["text_tokens"]
        rows = np.isin(tok.items, both)
        tokens.append(type(tok)(data=tok.data[rows], layers=tok.layers, readout=tok.readout, items=tok.items[rows]))
    return EvalSet(name, sides[0], sides[1], context=ReadContext.generated(), paired=True,
                   tokens=(tokens[0], tokens[1])), both


def _train_on_generations(name, questions_pos, questions_neg):
    gen = dict(max_new_tokens=10, batch_size=4, save=False)
    P = read_encoded(MODEL, TOK, generate(MODEL, TOK, questions_pos, system=POS_SYSTEM, **gen).encodings(),
                     cache=False)
    N = read_encoded(MODEL, TOK, generate(MODEL, TOK, questions_neg, system=NEG_SYSTEM, **gen).encodings(),
                     cache=False)
    gp = generate(MODEL, TOK, EVAL_QUESTIONS, system=POS_SYSTEM, **gen)
    gn = generate(MODEL, TOK, EVAL_QUESTIONS, system=NEG_SYSTEM, **gen)
    main, both = _generated_set("eval", gp, gn, None)
    with_system, _ = _generated_set("eval_with_system_prompt", gp, gn, "as generated")
    texts = _text_set("texts", EVAL_POS, EVAL_NEG)
    probe = ConceptProbe.fit(P, N, eval_sets=[main, with_system, texts], name=name, **FIT)
    return probe, (P, N), both


TRAININGS = {
    "texts": lambda: _train_on_texts("texts"),
    "texts_var": lambda: _train_on_texts("texts_var", estimator="mean_diff_var"),
    "texts_user": lambda: _train_on_texts("texts_user", role="user", system="x y z"),
    "shared": lambda: _train_on_generations("shared", QUESTIONS, QUESTIONS),
    "opposed": lambda: _train_on_generations("opposed", QUESTIONS[:4], QUESTIONS[2:]),
}


@pytest.fixture(scope="module")
def probes():
    out = {}
    for key, train in TRAININGS.items():
        probe, train_reps, eval_questions = train()
        probe.attach(MODEL, TOK)
        out[key] = (probe, train_reps, eval_questions)
    return out


def _summary(probe, train_reps, eval_questions):
    ev = probe.evaluation
    main = ev.main
    cal = probe.calibrations[0]
    out = {
        "best_layer": probe.layer, "best_d": main.d, "best_p": main.p, "best_d_ci95": main.d_ci95,
        "selection": ev.selection, "best_d_selection": ev.d_selection, "best_d_all_eval": float(main.d_by_layer[probe.layer]),
        "n_eval_selection": {"pos": len(ev.selection_idx[0]), "neg": len(ev.selection_idx[1])},
        "sweep_max_d": ev.max_d, "sweep_max_d_perm_p": ev.max_d_permutation_p,
        "sigma_best_layer": float(cal.scale[probe.layer]), "n_train_pos": len(train_reps[0].data),
        "n_train_neg": len(train_reps[1].data), "n_eval_pos": main.n_pos, "n_eval_neg": main.n_neg,
        "eval_paired": main.paired, "score_center": cal.center, "score_scale": cal.scale,
        "token_center": cal.token_center, "token_scale": cal.token_scale, "concept_vectors": probe.directions,
        "sweep_d": probe.d_by_layer, "sweep_p": probe.p_by_layer, "resid_norm": probe.resid_norm,
        "train_pos_reps": train_reps[0].data, "train_neg_reps": train_reps[1].data,
        "eval_pos_scores_mat": main.pos_scores, "eval_neg_scores_mat": main.neg_scores,
    }
    texts = ev.sets.get("texts")
    out["transfer_eval"] = None if texts is None else {
        "n_pos": texts.n_pos, "n_neg": texts.n_neg, "d_at_best_layer": texts.d, "p_at_best_layer": texts.p,
        "d_ci95_at_best_layer": texts.d_ci95}
    out["transfer_sweep_d"] = np.full(probe.num_layers, np.nan) if texts is None else texts.d_by_layer
    if eval_questions is not None:
        out["eval_question_idx"] = eval_questions
    return out


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("key", list(TRAININGS))
def test_fitting_a_probe(probes, key):
    _check(f"train_{key}", _summary(*probes[key]))


def test_score_texts(probes):
    out = {}
    for key, kwargs in (("texts", {}), ("texts_user", {"role": "user", "system": "x y z"}),
                        ("texts", {"reference_layer": 0})):
        texts = TEXTS_TO_SCORE if kwargs.get("role") == "user" else TEXTS_TO_SCORE[:3]
        name = f"{key}_" + "_".join(sorted(k if k != "system" else "system_prompt" for k in kwargs)) if kwargs else key
        scores = probes[key][0].score_texts(texts, save=False, **kwargs)
        n = scores.probes[0]
        out[name] = [
            dict(input_text=it.info["input_text"], text_span=it.info["text_span"], layers=it.layers[n],
                 score_mean=it.score_mean[n], score_mean_z=it.score_mean_z[n],
                 scores_standardized=it.token_z[n] is not None, scores_per_layer=it.scores[n],
                 scores_agg=it.token_scores[n], **({} if it.token_z[n] is None else {"scores_agg_z": it.token_z[n]}))
            for it in scores
        ]
    stored = _stored("score_texts")
    for name, items in out.items():
        expected = [{k: e[k] for k in a} for a, e in zip(items, stored[name])]
        _assert_close(json.loads(json.dumps(_jsonable(items))), expected, name)


def _by_condition(ev):
    """The completions of an evaluation by (dose, control, prompt)."""
    return {(float(it.info["dose"]), it.info["control"], int(it.index)): it for it in ev.scores}


def _prompt_index(record):
    return next(i for i, q in enumerate(QUESTIONS[:3]) if q in record["prompt_text"])


def test_steering_and_scoring_the_completions(probes):
    shared, texts_probe = probes["shared"][0], probes["texts"][0]
    run = dict(n_bootstrap=50, max_new_tokens=10, save=False)
    calls = {
        "raw": (shared, dict(doses=[4.0, -4.0], unit="raw", n_random=0)),
        "sigma_window": (shared, dict(doses=[2.0], unit="sd", window=1, distribute=True, n_random=0)),
        "resid_norm_list": (shared, dict(doses=[0.3], unit="resid_norm", layers=[1, 2], n_random=0)),
        "random_controls": (shared, dict(doses=[4.0], unit="raw", n_random=2)),
    }
    stored = _stored("score_prompts")
    for name, (probe, kwargs) in calls.items():
        got = _by_condition(evaluate_steering(probe, QUESTIONS[:3], **kwargs, **run))
        for rec in stored[name]:
            it = got[(float(rec["alpha"]), rec["control"], _prompt_index(rec))]
            assert it.text == rec["completion"], (name, rec["alpha"], rec["control"])
            _assert_close(float(it.score_mean[probe.name]), rec["score_mean"], f"{name}.score_mean")
    # a probe trained on texts scores the completions it did not steer
    got = _by_condition(evaluate_steering(texts_probe, QUESTIONS[:3], [4.0], unit="raw", n_random=0, **run))
    for rec in stored["texts_probe"]:
        it = got[(0.0, None, _prompt_index(rec))]
        assert it.text == rec["completion"]
        _assert_close(float(it.score_mean[texts_probe.name]), rec["score_mean"], "texts_probe.score_mean")


def test_several_probes_read_the_same_completions(probes):
    shared, opposed, texts_var = (probes[k][0] for k in ("shared", "opposed", "texts_var"))
    ev = evaluate_steering(shared, QUESTIONS[:3], [4.0], unit="raw", probes=[shared, opposed, texts_var], n_random=0,
                           n_bootstrap=50, max_new_tokens=10, save=False)
    got = _by_condition(ev)
    for j, rec in enumerate(_stored("multi_probe_score_prompts")["records"]):  # by prompt, then by dose
        it = got[(float(rec["alpha_value"]), None, j // 2)]
        assert it.text == rec["completion"]
        for name, value in rec["score_mean_by_probe"].items():
            _assert_close(float(it.score_mean[name]), value, f"multi[{j}].{name}")


def test_reading_texts_and_completions():
    out = {
        "text_reps": {k: v.data for k, v in read_texts(MODEL, TOK, EVAL_POS[:3], ["text_mean", "text_last",
                                                                                 "through_text_last_k"], last_k=2,
                                                       cache=False).items()},
        "text_reps_user": read_texts(MODEL, TOK, EVAL_NEG[:2], role="user", system="x y z", cache=False).data,
        "text_reps_assistant": read_texts(MODEL, TOK, EVAL_NEG[:2], role="assistant", prompt="say", cache=False).data,
        "completion_reps": read_texts(MODEL, TOK, ["pp x", "nn y"], role="assistant", prompt=["tell me", "why z"],
                                      system="z", cache=False).data,
    }
    stored = _stored("extract_reps_and_token_scores")
    old_names = {"text_mean": "assistant_all_mean", "text_last": "assistant_last",
                 "through_text_last_k": "sequence_last_k_mean"}
    expected = {k: v for k, v in stored.items() if k != "token_scores"}
    expected["text_reps"] = {new: stored["text_reps"][old] for new, old in old_names.items()}
    _assert_close(json.loads(json.dumps(_jsonable(out))), expected, "reading")


def test_scoring_every_token_of_a_text(probes):
    probe = probes["texts"][0]
    every = read_texts(MODEL, TOK, ["pp nx"], Readout("input", "tokens"), cache=False)
    scores = probe.project(every)[:, [1, 3]]
    stored = _stored("extract_reps_and_token_scores")["token_scores"]
    assert every.positions.tolist() == list(range(len(stored["token_ids"])))
    _assert_close(json.loads(json.dumps(_jsonable(scores))), stored["scores_per_layer"], "token_scores")
