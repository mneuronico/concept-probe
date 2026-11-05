"""Reference outputs of the existing API, on a tiny deterministic model (tests/tiny_model.py).

These tests pin what ``ProbeWorkspace.train_concept``, ``ConceptProbe.score_texts``,
``ConceptProbe.score_prompts``, ``multi_probe_score_prompts``, ``extract_text_reps``,
``extract_completion_reps`` and ``score_text_tokens`` return and write today, so every stage of the
redesign (docs/DESIGN.md) can show that the existing API still gives the same results. Numbers are
compared up to rounding (decision D8); texts, layer indices and file names exactly.

When a change is meant to alter these outputs (decisions D1 and D2, for example), regenerate the
reference and commit it together with a note of what changed and why:

    CP_UPDATE_REFERENCE=1 python -m pytest tests/test_reference_outputs.py
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from concept_probe import extract_completion_reps, extract_text_reps, multi_probe_score_prompts, score_text_tokens
from concept_probe.config import ConceptSpec
from concept_probe.probe import ProbeWorkspace
from concept_probe.utils import deep_merge
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


def _check(name, value):
    value = json.loads(json.dumps(_jsonable(value)))
    stored = json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.exists() else {}
    if UPDATE:
        stored[name] = value
        REFERENCE.parent.mkdir(exist_ok=True)
        REFERENCE.write_text(json.dumps(stored, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    assert name in stored, f"No reference for {name!r}; run with CP_UPDATE_REFERENCE=1 to create it."
    _assert_close(value, stored[name], name)


def _files(root):
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def _workspace(root, overrides=None):
    # The model is passed in already loaded (entry 4 of LIBRARY_ISSUES).
    return ProbeWorkspace(config_overrides=deep_merge({
        "output": {"root_dir": str(root), "console": False},
        "plots": {"enabled": False},
        "prompts": {"train_questions": QUESTIONS, "eval_questions": EVAL_QUESTIONS},
        "training": {"train_max_new_tokens": 10, "gen_batch_size": 4},
        "evaluation": {"eval_max_new_tokens": 10, "n_permutations": 200, "n_bootstrap": 200},
        "steering": {"steer_max_new_tokens": 10},
    }, overrides or {}), model=TinyCausalLM(), tokenizer=CharChatTok(), model_id="tiny-test")


def _concept(name, **kwargs):
    fields = dict(name=name, pos_label="pos", neg_label="neg", pos_system="pp pp pp", neg_system="nn nn nn",
                  eval_pos_texts=EVAL_POS, eval_neg_texts=EVAL_NEG)
    fields.update(kwargs)
    return ConceptSpec(**fields)


TRAININGS = {
    "texts": ({"training": {"train_prompt_mode": "texts"}},
              dict(pos_system="", neg_system="", train_pos_texts=TRAIN_POS, train_neg_texts=TRAIN_NEG)),
    "texts_var": ({"training": {"train_prompt_mode": "texts", "estimator": "mean_diff_var"}},
                  dict(pos_system="", neg_system="", train_pos_texts=TRAIN_POS, train_neg_texts=TRAIN_NEG)),
    "texts_user": ({"training": {"train_prompt_mode": "texts", "train_text_role": "user",
                                 "train_text_system": "x y z"}},
                   dict(pos_system="", neg_system="", train_pos_texts=TRAIN_POS, train_neg_texts=TRAIN_NEG)),
    "shared": ({"training": {"train_prompt_mode": "shared"}}, {}),
    "opposed": ({"training": {"train_prompt_mode": "opposed"},
                 "prompts": {"train_questions": [], "train_questions_pos": QUESTIONS[:4],
                             "train_questions_neg": QUESTIONS[2:]}}, {}),
}


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference")
    out = {}
    for key, (overrides, concept_fields) in TRAININGS.items():
        ws = _workspace(root / "out", overrides)
        out[key] = ws.train_concept(_concept(key, **concept_fields), run_name="ref")
    out["root"] = root
    return out


def _run_summary(probe):
    run = Path(probe.run_dir)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    tensors = np.load(run / "tensors.npz")
    out = {k: metrics.get(k) for k in (
        "best_layer", "best_d", "best_p", "best_d_ci95", "selection", "best_d_selection", "best_d_all_eval",
        "n_eval_selection", "sweep_max_d", "sweep_max_d_perm_p", "sigma_best_layer", "n_train_pos", "n_train_neg",
        "n_eval_pos", "n_eval_neg", "eval_mode_used", "eval_paired", "reference_context", "eval_readout",
        "dropped_items", "train_text_role",
    )}
    # the normalization of the difference of means, as the reference recorded it before the estimator
    # had its own setting
    out["direction_normalization"] = metrics["estimator"].get("normalization")
    transfer = metrics.get("transfer_eval")
    out["transfer_eval"] = None if transfer is None else {
        k: transfer.get(k) for k in ("n_pos", "n_neg", "d_at_best_layer", "p_at_best_layer", "d_ci95_at_best_layer")
    }
    for key in ("concept_vectors", "sweep_d", "sweep_p", "score_center", "score_scale", "token_center",
                "token_scale", "resid_norm", "eval_pos_scores_mat", "eval_neg_scores_mat", "eval_question_idx",
                "transfer_sweep_d", "train_pos_reps", "train_neg_reps"):
        out[key] = tensors[key]
    out["files"] = _files(run)
    return out


def _npz_arrays(path, keys):
    if not path:
        return {}
    data = np.load(path)
    return {k: data[k] for k in keys if k in data.files}


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("key", list(TRAININGS))
def test_train_concept(probes, key):
    _check(f"train_{key}", _run_summary(probes[key]))


def test_score_texts(probes):
    out = {}
    for key, kwargs in (("texts", {}), ("texts_user", {"role": "user", "system_prompt": "x y z"}),
                        ("shared", {"random_scoring_vector": True}), ("texts", {"reference_layer": 0})):
        texts = TEXTS_TO_SCORE if kwargs.get("role") == "user" else TEXTS_TO_SCORE[:3]
        name = f"{key}_" + "_".join(sorted(kwargs)) if kwargs else key
        records = probes[key].score_texts(texts, batch_subdir=name, **kwargs)
        out[name] = [
            dict({k: r[k] for k in ("input_text", "text_span", "layers", "score_mean", "score_mean_z",
                                    "scores_standardized", "random_scoring_vector_seed", "system_prompt")},
                 **_npz_arrays(r["npz_path"], ("scores_per_layer", "scores_agg", "scores_z_per_layer", "scores_agg_z")))
            for r in records
        ]
        out[name + "_files"] = _files(records[0]["batch_dir"])
    _check("score_texts", out)


PROMPT_KEYS = ("prompt_text", "completion", "prompt_len", "completion_end", "alpha", "alpha_unit", "alpha_value",
               "steer_layers", "steer_layer_scales", "steer_alpha_per_layer", "layers", "score_mean", "score_mean_z",
               "do_steering", "random_scoring_vector_seed", "random_steering_vector_seed", "sampling_seed")


def test_score_prompts(probes):
    calls = {
        "raw": dict(alphas=[0.0, 4.0, -4.0]),
        "sigma_window": dict(alphas=[0.0, 2.0], alpha_unit="sigma", steer_layers="window", steer_window_radius=1),
        "resid_norm_list": dict(alphas=[0.3], alpha_unit="resid_norm", steer_layers=[1, 2], steer_distribute=False),
        "random_steering": dict(alphas=[4.0], random_steering_vector=True),
        "random_scoring": dict(alphas=[0.0], random_scoring_vector=True),
        "sampled": dict(alphas=[0.0, 0.0], greedy=False, seed=3, temperature=0.8),
        "texts_probe": dict(alphas=[0.0]),
    }
    out = {}
    for name, kwargs in calls.items():
        probe = probes["texts" if name == "texts_probe" else "shared"]
        records = probe.score_prompts(QUESTIONS[:3], batch_subdir=name, **kwargs)
        out[name] = [
            dict({k: r[k] for k in PROMPT_KEYS},
                 **_npz_arrays(r["npz_path"], ("scores_per_layer", "scores_agg", "scores_agg_z")))
            for r in records
        ]
        out[name + "_files"] = _files(records[0]["batch_dir"])
    _check("score_prompts", out)


def test_multi_probe_score_prompts(probes):
    run = multi_probe_score_prompts(
        [probes["shared"], probes["opposed"], probes["texts_var"]], QUESTIONS[:3], alphas=[0.0, 4.0],
        steer_probe="shared", output_root=str(probes["root"] / "multi"), run_name="ref", batch_subdir="b",
    )
    out = {"records": [
        dict({k: r[k] for k in ("completion", "prompt_len", "completion_end", "alpha_value", "steer_alpha_per_layer",
                                "steer_layers", "score_mean_by_probe", "score_mean_z_by_probe", "probe_names")},
             **_npz_arrays(r["npz_path"], ("scores_agg", "scores_agg_z")))
        for r in run["results"]
    ], "files": _files(run["run_dir"])}
    _check("multi_probe_score_prompts", out)


def test_extract_reps_and_token_scores(probes):
    bundle = probes["texts"].model_bundle
    model, tok = bundle.model, bundle.tokenizer
    out = {
        "text_reps": extract_text_reps(model, tok, EVAL_POS[:3], pool=["assistant_all_mean", "assistant_last",
                                                                       "sequence_last_k_mean"], last_k=2),
        "text_reps_user": extract_text_reps(model, tok, EVAL_NEG[:2], role="user", system="x y z"),
        "text_reps_assistant": extract_text_reps(model, tok, EVAL_NEG[:2], role="assistant", prompt="say"),
        "completion_reps": extract_completion_reps(model, tok, [("tell me", "pp x"), ("why z", "nn y")], system="z"),
        "token_scores": score_text_tokens(model, tok, probes["texts"].directions, "pp nx", layer_indices=[1, 3]),
    }
    out["token_scores"].pop("offset_mapping")  # character offsets of the fake tokenizer, not the library's
    _check("extract_reps_and_token_scores", out)
