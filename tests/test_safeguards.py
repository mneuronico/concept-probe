"""Model-free tests for statistics that respect the design, z references by context, and safeguards
against silent wrong results."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from concept_probe import multi_probe_score_prompts
from concept_probe import probe as probe_mod
from concept_probe.config import ConceptSpec, resolve_config
from concept_probe.logger import JsonlLogger
from concept_probe.probe import ConceptProbe
from concept_probe.utils import deep_merge, json_dump
from fakes import CharChatTok, CharModel


# ------------------------------------------------------------- paired eval items
def _generate(model, tokenizer, system, prompts, *args, **kwargs):
    out = []
    for p in prompts:
        text = ("pp" if system == "POS" else "nn") + ("x" * (len(p) % 3)) + "$"
        ids = probe_mod.apply_chat(tokenizer, [{"role": "system", "content": system}, {"role": "user", "content": p}],
                                   add_generation_prompt=True)[0]
        out.append((torch.cat([ids, torch.tensor([ord(c) for c in text])]), int(ids.shape[-1])))
    return out


def _train(tmp_path, monkeypatch, overrides=None, concept=None, generate=_generate):
    monkeypatch.setattr(probe_mod, "_generate_pairs", generate)
    cfg = resolve_config(deep_merge({
        "prompts": {"train_questions": ["q1", "q2"], "eval_questions": [f"e{i}" + "+" * i for i in range(8)]},
        "evaluation": {"n_permutations": 20, "n_bootstrap": 20},
        "plots": {"enabled": False}, "output": {"console": False, "pretty_json_log": False},
    }, overrides or {}))
    concept = concept or ConceptSpec(name="c", pos_label="p", neg_label="n", pos_system="POS", neg_system="NEG",
                                     eval_pos_texts=["ppp", "ppx", "px"], eval_neg_texts=["nnn", "nnx", "nx"])
    bundle = SimpleNamespace(model_id="fake", tokenizer=CharChatTok(), model=CharModel(), num_layers=2)
    probe = ConceptProbe(model_bundle=bundle, config=cfg, concept=concept, run_dir=str(tmp_path),
                         logger=JsonlLogger(str(tmp_path / "log.jsonl"))).train()
    return probe, json.loads((tmp_path / "metrics.json").read_text()), np.load(tmp_path / "tensors.npz")


def test_generated_eval_is_split_by_question(tmp_path, monkeypatch):
    _, metrics, t = _train(tmp_path, monkeypatch)
    assert metrics["eval_paired"] and metrics["selection"] == "split"
    np.testing.assert_array_equal(t["eval_selection_pos_idx"], t["eval_selection_neg_idx"])
    assert t["eval_question_idx"].tolist() == list(range(8))


def test_an_empty_completion_drops_the_whole_question(tmp_path, monkeypatch):
    def generate(model, tokenizer, system, prompts, *args, **kwargs):
        out = _generate(model, tokenizer, system, prompts)
        if system == "NEG" and len(prompts) > 3:  # eval question 3 gets an empty negative completion
            ids, plen = out[3]
            out[3] = (torch.cat([ids[:plen], torch.tensor([ord("$")])]), plen)
        return out

    _, metrics, t = _train(tmp_path, monkeypatch, generate=generate)
    assert metrics["dropped_items"] == {"eval_neg": [3], "eval_pos": [3]}
    assert t["eval_question_idx"].tolist() == [0, 1, 2, 4, 5, 6, 7]


def test_paired_statistics():
    rng = np.random.default_rng(0)
    q = rng.normal(0, 5, (30, 2))  # large question effects shared by both poles
    pos, neg = q + 0.3 + rng.normal(0, 0.1, (30, 2)), q - 0.3 + rng.normal(0, 0.1, (30, 2))
    from scipy.stats import ttest_rel

    np.testing.assert_allclose(probe_mod._welch_p_columns(pos, neg, paired=True), ttest_rel(pos, neg, axis=0).pvalue)
    # resampling by pair keeps the question effect out of the interval's width
    lo_pair, hi_pair = probe_mod._bootstrap_d_ci(pos[:, 0], neg[:, 0], 200, np.random.default_rng(1), paired=True)
    lo_ind, hi_ind = probe_mod._bootstrap_d_ci(pos[:, 0], neg[:, 0], 200, np.random.default_rng(1))
    assert hi_pair - lo_pair < hi_ind - lo_ind
    _, p = probe_mod._max_d_permutation_p(pos, neg, 99, np.random.default_rng(2), paired=True)
    assert p < 0.05  # the paired difference is clear even though d (between poles) is small


# ------------------------------------------------------------- z references by context
def test_transfer_references_are_kept_and_used_for_raw_texts(tmp_path, monkeypatch):
    probe, metrics, t = _train(tmp_path, monkeypatch)
    assert metrics["reference_context"]["kind"] == "generated"
    assert metrics["evaluation"]["sets"]["texts"]["context"]["role"] == "raw"
    loaded = ConceptProbe.load(str(tmp_path))
    assert [c.source for c in loaded.calibrations] == ["evaluation", "texts"]
    raw = probe_mod._text_context("raw", None, None, "assistant_all_mean")
    assert loaded.calibration_for(raw).source == "texts"
    assert loaded.calibration_for(probe_mod._text_context("user", "S", None, "assistant_all_mean")) is None


def test_score_texts_has_no_z_without_references_from_the_same_reading(tmp_path, monkeypatch):
    probe, _, _ = _train(tmp_path, monkeypatch)
    probe.model_bundle.model = CharModel()
    probe.calibrations = [c for c in probe.calibrations if c.source == "evaluation"]  # generated completions only
    scores = probe.score_texts(["ppx"], save_html=False)
    assert scores[0].token_z["c"] is None and scores[0].score_mean_z["c"] is None
    assert "no calibration for items read as raw text" in scores.z_missing["c"]
    events = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any("no calibration for items read as raw text" in e.get("message", "") for e in events)


# ------------------------------------------------------------- inputs and configs
def test_null_system_prompts_mean_no_system_prompt():
    c = ConceptSpec.from_config({"concept": {"name": "c", "pos_label": "p", "neg_label": "n",
                                             "pos_system": "You are a pirate.", "neg_system": None}})
    assert c.neg_system == ""
    with pytest.raises(ValueError, match="null"):
        ConceptSpec.from_config({"concept": {"name": None}})


def test_questions_with_their_own_system_message_raise(tmp_path, monkeypatch):
    q = [{"role": "system", "content": "tutor"}, {"role": "user", "content": "q1"}]
    with pytest.raises(ValueError, match="own system message"):
        _train(tmp_path, monkeypatch, {"prompts": {"train_questions": [q, "q2"]}})
    assert not (tmp_path / "config.json").exists()


def test_the_random_control_is_one_direction_at_every_layer():
    from concept_probe import Add

    vectors = np.array([[3.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32)
    control = Add(vectors, alpha={0: 1.0, 1: 0.5, 2: 2.0}).random_like([0, 3])
    r = np.stack([v for _, v in control._entries])
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), [3, 2, 1], rtol=1e-6)
    unit = r / np.linalg.norm(r, axis=1, keepdims=True)
    np.testing.assert_allclose(unit[0], unit[1], atol=1e-6)
    np.testing.assert_allclose(unit[0], unit[2], atol=1e-6)
    assert control.alpha_per_layer == {0: 1.0, 1: 0.5, 2: 2.0} and control.control_seed == [0, 3]
    again = Add(vectors * 5).random_like([0, 3])  # the same seed: the same direction, whatever the lengths
    np.testing.assert_allclose(again._entries[0][1] / 15.0, unit[0], atol=1e-6)


# ------------------------------------------------------------- scoring outputs
def _scoring_probe(tmp_path, monkeypatch, name="c", config=None):

    bundle = SimpleNamespace(model_id="fake", tokenizer=CharChatTok(), model=CharModel(), num_layers=2)
    probe = ConceptProbe(model_bundle=bundle, config=resolve_config(config or {"output": {"console": False}}),
                         concept=ConceptSpec(name=name, pos_label="p", neg_label="n", pos_system="a", neg_system="b"),
                         run_dir=str(tmp_path / name), logger=JsonlLogger(str(tmp_path / name / "log.jsonl")))
    probe.directions = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    probe.steer_directions = probe.directions
    probe.layer = 0
    return probe


def test_repeated_alphas_get_their_own_files(tmp_path, monkeypatch):
    scores = _scoring_probe(tmp_path, monkeypatch).score_prompts(["hi"], alphas=[0.0, 0.0, 0.0], save_html=False)
    assert len(scores) == 3 and scores.column("alpha_index") == [0, 1, 2]


def test_a_reused_batch_directory_raises_unless_overwriting(tmp_path, monkeypatch):
    probe = _scoring_probe(tmp_path, monkeypatch)
    probe.score_prompts(["hi"], batch_subdir="b", save_html=False, save=True)
    with pytest.raises(FileExistsError):
        probe.score_prompts(["hi"], batch_subdir="b", save_html=False, save=True)
    old = os.path.join(probe.run_dir, "scores", "b", "coherence_rating.json")
    open(old, "w").write("[]")
    probe.score_prompts(["hi"], batch_subdir="b", save_html=False, overwrite=True, save=True)
    assert not os.path.exists(old)


def test_a_steering_probe_outside_the_list_is_checked_against_the_model(tmp_path, monkeypatch):
    # explicit dtypes: with the default "auto", the config alone does not say which dtype was used
    a = _scoring_probe(tmp_path, monkeypatch, "a", config={"output": {"console": False},
                                                           "model": {"dtype": "bfloat16"}})
    other = _scoring_probe(tmp_path, monkeypatch, "other", config={"output": {"console": False},
                                                                   "model": {"dtype": "float32"}})
    other.steer_sigma = 1.0
    with pytest.raises(ValueError, match="dtype"):
        multi_probe_score_prompts([a], ["hi"], alphas=[1.0], steer_probe=other, output_root=str(tmp_path / "m"))


# ------------------------------------------------------------- files
def test_json_takes_sets_numpy_bools_and_arrays(tmp_path):
    json_dump(str(tmp_path / "x.json"), {"expected": {"b", "a"}, "correct": np.bool_(True), "v": np.arange(2)})
    assert json.loads((tmp_path / "x.json").read_text()) == {"correct": True, "expected": ["a", "b"], "v": [0, 1]}
