"""Evaluating how a probe reads (layer 6b): named eval sets, the primary set read without the
pole's system prompt (decision D2), pairs, AUROC, accuracy, random-direction nulls, shuffled labels,
paired comparisons between scorers and probes read on each other's sets."""
import json
import os

import numpy as np
import pytest

from concept_probe import ConceptProbe, ConceptSpec, ProbeWorkspace
from concept_probe.evaluation import EvalSet, cross_reading, evaluate_reading, label_shuffle_control
from concept_probe.stats import auroc, compare_scorers, random_directions_null
from concept_probe.utils import deep_merge
from concept_probe.visuals import plot_sweep
from fakes import CharChatTok
from tiny_model import TinyCausalLM

from test_reference_outputs import (EVAL_NEG, EVAL_POS, EVAL_QUESTIONS, QUESTIONS, TRAIN_NEG, TRAIN_POS)


def _blobs(n=40, layers=4, dim=400, gap=1.0, seed=0, paired=False):
    # Many dimensions, as in a model: a random direction then barely aligns with the concept's (in 10
    # dimensions it would, and the random-direction null would rightly come out high).
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, layers, dim))
    neg = (pos + rng.normal(scale=0.3, size=pos.shape)) if paired else rng.normal(size=(n, layers, dim))
    pos[:, 2, 0] += gap
    return pos, neg


def _directions(layers=4, dim=400):
    d = np.zeros((layers, dim))
    d[:, 0] = 1.0
    return d


# ------------------------------------------------------------------------------- evaluate_reading
def test_the_primary_set_chooses_the_layer_and_the_others_are_measured_there():
    pos, neg = _blobs(gap=2.0)
    other_pos, other_neg = _blobs(gap=0.5, seed=3)
    ev = evaluate_reading(_directions(), [EvalSet("main", pos, neg), EvalSet("other", other_pos, other_neg)],
                          n_bootstrap=50, n_permutations=50)
    assert ev.layer == 2 and ev.selection == "split" and ev.primary == "main"
    main, other = ev.sets["main"], ev.sets["other"]
    assert main.measured_on == "the held-out part" and other.measured_on == "all items"
    assert main.d > 1 and 0 < main.auroc <= 1 and main.accuracy > 0.6 and main.null_d95 < 1
    assert other.d == pytest.approx(other.d_curve[2]) and other.accuracy is None  # no threshold in another context
    assert ev.threshold["layer"] == 2 and len(ev.selection_idx[0]) == 20


def test_paired_items_stay_together_and_get_paired_statistics():
    pos, neg = _blobs(gap=0.4, paired=True)
    paired = evaluate_reading(_directions(), [EvalSet("main", pos, neg, paired=True)], layer=2, n_bootstrap=0,
                              n_permutations=0)
    unpaired = evaluate_reading(_directions(), [EvalSet("main", pos, neg)], layer=2, n_bootstrap=0, n_permutations=0)
    assert paired.sets["main"].p < unpaired.sets["main"].p  # what each pair has in common cancels out
    split = evaluate_reading(_directions(), [EvalSet("main", pos, neg, paired=True)], n_bootstrap=0, n_permutations=0)
    np.testing.assert_array_equal(split.selection_idx[0], split.selection_idx[1])
    with pytest.raises(ValueError, match="paired"):
        EvalSet("x", pos, neg[:-1], paired=True)


def test_a_fixed_layer_uses_every_item_and_a_missing_primary_raises():
    pos, neg = _blobs()
    ev = evaluate_reading(_directions(), [EvalSet("main", pos, neg)], layer=1, n_bootstrap=0, n_permutations=0)
    assert ev.selection == "fixed_layer" and ev.sweep_max_d_perm_p is None and len(ev.report_idx[0]) == len(pos)
    with pytest.raises(ValueError, match="no eval set named"):
        evaluate_reading(_directions(), [EvalSet("main", pos, neg)], primary="nope")


def test_random_directions_give_the_null_of_d():
    pos, neg = _blobs(n=30, gap=3.0)
    null = random_directions_null(pos, neg, n_random=200, seed=1)
    assert null.shape == (4,) and np.all(null < 1.0)  # a random direction barely separates them
    v = _directions()[2]
    real = abs(np.mean(pos[:, 2] @ v) - np.mean(neg[:, 2] @ v))
    assert real > 2  # the concept direction does


def test_shuffled_labels_do_not_train_a_probe():
    pos, neg = _blobs(n=60, gap=3.0)
    out = label_shuffle_control("mean_diff", pos[:30], neg[:30], pos[30:], neg[30:], layer=2, n_shuffles=20)
    assert out["n_shuffles"] == 20 and out["abs_d_p95"] < 1.5


def test_compare_two_scorers_on_the_same_items():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1.5, 1, 80), rng.normal(0, 1, 80)
    noisy_pos, noisy_neg = pos + rng.normal(0, 3, 80), neg + rng.normal(0, 3, 80)
    out = compare_scorers(auroc, pos, neg, noisy_pos, noisy_neg, n_bootstrap=300)
    assert out["difference"] > 0 and out["ci95"][0] > 0


def test_probes_read_on_each_others_sets():
    pos, neg = _blobs(gap=2.0)

    class P:
        def __init__(self, name, directions, layer):
            self.concept = type("C", (), {"name": name})()
            self.directions, self.layer = directions, layer

    first = P("a", _directions(), 2)
    second_dirs = np.zeros((4, 400))
    second_dirs[:, 1] = 1.0
    second = P("b", second_dirs, 2)
    table = cross_reading([first, second], {"a_set": (pos, neg)}, n_random=50)
    assert table["d"]["a"]["a_set"]["d"] > 1 and abs(table["d"]["b"]["a_set"]["d"]) < 1
    assert table["cosine_at_row_probe_layer"]["a"]["a"] == pytest.approx(1.0)
    assert table["cosine_at_row_probe_layer"]["a"]["b"] == pytest.approx(0.0)


def test_the_sweep_plot_draws_the_reported_curve_and_the_null(tmp_path):
    d = np.array([0.5, 1.0, 2.0, 1.5])
    assert plot_sweep(d, np.full(4, 0.01), str(tmp_path / "s.png"), best_layer=2, selection_d=d + 0.2,
                      report_d=d - 0.1, null_d95=np.full(4, 0.4), best_d=1.9, title="t")
    assert (tmp_path / "s.png").stat().st_size > 0


# --------------------------------------------------------------------------------- in training
def _train(tmp_path, overrides=None, **concept):
    ws = ProbeWorkspace(config_overrides=deep_merge({
        "output": {"root_dir": str(tmp_path), "console": False}, "plots": {"enabled": False},
        "prompts": {"train_questions": QUESTIONS, "eval_questions": EVAL_QUESTIONS},
        "training": {"train_max_new_tokens": 10, "gen_batch_size": 4},
        "evaluation": {"eval_max_new_tokens": 10, "n_permutations": 200, "n_bootstrap": 200},
    }, overrides or {}), model=TinyCausalLM(), tokenizer=CharChatTok(), model_id="tiny-test")
    fields = dict(name="c", pos_label="p", neg_label="n", pos_system="pp pp pp", neg_system="nn nn nn",
                  eval_pos_texts=EVAL_POS, eval_neg_texts=EVAL_NEG)
    fields.update(concept)
    probe = ws.train_concept(ConceptSpec(**fields), run_name="r")
    return probe, json.loads(open(os.path.join(probe.run_dir, "metrics.json"), encoding="utf-8").read())


def test_generated_completions_are_judged_without_the_pole_system_prompt(tmp_path):
    probe, metrics = _train(tmp_path)
    sets = metrics["evaluation"]["sets"]
    assert metrics["evaluation"]["primary"] == "eval" and set(sets) == {"eval", "eval_with_system_prompt", "texts"}
    assert sets["eval"]["d"] < sets["eval_with_system_prompt"]["d"]  # the system prompt inflated it
    assert metrics["reference_context"]["kind"] == "generated"


def test_the_old_judgement_is_one_setting_away_and_gives_the_old_number(tmp_path):
    _, metrics = _train(tmp_path, {"evaluation": {"primary_set": "eval_with_system_prompt"}})
    assert metrics["best_d"] == pytest.approx(20.403938831070754, rel=1e-4)  # the number before decision D2


def test_minimal_pairs_of_eval_texts_are_split_and_tested_by_pair(tmp_path):
    texts = dict(pos_system="", neg_system="", train_pos_texts=TRAIN_POS, train_neg_texts=TRAIN_NEG)
    _, paired = _train(tmp_path / "p", {"training": {"train_prompt_mode": "texts"}, "evaluation": {"eval_pairs": True}},
                       **texts)
    assert paired["eval_paired"] is True and paired["evaluation"]["sets"]["eval"]["paired"] is True
    t = np.load(os.path.join(str(tmp_path / "p"), "c", "r", "tensors.npz"))
    np.testing.assert_array_equal(t["eval_selection_pos_idx"], t["eval_selection_neg_idx"])


def test_named_eval_sets_controls_and_new_numbers_are_reported(tmp_path):
    texts = dict(pos_system="", neg_system="", train_pos_texts=TRAIN_POS, train_neg_texts=TRAIN_NEG,
                 eval_sets={"short": {"pos": ["pp", "px", "zp"], "neg": ["nn", "nx", "zn"]}})
    _, metrics = _train(tmp_path, {"training": {"train_prompt_mode": "texts"}}, **texts)
    short = metrics["evaluation"]["sets"]["short"]
    assert short["n_pos"] == 3 and short["d"] is not None and short["null_d95"] is not None
    assert metrics["null_d95_at_best_layer"] < metrics["best_d"]
    assert metrics["best_auroc"] is not None and metrics["best_accuracy"] is not None
    assert metrics["label_shuffle"]["n_shuffles"] == 10
    assert "evaluation.n_label_shuffles" in [c["setting"] for c in metrics["choices"]]
