"""Model-free tests for eval analysis statistics."""
import numpy as np
import pytest

from concept_probe.eval_system import (
    _compute_coherence,
    _compute_stats,
    _per_sample_records,
    simple_equality_evaluator,
)
from concept_probe.scoring import ScoredItem, Scores


def test_score_stats_do_not_require_correct_labels():
    per_sample = [
        {"alpha": 0.0, "score_mean": 1.0},
        {"alpha": 0.0, "score_mean": 3.0},
        {"alpha": 1.0, "score_mean": 5.0},
    ]
    stats, alpha_vals = _compute_stats(per_sample)
    assert alpha_vals == [0.0, 1.0]
    by_alpha = {r["alpha"]: r for r in stats["score_by_alpha"]}
    assert by_alpha[0.0]["mean_score"] == 2.0
    assert by_alpha[0.0]["n"] == 2
    assert by_alpha[1.0]["mean_score"] == 5.0
    # accuracy has no labeled rows
    acc = {r["alpha"]: r for r in stats["accuracy_by_alpha"]}
    assert acc[0.0]["n"] == 0 and np.isnan(acc[0.0]["accuracy"])


def test_stats_tolerate_null_scores():
    per_sample = [
        {"alpha": 0.0, "score_mean": None, "correct": True},
        {"alpha": 0.0, "score_mean": 2.0, "correct": False},
    ]
    stats, _ = _compute_stats(per_sample)
    assert stats["score_by_alpha"][0]["mean_score"] == 2.0
    assert stats["accuracy_by_alpha"][0]["accuracy"] == 0.5


def test_compute_coherence_skips_rows_without_alpha():
    per_sample = [{"correct": True, "scores_index": 0}]  # no alpha key
    counts, accuracy = _compute_coherence(per_sample, [0.0], ratings={})
    assert all(c == [0] for c in counts.values())


def test_coherence_ratings_are_matched_by_the_generation_index():
    per_sample = [{"alpha": 0.0, "correct": True, "scores_index": 0},
                  {"alpha": 0.0, "correct": False, "scores_index": 1}]
    counts, accuracy = _compute_coherence(per_sample, [0.0], ratings={0: "COHERENT", 1: "NONSENSE"})
    assert counts["COHERENT"] == [1] and counts["NONSENSE"] == [1]
    assert accuracy["COHERENT"] == [1.0] and accuracy["NONSENSE"] == [0.0]


def _item(index, alpha, text, mean):
    return ScoredItem(index=index, text=text, token_ids=np.zeros(3, np.int64), tokens=["a", "b", "c"], span=(1, 3),
                      layers={"c": [0]}, scores={"c": np.zeros((3, 1), np.float32)},
                      token_scores={"c": np.zeros(3, np.float32)}, token_z={"c": None}, score_mean={"c": mean},
                      score_mean_z={"c": None}, info={"alpha": alpha, "alpha_unit": "raw", "prompt": f"q{index}"})


def test_records_come_from_the_scores_in_memory():
    scores = Scores("generations", ["c"], [_item(0, 0.0, "42", 1.5), _item(0, 2.0, "41", None), _item(1, 0.0, "7", 0.5)],
                    {"generation": {"greedy": True}})
    items = [{"prompt": "q0", "expected": "42"}, {"prompt": "q1", "expected": "7"}]
    rows = _per_sample_records(scores, items, lambda c, item: simple_equality_evaluator(c, item), None,
                               expected_key="expected", include_item=False, multi=False, show_progress=False)
    assert [r["item_index"] for r in rows] == [0, 0, 1] and [r["scores_index"] for r in rows] == [0, 1, 2]
    assert [r["correct"] for r in rows] == [True, False, True]
    assert rows[0]["score_mean"] == 1.5 and np.isnan(rows[1]["score_mean"])  # an empty completion has no score
    assert rows[1]["alpha"] == 2.0 and rows[0]["generation"] == {"greedy": True}


def test_simple_equality_evaluator_markers():
    item = {"expected": "42"}
    assert simple_equality_evaluator("the answer is: 42", item, marker=":")["correct"] is True
    assert simple_equality_evaluator("42 :rest", item, marker=":", marker_position="before")["correct"] is True
    assert simple_equality_evaluator("no marker here", item, marker=":")["correct"] is False
    assert simple_equality_evaluator("b", {"expected": ["a", "b"]})["correct"] is True
    assert simple_equality_evaluator("anything", {})["correct"] is None


def test_accuracy_intervals_are_wilson():
    per_sample = [{"alpha": 0.0, "correct": True} for _ in range(10)]
    acc = _compute_stats(per_sample)[0]["accuracy_by_alpha"][0]
    assert acc["accuracy"] == 1.0
    assert 0.6 < acc["ci_low"] < 0.75 and acc["ci_high"] == pytest.approx(1.0)  # normal interval: [1, 1]


def _rows_with_alphas(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for item in range(30):
        for alpha in (0.0, 4.0):
            score = rng.normal() + (1.0 if alpha else 0.0)
            rows.append({"item_index": item, "alpha": alpha, "score_mean": score,
                         "correct": bool(score + rng.normal(scale=0.5) > 0.5)})
    return rows


def test_correct_and_incorrect_are_never_pooled_across_alphas():
    stats, _ = _compute_stats(_rows_with_alphas())
    assert "correct_vs_incorrect" not in stats
    assert [r["alpha"] for r in stats["correct_vs_incorrect_by_alpha"]] == [0.0, 4.0]


def test_correctness_gee_adjusts_for_alpha_and_clusters_by_item():
    pytest.importorskip("statsmodels")
    gee = _compute_stats(_rows_with_alphas())[0]["correctness_gee"]
    assert gee["n_rows"] == 60 and gee["n_items"] == 30
    assert gee["coef_per_sd"] > 0 and gee["ci95"][0] <= gee["coef_per_sd"] <= gee["ci95"][1]
    assert 0 <= gee["p_value"] <= 1 and gee["note"] is None


def test_items_are_identified_by_their_prompt_across_batches():
    from concept_probe.eval_system import item_key

    # the same prompt in two batches is one item; the same index with different prompts is two
    assert item_key({"prompt": "2+2?", "item_index": 0, "batch_name": "a"}) == item_key(
        {"prompt": "2+2?", "item_index": 3, "batch_name": "b"})
    assert item_key({"prompt": "2+2?", "item_index": 0}) != item_key({"prompt": "3+3?", "item_index": 0})
    # without a prompt, the batch qualifies the index
    assert item_key({"item_index": 0, "batch_name": "a"}) != item_key({"item_index": 0, "batch_name": "b"})


def test_gee_clusters_rows_by_item_identity():
    pytest.importorskip("statsmodels")
    rows = _rows_with_alphas()
    # two batches whose items share indices but not prompts: 60 distinct items
    for batch in ("a", "b"):
        for r in _rows_with_alphas(seed=1 if batch == "b" else 0):
            rows.append(dict(r, prompt=f"{batch}:{r['item_index']}", batch_name=batch))
    rows = [r for r in rows if "prompt" in r]
    assert _compute_stats(rows)[0]["correctness_gee"]["n_items"] == 60
