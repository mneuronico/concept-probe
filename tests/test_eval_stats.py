"""Model-free tests for eval analysis statistics."""
import numpy as np

from concept_probe.eval_system import (
    _compute_coherence,
    _compute_stats,
    _mean_completion_score,
    simple_equality_evaluator,
)


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
    per_sample = [{"correct": True, "npz_path": "x.npz"}]  # no alpha key
    counts, accuracy = _compute_coherence(per_sample, [0.0], ratings={})
    assert all(c == [0] for c in counts.values())


def test_mean_completion_score_empty_span_is_nan(tmp_path):
    path = tmp_path / "a.npz"
    np.savez(path, scores_agg=np.array([1.0, 2.0, 3.0]), prompt_len=np.array([3]))
    assert np.isnan(_mean_completion_score(str(path)))

    path2 = tmp_path / "b.npz"
    np.savez(path2, scores_agg=np.array([1.0, 2.0, 3.0]), prompt_len=np.array([1]))
    assert _mean_completion_score(str(path2)) == 2.5


def test_simple_equality_evaluator_markers():
    item = {"expected": "42"}
    assert simple_equality_evaluator("the answer is: 42", item, marker=":")["correct"] is True
    assert simple_equality_evaluator("42 :rest", item, marker=":", marker_position="before")["correct"] is True
    assert simple_equality_evaluator("no marker here", item, marker=":")["correct"] is False
    assert simple_equality_evaluator("b", {"expected": ["a", "b"]})["correct"] is True
    assert simple_equality_evaluator("anything", {})["correct"] is None
