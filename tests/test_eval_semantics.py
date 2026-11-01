"""Model-free tests of how eval runs judge answers, what they record, and how batches are pooled."""
import json
import os

import numpy as np
import pytest

from concept_probe import coherence
from concept_probe import eval_system as E
from concept_probe.reporting import _build_rows, _compute_summary, _score_kind
from test_inputs import _probe


# ------------------------------------------------------------- default evaluator
def test_the_marker_takes_the_line_next_to_it():
    item = {"expected": "42"}
    out = E.simple_equality_evaluator("Work...\nANSWER: 42\nBecause 6*7.", item, marker="ANSWER:")
    assert out["extracted"] == "42" and out["correct"] is True
    out = E.simple_equality_evaluator("I think\n42\nis the answer", item, marker="is the answer",
                                      marker_position="before")
    assert out["extracted"] == "42" and out["correct"] is True
    out = E.simple_equality_evaluator("no marker here: 42", item, marker="ANSWER:")
    assert out["correct"] is False and out["extracted"] is None


def test_answers_match_as_numbers_ignoring_a_trailing_period():
    assert E.simple_equality_evaluator(" 4. ", {"expected": 4.0})["correct"] is True
    assert E.simple_equality_evaluator("4.5", {"expected": 4})["correct"] is False
    assert E.simple_equality_evaluator("Paris", {"expected": "paris"})["correct"] is False
    assert E.simple_equality_evaluator("Paris", {"expected": "paris"}, case_sensitive=False)["correct"] is True
    assert E.simple_equality_evaluator("b", {"expected": ["a", "b"]})["correct"] is True


def test_an_unknown_marker_position_raises():
    with pytest.raises(ValueError, match="marker_position"):
        E.simple_equality_evaluator("x", {"expected": "x"}, marker="A:", marker_position="after_last")


def test_items_without_an_expected_value_raise_with_the_default_evaluator(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Missing 'expected' for 1"):
        E.run_scored_eval(probe, [{"prompt": "a", "expected": "pp"}, {"prompt": "b", "expected": None}],
                          analyze=False, save_html=False)


# ------------------------------------------------------------- per-sample records
def test_per_sample_records_keep_z_scores_and_generation_settings(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    run = E.run_scored_eval(probe, [{"prompt": "a", "expected": "pp"}], analyze=False, save_html=False)
    row = run.per_sample[0]
    assert row["correct"] is True
    assert row["score_mean_z"] == pytest.approx(2.0)
    assert row["generation"]["greedy"] is True


def test_an_evaluator_cannot_overwrite_the_scoring_fields(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="score_mean"):
        E.run_scored_eval(probe, [{"prompt": "a"}], evaluator=lambda c, item: {"correct": True, "score_mean": 0.0},
                          analyze=False, save_html=False)


def test_an_empty_completion_has_no_score_in_any_probe():
    assert np.isnan(E._score_mean_by_probe_from_result({"score_mean_by_probe": {"a": None, "b": 1.0}})["a"])


# ------------------------------------------------------------- report
def _item(correct, raw, z, alpha=0.0):
    return {"correct": correct, "alpha": alpha, "prompt": f"q{raw}",
            "score_mean_by_probe": {"a": raw}, "score_mean_z_by_probe": {"a": z}}


def test_the_report_uses_z_scores_when_every_scored_item_has_them():
    items = [_item(True, 1.0, 2.0), _item(False, -1.0, -2.0)]
    assert _score_kind(items, ["a"]) == "z"
    assert _score_kind(items + [{**_item(True, 0.5, None)}], ["a"]) == "raw"
    # an empty completion (no raw score either) does not switch the report to raw
    assert _score_kind(items + [_item(True, None, None)], ["a"]) == "z"


def test_the_overall_mean_includes_unlabelled_rows():
    items = [_item(True, 1.0, 2.0), _item(None, 3.0, 6.0)]
    (summary,) = _compute_summary(_build_rows(items, ["a"], "z", labelled_only=False), ["a"]).values()
    assert summary["overall"]["a"] == pytest.approx(4.0)
    assert summary["correct"]["a"] == pytest.approx(2.0)


# ------------------------------------------------------------- pooling batches
def _write_batch(root, name, rows):
    path = root / name / "analysis"
    path.mkdir(parents=True)
    (path / "per_sample.json").write_text(json.dumps({"items": rows}))


def _rows(n, offset=0):
    return [{"item_index": i, "prompt": f"q{i + offset}", "alpha": a, "alpha_unit": "raw", "alpha_value": a,
             "correct": bool((i + int(a)) % 2), "score_mean": float(i) - a}
            for i in range(n) for a in (0.0, 1.0)]


def test_aggregation_finds_batches_by_their_analysis_folder(tmp_path):
    _write_batch(tmp_path, "first_run", _rows(6))
    _write_batch(tmp_path, "batch_0002", _rows(6, offset=100))
    out = E.aggregate_eval_batches(str(tmp_path))
    assert out is not None
    pooled = json.loads((tmp_path / "analysis_all" / "per_sample.json").read_text())["items"]
    assert {r["batch_name"] for r in pooled} == {"first_run", "batch_0002"}


def test_aggregation_warns_when_an_item_repeats_at_an_alpha(tmp_path, capsys):
    _write_batch(tmp_path, "a", _rows(6))
    _write_batch(tmp_path, "b", _rows(6))  # the same prompts again
    E.aggregate_eval_batches(str(tmp_path))
    assert "12 (item, alpha) pairs appear in more than one row" in capsys.readouterr().out


# ------------------------------------------------------------- coherence key
def test_an_explicit_env_file_wins_over_the_environment(tmp_path, monkeypatch):
    probe = _probe(tmp_path, monkeypatch)
    rec = probe.score_prompts(["hi"], save_html=False)[0]
    env = tmp_path / "keys.env"
    env.write_text("GROQ_API_KEY=from-file\n")
    monkeypatch.setenv("GROQ_API_KEY", "from-environment")
    used = []

    def fake_call(api_key, model, messages):
        used.append(api_key)
        raise RuntimeError("stop")

    monkeypatch.setattr(coherence, "_call_groq", fake_call)
    with pytest.raises(RuntimeError, match="stop"):
        coherence.rate_batch_coherence(os.path.dirname(rec["npz_path"]), env_path=str(env))
    assert used == ["from-file"]
