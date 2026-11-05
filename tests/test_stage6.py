"""Datasets and extras (stage 6 of docs/DESIGN.md): erasing confounds, learning curves, confound
checks, red teaming, several classes, regression, circles, and auditing a dataset."""
import json
import os

import numpy as np
import pytest

from concept_probe import (ConceptProbe, EvalSet, Eraser, audit_dataset, centroid_subspace, compare_poles,
                           confound_check, distinctive_words, diversity, fit_circle, fit_classes, fit_directions,
                           fit_regression, formulaicity, learning_curve, read_texts, red_team, surface_baseline)
from fakes import CharChatTok
from test_reference_outputs import EVAL_NEG, EVAL_POS, TRAIN_NEG, TRAIN_POS
from tiny_model import TinyCausalLM

L, D = 3, 60


def _confounded(n=300, seed=0):
    rng = np.random.default_rng(seed)
    concept = rng.integers(0, 2, n)
    confound = (rng.random(n) < np.where(concept == 1, 0.85, 0.15)).astype(int)  # goes with the concept
    X = rng.normal(size=(n, L, D))
    X[:, :, 0] += 2.0 * concept[:, None]
    X[:, :, 1] += 3.0 * confound[:, None]
    return X, concept, confound


# ------------------------------------------------------------------------------------ erasing
@pytest.mark.parametrize("method", ["leace", "project"])
def test_erasing_a_confound_leaves_no_trace_of_it_and_reads_raw_reps(method):
    X, concept, confound = _confounded()
    eraser = Eraser.fit(X, confound, method=method)
    erased = eraser.apply(X)
    np.testing.assert_allclose(erased[confound == 1].mean(0), erased[confound == 0].mean(0), atol=1e-8)
    plain = fit_directions("mean_diff", X[concept == 1], X[concept == 0])
    clean = fit_directions("mean_diff", X[concept == 1], X[concept == 0], erase=eraser)
    # the probe's projections no longer differ between the confound's groups...
    for v, expect_gap in ((plain, True), (clean, False)):
        gap = (X[confound == 1, 0] @ v[0]).mean() - (X[confound == 0, 0] @ v[0]).mean()
        assert bool(abs(gap) > 0.5) == expect_gap
    # ...and reading raw reps with the stored direction is reading the erased ones
    on_erased = fit_directions("mean_diff", erased[concept == 1], erased[concept == 0])
    assert np.corrcoef(X[:, 0] @ clean[0], erased[:, 0] @ on_erased[0])[0, 1] == pytest.approx(1.0)
    assert eraser.describe()["dimensions_erased"] == {0: 1, 1: 1, 2: 1}


def test_leace_leaves_no_linear_classifier_above_chance():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X, _, confound = _confounded(n=400)
    erased = Eraser.fit(X, confound).apply(X)
    acc = LogisticRegression(max_iter=2000).fit(erased[:, 1], confound).score(erased[:, 1], confound)
    assert acc < 0.6 and LogisticRegression(max_iter=2000).fit(X[:, 1], confound).score(X[:, 1], confound) > 0.85


def test_a_probe_can_be_fitted_without_a_confound():
    X, concept, confound = _confounded()
    probe = ConceptProbe.fit(X[concept == 1], X[concept == 0], layer=0, identity="synthetic", save=False,
                             erase=Eraser.fit(X, confound))
    assert probe.estimator["erased"]["method"] == "leace"
    gap = (X[confound == 1, 0] @ probe.directions[0]).mean() - (X[confound == 0, 0] @ probe.directions[0]).mean()
    assert abs(gap) < 1e-6
    assert abs((X[confound == 1, 0] @ probe.steer_directions[0]).mean()
               - (X[confound == 0, 0] @ probe.steer_directions[0]).mean()) < 1e-6


# ----------------------------------------------------------------------------- learning curves
def test_a_learning_curve_grows_with_items_and_has_its_null():
    rng = np.random.default_rng(0)
    P, N = rng.normal(size=(120, L, 200)), rng.normal(size=(120, L, 200))
    P[:, 1, 0] += 1.5
    EP, EN = rng.normal(size=(60, L, 200)), rng.normal(size=(60, L, 200))
    EP[:, 1, 0] += 1.5
    out = learning_curve(P, N, [EvalSet("eval", EP, EN), EvalSet("other", EP[:20], EN[:20])], [4, 64],
                         n_repeats=8, n_null=8, paired=True)
    probe, null = out["summary"]["probe"], out["summary"]["null"]
    assert probe[64]["d"] > probe[4]["d"] and probe[64]["d"] > null[64]["d_p95"]
    assert probe[64]["cos_full"] > probe[4]["cos_full"] and probe[64]["layer_mode_share"] > 0.8
    assert len(out["rows"]) == 2 * (8 + 8) and "d_other" in out["rows"][0]
    with pytest.raises(ValueError, match="larger than"):
        learning_curve(P, N, [EvalSet("eval", EP, EN)], [500])


# ------------------------------------------------------------------------------------ confounds
def test_a_confound_check_tells_what_the_probe_reads():
    rng = np.random.default_rng(1)
    dirs = np.zeros((L, D))
    dirs[:, 0] = 1.0
    A, B = rng.normal(size=(40, L, D)), rng.normal(size=(40, L, D))
    A[:, :, 5] += 2.0  # a confound off the probe's direction
    read = A.copy()
    read[:, :, 0] += 1.5  # a confound the probe reads
    out = confound_check(dirs, 1, {"other": (A, B), "read": (read, B)})
    assert out["other"]["reads_confound"] is False and abs(out["other"]["cosine_at_layer"]) < 0.2
    assert out["read"]["reads_confound"] is True and out["read"]["probe_d"] > out["read"]["null_d95"]
    assert out["other"]["confound_direction_d"] > 1  # the confound itself is easy to find


# ---------------------------------------------------------------------------------- red teaming
def test_red_team_reports_accuracy_and_neutral_shift():
    model, tok = TinyCausalLM(), CharChatTok()
    probe = ConceptProbe.fit(read_texts(model, tok, TRAIN_POS, cache=False), read_texts(model, tok, TRAIN_NEG, cache=False),
                             layer=1, model=model, tokenizer=tok, save=False,
                             calibrate=(read_texts(model, tok, EVAL_POS, cache=False),
                                        read_texts(model, tok, EVAL_NEG, cache=False)))
    out = red_team(probe, {"x_heavy": {"pos": ["pxxxx", "xxpxx"], "neg": ["nxxxx", "xxnxx"],
                                       "neutral": [("xxxx", "yyyy"), ("xyxy", "yxyx")]}})
    attack = out["x_heavy"]
    assert attack["pos"]["n"] == 2 and attack["pos"]["rule"] == "calibrated threshold"
    assert 0.0 <= attack["pos"]["accuracy"] <= 1.0 and len(attack["neg"]["z"]) == 2
    assert attack["neutral"]["n"] == 2 and len(attack["neutral"]["shift"]) == 2
    with pytest.raises(ValueError, match="no calibration"):
        red_team(probe, {"a": {"pos": ["p"]}}, role="user")


# ------------------------------------------------------------------------------------- geometry
def _days(n_per, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(7):
        x = rng.normal(size=(n_per, L, D)) * 0.3
        a = 2 * np.pi * k / 7
        x[:, 2, 0] += 3 * np.cos(a)
        x[:, 2, 1] += 3 * np.sin(a)
        out[f"day{k}"] = x
    return out


def test_classes_are_told_apart_where_they_differ():
    fit, test = fit_classes(_days(20, 1)), _days(10, 2)
    acc = fit.accuracy(test)
    assert acc["best_layer"] == 2 and acc["best_accuracy"] > 0.9 and acc["accuracy"][0] < 0.4
    assert acc["confusion"].sum() == 70 and acc["chance"] == pytest.approx(1 / 7)
    assert fit.predict(test["day3"], 2).count("day3") >= 9


def test_a_circle_fits_classes_in_their_cyclic_order():
    circle = fit_circle(_days(20, 1), n_shuffles=100)
    assert circle.r2[2] > 0.95 and circle.r2[2] > circle.r2_null_p95[2] and circle.r2[0] < circle.r2_null_p95[0]
    assert circle.accuracy(_days(10, 2))[2] > 0.9
    sub = centroid_subspace(_days(20, 1), k=2)
    assert sub["basis"].shape == (L, 2, D) and sub["share"][2] > 0.95


def test_a_regression_direction_predicts_a_number():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, L, D))
    y = 2.0 * X[:, 1, 3] + rng.normal(size=200) * 0.1
    reg = fit_regression(X[:150], y[:150], alpha=0.1)
    ev = reg.evaluate(X[150:], y[150:])
    assert ev["best_layer"] == 1 and ev["best_r2"] > 0.9 and ev["r2"][0] < 0.2
    assert abs(reg.directions[1, 3]) > 0.9  # it found the coordinate


# -------------------------------------------------------------------------------------- datasets
POS = ["I love this sunny day", "What a wonderful gift you gave me", "We won the final and I am thrilled",
       "Her smile made the morning bright", "The garden looks beautiful today", "I adore my new job",
       "Such a lovely evening with friends", "The trip was fantastic and fun"]
NEG = ["I hate this rainy day", "What a terrible gift you gave me", "We lost the final and I am crushed",
       "Her frown made the morning dark", "The garden looks awful today", "I despise my new job",
       "Such a dreadful evening with friends", "The trip was horrible and dull"]


def test_compare_poles_finds_a_surface_difference():
    rows = compare_poles([t + "!!" for t in POS], NEG, paired=True)
    marks = next(r for r in rows if r["feature"] == "exclamations")
    assert marks["more_pos"] == 8 and marks["p"] < 0.01 and rows.index(marks) < 3
    assert {r["feature"] for r in rows} >= {"words", "negation", "first_person"}


def test_surface_baselines_and_words():
    pytest.importorskip("sklearn")
    words = surface_baseline(POS, NEG, "words", paired=True, folds=4)
    assert words["accuracy"] >= 0.5 and words["chance"] == 0.5 and words["folds"] == 4
    marked = surface_baseline([t + "!" for t in POS], NEG, "surface", paired=True, folds=4)
    assert marked["accuracy"] == 1.0  # the exclamation gives the pole away
    custom = surface_baseline(POS, NEG, lambda texts: [[t.count("a")] for t in texts], folds=4)
    assert custom["n_features"] == 1
    top = distinctive_words(POS * 3, NEG * 3, top=5)
    assert {w["word"] for w in top["pos"]} & {"love", "wonderful", "won", "thrilled", "adore"}


def test_formulaic_and_varied_texts():
    f = formulaicity(["the old man sat down", "the old man sat up", "a cat ran"], ["0", "1", "2"],
                     jaccard_threshold=0.6)
    assert f["top_openings"][0] == ("the old", 2) and len(f["near_duplicates"]) == 1
    same = diversity(["a b c", "a b c"], embeddings=[[1.0, 0.0], [1.0, 0.0]])
    varied = diversity(["a b c", "d e f"], embeddings=[[1.0, 0.0], [0.0, 1.0]])
    assert same["type_token"] < varied["type_token"]
    assert same["embedding_dispersion"] == pytest.approx(0.0) and varied["embedding_dispersion"] == pytest.approx(1.0)


def test_an_audit_is_written(tmp_path):
    pytest.importorskip("sklearn")
    audit = audit_dataset([t + "!" for t in POS], NEG, paired=True, name="toy", folds=4, save=True,
                          out_dir=str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["audit.json", "audit.md"]
    assert json.load(open(tmp_path / "audit.json", encoding="utf-8"))["baselines"]["surface"]["accuracy"] == 1.0
    assert any("surface classifier" in n for n in audit.notes) and "# Audit: toy" in audit.markdown()
