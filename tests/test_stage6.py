"""Datasets and extras (stage 6 of docs/DESIGN.md): erasing confounds, learning curves, confound
checks, red teaming, several classes, regression, circles, and auditing a dataset."""
import json
import os

import numpy as np
import pytest

from concept_probe import (ConceptProbe, EvalSet, Eraser, Target, audit_dataset, compare_poles, confound_check,
                           count_matches,
                           distinctive_words, diversity, fit_directions, formulaicity, learning_curve, read_target,
                           read_texts, red_team, surface_baseline, target_shape)
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


# ------------------------------------------------------------------------------ targets and shapes
def _days(n_per, seed, circle=True):
    """Seven classes; at layer 2 their means lie on a circle in order (or in a scrambled order)."""
    rng = np.random.default_rng(seed)
    where = np.arange(7) if circle else np.array([0, 3, 6, 2, 5, 1, 4])
    # each class also has small features of its own, spread over many dimensions, as in a model
    own = np.random.default_rng(99).normal(size=(7, D)) * 0.25
    own[:, :2] = 0.0
    X, labels = [], []
    for k in range(7):
        x = rng.normal(size=(n_per, L, D)) * 0.3
        x[:, 2, :] += own[k]
        a = 2 * np.pi * where[k] / 7
        x[:, 2, 0] += 3 * np.cos(a)
        x[:, 2, 1] += 3 * np.sin(a)
        X.append(x)
        labels += [k] * n_per
    return np.concatenate(X), np.array(labels)


def test_the_shape_of_a_cycle_is_found_where_it_is_and_only_in_order():
    X, labels = _days(20, 1)
    shape = target_shape(X, Target.cycle(labels, range(7)), n_shuffles=100)
    assert shape.best_layer == 2 and shape.fit[2] > 0.95 and shape.fit[2] > shape.fit_null_p95[2]
    assert shape.similarity[2] > shape.similarity_null_p95[2] and shape.fit[0] < shape.fit_null_p95[0]
    assert shape.coords.shape == (L, 7, 2) and shape.share[2] > 0.75  # the plane keeps most of the spread
    # the same clusters out of order: a circle, but not the days' cycle
    Xs, labels_s = _days(20, 1, circle=False)
    scrambled = target_shape(Xs, Target.cycle(labels_s, range(7)), n_shuffles=100)
    assert scrambled.fit[2] < scrambled.fit_null_p95[2] + 0.05
    # an ellipse fits an affine map but not the shape itself as well
    Xe = X.copy()
    Xe[:, 2, 1] *= 0.3
    assert target_shape(Xe, Target.cycle(labels, range(7)), fit="affine", n_shuffles=0).fit[2] > \
        target_shape(Xe, Target.cycle(labels, range(7)), n_shuffles=0).fit[2] + 0.05


def test_reading_a_cycle_does_not_prove_its_shape_but_reading_through_the_shape_does():
    X, labels = _days(20, 1)
    t = Target.cycle(labels, range(7))
    linear = read_target(X, t, model="linear", n_shuffles=10)
    assert linear.best_score == 1.0 and linear.null_p95[2] >= 0.9  # any arrangement is readable
    through = read_target(X, t, model="shape", n_shuffles=10)
    assert through.score[2] > 0.9 and through.null_p95[2] < 0.7 and through.shuffle == "classes"
    with pytest.raises(ValueError, match="layout"):
        target_shape(X, Target.categories(labels))


def test_categories_numbers_and_orders_are_read_with_their_controls():
    X, labels = _days(20, 1)
    cats = read_target(X, Target.categories(labels), model="centroid", n_shuffles=10)
    assert cats.best_layer == 2 and cats.best_score > 0.9 and cats.null_p95[2] < 0.4
    assert cats.chance == pytest.approx(1 / 7) and cats.shuffle == "items"
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(200, L, D))
    y = 2.0 * Z[:, 1, 3] + rng.normal(size=200) * 0.1
    num = read_target(Z, Target.number(y), alpha=0.1, n_shuffles=10)
    assert num.best_layer == 1 and num.best_score > 0.9 and num.rank_correlation[1] > 0.9 and num.null_p95[1] < 0.2
    assert num.predict(Z[:3]).shape == (3, 1)
    ordered = Target.order(labels, range(7))
    assert ordered.places.shape == (7, 1) and ordered.structured and not Target.categories(labels).structured
    tree = Target.layout(["a", "b", "c", "a"], {"a": [0, 0], "b": [1, 0], "c": [0, 1]})
    assert tree.kind == "layout" and tree.values.tolist() == [[0, 0], [1, 0], [0, 1], [0, 0]]


def test_any_model_with_fit_and_predict_can_read():
    pytest.importorskip("sklearn")
    from sklearn.neighbors import KNeighborsRegressor

    X, labels = _days(20, 1)
    knn = read_target(X, Target.cycle(labels, range(7)), model=KNeighborsRegressor(3), layers=[2], n_shuffles=5)
    assert knn.model == "KNeighborsRegressor" and knn.score[2] > 0.9 and np.isnan(knn.score[0])


# -------------------------------------------------------------------------------------- datasets
POS = ["I love this sunny day", "What a wonderful gift you gave me", "We won the final and I am thrilled",
       "Her smile made the morning bright", "The garden looks beautiful today", "I adore my new job",
       "Such a lovely evening with friends", "The trip was fantastic and fun"]
NEG = ["I hate this rainy day", "What a terrible gift you gave me", "We lost the final and I am crushed",
       "Her frown made the morning dark", "The garden looks awful today", "I despise my new job",
       "Such a dreadful evening with friends", "The trip was horrible and dull"]


def test_compare_poles_finds_a_difference_in_length_or_in_a_check():
    rows = compare_poles([t + " indeed!!" for t in POS], NEG, paired=True,
                         checks={"exclamations": count_matches("!")})
    assert [r["feature"] for r in rows][:1] == ["exclamations"] and rows[0]["std_diff"] == float("inf")
    words = next(r for r in rows if r["feature"] == "words")
    assert words["more_pos"] == 8 and words["p"] < 0.01
    assert {r["feature"] for r in compare_poles(POS, NEG)} == {"chars", "words"}  # nothing else by default
    with pytest.raises(ValueError, match="built-in"):
        compare_poles(POS, NEG, checks={"words": len})


def test_surface_baselines_and_words():
    pytest.importorskip("sklearn")
    words = surface_baseline(POS, NEG, "words", paired=True, folds=4)
    assert words["accuracy"] >= 0.5 and words["chance"] == 0.5 and words["folds"] == 4
    marked = surface_baseline([t + "!" for t in POS], NEG, {"bangs": count_matches("!")}, paired=True, folds=4)
    assert marked["accuracy"] == 1.0  # the exclamation gives the pole away
    longer = surface_baseline([t + " and so on and so forth" for t in POS], NEG, "length", paired=True, folds=4)
    assert longer["accuracy"] == 1.0
    function = surface_baseline(POS, NEG, "words", vocabulary=["the", "a", "my"], paired=True, folds=4)
    assert function["n_features"] == 3
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
                          out_dir=str(tmp_path), checks={"bangs": count_matches("!")})
    assert sorted(os.listdir(tmp_path)) == ["audit.json", "audit.md"]
    saved = json.load(open(tmp_path / "audit.json", encoding="utf-8"))
    assert saved["baselines"]["checks"]["accuracy"] == 1.0 and set(saved["baselines"]) == {"words", "length", "checks"}
    assert any("checks classifier" in n for n in audit.notes) and "# Audit: toy" in audit.markdown()


def test_the_shape_can_be_drawn(tmp_path):
    pytest.importorskip("matplotlib")
    X, labels = _days(10, 1)
    shape = target_shape(X, Target.cycle(labels, range(7)), n_shuffles=10)
    assert shape.plot(str(tmp_path / "days.png")) and (tmp_path / "days.png").stat().st_size > 0


# --------------------------------------------------------------------------------- target probes
def _apply(intervention, H, layer):
    """An intervention's edit applied to activations ``(items, dim)`` at ``layer``, outside a model."""
    import torch
    from concept_probe import Where

    h = torch.tensor(H, dtype=torch.float64)[None]
    intervention.prepare(layer + 1)
    where = Where(torch.zeros(1, h.shape[1], dtype=torch.long), torch.ones(1, dtype=torch.long))
    return intervention.edit(layer, h, where)[0].numpy()


def test_a_target_probe_scores_new_activations_and_moves_them_around_the_cycle():
    from concept_probe import TargetProbe

    X, labels = _days(20, 1)
    new, new_labels = _days(10, 2)
    probe = TargetProbe.fit(X, Target.cycle(labels, range(7)), save=False)
    assert probe.layer == 2 and probe.reading["score"] > 0.9 and probe.shape["fit"] > probe.shape["fit_null_p95"]
    s = probe.score(new)
    assert np.mean(np.array(s.nearest) == new_labels) > 0.9
    on_own_day = (s.position - new_labels + 3.5) % 7 - 3.5
    assert np.abs(on_own_day).mean() < 0.3 and abs(s.fit.mean() - 1.0) < 0.3
    for steps in (1, -2):
        moved = probe.score(_apply(probe.steering(steps), new[:, 2, :], 2))
        delta = (moved.position - s.position + 3.5) % 7 - 3.5
        assert delta.mean() == pytest.approx(steps, abs=0.1)
    toward = probe.toward(source=3, steps=2)  # from class 3, two steps on
    assert probe.toward(2, source=3).label.endswith("to 2")  # to class 2
    with pytest.raises(ValueError, match="either"):
        probe.toward(2, source=3, steps=1)
    step = _apply(toward, new[new_labels == 3][:, 2, :], 2)
    assert np.array(probe.score(step).nearest).tolist().count(5) >= 9
    np.testing.assert_allclose(step - new[new_labels == 3][:, 2, :],
                               np.tile(probe.centroids[5, 2] - probe.centroids[3, 2], (10, 1)), atol=1e-5)
    rotated_elsewhere = probe.steering(1, layer=1)
    assert rotated_elsewhere.describe()["layers"] == [1]


def test_a_target_probe_for_numbers_and_categories(tmp_path):
    from concept_probe import TargetProbe

    rng = np.random.default_rng(3)
    Z = rng.normal(size=(200, L, D))
    y = 2.0 * Z[:, 1, 3] + rng.normal(size=200) * 0.1
    num = TargetProbe.fit(Z, Target.number(y), alpha=0.1, save=False)
    assert num.layer == 1 and num.reading["metric"] == "R2" and num.reading["score"] > 0.9
    before = num.score(Z[:5]).position
    after = num.score(_apply(num.steering(1.5), Z[:5, 1, :], 1)).position
    np.testing.assert_allclose(after - before, 1.5, atol=1e-6)  # moves the predicted value by exactly that
    X, labels = _days(20, 1)
    cats = TargetProbe.fit(X, Target.categories([f"c{k}" for k in labels]), save=True, root_dir=str(tmp_path))
    assert cats.score(X[:3]).nearest == ["c0", "c0", "c0"] and cats.score(X[:3]).class_scores.shape == (3, 7)
    with pytest.raises(ValueError, match="toward"):
        cats.steering(1)
    loaded = TargetProbe.load(cats.dir)
    assert loaded.score(X[:5]).nearest == cats.score(X[:5]).nearest and loaded.reading == cats.reading
    np.testing.assert_allclose(loaded.centroids, cats.centroids)


def test_a_target_probe_reads_texts_whole_and_token_by_token():
    from concept_probe import TargetProbe

    model, tok = TinyCausalLM(), CharChatTok()
    texts = ["pp xp", "p pp", "xpp", "pxp", "nn xn", "n nn", "xnn", "nxn", "zz yz", "z zz", "yzz", "zyz"] * 2
    labels = (["p"] * 4 + ["n"] * 4 + ["z"] * 4) * 2
    probe = TargetProbe.fit(read_texts(model, tok, texts, cache=False), Target.categories(labels), save=False)
    probe.attach(model, tok)
    assert probe.read(["ppp", "nnn"]).nearest == ["p", "n"]
    per_token = probe.score_tokens(["pn z"])[0]
    assert len(per_token) == 4 and per_token.nearest[0] == "p"
