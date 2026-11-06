"""Directions (layer 2) and the probe (layer 3): estimators, fitting with eval sets, calibrations,
thresholds, units, a separate steering direction (decision D1), saving and loading."""
import json
import os

import numpy as np
import pytest

import concept_probe
from concept_probe import ConceptProbe, EvalSet, ReadContext, read_texts
from concept_probe.directions import (
    MeanDiff,
    concept_direction,
    estimator_spec,
    fit_directions,
    make_estimator,
    register_estimator,
)
from concept_probe.linear_probe import fit_threshold
from fakes import CharChatTok
from tiny_model import TinyCausalLM


def _blobs(n=60, layers=3, dim=12, gap=1.5, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = rng.normal(size=(n, layers, dim)), rng.normal(size=(n, layers, dim))
    pos[:, :, 1] += gap
    return pos, neg


# ------------------------------------------------------------------------------------ estimators
@pytest.mark.parametrize("name", ["mean_diff", "mean_diff_var", "mean_diff_sd", "mean_diff_drop_outliers",
                                  "logistic", "lda"])
def test_every_estimator_gives_unit_directions_toward_the_first_group(name):
    pos, neg = _blobs()
    v = fit_directions(name, pos, neg)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.all(np.einsum("nld,ld->l", pos, v) > np.einsum("nld,ld->l", neg, v))
    assert abs(v[0, 1]) > 0.7  # found the coordinate that separates the groups


def test_mean_diff_is_the_original_concept_direction():
    pos, neg = _blobs()
    for norm in ("none", "var", "sd"):
        np.testing.assert_allclose(fit_directions({"kind": "mean_diff", "normalization": norm}, pos, neg)[2],
                                   concept_direction(pos[:, 2], neg[:, 2], norm), atol=1e-6)


def test_drop_outliers_zeroes_a_massive_coordinate_the_concept_does_not_move_and_says_how_many():
    pos, neg = _blobs(dim=40)
    rng = np.random.default_rng(1)
    pos[:, :, 0] += 300 * rng.normal(size=pos.shape[:2])
    neg[:, :, 0] += 300 * rng.normal(size=neg.shape[:2])
    raw = fit_directions("mean_diff", pos, neg)
    dropped = fit_directions("mean_diff_drop_outliers", pos, neg)
    assert np.all(dropped[:, 0] == 0) and np.any(np.abs(raw[:, 0]) > 0.05)
    probe = ConceptProbe.fit(pos, neg, estimator="mean_diff_drop_outliers", layer=0, identity="m", save=False)
    assert probe.estimator["dropped_coordinates"] == [1, 1, 1]
    assert "outlier coordinates" in [c.setting for c in probe.choices]


def test_estimators_from_names_dicts_objects_and_the_registry():
    assert estimator_spec("mean_diff_var") == {"kind": "mean_diff", "normalization": "var"}
    assert make_estimator({"kind": "logistic", "C": 1.0}).C == 1.0
    own = MeanDiff("sd")
    assert make_estimator(own) is own
    with pytest.raises(ValueError, match="Unknown estimator"):
        make_estimator("ridge")

    class FirstCoordinate:
        def fit(self, pos, neg):
            v = np.zeros(pos.shape[1])
            v[0] = -1.0  # pointing away from the positive pole: the library turns it around, and says so
            return v

        def spec(self):
            return {"kind": "first"}

    register_estimator("first", FirstCoordinate)
    pos, neg = _blobs()
    pos[:, :, 0] += 1.0
    assert np.all(fit_directions({"kind": "first"}, pos, neg)[:, 0] == 1.0)
    probe = ConceptProbe.fit(pos, neg, estimator={"kind": "first"}, layer=0, identity="m", save=False)
    assert probe.estimator["flipped_layers"] == [0, 1, 2]
    assert "direction sign" in [c.setting for c in probe.choices]


# ------------------------------------------------------------------------------ a probe from arrays
def _fit(pos, neg, **kw):
    kw.setdefault("identity", "tiny-test")
    kw.setdefault("save", False)
    return ConceptProbe.fit(pos, neg, **kw)


def test_a_probe_chooses_its_layer_on_eval_items_and_reads_z():
    pos, neg = _blobs(layers=4)
    pos[:, 2, 1] += 3.0  # layer 2 separates best
    sel_pos, sel_neg = _blobs(layers=4, seed=1)
    sel_pos[:, 2, 1] += 3.0
    ctx = ReadContext(role="raw")
    probe = _fit(pos, neg, eval=EvalSet("eval", sel_pos, sel_neg, context=ctx), n_permutations=50, n_bootstrap=50)
    assert probe.layer == 2 and probe.model_bundle is None and probe.metrics["d"] > 1
    z_pos, z_neg = probe.z(sel_pos, ctx), probe.z(sel_neg, ctx)
    assert z_pos.mean() > 0.5 > -0.5 > z_neg.mean()
    with pytest.raises(ValueError, match="no calibration for items read as replies"):
        probe.z(sel_pos, ReadContext.generated())
    with pytest.raises(ValueError, match="eval=.*layer="):
        _fit(pos, neg)


def test_z_in_the_poles_scale():
    concept, other = _blobs(gap=2.0)
    probe = _fit(concept, other, layer=0)
    probe.calibrate(concept, other)
    assert probe.z(concept, unit="poles").mean() == pytest.approx(1.0)
    assert probe.z(other, unit="poles").mean() == pytest.approx(-1.0)
    probe.calibrate(concept, other, baseline=True)
    assert probe.z(other, unit="poles").mean() == pytest.approx(0.0, abs=1e-9)
    assert probe.z(concept, unit="poles").mean() == pytest.approx(1.0)


def test_token_scores_are_summarized_per_text():
    model, tok = TinyCausalLM(), CharChatTok()
    P, N = read_texts(model, tok, ["pp x", "pxp"], cache=False), read_texts(model, tok, ["nn x", "nxn"], cache=False)
    probe = _fit(P, N, layer=1)
    tokens = read_texts(model, tok, ["pp x", "nxn y"], "text_tokens", cache=False)
    per_token = probe.scores(tokens)
    assert len(per_token) == len(tokens.data)
    for how in ("mean", "max", "min"):
        per_text = probe.scores(tokens, per_text=how)
        expected = [getattr(np, how)(per_token[tokens.items == i]) for i in (0, 1)]
        np.testing.assert_allclose(per_text, expected)
    with pytest.raises(ValueError, match="tokens readout"):
        probe.scores(P, per_text="max")


def test_a_probe_must_know_which_model_its_representations_come_from():
    pos, neg = _blobs()
    with pytest.raises(ValueError, match="Which model"):
        ConceptProbe.fit(pos, neg, layer=0, save=False)
    reps = read_texts(TinyCausalLM(), CharChatTok(), ["pp x", "p y"], cache=False)
    negs = read_texts(TinyCausalLM(), CharChatTok(), ["nn x", "n y"], cache=False)
    probe = ConceptProbe.fit(reps, negs, layer=1, save=False)
    assert probe.identity["dtype"] == "float64"  # recorded by read_texts
    with pytest.raises(ValueError, match="another model"):
        ConceptProbe.fit(reps, negs, layer=1, save=False, model=TinyCausalLM().float(), tokenizer=CharChatTok())


def test_what_needs_the_model_says_so_and_attach_gives_it(tmp_path):
    model, tok = TinyCausalLM(), CharChatTok()
    reps = read_texts(model, tok, ["pp x", "pxp", "p y"], cache=False)
    negs = read_texts(model, tok, ["nn x", "nxn", "n y"], cache=False)
    probe = ConceptProbe.fit(reps, negs, layer=1, save=False, root_dir=str(tmp_path))
    probe.calibrate(reps, negs)
    with pytest.raises(ValueError, match="needs the model loaded"):
        probe.score_texts(["pp"])
    with pytest.raises(ValueError, match="dtype"):
        probe.attach(TinyCausalLM().float(), CharChatTok())
    probe.attach(model, tok)
    scores = probe.score_texts(["pp x"], html=False, save=True)
    assert scores.dir.startswith(str(tmp_path))
    z = probe.z(read_texts(model, tok, ["pp x"], cache=False))
    np.testing.assert_allclose(scores.z()[0], z[0], rtol=1e-4)
    with pytest.raises(ValueError, match="read with 'text_last'"):  # pooled another way than the calibration
        probe.z(read_texts(model, tok, ["pp x"], "text_last", cache=False))


def test_classify_and_accuracy_use_the_calibrated_threshold():
    pos, neg = _blobs(gap=3.0)
    probe = _fit(pos, neg, layer=0)
    probe.calibrate(pos, neg, threshold_l2=0.1)
    labels, prob = probe.classify(np.concatenate([pos, neg]))
    accuracy = np.mean(labels == np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    assert accuracy > 0.9 and np.all((prob >= 0) & (prob <= 1))
    assert probe.calibrations[0].threshold["l2"] == 0.1
    out = probe.accuracy(pos, neg, n_bootstrap=200)
    assert out["accuracy"] == pytest.approx(accuracy) and out["ci95"][0] <= out["accuracy"] <= out["ci95"][1]
    assert probe.accuracy(pos, neg, cut="zero")["cut"] == "zero"


def test_fit_threshold_puts_one_half_at_the_threshold_and_stays_finite_when_separated():
    t = fit_threshold(np.array([3.0, 4.0, 5.0]), np.array([-1.0, 0.0, 1.0]))
    assert np.isfinite(t["slope"]) and 1.0 < t["threshold"] < 3.0
    assert abs(1 / (1 + np.exp(-(t["slope"] * t["threshold"] + t["intercept"]))) - 0.5) < 1e-9


def test_a_concept_against_a_baseline_puts_the_baseline_at_zero():
    concept, baseline = _blobs(gap=2.0)
    probe = _fit(concept, baseline, layer=0)
    probe.calibrate(concept, baseline, baseline=True)
    assert probe.calibrations[0].zero == "baseline"
    assert abs(probe.z(baseline).mean()) < 1e-9 and probe.z(concept).mean() > 1


def test_probes_save_by_default_and_load_back(tmp_path):
    pos, neg = _blobs()
    previous = concept_probe.configure(save=True, console=False)
    try:
        probe = ConceptProbe.fit(pos, neg, estimator="mean_diff_var", layer=1, identity="tiny-test",
                                 root_dir=str(tmp_path), run_name="r")
        assert probe.run_dir == os.path.join(str(tmp_path), "concept", "r")
        probe.calibrate(pos, neg, context=ReadContext(role="user"))
        probe.save()
        nothing = ConceptProbe.fit(pos, neg, layer=1, identity="tiny-test", save=False, root_dir=str(tmp_path / "none"))
        assert nothing.run_dir is None and not (tmp_path / "none").exists()
        with pytest.raises(FileExistsError):
            ConceptProbe.fit(pos, neg, layer=1, identity="tiny-test", root_dir=str(tmp_path), run_name="r")
    finally:
        concept_probe.configure(save=previous.save, console=previous.console)
    back = ConceptProbe.load(probe.run_dir)
    np.testing.assert_array_equal(back.directions, probe.directions)
    np.testing.assert_array_equal(back.steer_directions, probe.steer_directions)
    np.testing.assert_allclose(back.z(pos, ReadContext(role="user")), probe.z(pos, ReadContext(role="user")))
    assert back.estimator == {"kind": "mean_diff", "normalization": "var"} and back.identity == {"model_id": "tiny-test"}


def test_a_probe_from_directions_made_elsewhere_reads_with_unit_ones_and_steers_with_them_as_given():
    probe = ConceptProbe.from_directions(np.array([[3.0, 4.0]]), layer=0, identity="m", save=False)
    np.testing.assert_allclose(probe.directions, [[0.6, 0.8]])
    np.testing.assert_allclose(probe.scores(np.array([[[1.0, 1.0]]])), [1.4])
    np.testing.assert_allclose(probe.steer_directions, [[3.0, 4.0]])
    assert "reading directions" in [c.setting for c in probe.choices]


def test_reading_and_steering_directions_are_separate_by_default():
    pos, neg = _blobs(dim=20)
    pos[:, :, 5] *= 10
    neg[:, :, 5] *= 10
    probe = _fit(pos, neg, estimator="mean_diff_var", layer=0)
    np.testing.assert_allclose(probe.steer_directions, fit_directions("mean_diff", pos, neg), atol=1e-6)
    assert not np.allclose(probe.steer_directions, probe.directions)
    same = _fit(pos, neg, estimator="mean_diff_var", steer="read", layer=0)
    assert same.steer_directions is same.directions


# ------------------------------------------------------------------------------- fitting with eval sets
TRAIN_POS = ["pxp yz", "ppy x", "zp pp", "yxpp", "p zp x", "xpyp", "pp z", "ypxp z"]
EVAL_POS = ["py p", "xp zp", "pp yx", "zpx", "p p y", "yp z", "xxpp", "pzy p", "pyz", "zzp x", "xpp", "ppz y"]


def _flip(texts):
    return [t.replace("p", "n") for t in texts]


def _fit_texts(readout="text_mean", **kw):
    model, tok = TinyCausalLM(), CharChatTok()
    P = read_texts(model, tok, TRAIN_POS, readout, norms=True, cache=False)
    N = read_texts(model, tok, _flip(TRAIN_POS), readout, norms=True, cache=False)
    ev = EvalSet("eval", read_texts(model, tok, EVAL_POS, cache=False),
                 read_texts(model, tok, _flip(EVAL_POS), cache=False))
    probe = ConceptProbe.fit(P, N, eval=ev, n_permutations=20, n_bootstrap=20, save=False, **kw)
    return probe.attach(model, tok)


def test_fitting_reads_with_the_estimator_and_steers_with_the_raw_difference():
    probe = _fit_texts(estimator="mean_diff_var")
    assert probe.estimator == {"kind": "mean_diff", "normalization": "var"}
    assert probe.steer_estimator == {"kind": "mean_diff", "normalization": "none"}
    assert "steering direction" in [c.setting for c in probe.choices]
    assert not np.allclose(probe.steer_directions, probe.directions)
    assert probe.steer_sigma != pytest.approx(probe.calibrations[0].scale[probe.layer])
    add = probe.steering(2.0, "sigma")
    add.prepare(probe.num_layers)
    assert add.alpha_per_layer[probe.layer] == pytest.approx(2.0 * probe.steer_sigma)


def test_steering_can_use_the_reading_direction():
    probe = _fit_texts(estimator="mean_diff_var", steer="read")
    assert probe.steer_directions is probe.directions
    assert probe.steer_sigma == pytest.approx(probe.calibrations[0].scale[probe.layer])


def _old_run(folder, *, estimator_setting=None, steer=True):
    """A training run as earlier versions saved it: config.json, concept.json, metrics.json, tensors.npz."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cal = {"context": {"kind": "text", "role": "raw", "system": None, "prompt": None, "mode": "assistant_all_mean"},
           "center": [0.0] * 3, "scale": [1.0] * 3, "token_center": None, "token_scale": None, "zero": "midpoint",
           "neutral": None, "n_neutral": 0, "threshold": None, "n_pos": 4, "n_neg": 4, "source": "evaluation"}
    tensors = {"concept_vectors": v, "best_layer": np.array([1], np.int32), "sweep_d": np.zeros(3),
               "resid_norm": np.ones(3), "calibrations": np.array([json.dumps([cal])])}
    if steer:
        tensors.update(steer_vectors=v, steer_sigma_best_layer=np.array([0.5]))
    else:
        tensors["score_scale"] = np.ones(3)
    np.savez(os.path.join(folder, "tensors.npz"), **tensors)
    training = {"direction_normalization": estimator_setting} if estimator_setting else {"estimator": "mean_diff"}
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump({"model": {"model_id": "m", "dtype": "float16", "use_4bit": False}, "training": training,
                   "evaluation": {"layer_selection": {"mode": "window", "window_radius": 1}}}, f)
    with open(os.path.join(folder, "concept.json"), "w") as f:
        json.dump({"name": "old", "pos_label": "a", "neg_label": "b"}, f)
    with open(os.path.join(folder, "metrics.json"), "w") as f:
        json.dump({"best_layer": 1, "best_d": 2.0}, f)
    return folder


def test_runs_saved_by_earlier_versions_load_in_todays_terms(tmp_path):
    loaded = ConceptProbe.load(_old_run(str(tmp_path / "a"), estimator_setting="var"))
    assert loaded.estimator == {"kind": "mean_diff", "normalization": "var"}
    assert loaded.name == "old" and loaded.pos_label == "a" and loaded.layer == 1 and loaded.score_layers == [0, 1, 2]
    assert loaded.calibrations[0].context.readout == "text_mean"  # the readout's name of today
    assert loaded.identity["model_id"] == "m" and loaded.metrics["best_d"] == 2.0 and loaded.steer_sigma == 0.5


def test_an_older_run_steers_with_its_one_direction_and_says_so(tmp_path, capsys):
    previous = concept_probe.configure(console=True)
    try:
        capsys.readouterr()
        loaded = ConceptProbe.load(_old_run(str(tmp_path / "b"), estimator_setting="var", steer=False))
    finally:
        concept_probe.configure(console=previous.console)
    np.testing.assert_array_equal(loaded.steer_directions, loaded.directions)
    assert "saved before steering had its own direction" in capsys.readouterr().out


def test_fitting_with_another_estimator_and_its_label_shuffles():
    probe = _fit_texts(estimator="logistic")
    assert probe.estimator["kind"] == "logistic" and probe.metrics["d"] > 1
    assert probe.metrics["label_shuffle"] is None and "n_label_shuffles" in [c.setting for c in probe.choices]


def test_fitting_reports_a_threshold_and_where_neutral_texts_fall():
    model, tok = TinyCausalLM(), CharChatTok()
    neutral = read_texts(model, tok, ["xyz", "zy x", "yy", "x z y"], cache=False)
    probe = _fit_texts(neutral=neutral)
    threshold = probe.calibrations[0].threshold
    assert threshold["layer"] == probe.layer and threshold["fit_on"] == "the selection part of the primary set"
    report = probe.metrics["neutral"]
    assert report["n"] == 4 and np.isfinite(report["z"])


def test_fitting_on_every_token_as_a_sample():
    probe = _fit_texts("text_tokens")
    assert probe.metrics["n_train"]["pos"] > len(TRAIN_POS)


def test_a_saved_probe_keeps_its_steering_direction_and_calibrations(tmp_path):
    probe = _fit_texts(estimator="mean_diff_var")
    loaded = ConceptProbe.load(probe.save(str(tmp_path / "p")))
    np.testing.assert_array_equal(loaded.steer_directions, probe.steer_directions)
    assert loaded.steer_sigma == pytest.approx(probe.steer_sigma)
    assert [c.source for c in loaded.calibrations] == ["evaluation"] and loaded.calibrations[0].threshold
    np.testing.assert_allclose(loaded.resid_norm, probe.resid_norm)


def test_a_texts_probe_gives_completions_a_z_once_calibrated_on_replies():
    probe = _fit_texts()
    replies = read_texts(probe.model_bundle.model, probe.model_bundle.tokenizer, ["pp x", "nn y"], role="assistant",
                         prompt=["say", "say"], cache=False)
    with pytest.raises(ValueError, match="no calibration for items read as replies"):
        probe.z(replies)
    probe.calibrate(EVAL_POS, _flip(EVAL_POS), as_reply_to="say")
    assert np.isfinite(probe.z(replies)).all()


def test_reading_arrays_and_scoring_texts_give_the_same_z():
    probe = _fit_texts()
    texts = ["pp xp", "nn yn", "xyz"]
    from_records = probe.score_texts(texts, save=False).z()
    reps = read_texts(probe.model_bundle.model, probe.model_bundle.tokenizer, texts, cache=False)
    np.testing.assert_allclose(probe.z(reps), from_records, rtol=1e-4)
