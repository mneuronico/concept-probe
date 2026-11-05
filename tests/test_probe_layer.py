"""Directions (layer 2) and the probe in memory (layer 3): estimators, Probe, calibrations,
thresholds, a separate steering direction (decision D1), and training built from them."""
import json
import os

import numpy as np
import pytest

import concept_probe
from concept_probe import ConceptProbe, ConceptSpec, ProbeWorkspace, ReadContext, read_texts
from concept_probe.directions import (
    MeanDiff,
    concept_direction,
    estimator_spec,
    fit_directions,
    make_estimator,
    register_estimator,
)
from concept_probe.linear_probe import Calibration, fit_threshold
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


def test_drop_outliers_zeroes_a_massive_coordinate_the_concept_does_not_move():
    pos, neg = _blobs(dim=40)
    rng = np.random.default_rng(1)
    pos[:, :, 0] += 300 * rng.normal(size=pos.shape[:2])
    neg[:, :, 0] += 300 * rng.normal(size=neg.shape[:2])
    raw = fit_directions("mean_diff", pos, neg)
    dropped = fit_directions("mean_diff_drop_outliers", pos, neg)
    assert np.all(dropped[:, 0] == 0) and np.any(np.abs(raw[:, 0]) > 0.05)


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
            v[0] = 1.0
            return v

        def spec(self):
            return {"kind": "first"}

    register_estimator("first", FirstCoordinate)
    pos, neg = _blobs()
    pos[:, :, 0] += 1.0
    assert np.all(fit_directions({"kind": "first"}, pos, neg)[:, 0] == 1.0)


# ------------------------------------------------------------------------------ a probe from arrays
def _fit(pos, neg, **kw):
    kw.setdefault("identity", "tiny-test")
    kw.setdefault("save", False)
    return ConceptProbe.fit(pos, neg, **kw)


def test_a_probe_from_representations_chooses_its_layer_and_reads_z():
    pos, neg = _blobs(layers=4)
    pos[:, 2, 1] += 3.0  # layer 2 separates best
    sel_pos, sel_neg = _blobs(layers=4, seed=1)
    sel_pos[:, 2, 1] += 3.0
    ctx = ReadContext(role="raw")
    probe = _fit(pos[:40], neg[:40], select=(sel_pos, sel_neg), calibrate=(pos[40:], neg[40:]), context=ctx)
    assert probe.layer == 2 and probe.model_bundle is None
    z_pos, z_neg = probe.z(sel_pos, ctx), probe.z(sel_neg, ctx)
    assert z_pos.mean() > 1 > -1 > z_neg.mean()
    with pytest.raises(ValueError, match="no calibration for items read as generated completions"):
        probe.z(sel_pos, ReadContext.generated())


def test_a_probe_must_know_which_model_its_representations_come_from(tmp_path):
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
    probe = ConceptProbe.fit(reps, negs, layer=1, calibrate=(reps, negs), save=False,
                             config_overrides={"output": {"root_dir": str(tmp_path), "console": False}})
    with pytest.raises(ValueError, match="needs the model loaded"):
        probe.score_texts(["pp"])
    with pytest.raises(ValueError, match="dtype"):
        probe.attach(TinyCausalLM().float(), CharChatTok())
    probe.attach(model, tok)
    scores = probe.score_texts(["pp x"], batch_subdir="b", save_html=False, save=True)
    assert scores.dir.startswith(str(tmp_path))  # the probe got a run folder when it first saved
    z = probe.z(read_texts(model, tok, ["pp x"], cache=False))
    np.testing.assert_allclose(scores.z()[0], z[0], rtol=1e-4)


def test_classify_uses_the_calibrated_threshold():
    pos, neg = _blobs(gap=3.0)
    probe = _fit(pos, neg, layer=0, calibrate=(pos, neg))
    labels, prob = probe.classify(np.concatenate([pos, neg]))
    accuracy = np.mean(labels == np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    assert accuracy > 0.9 and np.all((prob >= 0) & (prob <= 1))


def test_fit_threshold_puts_one_half_at_the_threshold_and_stays_finite_when_separated():
    t = fit_threshold(np.array([3.0, 4.0, 5.0]), np.array([-1.0, 0.0, 1.0]))
    assert np.isfinite(t["slope"]) and 1.0 < t["threshold"] < 3.0
    assert abs(1 / (1 + np.exp(-(t["slope"] * t["threshold"] + t["intercept"]))) - 0.5) < 1e-9


def test_a_concept_against_a_baseline_puts_the_baseline_at_zero():
    concept, baseline = _blobs(gap=2.0)
    probe = _fit(concept, baseline, layer=0, calibrate=(concept, baseline), baseline=True)
    assert probe.calibrations[0].zero == "baseline"
    assert abs(probe.z(baseline).mean()) < 1e-9 and probe.z(concept).mean() > 1


def test_probes_save_by_default_and_load_back(tmp_path):
    pos, neg = _blobs()
    previous = concept_probe.configure(save=True)
    try:
        probe = ConceptProbe.fit(pos, neg, estimator="mean_diff_var", layer=1, calibrate=(pos, neg),
                                 context=ReadContext(role="user"), identity="tiny-test",
                                 config_overrides={"output": {"root_dir": str(tmp_path), "console": False}},
                                 run_name="r")
        assert probe.run_dir == os.path.join(str(tmp_path), "concept", "r")
        nothing = ConceptProbe.fit(pos, neg, layer=1, identity="tiny-test", save=False,
                                   config_overrides={"output": {"root_dir": str(tmp_path / "none")}})
        assert nothing.run_dir is None and not (tmp_path / "none").exists()
    finally:
        concept_probe.configure(save=previous.save)
    back = ConceptProbe.load(probe.run_dir)
    np.testing.assert_array_equal(back.directions, probe.directions)
    np.testing.assert_array_equal(back.steer_directions, probe.steer_directions)
    np.testing.assert_allclose(back.z(pos, ReadContext(role="user")), probe.z(pos, ReadContext(role="user")))
    assert back.estimator == {"kind": "mean_diff", "normalization": "var"} and back.identity == {"model_id": "tiny-test"}


def test_a_probe_from_directions_made_elsewhere():
    probe = ConceptProbe.from_directions(np.array([[3.0, 4.0]]), layer=0, identity="m", save=False)
    np.testing.assert_allclose(probe.directions, [[0.6, 0.8]])
    np.testing.assert_allclose(probe.scores(np.array([[[1.0, 1.0]]])), [1.4])
    assert probe.steer_directions is probe.directions


def test_reading_and_steering_directions_are_separate_by_default():
    pos, neg = _blobs(dim=20)
    pos[:, :, 5] *= 10
    neg[:, :, 5] *= 10
    probe = _fit(pos, neg, estimator="mean_diff_var", layer=0)
    np.testing.assert_allclose(probe.steer_directions, fit_directions("mean_diff", pos, neg), atol=1e-6)
    assert not np.allclose(probe.steer_directions, probe.directions)
    same = _fit(pos, neg, estimator="mean_diff_var", steer="read", layer=0)
    assert same.steer_directions is same.directions


# ------------------------------------------------------------------------ training (high level)
TRAIN_POS = ["pxp yz", "ppy x", "zp pp", "yxpp", "p zp x", "xpyp", "pp z", "ypxp z"]
EVAL_POS = ["py p", "xp zp", "pp yx", "zpx", "p p y", "yp z", "xxpp", "pzy p", "pyz", "zzp x", "xpp", "ppz y"]


def _flip(texts):
    return [t.replace("p", "n") for t in texts]


def _train(tmp_path, overrides=None, **concept):
    ws = ProbeWorkspace(config_overrides=concept_probe.utils.deep_merge({
        "output": {"root_dir": str(tmp_path), "console": False}, "plots": {"enabled": False},
        "training": {"train_prompt_mode": "texts"}, "evaluation": {"n_permutations": 20, "n_bootstrap": 20},
        "steering": {"steer_max_new_tokens": 6},
    }, overrides or {}), model=TinyCausalLM(), tokenizer=CharChatTok(), model_id="tiny-test")
    fields = dict(name="c", pos_label="p", neg_label="n", pos_system="", neg_system="",
                  train_pos_texts=TRAIN_POS, train_neg_texts=_flip(TRAIN_POS),
                  eval_pos_texts=EVAL_POS, eval_neg_texts=_flip(EVAL_POS))
    fields.update(concept)
    probe = ws.train_concept(ConceptSpec(**fields), run_name="r")
    metrics = json.loads(open(os.path.join(probe.run_dir, "metrics.json"), encoding="utf-8").read())
    return ws, probe, metrics


def test_training_reads_with_the_estimator_and_steers_with_the_raw_difference(tmp_path):
    _, probe, metrics = _train(tmp_path, {"training": {"estimator": "mean_diff_var"}})
    assert metrics["estimator"] == {"kind": "mean_diff", "normalization": "var"}
    assert metrics["steer_direction"] == {"kind": "mean_diff", "normalization": "none"}
    assert "steering.steer_direction" in [c["setting"] for c in metrics["choices"]]
    assert not np.allclose(probe.steer_directions, probe.directions)
    assert metrics["steer_sigma_best_layer"] != metrics["sigma_best_layer"]
    it = probe.score_prompts(["say"], alphas=[2.0], alpha_unit="sigma", batch_subdir="b")[0]
    assert it.info["steer_alpha_per_layer"][probe.layer] == pytest.approx(2.0 * probe.steer_sigma)


def test_steering_can_use_the_reading_direction(tmp_path):
    _, probe, metrics = _train(tmp_path, {"training": {"estimator": "mean_diff_var"},
                                          "steering": {"steer_direction": "read"}})
    assert probe.steer_directions is probe.directions and metrics["steer_sigma_best_layer"] == metrics["sigma_best_layer"]


def test_the_old_normalization_setting_is_gone_but_old_runs_load(tmp_path):
    with pytest.raises(ValueError, match="training.direction_normalization"):
        _train(tmp_path / "a", {"training": {"direction_normalization": "var"}})
    ws, probe, _ = _train(tmp_path / "b", {"training": {"estimator": "mean_diff_var"}})
    cfg_path = os.path.join(probe.run_dir, "config.json")
    cfg = json.loads(open(cfg_path, encoding="utf-8").read())
    cfg["training"].pop("estimator")
    cfg["training"]["direction_normalization"] = "var"  # as earlier versions saved it
    open(cfg_path, "w", encoding="utf-8").write(json.dumps(cfg))
    metrics_path = os.path.join(probe.run_dir, "metrics.json")
    metrics = json.loads(open(metrics_path, encoding="utf-8").read())
    metrics.pop("estimator")
    open(metrics_path, "w", encoding="utf-8").write(json.dumps(metrics))
    assert ConceptProbe.load(probe.run_dir).estimator == {"kind": "mean_diff", "normalization": "var"}


def test_training_with_another_estimator(tmp_path):
    _, probe, metrics = _train(tmp_path, {"training": {"estimator": "logistic"}})
    assert metrics["estimator"]["kind"] == "logistic" and metrics["best_d"] > 1


def test_training_reports_a_threshold_and_where_neutral_texts_fall(tmp_path):
    _, probe, metrics = _train(tmp_path, neutral_texts=["xyz", "zy x", "yy", "x z y"])
    assert metrics["threshold"]["layer"] == metrics["best_layer"]
    assert metrics["threshold"]["fit_on"] == "the selection part of the primary set"
    report = metrics["neutral_texts"]
    assert report["n"] == 4 and report["calibration"] == "evaluation" and np.isfinite(report["z_at_best_layer"])


def test_training_on_every_token_as_a_sample(tmp_path):
    _, probe, metrics = _train(tmp_path, {"readout": {"read_mode": "assistant_tokens"}})
    assert metrics["train_samples"]["pos"] > metrics["n_train_pos"] == len(TRAIN_POS)
    assert metrics["reference_context"]["mode"] == "assistant_all_mean"
    assert "evaluation readout" in [c["setting"] for c in metrics["choices"]]


def test_a_loaded_run_keeps_its_steering_direction_and_calibrations(tmp_path):
    ws, probe, _ = _train(tmp_path, {"training": {"estimator": "mean_diff_var"}})
    loaded = ws.get_probe(project_directory=probe.run_dir)
    np.testing.assert_array_equal(loaded.steer_directions, probe.steer_directions)
    assert loaded.steer_sigma == pytest.approx(probe.steer_sigma)
    assert [c.source for c in loaded.calibrations] == ["evaluation"] and loaded.calibrations[0].threshold


def test_an_older_run_steers_with_its_one_direction_and_says_so(tmp_path, capsys):
    ws, probe, _ = _train(tmp_path, {"training": {"estimator": "mean_diff_var"}, "output": {"console": True}})
    path = os.path.join(probe.run_dir, "tensors.npz")
    with np.load(path) as t:
        old = {k: t[k] for k in t.files if k not in ("steer_vectors", "steer_sigma_best_layer", "calibrations")}
    np.savez(path, **old)
    capsys.readouterr()
    loaded = ws.get_probe(project_directory=probe.run_dir)
    np.testing.assert_array_equal(loaded.steer_directions, loaded.directions)
    assert "saved before steering had its own direction" in capsys.readouterr().out


def test_a_texts_probe_says_why_generations_get_no_z_and_can_be_calibrated(tmp_path):
    _, probe, _ = _train(tmp_path)
    scores = probe.score_prompts(["say"], batch_subdir="before")
    assert np.isnan(scores.z()[0]) and "as generated completions" in scores.z_missing[scores.probes[0]]
    probe.calibrate(EVAL_POS, _flip(EVAL_POS), as_reply_to="say")
    scores = probe.score_prompts(["say"], batch_subdir="after")
    assert np.isfinite(scores.z()[0]) and scores.z_missing[scores.probes[0]] is None


def test_reading_arrays_and_scoring_texts_give_the_same_z(tmp_path):
    _, probe, _ = _train(tmp_path)
    texts = ["pp xp", "nn yn", "xyz"]
    from_records = probe.score_texts(texts, batch_subdir="z").z()
    reps = read_texts(probe.model_bundle.model, probe.model_bundle.tokenizer, texts, cache=False)
    np.testing.assert_allclose(probe.z(reps, ReadContext(role="raw")), from_records, rtol=1e-4)
