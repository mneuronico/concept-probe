"""The probe object (layers 2 and 3 of docs/DESIGN.md).

A :class:`ConceptProbe` is one direction per layer of a model to read a concept with, the layer it
reads at, the calibrations that turn its projections into z scores, and the directions it steers
with. It is built from representations already read (:meth:`ConceptProbe.fit`, which can also
evaluate it on named eval sets), from directions computed elsewhere
(:meth:`ConceptProbe.from_directions`), or from a saved probe (:meth:`ConceptProbe.load`, also for
runs saved by earlier versions). It reads arrays without a model; with one attached
(:meth:`ConceptProbe.attach`) it also scores texts token by token and builds its steering
intervention.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .directions import estimator_spec, fit_directions, make_estimator
from .evaluation import EvalSet, ReadingEvaluation, evaluate_reading, label_shuffle_control
from .inputs import (
    TEXT_ROLES,
    Encoded,
    PromptLike,
    ReadContext,
    _completion_input_ids,
    _normalize_messages,
    _prompt_to_text,
    encode_text,
)
from .interventions import Add
from .linear_probe import Calibration, calibrations_from_tensors, sigmoid, steer_directions_for
from .modeling import ModelBundle, identity_from_config, identity_mismatches, upgrade_saved_config
from .reading import Readout, _read
from .record import Choice, announce
from .scoring import ScoreInput, Scores, TokenScorer, score_inputs
from .settings import settings
from .stats import _pooled_sd_columns, bootstrap_ci, summarize_tokens
from .utils import ensure_dir, json_dump, now_tag, resolve_layer_index, safe_slug
from .visuals import plot_score_hist, plot_sweep

#: Units of a steering dose: as is, the eval items' within-pole SD on the steering direction, or each
#: layer's mean residual norm.
ALPHA_UNITS = ("raw", "sigma", "resid_norm")
#: Units of a probe's z: within-pole SDs, or the poles' scale (+1 the positive mean, 0 the zero).
Z_UNITS = ("sd", "poles")


# ------------------------------------------------------------------------------------- helpers
def _as_array(reps: Any) -> np.ndarray:
    return np.asarray(getattr(reps, "data", reps))


def _window(center: int, radius: Any, num_layers: int) -> Tuple[List[int], Optional[Choice]]:
    """The layers within ``radius`` of ``center``, and the choice made when the window is clipped."""
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"the window radius must be >= 0; got {radius}")
    lo, hi = center - radius, center + radius
    layers = list(range(max(0, lo), min(num_layers - 1, hi) + 1))
    clipped = None
    if lo < 0 or hi > num_layers - 1:
        clipped = Choice("steering layers", f"layers {lo} to {hi}", f"layers {layers[0]} to {layers[-1]}",
                         f"the window around layer {center} reaches past the model's {num_layers} layers")
    return layers, clipped


def _layer_list(values: Any, num_layers: int, what: str) -> List[int]:
    """A non-empty list of distinct layer indices (see :func:`resolve_layer_index`)."""
    if isinstance(values, (int, np.integer)) and not isinstance(values, bool):
        values = [values]
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError(f"{what} must be a non-empty list of layer indices; got {values!r}")
    layers = [resolve_layer_index(v, num_layers, f"{what} entries") for v in values]
    if len(set(layers)) != len(layers):
        raise ValueError(f"{what} repeats a layer: {layers}")
    return layers


# ------------------------------------------------------------------------------------- the probe
@dataclass
class ConceptProbe:
    """A concept probe: one direction per layer of a model to read the concept with, the layer it
    reads at, calibrations that turn its projections into z scores, and the directions it steers with.

    A probe always belongs to a model (``identity``: the model, dtype, quantization and chat
    template its directions were computed on), but needs that model loaded (``model_bundle``) only to
    score texts and to steer. Build one with :meth:`fit`, :meth:`from_directions` or :meth:`load`,
    and attach a loaded model with :meth:`attach`.
    """

    #: the concept's name and the labels of its two poles (the second may be a baseline)
    name: str = "concept"
    pos_label: str = "positive"
    neg_label: str = "negative"
    #: (layers, dim) unit directions to read with, pointing toward the positive pole
    directions: Optional[np.ndarray] = None
    #: the layer the probe reads at
    layer: Optional[int] = None
    #: the layers a score combines (their mean, or sum with ``score_aggregate="sum"``); default the probe's
    score_layers: Optional[List[int]] = None
    score_aggregate: str = "mean"
    #: (layers, dim) directions steering adds (decision D1: the raw difference of means by default)
    steer_directions: Optional[np.ndarray] = None
    #: one calibration per reading context: the zero and unit of z scores, a threshold, neutral texts
    calibrations: List[Calibration] = field(default_factory=list)
    #: the unit of ``unit="sigma"``: the eval items' pooled within-pole SD on the steering direction
    steer_sigma: Optional[float] = None
    #: the unit of ``unit="resid_norm"``: the mean residual norm per layer of the training texts
    resid_norm: Optional[np.ndarray] = None
    #: the model the directions belong to (see :func:`concept_probe.modeling.identity_of`)
    identity: Optional[Dict[str, Any]] = None
    #: the estimators the reading and steering directions came from
    estimator: Optional[Dict[str, Any]] = None
    steer_estimator: Optional[Dict[str, Any]] = None
    #: Cohen's d and its p per layer on the primary eval set (all its items)
    sweep_d: Optional[np.ndarray] = None
    sweep_p: Optional[np.ndarray] = None
    #: the evaluation made when the probe was fitted (in memory), and its numbers (also saved)
    evaluation: Optional[ReadingEvaluation] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    #: every choice made on the caller's behalf
    choices: List[Choice] = field(default_factory=list)
    model_bundle: Optional[ModelBundle] = None
    run_dir: Optional[str] = None
    #: where a probe without a folder creates one when it first saves (None: the session's output_root)
    save_root: Optional[str] = None

    # ------------------------------------------------------------------ building
    @classmethod
    def fit(
        cls,
        pos: Any,
        neg: Any,
        *,
        eval: Union[None, EvalSet, Sequence[EvalSet]] = None,
        layer: Optional[int] = None,
        estimator: Any = "mean_diff",
        steer: Any = "mean_diff",
        search_layers: Optional[Sequence[int]] = None,
        sweep_select: str = "effect_size",
        selection_split: Any = 0.5,
        n_random: int = 50,
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
        n_label_shuffles: Union[int, str] = "auto",
        neutral: Any = None,
        baseline: bool = False,
        erase: Any = None,
        threshold_l2: float = 1e-2,
        seed: int = 0,
        score_layers: Optional[Sequence[int]] = None,
        score_aggregate: str = "mean",
        name: str = "concept",
        pos_label: str = "positive",
        neg_label: str = "negative",
        model: Any = None,
        tokenizer: Any = None,
        identity: Optional[Union[Dict[str, Any], str]] = None,
        allow_model_mismatch: bool = False,
        root_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        save: Optional[bool] = None,
        overwrite: bool = False,
        plots: bool = True,
    ) -> "ConceptProbe":
        """Train a probe from representations already read: ``(items, layers, dim)`` of each pole
        (arrays, or what :func:`concept_probe.read_texts` returns, which says which model they come
        from and how they were read).

        ``estimator`` and ``steer`` set the reading and steering directions (see
        :mod:`concept_probe.directions`; ``steer="read"`` steers with the reading direction).
        ``erase`` (an :class:`~concept_probe.directions.Eraser`) fits both without a confound.

        With ``eval`` (an :class:`~concept_probe.evaluation.EvalSet`, or several: the first is the
        primary), the probe is evaluated as :func:`concept_probe.evaluate_reading` does: the layer is
        chosen on a ``selection_split`` part of the primary set (a fraction, or the items that choose
        it; see there) within ``search_layers``, its numbers come from the rest, every other set is
        measured there, with random-direction nulls (``n_random``), bootstrap intervals and a
        permutation p; ``n_label_shuffles`` refits with shuffled labels (``"auto"``: 10 for the
        difference of means, which is quick, else 0; announced). Each set's scores calibrate the
        probe for items read like it (the primary one with a threshold; ``baseline=True`` puts the
        zero at the second pole's mean, for a concept read against a baseline). ``neutral``, texts
        without the concept read like the primary set, are placed on its calibration (reported, not
        used as the zero). Without ``eval``, give the ``layer``.

        The probe belongs to the model the representations came from: the one ``read_texts``
        recorded, ``model`` (loaded, then attached; another one raises unless
        ``allow_model_mismatch``), or ``identity`` (a dict, or a model id). Representations read with
        ``norms=True`` give the unit of ``unit="resid_norm"``. The probe saves to
        ``<root>/<name>/<run_name or time>`` unless ``save=False`` (or the session does not save),
        with the plots of its evaluation unless ``plots=False``.
        """
        P, N = _as_array(pos), _as_array(neg)
        sets = [] if eval is None else ([eval] if isinstance(eval, EvalSet) else list(eval))
        if layer is None and not sets:
            raise ValueError("Pass eval= (eval sets, to choose the layer on items that did not train) or layer=.")
        probe = cls._new(name, pos_label, neg_label, model, tokenizer, identity, getattr(pos, "identity", None),
                         root_dir, allow_model_mismatch)
        choices: List[Choice] = []
        flips: List[int] = []
        est = make_estimator(estimator)
        read = fit_directions(est, P, N, erase=erase, flips=flips)
        probe.estimator = estimator_spec(est)
        _record_estimator_notes(probe, est, flips, choices)
        steer_dirs, steer_spec, choice = steer_directions_for(est, steer, P, N, read, erase=erase)
        if choice is not None:
            choices.append(choice)
        probe.directions, probe.steer_directions = read, steer_dirs
        probe.steer_estimator = steer_spec
        if erase is not None:
            probe.estimator = dict(probe.estimator, erased=erase.describe())
            probe.steer_estimator = dict(probe.steer_estimator, erased=erase.describe())
        num_layers = read.shape[0]
        norms = [getattr(r, "norms", None) for r in (pos, neg)]
        if all(n is not None for n in norms):
            probe.resid_norm = np.concatenate(norms, axis=0).mean(axis=0)

        if sets:
            ev = evaluate_reading(read, sets, layer=layer, selection_split=selection_split, search_layers=search_layers,
                                  sweep_select=sweep_select, n_random=int(n_random), n_bootstrap=int(n_bootstrap),
                                  n_permutations=int(n_permutations), threshold_l2=threshold_l2, seed=seed)
            probe.layer = int(ev.layer)
            probe.evaluation = ev
            main = ev.main
            probe.sweep_d, probe.sweep_p = main.d_curve, main.p_curve
            primary = sets[0]
            for s in sets:
                result = ev.sets.get(s.name)
                context = s.context or ReadContext()
                if result is None or any(c.context.matches(context) for c in probe.calibrations):
                    continue
                tokens = _token_projections(s, read)
                cal = Calibration.from_scores(result.pos_scores, result.neg_scores, context, pos_tokens=tokens[0],
                                              neg_tokens=tokens[1], zero="baseline" if baseline else "midpoint",
                                              source="evaluation" if s is primary else s.name)
                if s is primary:
                    cal.threshold = ev.threshold
                probe.calibrations.append(cal)
            n_shuffles = n_label_shuffles
            if n_shuffles == "auto":
                n_shuffles = 10 if probe.estimator.get("kind") == "mean_diff" else 0
                choices.append(Choice("n_label_shuffles", "auto", n_shuffles,
                                      "refitting the difference of means is quick, so ten shuffles run" if n_shuffles else
                                      f"refitting a {probe.estimator.get('kind')} direction for every shuffle is slow; "
                                      "pass n_label_shuffles to run them"))
            shuffle = None
            if int(n_shuffles) > 0:
                rp, rn = ev.report_idx
                shuffle = label_shuffle_control(est, P, N, primary.pos[rp], primary.neg[rn], probe.layer,
                                                int(n_shuffles), seed=seed)
            neutral_report = None
            if neutral is not None:
                cal = probe.calibrations[0]
                scores = np.einsum("nld,ld->nl", _as_array(neutral).astype(np.float64), read.astype(np.float64))
                cal.add_neutral(scores)
                z = cal.items_z(scores[:, [probe.layer]], [probe.layer])
                neutral_report = {"n": int(len(scores)), "mean_score": float(scores[:, probe.layer].mean()),
                                  "z": None if z is None else float(z.mean()),
                                  "z_sd": None if z is None or len(z) < 2 else float(z.std(ddof=1))}
            probe.metrics = _metrics(ev, shuffle, neutral_report, len(P), len(N))
            for message in ev.warnings:
                _warn(message)
        else:
            probe.layer = resolve_layer_index(layer, num_layers, "layer")
        probe.score_layers = None if score_layers is None else _layer_list(list(score_layers), num_layers, "score_layers")
        if score_aggregate not in ("mean", "sum"):
            raise ValueError(f"score_aggregate must be 'mean' or 'sum'; got {score_aggregate!r}")
        probe.score_aggregate = score_aggregate
        if probe.calibrations:
            primary_set = sets[0]
            if steer_dirs is not read:
                sp = np.einsum("nd,d->n", primary_set.pos[:, probe.layer].astype(np.float64), steer_dirs[probe.layer])
                sn = np.einsum("nd,d->n", primary_set.neg[:, probe.layer].astype(np.float64), steer_dirs[probe.layer])
                probe.steer_sigma = float(_pooled_sd_columns(sp[:, None], sn[:, None])[0])
            else:
                probe.steer_sigma = float(probe.calibrations[0].scale[probe.layer])
            if not np.isfinite(probe.steer_sigma):
                probe.steer_sigma = None
        probe.choices = choices
        announce(choices)
        probe._save_new(run_name, save, overwrite, "fit", plots=plots)
        return probe

    @classmethod
    def from_directions(
        cls,
        directions: np.ndarray,
        layer: int,
        *,
        steer_directions: Optional[np.ndarray] = None,
        name: str = "concept",
        pos_label: str = "positive",
        neg_label: str = "negative",
        model: Any = None,
        tokenizer: Any = None,
        identity: Optional[Union[Dict[str, Any], str]] = None,
        allow_model_mismatch: bool = False,
        root_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        save: Optional[bool] = None,
        overwrite: bool = False,
    ) -> "ConceptProbe":
        """A probe from directions computed elsewhere, ``(layers, dim)``. It reads with them scaled to
        unit length per layer (announced when they were not), and steers with ``steer_directions``,
        or with the directions as given. Calibrate it with :meth:`calibrate` to get z scores. The
        model and saving work as in :meth:`fit`."""
        probe = cls._new(name, pos_label, neg_label, model, tokenizer, identity, None, root_dir, allow_model_mismatch)
        v = np.asarray(directions, dtype=np.float64)
        if v.ndim != 2:
            raise ValueError(f"directions must be (layers, dim); got shape {v.shape}")
        lengths = np.linalg.norm(v, axis=1)
        probe.directions = (v / np.clip(lengths, 1e-12, None)[:, None]).astype(np.float32)
        if not np.allclose(lengths, 1.0, atol=1e-6):
            probe.choices.append(Choice("reading directions", "as given", "scaled to unit length",
                                        "a probe reads with unit directions, so its scores are projections; steering "
                                        "uses the directions as given"))
        probe.steer_directions = (np.asarray(directions, np.float32) if steer_directions is None
                                  else np.asarray(steer_directions, np.float32))
        probe.layer = resolve_layer_index(layer, v.shape[0], "layer")
        probe.estimator = {"kind": "given"}
        probe.steer_estimator = {"kind": "given"}
        announce(probe.choices)
        probe._save_new(run_name, save, overwrite, "from_directions", plots=False)
        return probe

    @classmethod
    def _new(cls, name, pos_label, neg_label, model, tokenizer, identity, reps_identity, root_dir,
             allow_model_mismatch) -> "ConceptProbe":
        probe = cls(name=str(name), pos_label=str(pos_label), neg_label=str(neg_label), save_root=root_dir)
        if isinstance(identity, str):
            identity = {"model_id": identity}
        if model is not None:
            probe.attach(model, tokenizer, allow_model_mismatch=True)
            mismatched = identity_mismatches(reps_identity, probe.identity) if reps_identity else []
            if mismatched and not allow_model_mismatch:
                raise ValueError("The representations were read with another model than the one passed ("
                                 + "; ".join(mismatched) + "); pass allow_model_mismatch=True to use it anyway.")
        else:
            probe.identity = reps_identity or identity
        if not probe.identity:
            raise ValueError("Which model are these representations from? Pass what read_texts returns (it records "
                             "the model), model= (the loaded model), or identity= (a dict, or the model id).")
        return probe

    def attach(self, model: Any, tokenizer: Any = None, *, allow_model_mismatch: bool = False) -> "ConceptProbe":
        """Give the probe a loaded model (a :class:`ModelBundle`, or a model and its tokenizer), so it
        can score texts and steer. A model of another identity than the probe's raises unless
        ``allow_model_mismatch``."""
        bundle = model if isinstance(model, ModelBundle) else None
        if bundle is None:
            if tokenizer is None:
                raise ValueError("attach a ModelBundle, or a model together with its tokenizer.")
            bundle = ModelBundle.from_loaded(model, tokenizer)
        mine = bundle.identity
        if self.identity:
            mismatched = identity_mismatches(self.identity, mine)
            if mismatched and not allow_model_mismatch:
                raise ValueError("The model differs from the one this probe belongs to (probe vs model: "
                                 + "; ".join(mismatched) + "); pass allow_model_mismatch=True to use it anyway.")
            if not mismatched:  # a probe given only the model id now knows the rest of its model
                self.identity = {**mine, **self.identity}
        else:
            self.identity = mine
        self.model_bundle = bundle
        return self

    # ------------------------------------------------------------------ saving and loading
    def _save_new(self, run_name: Optional[str], save: Optional[bool], overwrite: bool, how: str,
                  plots: bool = True) -> None:
        if not (settings().save if save is None else save):
            return
        root = self.save_root or settings().output_root
        run_dir = os.path.join(root, safe_slug(self.name), run_name or now_tag())
        if os.path.isdir(run_dir) and os.listdir(run_dir):
            if not overwrite:
                raise FileExistsError(f"{run_dir} already holds files; pass overwrite=True or another run_name.")
            import shutil

            shutil.rmtree(run_dir)
        self.save(run_dir, how=how)
        if plots and self.evaluation is not None:
            self._plots(os.path.join(run_dir, "plots"))
        if settings().console:
            print(f"[concept-probe] saved the probe to {run_dir}", flush=True)

    def _plots(self, folder: str) -> None:
        ev = self.evaluation
        ensure_dir(folder)
        main = ev.main
        plot_sweep(main.d_curve, main.p_curve, os.path.join(folder, "sweep.png"), best_layer=self.layer,
                   selection_d=ev.selection_d_curve, report_d=ev.report_d_curve if ev.selection == "split" else None,
                   null_d95=ev.report_null_d95_curve, best_d=main.d,
                   title=f"{self.name}: Cohen's d per layer on the {ev.primary} items")
        rp, rn = ev.report_idx
        plot_score_hist(main.pos_scores[rp, self.layer], main.neg_scores[rn, self.layer],
                        os.path.join(folder, "score_hist.png"),
                        title=f"Scores at layer {self.layer}: {main.measured_on} of the {ev.primary} items "
                              f"({len(rp)} {self.pos_label}, {len(rn)} {self.neg_label})",
                        labels=(self.pos_label, self.neg_label))
        for set_name, result in ev.sets.items():
            if set_name != ev.primary:
                plot_sweep(result.d_curve, result.p_curve, os.path.join(folder, f"sweep_{safe_slug(set_name)}.png"),
                           best_layer=self.layer, null_d95=result.null_d95_curve, best_d=result.d,
                           title=f"{self.name}: Cohen's d per layer on the {set_name} items")

    def save(self, run_dir: Optional[str] = None, *, how: str = "save") -> str:
        """Write the probe to ``run_dir`` (default: its own): ``probe.json`` (what it is, its
        calibrations, its evaluation's numbers, the choices made) and ``probe.npz`` (its directions
        and per-layer curves). :meth:`load` reads it back."""
        self._require_trained()
        run_dir = run_dir or self.run_dir
        if not run_dir:
            raise ValueError("The probe has no folder; pass run_dir.")
        ensure_dir(run_dir)
        self.run_dir = run_dir
        arrays = {"directions": np.asarray(self.directions, dtype=np.float32)}
        if self.steer_directions is not None and self.steer_directions is not self.directions:
            arrays["steer_directions"] = np.asarray(self.steer_directions, dtype=np.float32)
        for key in ("resid_norm", "sweep_d", "sweep_p"):
            if getattr(self, key) is not None:
                arrays[key] = np.asarray(getattr(self, key), dtype=np.float64)
        np.savez_compressed(os.path.join(run_dir, "probe.npz"), **arrays)
        json_dump(os.path.join(run_dir, "probe.json"), {
            "name": self.name, "pos_label": self.pos_label, "neg_label": self.neg_label, "layer": int(self.layer),
            "score_layers": self.score_layers, "score_aggregate": self.score_aggregate, "identity": self.identity,
            "estimator": self.estimator, "steer_estimator": self.steer_estimator, "steer_sigma": self.steer_sigma,
            "calibrations": [c.to_dict() for c in self.calibrations], "metrics": self.metrics,
            "choices": [c.to_dict() for c in self.choices], "built_by": how,
        })
        return run_dir

    @classmethod
    def load(cls, run_dir: str, model: Any = None, tokenizer: Any = None, *,
             allow_model_mismatch: bool = False) -> "ConceptProbe":
        """A probe saved in ``run_dir`` (by :meth:`save`, or a training run of an earlier version),
        with ``model`` attached if given (see :meth:`attach`)."""
        if os.path.exists(os.path.join(run_dir, "probe.json")):
            obj = cls._load_probe(run_dir)
        else:
            obj = cls._load_old_run(run_dir)
        obj.run_dir = run_dir
        if model is not None:
            obj.attach(model, tokenizer, allow_model_mismatch=allow_model_mismatch)
        return obj

    @classmethod
    def _load_probe(cls, run_dir: str) -> "ConceptProbe":
        with open(os.path.join(run_dir, "probe.json"), "r", encoding="utf-8") as f:
            d = json.load(f)
        with np.load(os.path.join(run_dir, "probe.npz")) as z:
            arrays = {k: z[k] for k in z.files}
        obj = cls(name=d["name"], pos_label=d["pos_label"], neg_label=d["neg_label"])
        obj.directions = arrays["directions"]
        obj.steer_directions = arrays.get("steer_directions", obj.directions)
        obj.layer = int(d["layer"])
        obj.score_layers, obj.score_aggregate = d.get("score_layers"), d.get("score_aggregate", "mean")
        obj.identity, obj.estimator, obj.steer_estimator = d.get("identity"), d.get("estimator"), d.get("steer_estimator")
        obj.steer_sigma = d.get("steer_sigma")
        obj.calibrations = [Calibration.from_dict(c) for c in d.get("calibrations", [])]
        obj.metrics = d.get("metrics") or {}
        obj.choices = [Choice(**c) for c in d.get("choices", [])]
        obj.resid_norm, obj.sweep_d, obj.sweep_p = (arrays.get(k) for k in ("resid_norm", "sweep_d", "sweep_p"))
        return obj

    @classmethod
    def _load_old_run(cls, run_dir: str) -> "ConceptProbe":
        """A training run of an earlier version: ``config.json``, ``concept.json``, ``metrics.json``
        and ``tensors.npz``, in the names of then."""
        with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = upgrade_saved_config(json.loads(f.read()))
        with open(os.path.join(run_dir, "concept.json"), "r", encoding="utf-8") as f:
            concept = json.loads(f.read())
        metrics: Dict[str, Any] = {}
        if os.path.exists(os.path.join(run_dir, "metrics.json")):
            with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.loads(f.read())
        with np.load(os.path.join(run_dir, "tensors.npz")) as t:
            tensors = {k: t[k] for k in t.files}
        obj = cls(name=concept.get("name", "concept"), pos_label=concept.get("pos_label", "positive"),
                  neg_label=concept.get("neg_label", "negative"))
        obj.directions = tensors["concept_vectors"]
        obj.layer = int(tensors["best_layer"][0])
        obj.sweep_d, obj.sweep_p = tensors.get("sweep_d"), tensors.get("sweep_p")
        obj.resid_norm = tensors.get("resid_norm")
        obj.calibrations = calibrations_from_tensors(tensors)
        obj.identity = metrics.get("model_identity") or identity_from_config(cfg.get("model", {}))
        obj.estimator = metrics.get("estimator") or estimator_spec(cfg.get("training", {}).get("estimator", "mean_diff"))
        evaluation = cfg.get("evaluation", {})
        selection = evaluation.get("layer_selection") or {"mode": "best"}
        num_layers = int(np.asarray(obj.directions).shape[0])
        mode = selection.get("mode", "best")
        if mode == "window":
            obj.score_layers = _window(obj.layer, selection.get("window_radius", 0), num_layers)[0]
        elif mode == "explicit":
            obj.score_layers = _layer_list(selection.get("layers", []), num_layers, "layer_selection.layers")
        elif mode == "all":
            obj.score_layers = list(range(num_layers))
        obj.score_aggregate = evaluation.get("score_aggregate", "mean")
        obj.metrics = {k: metrics.get(k) for k in ("best_layer", "best_d", "best_p", "best_d_ci95", "best_auroc",
                                                   "null_d95_at_best_layer", "selection") if k in metrics}
        if "steer_vectors" in tensors:
            obj.steer_directions = tensors["steer_vectors"]
            if np.array_equal(obj.steer_directions, obj.directions):
                obj.steer_directions = obj.directions
            sigma = float(tensors["steer_sigma_best_layer"][0])
            obj.steer_estimator = metrics.get("steer_direction") or obj.estimator
        else:
            # Saved before steering had its own direction: it steers with the one it reads with.
            obj.steer_directions, obj.steer_estimator = obj.directions, obj.estimator
            sigma = float(tensors["score_scale"][obj.layer]) if "score_scale" in tensors else float("nan")
            if obj.estimator != {"kind": "mean_diff", "normalization": "none"}:
                obj.choices.append(Choice(
                    "steering direction", "mean_diff", "the run's reading direction",
                    f"this run ({obj.estimator}) was saved before steering had its own direction, so it steers with "
                    "the one it reads with; refit it to steer with the raw difference of means"))
                announce(obj.choices)
        obj.steer_sigma = sigma if np.isfinite(sigma) else None
        return obj

    # ------------------------------------------------------------------ reading arrays
    @property
    def num_layers(self) -> int:
        self._require_trained()
        return int(np.asarray(self.directions).shape[0])

    def _score_layers(self) -> List[int]:
        return [int(self.layer)] if not self.score_layers else [int(l) for l in self.score_layers]

    def project(self, reps: Any) -> np.ndarray:
        """Each row's projection on each layer's reading direction: ``(rows, layers)``, from
        representations ``(rows, layers, dim)`` (an array, or what :func:`read_texts` returns)."""
        self._require_trained()
        R = _as_array(reps)
        if R.shape[1:] != np.asarray(self.directions).shape:
            raise ValueError(f"representations of shape {R.shape[1:]} (layers, dim) do not fit the probe's directions "
                             f"{np.asarray(self.directions).shape}; read every layer")
        return np.einsum("nld,ld->nl", R.astype(np.float64), np.asarray(self.directions, dtype=np.float64))

    def scores(self, reps: Any, *, per_text: Optional[str] = None) -> np.ndarray:
        """Each item's raw score: its projection at the probe's score layers (the probe's layer by
        default), combined as ``score_aggregate``. The offset is arbitrary; see :meth:`z`.

        For representations read token by token (a ``*_tokens`` readout), ``per_text`` summarizes
        each text's token scores into one (see :func:`summarize_tokens`: ``"mean"``, ``"max"``,
        ``"min"``, ``"median"``, ``"p90"``, ``"extreme"``); without it, every token gets a score."""
        layers = self._score_layers()
        proj = self.project(reps)[:, layers]
        out = proj.mean(axis=1) if self.score_aggregate == "mean" else proj.sum(axis=1)
        return self._per_text(out, reps, per_text, 0.0)

    def _per_text(self, values: np.ndarray, reps: Any, per_text: Optional[str], center: float) -> np.ndarray:
        if per_text is None:
            return values
        items = getattr(reps, "items", None)
        if items is None or getattr(getattr(reps, "readout", None), "agg", None) != "tokens":
            raise ValueError("per_text summarizes token scores: read the texts with a *_tokens readout")
        return summarize_tokens(values, items, per_text, center)[0]

    def z(self, reps: Any, context: Optional[Union[ReadContext, Dict[str, Any]]] = None, *, unit: str = "sd",
          per_text: Optional[str] = None) -> np.ndarray:
        """Each item's z score, with the calibration of how the items were read (``context``; by
        default what the representations say, or the only calibration). ``unit="sd"``: 0 is the
        calibration's zero (the midpoint between the poles, which is not "without the concept"; or
        the baseline's mean), 1 its within-pole SD. ``unit="poles"``: 0 the same zero, +1 the
        positive pole's mean (and -1 the negative's, with a midpoint). ``per_text`` summarizes the
        z of each text's tokens (see :meth:`scores`; single tokens get the calibration's per-item
        references). Raises when there is no calibration for that reading, or when the items were
        pooled another way than the calibration's items."""
        if unit not in Z_UNITS:
            raise ValueError(f"unit must be one of {Z_UNITS}; got {unit!r}")
        cal = self._calibration_of(reps, context, tokens=per_text is not None)
        layers = self._score_layers()
        z = cal.items_z(self.project(reps)[:, layers], layers, self.score_aggregate, unit=unit)
        if z is None:
            raise ValueError(f"the calibration for items read {cal.context.describe()} has no usable references "
                             f"{'in poles units ' if unit == 'poles' else ''}at layers {layers}.")
        return self._per_text(z, reps, per_text, 0.0)

    def _calibration_of(self, reps: Any, context: Any, tokens: bool = False) -> Calibration:
        """The calibration for ``reps``: of ``context``, or of the context the reps record; and the
        check that they were pooled like the calibration's items."""
        ctx = context if context is not None else getattr(reps, "context", None)
        cal = self.calibration_for(ctx)
        if cal is None and context is None and len(self.calibrations) == 1 and ctx is not None:
            raise ValueError(self._missing_z_reason(ctx))
        if cal is None:
            raise ValueError(self._missing_z_reason(ctx))
        readout = getattr(getattr(reps, "readout", None), "name", None)
        if readout is not None and not tokens and cal.context.readout and readout != cal.context.readout:
            raise ValueError(f"these items were read with {readout!r} and the calibration's with "
                             f"{cal.context.readout!r}: their z would not be comparable; calibrate on items read the "
                             "same way (probe.calibrate).")
        return cal

    def classify(self, reps: Any, context: Optional[Union[ReadContext, Dict[str, Any]]] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Whether each item belongs to the positive pole, and the probability, from the calibrated
        threshold (not from z = 0, which is the midpoint, not the best cut)."""
        cal = self._calibration_of(reps, context)
        if not cal.threshold:
            raise ValueError("That calibration has no threshold; calibrate again to fit one.")
        t = cal.threshold
        prob = sigmoid(t["slope"] * self.project(reps)[:, int(t["layer"])] + t["intercept"])
        return prob >= 0.5, prob

    def accuracy(self, pos: Any, neg: Any, *, cut: str = "threshold",
                 context: Optional[Union[ReadContext, Dict[str, Any]]] = None, n_bootstrap: int = 1000,
                 paired: bool = False, seed: int = 0) -> Dict[str, Any]:
        """The share of items of each pole on the right side of a cut, with a bootstrap 95% interval:
        ``cut="threshold"`` (the calibrated threshold) or ``"zero"`` (z = 0: the midpoint between
        the poles of the calibration, or the baseline)."""
        if cut == "threshold":
            right = np.r_[self.classify(pos, context)[0], ~self.classify(neg, context)[0]]
        elif cut == "zero":
            right = np.r_[self.z(pos, context) > 0, self.z(neg, context) <= 0]
        else:
            raise ValueError(f"cut must be 'threshold' or 'zero'; got {cut!r}")
        n_pos = len(_as_array(pos))
        a, b = right[:n_pos].astype(float), right[n_pos:].astype(float)
        share = lambda x, y: float(np.r_[x, y].mean())  # noqa: E731
        ci = bootstrap_ci(share, a, b, int(n_bootstrap), np.random.default_rng(seed), paired=paired)
        return {"accuracy": float(right.mean()), "ci95": ci, "cut": cut, "n": int(len(right))}

    # ------------------------------------------------------------------ calibrations
    def calibration_for(self, context: Optional[Union[ReadContext, Dict[str, Any]]] = None) -> Optional[Calibration]:
        """The calibration measured on items read like ``context`` (a :class:`ReadContext` or its
        dict); with no context, the only calibration if there is one."""
        if context is None:
            return self.calibrations[0] if len(self.calibrations) == 1 else None
        target = context if isinstance(context, ReadContext) else ReadContext.from_dict(context)
        return next((c for c in self.calibrations if c.context.matches(target)), None)

    def _missing_z_reason(self, context: Optional[Union[ReadContext, Dict[str, Any]]]) -> str:
        have = "; ".join(c.context.describe() for c in self.calibrations) or "in no context"
        if context is None:
            return (f"the probe has {len(self.calibrations)} calibrations (items read {have}); say which reading "
                    "context the items come from (context=...).")
        target = context if isinstance(context, ReadContext) else ReadContext.from_dict(context)
        return (f"the probe has no calibration for items read {target.describe()} (it has one for items read "
                f"{have}); z scores from another reading context are not comparable. Add one with "
                "probe.calibrate(pos, neg, ...).")

    def calibrate(
        self,
        pos: Any,
        neg: Any,
        *,
        context: Optional[Union[ReadContext, Dict[str, Any]]] = None,
        as_reply_to: Optional[Union[str, List[str]]] = None,
        role: str = "raw",
        system: Optional[str] = None,
        prompt: Optional[str] = None,
        neutral: Any = None,
        baseline: bool = False,
        threshold_l2: float = 1e-2,
        batch_size: int = 16,
    ) -> Calibration:
        """Measure a calibration on items of each pole, so items read the same way get z scores and
        a calibrated threshold. It replaces any calibration for the same reading context.

        ``pos`` / ``neg`` (and ``neutral``, texts without the concept, whose place is recorded) are
        representations already read, with ``context`` saying how (default: what they record, or raw
        text); or texts, which the probe reads with its model: as replies to ``as_reply_to`` (one
        user turn, or one per text), after ``system`` if given, which is how generated completions
        are read; or with ``role`` / ``system`` / ``prompt``, as :meth:`score_texts` reads them.
        Texts also give single tokens their references. ``baseline=True`` sets the zero at ``neg``'s
        mean instead of the midpoint. Call :meth:`save` to keep it.
        """
        self._require_trained()
        if isinstance(pos, np.ndarray) or hasattr(pos, "data"):
            ctx = context if context is not None else getattr(pos, "context", None)
            ctx = ReadContext() if ctx is None else (ctx if isinstance(ctx, ReadContext) else ReadContext.from_dict(ctx))
            scores = {"pos": self.project(pos), "neg": self.project(neg)}
            if neutral is not None:
                scores["neutral"] = self.project(neutral)
            tokens: Dict[str, Optional[List[np.ndarray]]] = {"pos": None, "neg": None}
        else:
            ctx, scores, tokens = self._read_for_calibration(pos, neg, neutral, as_reply_to, role, system, prompt,
                                                             batch_size)
        cal = Calibration.from_scores(scores["pos"], scores["neg"], ctx, pos_tokens=tokens["pos"],
                                      neg_tokens=tokens["neg"], zero="baseline" if baseline else "midpoint",
                                      source="calibrate()")
        layer = int(self.layer)
        cal.add_threshold(scores["pos"][:, layer], scores["neg"][:, layer], layer, "the calibration items",
                          l2=threshold_l2)
        if "neutral" in scores:
            cal.add_neutral(scores["neutral"])
        self.calibrations = [c for c in self.calibrations if not c.context.matches(ctx)] + [cal]
        return cal

    def _read_for_calibration(self, pos_texts, neg_texts, neutral_texts, as_reply_to, role, system, prompt,
                              batch_size):
        self._require_model("calibrating on texts")
        tokenizer, model = self.model_bundle.tokenizer, self.model_bundle.model

        def encode(texts: List[PromptLike], label: str) -> List[Encoded]:
            if as_reply_to is None:
                return [encode_text(tokenizer, t, system, role=role, prompt=prompt, warn=_warn,
                                    warn_prefix=f"calibrate.{label}[{i}]") for i, t in enumerate(texts)]
            turns = [as_reply_to] * len(texts) if isinstance(as_reply_to, str) else list(as_reply_to)
            if len(turns) != len(texts):
                raise ValueError(f"as_reply_to has {len(turns)} turns for {len(texts)} {label} texts.")
            out = []
            for turn, text in zip(turns, texts):
                messages, _ = _normalize_messages(turn, system)
                ids, plen = _completion_input_ids(tokenizer, messages, str(text))
                out.append(Encoded(ids[0], plen, int(ids.shape[-1])))
            return out

        if role not in TEXT_ROLES:
            raise ValueError(f"role must be one of {TEXT_ROLES}; got {role!r}")
        ctx = (ReadContext.generated("text_mean", system) if as_reply_to is not None else
               ReadContext(kind="text", role=role, system=None if role == "raw" else system,
                           prompt=prompt if role == "assistant" else None, readout="text_mean"))
        scores, tokens = {}, {}
        for label, texts in (("pos", pos_texts), ("neg", neg_texts), ("neutral", neutral_texts or [])):
            if not texts:
                continue
            reps = _read(model, tokenizer, encode(list(texts), label), [Readout.parse("text_mean")],
                         batch_size=batch_size, project_on=self.directions, cache=False,
                         label=f"calibrate.{label}")[0]
            announce(reps.choices)
            scores[label] = np.einsum("nld,ld->nl", reps.data, self.directions)
            tokens[label] = reps.projections
        return ctx, scores, tokens

    # ------------------------------------------------------------------ with the model
    def _token_scorer(self, context: Optional[Union[ReadContext, Dict[str, Any]]] = None, *,
                      vectors: Optional[np.ndarray] = None, reference_layer: Optional[int] = None) -> TokenScorer:
        """How this probe scores tokens read in ``context``: its directions at its score layers,
        combined as ``score_aggregate``, with the calibration for that context. ``vectors`` replaces
        the directions (a random control: then there is no calibration); ``reference_layer`` reads
        every layer with that one layer's direction (a logit-lens-style read, also without one)."""
        layers = self._score_layers()
        V = np.asarray(self.directions if vectors is None else vectors, dtype=np.float32)
        if reference_layer is not None:
            ref = resolve_layer_index(reference_layer, V.shape[0], "reference_layer")
            rows = np.stack([V[ref] for _ in layers])
        else:
            rows = V[layers]
        own = vectors is None and reference_layer is None
        cal = self.calibration_for(context) if own and context is not None else None
        return TokenScorer(str(self.name), list(layers), rows, self.score_aggregate, cal)

    def score_texts(
        self,
        texts: List[PromptLike],
        *,
        role: str = "raw",
        system: Optional[str] = None,
        prompt: Optional[str] = None,
        reference_layer: Optional[int] = None,
        batch_size: int = 16,
        save: Optional[bool] = None,
        out_dir: Optional[str] = None,
        html: bool = True,
    ) -> Scores:
        """Score every token of each text with the probe, in batches; returns
        :class:`~concept_probe.scoring.Scores`.

        Each text is read as :func:`concept_probe.read_texts` reads it (``role``, ``system``,
        ``prompt``; by default alone). Each item has raw projections per token and layer, their
        combination per token and, where the probe has a calibration for texts read this way,
        single-token z scores (0 the midpoint between the poles, 1 the within-pole SD of single
        tokens). ``score_mean`` is the text's mean token score and ``score_mean_z`` its z with the
        per-item references; without such a calibration only raw projections are given, and
        ``z_missing`` says why. ``reference_layer`` reads every layer with one layer's direction.
        ``Scores.per_text`` summarizes each text's tokens another way.

        Saved by default (``save``, or the session's setting) to ``out_dir``, or ``scores/<time>``
        in the probe's folder (or the session's ``output_root``): ``scores.npz``, ``scores.jsonl``,
        ``heatmaps.html`` (unless ``html=False``) and ``settings.json``.
        """
        self._require_model("score_texts")
        tokenizer, model = self.model_bundle.tokenizer, self.model_bundle.model
        if role not in TEXT_ROLES:
            raise ValueError(f"role must be one of {TEXT_ROLES}; got {role!r}")
        context = ReadContext(kind="text", role=role, system=None if role == "raw" else system,
                              prompt=prompt if role == "assistant" else None, readout="text_mean")
        scorer = self._token_scorer(context, reference_layer=reference_layer)
        z_missing = None
        if reference_layer is not None:
            z_missing = "read with one layer's direction"
        elif scorer.calibration is None:
            z_missing = self._missing_z_reason(context)
            _warn(f"score_texts: only raw projections are given: {z_missing}")
        elif (scorer.calibration.context.readout or "text_mean") != "text_mean":
            z_missing = (f"the calibration's items were pooled with {scorer.calibration.context.readout!r}, so a "
                         "text's mean token score has no z (single tokens still do)")
        inputs = []
        for i, text in enumerate(texts):
            enc = encode_text(tokenizer, text, system, role=role, prompt=prompt, warn=_warn,
                              warn_prefix=f"score_texts.texts[{i}]")
            span = (0, 0) if enc.start is None else (int(enc.start), int(enc.end))
            info = {"input": text, "role": role, "system": system,
                    "input_text": tokenizer.decode(enc.input_ids.tolist(), skip_special_tokens=False),
                    "text_span": None if enc.start is None else [int(enc.start), int(enc.end)]}
            inputs.append(ScoreInput(enc.input_ids, span, _prompt_to_text(text), i, info))
        items = score_inputs(model, tokenizer, inputs, [scorer], batch_size=int(batch_size))
        run_settings = {"role": role, "system": system, "prompt": prompt, "layers": scorer.layers,
                        "aggregate": scorer.aggregate, "reference_layer": reference_layer, "batch_size": int(batch_size),
                        "calibration": None if scorer.calibration is None else scorer.calibration.context.to_dict()}
        result = Scores("texts", [scorer.name], items, run_settings, {scorer.name: z_missing})
        if settings().save if save is None else save:
            if out_dir is None:
                root = self.run_dir or os.path.join(self.save_root or settings().output_root, safe_slug(self.name))
                out_dir = os.path.join(root, "scores", now_tag())
            ensure_dir(out_dir)
            result.save(out_dir, html=html)
        return result

    def steering(
        self,
        alpha: float,
        unit: str = "raw",
        *,
        layers: Optional[Sequence[int]] = None,
        window: Optional[int] = None,
        distribute: bool = False,
        positions: Any = None,
        vectors: Optional[np.ndarray] = None,
    ) -> Add:
        """The intervention that steers with this probe: an :class:`~concept_probe.interventions.Add`
        of ``alpha_l * v_l`` at each steered layer ``l``, where ``v_l`` is the probe's steering
        direction at that layer (``steer_directions``; decision D1) and ``alpha`` is in ``unit``:

        - ``"raw"``: as is;
        - ``"sigma"``: times ``steer_sigma``, the eval items' pooled within-pole SD on the steering
          direction at the probe's layer (a scale knob, not "n standard deviations of behavior");
        - ``"resid_norm"``: times each layer's mean residual norm on the training texts, so 0.1
          adds 10% of a typical token's norm.

        It steers at the probe's layer, at ``layers``, or at the layers within ``window`` of it (a
        window that reaches past the model is clipped, and that is announced). With several layers,
        each gets the whole dose unless ``distribute=True`` divides it among them. ``positions``
        restricts it to some tokens (see :mod:`concept_probe.interventions`), ``vectors`` replaces
        the directions (a random control). Use it with :func:`concept_probe.intervene`,
        :func:`concept_probe.generate` or :func:`concept_probe.answer_logprobs`.
        """
        self._require_trained()
        if unit not in ALPHA_UNITS:
            raise ValueError(f"unit must be one of {ALPHA_UNITS}; got {unit!r}")
        steer_layers = self.steer_layer_list(layers, window)
        if abs(float(alpha)) > 1e-12:
            self._check_alpha_unit(unit)
        value = float(alpha)
        if unit == "sigma" and abs(value) > 1e-12:
            value *= float(self.steer_sigma)
        per = value / len(steer_layers) if distribute else value
        alphas = {l: float(per * (float(self.resid_norm[l]) if unit == "resid_norm" else 1.0)) for l in steer_layers}
        V = self.steer_directions if vectors is None else vectors
        return Add({l: np.asarray(V[l], dtype=np.float32) for l in steer_layers}, alpha=alphas, positions=positions,
                   label=f"steering with {self.name!r}: alpha {float(alpha):g} ({unit})")

    def steer_layer_list(self, layers: Optional[Sequence[int]] = None, window: Optional[int] = None) -> List[int]:
        """The layers :meth:`steering` steers: the probe's layer, ``layers``, or the layers within
        ``window`` of the probe's layer."""
        self._require_trained()
        if layers is not None and window is not None:
            raise ValueError("give layers or window, not both")
        if window is not None:
            out, clipped = _window(int(self.layer), window, self.num_layers)
            if clipped is not None:
                announce([clipped])
            return out
        if layers is None:
            return [int(self.layer)]
        return _layer_list(list(layers) if not isinstance(layers, (int, np.integer)) else layers, self.num_layers,
                           "layers")

    def _check_alpha_unit(self, unit: str) -> None:
        """Raise if the probe lacks the reference that ``unit`` scales by."""
        if unit == "sigma" and (self.steer_sigma is None or not np.isfinite(self.steer_sigma)):
            raise ValueError("unit='sigma' needs the probe's steering unit (steer_sigma), which comes from evaluation "
                             "or calibration items; this probe has none.")
        if unit == "resid_norm" and self.resid_norm is None:
            raise ValueError("unit='resid_norm' needs the mean residual norm per layer: fit the probe on texts read "
                             "with norms=True, or set probe.resid_norm (evaluate_steering measures it on the prompts).")

    def describe(self) -> Dict[str, Any]:
        """What the probe is, in a dict: its concept, layer, estimators, calibrations and numbers."""
        self._require_trained()
        return {"name": self.name, "poles": [self.pos_label, self.neg_label], "layer": int(self.layer),
                "score_layers": self._score_layers(), "estimator": self.estimator,
                "steer_estimator": self.steer_estimator, "steer_sigma": self.steer_sigma, "identity": self.identity,
                "calibrations": [c.context.describe() for c in self.calibrations], "metrics": self.metrics,
                "run_dir": self.run_dir}

    # ------------------------------------------------------------------ checks
    def _require_trained(self) -> None:
        if self.directions is None or self.layer is None:
            raise ValueError("The probe has no directions yet: fit it, or load one.")

    def _require_model(self, what: str) -> None:
        self._require_trained()
        if self.model_bundle is None:
            raise ValueError(f"{what} needs the model loaded: attach it with probe.attach(model, tokenizer) "
                             "(the probe's identity says which model it belongs to).")


# ------------------------------------------------------------------------------------- fit helpers
def _warn(message: str) -> None:
    import warnings

    warnings.warn(message, stacklevel=3)


def _token_projections(s: EvalSet, directions: np.ndarray) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """Each item's per-token projections ``(tokens, layers)`` of a set read token by token, per pole."""
    if s.tokens is None:
        return None, None
    out = []
    for reps in s.tokens:
        data, items = _as_array(reps), np.asarray(reps.items)
        proj = np.einsum("nld,ld->nl", data.astype(np.float64), directions.astype(np.float64))
        out.append([proj[items == i] for i in dict.fromkeys(items.tolist())])
    return out[0], out[1]


def _record_estimator_notes(probe: ConceptProbe, est: Any, flips: List[int], choices: List[Choice]) -> None:
    """Record what the estimator did on the caller's behalf: directions turned around so the positive
    pole projects higher, and coordinates an outlier rule left out."""
    if flips:
        probe.estimator = dict(probe.estimator, flipped_layers=sorted(flips))
        choices.append(Choice("direction sign", "as the estimator gave it", "turned around",
                              f"at layers {sorted(flips)} the positive pole projected lower, so those directions were "
                              "turned around (a direction points toward the positive pole)"))
    dropped = list(getattr(est, "dropped", None) or [])
    if dropped:
        probe.estimator = dict(probe.estimator, dropped_coordinates=list(dropped))
        choices.append(Choice("outlier coordinates", "every coordinate", "some left out",
                              f"the outlier rule left out {sum(dropped)} coordinates over {len(dropped)} layers "
                              f"(per layer: {list(dropped)})"))


def _metrics(ev: ReadingEvaluation, shuffle: Optional[Dict[str, Any]], neutral: Optional[Dict[str, Any]],
             n_pos: int, n_neg: int) -> Dict[str, Any]:
    """The numbers a fitted probe saves: its evaluation's, the label-shuffle control and the neutral texts."""
    main = ev.main
    return {
        "layer": int(ev.layer), "d": main.d, "p": main.p, "d_ci95": main.d_ci95, "auroc": main.auroc,
        "auroc_ci95": main.auroc_ci95, "null_d95": main.null_d95, "accuracy": main.accuracy,
        "accuracy_ci95": main.accuracy_ci95, "d_all_eval": None if not np.isfinite(main.d_curve[ev.layer])
        else float(main.d_curve[ev.layer]), "n_train": {"pos": int(n_pos), "neg": int(n_neg)},
        "evaluation": ev.to_metrics(), "label_shuffle": shuffle, "neutral": neutral,
    }
