"""Scoring texts and generations with probes, in memory, and saving the scores (layers 5 and 7).

:func:`score_inputs` reads each input once (in batches) and projects every token on each probe's
directions; :class:`Scores` holds the result of one call (one :class:`ScoredItem` per text or
generation) and saves it as three files in one folder:

- ``scores.npz``: every item's tokens and per-token scores (per layer, per probe, and z);
- ``scores.jsonl``: one readable line per item (prompt, steering dose, text, mean score, z);
- ``heatmaps.html``: the heatmaps of all items, with a menu to pick the probe;

plus ``settings.json`` with what the call used. :meth:`Scores.load` reads them back.
``ConceptProbe.score_texts`` and :func:`concept_probe.evaluate_steering` are built on these pieces.
"""
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .generation import TokenScorer, TokenStep, _project_tokens, generate
from .interventions import Intervention
from .inputs import _decode_tokens
from .utils import ensure_dir, json_dump, now_tag, set_seed
from .visuals import render_heatmaps, segment_token_scores

SCORES_NPZ, SCORES_JSONL, HEATMAPS_HTML, SETTINGS_JSON = "scores.npz", "scores.jsonl", "heatmaps.html", "settings.json"


@dataclass
class ScoredItem:
    """One scored text or generation.

    ``span`` is the item's own tokens (the text, or the completion without its end-of-sequence
    token). Per probe name: ``scores`` are the raw projections of every token at the probe's
    scoring ``layers`` ``(tokens, layers)``; ``token_scores`` combines them per token; ``token_z``
    standardizes them with the probe's single-token references (None without a calibration for how
    the item was read); ``score_mean`` is the mean token score over the span and ``score_mean_z``
    its z with the per-item references. ``info`` has the rest: the prompt, the steering dose, the
    intervention, the generation settings.
    """

    index: int
    text: str
    token_ids: np.ndarray
    tokens: List[str]
    span: Tuple[int, int]
    layers: Dict[str, List[int]]
    scores: Dict[str, np.ndarray]
    token_scores: Dict[str, np.ndarray]
    token_z: Dict[str, Optional[np.ndarray]]
    score_mean: Dict[str, Optional[float]]
    score_mean_z: Dict[str, Optional[float]]
    info: Dict[str, Any] = field(default_factory=dict)
    logits: Optional[Dict[str, np.ndarray]] = None

    def record(self) -> Dict[str, Any]:
        """The item's line of ``scores.jsonl``."""
        out = {"index": self.index, "text": self.text, "span": [int(self.span[0]), int(self.span[1])],
               "n_tokens": int(len(self.token_ids)), "score_mean": dict(self.score_mean),
               "score_mean_z": dict(self.score_mean_z)}
        out.update(self.info)
        return out


def _single(probes: List[str], probe: Optional[str]) -> str:
    if probe is not None:
        if probe not in probes:
            raise KeyError(f"no probe named {probe!r}; the scores have {probes}")
        return probe
    if len(probes) != 1:
        raise ValueError(f"these scores have several probes ({probes}); say which one (probe=...)")
    return probes[0]


@dataclass
class Scores:
    """The scores of one call: one :class:`ScoredItem` per text or generation, in order.

    ``kind`` is ``"texts"`` or ``"generations"``; ``probes`` the probes' names; ``z_missing[name]``
    why a probe has no z for these items (None when it has); ``dir`` the folder the scores were
    saved to (None when not saved).
    """

    kind: str
    probes: List[str]
    items: List[ScoredItem]
    settings: Dict[str, Any] = field(default_factory=dict)
    z_missing: Dict[str, Optional[str]] = field(default_factory=dict)
    dir: Optional[str] = None

    def per_text(self, how: str = "mean", probe: Optional[str] = None, *, z: bool = False) -> np.ndarray:
        """One number per item from its own tokens' scores (the text, or the completion): their
        ``"mean"``, ``"max"``, ``"min"``, ``"median"``, a percentile ``"p<k>"``, or the most
        ``"extreme"`` token (farthest from 0, with its sign). ``z=True`` uses the single-token z
        scores (where the probe has them) instead of the raw ones."""
        from .stats import summarize_tokens

        name = _single(self.probes, probe)
        out = []
        for it in self.items:
            values = it.token_z[name] if z else it.token_scores[name]
            if values is None:
                raise ValueError(f"no z scores for {name!r}: {self.z_missing.get(name)}")
            own = np.asarray(values, dtype=np.float64)[it.span[0]:it.span[1]]
            out.append(summarize_tokens(own, np.zeros(len(own), dtype=int), how)[0][0] if len(own) else np.nan)
        return np.asarray(out, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[ScoredItem]:
        return iter(self.items)

    def __getitem__(self, i: int) -> ScoredItem:
        return self.items[i]

    def means(self, probe: Optional[str] = None) -> np.ndarray:
        """Each item's mean token score for ``probe`` (needed when there are several); NaN where empty."""
        name = _single(self.probes, probe)
        return np.array([np.nan if it.score_mean[name] is None else it.score_mean[name] for it in self.items])

    def z(self, probe: Optional[str] = None) -> np.ndarray:
        """Each item's z for ``probe``; NaN where there is none (see ``z_missing``)."""
        name = _single(self.probes, probe)
        return np.array([np.nan if it.score_mean_z[name] is None else it.score_mean_z[name] for it in self.items])

    def column(self, key: str) -> List[Any]:
        """``info[key]`` of every item (e.g. ``"alpha"`` or ``"prompt"``), to group or filter them."""
        return [it.info.get(key) for it in self.items]

    def records(self) -> List[Dict[str, Any]]:
        """The lines of ``scores.jsonl``."""
        return [it.record() for it in self.items]

    # ------------------------------------------------------------------------------ saving
    def save(self, out_dir: str, *, html: bool = True, segments: bool = False) -> str:
        """Write the scores to ``out_dir``. ``segments`` adds to each line the mean score of each
        message of the chat and of the completion (see :func:`concept_probe.visuals.segment_token_scores`)."""
        ensure_dir(out_dir)
        arrays: Dict[str, np.ndarray] = {
            "probe_names": np.array(self.probes),
            "token_ids": (np.concatenate([it.token_ids for it in self.items]).astype(np.int64) if self.items
                          else np.zeros(0, np.int64)),
            "offsets": np.cumsum([0] + [len(it.token_ids) for it in self.items]).astype(np.int64),
            "tokens": np.array([t for it in self.items for t in it.tokens], dtype=str),
            "span": np.array([it.span for it in self.items], dtype=np.int64).reshape(-1, 2),
        }
        for k, name in enumerate(self.probes):
            def cat(get, width=None):
                parts = [get(it) for it in self.items]
                return np.concatenate(parts) if parts else np.zeros((0,) if width is None else (0, width), np.float32)
            n_layers = len(self.items[0].layers[name]) if self.items else 0
            arrays[f"layers_{k}"] = np.array(self.items[0].layers[name] if self.items else [], dtype=np.int64)
            arrays[f"scores_{k}"] = cat(lambda it: it.scores[name], n_layers).astype(np.float32)
            arrays[f"token_scores_{k}"] = cat(lambda it: it.token_scores[name]).astype(np.float32)
            arrays[f"token_z_{k}"] = cat(lambda it: (np.full(len(it.token_ids), np.nan, np.float32)
                                                     if it.token_z[name] is None else it.token_z[name]))
        if self.items and all(it.logits is not None for it in self.items):
            arrays["logits_offsets"] = np.cumsum([0] + [len(it.logits["values"]) for it in self.items]).astype(np.int64)
            for key in self.items[0].logits:
                arrays[f"logits_{key}"] = np.concatenate([it.logits[key] for it in self.items])
        np.savez_compressed(os.path.join(out_dir, SCORES_NPZ), **arrays)
        with open(os.path.join(out_dir, SCORES_JSONL), "w", encoding="utf-8") as f:
            for it in self.items:
                rec = it.record()
                if segments:
                    rec["segments"] = {name: segment_token_scores(
                        it.tokens, (it.token_scores[name] if it.token_z[name] is None else it.token_z[name]).tolist(),
                        prompt_len=it.span[0] if self.kind == "generations" else None) for name in self.probes}
                f.write(json.dumps(rec, ensure_ascii=False, default=_plain) + "\n")
        json_dump(os.path.join(out_dir, SETTINGS_JSON), {"kind": self.kind, "probes": self.probes,
                                                          "z_missing": self.z_missing, "settings": self.settings})
        if html and self.items:
            self.heatmaps(os.path.join(out_dir, HEATMAPS_HTML))
        self.dir = out_dir
        return out_dir

    def heatmaps(self, path: str, title: Optional[str] = None) -> str:
        """One HTML page with every item's tokens colored by its score: token z where the probe has
        it, the raw projection otherwise (the menu says which)."""
        entries = []
        for it in self.items:
            by_probe = {}
            for name in self.probes:
                z = it.token_z[name]
                label = f"{name} (z: 0 = midpoint of the poles)" if z is not None else f"{name} (raw projection)"
                by_probe[label] = (it.token_scores[name] if z is None else z).tolist()
            entries.append((_entry_title(it), it.tokens, by_probe))
        render_heatmaps(entries, path, title=title or f"Token scores ({self.kind})")
        return path

    @classmethod
    def load(cls, path: str) -> "Scores":
        """Scores saved by :meth:`save`."""
        meta = json.loads(open(os.path.join(path, SETTINGS_JSON), encoding="utf-8").read())
        lines = [json.loads(l) for l in open(os.path.join(path, SCORES_JSONL), encoding="utf-8") if l.strip()]
        data = np.load(os.path.join(path, SCORES_NPZ))
        probes = [str(p) for p in data["probe_names"]]
        offsets, ids = data["offsets"], data["token_ids"]
        has_logits = "logits_offsets" in data.files
        items = []
        for i, rec in enumerate(lines):
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            token_ids = ids[lo:hi]
            info = {k: v for k, v in rec.items() if k not in ("index", "text", "span", "n_tokens", "score_mean",
                                                                "score_mean_z", "segments")}
            item = ScoredItem(index=int(rec["index"]), text=rec["text"], token_ids=token_ids,
                              tokens=[str(t) for t in data["tokens"][lo:hi]],
                              span=(int(rec["span"][0]), int(rec["span"][1])), layers={}, scores={}, token_scores={},
                              token_z={}, score_mean=dict(rec["score_mean"]), score_mean_z=dict(rec["score_mean_z"]),
                              info=info)
            for k, name in enumerate(probes):
                item.layers[name] = [int(l) for l in data[f"layers_{k}"]]
                item.scores[name] = data[f"scores_{k}"][lo:hi]
                item.token_scores[name] = data[f"token_scores_{k}"][lo:hi]
                z = data[f"token_z_{k}"][lo:hi]
                item.token_z[name] = None if np.all(np.isnan(z)) else z
            if has_logits:
                a, b = int(data["logits_offsets"][i]), int(data["logits_offsets"][i + 1])
                item.logits = {k[len("logits_"):]: data[k][a:b] for k in data.files
                               if k.startswith("logits_") and k != "logits_offsets"}
            items.append(item)
        return cls(kind=meta["kind"], probes=probes, items=items, settings=meta.get("settings", {}),
                   z_missing=meta.get("z_missing", {}), dir=path)


def _plain(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _entry_title(it: ScoredItem) -> str:
    info = it.info
    head = info.get("prompt", it.text)
    head = head if isinstance(head, str) else " | ".join(str(m.get("content", "")) for m in head)
    head = head if len(head) <= 60 else head[:57] + "..."
    parts = [f"{it.index:03d}", head]
    if info.get("alpha") is not None:
        parts.append(f"alpha {info['alpha']:g} ({info.get('alpha_unit', 'raw')})")
    if info.get("control") is not None:
        parts.append(f"control {info['control']}")
    return " | ".join(parts)


# ------------------------------------------------------------------------------------- scoring
@dataclass
class ScoreInput:
    """One input to score: its token ids, the item's own span, its text, and what to record with it."""

    token_ids: torch.Tensor
    span: Tuple[int, int]
    text: str
    index: int
    info: Dict[str, Any] = field(default_factory=dict)
    logits: Optional[Dict[str, np.ndarray]] = None


def item_z(scorer: TokenScorer, span_per_layer: np.ndarray) -> Optional[float]:
    """An item's z from its tokens' scores at the scorer's layers: their mean, standardized with the
    per-item references. None without a calibration, or when it was measured with a pooling other
    than the mean (its zero would not apply to a mean)."""
    cal = scorer.calibration
    span = np.asarray(span_per_layer, dtype=np.float64)
    if cal is None or (cal.context.readout or "text_mean") != "text_mean" or span.shape[0] == 0:
        return None
    z = cal.items_z(span.mean(axis=0)[None, :], scorer.layers, scorer.aggregate)
    return None if z is None else float(z[0])


def score_inputs(model, tokenizer, inputs: Sequence[ScoreInput], scorers: Sequence[TokenScorer], *,
                 batch_size: int = 16) -> List[ScoredItem]:
    """Score every token of each input with each scorer (see :class:`ScoredItem`)."""
    names = [s.name for s in scorers]
    if len(set(names)) != len(names):
        raise ValueError(f"the probes need distinct names, since scores are keyed by name; got {names}")
    projected = _project_tokens(model, tokenizer, [x.token_ids for x in inputs], scorers, batch_size=batch_size)
    items = []
    for x, per_probe in zip(inputs, projected):
        ids = np.asarray(x.token_ids, dtype=np.int64).reshape(-1)
        lo, hi = int(x.span[0]), int(x.span[1])
        item = ScoredItem(index=x.index, text=x.text, token_ids=ids, tokens=_decode_tokens(tokenizer, ids),
                          span=(lo, hi), layers={}, scores={}, token_scores={}, token_z={}, score_mean={},
                          score_mean_z={}, info=dict(x.info), logits=x.logits)
        for s in scorers:
            per_layer = per_probe[s.name]
            agg = s.combine(per_layer).astype(np.float32)
            item.layers[s.name] = list(s.layers)
            item.scores[s.name] = per_layer
            item.token_scores[s.name] = agg
            item.token_z[s.name] = s.token_z(per_layer)
            item.score_mean[s.name] = float(np.mean(agg[lo:hi])) if hi > lo else None
            item.score_mean_z[s.name] = item_z(s, per_layer[lo:hi])
        items.append(item)
    return items


@dataclass
class Condition:
    """One way of generating: an intervention (or none) and what to record about it in each item."""

    info: Dict[str, Any]
    intervention: Optional[Intervention] = None


def generate_and_score(
    model,
    tokenizer,
    prompts: Sequence[Any],
    conditions: Sequence[Condition],
    scorers: Sequence[TokenScorer],
    *,
    system: Optional[str] = None,
    max_new_tokens: int = 64,
    greedy: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: int = 16,
    read_batch_size: int = 16,
    on_token: Optional[Callable[[TokenStep], Any]] = None,
    watch: Sequence[Any] = (),
    keep_logits: Any = None,
    warn: Optional[Callable[[str], None]] = None,
    status: Optional[Callable[[str], None]] = None,
) -> List[ScoredItem]:
    """Generate every prompt under each condition (in batches) and score every completion, read
    again without any intervention. Items come prompt by prompt, each prompt's conditions in order;
    each item's ``info`` has its condition's ``info``, the prompt, where the completion is and the
    intervention it was generated under."""
    status = status or (lambda message: None)
    if not greedy and seed is not None:
        set_seed(int(seed))  # once: every condition draws from the same stream, in order
    generated = []
    for c, condition in enumerate(conditions):
        gens = generate(model, tokenizer, prompts, system=system, max_new_tokens=max_new_tokens, greedy=greedy,
                        temperature=temperature, top_p=top_p, batch_size=batch_size,
                        intervention=condition.intervention, on_token=on_token, watch=watch if on_token else (),
                        keep_logits=keep_logits, save=False, warn=warn)
        generated.extend((g.index, c, g, condition) for g in gens)
        status(f"Generated condition {c + 1}/{len(conditions)}")
    generated.sort(key=lambda x: (x[0], x[1]))
    inputs = [ScoreInput(torch.as_tensor(g.token_ids), (g.prompt_len, g.completion_end), g.text, g.index,
                         dict(condition.info, prompt=g.prompt, prompt_len=g.prompt_len, completion_end=g.completion_end,
                              system_prompt_overridden=g.own_system, intervention=g.intervention),
                         logits=g.logits)
              for _, _, g, condition in generated]
    status(f"Scoring {len(inputs)} completions")
    return score_inputs(model, tokenizer, inputs, scorers, batch_size=read_batch_size)


def batch_dir(base: str, batch_subdir: Optional[str], overwrite: bool) -> str:
    """The folder of one call's scores: ``base/batch_subdir`` (a new timestamped name when None).

    A folder that already holds files raises unless ``overwrite``, which deletes them and the
    analysis folder: an earlier batch's files, such as its coherence ratings, would otherwise be
    read together with the new ones.
    """
    out = base if batch_subdir == "" else os.path.join(base, batch_subdir or f"batch_{now_tag()}")
    if os.path.isdir(out):
        existing = [f for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))]
        if existing and not overwrite:
            raise FileExistsError(f"{out} already holds a scoring batch ({len(existing)} files); pass another "
                                  "batch_subdir, or overwrite=True to replace it.")
        for f in existing:
            os.remove(os.path.join(out, f))
        shutil.rmtree(os.path.join(out, "analysis"), ignore_errors=True)
    ensure_dir(out)
    return out
