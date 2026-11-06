"""Generating, scoring tokens and the probability of given answers (layer 5 of docs/DESIGN.md).

- :func:`generate` generates completions in batches (left padding), optionally under an
  intervention (layer 4), and returns them in memory; ``on_token`` is called at every step with
  the score of each watched probe on the token just read.
- :func:`answer_logprobs` gives the log-probability of given answers to each prompt, optionally
  under an intervention.
- ``_project_tokens`` projects every token of many inputs on the probes' directions, in batches;
  the scorers of :mod:`concept_probe.scoring` and :func:`score_text_tokens` are built on it.

Like everything that produces a result, :func:`generate` and :func:`answer_logprobs` save it by
default (decision D4): to ``out_dir``, or a new folder under the session's ``output_root``.
"""
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from .inputs import Encoded, PromptLike, _decode_tokens, _normalize_messages
from .interventions import Intervention, intervene
from .modeling import apply_chat, get_transformer_layers, safe_attention
from .reading import _forward_batch, check_finite_hidden_states
from .settings import settings
from .utils import ensure_dir, json_dump, now_tag, set_seed


# ---------------------------------------------------------------------------------- shared pieces
def eos_token_ids(model, tokenizer) -> Set[int]:
    """Token ids that end a generation: the model's generation config, plus the tokenizer's EOS."""
    ids: Set[int] = set()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is not None:
        ids.update(int(e) for e in (eos if isinstance(eos, (list, tuple)) else [eos]))
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if tok_eos is not None:
        ids.add(int(tok_eos))
    return ids


def completion_end(token_ids: Any, prompt_len: int, eos_ids: Set[int]) -> int:
    """End of a generated completion: the first end-of-sequence token at or after ``prompt_len``
    (excluded from the completion), or the end of the sequence. Pooling and scores never include it."""
    ids = [int(t) for t in token_ids]
    for i in range(int(prompt_len), len(ids)):
        if ids[i] in eos_ids:
            return i
    return len(ids)


def _pad_id(tokenizer) -> int:
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad is None else int(pad)


def _encode_prompts(tokenizer, prompts: Sequence[PromptLike], system: Optional[str],
                    warn: Optional[Callable[[str], None]], label: str) -> List[Tuple[torch.Tensor, bool]]:
    """Each prompt as the model's input for a reply: chat template with the generation prompt."""
    out = []
    for i, p in enumerate(prompts):
        messages, used_own_system = _normalize_messages(p, system, warn=warn, warn_prefix=f"{label}[{i}]")
        if warn is not None and messages and messages[-1].get("role") != "user":
            warn(f"{label}[{i}] ends with role='{messages[-1].get('role')}'; generation usually expects the last "
                 "message to be a user turn.")
        out.append((apply_chat(tokenizer, messages, add_generation_prompt=True)[0], used_own_system))
    return out


def _should_save(save: Optional[bool]) -> bool:
    return settings().save if save is None else bool(save)


def _new_dir(kind: str, out_dir: Optional[str]) -> str:
    path = out_dir or os.path.join(settings().output_root, kind, now_tag())
    ensure_dir(path)
    return path


# ------------------------------------------------------------------------------------- scorers
@dataclass
class TokenScorer:
    """How one probe scores tokens: its directions at the layers it scores with, how those layers'
    projections are combined, and the calibration that turns them into z (or None)."""

    name: str
    layers: List[int]
    vectors: np.ndarray  # (len(layers), dim)
    aggregate: str = "mean"
    calibration: Any = None

    def combine(self, per_layer: np.ndarray) -> np.ndarray:
        """``(tokens, len(layers))`` projections -> one score per token."""
        if self.aggregate == "mean":
            return per_layer.mean(axis=1)
        if self.aggregate == "sum":
            return per_layer.sum(axis=1)
        raise ValueError(f"Unknown aggregate: {self.aggregate}")

    def token_z(self, per_layer: np.ndarray) -> Optional[np.ndarray]:
        if self.calibration is None:
            return None
        out = self.calibration.tokens_z(per_layer, self.layers, self.aggregate)
        return None if out is None else np.asarray(out[1], dtype=np.float32)


@torch.no_grad()
def _project_tokens(model, tokenizer, inputs: Sequence[torch.Tensor], scorers: Sequence[TokenScorer], *,
                    batch_size: int = 16) -> List[Dict[str, np.ndarray]]:
    """Every token of each input projected on each scorer's directions: per input,
    ``{name: (tokens, len(layers)) float32}``. Inputs go through the model ``batch_size`` at a time
    (right padding); under an active intervention the projections read the edited stream."""
    needed = sorted({l for s in scorers for l in s.layers})
    vectors = {s.name: torch.as_tensor(np.asarray(s.vectors, dtype=np.float32)) for s in scorers}
    pad = _pad_id(tokenizer)
    out: List[Dict[str, np.ndarray]] = []
    step = max(1, int(batch_size))
    for b in range(0, len(inputs), step):
        hs, lens = _forward_batch(model, [x.reshape(-1) for x in inputs[b:b + step]], pad)
        for j, n in enumerate(lens):
            rows = {l: hs[l + 1][j, :n].float() for l in needed}
            check_finite_hidden_states([rows[l].cpu() for l in needed], model)
            item = {}
            for s in scorers:
                v = vectors[s.name]
                cols = [(rows[l] @ v[k].to(rows[l].device)).cpu() for k, l in enumerate(s.layers)]
                item[s.name] = torch.stack(cols, dim=1).numpy().astype(np.float32)
            out.append(item)
        del hs
    return out


# -------------------------------------------------------------------------------------- generate
@dataclass
class Generation:
    """One generated completion.

    ``token_ids`` are the prompt's tokens, then the completion's, then the end-of-sequence token if
    one was generated; the completion is ``token_ids[prompt_len:completion_end]``. ``logits`` holds,
    when asked for, the model's logits at each generated token (under the intervention, as it
    generated): ``{"values": (steps, k), "indices": (steps, k)}`` for the top ``k``, or
    ``{"values": (steps, vocab)}``.
    """

    index: int
    prompt: PromptLike
    system: Optional[str]
    token_ids: np.ndarray
    prompt_len: int
    completion_end: int
    text: str
    intervention: Optional[Dict[str, Any]] = None
    logits: Optional[Dict[str, np.ndarray]] = None
    #: the prompt was a message list with its own system message, so ``system`` was not used
    own_system: bool = False

    @property
    def completion_ids(self) -> np.ndarray:
        return self.token_ids[self.prompt_len:self.completion_end]

    def encoded(self) -> Encoded:
        """The prompt and the completion, with the completion as the span to read."""
        return Encoded(torch.as_tensor(self.token_ids, dtype=torch.long), self.prompt_len, self.completion_end,
                       self.text, None)

    def record(self) -> Dict[str, Any]:
        return {"index": self.index, "prompt": self.prompt, "system": None if self.own_system else self.system,
                "text": self.text,
                "prompt_len": self.prompt_len, "completion_end": self.completion_end,
                "n_tokens": int(len(self.token_ids)), "intervention": self.intervention}


class Generations(list):
    """The list of :class:`Generation` that :func:`generate` returns, with the call's ``settings``
    and the folder it was saved to (``dir``, None when not saved)."""

    def __init__(self, items: Sequence[Generation] = (), settings: Optional[Dict[str, Any]] = None,
                 dir: Optional[str] = None):
        super().__init__(items)
        self.settings = dict(settings or {})
        self.dir = dir

    def save(self, out_dir: str) -> str:
        """``generations.jsonl`` (one readable line each) and ``generations.npz`` (token ids, and
        logits when kept) in ``out_dir``."""
        ensure_dir(out_dir)
        with open(os.path.join(out_dir, "generations.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"settings": self.settings}, ensure_ascii=False, default=str) + "\n")
            for g in self:
                f.write(json.dumps(g.record(), ensure_ascii=False, default=str) + "\n")
        arrays: Dict[str, np.ndarray] = {
            "token_ids": np.concatenate([g.token_ids for g in self]) if self else np.zeros(0, np.int64),
            "offsets": np.cumsum([0] + [len(g.token_ids) for g in self]).astype(np.int64),
            "prompt_len": np.array([g.prompt_len for g in self], dtype=np.int64),
            "completion_end": np.array([g.completion_end for g in self], dtype=np.int64),
        }
        arrays.update(_pack_logits([g.logits for g in self]))
        np.savez_compressed(os.path.join(out_dir, "generations.npz"), **arrays)
        self.dir = out_dir
        return out_dir


def _pack_logits(per_item: Sequence[Optional[Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
    if not per_item or any(x is None for x in per_item):
        return {}
    out = {"logits_offsets": np.cumsum([0] + [len(x["values"]) for x in per_item]).astype(np.int64)}
    for key in per_item[0]:
        out[f"logits_{key}"] = np.concatenate([x[key] for x in per_item])
    return out


@dataclass
class TokenStep:
    """What ``on_token`` gets at each forward step of :func:`generate`.

    ``token_ids`` is the token each row just read: at step 0 the prompt's last token, then each
    generated token as the model reads it to predict the next. ``scores[name]`` is each watched
    probe's score of that token (its raw projection, combined over the probe's scoring layers), and
    ``z[name]`` its token z where the probe has a calibration for generated completions. Under an
    intervention this reading includes the edit (an added vector reads itself back); the scores
    that the scorers report are read again afterwards without it. ``active`` is False for rows
    that already ended (their token is padding). ``prompts`` says which prompt of the call each row is.
    """

    step: int
    prompts: List[int]
    token_ids: np.ndarray
    active: np.ndarray
    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    z: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)


class _Watcher:
    """Hooks that read the newest token at the watched probes' layers and call ``on_token``."""

    def __init__(self, model, scorers: Sequence[TokenScorer], on_token: Callable[[TokenStep], Any],
                 rows: List[int], eos_ids: Set[int]):
        self.model, self.scorers, self.on_token, self.rows, self.eos = model, list(scorers), on_token, rows, eos_ids
        self.layers = sorted({l for s in self.scorers for l in s.layers})
        self.handles: List[Any] = []
        self.last: Dict[int, torch.Tensor] = {}
        self.read: Optional[torch.Tensor] = None
        self.finished = torch.zeros(len(rows), dtype=torch.bool)
        self.step = 0

    def __enter__(self) -> "_Watcher":
        blocks = get_transformer_layers(self.model)

        def pre(module, args, kwargs):
            x = kwargs.get("input_ids")
            if x is None and args and isinstance(args[0], torch.Tensor):
                x = args[0]
            self.read = None if x is None else x[:, -1].detach().to("cpu")

        def keep(layer):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                self.last[layer] = h[:, -1, :].detach().float()
            return hook

        def post(module, args, kwargs, output):
            self._report()

        self.handles.append(self.model.register_forward_pre_hook(pre, with_kwargs=True))
        for l in self.layers:
            self.handles.append(blocks[l].register_forward_hook(keep(l)))
        self.handles.append(self.model.register_forward_hook(post, with_kwargs=True))
        return self

    def _report(self) -> None:
        token_ids = self.read.numpy() if self.read is not None else np.full(len(self.rows), -1)
        active = (~self.finished).numpy()
        scores, zs = {}, {}
        for s in self.scorers:
            per_layer = torch.stack([self.last[l].cpu() @ torch.as_tensor(np.asarray(s.vectors[k], np.float32))
                                     for k, l in enumerate(s.layers)], dim=1).numpy()
            scores[s.name] = s.combine(per_layer).astype(np.float32)
            zs[s.name] = s.token_z(per_layer)
        self.on_token(TokenStep(self.step, list(self.rows), token_ids, active, scores, zs))
        if self.read is not None:
            self.finished |= torch.tensor([int(t) in self.eos for t in token_ids])
        self.step += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class _LogitKeeper:
    """A hook on the output embeddings keeping the logits of the newest position at each step."""

    def __init__(self, model, keep: Union[int, str]):
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if head is None:
            raise ValueError("keep_logits needs the model's output embeddings (model.get_output_embeddings())")
        if keep != "all" and (not isinstance(keep, int) or keep < 1):
            raise ValueError(f"keep_logits must be None, 'all' or a number of top logits; got {keep!r}")
        self.head, self.keep = head, keep
        self.steps: List[Dict[str, torch.Tensor]] = []
        self.handle = None

    def __enter__(self) -> "_LogitKeeper":
        def hook(module, inputs, output):
            last = output[:, -1, :].detach().float()
            if self.keep == "all":
                self.steps.append({"values": last.cpu()})
            else:
                values, indices = torch.topk(last, k=min(int(self.keep), last.shape[-1]), dim=-1)
                self.steps.append({"values": values.cpu(), "indices": indices.to(torch.int32).cpu()})
        self.handle = self.head.register_forward_hook(hook)
        return self

    def for_row(self, row: int, n_steps: int) -> Dict[str, np.ndarray]:
        steps = self.steps[:n_steps]
        return {key: torch.stack([s[key][row] for s in steps]).numpy() for key in self.steps[0]}

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: Sequence[PromptLike],
    *,
    system: Optional[str] = None,
    max_new_tokens: int = 64,
    greedy: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: int = 16,
    intervention: Optional[Intervention] = None,
    on_token: Optional[Callable[[TokenStep], Any]] = None,
    watch: Sequence[Any] = (),
    keep_logits: Union[None, int, str] = None,
    save: Optional[bool] = None,
    out_dir: Optional[str] = None,
    warn: Optional[Callable[[str], None]] = None,
    label: str = "prompts",
) -> Generations:
    """Generate a completion for each prompt, ``batch_size`` prompts at a time.

    Each prompt is a user turn of the chat template (a message list is used as given) after
    ``system`` if one is set; :func:`concept_probe.preview_input` with ``generation=True`` shows the
    exact input. Prompts in a batch are padded on the left, with an attention mask; padding is
    removed from what is returned. In half precision a batched greedy completion can differ in its
    last tokens from the one generated alone (D8): use ``batch_size=1`` when that matters. Sampling
    (``greedy=False``) seeds the generators with ``seed`` first when it is given.

    ``intervention`` (see :mod:`concept_probe.interventions`) is applied while generating, with
    positions counted from each prompt's start. ``on_token(step)`` is called at every step with a
    :class:`TokenStep`: the token each row just read and, for each probe in ``watch``, its score.
    ``keep_logits`` keeps the logits of every generated token: ``"all"``, or the top ``k``.

    Returns a :class:`Generations` list, saved by default to ``out_dir`` (or a new folder under the
    session's ``output_root``) as ``generations.jsonl`` and ``generations.npz``.
    """
    if not prompts:
        return Generations([], {})
    if watch and on_token is None:
        raise ValueError("watch= needs on_token=, the function that gets the scores at each step")
    if not greedy and seed is not None:
        set_seed(int(seed))
    scorers = [p._token_scorer(generated=True) if hasattr(p, "_token_scorer") else p for p in watch]
    encoded = _encode_prompts(tokenizer, prompts, system, warn, label)
    eos_ids = eos_token_ids(model, tokenizer)
    pad = _pad_id(tokenizer)
    described = None if intervention is None else intervention.describe()
    results: List[Generation] = []
    step = max(1, int(batch_size))
    for start in range(0, len(encoded), step):
        chunk = [ids for ids, _ in encoded[start:start + step]]
        own = [o for _, o in encoded[start:start + step]]
        plens = [int(t.shape[-1]) for t in chunk]
        width = max(plens)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, t in enumerate(chunk):
            input_ids[j, width - plens[j]:] = t.to(torch.long)
            attn[j, width - plens[j]:] = 1
        rows = list(range(start, start + len(chunk)))
        applied = _Nothing() if intervention is None else intervene(model, intervention, prompt_lens=plens, items=rows)
        watcher = _Nothing() if on_token is None else _Watcher(model, scorers, on_token, rows, eos_ids)
        keeper = _Nothing() if keep_logits is None else _LogitKeeper(model, keep_logits)
        sampling = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None} if greedy else             {"do_sample": True, "temperature": temperature, "top_p": top_p}
        with applied, watcher, keeper, safe_attention(model):
            out = model.generate(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                                 max_new_tokens=int(max_new_tokens), pad_token_id=pad, **sampling)
        out = out.detach().cpu()
        for j, i in enumerate(rows):
            row = out[j][width - plens[j]:]  # without the left padding
            end = completion_end(row, plens[j], eos_ids)
            keep = min(int(row.shape[-1]), end + 1)  # keep the end-of-sequence token, drop padding after it
            ids = row[:keep].numpy().astype(np.int64)
            results.append(Generation(
                index=i, prompt=prompts[i], system=system, token_ids=ids, prompt_len=plens[j], completion_end=end,
                text=tokenizer.decode(ids[plens[j]:end].tolist(), skip_special_tokens=True),
                intervention=described,
                logits=None if keep_logits is None else keeper.for_row(j, keep - plens[j]),
                own_system=bool(own[j]),
            ))
    run_settings = {"system": system, "max_new_tokens": int(max_new_tokens), "greedy": bool(greedy),
                    "temperature": None if greedy else temperature, "top_p": None if greedy else top_p,
                    "seed": None if greedy else seed, "batch_size": step, "intervention": described,
                    "keep_logits": keep_logits}
    gens = Generations(results, run_settings)
    if _should_save(save):
        gens.save(_new_dir("generations", out_dir))
    return gens


# ------------------------------------------------------------------------------- answer probabilities
@dataclass
class AnswerLogprobs:
    """The log-probability of each given answer to each prompt (:func:`answer_logprobs`).

    ``logprobs[i, k]`` is log P(answer k | prompt i), summed over the answer's tokens;
    ``n_tokens[i, k]`` how many tokens that is, and ``answer_tokens[i][k]`` which (check them: an
    answer is tokenized on its own, so write it as the model would, e.g. with a leading space when
    the model would put one).
    """

    prompts: List[PromptLike]
    answers: List[List[str]]
    logprobs: np.ndarray
    n_tokens: np.ndarray
    answer_tokens: List[List[List[str]]]
    intervention: Optional[Dict[str, Any]] = None
    dir: Optional[str] = None

    def probs(self) -> np.ndarray:
        """Each answer's probability among the given ones (renormalized over them)."""
        lp = self.logprobs - self.logprobs.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def log_odds(self, first: int = 0, second: int = 1) -> np.ndarray:
        """log P(first) - log P(second) for each prompt (for one-token answers, the logit difference)."""
        return self.logprobs[:, first] - self.logprobs[:, second]

    def save(self, out_dir: str) -> str:
        ensure_dir(out_dir)
        json_dump(os.path.join(out_dir, "answer_logprobs.json"), {
            "prompts": self.prompts, "answers": self.answers, "logprobs": self.logprobs.tolist(),
            "n_tokens": self.n_tokens.tolist(), "answer_tokens": self.answer_tokens,
            "intervention": self.intervention,
        })
        self.dir = out_dir
        return out_dir


def _forward_accepts(model, name: str) -> bool:
    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


@torch.no_grad()
def _continuation_logprobs(model, pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]], *, pad: int, batch_size: int = 16,
                           intervention: Optional[Intervention] = None,
                           items: Optional[Sequence[int]] = None) -> List[np.ndarray]:
    """For each ``(prompt ids, continuation ids)``, the log-probability of every continuation token
    given what comes before it, ``batch_size`` pairs per forward pass (right padding, the logits of
    the continuation's positions only when the model can keep just those). ``intervention`` is
    applied meanwhile, with positions counted from each prompt's start."""
    out: List[np.ndarray] = [np.zeros(0) for _ in pairs]
    todo = [r for r in range(len(pairs)) if int(pairs[r][1].shape[0]) > 0]
    order = sorted(todo, key=lambda r: int(pairs[r][0].shape[0] + pairs[r][1].shape[0]))
    keep_arg = _forward_accepts(model, "logits_to_keep")
    step = max(1, int(batch_size))
    for b in range(0, len(order), step):
        rows = order[b:b + step]
        plens = [int(pairs[r][0].shape[0]) for r in rows]
        full = [torch.cat([pairs[r][0].reshape(-1), pairs[r][1].reshape(-1)]).to(torch.long) for r in rows]
        lens = [int(x.shape[0]) for x in full]
        input_ids = torch.full((len(full), max(lens)), int(pad), dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for j, x in enumerate(full):
            input_ids[j, :lens[j]] = x
            attn[j, :lens[j]] = 1
        lo, hi = min(plens) - 1, max(lens) - 1  # the positions whose logits predict continuation tokens
        kwargs: Dict[str, Any] = {"logits_to_keep": torch.arange(lo, hi, device=model.device)} if keep_arg else {}
        row_items = None if items is None else [int(items[r]) for r in rows]
        applied = _Nothing() if intervention is None else intervene(model, intervention, prompt_lens=plens,
                                                                     items=row_items)
        with applied, safe_attention(model):
            logits = model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                           use_cache=False, **kwargs).logits
        offset = lo if keep_arg else 0
        for j, r in enumerate(rows):
            cont = pairs[r][1].reshape(-1).to(torch.long)
            positions = torch.arange(plens[j] - 1, plens[j] - 1 + cont.shape[0]) - offset
            row = logits[j, positions.to(logits.device)]
            logp = torch.log_softmax(row if row.dtype == torch.float64 else row.float(), dim=-1)
            out[r] = logp[torch.arange(cont.shape[0], device=logp.device), cont.to(logp.device)].double().cpu().numpy()
        del logits
    return out


@torch.no_grad()
def answer_logprobs(
    model,
    tokenizer,
    prompts: Sequence[PromptLike],
    answers: Sequence[Any],
    *,
    system: Optional[str] = None,
    intervention: Optional[Intervention] = None,
    batch_size: int = 16,
    save: Optional[bool] = None,
    out_dir: Optional[str] = None,
    warn: Optional[Callable[[str], None]] = None,
) -> AnswerLogprobs:
    """How likely the model is to answer each prompt with each of ``answers``.

    Each prompt is formatted as for :func:`generate` (a user turn, the chat template's generation
    prompt); ``answers`` is one list of strings for every prompt, or one list per prompt (all of
    the same length). Each answer's tokens are appended to the prompt and their log-probabilities
    summed, ``batch_size`` (prompt, answer) pairs per forward pass. ``intervention`` is applied
    meanwhile, with negative positions counted back from the end of the prompt (``-1`` is the
    position that predicts the answer's first token). Saved by default, like :func:`generate`.
    """
    prompts = list(prompts)
    if answers and isinstance(answers[0], str):
        per_prompt = [list(answers) for _ in prompts]
    else:
        per_prompt = [list(a) for a in answers]
        if len(per_prompt) != len(prompts):
            raise ValueError(f"answers has {len(per_prompt)} lists for {len(prompts)} prompts")
    k = len(per_prompt[0]) if per_prompt else 0
    if k == 0 or any(len(a) != k for a in per_prompt):
        raise ValueError("every prompt needs the same, non-zero number of answers")
    encoded = _encode_prompts(tokenizer, prompts, system, warn, "prompts")
    pairs = []  # (prompt index, answer index, prompt ids, answer ids)
    answer_tokens: List[List[List[str]]] = []
    for i, (ids, _) in enumerate(encoded):
        answer_tokens.append([])
        for a, text in enumerate(per_prompt[i]):
            ans = tokenizer(str(text), add_special_tokens=False)["input_ids"]
            if hasattr(ans, "tolist"):
                ans = ans.tolist()
            ans = list(ans[0] if ans and isinstance(ans[0], list) else ans)
            if not ans:
                raise ValueError(f"answer {text!r} has no tokens")
            answer_tokens[i].append(_decode_tokens(tokenizer, ans))
            pairs.append((i, a, ids.to(torch.long), torch.tensor(ans, dtype=torch.long)))
    per_token = _continuation_logprobs(model, [(p, ans) for _, _, p, ans in pairs], pad=_pad_id(tokenizer),
                                       batch_size=batch_size, intervention=intervention,
                                       items=[i for i, _, _, _ in pairs])
    logprobs = np.zeros((len(prompts), k))
    n_tokens = np.zeros((len(prompts), k), dtype=np.int64)
    for (i, a, _, ans), lp in zip(pairs, per_token):
        logprobs[i, a] = float(lp.sum())
        n_tokens[i, a] = int(ans.shape[0])
    result = AnswerLogprobs(prompts, per_prompt, logprobs, n_tokens, answer_tokens,
                            None if intervention is None else intervention.describe())
    if _should_save(save):
        result.save(_new_dir("answer_logprobs", out_dir))
    return result
