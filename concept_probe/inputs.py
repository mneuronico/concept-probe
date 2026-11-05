"""Inputs: how a text or a conversation becomes the token ids the model reads (layer 1 of
docs/DESIGN.md, the part before the forward pass).

A text is read in a role (raw, as a user turn, or as the model's own reply), with an optional
system prompt and, for a reply, the user turn before it. Encoding returns the ids and the span of
the text's own tokens, which the ``assistant_*`` readouts pool over; :func:`preview_input` shows
the exact input before anything runs.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .modeling import _chat_template_kwargs, apply_chat


PromptMessage = Dict[str, str]
PromptLike = Union[str, List[PromptMessage]]


def _prompt_to_text(prompt: PromptLike) -> str:
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt)
    parts: List[str] = []
    for m in prompt:
        if not isinstance(m, dict):
            parts.append(str(m))
            continue
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        if role:
            parts.append(f"{role}: {content}")
        else:
            parts.append(content)
    return "\n".join(parts)



def _decode_tokens(tokenizer, token_ids: Iterable[int]) -> List[str]:
    return [tokenizer.decode([int(t)], skip_special_tokens=False) for t in token_ids]


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


_CHAT_TEMPLATE_FALLBACK_WARNED: set = set()


def _encode_for_read(
    tokenizer,
    prompt: "PromptLike",
    default_system: Optional[str],
    *,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "read",
) -> torch.Tensor:
    """Encode a chat-formatted 'read' input (no generation): ``prompt`` as a user turn, after
    ``default_system`` if one is set (see :func:`_normalize_messages`). Without a chat template
    it falls back to plain tokenization, and any system text is dropped.
    """
    if _has_chat_template(tokenizer):
        messages, _ = _normalize_messages(
            prompt, default_system, warn=warn, warn_prefix=warn_prefix
        )
        return apply_chat(tokenizer, messages, add_generation_prompt=False)

    text = prompt if isinstance(prompt, str) else _prompt_to_text(prompt)
    fallback_key = id(tokenizer)
    if warn is not None and fallback_key not in _CHAT_TEMPLATE_FALLBACK_WARNED:
        warn(
            f"{warn_prefix}: tokenizer has no chat_template; falling back to plain tokenization"
            + (" (the system prompt is dropped)." if default_system else ".")
        )
        _CHAT_TEMPLATE_FALLBACK_WARNED.add(fallback_key)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"]


def _completion_input_ids(tokenizer, messages: "List[PromptMessage]", completion: str) -> Tuple[torch.Tensor, int]:
    """Chat-format ``messages`` with an assistant generation prompt and append ``completion``.

    Returns ``(input_ids, prompt_len)``: the completion occupies ``input_ids[0, prompt_len:]``,
    with no end-of-turn token after it.
    """
    prompt_ids = apply_chat(tokenizer, messages, add_generation_prompt=True)
    plen = int(prompt_ids.shape[-1])
    comp_ids = tokenizer(str(completion), add_special_tokens=False, return_tensors="pt")["input_ids"]
    if comp_ids.dim() == 1:
        comp_ids = comp_ids.unsqueeze(0)
    return torch.cat([prompt_ids, comp_ids.to(prompt_ids.dtype)], dim=1), plen


TEXT_ROLES = ("user", "assistant", "raw")

_TEXT_SENTINEL = "CONCEPT_PROBE_TEXT_SENTINEL"


def _token_count(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(torch.as_tensor(ids).reshape(-1).shape[0])


def token_char_offsets(tokenizer, text: str) -> Optional[List[Tuple[int, int]]]:
    """Character span ``(a, b)`` in ``text`` of each of its tokens (no special tokens), or None
    when the tokenizer cannot say (slow tokenizers have no offsets)."""
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        om = enc["offset_mapping"]
    except Exception:
        return None
    return _as_pairs(om)


def _as_pairs(om: Any) -> List[Tuple[int, int]]:
    om = om.tolist() if hasattr(om, "tolist") else list(om)
    if om and isinstance(om[0], (list, tuple)) and om[0] and isinstance(om[0][0], (list, tuple)):
        om = om[0]  # batched
    return [(int(x), int(y)) for x, y in om]


def _render(tokenizer, messages, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False,
                                         **_chat_template_kwargs(tokenizer))


def _rendered_offsets(tokenizer, messages, add_generation_prompt: bool, input_ids: Optional[Sequence[int]]):
    """The rendered chat and its token offsets, when re-tokenizing it gives ``input_ids``."""
    try:
        rendered = _render(tokenizer, messages, add_generation_prompt)
        enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids, om = enc["input_ids"], enc["offset_mapping"]
    except Exception:
        return None
    if not isinstance(rendered, str):
        return None
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if input_ids is not None and [int(t) for t in ids] != [int(t) for t in input_ids]:
        return None
    return rendered, _as_pairs(om)


def _locate_message(
    tokenizer, messages: "List[PromptMessage]", index: int, add_generation_prompt: bool,
    rendered: str, offsets: List[Tuple[int, int]],
) -> Optional[Dict[str, Any]]:
    """Where message ``index``'s content sits in ``rendered`` (whose token offsets are ``offsets``):
    its token span, the text the template kept (it may strip whitespace), and each token's
    character span within that text. None if it cannot be located."""
    content = messages[index]["content"]
    try:
        # What the template writes before a message's content does not depend on that content, so
        # a sentinel in its place gives where the content starts.
        probe = messages[:index] + [dict(messages[index], content=_TEXT_SENTINEL)] + messages[index + 1:]
        probe_rendered = _render(tokenizer, probe, add_generation_prompt)
    except Exception:
        return None
    if not isinstance(probe_rendered, str):
        return None
    char_start = probe_rendered.find(_TEXT_SENTINEL)
    if char_start < 0:
        return None
    kept = next((n for n in (content, content.strip()) if n and rendered.startswith(n, char_start)), None)
    if kept is None:  # the template rewrote the text
        return None
    char_end = char_start + len(kept)
    covered = [i for i, (a, b) in enumerate(offsets) if b > a and a < char_end and b > char_start]
    if not covered:
        return None
    rel = [(max(0, offsets[i][0] - char_start), min(len(kept), offsets[i][1] - char_start))
           for i in range(covered[0], covered[-1] + 1)]
    return {"span": (covered[0], covered[-1] + 1), "text": kept, "trimmed": kept != content, "offsets": rel}


def message_spans(
    tokenizer, messages: "List[PromptMessage]", add_generation_prompt: bool = False,
    input_ids: Optional[Sequence[int]] = None,
) -> List[Optional[Tuple[int, int]]]:
    """Token span ``(start, end)`` of each message's content in the chat-formatted input, or None
    for a message whose content cannot be located (the template rewrote it, or the tokenizer has no
    offsets). ``input_ids``, if given, must be the encoding the spans refer to."""
    if not _has_chat_template(tokenizer):
        return [None] * len(messages)
    found = _rendered_offsets(tokenizer, messages, add_generation_prompt, input_ids)
    if found is None:
        return [None] * len(messages)
    out: List[Optional[Tuple[int, int]]] = []
    for i in range(len(messages)):
        loc = _locate_message(tokenizer, messages, i, add_generation_prompt, *found)
        out.append(None if loc is None else loc["span"])
    return out


def _user_text_location(
    tokenizer, text: "PromptLike", system: Optional[str], input_ids: torch.Tensor
) -> Optional[Dict[str, Any]]:
    """Where a user-role text (for a message list, its last message) sits in its encoding from
    :func:`_encode_for_read`: see :func:`_locate_message`. Without a chat template the text
    follows position 0 (the attention sink: the BOS, or else its first token, left out)."""
    total = int(input_ids.shape[-1])
    if not _has_chat_template(tokenizer):
        flat = text if isinstance(text, str) else _prompt_to_text(text)
        bos = getattr(tokenizer, "bos_token_id", None)
        has_bos = bos is not None and total > 0 and int(input_ids[0, 0]) == int(bos)
        n_text = _token_count(tokenizer, flat)
        start, end = 1, min(total, (1 if has_bos else 0) + n_text)
        offsets = token_char_offsets(tokenizer, flat)
        if offsets is not None:
            shift = 1 if has_bos else 0
            offsets = offsets[start - shift:end - shift]
        return {"span": (start, end), "text": flat, "trimmed": False, "offsets": offsets}
    messages, _ = _normalize_messages(text, system)
    found = _rendered_offsets(tokenizer, messages, False, input_ids[0].tolist())
    if found is None:
        return None
    return _locate_message(tokenizer, messages, len(messages) - 1, False, *found)


def _user_text_span(
    tokenizer, text: "PromptLike", system: Optional[str], input_ids: torch.Tensor
) -> Optional[Tuple[int, int, bool]]:
    """Token span of a user-role text inside its encoding from :func:`_encode_for_read`.

    Returns ``(start, end, trimmed)``: ``[start, end)`` are the text's own tokens, leaving out
    the BOS, system block and role headers before the text and the end-of-turn tokens after it
    (for a message-list text, the last message's content). ``trimmed`` is True when the chat
    template stripped whitespace from the text. Returns None if the span cannot be located.
    """
    loc = _user_text_location(tokenizer, text, system, input_ids)
    return None if loc is None else (loc["span"][0], loc["span"][1], loc["trimmed"])


_TEMPLATE_TRIM_WARNED: set = set()
_NO_BOS_WARNED: set = set()


def _no_bos_sink_start(tokenizer, warn: Optional[Callable[[str], None]], warn_prefix: str) -> int:
    """Where a raw text's own span starts when the tokenizer has no BOS token: after its first token.

    The first position of an input is the attention sink, with activations far larger than other
    tokens'; with a BOS it is the BOS, otherwise it is the text's first token, which is then left
    out of pooling.
    """
    if warn is not None and id(tokenizer) not in _NO_BOS_WARNED:
        warn(
            f"{warn_prefix}: the tokenizer has no BOS token, so the first token of each text is the attention "
            "sink (activations far larger than the rest); it is left out of pooling."
        )
        _NO_BOS_WARNED.add(id(tokenizer))
    return 1


@dataclass
class Encoded:
    """One input to read: its token ids and where the item's own tokens are.

    ``input_ids[start:end]`` are the item's own tokens (a text, or a generated completion), the
    span the ``assistant_*`` readouts pool over; ``start`` is None when the span is unknown.
    ``text`` is what that span holds as the model saw it (a chat template may strip whitespace),
    and ``offsets`` the character span in ``text`` of each token of the span, which readouts at a
    word use; both are None when unknown.
    """

    input_ids: torch.Tensor
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None
    offsets: Optional[List[Tuple[int, int]]] = None


def encode_text(
    tokenizer,
    text: "PromptLike",
    system: Optional[str] = None,
    *,
    role: str = "raw",
    prompt: Optional[str] = None,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "read",
) -> Encoded:
    """Encode a text to read (no generation), as texts-mode training, eval texts, ``score_texts``
    and ``extract_text_reps`` do.

    - ``role="raw"`` (default): the text alone, with no chat template and no system prompt: the
      BOS token (if the tokenizer has one) and the text. ``system`` and ``prompt`` are ignored.
      Without a BOS, the text's first token is the attention sink and is left out of the span.
    - ``role="user"``: the text as a user turn of the chat template, after ``system`` if one is
      set. A message list is read as given, and its last message is the text. The span leaves out
      the template tokens around the text; it is None when it cannot be located (``sequence_*``
      readouts still work then).
    - ``role="assistant"``: the text as the model's own reply: ``system`` if set, an optional user
      turn ``prompt``, the assistant header, then the text, with no end-of-turn token after it.
    """
    if role not in TEXT_ROLES:
        raise ValueError(f"training.train_text_role must be one of {TEXT_ROLES}; got {role!r}")
    if role == "user":
        ids = _encode_for_read(tokenizer, text, system, warn=warn, warn_prefix=warn_prefix)
        loc = _user_text_location(tokenizer, text, system, ids)
        if loc is None:
            if warn is not None:
                warn(f"{warn_prefix}: could not locate the text's own tokens in the chat-formatted input.")
            return Encoded(ids[0])
        if loc["trimmed"] and warn is not None and id(tokenizer) not in _TEMPLATE_TRIM_WARNED:
            warn(f"{warn_prefix}: the chat template strips leading/trailing whitespace from texts.")
            _TEMPLATE_TRIM_WARNED.add(id(tokenizer))
        return Encoded(ids[0], loc["span"][0], loc["span"][1], loc["text"], loc["offsets"])
    if not isinstance(text, str):
        raise TypeError(
            f"{warn_prefix}: role={role!r} needs plain-string texts; read a message list with role='user'."
        )
    if role == "raw":
        if (prompt is not None or system) and warn is not None:
            warn(f"{warn_prefix}: role='raw' reads the text alone; system and prompt are ignored.")
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        offsets = token_char_offsets(tokenizer, text)
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is None:
            start = _no_bos_sink_start(tokenizer, warn, warn_prefix)
            return Encoded(ids[0], start, int(ids.shape[-1]), text, None if offsets is None else offsets[start:])
        ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
        return Encoded(ids[0], 1, int(ids.shape[-1]), text, offsets)
    if not _has_chat_template(tokenizer):
        raise ValueError("role='assistant' needs a tokenizer with a chat template.")
    messages: "List[PromptMessage]" = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    if prompt is not None:
        messages.append({"role": "user", "content": str(prompt)})
    if not messages:
        raise ValueError(
            f"{warn_prefix}: reading a text as the assistant's reply needs a turn before it, since chat "
            "templates cannot format a conversation that starts with the reply: pass prompt= (a user turn) "
            "or system= (in training: training.train_text_prompt, or training.train_text_system / "
            "prompts.neutral_system)."
        )
    ids, plen = _completion_input_ids(tokenizer, messages, text)
    return Encoded(ids[0], plen, int(ids.shape[-1]), text, token_char_offsets(tokenizer, text))


@dataclass(frozen=True)
class ReadContext:
    """How items were read: what a calibration (the zero and unit of z scores) applies to.

    ``kind`` is ``"text"`` (texts read with ``role``, ``system`` and, for a reply, ``prompt``; see
    :func:`encode_text`) or ``"generated"`` (completions the model generated, read after their
    prompt). ``readout`` is how their tokens were pooled. Two contexts match when they read items
    the same way: generated completions always match each other, texts match on role, system
    prompt and prompt. The readout is recorded but not compared; item z scores need the mean
    pooling they were measured with, which the scorers check.
    """

    kind: str = "text"
    role: Optional[str] = "raw"
    system: Optional[str] = None
    prompt: Optional[str] = None
    readout: Optional[str] = "assistant_all_mean"

    def __post_init__(self):
        if self.kind not in ("text", "generated"):
            raise ValueError(f"ReadContext kind must be 'text' or 'generated'; got {self.kind!r}")
        if self.kind == "text" and self.role not in TEXT_ROLES:
            raise ValueError(f"ReadContext role must be one of {TEXT_ROLES}; got {self.role!r}")

    @classmethod
    def generated(cls, readout: Optional[str] = "assistant_all_mean") -> "ReadContext":
        return cls(kind="generated", role=None, system=None, prompt=None, readout=readout)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ReadContext":
        """From the dict runs save (``{"kind", "role", "system", "prompt", "mode"}``)."""
        d = dict(d or {})
        if d.get("kind") == "generated":
            return cls.generated(d.get("mode", d.get("readout")))
        return cls(kind="text", role=d.get("role") or "raw", system=d.get("system") or None,
                   prompt=d.get("prompt"), readout=d.get("mode", d.get("readout")))

    def to_dict(self) -> Dict[str, Any]:
        """The dict runs save, as before this class existed."""
        if self.kind == "generated":
            return {"kind": "generated", "mode": self.readout}
        return {"kind": "text", "role": self.role, "system": self.system or None, "prompt": self.prompt,
                "mode": self.readout}

    def matches(self, other: "ReadContext") -> bool:
        if self.kind != other.kind:
            return False
        if self.kind == "generated":
            return True
        return all((getattr(self, k) or None) == (getattr(other, k) or None) for k in ("role", "system", "prompt"))

    def describe(self) -> str:
        """A short description, for messages."""
        if self.kind == "generated":
            return "as generated completions"
        parts = [f"as {'raw text' if self.role == 'raw' else self.role + ' turns'}"]
        if self.system:
            parts.append(f"after the system prompt {self.system!r}")
        if self.prompt is not None:
            parts.append(f"after the user turn {self.prompt!r}")
        return " ".join(parts)


def _encode_text_for_read(
    tokenizer,
    text: "PromptLike",
    system: Optional[str] = None,
    *,
    role: str = "raw",
    prompt: Optional[str] = None,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "read",
) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
    """:func:`encode_text` as ``(input_ids of shape (1, seq_len), text_start, text_end)``."""
    enc = encode_text(tokenizer, text, system, role=role, prompt=prompt, warn=warn, warn_prefix=warn_prefix)
    return enc.input_ids.unsqueeze(0), enc.start, enc.end


def preview_input(
    tokenizer,
    text: "PromptLike",
    *,
    role: str = "raw",
    system: Optional[str] = None,
    prompt: Optional[str] = None,
    generation: bool = False,
) -> Dict[str, Any]:
    """Exactly what the library feeds the model for ``text``, before running anything.

    ``generation=False`` encodes ``text`` as a text to read, as texts-mode training, eval texts,
    ``score_texts`` and ``extract_text_reps`` do with the same ``role``/``system``/``prompt``.
    ``generation=True`` encodes it as a prompt to generate from, as ``score_prompts`` does: a user
    turn (after ``system`` if set) and the assistant header; ``role`` and ``prompt`` are ignored.

    Returns ``input_text`` (decoded with every special and template token), ``input_ids``,
    ``tokens``, ``text_span`` / ``span_text`` (the tokens the ``assistant_*`` readouts pool over;
    None for generation prompts), and ``message_spans``: for each message of the chat (the system
    prompt, the user turns, the reply being read), its ``role``, its token ``span`` (None where it
    cannot be located) and the ``text`` of that span. A raw text has no messages.
    """
    spans: List[Dict[str, Any]] = []
    if generation:
        messages, _ = _normalize_messages(text, system)
        ids = apply_chat(tokenizer, messages, add_generation_prompt=True)
        start = end = None
        located = message_spans(tokenizer, messages, True, ids[0].tolist())
        spans = [{"role": m["role"], "span": sp} for m, sp in zip(messages, located)]
    else:
        enc = encode_text(tokenizer, text, system, role=role, prompt=prompt)
        ids, start, end = enc.input_ids.unsqueeze(0), enc.start, enc.end
        if role == "user" and _has_chat_template(tokenizer):
            messages, _ = _normalize_messages(text, system)
            located = message_spans(tokenizer, messages, False, ids[0].tolist())
            spans = [{"role": m["role"], "span": sp} for m, sp in zip(messages, located)]
        elif role == "assistant":
            before = ([{"role": "system", "content": str(system)}] if system else []) + (
                [{"role": "user", "content": str(prompt)}] if prompt is not None else [])
            located = message_spans(tokenizer, before, True, ids[0, :start].tolist())
            spans = [{"role": m["role"], "span": sp} for m, sp in zip(before, located)]
            spans.append({"role": "assistant", "span": (int(start), int(end))})
    token_ids = [int(t) for t in ids[0].tolist()]
    for item in spans:
        sp = item["span"]
        item["text"] = None if sp is None else tokenizer.decode(token_ids[sp[0]:sp[1]], skip_special_tokens=False)
    return {
        "input_text": tokenizer.decode(token_ids, skip_special_tokens=False),
        "input_ids": token_ids,
        "tokens": _decode_tokens(tokenizer, token_ids),
        "text_span": None if start is None else (int(start), int(end)),
        "span_text": None if start is None else tokenizer.decode(token_ids[start:end], skip_special_tokens=False),
        "message_spans": spans,
    }


def _normalize_messages(
    prompt: PromptLike,
    default_system: Optional[str],
    *,
    warn: Optional[Callable[[str], None]] = None,
    warn_prefix: str = "prompt",
) -> Tuple[List[PromptMessage], bool]:
    """Return (messages, used_prompt_system).

    A system message is added only when ``default_system`` is set; None or "" add none.

    - A string prompt becomes a user turn, after the system message if any.
    - A message list is used as given. ``default_system`` is prepended only if the list has no
      system message; if it has one, that one wins (with a warning when ``default_system`` is set).
    """
    system_messages = [{"role": "system", "content": str(default_system)}] if default_system else []
    if isinstance(prompt, str):
        return system_messages + [{"role": "user", "content": prompt}], False

    if not isinstance(prompt, list):
        raise TypeError(f"{warn_prefix} must be str or list of messages; got {type(prompt)}")

    messages: List[PromptMessage] = []
    system_count = 0
    for i, m in enumerate(prompt):
        if not isinstance(m, dict):
            raise TypeError(f"{warn_prefix}[{i}] must be a dict with role/content; got {type(m)}")
        if "role" not in m or "content" not in m:
            raise ValueError(f"{warn_prefix}[{i}] must have 'role' and 'content' keys")
        role = str(m["role"])
        content = str(m["content"])
        if role == "system":
            system_count += 1
        messages.append({"role": role, "content": content})

    if system_count > 0:
        if default_system and warn is not None:
            warn(
                f"{warn_prefix} includes a system message; the provided default system prompt is ignored."
            )
        if system_count > 1 and warn is not None:
            warn(f"{warn_prefix} includes multiple system messages; using them as provided.")
        return (messages, True)

    return system_messages + messages, False
