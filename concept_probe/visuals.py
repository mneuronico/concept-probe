import html
import json
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


_SPECIAL_MARKERS = {
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|im_start|>",
    "<|im_end|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<bos>",
    "<eos>",
    "<pad>",
    "<|endoftext|>",
}

_ROLE_ALIASES = {
    "model": "assistant",
    "assistant": "assistant",
    "user": "user",
    "system": "system",
}


def _normalize_role_label(role: str) -> str:
    raw = str(role).strip().lower().strip(":")
    if not raw:
        return ""
    return _ROLE_ALIASES.get(raw, raw)


def _token_for_display(tok: str) -> str:
    # Normalize common tokenizer-visible whitespace markers so HTML views are readable.
    # Keep special markers untouched (they are handled separately by block splitting).
    if tok in _SPECIAL_MARKERS:
        return tok
    return tok.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")


def _split_chat_blocks_with_indices(tokens: List[str]) -> List[Tuple[str, List[str], List[int]]]:
    blocks: List[Tuple[str, List[str], List[int]]] = []
    cur_role = "text"
    cur_tokens: List[str] = []
    cur_indices: List[int] = []
    in_header = False
    header_tokens: List[str] = []
    expect_turn_role = False

    def flush():
        if cur_tokens:
            blocks.append((cur_role, list(cur_tokens), list(cur_indices)))
            cur_tokens.clear()
            cur_indices.clear()

    for idx, tok in enumerate(tokens):
        if tok in _SPECIAL_MARKERS:
            if tok == "<|start_header_id|>":
                flush()
                in_header = True
                header_tokens = []
            elif tok == "<|end_header_id|>":
                in_header = False
                role = _normalize_role_label("".join(header_tokens))
                if role:
                    cur_role = role
            elif tok in {"<start_of_turn>", "<|im_start|>"}:
                flush()
                expect_turn_role = True
            elif tok in {"<|eot_id|>", "<end_of_turn>", "<|im_end|>"}:
                flush()
                expect_turn_role = False
            continue

        if in_header:
            header_tokens.append(tok)
            continue

        if expect_turn_role:
            role_candidate = _normalize_role_label(_token_for_display(tok))
            if role_candidate:
                cur_role = role_candidate
                expect_turn_role = False
                continue
            continue

        cur_tokens.append(tok)
        cur_indices.append(idx)

    flush()
    return blocks


def segment_token_scores(
    tokens: List[str],
    scores: List[float],
    *,
    prompt_len: Optional[int] = None,
) -> List[Dict[str, object]]:
    blocks = _split_chat_blocks_with_indices(tokens)
    segments: List[Dict[str, object]] = []
    seg_idx = 0
    for block_idx, (role, block_tokens, block_indices) in enumerate(blocks):
        block_scores = [float(scores[i]) for i in block_indices]
        if prompt_len is None:
            if not block_indices:
                continue
            mean_score = float(np.mean(block_scores)) if block_scores else 0.0
            segments.append(
                {
                    "segment_index": seg_idx,
                    "message_index": block_idx,
                    "role": role,
                    "phase": "all",
                    "token_indices": list(block_indices),
                    "tokens": list(block_tokens),
                    "scores": list(block_scores),
                    "mean_score": mean_score,
                    "token_count": len(block_indices),
                }
            )
            seg_idx += 1
            continue

        prompt_tokens: List[str] = []
        prompt_indices: List[int] = []
        prompt_scores: List[float] = []
        completion_tokens: List[str] = []
        completion_indices: List[int] = []
        completion_scores: List[float] = []

        for tok, idx, sc in zip(block_tokens, block_indices, block_scores):
            if idx < int(prompt_len):
                prompt_tokens.append(tok)
                prompt_indices.append(idx)
                prompt_scores.append(sc)
            else:
                completion_tokens.append(tok)
                completion_indices.append(idx)
                completion_scores.append(sc)

        if prompt_indices:
            mean_score = float(np.mean(prompt_scores)) if prompt_scores else 0.0
            segments.append(
                {
                    "segment_index": seg_idx,
                    "message_index": block_idx,
                    "role": role,
                    "phase": "prompt",
                    "token_indices": list(prompt_indices),
                    "tokens": list(prompt_tokens),
                    "scores": list(prompt_scores),
                    "mean_score": mean_score,
                    "token_count": len(prompt_indices),
                }
            )
            seg_idx += 1
        if completion_indices:
            mean_score = float(np.mean(completion_scores)) if completion_scores else 0.0
            segments.append(
                {
                    "segment_index": seg_idx,
                    "message_index": block_idx,
                    "role": role,
                    "phase": "completion",
                    "token_indices": list(completion_indices),
                    "tokens": list(completion_tokens),
                    "scores": list(completion_scores),
                    "mean_score": mean_score,
                    "token_count": len(completion_indices),
                }
            )
            seg_idx += 1
    return segments


def _render_blocks_html_with_indices(blocks: List[Tuple[str, List[str], List[int]]]) -> str:
    parts = []
    token_idx = 0
    for role, block_tokens, _ in blocks:
        role_label = html.escape(role.strip() or "text")
        parts.append(f'<div class="block"><div class="role">{role_label}</div><div class="tokens">')
        for tok in block_tokens:
            safe_tok = html.escape(_token_for_display(tok), quote=False)
            parts.append(f'<span class="tok" data-idx="{token_idx}">{safe_tok}</span>')
            token_idx += 1
        parts.append("</div></div>")
    return "".join(parts)


def render_heatmaps(
    entries: List[Tuple[str, List[str], Union[List[float], Dict[str, List[float]]]]], out_path: str, title: str
) -> None:
    """One HTML page with each entry's tokens colored by their scores: ``(title, tokens, scores)``,
    where ``scores`` is one list, or ``{name: list}`` for several probes (a menu picks the probe).
    The positive side is blue and the negative orange, as in the other figures; each probe's scale
    is its largest absolute score over all entries."""
    if not entries:
        return
    entries = [(t, toks, sc if isinstance(sc, dict) else {"score": sc}) for t, toks, sc in entries]
    probe_names = list(entries[0][2].keys())
    scores_by_probe_entries: Dict[str, List[List[float]]] = {name: [] for name in probe_names}
    sections = []

    for entry_idx, (entry_title, tokens, scores_by_probe) in enumerate(entries):
        blocks = _split_chat_blocks_with_indices(tokens)
        visible_indices = [idx for _, _, idxs in blocks for idx in idxs]
        for name in probe_names:
            scores = scores_by_probe.get(name, [])
            scores_by_probe_entries[name].append(
                [float(scores[i]) for i in visible_indices] if visible_indices else []
            )

        section_title = html.escape(entry_title)
        body = _render_blocks_html_with_indices(blocks)
        sections.append(f'<div class="entry" data-entry="{entry_idx}"><h3>{section_title}</h3>{body}</div>')

    max_abs_by_probe: Dict[str, float] = {}
    for name in probe_names:
        flat = np.array([s for entry in scores_by_probe_entries[name] for s in entry], dtype=np.float32)
        max_abs_by_probe[name] = float(np.max(np.abs(flat))) if flat.size else 0.0

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
    .legend {{ margin: 8px 0 12px 0; font-size: 12px; }}
    .swatch {{ display: inline-block; width: 24px; height: 12px; margin-right: 6px; }}
    .tokens {{ white-space: pre-wrap; }}
    .tok {{ padding: 0 2px; }}
    .block {{ margin: 12px 0; }}
    .role {{ font-weight: bold; font-size: 12px; text-transform: uppercase; margin-bottom: 6px; }}
    .toolbar {{ margin: 8px 0 12px 0; font-size: 12px; }}
  </style>
</head>
<body>
<h2>{html.escape(title)}</h2>
<div class="toolbar">
  <label for="probe-select">Probe:</label>
  <select id="probe-select">
    {''.join([f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in probe_names])}
  </select>
</div>
<div class="legend">
  <span class="swatch" style="background: rgba(43,108,176,0.8);"></span>positive
  <span class="swatch" style="background: rgba(221,132,82,0.8); margin-left: 12px;"></span>negative
  <span id="scale-label" style="margin-left: 12px;">scale: +/-0.0000</span>
</div>
{''.join(sections)}
<script>
const scoresByProbe = {json.dumps(scores_by_probe_entries)};
const maxAbsByProbe = {json.dumps(max_abs_by_probe)};
const selectEl = document.getElementById("probe-select");

function colorForScore(value, maxAbs) {{
  if (!maxAbs || maxAbs <= 1e-12) return "rgba(0,0,0,0)";
  const v = Math.max(-1.0, Math.min(1.0, value / maxAbs));
  const alpha = Math.abs(v);
  if (v >= 0) {{
    return `rgba(43,108,176,${{alpha.toFixed(3)}})`;
  }}
  return `rgba(221,132,82,${{alpha.toFixed(3)}})`;
}}

function applyProbe(name) {{
  const entryScores = scoresByProbe[name] || [];
  const maxAbs = maxAbsByProbe[name] || 0.0;
  document.querySelectorAll(".entry").forEach((entry) => {{
    const entryIdx = parseInt(entry.dataset.entry, 10);
    const scores = entryScores[entryIdx] || [];
    entry.querySelectorAll(".tok").forEach((span) => {{
      const idx = parseInt(span.dataset.idx, 10);
      const sc = scores[idx] || 0.0;
      span.style.background = colorForScore(sc, maxAbs);
      span.title = (sc >= 0 ? "+" : "") + sc.toFixed(4);
    }});
  }});
  document.getElementById("scale-label").textContent = "scale: +/-" + maxAbs.toFixed(4);
}}

selectEl.addEventListener("change", () => applyProbe(selectEl.value));
applyProbe(selectEl.value);
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


D_COLOR = "#2b6cb0"      # Cohen's d of the probe
NULL_COLOR = "#9a9a9a"   # what random directions give
P_COLOR = "#553c9a"      # p-values
POS_COLOR = "#2b6cb0"    # the positive pole's items
NEG_COLOR = "#dd8452"    # the negative pole's items


def plot_sweep(
    sweep_d: Iterable[float],
    sweep_p: Iterable[float],
    out_path: str,
    *,
    best_layer: Optional[int] = None,
    search_intervals: Optional[Iterable[Tuple[int, int]]] = None,
    selection_d: Optional[Iterable[float]] = None,
    report_d: Optional[Iterable[float]] = None,
    null_d95: Optional[Iterable[float]] = None,
    best_d: Optional[float] = None,
    title: Optional[str] = None,
) -> bool:
    """Cohen's d and its p per layer.

    With ``report_d`` (the d on the held-out items, which the reported number comes from) the top
    panel draws that curve, with the reported ``best_d`` marked on it, and ``selection_d`` (the
    items that chose the layer) dashed; otherwise it draws ``sweep_d``, all eval items. ``null_d95``
    (the 95th percentile of |d| over random directions on the same items) is the gray band: a d
    inside it is what any direction would give. The bottom panel has the p-values on all items.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    d = np.array(list(sweep_d), dtype=np.float64)
    p = np.array(list(sweep_p), dtype=np.float64)
    layers = np.arange(len(d))

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    if search_intervals:
        for lo, hi in search_intervals:
            for axis in ax:
                axis.axvspan(lo - 0.5, hi + 0.5, color="#f2e6b0", alpha=0.35, zorder=0)
    if null_d95 is not None:
        null = np.array(list(null_d95), dtype=np.float64)
        ax[0].fill_between(layers, 0, null, color=NULL_COLOR, alpha=0.35, linewidth=0,
                           label="random directions (95th percentile of |d|)")
    if report_d is not None:
        ax[0].plot(layers, np.array(list(report_d), dtype=np.float64), color=D_COLOR,
                   label="held-out items (the reported d)")
        if selection_d is not None:
            ax[0].plot(layers, np.array(list(selection_d), dtype=np.float64), color=D_COLOR, linestyle="--",
                       alpha=0.6, label="selection items (chose the layer)")
    else:
        ax[0].plot(layers, d, color=D_COLOR, label="eval items")
        if selection_d is not None:
            ax[0].plot(layers, np.array(list(selection_d), dtype=np.float64), color=D_COLOR, linestyle="--",
                       alpha=0.6, label="selection items (chose the layer)")
    if best_layer is not None and best_d is not None and np.isfinite(best_d):
        ax[0].plot([best_layer], [best_d], "o", color=D_COLOR)
        ax[0].annotate(f"d = {best_d:.2f}", (best_layer, best_d), textcoords="offset points", xytext=(6, 6),
                       fontsize=9, color=D_COLOR)
    ax[0].axhline(0, color="#444444", linewidth=0.6)
    ax[0].set_ylabel("Cohen's d")
    ax[0].legend(fontsize=8, loc="best")
    ax[0].set_title(title or "Cohen's d per layer")

    safe_p = np.where(p <= 0, np.nan, p)
    ax[1].plot(layers, safe_p, color=P_COLOR)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("p-value (all items)")
    ax[1].set_xlabel("Layer")

    if best_layer is not None:
        for axis in ax:
            axis.axvline(best_layer, color="#222222", linestyle=":", linewidth=1.2, alpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def plot_score_hist(pos_scores: Iterable[float], neg_scores: Iterable[float], out_path: str, *,
                    title: Optional[str] = None, labels: Tuple[str, str] = ("positive", "negative")) -> bool:
    """Histograms of the two poles' scores at one layer; ``title`` should say which items and layer."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    pos = np.array(list(pos_scores), dtype=np.float32)
    neg = np.array(list(neg_scores), dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    bins = np.histogram_bin_edges(np.concatenate([pos, neg]), bins=20) if len(pos) + len(neg) else 20
    ax.hist(pos, bins=bins, alpha=0.6, color=POS_COLOR, label=labels[0])
    ax.hist(neg, bins=bins, alpha=0.6, color=NEG_COLOR, label=labels[1])
    ax.set_xlabel("Probe score (raw projection)")
    ax.set_ylabel("Items")
    ax.set_title(title or "Probe scores", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True
