import html
import json
from typing import Dict, Iterable, List, Optional, Tuple

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


def _visible_scores(tokens: List[str], scores: List[float]) -> List[float]:
    blocks = _split_chat_blocks(tokens, scores)
    out: List[float] = []
    for _, _, block_scores in blocks:
        out.extend(block_scores)
    return out


def _color_for_score(value: float, max_abs: float) -> str:
    if max_abs <= 1e-12:
        return "rgba(0,0,0,0)"
    v = max(-1.0, min(1.0, value / max_abs))
    if v >= 0:
        alpha = abs(v)
        return f"rgba(220,60,60,{alpha:.3f})"
    alpha = abs(v)
    return f"rgba(60,90,220,{alpha:.3f})"


def _split_chat_blocks(tokens: List[str], scores: List[float]) -> List[Tuple[str, List[str], List[float]]]:
    blocks: List[Tuple[str, List[str], List[float]]] = []
    cur_role = "text"
    cur_tokens: List[str] = []
    cur_scores: List[float] = []
    in_header = False
    header_tokens: List[str] = []
    expect_turn_role = False

    def flush():
        if cur_tokens:
            blocks.append((cur_role, list(cur_tokens), list(cur_scores)))
            cur_tokens.clear()
            cur_scores.clear()

    for tok, sc in zip(tokens, scores):
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
            # Skip pure-whitespace header tokens.
            continue

        cur_tokens.append(tok)
        cur_scores.append(float(sc))

    flush()
    return blocks


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


def split_chat_blocks_with_indices(tokens: List[str]) -> List[Tuple[str, List[str], List[int]]]:
    return _split_chat_blocks_with_indices(tokens)


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


def render_token_heatmap_multi(
    tokens: List[str],
    scores_by_probe: Dict[str, List[float]],
    out_path: str,
    title: Optional[str] = None,
    max_abs_by_probe: Optional[Dict[str, float]] = None,
) -> None:
    probe_names = list(scores_by_probe.keys())
    blocks = _split_chat_blocks_with_indices(tokens)
    visible_indices = [idx for _, _, idxs in blocks for idx in idxs]

    visible_scores: Dict[str, List[float]] = {}
    for name in probe_names:
        scores = scores_by_probe.get(name, [])
        visible_scores[name] = [float(scores[i]) for i in visible_indices] if visible_indices else []

    if max_abs_by_probe is None:
        max_abs_by_probe = {}
        for name in probe_names:
            vals = np.array(visible_scores[name], dtype=np.float32)
            max_abs_by_probe[name] = float(np.max(np.abs(vals))) if vals.size else 0.0

    title_html = f"<h3>{html.escape(title)}</h3>" if title else ""
    body = _render_blocks_html_with_indices(blocks)

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
{title_html}
<div class="toolbar">
  <label for="probe-select">Probe:</label>
  <select id="probe-select">
    {''.join([f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in probe_names])}
  </select>
</div>
<div class="legend">
  <span class="swatch" style="background: rgba(60,90,220,0.8);"></span>negative
  <span class="swatch" style="background: rgba(220,60,60,0.8); margin-left: 12px;"></span>positive
  <span id="scale-label" style="margin-left: 12px;">scale: +/-0.0000</span>
</div>
{body}
<script>
const scoresByProbe = {json.dumps(visible_scores)};
const maxAbsByProbe = {json.dumps(max_abs_by_probe)};
const selectEl = document.getElementById("probe-select");

function colorForScore(value, maxAbs) {{
  if (!maxAbs || maxAbs <= 1e-12) return "rgba(0,0,0,0)";
  const v = Math.max(-1.0, Math.min(1.0, value / maxAbs));
  const alpha = Math.abs(v);
  if (v >= 0) {{
    return `rgba(220,60,60,${{alpha.toFixed(3)}})`;
  }}
  return `rgba(60,90,220,${{alpha.toFixed(3)}})`;
}}

function applyProbe(name) {{
  const scores = scoresByProbe[name] || [];
  const maxAbs = maxAbsByProbe[name] || 0.0;
  const spans = document.querySelectorAll(".tok");
  spans.forEach((span) => {{
    const idx = parseInt(span.dataset.idx, 10);
    const sc = scores[idx] || 0.0;
    span.style.background = colorForScore(sc, maxAbs);
    span.title = (sc >= 0 ? "+" : "") + sc.toFixed(4);
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


def render_batch_heatmap_multi(
    entries: List[Tuple[str, List[str], Dict[str, List[float]]]], out_path: str, title: str
) -> None:
    if not entries:
        return
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
  <span class="swatch" style="background: rgba(60,90,220,0.8);"></span>negative
  <span class="swatch" style="background: rgba(220,60,60,0.8); margin-left: 12px;"></span>positive
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
    return `rgba(220,60,60,${{alpha.toFixed(3)}})`;
  }}
  return `rgba(60,90,220,${{alpha.toFixed(3)}})`;
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


def _render_blocks_html(
    tokens: List[str],
    scores: List[float],
    max_abs: float,
) -> str:
    blocks = _split_chat_blocks(tokens, scores)
    parts = []
    for role, block_tokens, block_scores in blocks:
        role_label = html.escape(role.strip() or "text")
        parts.append(f'<div class="block"><div class="role">{role_label}</div><div class="tokens">')
        for tok, sc in zip(block_tokens, block_scores):
            safe_tok = html.escape(_token_for_display(tok), quote=False)
            color = _color_for_score(float(sc), max_abs)
            title = f"{sc:+.4f}"
            parts.append(
                f'<span class="tok" style="background:{color};" title="{title}">{safe_tok}</span>'
            )
        parts.append("</div></div>")
    return "".join(parts)


def render_token_heatmap(
    tokens: List[str],
    scores: List[float],
    out_path: str,
    title: Optional[str] = None,
    max_abs: Optional[float] = None,
) -> None:
    if max_abs is None:
        vis = _visible_scores(tokens, scores)
        max_abs = float(np.max(np.abs(np.array(vis, dtype=np.float32)))) if vis else 0.0
    else:
        max_abs = float(max_abs)
    title_html = f"<h3>{html.escape(title)}</h3>" if title else ""
    body = _render_blocks_html(tokens, scores, max_abs=max_abs)
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
  </style>
</head>
<body>
{title_html}
<div class="legend">
  <span class="swatch" style="background: rgba(60,90,220,0.8);"></span>negative
  <span class="swatch" style="background: rgba(220,60,60,0.8); margin-left: 12px;"></span>positive
  <span style="margin-left: 12px;">scale: ±{max_abs:.4f}</span>
</div>
{body}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def render_batch_heatmap(entries: List[Tuple[str, List[str], List[float]]], out_path: str, title: str) -> None:
    all_scores: List[float] = []
    for _, tokens, scores in entries:
        all_scores.extend(_visible_scores(tokens, scores))
    max_abs = float(np.max(np.abs(np.array(all_scores, dtype=np.float32)))) if all_scores else 0.0

    sections = []
    for entry_title, tokens, scores in entries:
        section_title = html.escape(entry_title)
        body = _render_blocks_html(tokens, scores, max_abs=max_abs)
        sections.append(f"<h3>{section_title}</h3>{body}")

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
  </style>
</head>
<body>
<h2>{html.escape(title)}</h2>
<div class="legend">
  <span class="swatch" style="background: rgba(60,90,220,0.8);"></span>negative
  <span class="swatch" style="background: rgba(220,60,60,0.8); margin-left: 12px;"></span>positive
  <span style="margin-left: 12px;">scale: ±{max_abs:.4f}</span>
</div>
{''.join(sections)}
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)


def plot_sweep(
    sweep_d: Iterable[float],
    sweep_p: Iterable[float],
    out_path: str,
    *,
    best_layer: Optional[int] = None,
    search_intervals: Optional[Iterable[Tuple[int, int]]] = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    d = np.array(list(sweep_d), dtype=np.float32)
    p = np.array(list(sweep_p), dtype=np.float32)
    layers = np.arange(len(d))

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    if search_intervals:
        for lo, hi in search_intervals:
            for axis in ax:
                axis.axvspan(lo - 0.5, hi + 0.5, color="#f2e6b0", alpha=0.35, zorder=0)
    ax[0].plot(layers, d, color="#cc3d3d")
    ax[0].set_ylabel("Cohen's d")
    ax[0].set_title("Layer sweep")

    safe_p = np.where(p <= 0, np.nan, p)
    ax[1].plot(layers, safe_p, color="#3d5acc")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("p-value (log)")
    ax[1].set_xlabel("Layer")

    if best_layer is not None:
        for axis in ax:
            axis.axvline(best_layer, color="#222222", linestyle=":", linewidth=1.5, alpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def plot_score_hist(pos_scores: Iterable[float], neg_scores: Iterable[float], out_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    pos = np.array(list(pos_scores), dtype=np.float32)
    neg = np.array(list(neg_scores), dtype=np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(pos, bins=20, alpha=0.6, color="#cc3d3d", label="pos")
    ax.hist(neg, bins=20, alpha=0.6, color="#3d5acc", label="neg")
    ax.set_xlabel("Probe score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True
