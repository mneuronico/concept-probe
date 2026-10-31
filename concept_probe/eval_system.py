import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .probe import ConceptProbe
from .utils import ensure_dir, json_dump, safe_slug

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


PromptLike = Union[str, List[Dict[str, str]]]
ExampleItem = Dict[str, Any]
ExampleGenerator = Callable[..., Union[ExampleItem, List[ExampleItem]]]
Evaluator = Callable[[str, ExampleItem], Dict[str, Any]]

RATING_COLORS = {
    "COHERENT": "#33b233",
    "PARTIALLY_COHERENT": "#f2cc33",
    "NONSENSE": "#d93434",
}


@dataclass
class EvalRunResult:
    results: List[Dict[str, Any]]
    per_sample: List[Dict[str, Any]]
    batch_dir: str
    analysis_dir: Optional[str]


def _completion_slice(data: Any) -> slice:
    """The completion's tokens in a score npz: from prompt_len to completion_end (exclusive)."""
    prompt_len = int(data["prompt_len"][0]) if "prompt_len" in data else 0
    end = int(data["completion_end"][0]) if "completion_end" in data else None
    return slice(prompt_len, end)


def _mean_completion_score(npz_path: str) -> float:
    data = np.load(npz_path)
    scores = data["scores_agg"]
    span = scores[_completion_slice(data)]
    if span.size == 0:
        return float("nan")
    return float(np.mean(span))


def _score_mean_from_result(rec: Dict[str, Any]) -> float:
    direct = rec.get("score_mean")
    if direct is not None:
        try:
            direct_val = float(direct)
            if np.isfinite(direct_val):
                return direct_val
        except Exception:
            pass
    npz_path = rec.get("npz_path")
    if isinstance(npz_path, str):
        return _mean_completion_score(npz_path)
    return float("nan")


def _decode_probe_names(raw: Any) -> List[str]:
    names: List[str] = []
    if raw is None:
        return names
    for item in raw:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8", errors="ignore"))
        else:
            names.append(str(item))
    return names


def _mean_completion_score_multi(npz_path: str) -> Dict[str, float]:
    data = np.load(npz_path)
    scores = data["scores_agg"]
    completion = _completion_slice(data)

    if scores.ndim == 1:
        span = scores[completion]
        mean_val = float(np.mean(span)) if span.size else float("nan")
        return {"_probe_0": mean_val}

    if scores.ndim != 2:
        raise ValueError(f"Unexpected scores_agg shape: {scores.shape}")

    raw_names = data.get("probe_names")
    probe_names = _decode_probe_names(raw_names)
    if not probe_names or len(probe_names) != scores.shape[0]:
        probe_names = [f"probe_{i}" for i in range(scores.shape[0])]

    score_map: Dict[str, float] = {}
    for idx, name in enumerate(probe_names):
        span = scores[idx][completion]
        mean_val = float(np.mean(span)) if span.size else float("nan")
        score_map[name] = mean_val
    return score_map


def _score_mean_by_probe_from_result(rec: Dict[str, Any]) -> Dict[str, float]:
    direct = rec.get("score_mean_by_probe")
    if isinstance(direct, dict):
        out: Dict[str, float] = {}
        for key, value in direct.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
        if out:
            return out
    npz_path = rec.get("npz_path")
    if isinstance(npz_path, str):
        return _mean_completion_score_multi(npz_path)
    return {}


def _finite_float(value: Any) -> float:
    """Coerce to float, mapping None/non-numeric to NaN (per_sample.json stores NaN as null)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sem(values: List[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    nx, ny = x.size, y.size
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / (np.sqrt(pooled) + 1e-12))


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _wilson_interval(successes: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """95% Wilson score interval of a proportion (stays inside [0, 1], unlike the normal one)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _correctness_gee(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Logistic GEE of correctness on the probe score, adjusting for alpha, clustered by item.

    Steering moves both the score and the accuracy, so alpha enters as a categorical
    covariate; each item appears once per alpha, so rows are clustered by item (robust
    standard errors). The score is standardized across rows: the coefficient is per 1 SD.
    """
    usable = [
        r for r in rows
        if _is_bool(r.get("correct")) and np.isfinite(_finite_float(r.get("score_mean")))
        and r.get("item_index") is not None and "alpha" in r
    ]
    out: Dict[str, Any] = {
        "model": "logit P(correct) ~ score_z + C(alpha); GEE, exchangeable within item, robust SE",
        "n_rows": len(usable),
        "n_items": len({r["item_index"] for r in usable}),
        "note": None,
    }
    y = np.array([1.0 if r["correct"] else 0.0 for r in usable])
    if len(usable) < 4 or y.min() == y.max():
        out["note"] = "Needs at least 4 labelled rows with both correct and incorrect answers."
        return out
    try:
        import statsmodels.api as sm
    except Exception:
        out["note"] = "statsmodels not available; model not fit."
        return out
    score = np.array([_finite_float(r["score_mean"]) for r in usable])
    sd = float(score.std())
    if sd == 0:
        out["note"] = "The score is constant across rows."
        return out
    alphas = sorted({float(r["alpha"]) for r in usable})
    columns = [np.ones(len(usable)), (score - score.mean()) / sd]
    columns += [np.array([1.0 if float(r["alpha"]) == a else 0.0 for r in usable]) for a in alphas[1:]]
    X = np.column_stack(columns)
    groups = np.array([r["item_index"] for r in usable])
    try:
        fit = sm.GEE(y, X, groups=groups, family=sm.families.Binomial(),
                     cov_struct=sm.cov_struct.Exchangeable()).fit()
    except Exception as exc:  # e.g. perfect separation
        out["note"] = f"GEE fit failed: {exc}"
        return out
    ci = np.asarray(fit.conf_int())
    out.update({
        "score_sd": sd,
        "coef_per_sd": float(fit.params[1]),
        "odds_ratio_per_sd": float(np.exp(fit.params[1])),
        "ci95": [float(ci[1, 0]), float(ci[1, 1])],
        "p_value": float(fit.pvalues[1]),
        "reference_alpha": alphas[0],
    })
    return out


def _alpha_label(alpha: float) -> str:
    return f"{alpha:+g}"


def _plot_accuracy_by_alpha(
    alpha_vals: List[float], acc_vals: List[float], ci_low: List[float], ci_high: List[float], out_path: str
) -> None:
    if plt is None:
        return
    yerr = [
        [a - lo if np.isfinite(lo) else 0.0 for a, lo in zip(acc_vals, ci_low)],
        [hi - a if np.isfinite(hi) else 0.0 for a, hi in zip(acc_vals, ci_high)],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(alpha_vals, acc_vals, yerr=yerr, marker="o", color="#2b6aa6", linewidth=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Accuracy (95% Wilson interval)")
    ax.set_title("Accuracy vs steering alpha")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_score_by_alpha(
    alpha_vals: List[float], mean_scores: List[float], sem_scores: List[float], out_path: str
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(alpha_vals, mean_scores, yerr=sem_scores, marker="o", color="#b24a3b", linewidth=2)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Mean probe score")
    ax.set_title("Mean probe score vs steering alpha")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_score_by_correctness_by_alpha(
    alpha_vals: List[float],
    mean_correct: List[float],
    mean_incorrect: List[float],
    sem_correct: List[float],
    sem_incorrect: List[float],
    out_path: str,
) -> None:
    if plt is None:
        return
    x = np.arange(len(alpha_vals))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, mean_correct, width, yerr=sem_correct, color="#2c8d5b", label="correct")
    ax.bar(x + width / 2, mean_incorrect, width, yerr=sem_incorrect, color="#b24a3b", label="incorrect")
    ax.set_xticks(x, [str(a) for a in alpha_vals])
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Mean probe score")
    ax.set_title("Probe score by correctness per alpha")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_accuracy_by_coherence(
    alpha_vals: List[float], accuracy_by_rating: Dict[str, List[float]], out_path: str
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, accuracy_by_rating[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs steering alpha (by coherence)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_coherence_counts(alpha_vals: List[float], counts_by_rating: Dict[str, List[int]], out_path: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for rating, color in RATING_COLORS.items():
        ax.plot(alpha_vals, counts_by_rating[rating], marker="o", linewidth=2, color=color, label=rating)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Count")
    ax.set_title("Coherence rating count by alpha")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _render_progress(current: int, total: int, *, width: int = 28) -> str:
    if total <= 0:
        return "[----------------------------] 0/0"
    frac = min(1.0, max(0.0, current / total))
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def _progress_print(
    current: int,
    total: int,
    *,
    label: str,
    enabled: bool,
    last: bool = False,
) -> None:
    if not enabled:
        return
    inline_env = os.environ.get("CP_PROGRESS_INLINE", "").strip().lower()
    if inline_env in {"1", "true", "yes"}:
        inline = True
    elif inline_env in {"0", "false", "no"}:
        inline = False
    else:
        inline = sys.stdout.isatty()
    bar = _render_progress(current, total)
    end = "\n" if (last or not inline) else "\r"
    print(f"{label} {bar}", end=end, flush=True)


def _normalize_items(items: Sequence[Any]) -> List[ExampleItem]:
    normalized: List[ExampleItem] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append({"prompt": item})
    return normalized


def _generate_items(
    generator: ExampleGenerator,
    *,
    num_items: Optional[int],
    generator_kwargs: Optional[Dict[str, Any]],
) -> List[ExampleItem]:
    kwargs = generator_kwargs or {}
    if num_items is None:
        produced = generator(**kwargs)
        if isinstance(produced, list):
            return _normalize_items(produced)
        return _normalize_items([produced])

    items: List[ExampleItem] = []
    for _ in range(int(num_items)):
        produced = generator(**kwargs)
        if isinstance(produced, list):
            if len(produced) == 1:
                items.append(produced[0])
            else:
                raise ValueError(
                    "Generator returned a list while num_items is set. "
                    "Return a single item per call, or omit num_items."
                )
        else:
            items.append(produced)
    return _normalize_items(items)


def _build_prompt_from_item(
    item: ExampleItem,
    *,
    prompt_builder: Optional[Callable[[ExampleItem], PromptLike]],
    prompt_template: Optional[str],
    prompt_prefix: Optional[str],
    prompt_key: str,
    variable_key: str,
) -> PromptLike:
    if prompt_builder is not None:
        return prompt_builder(item)
    if prompt_key in item:
        return item[prompt_key]
    if prompt_template is not None:
        return prompt_template.format(**item)
    if prompt_prefix is not None and variable_key in item:
        return f"{prompt_prefix}{item[variable_key]}"
    raise ValueError("Unable to build prompt: provide prompt_builder, prompt_template, prompt_prefix, or prompt_key.")


def _extract_with_marker(
    completion: str,
    marker: str,
    *,
    marker_position: str = "after",
) -> Optional[str]:
    if not marker:
        return completion.strip()
    idx = completion.find(marker)
    if idx == -1:
        return None
    if marker_position == "before":
        return completion[:idx].strip()
    return completion[idx + len(marker):].strip()


def simple_equality_evaluator(
    completion: str,
    item: ExampleItem,
    expected_key: str = "expected",
    *,
    marker: Optional[str] = None,
    marker_position: str = "after",
) -> Dict[str, Any]:
    expected = item.get(expected_key)
    if expected is None:
        return {"correct": None, "expected": None}

    if marker:
        extracted = _extract_with_marker(completion, marker, marker_position=marker_position)
        if extracted is None:
            return {"correct": False, "expected": expected, "extracted": None}
        candidate = extracted
    else:
        candidate = completion.strip()

    if isinstance(expected, (list, tuple, set)):
        expected_strs = [str(v).strip() for v in expected]
        correct = candidate.strip() in expected_strs
    else:
        correct = candidate.strip() == str(expected).strip()
    return {"correct": bool(correct), "expected": expected, "extracted": candidate}


def run_scored_eval(
    probe: ConceptProbe,
    items: Optional[Sequence[Any]] = None,
    *,
    num_items: Optional[int] = None,
    generator: Optional[ExampleGenerator] = None,
    generator_kwargs: Optional[Dict[str, Any]] = None,
    evaluator: Optional[Evaluator] = None,
    evaluator_kwargs: Optional[Dict[str, Any]] = None,
    prompt_builder: Optional[Callable[[ExampleItem], PromptLike]] = None,
    prompt_template: Optional[str] = None,
    prompt_prefix: Optional[str] = None,
    prompt_key: str = "prompt",
    variable_key: str = "input",
    expected_key: str = "expected",
    include_item: bool = True,
    output_subdir: str = "eval",
    batch_subdir: Optional[str] = None,
    alphas: Optional[List[float]] = None,
    alpha_unit: str = "raw",
    analysis_name: str = "analysis",
    analyze: bool = True,
    rate_coherence: bool = False,
    coherence_model: Optional[str] = None,
    coherence_max_per_request: int = 8,
    coherence_env_path: Optional[str] = None,
    marker: Optional[str] = None,
    marker_position: str = "after",
    **score_kwargs: Any,
) -> EvalRunResult:
    def _status(message: str) -> None:
        if probe.console is not None:
            probe.console.info(message)

    if items is not None and num_items is not None:
        _status("Note: num_items ignored because explicit items were provided.")

    if items is None:
        if generator is None:
            raise ValueError("Provide items or a generator.")
        _status("Generating eval items...")
        items = _generate_items(
            generator,
            num_items=num_items,
            generator_kwargs=generator_kwargs,
        )

    normalized_items = _normalize_items(items)
    _status(f"Prepared {len(normalized_items)} eval items.")
    prompts = [
        _build_prompt_from_item(
            item,
            prompt_builder=prompt_builder,
            prompt_template=prompt_template,
            prompt_prefix=prompt_prefix,
            prompt_key=prompt_key,
            variable_key=variable_key,
        )
        for item in normalized_items
    ]

    if evaluator is None:
        missing_expected = [
            i for i, item in enumerate(normalized_items) if expected_key not in item
        ]
        if missing_expected:
            raise ValueError(
                f"Missing '{expected_key}' for {len(missing_expected)} item(s); provide an evaluator or "
                f"include '{expected_key}' in every item."
            )
        def _default_eval(completion: str, item: ExampleItem) -> Dict[str, Any]:
            return simple_equality_evaluator(
                completion,
                item,
                expected_key=expected_key,
                marker=marker,
                marker_position=marker_position,
            )
        evaluator = _default_eval

    expected_alphas = len(alphas) if alphas is not None else 1
    total_runs = len(prompts) * expected_alphas
    _status(f"Scoring prompts ({len(prompts)} prompts x {expected_alphas} alphas = {total_runs} runs)...")
    results = probe.score_prompts(
        prompts=prompts,
        output_subdir=output_subdir,
        batch_subdir=batch_subdir,
        alphas=alphas,
        alpha_unit=alpha_unit,
        **score_kwargs,
    )

    if results and normalized_items:
        alpha_count = max(1, len(results) // len(normalized_items))
    else:
        alpha_count = len(alphas) if alphas is not None else 1
    per_sample: List[Dict[str, Any]] = []
    total_eval = len(results)
    progress_every = max(1, total_eval // 25) if total_eval else 1
    show_progress = probe.console is not None and probe.console.enabled
    if show_progress and total_eval:
        _progress_print(0, total_eval, label="Evaluating", enabled=True)
    for idx, rec in enumerate(results):
        item_idx = idx // alpha_count
        if item_idx >= len(normalized_items):
            break
        item = normalized_items[item_idx]
        completion = rec.get("completion", "")
        if evaluator_kwargs:
            eval_result = evaluator(completion, item, **evaluator_kwargs)
        else:
            eval_result = evaluator(completion, item)

        record = {
            "item_index": int(item_idx),
            "alpha": float(rec.get("alpha", 0.0)),
            "prompt": rec.get("prompt"),
            "completion": completion,
            "score_mean": _score_mean_from_result(rec),
            "npz_path": rec.get("npz_path"),
            "html_path": rec.get("html_path"),
        }
        if expected_key in item:
            record["expected"] = item.get(expected_key)
        if include_item:
            record["item"] = item
        if isinstance(eval_result, dict):
            record.update(eval_result)
        per_sample.append(record)
        if total_eval and ((idx + 1) % progress_every == 0 or (idx + 1) == total_eval):
            _progress_print(
                idx + 1,
                total_eval,
                label="Evaluating",
                enabled=show_progress,
                last=(idx + 1) == total_eval,
            )

    if results:
        first_npz_path = results[0].get("npz_path")
        if isinstance(first_npz_path, str):
            batch_dir = Path(first_npz_path).resolve().parent
        else:
            batch_dir = Path(results[0].get("batch_dir", ".")).resolve()
    else:
        batch_dir = Path(".")
    analysis_dir = None
    if analyze and results:
        _status("Writing analysis...")
        analysis_dir = str(batch_dir / analysis_name)
        if rate_coherence:
            _status("Rating coherence (best-effort)...")
            rate_batch_coherence_safe(
                str(batch_dir),
                max_elements_per_request=coherence_max_per_request,
                model=coherence_model,
                env_path=coherence_env_path,
            )
            _status("Rebuilding analysis with coherence data...")
        _write_analysis(batch_dir, per_sample, analysis_name=analysis_name)
        _status(f"Analysis complete: {analysis_dir}")

    return EvalRunResult(results=results, per_sample=per_sample, batch_dir=str(batch_dir), analysis_dir=analysis_dir)


def run_multi_scored_eval(
    probes: Sequence["ConceptProbe"],
    items: Optional[Sequence[Any]] = None,
    *,
    num_items: Optional[int] = None,
    generator: Optional[ExampleGenerator] = None,
    generator_kwargs: Optional[Dict[str, Any]] = None,
    evaluator: Optional[Evaluator] = None,
    evaluator_kwargs: Optional[Dict[str, Any]] = None,
    prompt_builder: Optional[Callable[[ExampleItem], PromptLike]] = None,
    prompt_template: Optional[str] = None,
    prompt_prefix: Optional[str] = None,
    prompt_key: str = "prompt",
    variable_key: str = "input",
    expected_key: str = "expected",
    include_item: bool = True,
    output_root: str = "outputs_multi",
    project_name: str = "multi_probe",
    run_name: Optional[str] = None,
    output_subdir: str = "eval",
    batch_subdir: Optional[str] = None,
    alphas: Optional[List[float]] = None,
    alpha_unit: str = "raw",
    analysis_name: str = "analysis",
    analyze: bool = True,
    rate_coherence: bool = False,
    coherence_model: Optional[str] = None,
    coherence_max_per_request: int = 8,
    coherence_env_path: Optional[str] = None,
    **score_kwargs: Any,
) -> EvalRunResult:
    from .multi_probe import multi_probe_score_prompts

    if items is not None and num_items is not None:
        if probes and probes[0].console is not None:
            probes[0].console.info("Note: num_items ignored because explicit items were provided.")

    if items is None:
        if generator is None:
            raise ValueError("Provide items or a generator.")
        if probes and probes[0].console is not None:
            probes[0].console.info("Generating eval items...")
        items = _generate_items(
            generator,
            num_items=num_items,
            generator_kwargs=generator_kwargs,
        )

    normalized_items = _normalize_items(items)
    if probes and probes[0].console is not None:
        probes[0].console.info(f"Prepared {len(normalized_items)} eval items.")

    prompts = [
        _build_prompt_from_item(
            item,
            prompt_builder=prompt_builder,
            prompt_template=prompt_template,
            prompt_prefix=prompt_prefix,
            prompt_key=prompt_key,
            variable_key=variable_key,
        )
        for item in normalized_items
    ]

    if evaluator is None:
        missing_expected = [
            i for i, item in enumerate(normalized_items) if expected_key not in item
        ]
        if missing_expected:
            raise ValueError(
                f"Missing '{expected_key}' for {len(missing_expected)} item(s); provide an evaluator or "
                f"include '{expected_key}' in every item."
            )

        def _default_eval(completion: str, item: ExampleItem) -> Dict[str, Any]:
            return simple_equality_evaluator(
                completion,
                item,
                expected_key=expected_key,
            )

        evaluator = _default_eval

    if probes and probes[0].console is not None:
        expected_alphas = len(alphas) if alphas is not None else 1
        total_runs = len(prompts) * expected_alphas
        probes[0].console.info(
            f"Scoring prompts (multi-probe) ({len(prompts)} prompts x {expected_alphas} alphas = {total_runs} runs)..."
        )

    result_payload = multi_probe_score_prompts(
        list(probes),
        prompts,
        output_root=output_root,
        project_name=project_name,
        run_name=run_name,
        output_subdir=output_subdir,
        batch_subdir=batch_subdir,
        alphas=alphas,
        alpha_unit=alpha_unit,
        **score_kwargs,
    )
    results = result_payload.get("results", [])

    if results and normalized_items:
        alpha_count = max(1, len(results) // len(normalized_items))
    else:
        alpha_count = len(alphas) if alphas is not None else 1
    per_sample: List[Dict[str, Any]] = []
    total_eval = len(results)
    progress_every = max(1, total_eval // 25) if total_eval else 1
    show_progress = probes and probes[0].console is not None and probes[0].console.enabled
    if show_progress and total_eval:
        _progress_print(0, total_eval, label="Evaluating", enabled=True)
    for idx, rec in enumerate(results):
        item_idx = idx // alpha_count
        if item_idx >= len(normalized_items):
            break
        item = normalized_items[item_idx]
        completion = rec.get("completion", "")
        if evaluator_kwargs:
            eval_result = evaluator(completion, item, **evaluator_kwargs)
        else:
            eval_result = evaluator(completion, item)

        score_map = _score_mean_by_probe_from_result(rec)
        record = {
            "item_index": int(item_idx),
            "alpha": float(rec.get("alpha", 0.0)),
            "prompt": rec.get("prompt"),
            "completion": completion,
            "score_mean_by_probe": score_map,
            "npz_path": rec.get("npz_path"),
            "html_path": rec.get("html_path"),
            "probe_names": rec.get("probe_names"),
        }
        if expected_key in item:
            record["expected"] = item.get(expected_key)
        if include_item:
            record["item"] = item
        if isinstance(eval_result, dict):
            record.update(eval_result)
        per_sample.append(record)
        if total_eval and ((idx + 1) % progress_every == 0 or (idx + 1) == total_eval):
            _progress_print(
                idx + 1,
                total_eval,
                label="Evaluating",
                enabled=show_progress,
                last=(idx + 1) == total_eval,
            )

    if results:
        first_npz_path = results[0].get("npz_path")
        if isinstance(first_npz_path, str):
            batch_dir = Path(first_npz_path).resolve().parent
        else:
            batch_dir = Path(results[0].get("batch_dir", ".")).resolve()
    else:
        batch_dir = Path(".")
    analysis_dir = None
    if analyze and results:
        if probes and probes[0].console is not None:
            probes[0].console.info("Writing analysis...")
        analysis_dir = str(batch_dir / analysis_name)
        if rate_coherence:
            if probes and probes[0].console is not None:
                probes[0].console.info("Rating coherence (best-effort)...")
            rate_batch_coherence_safe(
                str(batch_dir),
                max_elements_per_request=coherence_max_per_request,
                model=coherence_model,
                env_path=coherence_env_path,
            )
            if probes and probes[0].console is not None:
                probes[0].console.info("Rebuilding analysis with coherence data...")
        _write_analysis_multi(batch_dir, per_sample, analysis_name=analysis_name)
        if probes and probes[0].console is not None:
            probes[0].console.info(f"Analysis complete: {analysis_dir}")

    return EvalRunResult(results=results, per_sample=per_sample, batch_dir=str(batch_dir), analysis_dir=analysis_dir)


def _load_per_sample(per_sample_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(per_sample_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _load_coherence_ratings(batch_dir: Path) -> Dict[str, str]:
    ratings_path = batch_dir / "coherence_rating.json"
    if not ratings_path.exists():
        return {}
    data = json.loads(ratings_path.read_text(encoding="utf-8"))
    ratings: Dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        example = entry.get("example")
        rating = entry.get("rating")
        if isinstance(example, str) and isinstance(rating, str):
            ratings[str((batch_dir / example).resolve())] = rating
            # cwd-independent key: per_sample.json may store npz paths relative to a
            # different working directory, so also match on <batch_name>/<file_name>.
            ratings[f"{batch_dir.name}/{example}"] = rating
    return ratings


def _is_multi_probe_per_sample(per_sample: List[Dict[str, Any]]) -> bool:
    for item in per_sample:
        if isinstance(item.get("score_mean_by_probe"), dict):
            return True
    return False


def _accuracy_by_alpha(by_alpha: Dict[float, List[Dict[str, Any]]], alpha_vals: List[float]) -> List[Dict[str, Any]]:
    out = []
    for alpha in alpha_vals:
        rows = [r for r in by_alpha.get(alpha, []) if _is_bool(r.get("correct"))]
        k = sum(1 for r in rows if r.get("correct") is True)
        lo, hi = _wilson_interval(k, len(rows))
        out.append({
            "alpha": float(alpha),
            "accuracy": float(k / len(rows)) if rows else float("nan"),
            "n": int(len(rows)),
            "ci_low": lo,
            "ci_high": hi,
        })
    return out


def _compute_stats(per_sample: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[float]]:
    """Accuracy and probe score per alpha, correct vs incorrect within each alpha, and a GEE of
    correctness on score across alphas (alpha as covariate, rows clustered by item).

    Correct and incorrect rows are never pooled across alphas: steering moves both the score
    and the accuracy, and the same item recurs at every alpha.
    """
    alpha_vals = sorted({float(item["alpha"]) for item in per_sample if "alpha" in item})
    by_alpha: Dict[float, List[Dict[str, Any]]] = {}
    for row in per_sample:
        if "alpha" not in row:
            continue
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    score_by_alpha: List[Dict[str, Any]] = []
    per_alpha_correctness: List[Dict[str, Any]] = []
    for alpha in alpha_vals:
        all_rows = by_alpha.get(alpha, [])
        # Score stats use every row with a finite score, independent of correctness labels:
        # a custom evaluator that emits no "correct" field must not blank the score analysis.
        scores = [s for s in (_finite_float(r.get("score_mean")) for r in all_rows) if np.isfinite(s)]
        score_by_alpha.append({
            "alpha": float(alpha),
            "mean_score": float(np.mean(scores)) if scores else float("nan"),
            "sem": _sem(scores),
            "n": int(len(scores)),
        })
        labelled = [r for r in all_rows if _is_bool(r.get("correct"))]
        groups = {}
        for flag in (True, False):
            groups[flag] = [
                s for s in (_finite_float(r.get("score_mean")) for r in labelled if r.get("correct") is flag)
                if np.isfinite(s)
            ]
        per_alpha_correctness.append({
            "alpha": float(alpha),
            "mean_correct": float(np.mean(groups[True])) if groups[True] else float("nan"),
            "mean_incorrect": float(np.mean(groups[False])) if groups[False] else float("nan"),
            "sem_correct": _sem(groups[True]),
            "sem_incorrect": _sem(groups[False]),
            "cohen_d": _cohen_d(np.array(groups[True]), np.array(groups[False])),
            "n_correct": int(len(groups[True])),
            "n_incorrect": int(len(groups[False])),
        })

    stats = {
        "accuracy_by_alpha": _accuracy_by_alpha(by_alpha, alpha_vals),
        "score_by_alpha": score_by_alpha,
        "correct_vs_incorrect_by_alpha": per_alpha_correctness,
        "correctness_gee": _correctness_gee(per_sample),
    }
    return stats, alpha_vals


def _compute_accuracy_stats(per_sample: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[float]]:
    alpha_vals = sorted({float(item["alpha"]) for item in per_sample if "alpha" in item})
    by_alpha: Dict[float, List[Dict[str, Any]]] = {}
    for row in per_sample:
        if "alpha" not in row:
            continue
        by_alpha.setdefault(float(row["alpha"]), []).append(row)
    return _accuracy_by_alpha(by_alpha, alpha_vals), alpha_vals


def _compute_coherence(
    per_sample: List[Dict[str, Any]], alpha_vals: List[float], ratings: Dict[str, str]
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    counts_by_rating: Dict[str, List[int]] = {k: [0] * len(alpha_vals) for k in RATING_COLORS}
    accuracy_by_rating: Dict[str, List[float]] = {k: [float("nan")] * len(alpha_vals) for k in RATING_COLORS}
    by_alpha: Dict[float, List[Dict[str, Any]]] = {}
    for row in per_sample:
        if "alpha" not in row:
            continue
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    for i, alpha in enumerate(alpha_vals):
        rows = [r for r in by_alpha.get(alpha, []) if _is_bool(r.get("correct"))]
        for rating in RATING_COLORS:
            rated_rows = []
            for item in rows:
                npz_path = item.get("npz_path")
                if not isinstance(npz_path, str):
                    continue
                p = Path(npz_path)
                item_rating = ratings.get(str(p.resolve()))
                if item_rating is None:
                    item_rating = ratings.get(f"{p.parent.name}/{p.name}")
                if item_rating == rating:
                    rated_rows.append(item)
            counts_by_rating[rating][i] = len(rated_rows)
            if rated_rows:
                accuracy_by_rating[rating][i] = sum(1 for r in rated_rows if r.get("correct") is True) / len(rated_rows)
    return counts_by_rating, accuracy_by_rating


def _write_analysis(
    batch_dir: Path,
    per_sample: List[Dict[str, Any]],
    *,
    analysis_name: str = "analysis",
    coherence_ratings: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    analysis_dir = batch_dir / analysis_name
    plots_dir = analysis_dir / "plots"
    ensure_dir(str(plots_dir))

    json_dump(str(analysis_dir / "per_sample.json"), {"items": per_sample})

    stats, alpha_vals = _compute_stats(per_sample)
    json_dump(str(analysis_dir / "stats.json"), stats)

    _plot_accuracy_by_alpha(
        alpha_vals,
        [r["accuracy"] for r in stats["accuracy_by_alpha"]],
        [r["ci_low"] for r in stats["accuracy_by_alpha"]],
        [r["ci_high"] for r in stats["accuracy_by_alpha"]],
        str(plots_dir / "accuracy_vs_alpha.png"),
    )
    _plot_score_by_alpha(
        alpha_vals,
        [r["mean_score"] for r in stats["score_by_alpha"]],
        [r["sem"] for r in stats["score_by_alpha"]],
        str(plots_dir / "score_vs_alpha.png"),
    )

    per_alpha_correctness = stats["correct_vs_incorrect_by_alpha"]
    _plot_score_by_correctness_by_alpha(
        alpha_vals,
        [r["mean_correct"] for r in per_alpha_correctness],
        [r["mean_incorrect"] for r in per_alpha_correctness],
        [r["sem_correct"] for r in per_alpha_correctness],
        [r["sem_incorrect"] for r in per_alpha_correctness],
        str(plots_dir / "score_by_correctness_by_alpha.png"),
    )

    if coherence_ratings is None:
        coherence_ratings = _load_coherence_ratings(batch_dir)
    if coherence_ratings:
        counts_by_rating, accuracy_by_rating = _compute_coherence(per_sample, alpha_vals, coherence_ratings)
        _plot_accuracy_by_coherence(
            alpha_vals, accuracy_by_rating, str(plots_dir / "accuracy_vs_alpha_by_coherence.png")
        )
        _plot_coherence_counts(
            alpha_vals, counts_by_rating, str(plots_dir / "coherence_counts_by_alpha.png")
        )

    return {"analysis_dir": str(analysis_dir), "plots_dir": str(plots_dir)}


def _collect_probe_names(per_sample: List[Dict[str, Any]]) -> List[str]:
    for item in per_sample:
        probe_names = item.get("probe_names")
        if isinstance(probe_names, list) and probe_names:
            return [str(p) for p in probe_names]
    names: List[str] = []
    for item in per_sample:
        score_map = item.get("score_mean_by_probe")
        if isinstance(score_map, dict):
            for key in score_map.keys():
                if key not in names:
                    names.append(str(key))
    return names


def _write_analysis_multi(
    batch_dir: Path,
    per_sample: List[Dict[str, Any]],
    *,
    analysis_name: str = "analysis",
    coherence_ratings: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    analysis_dir = batch_dir / analysis_name
    plots_dir = analysis_dir / "plots"
    ensure_dir(str(plots_dir))

    json_dump(str(analysis_dir / "per_sample.json"), {"items": per_sample})

    probe_names = _collect_probe_names(per_sample)
    per_probe_stats: Dict[str, Any] = {}
    accuracy_by_alpha, alpha_vals_master = _compute_accuracy_stats(per_sample)

    if accuracy_by_alpha:
        _plot_accuracy_by_alpha(
            alpha_vals_master,
            [r["accuracy"] for r in accuracy_by_alpha],
            [r["ci_low"] for r in accuracy_by_alpha],
            [r["ci_high"] for r in accuracy_by_alpha],
            str(plots_dir / "accuracy_vs_alpha.png"),
        )

    for probe in probe_names:
        probe_rows: List[Dict[str, Any]] = []
        for row in per_sample:
            score_map = row.get("score_mean_by_probe", {})
            if not isinstance(score_map, dict):
                continue
            if probe not in score_map:
                continue
            probe_rows.append(
                {
                    "alpha": row.get("alpha"),
                    "item_index": row.get("item_index"),
                    "correct": row.get("correct"),
                    "score_mean": score_map.get(probe),
                }
            )
        if not probe_rows:
            continue
        stats, alpha_vals = _compute_stats(probe_rows)
        per_probe_stats[probe] = {
            "score_by_alpha": stats["score_by_alpha"],
            "correct_vs_incorrect_by_alpha": stats["correct_vs_incorrect_by_alpha"],
            "correctness_gee": stats["correctness_gee"],
        }
        if not alpha_vals_master:
            alpha_vals_master = alpha_vals

        probe_slug = safe_slug(probe) or f"probe_{len(per_probe_stats)}"
        probe_plot_dir = plots_dir / probe_slug
        ensure_dir(str(probe_plot_dir))

        _plot_score_by_alpha(
            alpha_vals,
            [r["mean_score"] for r in stats["score_by_alpha"]],
            [r["sem"] for r in stats["score_by_alpha"]],
            str(probe_plot_dir / "score_vs_alpha.png"),
        )
        per_alpha_correctness = stats["correct_vs_incorrect_by_alpha"]
        _plot_score_by_correctness_by_alpha(
            alpha_vals,
            [r["mean_correct"] for r in per_alpha_correctness],
            [r["mean_incorrect"] for r in per_alpha_correctness],
            [r["sem_correct"] for r in per_alpha_correctness],
            [r["sem_incorrect"] for r in per_alpha_correctness],
            str(probe_plot_dir / "score_by_correctness_by_alpha.png"),
        )

    stats_out = {
        "probe_names": probe_names,
        "accuracy_by_alpha": accuracy_by_alpha,
        "per_probe": per_probe_stats,
    }
    json_dump(str(analysis_dir / "stats.json"), stats_out)

    if coherence_ratings is None:
        coherence_ratings = _load_coherence_ratings(batch_dir)
    if coherence_ratings:
        if not alpha_vals_master:
            alpha_vals_master = sorted({float(item["alpha"]) for item in per_sample if "alpha" in item})
        counts_by_rating, accuracy_by_rating = _compute_coherence(
            per_sample, alpha_vals_master, coherence_ratings
        )
        _plot_accuracy_by_coherence(
            alpha_vals_master, accuracy_by_rating, str(plots_dir / "accuracy_vs_alpha_by_coherence.png")
        )
        _plot_coherence_counts(
            alpha_vals_master, counts_by_rating, str(plots_dir / "coherence_counts_by_alpha.png")
        )

    return {"analysis_dir": str(analysis_dir), "plots_dir": str(plots_dir)}


def rate_batch_coherence_safe(
    batch_dir: str,
    *,
    max_elements_per_request: int = 8,
    model: Optional[str] = None,
    env_path: Optional[str] = None,
) -> bool:
    try:
        from .coherence import rate_batch_coherence
    except Exception as exc:
        print(f"Warning: coherence rater unavailable: {exc}")
        return False

    try:
        rate_batch_coherence(
            batch_dir,
            max_elements_per_request=max_elements_per_request,
            model=model,
            env_path=env_path,
        )
        return True
    except Exception as exc:
        print(f"Warning: coherence rating failed for {batch_dir}: {exc}")
        return False


def rehydrate_batch_analysis(
    batch_dir: str,
    *,
    analysis_name: str = "analysis",
    rate_coherence: bool = False,
    coherence_model: Optional[str] = None,
    coherence_max_per_request: int = 8,
    coherence_env_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    batch_path = Path(batch_dir).resolve()
    per_sample_path = batch_path / analysis_name / "per_sample.json"
    if not per_sample_path.exists():
        print(f"Warning: missing per_sample.json in {batch_path}")
        return None

    if rate_coherence:
        rate_batch_coherence_safe(
            str(batch_path),
            max_elements_per_request=coherence_max_per_request,
            model=coherence_model,
            env_path=coherence_env_path,
        )

    per_sample = _load_per_sample(per_sample_path)
    if _is_multi_probe_per_sample(per_sample):
        return _write_analysis_multi(batch_path, per_sample, analysis_name=analysis_name)
    return _write_analysis(batch_path, per_sample, analysis_name=analysis_name)


def aggregate_eval_batches(
    eval_dir: str,
    *,
    output_name: str = "analysis_all",
    batch_dirs: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    eval_path = Path(eval_dir).resolve()
    if not eval_path.exists():
        print(f"Warning: eval dir not found: {eval_path}")
        return None

    batches: List[Path] = []
    if batch_dirs is None:
        batches = sorted(
            [p for p in eval_path.iterdir() if p.is_dir() and p.name.startswith("batch_")],
            key=lambda p: p.name,
        )
    else:
        for raw in batch_dirs:
            path = Path(raw).resolve()
            batches.append(path)

    if not batches:
        print(f"Warning: no batch directories found under {eval_path}")
        return None

    per_sample: List[Dict[str, Any]] = []
    ratings: Dict[str, str] = {}
    for batch in batches:
        per_sample_path = batch / "analysis" / "per_sample.json"
        if not per_sample_path.exists():
            print(f"Warning: missing per_sample.json in {batch}")
            continue
        items = _load_per_sample(per_sample_path)
        for item in items:
            item["batch_name"] = batch.name
            item["batch_dir"] = str(batch)
        per_sample.extend(items)
        ratings.update(_load_coherence_ratings(batch))

    if not per_sample:
        print(f"Warning: no per-sample items found under {eval_path}")
        return None

    if _is_multi_probe_per_sample(per_sample):
        return _write_analysis_multi(
            eval_path,
            per_sample,
            analysis_name=output_name,
            coherence_ratings=ratings if ratings else None,
        )
    return _write_analysis(
        eval_path,
        per_sample,
        analysis_name=output_name,
        coherence_ratings=ratings if ratings else None,
    )


__all__ = [
    "EvalRunResult",
    "run_scored_eval",
    "run_multi_scored_eval",
    "simple_equality_evaluator",
    "rehydrate_batch_analysis",
    "aggregate_eval_batches",
    "rate_batch_coherence_safe",
]
