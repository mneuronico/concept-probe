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


def _mean_completion_score(npz_path: str) -> float:
    data = np.load(npz_path)
    scores = data["scores_agg"]
    prompt_len = int(data["prompt_len"][0]) if "prompt_len" in data else 0
    if prompt_len < scores.shape[0]:
        span = scores[prompt_len:]
    else:
        span = scores
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
    prompt_len = int(data["prompt_len"][0]) if "prompt_len" in data else 0

    if scores.ndim == 1:
        span = scores[prompt_len:] if prompt_len < scores.shape[0] else scores
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
        row = scores[idx]
        span = row[prompt_len:] if prompt_len < row.shape[0] else row
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


def _alpha_label(alpha: float) -> str:
    return f"{alpha:+g}"


def _plot_accuracy_by_alpha(alpha_vals: List[float], acc_vals: List[float], n_vals: List[int], out_path: str) -> None:
    if plt is None:
        return
    se = [np.sqrt(a * (1 - a) / n) if n > 0 else float("nan") for a, n in zip(acc_vals, n_vals)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(alpha_vals, acc_vals, yerr=se, marker="o", color="#2b6aa6", linewidth=2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Accuracy")
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


def _plot_score_by_correctness(
    mean_correct: float,
    mean_incorrect: float,
    sem_correct: float,
    sem_incorrect: float,
    out_path: str,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [mean_correct, mean_incorrect], yerr=[sem_correct, sem_incorrect], color=["#2c8d5b", "#b24a3b"])
    ax.set_xticks([0, 1], ["correct", "incorrect"])
    ax.set_ylabel("Mean probe score")
    ax.set_title("Probe score by correctness")
    ax.grid(True, axis="y", alpha=0.25)
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
    rate_coherence: bool = True,
    coherence_model: str = "openai/gpt-oss-20b",
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
    rate_coherence: bool = True,
    coherence_model: str = "openai/gpt-oss-20b",
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
    return ratings


def _is_multi_probe_per_sample(per_sample: List[Dict[str, Any]]) -> bool:
    for item in per_sample:
        if isinstance(item.get("score_mean_by_probe"), dict):
            return True
    return False


def _compute_stats(per_sample: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[float]]:
    alpha_vals = sorted({float(item["alpha"]) for item in per_sample if "alpha" in item})
    by_alpha: Dict[float, List[Dict[str, Any]]] = {}
    for row in per_sample:
        if "alpha" not in row:
            continue
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    accuracy_vals: List[float] = []
    mean_score_vals: List[float] = []
    sem_score_vals: List[float] = []
    counts: List[int] = []
    for alpha in alpha_vals:
        rows = [r for r in by_alpha.get(alpha, []) if _is_bool(r.get("correct"))]
        counts.append(len(rows))
        if rows:
            acc = sum(1 for r in rows if r.get("correct") is True) / len(rows)
        else:
            acc = float("nan")
        accuracy_vals.append(float(acc))
        scores = [
            float(r.get("score_mean"))
            for r in rows
            if np.isfinite(r.get("score_mean", float("nan")))
        ]
        mean_score_vals.append(float(np.mean(scores)) if scores else float("nan"))
        sem_score_vals.append(_sem(scores))

    correct_scores = np.array(
        [
            float(r.get("score_mean"))
            for r in per_sample
            if r.get("correct") is True and np.isfinite(r.get("score_mean", float("nan")))
        ],
        dtype=np.float32,
    )
    incorrect_scores = np.array(
        [
            float(r.get("score_mean"))
            for r in per_sample
            if r.get("correct") is False and np.isfinite(r.get("score_mean", float("nan")))
        ],
        dtype=np.float32,
    )
    mean_correct = float(np.mean(correct_scores)) if correct_scores.size else float("nan")
    mean_incorrect = float(np.mean(incorrect_scores)) if incorrect_scores.size else float("nan")
    sem_correct = _sem(correct_scores.tolist())
    sem_incorrect = _sem(incorrect_scores.tolist())

    mean_correct_by_alpha: List[float] = []
    mean_incorrect_by_alpha: List[float] = []
    sem_correct_by_alpha: List[float] = []
    sem_incorrect_by_alpha: List[float] = []
    per_alpha_correctness: List[Dict[str, Any]] = []
    for alpha in alpha_vals:
        rows = [r for r in by_alpha.get(alpha, []) if _is_bool(r.get("correct"))]
        correct_scores_alpha = [
            float(r.get("score_mean"))
            for r in rows
            if r.get("correct") is True and np.isfinite(r.get("score_mean", float("nan")))
        ]
        incorrect_scores_alpha = [
            float(r.get("score_mean"))
            for r in rows
            if r.get("correct") is False and np.isfinite(r.get("score_mean", float("nan")))
        ]
        mean_c = float(np.mean(correct_scores_alpha)) if correct_scores_alpha else float("nan")
        mean_i = float(np.mean(incorrect_scores_alpha)) if incorrect_scores_alpha else float("nan")
        sem_c = _sem(correct_scores_alpha)
        sem_i = _sem(incorrect_scores_alpha)
        mean_correct_by_alpha.append(mean_c)
        mean_incorrect_by_alpha.append(mean_i)
        sem_correct_by_alpha.append(sem_c)
        sem_incorrect_by_alpha.append(sem_i)
        per_alpha_correctness.append(
            {
                "alpha": float(alpha),
                "mean_correct": mean_c,
                "mean_incorrect": mean_i,
                "sem_correct": sem_c,
                "sem_incorrect": sem_i,
                "n_correct": int(len(correct_scores_alpha)),
                "n_incorrect": int(len(incorrect_scores_alpha)),
            }
        )

    stats = {
        "accuracy_by_alpha": [
            {"alpha": float(a), "accuracy": float(acc), "n": int(n)}
            for a, acc, n in zip(alpha_vals, accuracy_vals, counts)
        ],
        "score_by_alpha": [
            {"alpha": float(a), "mean_score": float(m), "sem": float(s), "n": int(n)}
            for a, m, s, n in zip(alpha_vals, mean_score_vals, sem_score_vals, counts)
        ],
        "correct_vs_incorrect": {
            "mean_correct": mean_correct,
            "mean_incorrect": mean_incorrect,
            "sem_correct": sem_correct,
            "sem_incorrect": sem_incorrect,
            "cohen_d": _cohen_d(correct_scores, incorrect_scores),
        },
        "correct_vs_incorrect_by_alpha": per_alpha_correctness,
    }
    return stats, alpha_vals


def _compute_accuracy_stats(per_sample: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[float]]:
    alpha_vals = sorted({float(item["alpha"]) for item in per_sample if "alpha" in item})
    by_alpha: Dict[float, List[Dict[str, Any]]] = {}
    for row in per_sample:
        if "alpha" not in row:
            continue
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    accuracy_vals: List[float] = []
    counts: List[int] = []
    for alpha in alpha_vals:
        rows = [r for r in by_alpha.get(alpha, []) if _is_bool(r.get("correct"))]
        counts.append(len(rows))
        if rows:
            acc = sum(1 for r in rows if r.get("correct") is True) / len(rows)
        else:
            acc = float("nan")
        accuracy_vals.append(float(acc))

    stats = [
        {"alpha": float(a), "accuracy": float(acc), "n": int(n)}
        for a, acc, n in zip(alpha_vals, accuracy_vals, counts)
    ]
    return stats, alpha_vals


def _compute_coherence(
    per_sample: List[Dict[str, Any]], alpha_vals: List[float], ratings: Dict[str, str]
) -> Tuple[Dict[str, List[int]], Dict[str, List[float]]]:
    counts_by_rating: Dict[str, List[int]] = {k: [0] * len(alpha_vals) for k in RATING_COLORS}
    accuracy_by_rating: Dict[str, List[float]] = {k: [float("nan")] * len(alpha_vals) for k in RATING_COLORS}
    by_alpha: Dict[float, List[Dict[str, Any]]] = {}
    for row in per_sample:
        by_alpha.setdefault(float(row["alpha"]), []).append(row)

    for i, alpha in enumerate(alpha_vals):
        rows = [r for r in by_alpha.get(alpha, []) if _is_bool(r.get("correct"))]
        for rating in RATING_COLORS:
            rated_rows = []
            for item in rows:
                npz_path = item.get("npz_path")
                if not isinstance(npz_path, str):
                    continue
                key = str(Path(npz_path).resolve())
                if ratings.get(key) == rating:
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
        [r["n"] for r in stats["accuracy_by_alpha"]],
        str(plots_dir / "accuracy_vs_alpha.png"),
    )
    _plot_score_by_alpha(
        alpha_vals,
        [r["mean_score"] for r in stats["score_by_alpha"]],
        [r["sem"] for r in stats["score_by_alpha"]],
        str(plots_dir / "score_vs_alpha.png"),
    )

    correct_stats = stats["correct_vs_incorrect"]
    _plot_score_by_correctness(
        correct_stats["mean_correct"],
        correct_stats["mean_incorrect"],
        correct_stats["sem_correct"],
        correct_stats["sem_incorrect"],
        str(plots_dir / "score_by_correctness.png"),
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
            [r["n"] for r in accuracy_by_alpha],
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
                    "correct": row.get("correct"),
                    "score_mean": score_map.get(probe),
                }
            )
        if not probe_rows:
            continue
        stats, alpha_vals = _compute_stats(probe_rows)
        per_probe_stats[probe] = {
            "score_by_alpha": stats["score_by_alpha"],
            "correct_vs_incorrect": stats["correct_vs_incorrect"],
            "correct_vs_incorrect_by_alpha": stats["correct_vs_incorrect_by_alpha"],
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
        correct_stats = stats["correct_vs_incorrect"]
        _plot_score_by_correctness(
            correct_stats["mean_correct"],
            correct_stats["mean_incorrect"],
            correct_stats["sem_correct"],
            correct_stats["sem_incorrect"],
            str(probe_plot_dir / "score_by_correctness.png"),
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
    model: str = "openai/gpt-oss-20b",
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
    rate_coherence: bool = True,
    coherence_model: str = "openai/gpt-oss-20b",
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
