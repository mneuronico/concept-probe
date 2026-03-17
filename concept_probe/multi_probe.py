import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .logger import JsonlLogger
from .modeling import apply_chat, attention_mask_from_ids
from .probe import (
    ConceptProbe,
    PromptLike,
    _pack_generation_logits_arrays,
    _alpha_label,
    _decode_tokens,
    _mix_random_control_seed,
    _normalize_messages,
    _prompt_slug_any,
    _random_vectors_same_norm,
    _resolve_random_control_seed,
    forward_hidden_states_all_layers,
    token_scores_from_hs_layers,
)
from .steering import MultiLayerSteererLayerwise
from .utils import ensure_dir, json_dump, now_tag, safe_slug
from .visuals import render_batch_heatmap_multi, render_token_heatmap_multi, segment_token_scores


def _resolve_steer_probe(
    probes: List[ConceptProbe], steer_probe: Optional[Union[int, str, ConceptProbe]]
) -> Optional[ConceptProbe]:
    if steer_probe is None:
        return None
    if isinstance(steer_probe, ConceptProbe):
        return steer_probe
    if isinstance(steer_probe, int):
        if steer_probe < 0 or steer_probe >= len(probes):
            raise IndexError(f"steer_probe index out of range: {steer_probe}")
        return probes[steer_probe]
    target = safe_slug(str(steer_probe))
    for probe in probes:
        if safe_slug(str(probe.concept.name)) == target:
            return probe
    raise ValueError(f"steer_probe '{steer_probe}' not found in probes.")


def _render_progress(current: int, total: int, *, width: int = 28) -> str:
    if total <= 0:
        return "[----------------------------] 0/0"
    frac = min(1.0, max(0.0, current / total))
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def _progress_print(current: int, total: int, *, label: str, enabled: bool, last: bool = False) -> None:
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


def _steer_layer_list(
    probe: ConceptProbe,
    steer_layers: Optional[Union[str, List[int]]],
    steer_window_radius: Optional[int],
) -> List[int]:
    if steer_layers is None:
        steer_layers = probe.config["steering"]["steer_layers"]
    if steer_window_radius is None:
        steer_window_radius = probe.config["steering"]["steer_window_radius"]

    if steer_layers == "probe":
        return [int(probe.best_layer)]
    if steer_layers == "window":
        lo = max(0, int(probe.best_layer) - int(steer_window_radius))
        hi = min(probe.model_bundle.num_layers - 1, int(probe.best_layer) + int(steer_window_radius))
        return list(range(lo, hi + 1))
    if isinstance(steer_layers, list):
        return list(map(int, steer_layers))
    return list(map(int, steer_layers))


def multi_probe_score_prompts(
    probes: List[ConceptProbe],
    prompts: List[PromptLike],
    *,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    greedy: Optional[bool] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    layer_selection: Optional[Dict[str, Any]] = None,
    output_root: str = "outputs_multi",
    project_name: str = "multi_probe",
    run_name: Optional[str] = None,
    output_subdir: str = "scores",
    alphas: Optional[List[float]] = None,
    alpha_unit: str = "raw",
    steer_probe: Optional[Union[int, str, ConceptProbe]] = None,
    steer_layers: Optional[Union[str, List[int]]] = None,
    steer_window_radius: Optional[int] = None,
    steer_distribute: Optional[bool] = None,
    save_html: Optional[bool] = None,
    save_segments: Optional[bool] = None,
    save_generation_logits: Optional[bool] = None,
    generation_logits_top_k: Optional[int] = None,
    generation_logits_dtype: Optional[str] = None,
    batch_subdir: Optional[str] = None,
    random_scoring_vector: bool = False,
    random_steering_vector: bool = False,
    random_vector_seed: Optional[int] = None,
) -> Dict[str, Any]:
    if not probes:
        raise ValueError("probes must be a non-empty list of ConceptProbe objects.")
    if not prompts:
        return {"run_dir": None, "results": []}

    for probe in probes:
        probe._require_trained()

    model_bundle = probes[0].model_bundle
    for probe in probes[1:]:
        if probe.model_bundle.model_id != model_bundle.model_id:
            raise ValueError("All probes must use the same model_id for multi-probe scoring.")

    steer_probe_obj = _resolve_steer_probe(probes, steer_probe)

    if alphas is None or len(alphas) == 0:
        alphas = [0.0]
    nonzero_alpha_requested = any(abs(float(a)) > 1e-12 for a in alphas)
    if steer_probe_obj is None and nonzero_alpha_requested:
        steer_probe_obj = probes[0]
        probes[0]._warn(
            "Non-zero alphas provided without steer_probe; defaulting steer_probe to the first probe."
        )
    if steer_probe_obj is None:
        alphas = [0.0]
        nonzero_alpha_requested = False

    sys_prompt = system_prompt or probes[0].config["prompts"].get("neutral_system", "You are a helpful assistant.")
    steer_cfg = probes[0].config.get("steering", {})
    max_new_tokens = max_new_tokens if max_new_tokens is not None else steer_cfg.get("steer_max_new_tokens", 128)
    greedy = greedy if greedy is not None else (not steer_cfg.get("steer_sampling", False))
    temperature = temperature if temperature is not None else steer_cfg.get("steer_temperature", 0.7)
    top_p = top_p if top_p is not None else steer_cfg.get("steer_top_p", 0.9)
    save_html = probes[0].config["plots"].get("heatmap_html", True) if save_html is None else save_html
    save_segments = False if save_segments is None else save_segments
    do_steering = bool(steer_cfg.get("do_steering", True))
    save_token_scores_npz = bool(probes[0].config.get("output", {}).get("save_token_scores_npz", True))
    if save_generation_logits is None:
        save_generation_logits = bool(probes[0].config.get("output", {}).get("save_generation_logits", False))
    else:
        save_generation_logits = bool(save_generation_logits)
    if generation_logits_top_k is None:
        generation_logits_top_k = probes[0].config.get("output", {}).get("generation_logits_top_k", None)
    if generation_logits_dtype is None:
        generation_logits_dtype = probes[0].config.get("output", {}).get("generation_logits_dtype", "float16")
    if save_generation_logits and not save_token_scores_npz:
        probes[0]._warn(
            "save_generation_logits requested but output.save_token_scores_npz=false; logits will not be saved."
        )
    save_generation_logits = bool(save_generation_logits and save_token_scores_npz)
    base_random_seed = _resolve_random_control_seed(
        random_vector_seed,
        probes[0].config.get("random", {}).get("seed", 0),
    )
    random_scoring_seed_map: Dict[str, Optional[int]] = {}
    scoring_vectors_by_probe_name: Dict[str, np.ndarray] = {}
    for idx, probe in enumerate(probes):
        probe_name = str(probe.concept.name)
        if random_scoring_vector:
            probe_seed = _mix_random_control_seed(base_random_seed, 3000 + idx)
            random_scoring_seed_map[probe_name] = probe_seed
            scoring_vectors_by_probe_name[probe_name] = _random_vectors_same_norm(
                probe.concept_vectors, seed=probe_seed
            )
        else:
            random_scoring_seed_map[probe_name] = None
            scoring_vectors_by_probe_name[probe_name] = probe.concept_vectors
    steering_seed: Optional[int] = None
    steering_vectors: Optional[np.ndarray] = None
    if steer_distribute is None:
        steer_distribute = steer_cfg.get("steer_distribute", True)

    if nonzero_alpha_requested and not do_steering:
        probes[0]._warn(
            "Non-zero alphas provided; overriding steering.do_steering=false and enabling steering."
        )
        do_steering = True

    if not do_steering:
        steer_probe_obj = None
    if random_steering_vector and steer_probe_obj is None:
        probes[0]._warn("random_steering_vector=true but steering is inactive; no steering is applied.")
    if steer_probe_obj is not None:
        if random_steering_vector:
            steer_idx = probes.index(steer_probe_obj)
            steering_seed = _mix_random_control_seed(base_random_seed, 4000 + steer_idx)
            steering_vectors = _random_vectors_same_norm(steer_probe_obj.concept_vectors, seed=steering_seed)
            probes[0]._warn(
                f"multi_probe_score_prompts using random steering vector control "
                f"(same-norm per layer, seed={steering_seed}, steer_probe={steer_probe_obj.concept.name})."
            )
        else:
            steering_vectors = steer_probe_obj.concept_vectors
    if random_scoring_vector:
        probes[0]._warn(
            f"multi_probe_score_prompts using random scoring vector control "
            f"(same-norm per layer, base_seed={base_random_seed})."
        )

    run_tag = run_name or now_tag()
    run_dir = os.path.join(output_root, safe_slug(project_name), run_tag)
    ensure_dir(run_dir)
    log_path = os.path.join(run_dir, "log.jsonl")
    logger = JsonlLogger(
        log_path,
        enabled=bool(probes[0].config.get("output", {}).get("log_jsonl", True)),
    )

    probes_info = [
        {
            "name": probe.concept.name,
            "pos_label": probe.concept.pos_label,
            "neg_label": probe.concept.neg_label,
            "run_dir": probe.run_dir,
        }
        for probe in probes
    ]

    config = {
        "project_name": project_name,
        "model_id": model_bundle.model_id,
        "system_prompt": sys_prompt,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "greedy": greedy,
            "temperature": temperature,
            "top_p": top_p,
            "save_generation_logits": bool(save_generation_logits),
            "generation_logits_top_k": generation_logits_top_k,
            "generation_logits_dtype": generation_logits_dtype,
        },
        "steering": {
            "do_steering": do_steering,
            "alpha_unit": alpha_unit,
            "alphas": alphas,
            "steer_probe": None if steer_probe_obj is None else steer_probe_obj.concept.name,
            "steer_layers": steer_layers,
            "steer_window_radius": steer_window_radius,
            "steer_distribute": steer_distribute,
        },
        "random_vector_controls": {
            "random_scoring_vector": bool(random_scoring_vector),
            "random_steering_vector": bool(random_steering_vector),
            "random_vector_seed_input": (
                int(random_vector_seed) if random_vector_seed is not None else None
            ),
            "random_vector_seed_base": int(base_random_seed),
            "random_scoring_vector_seed_by_probe": random_scoring_seed_map,
            "random_steering_vector_seed": steering_seed,
        },
        "probes": probes_info,
    }
    json_dump(os.path.join(run_dir, "config.json"), config)
    json_dump(os.path.join(run_dir, "probes.json"), {"probes": probes_info})
    logger.log("config", config)

    base_dir = os.path.join(run_dir, output_subdir)
    ensure_dir(base_dir)
    if batch_subdir is None:
        batch_subdir = f"batch_{now_tag()}"
    out_dir = base_dir if batch_subdir == "" else os.path.join(base_dir, batch_subdir)
    ensure_dir(out_dir)

    tokenizer = model_bundle.tokenizer
    model = model_bundle.model
    results: List[Dict[str, Any]] = []
    batch_entries: List[Tuple[str, List[str], Dict[str, List[float]]]] = []
    console = probes[0].console
    total_prompts = len(prompts)
    total_alphas = len(alphas)
    total_runs = total_prompts * total_alphas
    if console is not None:
        console.info(
            f"Multi-probe scoring {total_prompts} prompts x {total_alphas} alphas ({total_runs} generations) -> {out_dir}"
        )
    show_progress = console is not None and console.enabled
    if show_progress and total_runs:
        _progress_print(0, total_runs, label="Generating", enabled=True)

    if steer_probe_obj is not None:
        steer_layer_list = _steer_layer_list(steer_probe_obj, steer_layers, steer_window_radius)
    else:
        steer_layer_list = []

    warn_fn = probes[0]._warn if probes[0].console is not None else None

    run_idx = 0
    progress_every = 1
    for i, prompt in enumerate(prompts):
        prompt_slug = _prompt_slug_any(prompt)
        for alpha in alphas:
            alpha_val = float(alpha)
            if steer_probe_obj is not None and alpha_unit == "sigma":
                if steer_probe_obj.proj_std_best_layer is None or np.isnan(steer_probe_obj.proj_std_best_layer):
                    raise ValueError("proj_std_best_layer is not available for sigma scaling.")
                alpha_val = alpha_val * float(steer_probe_obj.proj_std_best_layer)

            messages, used_prompt_system = _normalize_messages(
                prompt,
                sys_prompt,
                warn=warn_fn,
                warn_prefix=f"multi_probe.prompts[{i}]",
            )
            input_ids = apply_chat(tokenizer, messages, add_generation_prompt=True).to(model.device)
            attn = attention_mask_from_ids(input_ids).to(model.device)
            prompt_len = int(input_ids.shape[-1])

            if steer_probe_obj is not None:
                with MultiLayerSteererLayerwise(
                    model,
                    steer_layer_list,
                    steering_vectors,
                    float(alpha_val),
                    distribute=steer_distribute,
                ):
                    gen_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=max_new_tokens,
                        do_sample=not greedy,
                        temperature=temperature if not greedy else None,
                        top_p=top_p if not greedy else None,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0].detach().cpu()
                    if save_generation_logits:
                        hs_layers, generation_step_logits = forward_hidden_states_all_layers(
                            model,
                            gen_ids.unsqueeze(0),
                            return_generation_logits=True,
                            generation_prompt_len=prompt_len,
                        )
                    else:
                        hs_layers = forward_hidden_states_all_layers(model, gen_ids.unsqueeze(0))
                        generation_step_logits = None
            else:
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=not greedy,
                    temperature=temperature if not greedy else None,
                    top_p=top_p if not greedy else None,
                    pad_token_id=tokenizer.eos_token_id,
                )[0].detach().cpu()
                if save_generation_logits:
                    hs_layers, generation_step_logits = forward_hidden_states_all_layers(
                        model,
                        gen_ids.unsqueeze(0),
                        return_generation_logits=True,
                        generation_prompt_len=prompt_len,
                    )
                else:
                    hs_layers = forward_hidden_states_all_layers(model, gen_ids.unsqueeze(0))
                    generation_step_logits = None
            token_ids = gen_ids.numpy().astype(np.int32)
            tokens = _decode_tokens(tokenizer, token_ids)
            completion_ids = token_ids[prompt_len:]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            generation_logits_payload: Dict[str, np.ndarray] = {}
            if save_generation_logits and generation_step_logits is not None:
                generation_logits_payload = _pack_generation_logits_arrays(
                    generation_step_logits,
                    logits_top_k=generation_logits_top_k,
                    logits_dtype_name=generation_logits_dtype,
                    logits_source="raw_model",
                )

            scores_agg_list: List[np.ndarray] = []
            for probe in probes:
                layers = probe._layer_selection(layer_selection)
                aggregate = probe.config["evaluation"]["score_aggregate"]
                scoring_vectors = scoring_vectors_by_probe_name[str(probe.concept.name)]
                _, scores_agg = token_scores_from_hs_layers(
                    hs_layers, scoring_vectors, layers, aggregate=aggregate
                )
                scores_agg_list.append(scores_agg.astype(np.float32))

            scores_agg_stack = np.stack(scores_agg_list, axis=0)
            probe_names = [probe.concept.name for probe in probes]
            score_mean_by_probe: Dict[str, float] = {}
            for idx, name in enumerate(probe_names):
                row = scores_agg_stack[idx]
                completion_span = row[prompt_len:] if prompt_len < row.shape[0] else row
                score_mean_by_probe[name] = float(np.mean(completion_span)) if completion_span.size else float("nan")

            alpha_label = _alpha_label(float(alpha), alpha_unit)
            base = f"prompt_{i:03d}_{prompt_slug}_{alpha_label}"
            if save_token_scores_npz:
                npz_path = os.path.join(out_dir, f"{base}.npz")
                npz_payload: Dict[str, Any] = dict(
                    token_ids=token_ids,
                    prompt_len=np.array([prompt_len], dtype=np.int32),
                    scores_agg=scores_agg_stack,
                    probe_names=np.array(probe_names),
                )
                npz_payload.update(generation_logits_payload)
                np.savez_compressed(npz_path, **npz_payload)
            else:
                npz_path = None

            if save_html:
                html_path = os.path.join(out_dir, f"{base}.html")
                scores_by_probe = {
                    name: scores_agg_stack[idx].tolist() for idx, name in enumerate(probe_names)
                }
                title = f"prompt {i:03d} | {prompt_slug} | {alpha_label}"
                render_token_heatmap_multi(tokens, scores_by_probe, html_path, title=title)
                batch_entries.append((title, tokens, scores_by_probe))
            else:
                html_path = None

            segments_path = None
            if save_segments:
                segments_path = os.path.join(out_dir, f"{base}_segments.json")
                segments_by_probe: Dict[str, List[Dict[str, object]]] = {}
                for idx, name in enumerate(probe_names):
                    segments_by_probe[name] = segment_token_scores(
                        tokens,
                        scores_agg_stack[idx].tolist(),
                        prompt_len=prompt_len,
                    )
                json_dump(
                    segments_path,
                    {"prompt_len": prompt_len, "segments_by_probe": segments_by_probe},
                )

            rec = {
                "prompt": prompt,
                "system_prompt": sys_prompt,
                "system_prompt_overridden": bool(used_prompt_system),
                "alpha": float(alpha),
                "alpha_unit": alpha_unit,
                "alpha_value": float(alpha_val),
                "steer_probe": None if steer_probe_obj is None else steer_probe_obj.concept.name,
                "steer_layers": steer_layer_list,
                "random_scoring_vector": bool(random_scoring_vector),
                "random_scoring_vector_seed_by_probe": random_scoring_seed_map,
                "random_steering_vector": bool(random_steering_vector),
                "random_steering_vector_seed": steering_seed,
                "completion": completion_text,
                "prompt_len": prompt_len,
                "probe_names": probe_names,
                "npz_path": npz_path,
                "html_path": html_path,
                "segments_path": segments_path,
                "score_mean_by_probe": score_mean_by_probe,
                "generation_logits_saved": bool(generation_logits_payload),
                "generation_logits_top_k": (
                    int(generation_logits_payload["generation_logits_topk_k"][0])
                    if "generation_logits_topk_k" in generation_logits_payload
                    else None
                ),
                "generation_logits_steps": (
                    int(generation_logits_payload["generation_logits_steps"][0])
                    if "generation_logits_steps" in generation_logits_payload
                    else 0
                ),
                "generation_logits_source": (
                    str(generation_logits_payload["generation_logits_source"][0])
                    if "generation_logits_source" in generation_logits_payload
                    else None
                ),
                "batch_dir": out_dir,
            }
            results.append(rec)
            logger.log("multi_score_prompt", rec)
            run_idx += 1
            if total_runs and (
                run_idx % progress_every == 0 or run_idx == total_runs
            ):
                _progress_print(
                    run_idx,
                    total_runs,
                    label="Generating",
                    enabled=show_progress,
                    last=run_idx == total_runs,
                )
        # Progress bar handles per-run updates; avoid extra log spam here.

    if save_html and batch_entries:
        batch_path = os.path.join(out_dir, "batch_prompts.html")
        render_batch_heatmap_multi(batch_entries, batch_path, title="Batch prompt scores")
        logger.log("multi_score_prompt_batch", {"html_path": batch_path, "count": len(batch_entries)})

    return {"run_dir": run_dir, "results": results, "probes": probes_info}
