import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .logger import JsonlLogger
from .modeling import apply_chat, attention_mask_from_ids, identity_mismatches
from .probe import (
    GENERATED_CONTEXT,
    ConceptProbe,
    PromptLike,
    _pack_generation_logits_arrays,
    _alpha_label,
    _completion_end,
    _eos_token_ids,
    _resolve_steer_layers,
    _validate_alpha_unit,
    _decode_tokens,
    _normalize_messages,
    _prompt_slug_any,
    _resolve_random_control_seed,
    forward_hidden_states_all_layers,
    model_identity_of,
    token_scores_from_hs_layers,
)
from .steering import MultiLayerSteererLayerwise
from .utils import ensure_dir, json_dump, now_tag, safe_slug, set_seed
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


def _shared_setting(probes: List[ConceptProbe], section: str, key: str, explicit: Any, argument: str) -> Any:
    """A setting the whole run uses: ``explicit`` if given, else the probes' common config value.

    Probes whose configs disagree raise, instead of the first probe's value being used silently.
    """
    if explicit is not None:
        return explicit
    values = [probe.config.get(section, {}).get(key) for probe in probes]
    if any(v != values[0] for v in values[1:]):
        named = ", ".join(f"{p.concept.name}={v!r}" for p, v in zip(probes, values))
        raise ValueError(f"The probes disagree on {section}.{key} ({named}); pass {argument}= to choose one.")
    return values[0]


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
    return _resolve_steer_layers(
        steer_layers, int(probe.layer), steer_window_radius, probe.num_layers
    )


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
    seed: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate (optionally steered by one probe) completions and score them with every probe.

    All probes must come from the same model and model settings. Prompts follow
    :meth:`ConceptProbe.score_prompts`: a user turn after the system prompt, if one is set.
    Generation settings (system prompt, max_new_tokens, sampling, temperature, top_p) come
    from the arguments, or else from the probes' configs, which must then agree; the steering
    layers and distribution come from the steering probe's config. ``score_mean_by_probe``
    averages the completion's tokens up to and not including the end-of-sequence token. A
    ``run_name`` whose directory already holds files raises unless ``overwrite=True``.
    """
    if not probes:
        raise ValueError("probes must be a non-empty list of ConceptProbe objects.")
    # Warnings go to the console and to this run's log (buffered until the log exists).
    pending_warnings: List[str] = []
    run_logger: List[JsonlLogger] = []

    def warn(message: str) -> None:
        if probes[0].console is not None:
            probes[0].console.warn(message)
        if run_logger:
            run_logger[0].log("warning", {"message": message})
        else:
            pending_warnings.append(message)

    alpha_unit = _validate_alpha_unit(alpha_unit)
    # Scores, plots, per-sample records and reports are all keyed by concept name.
    slugs = [safe_slug(str(probe.concept.name)) for probe in probes]
    repeated = sorted({str(probe.concept.name) for probe, slug in zip(probes, slugs) if slugs.count(slug) > 1})
    if repeated:
        raise ValueError(
            f"Probes must have distinct concept names, since outputs are keyed by name; repeated: {repeated}. "
            "Rename them first, e.g. probe.concept.name = '<name>_var'."
        )
    if not prompts:
        return {"run_dir": None, "results": []}

    for probe in probes:
        probe._require_trained()

    # Every probe is scored on the activations of probes[0]'s model, so all of them must have been
    # trained on that model, with the same settings that shape the activations.
    model_bundle = probes[0].model_bundle

    def check_same_model(probe: ConceptProbe) -> None:
        mismatched = identity_mismatches(model_identity_of(probe), model_identity_of(probes[0]))
        if probe.model_bundle.model_id != model_bundle.model_id or mismatched:
            raise ValueError(
                f"Probe {probe.concept.name!r} was trained with other model settings than "
                f"{probes[0].concept.name!r} ({'; '.join(mismatched) or 'model_id'}); multi-probe scoring "
                "runs one model for all probes."
            )

    for probe in probes[1:]:
        check_same_model(probe)

    steer_probe_obj = _resolve_steer_probe(probes, steer_probe)
    if steer_probe_obj is not None and all(steer_probe_obj is not p for p in probes):
        steer_probe_obj._require_trained()
        check_same_model(steer_probe_obj)

    if alphas is None:
        alphas = [0.0]
    if len(alphas) == 0:
        raise ValueError("alphas is empty; pass at least one alpha (0.0 for no steering).")
    nonzero_alpha_requested = any(abs(float(a)) > 1e-12 for a in alphas)
    if steer_probe_obj is None and nonzero_alpha_requested:
        raise ValueError("Non-zero alphas need steer_probe: the probe (index, name or object) whose vectors steer.")

    # Settings of the generation itself: an explicit argument, or the value all probes' configs share.
    sys_prompt = _shared_setting(probes, "prompts", "neutral_system", system_prompt, "system_prompt")
    max_new_tokens = _shared_setting(probes, "steering", "steer_max_new_tokens", max_new_tokens, "max_new_tokens")
    greedy = greedy if greedy is not None else not _shared_setting(probes, "steering", "steer_sampling", None, "greedy")
    if not greedy:
        temperature = _shared_setting(probes, "steering", "steer_temperature", temperature, "temperature")
        top_p = _shared_setting(probes, "steering", "steer_top_p", top_p, "top_p")
        # Sampled completions are reproducible: without an explicit seed, random.seed is used.
        seed = _shared_setting(probes, "random", "seed", seed, "seed")
        set_seed(int(seed))
    elif seed is not None:
        set_seed(int(seed))
    generation = {"greedy": bool(greedy), "max_new_tokens": int(max_new_tokens),
                  "temperature": None if greedy else temperature, "top_p": None if greedy else top_p,
                  "seed": None if greedy else int(seed)}
    save_html = probes[0].config["plots"].get("heatmap_html", True) if save_html is None else save_html
    save_segments = False if save_segments is None else save_segments
    do_steering = bool(probes[0].config.get("steering", {}).get("do_steering", True))
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
        warn(
            "save_generation_logits requested but output.save_token_scores_npz=false; logits will not be saved."
        )
    save_generation_logits = bool(save_generation_logits and save_token_scores_npz)
    base_random_seed = _resolve_random_control_seed(
        random_vector_seed,
        probes[0].config.get("random", {}).get("seed", 0),
    )
    # Each probe's random control comes from the probe itself, so it is the same vector that
    # score_texts and score_prompts use for it.
    random_scoring_seed_map: Dict[str, Optional[int]] = {}
    scoring_vectors_by_probe: List[np.ndarray] = []
    for probe in probes:
        probe_name = str(probe.concept.name)
        if random_scoring_vector:
            vectors, probe_seed = probe._random_control_vectors(random_vector_seed)
            random_scoring_seed_map[probe_name] = probe_seed
            scoring_vectors_by_probe.append(vectors)
        else:
            random_scoring_seed_map[probe_name] = None
            scoring_vectors_by_probe.append(probe.directions)
    steering_seed: Optional[int] = None
    steering_vectors: Optional[np.ndarray] = None

    if nonzero_alpha_requested and not do_steering:
        warn(
            "Non-zero alphas provided; overriding steering.do_steering=false and enabling steering."
        )
        do_steering = True

    if not do_steering:
        steer_probe_obj = None
    if random_steering_vector and steer_probe_obj is None:
        warn("random_steering_vector=true but steering is inactive; no steering is applied.")
    if steer_probe_obj is not None:
        if random_steering_vector:
            steering_vectors, steering_seed = steer_probe_obj._random_control_vectors(random_vector_seed)
            warn(
                f"multi_probe_score_prompts using random steering vector control "
                f"(same-norm per layer, seed={steering_seed}, steer_probe={steer_probe_obj.concept.name})."
            )
        else:
            steering_vectors = steer_probe_obj.steer_directions
    if random_scoring_vector:
        warn(
            f"multi_probe_score_prompts using random scoring vector control "
            f"(same-norm per layer, base_seed={base_random_seed})."
        )
    if steer_probe_obj is not None:
        # How to steer comes from the steering probe's config (its layers, window and distribution).
        steer_layer_list = _steer_layer_list(steer_probe_obj, steer_layers, steer_window_radius)
        if steer_distribute is None:
            steer_distribute = steer_probe_obj.config["steering"]["steer_distribute"]
        if nonzero_alpha_requested:
            steer_probe_obj._check_alpha_unit(alpha_unit)
    else:
        steer_layer_list = []

    run_tag = run_name or now_tag()
    run_dir = os.path.join(output_root, safe_slug(project_name), run_tag)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        if not overwrite:
            raise FileExistsError(f"{run_dir} already holds a run; pass another run_name, or overwrite=True to replace it.")
        shutil.rmtree(run_dir)
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
            "sampling_seed": seed,
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
    run_logger.append(logger)
    for message in pending_warnings:
        logger.log("warning", {"message": message})

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

    warn_fn = warn

    eos_ids = _eos_token_ids(model, tokenizer)
    z_missing_by_probe: Dict[str, Optional[str]] = {}
    for probe in probes:
        reason = None
        if not random_scoring_vector and probe.calibration_for(GENERATED_CONTEXT) is None:
            reason = probe._missing_z_reason(GENERATED_CONTEXT)
            warn(f"multi_probe_score_prompts: {probe.concept.name}: score_mean_z is None: {reason}")
        z_missing_by_probe[str(probe.concept.name)] = reason
    run_idx = 0
    progress_every = 1
    for i, prompt in enumerate(prompts):
        prompt_slug = _prompt_slug_any(prompt)
        for j, alpha in enumerate(alphas):
            alpha_val, layer_scales = float(alpha), None
            if steer_probe_obj is not None:
                alpha_val, layer_scales = steer_probe_obj._steering_alpha(alpha, alpha_unit, steer_layer_list)

            messages, used_prompt_system = _normalize_messages(
                prompt,
                sys_prompt,
                warn=warn_fn,
                warn_prefix=f"multi_probe.prompts[{i}]",
            )
            input_ids = apply_chat(tokenizer, messages, add_generation_prompt=True).to(model.device)
            attn = attention_mask_from_ids(input_ids).to(model.device)
            prompt_len = int(input_ids.shape[-1])

            should_steer = steer_probe_obj is not None and abs(float(alpha_val)) > 1e-12
            steer_alpha_per_layer: Dict[int, float] = {}
            if steer_probe_obj is not None:
                steerer = MultiLayerSteererLayerwise(
                    model,
                    steer_layer_list,
                    steering_vectors,
                    float(alpha_val),
                    distribute=steer_distribute,
                    layer_scales=layer_scales,
                )
                steer_alpha_per_layer = steerer.per_layer_alpha() if should_steer else {}
                with steerer:
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
                        # Generation logits must reflect the steered model that produced the text.
                        _, generation_step_logits = forward_hidden_states_all_layers(
                            model,
                            gen_ids.unsqueeze(0),
                            return_generation_logits=True,
                            generation_prompt_len=prompt_len,
                        )
                    else:
                        generation_step_logits = None
                # Score with the hooks removed: with them active, the injected vector reads itself
                # back at the steered layers, contaminating every probe whose direction overlaps it.
                hs_layers = forward_hidden_states_all_layers(model, gen_ids.unsqueeze(0))
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
            completion_end = _completion_end(token_ids, prompt_len, eos_ids)
            completion_text = tokenizer.decode(token_ids[prompt_len:completion_end], skip_special_tokens=True)
            generation_logits_payload: Dict[str, np.ndarray] = {}
            if save_generation_logits and generation_step_logits is not None:
                generation_logits_payload = _pack_generation_logits_arrays(
                    generation_step_logits,
                    logits_top_k=generation_logits_top_k,
                    logits_dtype_name=generation_logits_dtype,
                    logits_source="steered_model" if should_steer else "raw_model",
                )

            scores_agg_list: List[np.ndarray] = []
            per_layer_payload: Dict[str, np.ndarray] = {}
            z_agg_list: List[np.ndarray] = []
            item_z: List[Optional[float]] = []
            for probe, scoring_vectors in zip(probes, scoring_vectors_by_probe):
                layers = probe._layer_selection(layer_selection)
                aggregate = probe.config["evaluation"]["score_aggregate"]
                scores_per_layer, scores_agg = token_scores_from_hs_layers(
                    hs_layers, scoring_vectors, layers, aggregate=aggregate
                )
                scores_agg_list.append(scores_agg.astype(np.float32))
                # each probe scores its own layers, so its per-layer scores are stored under its index
                per_layer_payload[f"scores_per_layer_{len(scores_agg_list) - 1}"] = scores_per_layer.astype(np.float32)
                per_layer_payload[f"layer_indices_{len(scores_agg_list) - 1}"] = np.array(layers, dtype=np.int32)
                standardized = None if random_scoring_vector else probe._token_z(
                    scores_per_layer, layers, aggregate, GENERATED_CONTEXT
                )
                item_z.append(
                    None if random_scoring_vector
                    else probe._item_z(scores_per_layer[prompt_len:completion_end], layers, aggregate, GENERATED_CONTEXT)
                )
                z_agg_list.append(
                    np.full(scores_agg.shape, np.nan, dtype=np.float32) if standardized is None else standardized[1]
                )

            scores_agg_stack = np.stack(scores_agg_list, axis=0)
            scores_agg_z_stack = np.stack(z_agg_list, axis=0)
            probe_names = [probe.concept.name for probe in probes]
            score_mean_by_probe: Dict[str, Optional[float]] = {}
            score_mean_z_by_probe: Dict[str, Optional[float]] = {}
            for idx, name in enumerate(probe_names):
                completion_span = scores_agg_stack[idx][prompt_len:completion_end]
                score_mean_by_probe[name] = float(np.mean(completion_span)) if completion_span.size else None
                score_mean_z_by_probe[name] = item_z[idx]

            alpha_label = _alpha_label(float(alpha), alpha_unit)
            base = f"prompt_{i:03d}_{prompt_slug}_a{j:02d}_{alpha_label}"
            if save_token_scores_npz:
                npz_path = os.path.join(out_dir, f"{base}.npz")
                npz_payload: Dict[str, Any] = dict(
                    token_ids=token_ids,
                    prompt_len=np.array([prompt_len], dtype=np.int32),
                    completion_end=np.array([completion_end], dtype=np.int32),
                    scores_agg=scores_agg_stack,
                    scores_agg_z=scores_agg_z_stack,
                    probe_names=np.array(probe_names),
                    **per_layer_payload,
                )
                npz_payload.update(generation_logits_payload)
                np.savez_compressed(npz_path, **npz_payload)
            else:
                npz_path = None

            if save_html:
                html_path = os.path.join(out_dir, f"{base}.html")
                # Standardized scores (0 = midpoint of the eval poles) where the probe has them, raw otherwise.
                scores_by_probe = {}
                for idx, name in enumerate(probe_names):
                    if np.all(np.isfinite(scores_agg_z_stack[idx])):
                        scores_by_probe[f"{name} (z)"] = scores_agg_z_stack[idx].tolist()
                    else:
                        scores_by_probe[f"{name} (raw)"] = scores_agg_stack[idx].tolist()
                title = f"prompt {i:03d} | {prompt_slug} | {alpha_label}"
                render_token_heatmap_multi(tokens, scores_by_probe, html_path, title=title)
                batch_entries.append((title, tokens, scores_by_probe))
            else:
                html_path = None

            segments_path = None
            if save_segments:
                segments_path = os.path.join(out_dir, f"{base}_segments.json")
                # Like the heatmap: token z where the probe has references, raw projections otherwise.
                segments_by_probe: Dict[str, List[Dict[str, object]]] = {}
                score_kind_by_probe: Dict[str, str] = {}
                for idx, name in enumerate(probe_names):
                    has_z = bool(np.all(np.isfinite(scores_agg_z_stack[idx])))
                    score_kind_by_probe[name] = "z" if has_z else "raw"
                    segments_by_probe[name] = segment_token_scores(
                        tokens,
                        (scores_agg_z_stack[idx] if has_z else scores_agg_stack[idx]).tolist(),
                        prompt_len=prompt_len,
                    )
                json_dump(
                    segments_path,
                    {"prompt_len": prompt_len, "score_kind_by_probe": score_kind_by_probe,
                     "segments_by_probe": segments_by_probe},
                )

            rec = {
                "prompt": prompt,
                "system_prompt": sys_prompt,
                "system_prompt_overridden": bool(used_prompt_system),
                "prompt_text": tokenizer.decode(token_ids[:prompt_len].tolist(), skip_special_tokens=False),
                "alpha": float(alpha),
                "alpha_unit": alpha_unit,
                "alpha_value": float(alpha_val),
                "steer_layer_scales": layer_scales,
                "steer_alpha_per_layer": steer_alpha_per_layer,
                "steer_vectors": "each steered layer's own probe vector",
                "generation": generation,
                "steer_probe": None if steer_probe_obj is None else steer_probe_obj.concept.name,
                "steer_layers": steer_layer_list,
                "random_scoring_vector": bool(random_scoring_vector),
                "random_scoring_vector_seed_by_probe": random_scoring_seed_map,
                "random_steering_vector": bool(random_steering_vector),
                "random_steering_vector_seed": steering_seed,
                "completion": completion_text,
                "prompt_len": prompt_len,
                "completion_end": completion_end,
                "probe_names": probe_names,
                "npz_path": npz_path,
                "html_path": html_path,
                "segments_path": segments_path,
                "score_mean_by_probe": score_mean_by_probe,
                "score_mean_z_by_probe": score_mean_z_by_probe,
                "z_missing_by_probe": z_missing_by_probe,
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
