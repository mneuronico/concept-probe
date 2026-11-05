"""Several probes at once: generate (optionally steered by one of them) and score with all of them.

It is :meth:`ConceptProbe.score_prompts` with a list of probes: the same generation in batches,
the same scoring and the same files, plus a run folder of its own with the probes' names.
"""
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Union

from .generation import TokenStep
from .logger import JsonlLogger
from .modeling import identity_mismatches
from .probe import ConceptProbe, PromptLike, _score_generations, model_identity_of
from .scoring import Scores, batch_dir
from .settings import settings
from .utils import ensure_dir, json_dump, now_tag, safe_slug


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
    keep_logits: Union[None, int, str] = None,
    batch_subdir: Optional[str] = None,
    n_random: int = 0,
    random_seed: Optional[int] = None,
    seed: Optional[int] = None,
    overwrite: bool = False,
    batch_size: Optional[int] = None,
    on_token: Optional[Callable[[TokenStep], Any]] = None,
    save: Optional[bool] = None,
) -> Scores:
    """Generate (optionally steered by ``steer_probe``) completions and score them with every probe.

    All probes must come from the same model and model settings, and have distinct names (scores
    are keyed by name). Prompts, alphas, generation and scoring follow
    :meth:`ConceptProbe.score_prompts`, including ``n_random`` random controls of the steering
    probe's steering. Settings of the generation (system prompt,
    max_new_tokens, sampling, temperature, top_p, batch size) come from the arguments, or else from
    the probes' configs, which must then agree; the steering layers and their distribution come
    from the steering probe's config.

    Returns :class:`concept_probe.scoring.Scores` with one entry per probe in each item. Saved by
    default to ``<output_root>/<project_name>/<run_name>``: ``config.json``, ``probes.json`` and
    ``log.jsonl``, and the scores in ``<output_subdir>/<batch_subdir>``. A ``run_name`` whose folder
    already holds files raises unless ``overwrite=True``.
    """
    if not probes:
        raise ValueError("probes must be a non-empty list of ConceptProbe objects.")
    slugs = [safe_slug(str(probe.concept.name)) for probe in probes]
    repeated = sorted({str(probe.concept.name) for probe, slug in zip(probes, slugs) if slugs.count(slug) > 1})
    if repeated:
        raise ValueError(
            f"Probes must have distinct concept names, since outputs are keyed by name; repeated: {repeated}. "
            "Rename them first, e.g. probe.concept.name = '<name>_var'."
        )
    for probe in probes:
        probe._require_model("multi_probe_score_prompts")
    first = probes[0]

    def check_same_model(probe: ConceptProbe) -> None:
        mismatched = identity_mismatches(model_identity_of(probe), model_identity_of(first))
        if probe.model_bundle.model_id != first.model_bundle.model_id or mismatched:
            raise ValueError(
                f"Probe {probe.concept.name!r} was trained with other model settings than "
                f"{first.concept.name!r} ({'; '.join(mismatched) or 'model_id'}); multi-probe scoring "
                "runs one model for all probes."
            )

    for probe in probes[1:]:
        check_same_model(probe)
    steer = _resolve_steer_probe(probes, steer_probe)
    if steer is not None and all(steer is not p for p in probes):
        steer._require_model("multi_probe_score_prompts")
        check_same_model(steer)

    saving = settings().save if save is None else bool(save)
    run_dir, logger = None, None
    if saving:
        run_dir = os.path.join(output_root, safe_slug(project_name), run_name or now_tag())
        if os.path.isdir(run_dir) and os.listdir(run_dir):
            if not overwrite:
                raise FileExistsError(f"{run_dir} already holds a run; pass another run_name, or overwrite=True to "
                                      "replace it.")
            shutil.rmtree(run_dir)
        ensure_dir(run_dir)
        logger = JsonlLogger(os.path.join(run_dir, "log.jsonl"),
                             enabled=bool(first.config.get("output", {}).get("log_jsonl", True)))

    def warn(message: str) -> None:
        if first.console is not None:
            first.console.warn(message)
        if logger is not None:
            logger.log("warning", {"message": message})

    def status(message: str) -> None:
        if first.console is not None:
            first.console.info(message)

    if not prompts:
        return Scores("generations", [str(p.concept.name) for p in probes], [], {"run_dir": run_dir})
    result = _score_generations(
        probes, prompts, steer=steer, alphas=alphas, alpha_unit=alpha_unit, steer_layers=steer_layers,
        steer_window_radius=steer_window_radius, steer_distribute=steer_distribute, system_prompt=system_prompt,
        max_new_tokens=max_new_tokens, greedy=greedy, temperature=temperature, top_p=top_p, seed=seed,
        batch_size=batch_size, layer_selection=layer_selection, n_random=n_random, random_seed=random_seed,
        keep_logits=keep_logits, on_token=on_token, warn=warn, status=status,
    )
    probes_info = [{"name": p.concept.name, "pos_label": p.concept.pos_label, "neg_label": p.concept.neg_label,
                    "run_dir": p.run_dir} for p in probes]
    result.settings.update(run_dir=run_dir, model_id=first.model_bundle.model_id, project_name=project_name,
                           probes=probes_info)
    if saving:
        json_dump(os.path.join(run_dir, "config.json"), result.settings)
        json_dump(os.path.join(run_dir, "probes.json"), {"probes": probes_info})
        logger.log("config", result.settings)
        save_html = first.config["plots"].get("heatmap_html", True) if save_html is None else save_html
        out = batch_dir(os.path.join(run_dir, output_subdir), batch_subdir, overwrite)
        result.save(out, html=save_html, segments=bool(save_segments))
        logger.log("multi_score_prompts", {"n_generations": len(result), "dir": out})
        status(f"Saved the scores of {len(result)} generations to {out}")
    return result
