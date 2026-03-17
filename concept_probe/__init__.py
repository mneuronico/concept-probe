from .config import ConceptSpec, load_defaults, resolve_config
from .probe import ConceptProbe, ProbeWorkspace
from .multi_probe import multi_probe_score_prompts
from .console import ConsoleLogger
from .concept_runner import train_concept_from_json
from .reporting import generate_multi_probe_report
from .eval_system import (
    EvalRunResult,
    aggregate_eval_batches,
    rate_batch_coherence_safe,
    rehydrate_batch_analysis,
    run_multi_scored_eval,
    run_scored_eval,
    simple_equality_evaluator,
)


def rate_batch_coherence(*args, **kwargs):
    from .coherence import rate_batch_coherence as _rate_batch_coherence

    return _rate_batch_coherence(*args, **kwargs)


__all__ = [
    "ConceptSpec",
    "load_defaults",
    "resolve_config",
    "ConceptProbe",
    "ProbeWorkspace",
    "multi_probe_score_prompts",
    "ConsoleLogger",
    "train_concept_from_json",
    "EvalRunResult",
    "run_scored_eval",
    "run_multi_scored_eval",
    "simple_equality_evaluator",
    "rehydrate_batch_analysis",
    "aggregate_eval_batches",
    "rate_batch_coherence_safe",
    "rate_batch_coherence",
    "generate_multi_probe_report",
]
