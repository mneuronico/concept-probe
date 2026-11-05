from .config import ConceptSpec, load_defaults, resolve_config
from .probe import (
    ConceptProbe,
    NonFiniteActivationsError,
    ProbeWorkspace,
    concept_direction,
    extract_completion_reps,
    extract_text_reps,
    preview_input,
    score_text_tokens,
)
from .multi_probe import multi_probe_score_prompts
from .inputs import Encoded, ReadContext, encode_text, message_spans
from .directions import fit_directions, make_estimator, register_estimator
from .linear_probe import Calibration
from .evaluation import EvalSet, ReadingEvaluation, cross_reading, evaluate_reading, label_shuffle_control
from .stats import accuracy, auroc, compare_scorers, random_directions_null
from .modeling import ModelBundle, QUANTIZATION_LEVELS
from .reading import EmptySpanError, Readout, Reps, read_encoded, read_texts
from .record import Choice
from .settings import configure
from .interventions import Ablate, Add, Compose, Edit, Intervention, Patch, Rotate, Where, intervene
from .generation import AnswerLogprobs, Generation, Generations, TokenStep, answer_logprobs, generate
from .scoring import ScoredItem, Scores
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


def rate_coherence(*args, **kwargs):
    from .coherence import rate_coherence as _rate_coherence

    return _rate_coherence(*args, **kwargs)


__all__ = [
    "ConceptSpec",
    "load_defaults",
    "resolve_config",
    "ConceptProbe",
    "ProbeWorkspace",
    "NonFiniteActivationsError",
    "score_text_tokens",
    "extract_completion_reps",
    "concept_direction",
    "extract_text_reps",
    "preview_input",
    "encode_text",
    "message_spans",
    "Encoded",
    "ReadContext",
    "Calibration",
    "EvalSet",
    "ReadingEvaluation",
    "evaluate_reading",
    "label_shuffle_control",
    "cross_reading",
    "auroc",
    "accuracy",
    "compare_scorers",
    "random_directions_null",
    "fit_directions",
    "make_estimator",
    "register_estimator",
    "ModelBundle",
    "QUANTIZATION_LEVELS",
    "Readout",
    "Reps",
    "EmptySpanError",
    "read_texts",
    "read_encoded",
    "Choice",
    "configure",
    "multi_probe_score_prompts",
    "Intervention",
    "Add",
    "Ablate",
    "Patch",
    "Rotate",
    "Edit",
    "Compose",
    "Where",
    "intervene",
    "generate",
    "Generation",
    "Generations",
    "TokenStep",
    "answer_logprobs",
    "AnswerLogprobs",
    "Scores",
    "ScoredItem",
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
    "rate_coherence",
    "generate_multi_probe_report",
]
