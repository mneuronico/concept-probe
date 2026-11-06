"""concept-probe: train, evaluate and steer linear probes of concepts in open-weights language models.

The library is in layers, each usable alone: load a model (:class:`ModelBundle`), read texts into
representations (:func:`read_texts`), fit directions and a probe on them (:func:`fit_directions`,
:class:`ConceptProbe`), evaluate how it reads against controls (:func:`evaluate_reading`), intervene
in the model (:class:`Add`, :class:`Ablate`, :class:`Patch`, :class:`Rotate`), generate and score
(:func:`generate`, :func:`answer_logprobs`), and evaluate steering against random controls
(:func:`evaluate_steering`). Beyond two poles, :class:`Target` and :class:`TargetProbe`. The guides
(:func:`guide`) say in what order, and what makes a probe look better than it is;
:func:`load_example` gives texts to try it with.
"""
__version__ = "0.3.0"

from .settings import configure
from .record import Choice
from .modeling import ModelBundle
from .inputs import Encoded, ReadContext, encode_text, preview_input
from .reading import EmptySpanError, NonFiniteActivationsError, Readout, Reps, read_encoded, read_texts
from .directions import Eraser, fit_directions, register_estimator
from .probe import ConceptProbe, summarize_tokens
from .evaluation import (EvalSet, ReadingEvaluation, confound_check, cross_reading, evaluate_reading,
                         label_shuffle_control, learning_curve, red_team)
from .stats import accuracy, auroc, cohen_d, compare_scorers, random_directions_null
from .structure import Target, TargetProbe, TargetReading, TargetScores, TargetShape, read_target, target_shape
from .datasets import (DatasetAudit, audit_dataset, compare_poles, count_matches, distinctive_words, diversity,
                       formulaicity, surface_baseline, text_features)
from .interventions import Ablate, Add, Compose, Edit, Intervention, Patch, Rotate, Where, intervene
from .generation import AnswerLogprobs, Generation, Generations, TokenStep, answer_logprobs, generate
from .scoring import ScoredItem, Scores
from .steering_eval import CoherenceRule, SteeringEvaluation, evaluate_steering, exact_match, repetition_rate
from .judges import COHERENCE, AgentJudge, AnswersPending, APIJudge, Judge, LocalJudge, Rubric, rate_saved
from .critique import Critique, critique_dataset, critique_task, load_critique
from .guide import guide
from .example_data import ExampleDataset, load_example

__all__ = [
    # session and loading
    "configure", "Choice", "ModelBundle",
    # reading
    "read_texts", "read_encoded", "Readout", "Reps", "ReadContext", "Encoded", "encode_text", "preview_input",
    "EmptySpanError", "NonFiniteActivationsError",
    # directions and the probe
    "fit_directions", "register_estimator", "Eraser", "ConceptProbe", "summarize_tokens",
    # evaluating how it reads
    "EvalSet", "ReadingEvaluation", "evaluate_reading", "label_shuffle_control", "cross_reading", "learning_curve",
    "confound_check", "red_team", "cohen_d", "auroc", "accuracy", "compare_scorers", "random_directions_null",
    # beyond two poles
    "Target", "TargetProbe", "TargetReading", "TargetScores", "TargetShape", "read_target", "target_shape",
    # datasets
    "audit_dataset", "DatasetAudit", "compare_poles", "surface_baseline", "text_features", "count_matches",
    "distinctive_words", "formulaicity", "diversity", "Critique", "critique_task", "load_critique",
    "critique_dataset",
    # interventions, generation and scoring
    "Intervention", "Add", "Ablate", "Patch", "Rotate", "Edit", "Compose", "Where", "intervene",
    "generate", "Generation", "Generations", "TokenStep", "answer_logprobs", "AnswerLogprobs", "Scores", "ScoredItem",
    # evaluating steering, and judges
    "evaluate_steering", "SteeringEvaluation", "CoherenceRule", "exact_match", "repetition_rate",
    "Rubric", "COHERENCE", "Judge", "APIJudge", "LocalJudge", "AgentJudge", "AnswersPending", "rate_saved",
    # guides and an example dataset
    "guide", "load_example", "ExampleDataset",
]
