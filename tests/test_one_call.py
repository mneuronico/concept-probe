"""The one-call constructors: the same probe as reading, the eval sets and fit step by step."""
import numpy as np
import pytest

from concept_probe import ConceptProbe, EvalSet, ModelBundle, read_texts
from test_reference_outputs import (EVAL_NEG, EVAL_POS, EVAL_QUESTIONS, MODEL, NEG_SYSTEM, POS_SYSTEM, QUESTIONS, TOK,
                                    TRAIN_NEG, TRAIN_POS, _train_on_generations)

FIT = dict(n_permutations=200, n_bootstrap=200, seed=0, save=False)


def _bundle():
    return ModelBundle.from_loaded(MODEL, TOK, model_id="tiny-test", console=False)


def test_from_texts_is_reading_an_eval_set_and_fit_in_a_row():
    bundle = _bundle()
    one = ConceptProbe.from_texts(bundle, TRAIN_POS, TRAIN_NEG, EVAL_POS, EVAL_NEG, paired=True, name="c", **FIT)
    read = lambda texts: read_texts(MODEL, TOK, texts)  # noqa: E731
    steps = ConceptProbe.fit(read(TRAIN_POS), read(TRAIN_NEG),
                             eval_sets=EvalSet("eval", read(EVAL_POS), read(EVAL_NEG), paired=True), model=bundle,
                             name="c", **FIT)
    assert one.layer == steps.layer and one.metrics["d"] == pytest.approx(steps.metrics["d"])
    np.testing.assert_allclose(one.directions, steps.directions)
    assert one.evaluation.sets["eval"].paired


def test_from_system_prompts_trains_on_answers_and_reads_the_eval_answers_without_the_system_prompt():
    probe = ConceptProbe.from_system_prompts(_bundle(), POS_SYSTEM, NEG_SYSTEM, QUESTIONS, EVAL_QUESTIONS,
                                             eval_texts=(EVAL_POS, EVAL_NEG), max_new_tokens=10, batch_size=4,
                                             name="shared", **FIT)
    reference, _, _ = _train_on_generations("shared", QUESTIONS, QUESTIONS)
    assert set(probe.evaluation.sets) == {"eval", "eval_with_system_prompt", "texts"}
    assert probe.layer == reference.layer and probe.metrics["d"] == pytest.approx(reference.metrics["d"])
    np.testing.assert_allclose(probe.directions, reference.directions, atol=1e-6)
    sets = probe.evaluation.sets
    assert sets["eval"].d < sets["eval_with_system_prompt"].d  # the system prompt inflates it


def test_from_system_prompts_saves_the_answers_with_the_probe(tmp_path):
    probe = ConceptProbe.from_system_prompts(_bundle(), POS_SYSTEM, NEG_SYSTEM, QUESTIONS[:3], EVAL_QUESTIONS[:4],
                                             max_new_tokens=4, batch_size=4, name="s", output_root=str(tmp_path),
                                             **{**FIT, "save": True, "plots": False})
    folder = tmp_path / "s"
    run = next(folder.iterdir())
    assert sorted(p.name for p in (run / "generations").iterdir()) == ["eval_neg", "eval_pos", "train_neg", "train_pos"]
    assert probe.out_dir == str(run)
