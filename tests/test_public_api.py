from importlib.resources import files

import concept_probe


def test_public_api_surface_is_exposed():
    expected = {
        "ModelBundle", "read_texts", "read_encoded", "Readout", "Reps", "fit_directions", "ConceptProbe", "EvalSet",
        "evaluate_reading", "summarize_tokens", "Add", "Ablate", "Patch", "Rotate", "intervene", "generate",
        "answer_logprobs", "evaluate_steering", "exact_match", "SteeringEvaluation", "Rubric", "APIJudge",
        "LocalJudge", "AgentJudge", "rate_saved", "critique_task", "Target", "TargetProbe", "read_target",
        "target_shape", "audit_dataset", "guide", "configure",
    }
    assert expected.issubset(set(concept_probe.__all__))
    assert all(hasattr(concept_probe, name) for name in concept_probe.__all__)


def test_what_was_replaced_is_gone():
    # one way to do each thing: the configuration-driven pipeline and the duplicates are not there
    gone = {"ConceptSpec", "ProbeWorkspace", "load_defaults", "resolve_config", "train_concept_from_json",
            "multi_probe_score_prompts", "score_text_tokens", "extract_text_reps", "extract_completion_reps",
            "concept_direction", "ConsoleLogger"}
    assert not gone & set(concept_probe.__all__)
    assert not any(hasattr(concept_probe, name) for name in gone)
    assert not files("concept_probe").joinpath("defaults.json").is_file()


def test_the_guides_ship_with_the_package():
    listing = concept_probe.guide()
    for name in ("workflow", "datasets", "steering", "delegating"):
        assert f"- {name}:" in listing and concept_probe.guide(name).startswith("# ")
    try:
        concept_probe.guide("nope")
    except ValueError as exc:
        assert "workflow" in str(exc)
    else:
        raise AssertionError("an unknown guide should raise")
