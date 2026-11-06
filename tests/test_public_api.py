from importlib.resources import files

import concept_probe


def test_public_api_surface_is_exposed():
    expected = {
        "ConceptSpec",
        "ConceptProbe",
        "ProbeWorkspace",
        "ConsoleLogger",
        "load_defaults",
        "multi_probe_score_prompts",
        "rate_saved",
        "APIJudge",
        "LocalJudge",
        "Rubric",
        "evaluate_steering",
        "exact_match",
        "SteeringEvaluation",
        "AgentJudge",
        "critique_task",
        "read_target",
        "target_shape",
        "intervene",
        "generate",
        "resolve_config",
        "score_text_tokens",
        "train_concept_from_json",
    }

    assert expected.issubset(set(concept_probe.__all__))


def test_defaults_json_is_packaged():
    defaults_path = files("concept_probe").joinpath("defaults.json")

    assert defaults_path.is_file()


def test_load_defaults_contains_expected_sections():
    defaults = concept_probe.load_defaults()

    for key in ("model", "training", "evaluation", "steering", "output", "prompts"):
        assert key in defaults


def test_resolve_config_applies_nested_overrides():
    resolved = concept_probe.resolve_config(
        {
            "model": {"model_id": "test/model"},
            "output": {"root_dir": "tmp-output"},
        }
    )

    assert resolved["model"]["model_id"] == "test/model"
    assert resolved["output"]["root_dir"] == "tmp-output"


def test_concept_spec_round_trip():
    concept = concept_probe.ConceptSpec.from_config(
        {
            "concept": {
                "name": "calm_vs_urgent",
                "pos_label": "calm",
                "neg_label": "urgent",
                "pos_system": "calm system",
                "neg_system": "urgent system",
                "eval_pos_texts": ["steady"],
                "eval_neg_texts": ["rush"],
            }
        }
    )

    assert concept.to_dict()["name"] == "calm_vs_urgent"
    assert concept.eval_pos_texts == ["steady"]
    assert concept.eval_neg_texts == ["rush"]


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
