import json
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "concepts"


def test_example_concepts_exist():
    example_paths = sorted(EXAMPLES_DIR.glob("*.json"))

    assert example_paths, "expected example concept JSON files"


def test_example_concepts_have_required_sections():
    example_paths = sorted(EXAMPLES_DIR.glob("*.json"))

    for path in example_paths:
        data = json.loads(path.read_text(encoding="utf-8"))

        assert "concept" in data, f"missing concept section in {path.name}"
        assert "prompts" in data, f"missing prompts section in {path.name}"

        concept = data["concept"]
        prompts = data["prompts"]

        for key in ("name", "pos_label", "neg_label", "pos_system", "neg_system"):
            assert concept.get(key), f"missing concept.{key} in {path.name}"

        assert concept.get("eval_pos_texts"), f"missing eval_pos_texts in {path.name}"
        assert concept.get("eval_neg_texts"), f"missing eval_neg_texts in {path.name}"
        assert prompts.get("train_questions"), f"missing train_questions in {path.name}"
        assert prompts.get("eval_questions"), f"missing eval_questions in {path.name}"
        assert prompts.get("neutral_system"), f"missing neutral_system in {path.name}"
