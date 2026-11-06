"""The example datasets that ship with the package."""
import pytest

from concept_probe import ExampleDataset, load_example
from concept_probe.example_data import EXAMPLES


def test_the_valence_example_is_paired_and_split():
    data = load_example("valence")
    assert isinstance(data, ExampleDataset) and data.pos_label == "positive"
    assert len(data.train_pos) == len(data.train_neg) == 141 and len(data.eval_pos) == len(data.eval_neg) == 83
    train, held_out = set(data.column("id", split="train")), set(data.column("id", split="eval"))
    assert not train & held_out and len(train | held_out) == len(data.records)
    assert all(r["pos"] != r["neg"] for r in data.records)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_example_loads_with_its_description(name):
    data = load_example(name)
    assert data.records and data.description and data.name == name


def test_texts_and_columns_select_records():
    topics = load_example("topics")
    assert len(topics.column("text", set="food")) == 40 and len(topics.texts) == 240
    with pytest.raises(KeyError, match="no 'pos'"):
        topics.pos


def test_a_concept_written_as_system_prompts_has_them_in_extra():
    data = load_example("bored_vs_interested")
    assert data.pos_label == "interested" and set(data.extra) == {"pos_system", "neg_system", "train_questions",
                                                                   "eval_questions"}
    assert len(data.eval_pos) == len(data.eval_neg) == 20


def test_an_unknown_example_names_the_ones_there_are():
    with pytest.raises(ValueError, match="valence"):
        load_example("sentiment")
