"""The example dataset that ships with the package."""
import pytest

from concept_probe import ExampleDataset, load_example


def test_the_valence_example_is_paired_and_split():
    data = load_example("valence")
    assert isinstance(data, ExampleDataset) and data.pos_label == "positive"
    assert len(data.train_pos) == len(data.train_neg) == 141 and len(data.eval_pos) == len(data.eval_neg) == 83
    train = {p["id"] for p in data.pairs if p["split"] == "train"}
    held_out = {p["id"] for p in data.pairs if p["split"] == "eval"}
    assert not train & held_out and len(train | held_out) == len(data.pairs)
    assert all(p["pos"] != p["neg"] for p in data.pairs)


def test_an_unknown_example_names_the_ones_there_are():
    with pytest.raises(ValueError, match="valence"):
        load_example("sentiment")
