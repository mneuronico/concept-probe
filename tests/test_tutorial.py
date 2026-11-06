"""The tutorial notebook: valid, and every code cell compiles (it runs end to end on a GPU, not here)."""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "tutorial.ipynb"


def test_every_code_cell_of_the_tutorial_compiles():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert nb["nbformat"] == 4 and len(code) > 30
    for i, cell in enumerate(code):
        compile("".join(cell["source"]), f"tutorial cell {i}", "exec")


def test_the_tutorial_uses_only_example_datasets_that_exist():
    import re

    from concept_probe.example_data import EXAMPLES

    text = NOTEBOOK.read_text(encoding="utf-8")
    assert set(re.findall(r'load_example\(\\"(\w+)\\"\)', text)) <= set(EXAMPLES)
