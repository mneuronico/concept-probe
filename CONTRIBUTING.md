# Contributing

Thanks for taking an interest in `concept-probe`.

## Local setup

Create a virtual environment and install the project with development dependencies:

```bash
pip install -e .[dev,stats,plots]
```

## Running checks

Run the test suite (it runs on a tiny deterministic model, on the CPU, with no downloads):

```bash
python -m pytest -q
```

You can also sanity-check the example scripts for syntax:

```bash
python -m py_compile examples/quickstart.py examples/system_prompts.py
```

## Pull requests

Before opening a PR:

- Keep changes narrowly scoped.
- Add or update tests when behavior changes.
- Update the README, `CHANGELOG.md` and the guides (`concept_probe/guides/`) if the public API or
  installation story changes.
- Keep the layers apart: each piece does one thing, announces the choices it
  makes for you, and works on arrays without a model where it can.
