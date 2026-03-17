# Contributing

Thanks for taking an interest in `concept-probe`.

## Local setup

Create a virtual environment and install the project with development dependencies:

```bash
pip install -e .[dev,stats,plots]
```

If you want to use coherence helpers, also install:

```bash
pip install -e .[coherence]
```

## Running checks

Run the lightweight test suite:

```bash
python -m pytest -q
```

You can also sanity-check the example scripts for syntax:

```bash
python -m py_compile examples/quickstart.py examples/train_from_json.py
```

## Pull requests

Before opening a PR:

- Keep changes narrowly scoped.
- Add or update tests when behavior changes.
- Update the README if the public API or installation story changes.
- Preserve the current package structure unless there is a strong reason to alter it.
