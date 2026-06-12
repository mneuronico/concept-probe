# concept-probe

`concept-probe` is a small, config-driven Python library for training, evaluating, scoring, and steering concept probes in open-weights language models.

## What it includes

- Contrastive probe training from positive vs negative system prompts
- Probe training from labelled example texts (no system prompts required)
- Layer sweeps with effect sizes and optional p-values
- Token-level and sequence-level scoring
- Generation-time steering with learned concept vectors
- Multi-probe scoring on the same model
- Saved artifacts for configs, metrics, tensors, logs, plots, and HTML heatmaps
- Behavioral eval helpers and multi-probe reporting utilities
- Optional generation-logit capture

## What it does not include

- Paper datasets
- Paper figures
- Paper analysis outputs
- Repo-level experiment history

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/mneuronico/concept-probe.git
```

Local editable install:

```bash
pip install -e .
```

Minimal runtime dependencies:

```bash
pip install numpy torch transformers
```

Optional extras:

```bash
pip install -e .[plots,stats]
pip install -e .[coherence]
pip install -e .[full]
pip install -e .[dev,stats,plots]
```

Notes:

- The library expects a local PyTorch environment and a Hugging Face-compatible causal LM.
- `scipy` is used for sweep p-values.
- `matplotlib` is used for plots.
- `statsmodels` and `scikit-learn` are used by reporting utilities.
- `groq` is only needed for coherence rating helpers.
- `bitsandbytes` is only needed for 4-bit loading.

## Quickstart

```python
from concept_probe import ConceptSpec, ProbeWorkspace

SAD_SYSTEM = "You are a helpful assistant with a subdued, melancholic tone."
HAPPY_SYSTEM = "You are a helpful assistant with a warm, optimistic tone."

concept = ConceptSpec(
    name="sad_vs_happy",
    pos_label="sad",
    neg_label="happy",
    pos_system=SAD_SYSTEM,
    neg_system=HAPPY_SYSTEM,
    eval_pos_texts=[
        "I feel heavy and distant today.",
        "It is hard to find hope right now.",
    ],
    eval_neg_texts=[
        "I feel light and excited about today.",
        "Everything feels uplifting and positive.",
    ],
)

workspace = ProbeWorkspace(model_id="meta-llama/Llama-3.2-3B-Instruct")
probe = workspace.train_concept(concept)

probe.score_prompts(
    prompts=["Write a short paragraph about the ocean."],
    alphas=[0.0, 6.0, -6.0],
    alpha_unit="sigma",
)
```

Outputs are written under `outputs/<concept_name>/<timestamp>/`.

## Training from labelled texts

If you already have labelled example sentences for each pole, set
`training.train_prompt_mode = "texts"` and provide them on the concept
itself via `train_pos_texts` / `train_neg_texts`. No `pos_system` /
`neg_system` is needed; pooling falls back to `readout.read_mode`
(default `sequence_all_mean`), the same path the read-side eval uses.

```python
from concept_probe import ConceptSpec, ProbeWorkspace

concept = ConceptSpec(
    name="raya_animal_vs_geom",
    pos_label="animal",
    neg_label="geometry",
    pos_system="",
    neg_system="",
    train_pos_texts=[
        "La raya nadaba sobre el lecho arenoso.",
        # ... 20+ examples
    ],
    train_neg_texts=[
        "Trazó una raya recta con la regla.",
        # ... 20+ examples
    ],
    eval_pos_texts=[...],   # held-out animal examples
    eval_neg_texts=[...],   # held-out geometry examples
)

workspace = ProbeWorkspace(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    config_overrides={"training": {"train_prompt_mode": "texts"}},
)
probe = workspace.train_concept(concept)
```

## JSON-driven workflow

You can train from a JSON concept spec:

```python
from concept_probe import train_concept_from_json

probe = train_concept_from_json("examples/concepts/bored_vs_interested.json")
```

The example configs in `examples/concepts/` are intentionally general and are not tied to the paper repo.

## Repo layout

```text
concept-probe/
  concept_probe/          # library package
  examples/               # general examples only
  pyproject.toml
  README.md
  LICENSE
```

## Public API

Main imports:

```python
from concept_probe import (
    ConceptSpec,
    ProbeWorkspace,
    ConceptProbe,
    multi_probe_score_prompts,
    train_concept_from_json,
    generate_multi_probe_report,
    run_scored_eval,
    run_multi_scored_eval,
)
```

## Development

Run the lightweight test suite:

```bash
python -m pytest -q
```

Compile the example entrypoints:

```bash
python -m py_compile examples/quickstart.py examples/train_from_json.py
```

GitHub Actions CI runs the same lightweight checks and also builds source and wheel distributions.

## Release notes

- Steered generations are now scored with the steering hooks removed. Previously the
  scoring forward pass ran with the hooks still active, so the injected vector "read
  itself back" and inflated probe scores under steering by roughly
  `alpha * cos(steer_dir, score_dir)` at the steered layers — a large effect when
  steering and scoring with the same probe (and ~zero for random-vector controls,
  which therefore could not detect it). Probe-score-vs-alpha curves from earlier
  versions should be regenerated by re-scoring the saved completions. Saved
  generation logits still reflect the steered model and are labeled
  `generation_logits_source = "steered_model"`.
- `best_d`/`best_p` in `metrics.json` are post-selection values (the best layer is
  chosen by optimizing over the layer sweep on the same eval data); confirm effect
  sizes on held-out data before reporting them.
- The default config in `concept_probe/defaults.json` is usable, but opinionated.
- Behavioral eval helpers depend on optional packages and, for coherence rating, an external API key.
- Example scripts are intended as starting points and will download model weights when you run them.
