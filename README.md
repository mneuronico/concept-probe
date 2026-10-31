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

Outputs are written under `outputs/<concept_name>/<timestamp>/`. A `run_name` that already
holds a run raises instead of mixing two runs; pass `overwrite=True` to replace it.

## What the model sees

By default the library adds nothing to your text that you did not ask for:

- **Texts to read** (`score_texts`, `extract_text_reps`, texts-mode training, eval texts) are
  read raw: the tokenizer's BOS token and the text, with no chat template and no system prompt.
- **Prompts to generate from** (`score_prompts`, `multi_probe_score_prompts`, training
  questions) are a user turn of the chat template, which generation needs. No system prompt is
  added unless you set one: `prompts.neutral_system` (default null), or `system_prompt=` in the
  call. In shared and opposed training the concept's `pos_system`/`neg_system` are the system
  prompts, since they define the concept.
- A message list is used as given. A system prompt you set is added only if the list has none.

To read texts inside the chat template instead, use `role="user"` (the text as a user turn,
after the system prompt if one is set) or `role="assistant"` (the text as the model's reply,
after the system prompt and an optional user turn `prompt`). Settings a role would ignore
raise: a system prompt with `role="raw"`, or a `prompt` with `role="user"`.

The chat template itself can add or change things, and it differs by model:

- Llama 3.x writes a "Cutting Knowledge Date / Today Date" block into every prompt, with or
  without a system prompt. Llama 3.2/3.3 write *today's* date there, so the same input would
  tokenize differently on different days; `model.chat_template_date` (ISO `YYYY-MM-DD`, default
  `"2024-07-26"`, the date those templates fall back to) fixes it, and null writes the real date.
- Qwen 2.5 inserts its own system prompt ("You are Qwen, created by Alibaba Cloud...") when
  there is none.
- Gemma 3 has no system role: it puts the system prompt inside the first user turn.
- Llama and Gemma strip leading and trailing whitespace from each message; the library warns
  once when that happens to a text.

`model.chat_template` (a Jinja string) replaces the tokenizer's template for everything the
bundle does. `score_text_tokens`, the low-level per-token scorer, reads its text raw too (BOS +
text; `add_bos=False` drops the BOS). To see exactly what goes into the model, before running
anything:

```python
from concept_probe import preview_input

preview_input(tokenizer, "The river was calm.")                              # raw (default)
preview_input(tokenizer, "The river was calm.", role="user", system="...")   # chat-formatted
preview_input(tokenizer, "Write about the sea.", generation=True)            # as score_prompts
```

It returns the decoded input with every special and template token, the token ids, and the
span of the text's own tokens. `score_texts` records the same `input_text` and `text_span`, and
`score_prompts` records `prompt_text`.

## Training from labelled texts

If you already have labelled example sentences for each pole, set
`training.train_prompt_mode = "texts"` and provide them on the concept itself via
`train_pos_texts` / `train_neg_texts`. No `pos_system` / `neg_system` is needed. Each text is
read raw (see above) and pooled with `readout.read_mode` (default `assistant_all_mean`: the mean
over the text's own tokens).

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

`training.train_text_role` (`"raw"`, `"user"` or `"assistant"`) sets how the labelled texts are
read, with `training.train_text_system` (null: `prompts.neutral_system`) and, for the assistant
role, `training.train_text_prompt`. The `assistant_*` readouts pool over the text's own tokens
in every role: they leave out the BOS, the system block and role headers before the text, and
the end-of-turn tokens after it. The `sequence_*` readouts pool over the whole input instead;
the template part is identical in every input, so under mean pooling its weight is 1/length and
scores and the direction partly track text length. Training warns when a `sequence_*` readout
is used.

To score other texts under the same readout, use `extract_text_reps` with the same `role`,
`prompt` and `system` (its defaults are the training defaults). It can return several readouts
from one forward pass:

```python
reps = extract_text_reps(model, tokenizer, new_texts, pool=["assistant_all_mean", "assistant_last"])
```

## Evaluation

The evaluation that picks the layer and measures the effect uses the same kind of data as
training, read the same way:

- **Texts mode**: the concept's `eval_pos_texts` / `eval_neg_texts`, read with the training
  texts' role and system prompt and pooled with `readout.read_mode`. `evaluation.eval_system`
  (null by default) overrides the system prompt, with a warning.
- **Shared and opposed modes**: completions generated for `prompts.eval_questions` under
  `pos_system` / `neg_system`, pooled with `readout.train_mode`, like the training completions.
  `eval_questions` (20 by default) should not repeat the training questions; training warns if
  they do.

With fewer than 10 eval items per pole, training warns that the layer choice and `best_d` rest
on very few items. An item with no tokens of its own (a completion that ends right away, or an
empty text) is left out with a warning and listed under `dropped_items` in `metrics.json`; it is
never pooled over the prompt in its place, which would put the system prompt into the vector.

In shared and opposed modes, eval texts given with the concept are a **transfer check**: they are
read with `training.train_text_role` (raw by default), scored at the chosen layer and reported
under `transfer_eval` in `metrics.json` (`d_at_best_layer`, its p and bootstrap interval), plus a
`transfer_sweep.png` plot. They do not take part in choosing the layer. The check matters
because a generated completion has the system prompt that defines the concept in its context: a
vector could partly encode that context, which the generated eval shares and plain texts do not.
`evaluation.eval_mode = "read"` chooses the layer on the eval texts instead, with a warning.

`metrics.json` records each readout (`eval_readout`, `transfer_eval.readout`). Generated
completions (training, evaluation, scoring) are pooled up to and not including their
end-of-sequence token, whichever generation path produced them.

## Layers

Every layer argument (`training.layer_idx`, `best_layer_search`, `layer_selection.layers`,
`steer_layers`, `reference_layer`, patch layers) takes layer indices: negative ones count from
the end, as in Python, and indices outside the model raise. A fraction of depth is written
explicitly, `{"frac": 0.75}`, so that `1` and `1.0` cannot mean different layers:

```python
config_overrides = {"training": {"best_layer_search": {"frac": [0.2, 0.8]}}}  # or [6, 20], or [[0, 3], [10, -1]]
```

## Layer selection and its statistics

`best_layer` is chosen by a sweep over `training.best_layer_search` on the eval items, by
largest Cohen's d (`sweep_select = "effect_size"`) or smallest Welch p among layers with d > 0
(`"p_value"`). Choosing the layer and measuring its effect on the same items overstates the
effect, so by default (`evaluation.selection_split = 0.5`) the layer is chosen on half of the
eval items and `best_d`, `best_p` and `best_d_ci95` (bootstrap, `evaluation.n_bootstrap`) are
measured on the other half. `metrics.json` also reports:

- `selection`: `"split"`, `"all_eval"` (too few items to split, or `selection_split = null`:
  `best_d` is then post-selection and a warning says so) or `"fixed_layer"`.
- `best_d_all_eval`: d at that layer on all eval items (post-selection).
- `sweep_max_d` and `sweep_max_d_perm_p`: the largest d over the searched layers and its
  permutation p-value (`evaluation.n_permutations`), which accounts for having searched them.

Without evaluation, `best_d`/`best_p` are null and the sweep is NaN.

## Scores and steering units

Probe scores are raw projections onto the unit concept direction, with an arbitrary offset.
Each probe stores two references per layer, both from the evaluation items:

- per item (`score_center`, `score_scale`): the midpoint between the poles' mean item score and
  the pooled within-pole SD of item scores, where an item's score is its mean over its tokens;
- per token (`token_center`, `token_scale`): the same over single tokens.

A single token spreads much more than a mean over a text's tokens, which averages most of the
token-to-token variation away, so the two scales differ (by 1.9-3.4x across layers in a 1B
Llama on ~16-token texts). Scoring functions save standardized scores next to the raw ones, each with its own
reference: per-token z (`scores_z_per_layer`, `scores_agg_z`, shown in the heatmaps) and the z of
an item's mean (`score_mean_z`). In both, 0 is the midpoint between the poles, so positive leans
toward the positive pole, and 1 is a within-pole SD.

Steering adds `alpha * v_l` to every token position of each steered layer. `alpha_unit`:

- `"raw"`: alpha as is.
- `"sigma"`: alpha times `sigma_best_layer`, the pooled within-pole SD at the best layer. It is a
  scale knob, not "n standard deviations of behavior".
- `"resid_norm"`: alpha times each layer's mean residual norm on the training tokens, so
  `alpha=0.1` adds 10% of a typical token's norm at each steered layer.

With `steer_distribute = true` (default) alpha is split across the steered layers.
`random_scoring_vector` / `random_steering_vector` use one random direction per probe, derived
from its name and run directory, so the control is the same vector in every function.

## Variance-normalized directions

`training.direction_normalization` chooses how the per-layer direction is estimated from the
pole representations:

- `"none"` (default): the difference of means, `mu_pos - mu_neg`.
- `"var"`: the difference of means divided coordinate-wise by the pooled within-pole variance,
  `0.5 * (var_pos + var_neg)`. This is the diagonal form of the covariance correction of
  mass-mean probing. It keeps a few very high-variance coordinates from dominating the
  direction, as with the massive activations of Gemma 3.
- `"sd"`: the same, divided by the pooled standard deviation.

The direction is always normalized to unit length and stored in `tensors.npz`. Every later use
of the probe (scoring, steering, `ConceptProbe.load`) reads the stored vector, so a probe trained
with `"var"` stays variance-normalized wherever it is used. The layer sweep, and therefore
`best_layer`, is computed with the normalized directions. `concept_direction(pos, neg,
normalization)` exposes the same estimator for representations computed elsewhere.

```python
config_overrides = {"training": {"direction_normalization": "var", "best_layer_search": {"frac": [0.2, 0.8]}}}
```

## Behavioral evals and reports

`run_scored_eval` / `run_multi_scored_eval` score prompts at several alphas and check
correctness. Their analysis never pools correct and incorrect answers across alphas (steering
moves both the scores and the accuracy, and each item recurs at every alpha):

- per alpha: accuracy with 95% Wilson intervals, mean score, and correct vs incorrect scores;
- across alphas: a logistic GEE of correctness on the standardized score with alpha as a
  covariate, clustered by item (`correctness_gee` in `stats.json`).

`generate_multi_probe_report` shows one alpha at a time. Its correctness model has alpha as a
covariate, cross-validation folds that keep all rows of an item together, and standard errors
clustered by item. Items are identified by their prompt, so `aggregate_eval_batches` treats the
same prompt in two batches as one item and different prompts that share an index as two.

With the default evaluator (exact match against `expected`), `marker` / `marker_position` pick
the part of the completion to compare in both `run_scored_eval` and `run_multi_scored_eval`; a
`marker` together with your own `evaluator` raises, since it would not be used.

`multi_probe_score_prompts` runs one model for every probe, so the probes must come from the same
model settings (`model_id`, `dtype`, `use_4bit`, chat template and its date). The generation
settings (system prompt, `max_new_tokens`, sampling, `temperature`, `top_p`) come from the
arguments or, when not passed, from the probes' configs, which must then agree. The steering
layers and their distribution come from the steering probe's config.

Coherence rating sends completions to the external Groq API, so it is off by default
(`rate_coherence=False`). The key comes from `GROQ_API_KEY` in the environment, or from a file
you pass as `coherence_env_path`; no `.env` is read otherwise. Ratings record the judge model and
date: they are labels from another model, not ground truth.

## Loading a trained probe

```python
workspace = ProbeWorkspace(project_directory="outputs/sad_vs_happy/<run>")
probe = workspace.get_probe()
```

This loads the run's own model settings. `get_probe` on a workspace created otherwise raises if
the model (`model_id`, `dtype`, `use_4bit`, chat template or its date) differs from the run's,
since the directions only apply to the activations they were trained on;
`allow_model_mismatch=True` loads it anyway.

## Configuration

`concept_probe/defaults.json` holds every setting. Overrides with a key the defaults do not have
(a misspelling, for instance) raise instead of being ignored. `model.use_4bit` (default false) loads NF4 4-bit weights; it needs bitsandbytes and
fails rather than loading the full-precision model in its place.

## JSON-driven workflow

You can train from a JSON concept spec:

```python
from concept_probe import train_concept_from_json

probe = train_concept_from_json("examples/concepts/bored_vs_interested.json")
```

It trains the probe and scores the eval prompts with the config's `steering` settings. The
example configs in `examples/concepts/` are intentionally general and are not tied to the paper
repo.

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
    concept_direction,
    extract_completion_reps,
    extract_text_reps,
    preview_input,
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

### Unreleased

- **Nothing is added to your text by default.** Texts are read raw (BOS + text);
  `training.train_text_role` defaults to `"raw"`, `prompts.neutral_system` to null, and
  `score_texts` / `extract_text_reps` default to the raw text. Generation prompts are a user turn
  with no system prompt unless one is set. Message lists are no longer given a system message.
  Previously every text and prompt got "You are a helpful assistant. Answer clearly and
  concisely." and the chat template. `preview_input` shows the exact input.
- **Evaluation uses the same kind of data as training.** Shared and opposed modes evaluate on
  completions generated for `eval_questions`, read like the training completions; eval texts
  given with such a concept are reported as a transfer check (`transfer_eval`) and do not pick
  the layer. Previously shared-mode eval texts picked the layer, read as user turns pooled over
  the whole chat sequence, so the sweep measured another readout, partly text length.
- **`assistant_*` readouts pool over the text's own tokens in every role**, and
  `readout.read_mode` defaults to `assistant_all_mean`. Previously `sequence_all_mean` mixed in
  the chat-template prefix with weight 1/length.
- **One end-of-sequence rule**: every generation path keeps the EOS token in its ids and pools
  up to it. Previously batched generation dropped it and the other paths pooled it.
- **Layer selection is measured on held-out eval items** (`evaluation.selection_split`), with a
  bootstrap interval and a permutation p for the largest d. `sweep_select = "p_value"` only
  considers layers with d > 0, and p-values are kept in float64.
- **Standardized scores and new units**: per-item and per-token references per layer, z scores
  for single tokens and for item means, `sigma_best_layer` (pooled within-pole SD, used by
  `alpha_unit="sigma"`; it used to include the between-pole gap) and `alpha_unit="resid_norm"`.
- **Fixed chat-template date**: `model.chat_template_date` (default `"2024-07-26"`). Llama 3.2
  inputs used to contain the date of the run.
- **Errors instead of silent fallbacks**: unknown config and concept keys; `alpha_unit`,
  `steer_layers`, `sweep_select` and layer indices outside their values (previously `"12"`
  steered layers 1 and 2, and out-of-range layer bounds were clipped); settings a text role
  would ignore; `pos_system == neg_system` in contrastive modes; `use_4bit` without bitsandbytes;
  a different model in `get_probe`; a reused `run_name`. Most are checked before anything runs.
- **Fixes**: `multi_probe_score_prompts` scored every probe sharing a concept name with the last
  one's vectors (names must now be distinct); `ActivationPatch` layers outside the model,
  including negative indices, were skipped; `extract_text_reps(role="assistant")` with no system
  or prompt raised an `IndexError`.
- **Statistics**: eval analyses no longer pool correct vs incorrect across alphas (per-alpha
  stats plus a GEE clustered by item), accuracy uses Wilson intervals, and the multi-probe report
  groups its CV folds and standard errors by item, adjusts for alpha and shows one alpha at a time.
- **No silent stand-ins**: an item with no tokens of its own is left out (it used to be pooled
  over the whole prompt, system prompt included); fewer than 10 eval items per pole warns, and
  there are 20 default `eval_questions` instead of 5; eval items are clustered by their prompt
  across batches; multi-probe scoring requires one model and agreeing generation settings
  instead of taking the first probe's; `run_multi_scored_eval` takes `marker`, and a `marker`
  with a custom evaluator raises; `score_text_tokens` adds only the BOS, like every raw read.
- **Other defaults**: `use_4bit` false (NF4 when on); coherence rating off, and no `.env` read
  unless passed; `train_concept_from_json` follows the config's steering settings; the random
  control vector is the same for a probe in every function; `metrics.json` reports null instead
  of 0 when there was no evaluation.

### Earlier

- `training.direction_normalization` (`"none"`, `"var"`, `"sd"`) for variance-normalized
  concept directions; `extract_text_reps`; `model.chat_template`.
- `apply_chat` normalizes `apply_chat_template` output to a `(1, seq_len)` LongTensor, whatever
  the transformers version returns.
- Steered generations are scored with the steering hooks removed. Previously the scoring
  forward pass ran with the hooks still active, so the injected vector "read itself back" and
  inflated probe scores under steering by roughly `alpha * cos(steer_dir, score_dir)` at the
  steered layers. Probe-score-vs-alpha curves from those versions should be regenerated by
  re-scoring the saved completions. Saved generation logits still reflect the steered model and
  are labeled `generation_logits_source = "steered_model"`.
- `score_prompts(..., seed=...)` / `multi_probe_score_prompts(..., seed=...)` seed the RNGs for
  reproducible sampling (best-effort: GPU kernels are not bit-deterministic across
  hardware/library versions). Batched generation can differ from unbatched generation in half
  precision; use `training.gen_batch_size = 1` when that matters.
- Behavioral eval helpers depend on optional packages and, for coherence rating, an external API
  key. Example scripts are intended as starting points and will download model weights when you
  run them.
