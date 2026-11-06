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
- Evaluation against controls: random directions, shuffled labels, random steering directions
- Interventions (add, erase, patch, rotate), batched generation, answer probabilities
- Steering evaluation with coherence, judges (any provider, a local model or an agent) and behavioral measures
- Dataset audits and critiques, confound checks and erasure, probes toward any target and the shape of representations
- Guides for people and agents (`concept_probe.guide()`, `AGENTS.md`)

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
- `statsmodels` and `scikit-learn` are used by reporting utilities, the logistic and LDA
  estimators and the dataset audit's classifiers.
- Judges behind an API (OpenAI, Anthropic, Gemini, OpenRouter, Groq, local servers) need no client
  package: the library calls them with Python's standard library.
- `bitsandbytes` is only needed for the `nf4` and `int8` quantization levels, and `hqq` for the
  `hqq*` levels.

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

scores = probe.score_prompts(
    prompts=["Write a short paragraph about the ocean."],
    alphas=[0.0, 6.0, -6.0],
    alpha_unit="sigma",
)
scores.z()                       # each completion's z: the prompt at each alpha, in order
[item.text for item in scores]   # the completions
```

Outputs are written under `outputs/<concept_name>/<timestamp>/`, and each call's scores in a
folder of their own under `scores/` (see [Generating and scoring](#generating-and-scoring)). A
`run_name` that already holds a run raises instead of mixing two runs; pass `overwrite=True` to
replace it.

## Loading the model

`ProbeWorkspace` loads the model of the config's `model` section:

- `model.dtype` is `"auto"` by default: bfloat16 on GPUs that compute in it (compute capability
  8.0 and up), float16 on older GPUs (the T4 and the RTX 20xx only emulate bfloat16, slowly),
  except float32 for Gemma 3, whose activations overflow in float16; float32 without a CUDA GPU.
  An explicit `"bfloat16"`, `"float16"` or `"float32"` is used as given.
- `model.quantization` is null (default), a level (`"nf4"`, `"int8"`, `"hqq4"`, `"hqq3"`,
  `"hqq2"`), a transformers quantization config, or a dict with its `quant_method` (for
  bitsandbytes with double quantization, for instance). NF4 computes in `model.dtype`. A level
  that cannot load fails instead of loading the full-precision model.
- With `HF_HUB_OFFLINE` set, a repo id loads from its snapshot in the local cache.

A model that is already loaded can be passed in instead, for instance one quantized another
way, or one model shared by several workspaces: `ProbeWorkspace(model=model, tokenizer=tokenizer)`,
or `model=` a `ModelBundle`. Its own name, dtype and quantization replace the config's.

What makes the activations what they are (the model and its revision, the dtype actually used,
the quantization, the chat template and its date) is the model's **identity**. A trained probe
records it in `metrics.json` (`model_identity`), and loading a probe with a model of another
identity raises (see [Loading a trained probe](#loading-a-trained-probe)).

**Every choice the library makes on its own is announced.** The dtype chosen for `"auto"`, the
cached snapshot loaded offline, where read representations are cached: each is printed with
its reason (when `output.console` is on), written to the run's log, and listed under `choices`
in `metrics.json`. Each has a setting to choose otherwise.

**Results are saved by default.** Runs write their folder, and reading stores the
representations it computes (see [Reading representations](#reading-representations)), so the
same text does not go through the same model twice. `concept_probe.configure(save=False)`
turns off every write the caller did not ask for, for the rest of the session.

## What the model sees

By default the library adds nothing to your text that you did not ask for:

- **Texts to read** (`score_texts`, `extract_text_reps`, texts-mode training, eval texts) are
  read raw: the tokenizer's BOS token and the text, with no chat template and no system prompt.
  The first position of any input is the attention sink, with activations far larger than other
  tokens'. With a BOS it is the BOS; with tokenizers that have none (Qwen, for instance) it is
  the text's first token, which is then left out of pooling, with a warning.
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

It returns the decoded input with every special and template token, the token ids, the span of
the text's own tokens, and `message_spans`: the role, token span and text of every message of
the chat (system prompt, user turns, the reply being read). `message_spans(tokenizer, messages)`
gives the spans for any conversation. `score_texts` records the same `input_text` and
`text_span`, and `score_prompts` records `prompt_text`.

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
role, `training.train_text_prompt`. The assistant role reads each text as the model's reply, and a
chat template cannot start a conversation with a reply, so it needs a turn before the text: set
`training.train_text_prompt` (a user turn, the same before every text) or a system prompt.
Without one, training stops before running anything. The library does not add one for you,
because whatever goes there is part of what the model reads; `preview_input` shows the result.
The `assistant_*` readouts pool over the text's own tokens
in every role: they leave out the BOS, the system block and role headers before the text, and
the end-of-turn tokens after it. The `sequence_*` readouts pool from the start of the input to
the end of the text, template included (the tokens after the text are left out, as in every
readout); the template part is identical in every input, so under mean pooling its weight is
1/length and scores and the direction partly track text length. Training warns when a
`sequence_*` readout is used.

To score other texts under the same readout, use `extract_text_reps` with the same `role`,
`prompt` and `system` (its defaults are the training defaults). It can return several readouts
from one forward pass:

```python
reps = extract_text_reps(model, tokenizer, new_texts, pool=["assistant_all_mean", "assistant_last"])
```

## Reading representations

`read_texts` reads texts with a model and returns their representations as arrays, with no run
folder and no probe:

```python
from concept_probe import Readout, read_texts

reps = read_texts(model, tokenizer, texts)                          # mean over each text's tokens
reps.data                                                           # (texts, layers, hidden_dim)
read_texts(model, tokenizer, texts, "assistant_max")                # per-coordinate max over the tokens
read_texts(model, tokenizer, texts, Readout(position=0))            # the text's first token
read_texts(model, tokenizer, texts, Readout(substring="river"))     # the tokens of a word
read_texts(model, tokenizer, texts, Readout(substring=words))       # one word per text
read_texts(model, tokenizer, texts, "assistant_tokens")             # every token is a row
read_texts(model, tokenizer, texts, layers=["embeddings", 0, -1])   # the embedding output too
```

Texts are encoded as everywhere else (`role`, `system`, `prompt`). A `Readout` takes the text's
own tokens (`span="text"`) or the input up to the end of the text (`span="sequence"`), narrows
them to the last `last_k`, one `position` or a `substring`, and reduces them with `agg`
(`"mean"`, `"max"`, `"first"`, `"last"`, or `"tokens"`, which keeps every token as a row, with
`reps.items` and `reps.positions` saying where each row comes from). The readout names of the
config (`assistant_all_mean`, `assistant_max`, `assistant_tokens`, ...) are shorthands for these.
A text without the position or substring raises `EmptySpanError` naming it, or, with
`skip_empty=True`, is listed in `reps.skipped`. Training takes the pooled readouts; training on
every token as a sample is not supported yet. `read_encoded` reads inputs already tokenized, such
as generated completions, without tokenizing them again.

Inputs go through the model `batch_size` at a time (default 16; `readout.batch_size` in the
config, which training uses). Batching is several times faster, and in half precision it changes
the last digits: with Qwen 2.5 0.5B in float16, reading 64 texts in batches of 16 instead of one
by one took 0.5 s instead of 3.8 s and moved a d of 3.4043 to 3.4041. `batch_size=1` reproduces
reading one input at a time exactly.

Reading stores what it reads in a cache and reuses it: an entry is used again only for the same
model identity, the same token ids, the same span and the same readout. The cache is one SQLite
file per model under `<output_root>/rep_cache` (`concept_probe.configure(output_root=...)`;
`<output.root_dir>/rep_cache` for a workspace, or `output.rep_cache_dir`). `cache=False`, or
`output.rep_cache = false` in a workspace, turns it off. Entries hold float32, so a cached read
equals a fresh one; `cache_dtype="float16"` (`output.rep_cache_dtype`) halves the size and rounds,
and keeps float32 for an entry with values float16 cannot hold.

The probe learns whatever separates the two sets of texts, so if the poles also differ in
length, negation, punctuation, topic or phrasing, the direction mixes the concept with that, and
held-out accuracy does not show it. The `datasets` guide (`concept_probe.guide("datasets")`)
describes how to write varied texts, audit them, and attack the probe until it cannot be fooled
by a confound in the data.

## Evaluation

Training measures how well the probe reads on **named eval sets**. One of them, the primary set,
chooses the layer and gives the numbers reported for the probe; every other set is measured at
that layer. `metrics.json` has them all under `evaluation.sets`, and the primary set's numbers
also at the top (`best_d`, `best_auroc`, ...).

- **Texts mode**: the primary set `"eval"` is the concept's `eval_pos_texts` / `eval_neg_texts`,
  read with the training texts' role and system prompt and pooled with `readout.read_mode`.
  `evaluation.eval_system` (null by default) overrides the system prompt, with a warning.
- **Shared and opposed modes**: completions generated for `prompts.eval_questions` under
  `pos_system` / `neg_system`. The primary set `"eval"` reads each completion as the reply to its
  question alone, without that system prompt: the prompt itself pushes the score toward its pole,
  so reading the completions after it inflates d (in the workshop's valence probes, about 10 with
  it against about 2.5 for the same probe on written texts). Reading them after it, as they were
  generated, is the set `"eval_with_system_prompt"`; `evaluation.primary_set =
  "eval_with_system_prompt"` makes it the primary one, as before. Eval texts given with such a
  concept are the set `"texts"`. `eval_questions` (20 by default) should not repeat the training
  questions; training warns if they do.
- **More sets**: `concept.eval_sets` = `{name: {"pos": [...], "neg": [...]}}` (optionally with
  `"paired"`, `"role"`, `"system"`, `"prompt"`), read like the eval texts, for out-of-distribution
  checks next to the main number.

For every set: Cohen's d at the probe's layer, its p and a bootstrap interval, the AUROC with its
interval, and **what random directions give** on the same items (`null_d95`: the 95th percentile of
|d| over `evaluation.n_random_directions` = 50 random directions). A d inside that is what any
direction would give: how far apart the sets are on their own, which varies a lot from one set of
texts to another. The primary set also has the **accuracy** with the calibrated threshold and its
interval. As a control of the whole procedure, training also refits the direction with the
training labels shuffled (`evaluation.n_label_shuffles`; by default 10 for the difference of
means, where it is quick, and none for slower estimators) and reports the |d| those directions get
(`label_shuffle`).

**Pairs.** Generated completions come in pairs: both poles answer the same question. Eval texts can
too (`evaluation.eval_pairs = true`: `eval_pos_texts[i]` and `eval_neg_texts[i]` are a minimal pair).
Paired items are split, resampled and permuted by pair, and their p-values come from the paired
t-test; an item whose pair-mate is empty is left out with it. Splitting the poles independently
would let a pair help choose the layer through one pole and then be measured through the other.

With fewer than 10 eval items per pole, training warns that the layer choice and `best_d` rest
on very few items. An item with no tokens of its own (a completion that ends right away, or an
empty text) is left out with a warning and listed under `dropped_items` in `metrics.json`; it is
never pooled over the prompt in its place, which would put the system prompt into the vector.
Generated completions are pooled up to and not including their end-of-sequence token.

The same evaluation works on representations already read, with no training run:

```python
from concept_probe import EvalSet, evaluate_reading, compare_scorers, cross_reading, auroc

ev = evaluate_reading(directions, [EvalSet("main", pos, neg, paired=True), EvalSet("ood", pos2, neg2)])
ev.layer, ev.sets["ood"].d, ev.sets["ood"].null_d95
compare_scorers(auroc, probe_pos, probe_neg, judge_pos, judge_neg)   # paired difference with its interval
cross_reading([valence_probe, interest_probe], {"valence": (vp, vn), "interest": (ip, inn)})
```

`compare_scorers` compares two ways of scoring the same items (two probes, or a probe and a judge)
with a paired bootstrap interval. `cross_reading` reads each probe on each set, with the
random-direction null and the cosine between the probes' directions: a probe can encode another
concept without anything showing it until it is read on that concept's texts.

## Layers

Every layer argument (`training.layer_idx`, `best_layer_search`, `layer_selection.layers`,
`steer_layers`, `reference_layer`, patch layers) takes layer indices: negative ones count from
the end, as in Python, and indices outside the model raise. A fraction of depth is written
explicitly, `{"frac": 0.75}`, so that `1` and `1.0` cannot mean different layers:

```python
config_overrides = {"training": {"best_layer_search": {"frac": [0.2, 0.8]}}}  # or [6, 20], or [[0, 3], [10, -1]]
```

## Layer selection and its statistics

The layer is chosen by a sweep over `training.best_layer_search` on the primary set, by largest
Cohen's d (`sweep_select = "effect_size"`) or smallest p among layers with d > 0 (`"p_value"`).
Choosing the layer and measuring its effect on the same items overstates the effect, so by default
(`evaluation.selection_split = 0.5`) the layer is chosen on half of the items and the reported
numbers come from the other half. `metrics.json` also reports:

- `selection`: `"split"`, `"all_eval"` (too few items to split, or `selection_split = null`: the
  numbers are then post-selection and a warning says so) or `"fixed_layer"`.
- `best_d_all_eval`: d at that layer on all of the primary set's items (post-selection).
- `sweep_max_d` and `sweep_max_d_perm_p`: the largest d over the searched layers and its
  permutation p-value (`evaluation.n_permutations`), which accounts for having searched them.
  It is computed on all items: it tests whether any searched layer separates the poles, not the
  held-out `best_d`.

`plots/sweep.png` draws d per layer on the held-out items (the curve the reported d comes from,
with that d marked), dashed on the items that chose the layer, and the random-direction null as a
gray band; `plots/score_hist.png` the held-out items' scores at the layer, and `sweep_<set>.png`
each other set.

Without evaluation, `best_d`/`best_p` are null and the sweep is NaN.

## Scores and steering units

Probe scores are raw projections onto the unit concept direction, with an arbitrary offset.
Each probe stores two references per layer, both from the evaluation items:

- per item (`center`, `scale` of its calibration): the midpoint between the poles' mean item
  score and the pooled within-pole SD of item scores, where an item's score is its mean over its
  tokens;
- per token (`token_center`, `token_scale`): the same over single tokens.

A single token spreads much more than a mean over a text's tokens, which averages most of the
token-to-token variation away, so the two scales differ (by 1.9-3.4x across layers in a 1B
Llama on ~16-token texts). Scoring functions give standardized scores next to the raw ones, each
with its own reference: per-token z (`token_z`, shown in the heatmaps) and the z of an item's mean
(`score_mean_z`). In both, 0 is the midpoint between the poles, so
positive leans toward the positive pole, and 1 is a within-pole SD.

**z = 0 does not mean "without the concept".** It is the midpoint between the mean scores of the
two poles' eval items, and text that belongs to neither pole need not land there. With a valence
probe on Llama 3.2 3B (layer 11), 50 neutral texts scored z = +0.54 on average, and texts about
food, cars, days of the week or plain statements between +0.3 and +0.7. Where neutral text lands
also depends on the training texts: valence probes trained on other sources put the same neutral
texts between -0.45 and -0.16. So z > 0 on a neutral text is not "positive"; to know where text
without the concept sits, score a set of it with the same probe. A concept's `neutral_texts`
does that during training: `metrics.json` reports where they fall (`neutral_texts`, in z of the
calibration measured on texts read the same way), without using them as the zero.

**A threshold to classify.** The midpoint is not the best cut between the poles: a probe can
order items well and cut them badly. Training fits a one-variable logistic regression of the pole
on the item score at the best layer, on the half of the eval items that chose the layer (or all of
them without a split), and saves it as `threshold` in `metrics.json` and in the calibration;
`ConceptProbe.classify` uses it.

**z for another reading context.** A probe trained on texts has no z references for the
completions `score_prompts` generates (they are read differently), so their `score_mean_z` is
None. The scorers say why (`scores.z_missing`) and warn. `probe.calibrate(pos_texts,
neg_texts, as_reply_to=prompts)` measures references on texts read as replies to those prompts,
which is how generations are read; with `role` / `system_prompt` / `prompt` instead, it calibrates
texts read that way.

References only apply to inputs read the way they were measured: each calibration says how its
items were read. In shared and opposed modes the primary set's calibration comes from completions
read as replies to their question, which is how `score_prompts` reads generations; texts read raw
sit elsewhere entirely (in a 1B Llama, the eval texts of the negative pole came out on the positive
side of the generated-completion midpoint). `score_texts` therefore standardizes only with a
calibration measured on texts read with the same role, system prompt and prompt: the primary
set's in texts mode, the `texts` set's in shared and opposed modes. Without one it saves only raw
projections, with a warning.

### Steering

By default steering touches one layer: the probe's `best_layer` (`steering.steer_layers =
"probe"`). It adds `alpha_l * v_l` to every token position of each steered layer `l`, where `v_l`
is the probe's own unit direction at that layer: with several layers, each uses the direction
trained there, not the chosen layer's vector copied to the others. Several layers are steered
with `steer_layers = "window"` (`best_layer` +/- `steer_window_radius`, clipped at the model's
edges) or a list of layers. Then, with `steer_distribute = true` (default), alpha is divided by the
number of layers actually steered (3 instead of 5 for a window at the first layer).

`alpha_unit` sets what alpha is measured in:

- `"raw"` (default everywhere): alpha as is.
- `"sigma"`: alpha times `steer_sigma`, the eval items' pooled within-pole SD on the steering
  direction at the probe's layer. It is a
  scale knob, not "n standard deviations of behavior".
- `"resid_norm"`: alpha times each steered layer's mean residual norm on the training tokens.
  With one steered layer, `alpha=0.1` adds 10% of a typical token's norm; distributed over n
  layers, each adds `0.1 / n` of its own layer's norm.

The scores' `settings` have the steered layers (`steer_layers`), and each item's `info` the alpha
each layer applied after distribution and units (`steer_alpha_per_layer`) and the intervention.
`probe.steering(alpha, unit)` gives that intervention, to use anywhere (see [Interventions](#interventions)).

Generation is greedy by default (`steering.steer_sampling = false`), so a prompt at an alpha has
one completion. With sampling, the RNGs are seeded with `seed`, or else `random.seed`, so the
completions are reproducible for the same batch size (best effort: GPU kernels are not
bit-deterministic). The scores' `settings` have the generation settings under `generation`. An
empty completion has `score_mean` null.

`n_random` adds random controls in the same call: at each non-zero alpha, the same steering with a
random direction of the same length at each layer (`info["control"]` is its number, None for the
probe's own steering). Control `k` is the same direction at every alpha, seeded by `random_seed`
(else `random.seed`). It is one direction for all the steered layers, with each layer's norm:
concept vectors of adjacent layers point nearly the same way, so independent random directions
would partly cancel and push the residual stream about half as far as the concept. One control
says little; [Evaluating steering](#evaluating-steering) compares the steering with ten.

Repeated alphas (several samples, with sampling on) are separate items (`info["alpha_index"]`). A
`batch_subdir` that already holds files raises unless `overwrite=True`.

## Generating and scoring

`score_texts`, `score_prompts` and `multi_probe_score_prompts` return a `Scores` object in memory:
one item per text or generation, in order (for generations, prompt by prompt, each prompt's
alphas in the order given).

```python
scores = probe.score_prompts(prompts, alphas=[-4.0, 0.0, 4.0])
scores.means(), scores.z()            # each item's mean token score and its z (NaN where none)
alphas = scores.column("alpha")       # any value of the items' info, to group or filter them
item = scores[0]
item.text, item.tokens, item.span     # the completion, every token, and the completion's span
item.token_scores["valence"]          # the probe's score of every token (keyed by probe name)
item.token_z["valence"]               # their z, or None without a calibration for this reading
item.scores["valence"]                # raw projections per token and scoring layer
```

Prompts are generated `batch_size` at a time (`steering.gen_batch_size`, by default
`training.gen_batch_size`, 16), padded on the left; completions are then read again, without the
steering, in batches of `readout.batch_size`. On a Qwen 2.5 0.5B in an RTX 2060, eight prompts went
from 10.5 s one at a time to 1.3 s in a batch, with the same greedy completions (in half precision
one of eight can differ in its last tokens; `batch_size=1` reproduces generating alone).

Each call saves (decision D4; `save=False`, or `concept_probe.configure(save=False)`, turns it off)
one folder: `scores.npz` (every item's token ids, tokens and per-token scores per layer, per probe
and in z), `scores.jsonl` (one readable line per item: prompt, alpha, text, mean score, z),
`heatmaps.html` (every item, with a menu to pick the probe; positive in blue, negative in orange)
and `settings.json`. `Scores.load(folder)` reads it back. `save_segments=True` adds each message's
mean score to the lines.

The pieces are public:

- `generate(model, tokenizer, prompts, ...)` generates in batches, optionally under an
  intervention, and returns the generations (token ids, where the completion is, text);
  `on_token=` is called at every step with the score of each probe in `watch=` on the token just
  read (with steering on, that reading includes the added vector); `keep_logits="all"` or a number
  keeps the logits (or the top ones) of each generated token, as the steered model produced them.
- `answer_logprobs(model, tokenizer, prompts, answers, intervention=...)` gives log P(answer |
  prompt) for given answers, in batches: `probs()` renormalizes them, `log_odds(0, 1)` compares
  two. An answer is tokenized on its own, so write it as the model would (with a leading space
  when it would put one); `answer_tokens` shows the tokens.
- `score_text_tokens(model, tokenizer, vectors, text)` scores every token of a raw text with any
  directions.

## Evaluating steering

`evaluate_steering` sweeps the doses of a steering intervention, with random controls, and
measures every completion:

```python
from concept_probe import evaluate_steering

ev = evaluate_steering(probe, prompts, doses=[-0.25, -0.1, 0.1, 0.25], unit="resid_norm",
                       n_random=10, measures={"positivity": my_classifier})
ev.row(0.25)["effects"]["positivity"]   # change from dose 0, its interval, the controls' p95
ev.largest_coherent_dose                # {"positive": 0.1, "negative": -0.25}, say
```

- **Doses** are multiples of the unit (a probe's `steering(dose, unit)`), of an `Add` given at dose
  1, or the argument of a function `dose -> Add`. Dose 0 is the baseline, added when missing.
- **Random controls**: at each non-zero dose, `n_random` (10 by default) same-length random
  directions, each the same at every dose. For each measure the table gives their mean changes and
  the 95th percentile of their absolute values, next to the steering's change: what a direction
  without the concept does to that measure.
- **Measures of effect**: each probe in `probes` and the steering probe, read without the
  intervention (in z where the probe has a calibration for generated completions, raw otherwise;
  the steering probe's reading is informative but not independent of its steering), and any
  function of the texts in `measures` (a classifier, an embedding model, or a judge's
  `measure(rubric)`, see [Judges](#judges)). Each is reported as the mean change from dose 0,
  paired by prompt, with a bootstrap interval over prompts.
- **Coherence**: each completion's perplexity under the model without the intervention (as the
  reply to its prompt), its share of repeated word 3-grams and its length. By default
  (`CoherenceRule`) a dose is coherent when the median perplexity stays within half and twice
  that of dose 0 and the repetition rises by at most 0.2; `coherence=` takes any
  `rule(at_dose, at_zero) -> bool`. The largest coherent dose on each side is the largest one
  whose smaller doses are all coherent too.

It saves (by default, under the probe's run folder or the session's `output_root`) the scores of
every completion with their measures, `steering.json` with the table, and `dose_response.png`: the
change of each measure per dose with the controls in gray, and perplexity and repetition with the
rule's limits, incoherent doses hollow. On a Qwen 2.5 0.5B in an RTX 2060, 8 prompts at 5 doses
with 5 controls each (248 generations of 32 tokens) took a minute.

## Datasets, confounds and beyond two poles

**Auditing a dataset** (no model): `audit_dataset(pos_texts, neg_texts, paired=..., checks=...)`
compares the poles' length (characters, words, and tokens with `n_tokens`), lists the words much
more frequent in one pole, flags repeated openings and near duplicates, measures variety, and
trains classifiers that see only words, only length, only your checks (and any embedding you pass
as `embed`); it writes `audit.md` and `audit.json`. Nothing about a particular concept or language
is checked by default: anything else you suspect is a check, a function `text -> number`, e.g.
`checks={"negation": count_matches(r"\b(not|no|never)\b|n't\b")}`. The pieces are separate
functions (`compare_poles`, `surface_baseline`, `distinctive_words`, `formulaicity`, `diversity`).
`surface_baseline` is also the number a probe has to beat: a probe that does not read better than
a bag of words did not need the model.

**Critiquing a dataset.** A reader notices what no check was written for. `critique_task(pos, neg,
folder, concept=...)` writes a task for an agent (a sample of each pole, what to look for: differences
besides the concept, formulaic writing, mislabeled items, and the exact JSON to write), and
`read_critique(folder)` reads and checks the answer; `critique_dataset(pos, neg, judge)` asks a
judge behind an API or loaded here directly. `critique.checks()` turns the differences that come
with a regular expression into checks for the audit, so each hypothesis is measured on every text.

**Confounds.** `confound_check(directions, layer, {"negation": (with, without)})` says, for
representations of items with and without a feature, whether the probe separates them beyond
random directions, how separable the feature is, and the cosine between its direction and the
probe's. To train without it, fit an `Eraser` on the training representations and each item's
confound label, and pass it to `ConceptProbe.fit(..., erase=eraser)` (or `fit_directions`):

```python
eraser = Eraser.fit(np.concatenate([pos, neg]), has_negation, method="leace")   # or "project"
probe = ConceptProbe.fit(pos, neg, layer=14, erase=eraser, model=model, tokenizer=tokenizer)
```

`"leace"` (least-squares concept erasure, Belrose et al. 2023) makes the smallest change after which
no linear classifier tells the confound apart; `"project"` removes the confound's directions. The
probe stores directions that read raw representations, so it is used as any other. On Qwen 2.5
0.5B, the workshop's valence probe read negation (d -0.47 between texts with and without one;
random directions 0.34); with negation erased it read it at d 0.00 and valence as well as before.

**Learning curves.** `learning_curve(pos, neg, eval_sets, sizes=[8, 32, 128])` refits the probe on
random subsets of each size and measures it on fixed eval sets, with the same done on random groups
of the same items as the null, the cosine with the direction from every item, and how often the
same layer is chosen.

**Attacking a probe.** `red_team(probe, {"name": {"pos": [...], "neg": [...], "neutral": [(a, b)]}})`
reads items written to exploit a suspected confound: each pole's items that carry the other pole's
surface, and pairs of neutral texts that differ only in the feature. It reports each pole's
accuracy and how far the feature alone moves the probe, in z.

**Beyond two poles: any target, and its shape.** A `Target` says what each item is:
`Target.categories(labels)`, `Target.number(values)`, `Target.order(labels, classes)`,
`Target.cycle(labels, classes)` (e.g. days of the week, evenly around a circle) or
`Target.layout(labels, {class: coordinates})`. Two questions, each layer by layer and with its
control:

- `read_target(reps, target, model=...)`: can it be read? A map fitted on some items and tested on
  others: `"centroid"` (the nearest class mean), `"linear"` (ridge, optionally of low `rank`),
  `"shape"` (through the plane of the class centroids), or any model with `fit`/`predict`, nonlinear
  ones included. Accuracy for classes, R² and rank correlation for numbers, next to the same with the
  target shuffled (between items, or between classes: which class gets which place).
- `target_shape(reps, target)`: do the class centroids have the target's shape? The correlation
  between their distances and the places' distances, and how well their own top plane (or line)
  matches the places up to rotation and scale (`fit="affine"` also allows stretching), next to
  shuffled orders. `shape.plot(path)` draws the centroids joined in order.

With many dimensions and few classes almost any arrangement can be read, so reading a cycle says
little about its shape: in Llama 3.2 3B, a rank-2 linear map read the days of the week perfectly,
and just as perfectly with the days shuffled. Their centroids lie on a circle in order (fit 0.97,
shuffled orders 0.54, at layer 18), and reading through that plane names the day of new sentences
(100%, shuffled 57%). The two-pole `ConceptProbe` does not use this module and is not changed by it.

**A probe toward a target.** `TargetProbe.fit(reps, target)` fits the map once (at the layer that
reads the target best on held-out items; among equally good layers, the one whose centroids fit
the shape best) and then works like any probe:

```python
probe = TargetProbe.fit(read_texts(model, tokenizer, day_sentences, "assistant_last"),
                        Target.cycle(day_names, ["Monday", "Tuesday", ..., "Sunday"]))
probe.attach(model, tokenizer)
s = probe.read(["The party is on Thursday"])    # s.position: 2.86 (steps from Monday), s.nearest: Thursday
probe.score_tokens(["I will call you on Friday"])  # the same for every token
probe.score(new_reps)                          # on representations already read
probe.steering(1, layer=6)                     # rotate one step around the days' circle
probe.toward(source="Monday", steps=1, layer=6)  # add the whole step from Monday's centroid to Tuesday's
```

Each activation gets a score that fits the target: for a cycle, its place on the circle in class
steps (continuous), the nearest class, and how far it sits from the center (1 is on the circle); for
an order or a number, the value; for categories, a score per class and the nearest one. The probe
saves and loads (`save`, `TargetProbe.load`), with how well it read held-out items and, for a
structured target, how well the centroids have its shape.

It moves activations in two ways, at any layer. `steering(steps)` moves them along the shape: a
rotation in the centroids' plane about the cycle's center (one step is one class), or a step along
an order or a number's direction. `toward(...)` adds the whole difference between two classes'
centroids, in every dimension, for any target with classes. They need not do the same: the days
form a circle, but rotating within it did not change what Llama 3.2 3B (0 of 7) or Qwen 2.5 0.5B
(at most 2 of 7) answered for "tomorrow", while the whole step did (Llama 7 of 7 at layers 4 to 12;
Qwen 4 to 6 of 7 at layers 2 to 16, against at most 2 of 7 for a random step of the same length).

**Several models.** There is no batch runner: load each model, read, train and evaluate, and
compare at the same relative depth.

```python
import gc
import numpy as np
import torch
from concept_probe import ModelBundle, read_texts, ConceptProbe, EvalSet, evaluate_reading
from concept_probe.config import resolve_config

results = {}
for model_id in ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]:
    bundle = ModelBundle.load(dict(resolve_config()["model"], model_id=model_id))
    read = lambda texts: read_texts(bundle.model, bundle.tokenizer, texts)
    pos, neg, eval_pos, eval_neg = read(train_pos), read(train_neg), read(test_pos), read(test_neg)
    probe = ConceptProbe.fit(pos, neg, select=(eval_pos, eval_neg), model=bundle.model, tokenizer=bundle.tokenizer)
    ev = evaluate_reading(probe.directions, [EvalSet("eval", eval_pos.data, eval_neg.data)])
    n = probe.num_layers
    results[model_id] = {"layer": ev.layer, "depth": ev.layer / (n - 1), "d": ev.main.d,
                         "d_by_depth": dict(zip((np.arange(n) / (n - 1)).round(2), ev.main.d_curve))}
    del bundle, probe; gc.collect(); torch.cuda.empty_cache()
```

Each reading is cached by model, so a second pass over the same texts does not run the model again.

## Interventions

An intervention says what to change in the residual stream, at which layers and positions, with
any vectors, and `intervene(model, intervention)` applies it while a `with` block runs: whatever
the model does inside (generating, reading, a forward pass) runs on the edited stream.

```python
from concept_probe import Add, Ablate, Patch, Rotate, Edit, intervene

Add({12: v}, alpha=4.0)                       # steer: add 4 v at layer 12 (a dict, or an array per layer)
Ablate(directions, layers=[10, 11, 12])       # project the direction out (several rows: a subspace)
Ablate({12: v}, value=0.5)                    # set the stream's coordinate along v to 0.5
Patch(12, positions=[-1], values=donor)       # overwrite the last prompt token with a donor's activation
Rotate(u, w, angle=np.pi / 4, layers=[12])    # rotate in the plane of u and w
Edit(my_function, layers=[12])                # anything: my_function(layer, hidden, where) -> hidden
Add({12: v}, 4.0) + Ablate({20: u})           # several, in order

with intervene(model, Ablate(probe.directions, layers=range(16))):
    reps = read_texts(model, tokenizer, texts)   # read with the direction removed
```

Layer `l` is the output of block `l`, as everywhere. Positions count each row's own tokens from 0,
without padding, so an intervention means the same in a batch padded on either side and while
generating with a cache; negative positions count back from the end of the prompt, and
`"prompt"` / `"generated"` select those parts (`None`, the default, is every position).
`probe.steering(alpha, unit)` builds the `Add` a probe steers with, and `generate`,
`answer_logprobs` and `score_prompts` take any intervention. Reading under an intervention reports
the stream after the edit at an edited layer (what the next block receives; Hugging Face's
`output_hidden_states` records it before) and neither reads nor fills the representations cache.

## Directions: reading and steering

`training.estimator` chooses how each layer's direction is estimated from the representations of
the two poles:

- `"mean_diff"` (default): the difference of means, `mu_pos - mu_neg`.
- `"mean_diff_var"` / `"mean_diff_sd"`: the difference of means divided coordinate-wise by the
  pooled within-pole variance (or SD), `0.5 * (var_pos + var_neg)`. This is the diagonal form of
  the covariance correction of mass-mean probing; it keeps a few very high-variance coordinates
  from dominating the direction, as with the massive activations of Gemma 3.
- `"mean_diff_drop_outliers"`: the raw difference of means without the massive coordinates the
  concept does not move (pooled SD more than 10 times the layer's median, |t| below 2).
- `"logistic"`: L2-regularized logistic regression on standardized coordinates (`C` = 0.05).
- `"lda"`: linear discriminant analysis with a Ledoit-Wolf shrunk covariance.

A dict sets parameters (`{"kind": "logistic", "C": 1.0}`), and `register_estimator(kind, factory)`
adds an estimator: any object with `fit(pos, neg)` returning a direction. Every direction has unit
length and points toward the positive pole. `logistic` and `lda` need scikit-learn (the `stats`
extra).

**Steering has its own direction.** Reading and steering do not favor the same estimator. Across
Llama, Qwen and Gemma 3 models, `"mean_diff_var"` read better (by a wide margin in Gemma and Qwen)
but steered less: with a valence probe, the change in an external classifier's positivity between
steering at +alpha and at -alpha (at the largest alpha that kept the text coherent) fell from +0.22
to +0.12 in Llama 3.2 3B and from +0.18 to +0.03 in Qwen 2.5 3B, and some models lost coherence at
smaller alphas. So `steering.steer_direction` is a separate setting: `"mean_diff"` (default, the raw
difference of means), `"read"` (the reading direction), or any estimator. When the two differ,
training says so, saves both (`concept_vectors`, `steer_vectors` in `tensors.npz`), and measures the
unit of `alpha_unit="sigma"` on the steering direction (`steer_sigma_best_layer`). Probes saved
before this steer with the one direction they have, and loading one trained with another estimator
says so.

```python
config_overrides = {"training": {"estimator": "mean_diff_var"}, "steering": {"steer_direction": "mean_diff"}}
```

## The probe object

A `ConceptProbe` holds what a probe is: one direction per layer to read with (`directions`), the
layer it reads at (`layer`), the directions it steers with (`steer_directions`), and its
`calibrations`. It always belongs to a model (`identity`: the model, dtype, quantization and chat
template its directions were computed on), but needs that model loaded only to read texts,
generate and steer. A probe is built:

- by training it from a concept (`ProbeWorkspace.train_concept`, as above);
- from representations already read, without loading the model again:

```python
from concept_probe import ConceptProbe, ReadContext, read_texts

pos = read_texts(model, tokenizer, pos_texts)           # records which model it read them with
neg = read_texts(model, tokenizer, neg_texts)
probe = ConceptProbe.fit(pos.data[:40], neg.data[:40], identity=pos.identity,
                         estimator="mean_diff_var",
                         select=(pos.data[40:60], neg.data[40:60]),  # chooses the layer
                         calibrate=(pos.data[60:], neg.data[60:]),   # zero, unit and threshold of z
                         context=ReadContext(role="raw"))            # how those items were read
```

  `fit` takes the arrays or what `read_texts` returns (then the model is known); otherwise
  `identity=` (a dict or the model id) or `model=` says which model they come from, and one of
  them is required. `baseline=True` reads the second group as a baseline (text without the
  concept) rather than an opposite pole: the zero is then the baseline's mean;
- from directions computed elsewhere: `ConceptProbe.from_directions(directions, layer=12, identity=...)`;
- from a saved run: `ConceptProbe.load(run_dir)`, or `ProbeWorkspace.get_probe` (see below).

With representations, a probe needs no model:

```python
probe.z(reps, context)          # z at the probe's layer, with the calibration of that reading context
probe.classify(reps, context)   # positive?, and the probability, from the calibrated threshold
probe.calibrate(pos, neg, context=...)   # a calibration for another reading context
probe.project(reps)             # raw projections, every layer
```

Reading texts, generating and steering need the model: `probe.attach(model, tokenizer)` gives it
one (a model of another identity raises unless `allow_model_mismatch=True`), and then
`score_texts`, `score_prompts` and `calibrate(pos_texts, neg_texts, ...)` work as described above.

A `ReadContext` says how items were read (texts with a role, system prompt and prompt, or
generated completions). A `Calibration` holds the zero and unit of z measured on items read that
way, a threshold, and where neutral texts fall; a probe has one per context it was calibrated in,
and asking for z in another context raises, naming the contexts it has.

Probes save to their run folder by default: `fit` and `from_directions` write one under
`<root>/<concept name>/<run_name or timestamp>`, where `root` is their `root_dir` argument, or else
the session's `output_root` (`concept_probe.configure(output_root=...)`), unless `save=False`. A
probe without a folder gets one there the first time it saves scores, and `probe.save()` writes its
current state (a new calibration, for instance) into its folder.

## Behavioral evals

A behavioral eval is a steering evaluation with a measure of the task: give each prompt its item,
and measure the completions with a function of the texts and their items.

```python
items = [{"prompt": "What is 2 + 2? Answer after ANSWER:", "expected": "4"}, ...]
ev = evaluate_steering(probe, [it["prompt"] for it in items], doses=[-0.2, 0.2], items=items,
                       measures={"correct": exact_match(marker="ANSWER:")})
ev.row(0.2)["effects"]["correct"]        # change in accuracy from dose 0, against random directions
ev.relate(probe.concept.name, "correct")  # does the probe's score predict being right?
```

`exact_match` compares the completion (or the first line after `marker`, or the last before it)
with `item["expected"]` (a value or a list of accepted values), after stripping whitespace and a
trailing period, as numbers when both read as numbers, case-sensitively unless
`case_sensitive=False`. Under strong steering, format drift counts as a wrong answer, so read
completions before taking a drop as the model knowing less. Any other check is a function
`(texts, items) -> numbers`.

`relate(predictor, outcome)` fits a GEE of the outcome on the predictor (logistic for a 0/1
outcome, linear otherwise), with the dose as a covariate and each prompt as a cluster (it recurs at
every dose), with standard errors corrected for few clusters. The predictor is in SDs within a
dose: across doses it would include the steering shift, and the effect would grow with the range of
doses run.

`multi_probe_score_prompts` runs one model for every probe, so the probes must come from the same
model identity (the model, the dtype used, the quantization, the chat template and its date). The generation
settings (system prompt, `max_new_tokens`, sampling and, when sampling, `temperature`, `top_p`
and the seed) come from the arguments or, when not passed, from the probes' configs, which must
then agree. Non-zero alphas need `steer_probe` (index, name or probe); the steering layers and
their distribution come from its config. Each item has every probe's scores, keyed by its name;
`scores.npz` has them as `scores_<i>` and `layers_<i>` in the order of `probe_names`. Warnings go
to the run's log as well as the console.

`rate_saved(folder, judge, rubric)` rates a saved batch's completions on any rubric and saves
`<rubric>_rating.json`, one answer per generation by its index in the scores, with the judge and
the date: they are labels from another model, not ground truth.

## Judges

A judge is another model that rates texts on any dimension you describe. A `Rubric` says what to
ask about each text and which answers count, each with a number; `COHERENCE` is one, ready made.

```python
from concept_probe import APIJudge, LocalJudge, Rubric, COHERENCE

positivity = Rubric("positivity", "How positive is the feeling this text expresses?",
                    {str(k): float(k) for k in range(1, 6)}, guide="1 very negative, 3 neutral, 5 very positive.")

judge = APIJudge("anthropic", model="<a model id>")          # or "openai", "gemini", "openrouter", "groq"
judge = APIJudge("ollama", model="llama3.1")                # a local Ollama server
judge = APIJudge("openai-compatible", model="m", base_url="http://localhost:8000/v1")   # vLLM, LM Studio...
judge = LocalJudge(judge_model, judge_tokenizer)            # a model loaded here
judge = AgentJudge("tasks")                                 # an agent, a subagent or a person (see below)

judge.rate(texts, positivity)        # the answers
judge.score(texts, positivity)       # their numbers
judge.measure(COHERENCE)             # texts -> numbers, for evaluate_steering(measures=...)
```

- **Behind an API**, a judge gets several texts per request (`texts_per_request`) as a JSON list
  and returns one answer per text, at temperature 0. The key is `api_key`, else the provider's
  variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`,
  `GROQ_API_KEY`) from a file given as `env_path`, else from the environment; it is never saved.
  Building one is what sends texts out. `model` has no default: it is the service's id of the
  model you choose.
- **Loaded here**, a judge reads the probability of each answer as the start of its reply, with no
  sampling, and nothing leaves the machine: `rate` takes the most probable answer and `score` the
  expected value. Use a model other than the one you study, so it does not share its
  representations, and write the answers as it would start its reply.
- **An agent** (Claude Code, a subagent it starts, any program or a person): `AgentJudge` writes the
  task to a folder (`TASK.md` with the question, the allowed answers and the exact format of the
  answer; `items.jsonl` with the texts) and raises `AnswersPending`; once `answers.jsonl` is written
  there, the same call reads it and checks it, reporting anything missing or not allowed. No API and
  no model. In a steering evaluation, add the measure after the sweep:
  `ev.add_measure("coherence", judge.measure(COHERENCE))`. The `delegating` guide has the details.

## Loading a trained probe

```python
workspace = ProbeWorkspace(project_directory="outputs/sad_vs_happy/<run>")
probe = workspace.get_probe()
# or, with a model already loaded, or none:
probe = ConceptProbe.load("outputs/sad_vs_happy/<run>", model, tokenizer)
```

This loads the run's own model settings. `get_probe` on a workspace created otherwise raises if
the model identity (the model, the dtype actually used, the quantization, the chat template or
its date) differs from the run's, since the directions only apply to the activations they were
trained on; `allow_model_mismatch=True` loads it anyway. Runs saved before identities were
recorded are compared on their config, whose dtype may be `"auto"`: that one is not compared.

## Configuration

`concept_probe/defaults.json` holds every setting. Overrides with a key the defaults do not have
(a misspelling, for instance) raise instead of being ignored. The model settings are described in
[Loading the model](#loading-the-model); `readout.batch_size` and `output.rep_cache`,
`output.rep_cache_dir` and `output.rep_cache_dtype` in [Reading representations](#reading-representations);
`steering.gen_batch_size` and `output.generation_logits` (null, `"all"` or a number of top logits
to keep) in [Generating and scoring](#generating-and-scoring).

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
  concept_probe/guides/   # guides for people and agents (concept_probe.guide())
  docs/                   # the design (DESIGN.md) and the list it answers (LIBRARY_ISSUES.md)
  AGENTS.md               # where agents start
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
    read_texts,
    read_encoded,
    Readout,
    ReadContext,
    evaluate_reading,
    EvalSet,
    fit_directions,
    message_spans,
    ModelBundle,
    configure,
    multi_probe_score_prompts,
    Scores,
    generate,
    answer_logprobs,
    evaluate_steering,
    APIJudge,
    LocalJudge,
    Rubric,
    audit_dataset,
    learning_curve,
    confound_check,
    Eraser,
    Target,
    TargetProbe,
    read_target,
    target_shape,
    AgentJudge,
    critique_task,
    exact_match,
    guide,
    intervene,
    Add,
    Ablate,
    Patch,
    Rotate,
    Edit,
    train_concept_from_json,
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

- **Fixed (found re-running the workshop with this version):**
  - in float32 on a GPU, reading in batches could read one text wrong at every layer, silently
    (Gemma 3 1B: the longest text of a padded batch, through PyTorch's memory-efficient attention
    kernel). Reading, generating and `answer_logprobs` now leave that kernel out in float32 on a GPU,
    and say so when the model loads;
  - `learning_curve` split the eval items differently for every draw, so the spread across draws
    mixed in which items measured; now they are split once;
  - a probe saved with the model's path in the Hugging Face cache refused the same model loaded by
    its repo name, and a probe given only the model id (`identity="org/name"`) refused that model;
    identities now compare the repo, and only what both sides say.
- **Probes toward any target, and the shape of representations** (`structure.py`): `Target`
  (categories, a number, an order, a cycle, a layout), `read_target` (centroid, linear, through the
  shape, or any model with fit/predict; with a shuffled-target control) and `target_shape`
  (representational similarity and Procrustes fit of the class centroids, with shuffled orders;
  `plot`). They replace `fit_classes`, `fit_regression`, `centroid_subspace` and `fit_circle`. The
  two-pole probe is not changed (the reference outputs are the same).
- **`TargetProbe`**: a probe toward any target, fitted once, that scores new activations (`score`,
  `read`, `score_tokens`: a place on a cycle, a value, the nearest class, how well it fits), saves
  and loads, and moves activations along the shape (`steering`) or between classes (`toward`), at
  any layer. `Rotate` takes a `center`.
- **The audit checks nothing particular by default**: length, words, duplicates and variety;
  `checks={name: text -> number}` (and `count_matches(regex)`) for anything else. The word groups
  (negation, persons...) and the function-word list are gone; `surface_features` is `text_features`,
  and `surface_baseline` takes `"words"` (with an optional `vocabulary`), `"length"`, a dict of
  checks or a function.
- **Agents as readers**: `AgentJudge` writes a rating task to a folder and reads the answers;
  `critique_task` / `read_critique` (and `critique_dataset` for a judge) critique a dataset, and
  `Critique.checks()` feeds the audit. Guides ship with the package (`concept_probe.guide()`), and
  `AGENTS.md` points agents to them; `docs/building_datasets.md` became the `datasets` guide.
- **Behavioral evals are steering evaluations**: `evaluate_steering(..., items=, measures=)` with
  measures of the texts and their items (`exact_match`), `SteeringEvaluation.add_measure` (after the
  sweep, e.g. from an agent), `SteeringEvaluation.load` and `relate` (the GEE of one measure on
  another, with the dose as a covariate). They replace `run_scored_eval`, `run_multi_scored_eval`,
  `simple_equality_evaluator`, `aggregate_eval_batches`, `rehydrate_batch_analysis`,
  `rate_batch_coherence_safe`, `generate_multi_probe_report` and their modules.
- The default coherence rule is documented as a heuristic.
- **Judges on any dimension, from any provider**: a `Rubric` (question, allowed answers with their
  numbers) answered by an `APIJudge` (OpenAI, Anthropic, Gemini, OpenRouter, Groq, Ollama or any
  OpenAI-compatible server, through the standard library) or a `LocalJudge` (a model loaded here,
  reading each answer's probability). `COHERENCE` is one rubric. They replace `coherence.py`
  (`rate_coherence`, `rate_batch_coherence`, the `groq` extra) and `coherence_judge`; the eval runs
  take `coherence_judge=` (a judge) instead of `rate_coherence`, `coherence_model`,
  `coherence_max_per_request` and `coherence_env_path`, and `rate_saved` rates a saved batch on any
  rubric. Ratings record the judge (`judge`) and the answer's number (`value`).
- **Datasets and extras** (stage 6): `audit_dataset` and its pieces (surface features, surface /
  function-word / word / embedding baselines, distinctive words, formulaicity, diversity);
  `confound_check`; `Eraser` (LEACE or projection) to fit directions without a confound
  (`fit_directions(..., erase=)`, `ConceptProbe.fit(..., erase=)`); `learning_curve` with its
  random-group null; `red_team`; `fit_classes`, `fit_regression`, `centroid_subspace` and
  `fit_circle`; and a recipe for several models.
- **Evaluating steering** (`evaluate_steering`, entries 23 and 24 of the workshop's list): a dose
  sweep with `n_random` random controls of the same length, local coherence measures (perplexity
  under the unsteered model, repetition, length), probes and any function of the texts as measures
  of effect (the Groq judge is one, `coherence_judge`), a replaceable coherence rule and the
  largest coherent dose on each side, with a table, its files and a plot.
- **Random controls come in the same call**: `score_prompts` and `multi_probe_score_prompts` take
  `n_random` (and `random_seed`) instead of `random_steering_vector` (one random direction per
  call) and `random_vector_seed`; a control now has the steering vectors' length at each layer
  (it had the reading directions'). `random_scoring_vector` is gone from `score_texts`,
  `score_prompts` and `multi_probe_score_prompts`: the random-direction null of reading is in the
  evaluation (`null_d95`). `Add.scaled(factor)` and `Add.random_like(seed)` give a dose of an
  addition and its control.
- **Interventions** (`concept_probe.interventions`): `Add`, `Ablate` (project out, or set a
  coordinate; one direction or a subspace), `Patch`, `Rotate`, `Edit` (a function of your own) and
  their sums, applied with `intervene(model, ...)`, at layers and positions counted per row without
  padding. Reading under one reports the edited stream at the edited layer and skips the cache.
  `probe.steering(alpha, unit)` builds a probe's steering. They replace `ResidualStreamEditor`,
  `ActivationPatch`, `build_patch_edit_fn`, `patched_forward_hidden_states`,
  `cache_residual_stream`, `LayerSteerer` and `MultiLayerSteererLayerwise`, which are gone.
- **Generation in batches, in memory**: `generate` (left padding, an optional intervention,
  `on_token` with each watched probe's score while generating, `keep_logits`) replaces
  `generate_once` and `generate_batch`; training's opposed mode now generates in batches too (the
  same completions on the test model). `answer_logprobs` gives the probability of given answers
  under an intervention.
- **Scoring returns `Scores`**: `score_texts`, `score_prompts` and `multi_probe_score_prompts`
  return every item's tokens and per-token scores in memory, instead of a list of records (or a
  dict) pointing at one npz and one HTML file per generation. They generate and read in batches
  (8x faster in a 0.5B model), save one folder per call (`scores.npz`, `scores.jsonl`,
  `heatmaps.html`, `settings.json`) and follow the session's `save` setting. The per-layer token z
  (`scores_z_per_layer`) is no longer saved (it is the per-layer scores standardized with the token
  references). Sampled completions differ from before, since a batch draws its tokens in another
  order. Heatmaps show positive in blue and negative in orange, as the other figures.
- **Settings removed**: `steering.do_steering` (the alphas decide), `output.save_token_scores_npz`
  (saving is on or off as a whole), `output.save_generation_logits`, `generation_logits_top_k` and
  `generation_logits_dtype` (now `output.generation_logits`; the logits come from the generation
  itself, with no extra pass). Runs saved with them still load. `steering.gen_batch_size` is new.
- **Removed**: `forward_hidden_states_all_layers` (`read_encoded(..., "sequence_tokens")` reads
  every token of an input; the readout is new), `token_scores_from_hs_layers`,
  `render_token_heatmap`, `render_token_heatmap_multi` and `render_batch_heatmap` (one renderer,
  `render_heatmaps`).
- **Evals read from memory**: `run_scored_eval` / `run_multi_scored_eval` build their records
  from the returned scores (`scores_index` instead of `npz_path` / `html_path`); coherence ratings
  are matched by that index instead of by file name, and `rate_coherence(texts)` rates texts in
  memory. `generate_multi_probe_report` accepts the returned run.
- **Loading.** `model.dtype` defaults to `"auto"`: bfloat16 on GPUs that compute in it, float16
  on older ones (float32 for Gemma 3), float32 on the CPU. It used to be bfloat16 everywhere,
  which older GPUs only emulate, slowly. `model.quantization` takes `"nf4"`, `"int8"`, `"hqq4"`,
  `"hqq3"`, `"hqq2"`, a dict or a transformers config, and replaces `model.use_4bit`, which is gone
  (runs saved with it still load, as `"nf4"`); HQQ layers compute in the model's dtype (transformers 4.57 forced
  float32). `ProbeWorkspace(model=..., tokenizer=...)` takes a model that is already loaded.
- **Model identity.** Probes record the identity of the model they were trained with (model,
  revision, dtype actually used, quantization, chat template and date) in `metrics.json`, and
  `get_probe` and `multi_probe_score_prompts` compare identities instead of config keys.
- **Automatic choices are announced** (printed, logged, and listed under `choices` in
  `metrics.json`), and **results are saved by default**; `concept_probe.configure(save=False)`
  turns off every write the caller did not ask for.
- **Batched reading.** Training, eval texts, `extract_text_reps` and `extract_completion_reps`
  read 16 inputs per forward pass (`readout.batch_size`), several times faster than one by one;
  in half precision the last digits change. Metrics record `read_batch_size`.
- **Representations cache.** Reading stores what it computes (one SQLite file per model under
  `<root_dir>/rep_cache`) and reuses it for the same model identity, token ids, span and readout.
- **Readouts.** `read_texts` / `read_encoded` return arrays with any `Readout`: max, first token,
  a position, a word (one per text if needed), every token as a row, and the embedding output;
  `assistant_max`, `assistant_first`, `sequence_max` and `assistant_tokens` are new readout names.
- **Message spans.** `preview_input` returns the token span of every message, and
  `message_spans(tokenizer, messages)` does for any conversation.
- **Estimators.** `training.estimator` takes `mean_diff` (default), `mean_diff_var`, `mean_diff_sd`,
  `mean_diff_drop_outliers`, `logistic`, `lda`, a dict with parameters, or a registered estimator.
  It replaces `training.direction_normalization`, which is gone (runs saved with it still load).
- **Steering has its own direction** (`steering.steer_direction`, default the raw difference of
  means): a probe trained with `mean_diff_var` reads with it and steers with the raw difference,
  which pushes further. Training announces it and saves both; `alpha_unit="sigma"` is measured on the
  steering direction. Before, the one direction did both, so steering with a `var` probe changes.
- **One probe object, which needs its model only to read texts, generate and steer.**
  `ConceptProbe` is built by training, from representations already read (`ConceptProbe.fit`,
  which chooses the layer on held-out items), from directions computed elsewhere
  (`ConceptProbe.from_directions`) or from a run (`ConceptProbe.load(run_dir, model=None)`), and
  `attach` gives it a model later. It always records which model it belongs to. Its fields are
  renamed and simpler: `directions` (was `concept_vectors`), `layer` (was `best_layer`),
  `steer_directions`, `identity` (was `model_identity`), and `calibrations`, a list of
  `Calibration` objects that replaces `score_center`, `score_scale`, `token_center`, `token_scale`,
  `reference_context` and `transfer_references`; `sigma_best_layer` is only reported in
  `metrics.json`, and steering in "sigma" uses `steer_sigma`. `z`, `classify`, `project` and
  `calibrate` work on representations; every z score goes through the calibration's one formula.
- **Training** also reports a threshold fitted on the selection half of the eval items, where a
  concept's `neutral_texts` fall, and can train on every token as a sample
  (`readout.read_mode = "assistant_tokens"`; eval items are then scored by their mean).
- **Evaluation on named eval sets** (`metrics.json` `evaluation.sets`): each with d, p, a bootstrap
  interval, the AUROC with its interval and the d random directions give on the same items
  (`null_d95`, also drawn in `plots/sweep.png`); the primary set also has the accuracy with the
  calibrated threshold. In shared and opposed modes the primary set now reads the eval completions
  without the pole's system prompt (which inflated d); the old reading is the set
  `eval_with_system_prompt` (`evaluation.primary_set` picks it), and eval texts are the set `texts`
  (this replaces `transfer_eval`). `concept.eval_sets` adds more sets; `evaluation.eval_pairs` pairs
  the eval texts; `label_shuffle` reports directions refitted with shuffled training labels. The
  sweep plot draws the held-out curve the reported d comes from. `evaluate_reading`,
  `compare_scorers` (a paired difference between two scorers) and `cross_reading` (probes read on
  each other's sets) work on representations. `tensors.npz` holds the calibrations instead of
  `score_center` and the other reference arrays.
- **z in another context.** `score_prompts` and `multi_probe_score_prompts` say why `score_mean_z`
  is None (`z_missing`, and a warning), and `ConceptProbe.calibrate(...)` adds references for
  generated completions or for texts read another way.
- **NaN or inf hidden states stop the run** with `NonFiniteActivationsError`, which names the
  layers and the model's dtype; loading a Gemma 3 model in float16 warns that it overflows.
  Previously the non-finite layers gave NaN directions and scores, and the layer sweep skipped
  them without saying so.
- **Loading works offline**: with `HF_HUB_OFFLINE` set, a repo id loads from its snapshot in the
  local cache (transformers 4.57.3 failed to load a tokenizer by repo id offline). Models load
  with `dtype=` on transformers 4.56 and later, which deprecate `torch_dtype=`.
- **Steering follows the device of each block's output** instead of its parameters, so it works
  on models offloaded to the CPU, whose parameters sit on the `meta` device between passes.
- **`pip install concept-probe[quantization]` installs bitsandbytes on Windows too**; it was
  excluded there, although bitsandbytes supports Windows since 0.43.
- **Assistant-role training texts with no turn before them stop before anything runs**, with a
  message naming `training.train_text_prompt` and the system prompt settings.
- Documented that z = 0 is the midpoint between the poles, not "without the concept".
- `tests/test_reference_outputs.py` pins what the existing API returns and writes on a tiny
  deterministic model, so the redesign in [docs/DESIGN.md](docs/DESIGN.md) can show that it keeps
  the same results.
- **Steering touches only the probe's layer by default** (`steering.steer_layers = "probe"`).
  It used to steer a window of 5 layers around it with alpha split among them. With several
  layers, each adds its own probe direction (unchanged); records now say so and include
  `steer_alpha_per_layer`. Generation is greedy by default (`steer_sampling = false`); sampled
  runs are seeded from `random.seed` when no seed is passed, and records hold the settings under
  `generation`. Every entry point uses `alpha_unit="raw"` by default
  (`train_concept_from_json` used `"sigma"`).
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
  would ignore; `pos_system == neg_system` in contrastive modes; a quantization without its library;
  a different model in `get_probe`; a reused `run_name`. Most are checked before anything runs.
- **Fixes**: `multi_probe_score_prompts` scored every probe sharing a concept name with the last
  one's vectors (names must now be distinct); `ActivationPatch` layers outside the model,
  including negative indices, were skipped; `extract_text_reps(role="assistant")` with no system
  or prompt raised an `IndexError`.
- **Statistics**: eval analyses no longer pool correct vs incorrect across alphas (per-alpha
  stats plus a GEE clustered by item), accuracy uses Wilson intervals, and the multi-probe report
  groups its CV folds and standard errors by item, adjusts for alpha and shows one alpha at a time.
- **Statistics that respect the design**: generated eval items stay paired by question in the
  split, the bootstrap, the permutations and the p-values (independent splits biased the
  held-out d downward); the GEE effect size is per within-alpha SD and its SEs are corrected for
  few items; the report shows an alpha-only baseline next to its correctness model.
- **z references by context**: `score_texts` standardizes only with references measured on texts
  read the same way (the transfer check's in shared/opposed modes), instead of with the
  generated-completion references, which put raw texts on the wrong side of the midpoint.
- **More safeguards**: tokenizers without a BOS leave the first text token (the attention sink)
  out of pooling; the random control is one direction at every layer (independent directions
  pushed about half as far); a null system prompt means none (it became the text "None");
  questions carrying their own system message raise; repeated alphas get their own files; a
  reused batch directory raises unless `overwrite=True`; a steering probe outside `probes` is
  checked against the model; correctness labels are normalized (numpy bools counted as wrong);
  the eval analysis no longer crashes at 0% or 100% accuracy, or on set / numpy values.
- **No silent stand-ins**: an item with no tokens of its own is left out (it used to be pooled
  over the whole prompt, system prompt included); fewer than 10 eval items per pole warns, and
  there are 20 default `eval_questions` instead of 5; eval items are clustered by their prompt
  across batches; multi-probe scoring requires one model and agreeing generation settings
  instead of taking the first probe's; `run_multi_scored_eval` takes `marker`, and a `marker`
  with a custom evaluator raises; `score_text_tokens` adds only the BOS, like every raw read.
- **Clearer semantics**: `sequence_*` readouts stop at the end of the item (they included the
  EOS of generated completions); texts-mode probes evaluated on generations pool them with
  `read_mode`; a custom defaults file is merged over the packaged defaults; `config.json` holds the
  concept that was trained; `extract_completion_reps` adds `system` like every other path;
  multi-probe scoring requires `steer_probe` for non-zero alphas (it used the first probe) and
  logs its warnings; the default evaluator reads the first line after a marker, compares numbers
  as numbers and raises on null `expected`; evaluators cannot overwrite scoring fields; the
  report puts probes on a z axis when it can and its overall means include unlabelled rows;
  `n_train_pos`/`n_train_neg` count the items in the vector; segment files use the heatmap's
  scores; `aggregate_eval_batches` finds batches by their analysis folder; an explicit
  coherence env file wins over the environment; the example concepts set no system prompt.
- **Other defaults**: no quantization; coherence rating off, and no `.env` read
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
