# concept-probe

`concept-probe` trains, evaluates and steers linear probes of concepts in open-weights language
models (Hugging Face causal LMs), with the controls that tell a real effect from what any direction
or any label would give.

It is built in layers, each usable alone: load a model, read texts into representations, fit
directions and a probe on them, evaluate how it reads, intervene in the model, generate and score,
and evaluate steering. Every number comes with its control, every automatic choice is announced, and
every result is saved by default.

[![Open the tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mneuronico/concept-probe/blob/main/examples/tutorial.ipynb)
**The tutorial** (`examples/tutorial.ipynb`) goes through everything step by step on a small model, on
Colab's free GPU in about an hour: a first probe, where to read, how many examples, who writes the
texts, surface against concept, steering and ablation, null controls, geometry beyond one direction,
baselines, and how to audit and attack a dataset.

## What it includes

- Loading any causal LM, with a precision chosen for your GPU and model (`dtype="auto"`), and
  quantization by name (`"nf4"`, `"int8"`, `"hqq4"`, ...), also offline from the cache
- Reading texts, prompts and completions at every layer, with several readouts in one pass (mean over
  the text, last token, a word, every token, ...), in batches, with a cache
- Directions from several estimators (difference of means, variance-normalized, logistic, LDA, your
  own), with a confound erased (LEACE) if you ask
- Probes that choose their layer on held-out items and report Cohen's d, AUROC and accuracy with
  intervals, next to random directions, shuffled labels and a permutation p over layers
- Calibrated scores: z per reading context, in SDs or in the poles' scale, per text or per token,
  thresholds, neutral texts
- Interventions (add, erase, patch, rotate, any edit), batched generation, answer probabilities
- Steering evaluation against random directions of the same length, with coherence and any measure
  (a classifier, a judge behind an API, a local model or an agent)
- Dataset audits and critiques, confound checks, learning curves, red teaming
- Probes toward any target (classes, numbers, orders, cycles) and the shape of representations
- Guides for people and agents (`concept_probe.guide()`, `AGENTS.md`)

## Installation

```bash
pip install git+https://github.com/mneuronico/concept-probe.git
```

or, from a clone, `pip install -e .`. The core needs `numpy`, `torch` and `transformers`. Extras:
`pip install -e .[stats]` (`scipy`, `statsmodels`, `scikit-learn`: p-values, the logistic and LDA
estimators, the audit's classifiers), `.[plots]` (`matplotlib`), `.[quantization]`
(`bitsandbytes`, for `"nf4"` and `"int8"`; HQQ needs `hqq`), `.[full]` for all of them. Judges behind
an API need no client package.

## Quickstart

```python
from concept_probe import ConceptProbe, ModelBundle, evaluate_steering, load_example, read_texts

bundle = ModelBundle.load("meta-llama/Llama-3.2-3B-Instruct")
data = load_example("valence")      # minimal pairs to try it with; your own lists of texts go here

probe = ConceptProbe.from_texts(bundle, data.train_pos, data.train_neg, data.eval_pos, data.eval_neg,
                                paired=True, name="valence")   # minimal pairs stay together
probe.metrics["d"], probe.metrics["null_d95"]       # d on the held-out half, and what random directions give
probe.z(read_texts(bundle.model, bundle.tokenizer, ["The storm passed and nobody was hurt."]))
probe.score_texts(["The storm passed and nobody was hurt."])        # every token, with an HTML heatmap

prompts = ["Write a short story about a student getting exam results.", "Tell me about your weekend."]
ev = evaluate_steering(probe, prompts, doses=[-0.25, 0.25])          # against 10 random directions
ev.row(0.25)["effects"]["valence"], ev.largest_coherent_dose
```

`load_example("valence")` is a dataset that ships with the library: 224 pairs of English texts, each
one situation told with a good and with a bad outcome (141 pairs to train, 83 to evaluate). Others
test a valence probe (paraphrases, other genres, Spanish, attacks, confounds, negation), give
baselines (neutral texts, topics) and other concepts (truth, days of the week, emotions, concepts
written as system prompts); `concept_probe.example_data.EXAMPLES` lists them. They are there to see
the library work; for your concept, write its texts (`concept_probe.guide("datasets")`).

`ConceptProbe.from_texts` is three steps in a row, which you can also take yourself (to read
another way, add eval sets, or look at what each step gives):

```python
from concept_probe import EvalSet

model, tok = bundle.model, bundle.tokenizer
pos, neg = read_texts(model, tok, data.train_pos), read_texts(model, tok, data.train_neg)  # (texts, layers, dim)
eval_set = EvalSet("eval", read_texts(model, tok, data.eval_pos), read_texts(model, tok, data.eval_neg), paired=True)
probe = ConceptProbe.fit(pos, neg, eval_sets=eval_set, name="valence", model=bundle)
```

Everything is saved by default, in `concept_probe_outputs/` under the folder you run from: the
probe with its plots, the steering evaluation, and a cache of what was read.
`concept_probe.configure(save=False)` turns that off for the session, and `output_root=` puts it
elsewhere.

The layer is chosen on half of the eval items and the numbers are measured on the other half. The
probe saves itself (`probe.out_dir`) with its plots; `examples/quickstart.py` runs this end to end,
and `examples/system_prompts.py` trains on what the model writes under a system prompt per pole.

**Before you trust a probe**, read the `workflow` guide (`print(concept_probe.guide("workflow"))`):
the order of work and the traps that make a probe look better than it is.

## Loading a model

```python
bundle = ModelBundle.load("google/gemma-3-4b-it", quantization="nf4")
bundle = ModelBundle.from_loaded(model, tokenizer)        # a model you loaded yourself
```

- `dtype="auto"` (the default) is bfloat16 on GPUs that compute in it (compute capability 8.0 and
  up), float16 on older ones, float32 for Gemma 3 there (it overflows in float16) and on the CPU.
  Any dtype can be given; Gemma 3 in float16 warns.
- `quantization` is None, a level (`"nf4"`, `"int8"`, `"hqq4"`, `"hqq3"`, `"hqq2"`), a dict with a
  `quant_method`, or a transformers config. NF4 computes in the model's dtype.
- With `HF_HUB_OFFLINE` set, a repo id loads from its cached snapshot.
- Templates that write today's date (Llama 3) get a fixed one (`chat_template_date="2024-07-26"`),
  so the same text is the same input any day; `chat_template` replaces the template.
- In float32 on a GPU, the library's forward passes leave out PyTorch's memory-efficient attention
  kernel, which read one text of a padded batch wrong (Gemma 3, torch 2.7).

Every choice made on your behalf is printed (once per session) and kept in `bundle.choices`. A
model's **identity** (its repo, dtype, quantization, template and date) goes with everything read
from it: a probe refuses another model unless `allow_model_mismatch=True`.

## What the model sees

```python
preview_input(tok, "The storm passed.")                         # BOS + the text, and the span read
preview_input(tok, "The storm passed.", role="user", system="You are helpful.")
preview_input(tok, "Sure, here it is.", role="assistant", prompt="Write a story.")
preview_input(tok, "Write a story.", add_generation_prompt=True)          # a prompt to generate from
```

A text is read **alone** by default (`role="raw"`: the BOS and the text, no template): the probe
learns the text, not a chat around it. `role="user"` reads it as a user turn, `role="assistant"` as
the model's reply after `prompt` (one, or one per text) and `system`. With `add_generation_prompt=True`, a text
is a prompt to generate from, exactly as `generate` and `answer_logprobs` feed it. Whatever a role
would leave out (a system prompt with `role="raw"`, a template the tokenizer lacks) is warned about.

## Reading representations

```python
reps = read_texts(model, tok, texts)                                 # mean over each text's tokens
reps.data, reps.layers, reps.context                                 # (texts, layers, dim), how it was read
read_texts(model, tok, texts, "text_last")                          # the text's last token
read_texts(model, tok, texts, ["text_mean", "text_last"])           # {name: reps}, from one pass
read_texts(model, tok, texts, Readout(substring="river"))           # the tokens of a word
read_texts(model, tok, texts, "text_tokens")                        # every token is a row (reps.items)
read_texts(model, tok, prompts, Readout("input", "last"), add_generation_prompt=True)  # the token that predicts the reply
read_texts(model, tok, texts, layers=["embeddings", 0, -1])
read_encoded(model, tok, generations.encodings())                    # completions, as generated
```

A `Readout` takes the text's own tokens (`span="text"`), the input up to the end of the text
(`"through_text"`: from the start of the input through the text, template tokens too) or the whole input (`"input"`), narrows them to the last
`last_k`, one `position` or a `substring`, and reduces them with `agg` (`"mean"`, `"max"`,
`"first"`, `"last"`, or `"tokens"`). The names `text_mean`, `text_last`, `text_last_k`,
`text_first`, `text_max`, `text_tokens` and their `through_text_*` versions are shorthands. A text
without the position or substring raises `EmptySpanError` naming it (or is skipped with
`skip_empty=True`). Activations that are not finite raise at the layers read, naming them.

Inputs go through the model 16 at a time; in half precision batching moves the last digits
(`batch_size=1` reproduces reading one at a time). What is read is cached on disk by model identity,
token ids, span and readout, so the same text is never read twice (`cache=False` turns it off;
`configure(save=False)` turns off every write).

## Fitting a probe

```python
probe = ConceptProbe.fit(pos, neg, eval_sets=[eval_set, other_set], estimator="mean_diff", model=bundle)
```

- **Directions**: one per layer, from `estimator`: `"mean_diff"` (the difference of means),
  `"mean_diff_var"` / `"mean_diff_sd"` (divided by the variance or SD of each coordinate),
  `"mean_diff_drop_outliers"`, `"logistic"`, `"lda"`, a dict with parameters, or your own
  (`register_estimator`). Each points toward the first pole (a turned direction is announced).
  `erase=Eraser.fit(reps, confound_labels)` fits without a confound.
- **Steering directions** are the raw difference of means by default, whatever reads
  (normalized directions read better and push less); `steer_estimator="read"` steers with the reading ones.
- **The layer** is chosen on a `selection_split` part of the primary eval set (half, at random, pairs
  kept together; or the indices you give), within `search_layers`, and its numbers come from the
  rest: Cohen's d with a bootstrap interval and its p, AUROC, accuracy at a calibrated threshold, and
  `null_d95`, what 50 random directions give on the same items. A permutation p corrects the largest
  d for the search over layers; `n_label_shuffles` refits with the labels shuffled. Every other set is
  measured at that layer. Without `eval`, give `layer=`.
- **Calibrations**: each eval set's scores give the zero and unit of z for items read like it (the
  primary one with a threshold; `zero="baseline"` puts the zero at the second group's mean, for a
  concept read against a baseline). `neutral=` texts are placed on it, not used as the zero.
- The probe belongs to the model the representations came from (`read_texts` records it; else pass
  `model=` or `identity=`). It saves to `<output_root>/<name>/<time>` by default (`probe.json`,
  `probe.npz`, `plots/`), and `ConceptProbe.load(folder, model, tokenizer)` reads it back.
  `ConceptProbe.from_directions(directions, layer)` makes one from directions computed elsewhere.

`probe.metrics` has the numbers (layer, d, p, intervals, AUROC, accuracy, nulls, every set, the
label shuffles, the neutral texts); `probe.evaluation` the full evaluation, with per-layer curves;
`probe.choices` what was decided on your behalf.

## Reading with a probe

```python
probe.scores(reps)                        # raw projections at the probe's layer (the offset is arbitrary)
probe.z(reps)                             # in within-pole SDs; 0 is the midpoint between the poles
probe.z(reps, unit="poles")               # +1 the positive pole's mean, -1 the negative's
probe.z(token_reps, per_text="max")       # each text's tokens summarized: mean, max, min, median, p90, extreme
probe.classify(reps)                      # the calibrated threshold, and the probability
probe.accuracy(pos_reps, neg_reps)        # with a bootstrap interval
probe.calibrate(pos_texts, neg_texts, role="assistant", prompt=prompts)  # another reading: replies
```

**z = 0 is the midpoint between the poles, not "no concept"**: neutral texts can sit far from it
(`neutral=` shows where). A z needs a calibration measured on items read the same way: a probe
calibrated on texts read alone has no z for completions read as replies, and `z` refuses items
pooled another way than its calibration's items. `probe.score_texts(texts)` scores every token of
each text (raw, and in z where there is a calibration), saving an HTML heatmap; `Scores.per_text`
summarizes each text's tokens.

## Evaluating how it reads

`evaluate_reading(directions, [EvalSet(...), ...])` is what `fit` runs: the layer on part of the
primary set, every set measured there, with nulls and intervals. Around it:

- `cross_reading(probes, sets)`: each probe on each other's sets, with nulls and the cosines between
  them (two nearly orthogonal probes can still read each other's concept).
- `compare_scorers(stat, a_pos, a_neg, b_pos, b_neg)`: how much better one scorer does than another on
  the same items, by `stat` (`auroc`, `cohen_d`...), with a paired bootstrap.
- `cohen_d(x, y)`, `auroc(pos, neg)`, `accuracy(pos, neg, threshold)`: the same numbers for any two
  groups of scores (a per-text summary of token scores, say).
- `label_shuffle_control`, `random_directions_null`, `learning_curve(pos, neg, eval_sets, sizes)`
  (the eval items split once; the null drawn the same way), `confound_check`, `red_team`.

## Training on what the model writes

A concept can be easier to ask for than to write: generate each pole's texts under its system prompt
and fit on them. In one call:

```python
probe = ConceptProbe.from_system_prompts(bundle, pos_system, neg_system, questions, eval_questions,
                                         eval_texts=(eval_pos_texts, eval_neg_texts))
```

or step by step:

```python
pos_gens = generate(model, tok, questions, system=pos_system)
neg_gens = generate(model, tok, questions, system=neg_system)
pos = read_encoded(model, tok, pos_gens.encodings())
neg = read_encoded(model, tok, neg_gens.encodings())
```

Evaluate on answers to other questions **read without the system prompt they were written under**
(`gens.encodings(system=None)`): read with it, the system prompt itself pushes the score
toward its pole. In one study the same answers gave d 8.5 read with their system prompt and 4.9
without. `from_system_prompts` evaluates that way, with the sets paired by question, and also reports
the answers read with their system prompt; `examples/system_prompts.py` runs it.
Probes trained on written texts read text the model did not write better.

## Interventions

```python
from concept_probe import Add, Ablate, Patch, Rotate, Edit, intervene

Add({12: v}, alpha=4.0)                       # steer: add 4 v at layer 12 (a dict, or an array per layer)
Ablate(directions, layers=[10, 11, 12])       # project the direction out (several rows: a subspace)
Ablate({12: v}, value=0.5)                    # set the stream's coordinate along v to 0.5
Patch(12, positions=[-1], values=donor)       # overwrite the last prompt token with a donor's activation
Patch(12, [-1], donors)                       # donors (prompts, 1, dim): each prompt its own donor
Rotate(u, w, angle=np.pi / 4, layers=[12])    # rotate in the plane of u and w
Edit(my_function, layers=[12])                # anything: my_function(layer, hidden, where) -> hidden
Add({12: v}, 4.0) + Ablate({20: u})           # several, in order

with intervene(model, Ablate(probe.directions, layers=range(16))):
    reps = read_texts(model, tok, texts)       # read with the direction removed
```

Layer `l` is the output of block `l`. Positions count each row's own tokens from 0, without padding;
negative ones count back from the end of the prompt, and `"prompt"` / `"generated"` select those
parts. `probe.steering(dose, unit, layers=..., window=...)` builds the `Add` a probe steers with:
`unit` is `"resid_norm"` (each layer's typical residual norm; the default here and in
`evaluate_steering`), `"sd"` (the eval items' SD on the steering direction) or `"raw"`; each layer steered gets the whole dose unless `distribute=True`.
`generate`, `answer_logprobs` and `evaluate_steering` take any intervention and say which prompt
each row of a batch is, so a `Patch` can carry one donor per prompt. Reading under an intervention
reports the stream after the edit.

## Generating and answer probabilities

```python
gens = generate(model, tok, prompts, intervention=probe.steering(0.25, "resid_norm"))
gens.texts, gens[0].token_ids, gens.encodings()
answer_logprobs(model, tok, prompts, ["Positive", "Negative"], intervention=Ablate(...)).log_odds()
```

`generate` works in batches padded on the left (greedy by default; when sampling, settings left
unset take the model's own and that is announced). `on_token=` gets each step's token and the
scores of the probes in `probes=`; `keep_logits=` keeps the logits. `answer_logprobs` gives
log P(answer | prompt) for given answers, with or without an intervention (write each answer as the
model would start it; `answer_tokens` shows the tokens).

## Evaluating steering

```python
ev = evaluate_steering(probe, prompts, doses=[-0.25, -0.1, 0.1, 0.25],
                       probes=[other_probe], measures={"positivity": my_classifier})
ev.row(0.25)["effects"]["positivity"]   # change from dose 0, its interval, the controls' p95
ev.largest_coherent_dose                # {"positive": 0.1, "negative": -0.25}, say
ev.relate("valence", "positivity")      # does the probe's score predict the outcome? (GEE)
```

- **Doses** are in residual-stream norms of each layer by default (a probe without those norms gets
  them measured on the prompts, and that is said), or `unit="sd"` / `"raw"`; an `Add` (given at
  dose 1) or a function `dose -> Add` also steer. Dose 0 is the baseline, added when missing.
- **Random controls**: at each dose, `n_random` (10) random directions of the same length, each the
  same at every dose; every measure reports the steering's mean change next to the 95th percentile
  of theirs.
- **Measures**: each probe's score of the completions, read without the intervention (raw, and in
  SDs of its own eval items), and any function of the texts (a classifier, a judge's
  `as_measure(rubric)`, `exact_match` for answers with `items=`), each with a bootstrap interval over
  prompts. `ev.add_measure` adds one after the sweep.
- **Coherence**: perplexity under the model without the intervention, repetition and length.
  `CoherenceRule` (perplexity within half and twice that of dose 0, repetition up by at most 0.2) is
  a heuristic that flags doses and gives the largest coherent one; it filters nothing, and
  `coherence_rule=` takes your own rule.

It saves the completions, `steering.json` and `dose_response.png`; `SteeringEvaluation.load`
reads them back.

## Judges

A judge rates texts on any dimension you describe: a `Rubric` says what to ask and which answers
count, each with a number (`COHERENCE` is ready made).

```python
positivity = Rubric("positivity", "How positive is the feeling this text expresses?",
                    {str(k): float(k) for k in range(1, 6)}, guide="1 very negative, 3 neutral, 5 very positive.")
judge = APIJudge("anthropic", model="<a model id>")      # or "openai", "gemini", "openrouter", "groq", "ollama"
judge = LocalJudge(judge_model, judge_tokenizer)         # a model loaded here: answer probabilities
judge = AgentJudge("tasks")                              # an agent or a person, through files
judge.score(texts, positivity), judge.as_measure(COHERENCE)
```

Behind an API, several texts go per request at temperature 0; the key comes from `api_key`, a file
(`env_path`) or the environment and is never saved, and an answer that is not allowed scores NaN
(warned). A local judge reads each answer's probability, with no sampling. `AgentJudge` writes the
task to a folder and reads the answers back, checked (see the `delegating` guide).

## Datasets, confounds and beyond two poles

**Auditing a dataset** (no model): `audit_dataset(pos_texts, neg_texts, paired=..., checks=...)`
compares the poles' length, lists the words much more frequent in one pole, flags repeated openings
and near duplicates, measures variety, and trains classifiers that see only words, only length or
only your checks; nothing about a particular concept is checked unless you pass it
(`checks={"negation": count_matches(r"\b(not|no|never)\b|n't\b")}`). `surface_baseline` is the
number a probe has to beat. **Critiquing**: `critique_task(pos, neg, folder)` writes a task for an
agent (`load_critique` reads it), `critique_dataset(pos, neg, judge)` asks a judge. The `datasets`
guide says how to write the texts.

**Confounds**: `confound_check(directions, layer, {"negation": (with, without)})`, and to fit
without one, `Eraser.fit(reps, labels)` (LEACE, or `"project"`) passed as `erase=`.

**Beyond two poles.** A `Target` says what each item is: `Target.categories(labels)`,
`Target.number(values)`, `Target.order(labels, classes)`, `Target.cycle(labels, classes)` or
`Target.layout(labels, {class: coordinates})`. `read_target(reps, target, model=...)` asks whether it
can be read (centroid, linear, through the centroids' plane, or any model with `fit`/`predict`), next
to the target shuffled; `target_shape(reps, target)` whether the class centroids have the target's
shape, next to shuffled orders. With many dimensions and few classes almost any arrangement can be
read, so report both. `TargetProbe.fit(reps, target)` keeps the map: it scores new activations (a
place on a cycle, a value, the nearest class), reads texts whole or token by token, and moves
activations along the shape (`steering`) or between classes (`toward`).

**Several models.** Load each one, read, fit and evaluate, and compare at the same relative depth:

```python
for model_id in ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]:
    bundle = ModelBundle.load(model_id)
    probe = ConceptProbe.from_texts(bundle, train_pos, train_neg, test_pos, test_neg)
    results[model_id] = (probe.layer / (probe.num_layers - 1), probe.metrics["d"], probe.metrics["null_d95"])
    del bundle
```

## Saving, choices and the session

Every function that produces a result saves it by default (in `concept_probe_outputs/`, unless you
give a folder), and every automatic choice (dtype, cache,
steering direction, controls, sampling settings, ...) is printed once per session and kept with the
result. `concept_probe.configure(save=False, console=False, output_root="...")` changes that for the
session.

## Repo layout

```text
concept_probe/          the library
concept_probe/guides/   guides for people and agents (concept_probe.guide())
examples/               tutorial.ipynb (the Colab tutorial), quickstart.py, system_prompts.py
AGENTS.md               where agents start
CHANGELOG.md            what changed
```

## Development

```bash
python -m pytest -q
```

The tests run on a tiny deterministic model (`tests/tiny_model.py`); `tests/test_reference_outputs.py`
checks that today's layers give the numbers the library gave before its redesign.
