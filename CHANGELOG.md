# Changelog

## Unreleased

- **One way to do each thing.** The configuration-driven pipeline is gone: `ProbeWorkspace`,
  `ConceptSpec`, `ConceptProbe.train`, `score_prompts`, the JSON runner (`train_concept_from_json`),
  `defaults.json`, `resolve_config` and `load_defaults`. So are the duplicates of today's functions:
  `extract_text_reps` and `extract_completion_reps` (`read_texts`), `score_text_tokens`
  (`read_texts` token by token and the probe), `multi_probe_score_prompts` (`evaluate_steering`
  with `probes=`), and `concept_direction`, `ConsoleLogger` and `message_spans` as top-level names.
  `fit(calibrate=...)` is `probe.calibrate(...)` after fitting, and `fit(select=...)` an eval set
  that chooses the layer (`eval=EvalSet(...)`, `selection_split=None`).
  Every number they gave is given by today's layers: the reference tests compute the old outputs
  with them and get the same numbers.
- **`ConceptProbe.fit(pos, neg, eval=...)`** evaluates the probe on named eval sets (the layer chosen
  on part of the primary set, controls, intervals, a label-shuffle control, neutral texts) and
  calibrates it on them; the numbers are `probe.metrics` and `probe.evaluation`, saved with the probe
  (`probe.json`, `probe.npz`, and the plots). Runs saved by earlier versions still load.
- **`ModelBundle.load(model_id, dtype="auto", quantization=None, ...)`** takes arguments instead of a
  configuration dict; `ModelBundle.from_loaded` sets up a tokenizer and says what it changed.
- **Readouts are called `text_*`** (`text_mean`, `text_last`, `text_last_k`, `text_first`,
  `text_max`, `text_tokens`) and `sequence_*` (`sequence_mean`, `sequence_last_k`, ...); the old
  names raise with the new one, and saved runs are translated.
- **What the workshop wrote by hand is in the library:** the items that choose the layer can be given
  (`selection_split=[indices]`); token scores are summarized per text (`probe.scores(reps,
  per_text="max")`, `Scores.per_text`, `summarize_tokens`); z in the poles' scale (`unit="poles"`: +1
  the positive mean); an accuracy with its interval (`probe.accuracy`); completions read without the
  system prompt they were generated under (`Generations.encodings(system=None)`, or `read_texts` with
  one prompt per text); `evaluate_steering` measures a probe always in raw scores and in SDs of its
  eval items; `cohen_d(x, y)` is public, next to `auroc` and `accuracy`.
- **Defaults follow the workshop:** `evaluate_steering` doses are in residual-stream norms by default
  (measured on the prompts, and said, when a probe lacks them), and steering every listed layer gets
  the whole dose unless `distribute=True`.
- **Nothing is decided without saying so:** sampling settings taken from the model are announced and
  saved, settings that do not apply warn (greedy decoding with a temperature, a raw text with a
  system prompt, a user turn with a prompt), a probe's z refuses items pooled another way than its
  calibration, a target probe finds its layer by name, a window clipped at the model's edges, dose
  0 added, directions turned around and outlier coordinates left out are announced, an API judge's
  unusable answer scores NaN instead of failing the call, finite layers can be read from a model
  whose later layers overflow, a reply with no turn before it is tried, and `fit` and
  `evaluate_steering` take `allow_model_mismatch`.

## The redesign, stages 0 to 7 (docs/DESIGN.md)

- **Reading several readouts at once, prompts as generation sees them, and one donor per prompt**
  (also found re-running the workshop): `read_texts` and `read_encoded` take a list of readouts and
  return `{name: Reps}` from the same forward passes; `read_texts(..., generation=True)` (and
  `encode_text`) encode prompts exactly as `generate` and `answer_logprobs` do, and
  `Readout(span="input")` reads the whole input, up to the token that predicts the reply;
  `preview_input(generation=True)` now gives the text's span too. `Patch` takes one donor per prompt
  (values with a leading axis of prompts, or one list of positions per prompt); `generate` and
  `answer_logprobs` tell interventions which prompt each row is (`Where.items`, and `items=` in
  `intervene` for a direct call).
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

## Earlier

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
