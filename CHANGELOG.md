# Changelog

## 0.2.0 (2026-11-06)

A redesign. The library is now a set of layers, each usable alone, instead of one configuration-driven
pipeline. Code written for 0.1.0 does not run unchanged, and runs saved by 0.1.0 do not load: fit
the probe again.

### The layers

- **In one call**: `ConceptProbe.from_texts(model, pos, neg, eval_pos, eval_neg)` reads texts and
  fits and evaluates a probe; `ConceptProbe.from_system_prompts(model, pos_system, neg_system,
  questions, eval_questions)` does it on what the model writes under one system prompt per pole.
  Both are the layers below in a row.
- **Loading**: `ModelBundle.load(model_id, dtype="auto", quantization=None, ...)` picks a precision
  your GPU computes in (and float32 for Gemma 3, which overflows in float16), takes quantization by
  name (`"nf4"`, `"int8"`, `"hqq4"`, `"hqq3"`, `"hqq2"`) or as a config, works offline from the
  cache, and wraps a model you already loaded (`ModelBundle.from_loaded`).
- **Reading**: `read_texts` and `read_encoded` return per-layer representations (`Reps`) in batches,
  with several readouts from one pass (`text_mean`, `text_last`, `text_tokens`, a word, a position,
  ...), each text's residual norms, a cache on disk, and the input exactly as `generate` builds it
  (`add_generation_prompt=True`).
  `preview_input` shows what the model sees. Non-finite activations stop with
  `NonFiniteActivationsError`, which names the layers.
- **Directions**: `fit_directions` with the difference of means (raw, variance- or SD-normalized,
  without outlier coordinates), logistic regression, LDA or an estimator you register; `Eraser`
  removes a confound first (LEACE or projection).
- **The probe**: `ConceptProbe.fit(pos, neg, eval_sets=EvalSet(...))` chooses the layer on part of the
  primary eval set and reports Cohen's d, its p, intervals, AUROC and accuracy on the rest, next to
  what random directions (`null_d95`) and shuffled labels give. It reads with its estimator's
  direction and steers with the raw difference of means by default. `calibrate`, `z` (in SDs or in
  the poles' scale), `classify`, `accuracy`, `scores`, `score_texts`, `steering`, `save` and `load`.
  `probe.steering(dose)` and `evaluate_steering` take doses in the same unit, residual-stream
  norms by default.
- **Interventions**: `Add`, `Ablate`, `Patch` (one donor per prompt), `Rotate`, `Edit` (your own
  function) and their sums, at any layers and positions, applied with `intervene`, `generate` or
  `answer_logprobs`.
- **Generating**: `generate` in batches, under an intervention, with `on_token` to read probes while
  generating; `answer_logprobs` for the probability of given answers.
- **Evaluating steering**: `evaluate_steering` sweeps doses (in residual-stream norms by default)
  against random directions of the same length (ten by default), measures coherence (perplexity,
  repetition) and any measure you give (probes, a classifier, a judge), and reports the largest
  coherent dose.
- **Judges**: a `Rubric` answered by an `APIJudge` (OpenAI, Anthropic, Gemini, OpenRouter, Groq,
  Ollama or any OpenAI-compatible server), a `LocalJudge` or an `AgentJudge`.
- **Datasets**: `audit_dataset`, `compare_poles`, `surface_baseline`, `confound_check`,
  `learning_curve`, `red_team`, and critiques by an agent or a judge (`critique_task`,
  `critique_dataset`).
- **Beyond two poles**: `Target` (classes, a number, an order, a cycle, a layout), `read_target`,
  `target_shape` and `TargetProbe`.
- **Statistics**: `cohen_d`, `auroc`, `accuracy`, `compare_scorers`, `random_directions_null`,
  `summarize_tokens`.
- **Guides** for people and agents (`concept_probe.guide()`, `AGENTS.md`), and an example dataset
  to try the library with (`load_example("valence")`: 224 minimal pairs of positive and negative
  texts).

### Defaults that changed

- Texts are read raw (BOS and the text, no chat template) unless you give a role.
- The layer is chosen on held-out items, and the reported numbers come from the others.
- Answers the model wrote under a system prompt per pole are evaluated without that system prompt
  (read with it, the system prompt itself pushes the score toward its pole and inflates d).
- Steering touches only the probe's layer, and each listed layer gets the whole dose unless
  `distribute=True`.
- The chat template's date is fixed (`chat_template_date="2024-07-26"`), so the same text is the
  same input on any day.
- Steering doses are in residual-stream norms (0.1 adds a tenth of a typical token's norm at each
  steered layer); `unit="sd"` and `unit="raw"` are the others.
- Results are saved by default, in `concept_probe_outputs/` unless you give a folder, and every
  choice the library makes for you is announced, saved with the result, and has an option to
  choose otherwise; `concept_probe.configure(save=False)` turns saving off for the session.

### Removed

- The configuration-driven pipeline: `ProbeWorkspace`, `ConceptSpec`, `ConceptProbe.train`, the
  JSON runner, `defaults.json`. `ConceptProbe.from_texts` and `from_system_prompts` do it in one
  call, and the layers step by step;
  `examples/quickstart.py` and `examples/system_prompts.py` show the whole recipe.
- `score_prompts`, `multi_probe_score_prompts` and `score_text_tokens` (`generate`,
  `ConceptProbe.score_texts` and `evaluate_steering(probes=...)`); `extract_text_reps` and
  `extract_completion_reps` (`read_texts`, `read_encoded`); the behavioral eval runners and reports
  (`evaluate_steering` with `measures=`); the coherence helpers (`Rubric` and a judge);
  `ResidualStreamEditor`, `ActivationPatch` and the steering hooks (the interventions).

### Fixed

- Layers whose activations overflowed (Gemma 3 in float16) gave NaN directions and scores, and the
  layer search skipped them without saying so; now reading stops and says which layers.
- An item with no tokens of its own was pooled from other tokens; now it is left out, and said.
- Eval statistics treated minimal pairs as independent items; `EvalSet(..., paired=True)` keeps
  them paired.
- Loading a tokenizer by repo id failed offline (transformers 4.57); a repo id now loads from its
  snapshot in the cache.
- Steering took the device of a block's parameters, which fails on models offloaded to the CPU; it
  now follows the device of the block's output.
- `pip install concept-probe[quantization]` skipped bitsandbytes on Windows, which it supports.
- `generate` and `answer_logprobs` did not warn about a prompt that ended with another role than
  the user's; they do now, as reading does.

## 0.1.0

- `training.direction_normalization` (`"none"`, `"var"`, `"sd"`) for variance-normalized
  concept directions; `extract_text_reps`; `model.chat_template`.
- `apply_chat` normalizes `apply_chat_template` output to a `(1, seq_len)` LongTensor, whatever
  the transformers version returns.
- Steered generations are scored with the steering hooks removed. Previously the scoring
  forward pass ran with the hooks still active, so the injected vector "read itself back" and
  inflated probe scores under steering by roughly `alpha * cos(steer_dir, score_dir)` at the
  steered layers.
- `score_prompts(..., seed=...)` / `multi_probe_score_prompts(..., seed=...)` seed the RNGs for
  reproducible sampling.
