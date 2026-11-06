# Working with concept-probe

This guide is for anyone using the library, people or agents. It gives the order of work and the
traps that make a probe look better than it is. Each step names the functions to use; their
docstrings have the details.

## The order of work

1. **Load the model once** (`ModelBundle.load`, or pass an already loaded model and tokenizer).
   `dtype="auto"` picks a safe precision and says which; some models (Gemma 3) overflow in float16.
2. **Write or audit the data before training** (see the `datasets` guide). A probe learns whatever
   separates the two sets of texts, including length, topic or a repeated phrasing.
3. **Read** the texts (`read_texts`): exactly as the probe will read them later (role, system
   prompt, which tokens). `preview_input` shows what the model sees.
4. **Fit the probe** (`ConceptProbe.fit` on representations, or `ProbeWorkspace.train_concept`
   from a spec). Choose the layer on items that did not train the directions.
5. **Evaluate on held-out items, against controls** (`evaluate_reading`): every d next to what
   random directions give on the same items (`null_d95`), and a refit with shuffled labels. Add
   eval sets from other sources (out of distribution) and read each probe on the others' sets
   (`cross_reading`).
6. **Check for confounds** (`confound_check`, `red_team`), and fit without them if needed
   (`Eraser`).
7. **Steer only after reading works**, and evaluate steering with controls and coherence
   (`evaluate_steering`; see the `steering` guide).
8. **Beyond two poles** (several classes, a number, an order, a cycle): `read_target` asks whether
   the target can be read, `target_shape` whether the classes' centroids have its shape. Report
   both: with few classes and many dimensions almost any arrangement can be read. `TargetProbe`
   keeps the map to score new activations (a place on a cycle, a value), read texts and steer;
   moving along the shape (`steering`) and moving between classes (`toward`) need not have the
   same effect on the model, so test both against random steps of the same length.

## Traps

- **A d without its null means little.** How far apart two sets are on a random direction varies a
  lot between sets (in one study, from 0.4 on paired texts to 3 on texts sampled from one request).
  Always report `null_d95` next to d, and AUROC next to both.
- **The system prompt inflates the reading.** Completions generated under a system prompt per
  pole carry that prompt's signature: read after it, the probe separates them by the prompt, not
  only by the text. The library reads them without it by default (`eval_with_system_prompt` is the
  other reading); keep it that way.
- **z only applies to items read like its calibration.** A probe calibrated on raw texts has no z
  for generated completions, and vice versa. The library says why a z is missing; calibrate on items
  read the right way (`probe.calibrate`) instead of borrowing another context's zero.
- **z = 0 is the midpoint between the poles, not "no concept".** Neutral text can sit far from 0.
  Give the concept `neutral_texts` to see where it lands.
- **Choosing the layer on the items you report overstates the effect.** The library splits the
  eval items by default; do the same in your own analyses.
- **Steering vectors read themselves back.** Scores of steered text are read again without the
  intervention; a reading taken while steering (`on_token`) includes the added vector.
- **One random control is not a control.** Use several (`n_random=10`) and compare with their 95th
  percentile.
- **Text written by the model under study comes with shortcuts** (in one study, surface features
  alone told its positive and negative texts apart 80-100% of the time). Audit it like any other.

## Saving and reproducing

Every function that produces a result also saves it by default (`concept_probe.configure(save=False)`
turns it off), and every automatic choice (dtype, cache, steering direction, controls) is printed,
logged and saved with the result. Keep the run folders: they hold what the numbers came from.
