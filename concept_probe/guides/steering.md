# Evaluating steering

Adding a probe's direction to the residual stream (steering) can change what a model writes. Three
things can make a steering result mean less than it seems: any direction of the same length might
change the text as much; the text may simply break; and the measure of the effect may be the probe
itself. `evaluate_steering` handles the three.

## How

```python
ev = evaluate_steering(probe, prompts, doses=[-0.25, -0.1, 0.1, 0.25], unit="resid_norm",
                       n_random=10, measures={"positivity": my_classifier})
ev.row(0.25)["effects"]["positivity"]   # change from dose 0, its interval, the controls' p95
ev.largest_coherent_dose
```

- **Random controls.** At each dose, `n_random` (10) random directions of the same length at each
  layer. An effect counts when it goes beyond the 95th percentile of the controls' effects
  (`beyond_controls`).
- **Coherence.** Each completion's perplexity under the model without the intervention, its
  repetition and its length. The default rule (perplexity within half and twice that of dose 0,
  repetition up by at most 0.2) is a heuristic: read completions near the limit and adjust it
  (`CoherenceRule`) or pass your own. Report effects only at coherent doses.
- **Measures of effect independent of the probe.** The steering probe's own reading of the steered
  text is informative but not independent of the steering. Add at least one measure that is not
  the probe: a classifier, a judge (`judge.measure(rubric)`), another probe (`probes=`), or whether
  an answer is right (`items=` and `exact_match()`).

## Measures added later

`ev.add_measure(name, measure)` measures every completion after the sweep and recomputes the table.
With an `AgentJudge`, the first call writes the task and raises `AnswersPending`; once the answers
are written (see the `delegating` guide), the same call adds them. `SteeringEvaluation.load(folder)`
picks a saved evaluation up again.

## Relating measures

`ev.relate("probe_name", "correct")` asks whether the probe's score predicts being right, with the
dose as a covariate and each prompt as a cluster. The predictor is standardized within each dose,
because steering itself moves it.

## What to report

The dose-response plot (`dose_response.png`), the largest coherent dose on each side, each effect
at coherent doses next to its controls' 95th percentile, and some completions at the largest
coherent dose and just past it.
