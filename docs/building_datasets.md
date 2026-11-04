# Building concept datasets: red teaming and iteration

A probe finds whatever direction separates the two sets of texts it is trained on. If the poles
differ in anything besides the concept (length, negation, punctuation, register, topic, who wrote
them, a handful of repeated phrasings), the direction mixes the concept with that difference.
Held-out accuracy does not reveal it, because the eval split was written the same way and shares
the confound. This guide is the procedure we use to write labelled texts for `texts` mode and to
check them before trusting a probe trained on them.

## The loop

1. **Write with variety on purpose.** Minimal pairs (the same situation in both poles) hold topic,
   length and style fixed within a pair. Across pairs, vary everything that is not the concept:
   genre (chat message, review, diary, email, dialogue line, post, plain statement), register,
   length, grammatical person, tense, topic, and whether the concept is explicit (a keyword such
   as "happy") or implicit ("the biopsy came back clean"). Vary also *how* the concept flips
   within a pair. For valence: the outcome, an adjective, a verb, a quantity; negation sometimes
   in the positive item ("no one was hurt") and sometimes in the negative one ("she didn't
   recognize us"). Avoid templates, and keep a genre or a language out of training to use as an
   out-of-distribution test.

2. **Audit without a model.** For each pair, compare the two members on every surface feature you
   can count: tokens, punctuation, exclamations, negations, pronouns, temporal adverbs ("again",
   "finally", "still"), numbers, casing. Then fit classifiers that should fail, and check that
   they stay at chance (0.5 on pairs):
   - one that sees only surface features;
   - one that sees only function words.

   Read the words that separate the poles best (weighted log-odds with a prior): they should be
   concept words, not function words or a phrase you reused. Check for repeated openings and
   near-duplicate items.

3. **Reread the texts as an adversary.** Ask what else the poles differ in, on average. Some
   confounds are semantic and no counter finds them. In a valence set we wrote, the positives
   leaned toward beginnings ("for the first time") and completeness ("every", "the whole"). The
   negatives leaned toward endings, accumulations of events joined by "and", and the passive voice
   of things happening to someone. None of these is valence.

4. **Red-team the probe.** Train it as you will use it, then try to beat it with items written
   for each suspected confound:
   - items of each pole that carry the *other* pole's feature (a positive text full of negations,
     a negative one that says "finally");
   - pairs of neutral texts that differ *only* in the feature ("The first train leaves at 5:40" /
     "The last train leaves at 23:40").

   The question is: can you find an example that expresses the concept and the probe misses, or
   one that does not and the probe fires, only because of a confound you put in the dataset? For
   the neutral pairs, report the score difference in units of the gap between the poles: 0 means
   the feature does not move the probe, 1 means it moves it as much as the concept does.

5. **Attribute each failure.** It is the dataset's fault if the feature is imbalanced between the
   poles in training (step 2), or if balancing it removes the failure. If the feature is balanced
   and the probe still follows it, the association comes from the model's representation. Report
   that as a finding; it is not a bug in the data.

6. **Fix and repeat.** Balance or remove the feature, add training pairs that dissociate it from
   the concept, and run steps 2-5 again. Stop when:
   - the surface classifiers are at chance;
   - the separating words are concept words;
   - no attack beats the probe by more than its held-out error;
   - all of this holds in more than one model, preferably from different families, so the data
     is not tuned to one model's quirks.

7. **Keep a final red-team set out of the loop.** Write it after the last iteration and never use
   it to fix the data. The loop otherwise overfits to your own attacks, just as choosing a layer
   on the data you report on inflates the effect. Report its results next to the probe's.

Keep a log of each iteration: what changed in the data and which numbers moved. It is the
evidence that the probe measures the concept, and it makes a good figure.

## Things that came up on other datasets

- **The readout can create a confound.** On a true/false dataset, a probe on the mean over the
  statement's tokens followed how familiar its entities were ("Paris is in France" read truer
  than "Rennes is in France"). Whether a statement is true is only settled at its end; read at
  the last token, that shift disappeared. Try the readout you will use, and if a confound sits
  in some tokens of the text, try another readout before rewriting the data.
- **Easy contrasts inflate held-out separation.** False statements written to be obviously false
  ("Gold is a gas") gave twice the held-out d of plausible ones ("Picasso was an Italian
  painter"). The extra separation was absurdity, not truth.
- **Neutral-pair shifts are in units of the pole gap.** When a fix makes the poles harder to tell
  apart, the gap shrinks and the same raw shift looks bigger. Compare rounds on fixed external
  sets (out-of-distribution data, a final red-team set) rather than on the shift alone.
- **For factual concepts, check what the model knows.** Compare the log-probability of the two
  members of each pair; a pair whose false member the model finds likelier is noise to it. Do
  not filter on it blindly: likelihood is not truth (a negated true statement is often less
  likely than its false counterpart), and filtering keeps exactly the pairs where they agree.
- **Text written by the model under study comes with surface shortcuts.** Positive and negative
  texts generated by the probed model were separable from surface features alone (0.8-1.0),
  against 0.5 for hand-written, audited ones.
- **Sets that never train a probe still need checking.** Test sets, confound variants and neutral
  baselines fail differently: a "neutral" text that reads as bad news, a negation that does not
  flip the label ("not bad" is not positive), a confound variant that also shifts the concept.
  Check each item's label with a judge outside the models under study. Note that the judge has
  biases of its own (it may also react to a valence word inside a name) before treating its
  verdict as ground truth.

## Doing it with concept-probe

`extract_text_reps` reads texts exactly as `texts`-mode training does, and `concept_direction`
estimates the direction. A neutral-pair shift at layer `l` is then a few lines:

```python
import numpy as np
from concept_probe import concept_direction, extract_text_reps

pos = extract_text_reps(model, tokenizer, train_pos_texts)   # (n, layers, dim)
neg = extract_text_reps(model, tokenizer, train_neg_texts)
v = concept_direction(pos[:, l], neg[:, l])
mp, mn = pos[:, l] @ v, neg[:, l] @ v
half_gap = (mp.mean() - mn.mean()) / 2

a = extract_text_reps(model, tokenizer, ["The first train leaves at 5:40."])[:, l] @ v
b = extract_text_reps(model, tokenizer, ["The last train leaves at 23:40."])[:, l] @ v
shift = float((a - b)[0] / half_gap)   # 0: the feature does not move the probe
```

Choose `l` on eval data, never on the items you report (see
[Layer selection and its statistics](../README.md#layer-selection-and-its-statistics)).
