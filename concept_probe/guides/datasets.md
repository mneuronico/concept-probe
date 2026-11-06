# Building and checking a dataset of two poles

A probe finds whatever direction separates the two sets of texts it is trained on. If the poles
differ in anything besides the concept (length, negation, punctuation, register, topic, who wrote
them, a handful of repeated phrasings), the direction mixes the concept with that difference.
Held-out accuracy does not reveal it, because the eval split was written the same way and shares
the confound. This is the procedure for writing labelled texts and checking them before trusting a
probe trained on them. An agent can do every step, delegating the reading to subagents (see the
`delegating` guide).

## The loop

1. **Write with variety on purpose.** Minimal pairs (the same situation in both poles) hold topic,
   length and style fixed within a pair. Across pairs, vary everything that is not the concept:
   genre (chat message, review, diary, email, dialogue line, post, plain statement), register,
   length, grammatical person, tense, topic, and whether the concept is explicit (a keyword) or
   implicit (an event that implies it). Vary also *how* the concept flips within a pair: an outcome,
   an adjective, a verb, a quantity; put negation sometimes in one pole and sometimes in the other.
   Avoid templates and repeated openings, and keep a genre or a language out of training to use as
   an out-of-distribution test.

2. **Audit without a model** (`audit_dataset(pos, neg, paired=True, checks=...)`). It compares the
   poles' length, lists the words much more frequent in one pole, flags repeated openings and near
   duplicates, measures variety, and trains classifiers that see only words, only length or only your
   checks. Words should separate the poles (the concept is in the words); length and your checks
   should stay at chance (0.5 on pairs). The words that separate the poles best should be concept
   words, not function words or a phrase you reused. Add a check for every feature you suspect:
   `checks={"negation": count_matches(r"\b(not|no|never)\b|n't\b")}`.

3. **Have the texts read by someone looking for confounds** (`critique_task` for an agent or a
   subagent, `critique_dataset` for a model behind an API or loaded here). Some confounds are
   semantic and no counter finds them: in one valence set, the positives leaned toward beginnings
   ("for the first time") and completeness ("every", "the whole"), the negatives toward endings,
   accumulations of events joined by "and", and the passive voice. None of these is valence. The
   critique's differences that come with a regular expression become checks for step 2
   (`critique.checks()`), so each hypothesis is measured on every text.

4. **Attack the probe** (`red_team`). Train it as you will use it, then try to beat it with items
   written for each suspected confound:
   - items of each pole that carry the *other* pole's feature (a positive text full of negations,
     a negative one that says "finally");
   - pairs of neutral texts that differ *only* in the feature ("The first train leaves at 5:40" /
     "The last train leaves at 23:40").

   Can you find an example that expresses the concept and the probe misses, or one that does not
   and the probe fires, only because of a confound in the dataset? `red_team` reports each pole's
   accuracy and how far the feature alone moves the probe, in within-pole standard deviations
   (0: not at all). `confound_check` asks the same of representations of items with and without the
   feature.

5. **Attribute each failure.** It is the dataset's fault if the feature is imbalanced between the
   poles in training (step 2), or if balancing it removes the failure. If the feature is balanced
   and the probe still follows it, the association comes from the model's representation: report it
   as a finding, not a bug in the data. To train without a feature, `Eraser` removes it from the
   representations before fitting.

6. **Fix and repeat.** Balance or remove the feature, add training pairs that dissociate it from the
   concept, and run steps 2-5 again. Stop when:
   - length and your checks are at chance;
   - the separating words are concept words;
   - no attack beats the probe by more than its held-out error;
   - all of this holds in more than one model, preferably from different families.

7. **Keep a final attack set out of the loop.** Write it after the last iteration and never use it
   to fix the data: the loop otherwise fits your own attacks, as choosing a layer on the data you
   report on inflates the effect. Report its results next to the probe's.

Keep a log of each iteration: what changed in the data and which numbers moved. It is the evidence
that the probe measures the concept.

## Things that came up before

- **The readout can create a confound.** On a true/false dataset, a probe on the mean over the
  statement's tokens followed how familiar its entities were; read at the last token, where the
  statement is settled, that shift disappeared. Try the readout you will use.
- **Easy contrasts inflate held-out separation.** False statements written to be obviously false
  gave twice the held-out d of plausible ones: the extra separation was absurdity, not truth.
- **Compare rounds on fixed sets.** When a fix makes the poles harder to tell apart, shifts measured
  in units of the poles' spread look bigger; compare rounds on fixed external sets.
- **For factual concepts, check what the model knows** (`answer_logprobs`), without filtering on it
  blindly: likelihood is not truth.
- **Text written by the model under study comes with surface shortcuts.** Audit it like any other.
- **Sets that never train a probe still need checking.** Test sets, confound variants and neutral
  texts fail too (a "neutral" text that reads as bad news, a negation that does not flip the label).
  Have each item's label checked by a reader outside the models under study (a judge, an agent),
  knowing it has biases of its own.

## For agents writing the texts

When you are asked to write or fix a dataset with this library: write in batches, run
`audit_dataset` after each batch, read its `audit.md`, delegate a critique (`critique_task`) to a
subagent with no stake in the texts, turn the critique's differences into checks, and fix what the
numbers confirm. Do not write the attack set (step 7) in the same context that wrote the training
texts; a fresh subagent should.
