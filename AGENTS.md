# Instructions for agents using concept-probe

concept-probe trains, evaluates and steers linear probes of concepts in open-weights language
models. If you are an agent (Claude Code or another) working with it, read the guides it ships
before acting; they are in the package, so they are there wherever it is installed:

```bash
python -c "import concept_probe; print(concept_probe.guide())"            # the list
python -c "import concept_probe; print(concept_probe.guide('workflow'))"  # one guide
```

- `workflow`: the order of work, and the traps that make a probe look better than it is.
- `datasets`: writing and checking the texts of the two poles (variety, pairs, no confounds, not
  formulaic), with the audit, the critique and the attacks.
- `steering`: evaluating steering against random controls, with coherence and independent measures.
- `delegating`: how a reading task (rating texts, critiquing a dataset) is written to a folder for an
  agent or a subagent to answer, and the exact format of the answer.

## Rules of thumb

- Report every number next to its control: d next to `null_d95`, a steering effect next to its
  random controls' 95th percentile, a target's reading next to its shuffled-target null.
- Choose layers, thresholds and doses on items you do not report on.
- Do not write the texts and judge them in the same context: delegate the critique, the ratings and
  the final attack set to a subagent that did not write them.
- Results are saved by default, with every automatic choice; keep the folders and cite them.
- The docstrings of the public functions are the reference; `README.md` has the overview and
  `docs/DESIGN.md` the reasons for the design.
