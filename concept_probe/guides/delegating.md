# Delegating reading to an agent (or anyone)

Some steps need a reader: rating texts on a question (is this coherent? how positive is it?) and
critiquing a dataset (what else differs between the poles?). The library can use a model behind an
API (`APIJudge`) or loaded locally (`LocalJudge`), or hand the task to an agent: you, a subagent you
start, or a person. The task is written to a folder with everything needed; nothing else is assumed.

## Rating texts: `AgentJudge`

```python
judge = AgentJudge("tasks")
judge.rate(texts, rubric)    # writes tasks/<rubric>-<id>/ and raises AnswersPending(task_dir)
```

The folder has:

- `TASK.md`: the question, the allowed answers and what they mean, and the exact format;
- `items.jsonl`: one JSON object per line, `{"id": <integer>, "text": "<the text>"}`.

The answer is `answers.jsonl` in the same folder: one line per text,
`{"id": <the text's id>, "answer": "<an allowed answer>"}`, every id once. Then the same call
(`judge.rate(texts, rubric)`, or `ev.add_measure(...)`) reads it, checks every id and every answer,
and reports what is missing or not allowed.

If you are an agent with subagents, you can split `items.jsonl` among them (by id ranges), give each
the `TASK.md` and its part, and merge their answers into the one `answers.jsonl`. Ask each to judge
every text on its own; do not show a subagent the probe's scores or which pole a text came from.

## Critiquing a dataset: `critique_task`

```python
folder = critique_task(pos, neg, "critique", concept="positive vs negative feeling", paired=True)
# an agent answers critique/TASK.md, writing critique/critique.json
critique = load_critique(folder)
audit_dataset(pos, neg, checks=critique.checks())
```

The folder has `TASK.md` (what to look for, and the JSON format), `sample.json` (a random sample of
each pole, or of the pairs, with ids like `pos:3`) and, once answered, `critique.json`.
`load_critique` checks the answer: the names, the poles, that the ids are in the sample and that
the regular expressions compile. A critique works best from a reader that did not write the texts.

## Good practice

- Give the reader only what the task file gives: no scores, no hypotheses to confirm.
- Prefer a reader from outside the models under study; a judge's labels are another model's
  opinion, not ground truth. Check a sample of its answers.
- Keep the task folders: they record what was asked and what was answered.
