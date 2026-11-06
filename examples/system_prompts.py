"""A probe trained on what the model writes under a system prompt per pole.

    python examples/system_prompts.py examples/concepts/bored_vs_interested.json

The concept file has each pole's system prompt and label, questions to answer for training and for
evaluation, and eval texts. The model answers every question under each pole's system prompt; the
probe is fitted on those answers and evaluated on answers to other questions, read without the
system prompt they were written under (decision D2: read with it, the system prompt itself pushes
the score toward its pole and inflates d), next to the same answers read with it and to the eval
texts read alone. Training on written texts (examples/quickstart.py) generalizes better to text the
model did not write; this is for concepts that are easier to ask for than to write.
"""
import json
import sys

import numpy as np

from concept_probe import ConceptProbe, EvalSet, ModelBundle, ReadContext, generate, read_encoded, read_texts


def answers_set(name, model, tok, gens_pos, gens_neg, system):
    """Answers to the same questions as an eval set, paired by question (an empty answer drops its pair):
    read after the question alone (``system=None``) or as written (``"as generated"``)."""
    reads = [read_encoded(model, tok, g.encodings(system), skip_empty=True) for g in (gens_pos, gens_neg)]
    both = sorted(set(reads[0].read_items) & set(reads[1].read_items))
    pos, neg = (r.data[[r.read_items.index(q) for q in both]] for r in reads)
    return EvalSet(name, pos, neg, context=ReadContext.generated(system=None), paired=True)


def main(path: str) -> None:
    spec = json.load(open(path, encoding="utf-8"))
    concept, prompts = spec["concept"], spec["prompts"]
    bundle = ModelBundle.load("meta-llama/Llama-3.2-3B-Instruct")
    model, tok = bundle.model, bundle.tokenizer

    def answers(questions, system):
        return generate(model, tok, questions, system=system, max_new_tokens=64)

    train_pos = answers(prompts["train_questions"], concept["pos_system"])
    train_neg = answers(prompts["train_questions"], concept["neg_system"])
    pos = read_encoded(model, tok, train_pos.encodings(), norms=True)  # as written, after the pole's prompt
    neg = read_encoded(model, tok, train_neg.encodings(), norms=True)
    eval_pos = answers(prompts["eval_questions"], concept["pos_system"])
    eval_neg = answers(prompts["eval_questions"], concept["neg_system"])
    sets = [answers_set("eval", model, tok, eval_pos, eval_neg, None),
            answers_set("eval_with_system_prompt", model, tok, eval_pos, eval_neg, "as generated"),
            EvalSet("texts", read_texts(model, tok, concept["eval_pos_texts"]),
                    read_texts(model, tok, concept["eval_neg_texts"]))]
    probe = ConceptProbe.fit(pos, neg, eval=sets, name=concept["name"], pos_label=concept["pos_label"],
                             neg_label=concept["neg_label"], model=bundle)
    for name, result in probe.evaluation.sets.items():
        print(f"{name}: d {result.d:.2f} (random directions {result.null_d95:.2f})")
    print("cosine of the reading and steering directions:",
          float(np.dot(probe.directions[probe.layer], probe.steer_directions[probe.layer])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/concepts/bored_vs_interested.json")
