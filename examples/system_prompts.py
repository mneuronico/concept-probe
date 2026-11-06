"""A probe trained on what the model writes under a system prompt per pole.

    python examples/system_prompts.py examples/concepts/bored_vs_interested.json

The concept file has each pole's system prompt and label, questions to answer for training and for
evaluation, and eval texts. The model answers every question under each pole's system prompt; the
probe is fitted on those answers and evaluated on answers to other questions, read without the
system prompt they were written under (read with it, the system prompt itself pushes the score
toward its pole and inflates d), next to the same answers read with it and to the eval texts read
alone. Training on written texts (examples/quickstart.py) generalizes better to text the model did
not write; this is for concepts that are easier to ask for than to write.
"""
import json
import sys

import numpy as np

from concept_probe import ConceptProbe, ModelBundle


def main(path: str) -> None:
    spec = json.load(open(path, encoding="utf-8"))
    concept, prompts = spec["concept"], spec["prompts"]
    bundle = ModelBundle.load("meta-llama/Llama-3.2-3B-Instruct")

    probe = ConceptProbe.from_system_prompts(
        bundle, concept["pos_system"], concept["neg_system"], prompts["train_questions"], prompts["eval_questions"],
        eval_texts=(concept["eval_pos_texts"], concept["eval_neg_texts"]),
        name=concept["name"], pos_label=concept["pos_label"], neg_label=concept["neg_label"])
    for name, result in probe.evaluation.sets.items():
        print(f"{name}: d {result.d:.2f} (random directions {result.null_d95:.2f})")
    print("cosine of the reading and steering directions:",
          float(np.dot(probe.directions[probe.layer], probe.steer_directions[probe.layer])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/concepts/bored_vs_interested.json")
