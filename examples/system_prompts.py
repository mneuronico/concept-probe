"""A probe trained on what the model writes under a system prompt per pole.

    python examples/system_prompts.py [bored_vs_interested | distracted_vs_focused | impulsive_vs_planning]

The concept is one of the library's examples: each pole's system prompt and label, questions to
answer for training and for evaluation, and eval texts written by hand. The model answers every
question under each pole's system prompt; the probe is fitted on those answers and evaluated on
answers to other questions, read without the system prompt they were written under (read with it,
the system prompt itself pushes the score toward its pole and inflates d), next to the same answers
read with it and to the eval texts read alone. Training on written texts (examples/quickstart.py)
generalizes better to text the model did not write; this is for concepts that are easier to ask for
than to write.
"""
import sys

import numpy as np

from concept_probe import ConceptProbe, ModelBundle, load_example


def main(name: str) -> None:
    concept = load_example(name)
    bundle = ModelBundle.load("meta-llama/Llama-3.2-3B-Instruct")

    probe = ConceptProbe.from_system_prompts(
        bundle, concept.extra["pos_system"], concept.extra["neg_system"], concept.extra["train_questions"],
        concept.extra["eval_questions"], eval_texts=(concept.pos, concept.neg),
        name=name, pos_label=concept.pos_label, neg_label=concept.neg_label)
    for set_name, result in probe.evaluation.sets.items():
        print(f"{set_name}: d {result.d:.2f} (random directions {result.null_d95:.2f})")
    print("cosine of the reading and steering directions:",
          float(np.dot(probe.directions[probe.layer], probe.steer_directions[probe.layer])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bored_vs_interested")
