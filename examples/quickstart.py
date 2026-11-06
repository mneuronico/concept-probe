"""A probe of valence from written texts, evaluated against controls, read on new texts and used to steer.

    python examples/quickstart.py

The texts are the example dataset that ships with the library (``load_example("valence")``: minimal
pairs of positive and negative texts). For a concept of your own, put your texts in their place: a
few dozen pairs per pole for training, others for evaluation (the ``datasets`` guide says how to
write them: ``concept_probe.guide("datasets")``).
"""
from concept_probe import ConceptProbe, ModelBundle, evaluate_steering, load_example, read_texts


def main() -> None:
    bundle = ModelBundle.load("meta-llama/Llama-3.2-3B-Instruct")  # dtype chosen for your GPU, and said
    model, tok = bundle.model, bundle.tokenizer
    data = load_example("valence")  # 141 training pairs and 83 eval pairs

    # The texts are read alone (no chat template), as the mean over each text's tokens, at every layer.
    # The layer is chosen on half of the eval pairs and the numbers come from the other half, next to
    # what random directions and shuffled labels give on the same items. (This is read_texts, an
    # EvalSet and ConceptProbe.fit in a row; the README shows them one by one.)
    probe = ConceptProbe.from_texts(bundle, data.train_pos, data.train_neg, data.eval_pos, data.eval_neg,
                                    paired=True, name="valence")
    m = probe.metrics
    print(f"layer {probe.layer}: d {m['d']:.2f} (random directions {m['null_d95']:.2f}), AUROC {m['auroc']:.2f}")

    print(probe.z(read_texts(model, tok, ["The storm passed and nobody was hurt."])))   # in within-pole SDs
    probe.score_texts(["The storm passed and nobody was hurt."])                        # every token, saved as HTML

    # Steering, against ten random directions of the same length, with coherence measured.
    prompts = ["Write a short story about a student getting exam results.",
               "Describe your last trip to the beach.", "Tell me about your weekend."]
    ev = evaluate_steering(probe, prompts, doses=[-0.25, 0.25])
    print(ev.row(0.25)["effects"][probe.name], ev.largest_coherent_dose)


if __name__ == "__main__":
    main()
