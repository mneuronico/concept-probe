"""A probe of valence from written texts, evaluated against controls, read on new texts and used to steer.

    python examples/quickstart.py

Replace the texts with your own: a few dozen pairs per pole for training, others for evaluation (the
``datasets`` guide says how to write them: concept_probe.guide("datasets")).
"""
from concept_probe import ConceptProbe, EvalSet, ModelBundle, evaluate_steering, read_texts

TRAIN_POS = ["The exam results came back and she passed with honors.", "The doctor said the scan was clear.",
             "Our team won the final in the last minute.", "The new apartment is bright and quiet."]
TRAIN_NEG = ["The exam results came back and she failed again.", "The doctor said the scan showed a tumor.",
             "Our team lost the final in the last minute.", "The new apartment is dark and noisy."]
EVAL_POS = ["The flight landed early and smoothly.", "My sister is coming to visit this weekend.",
            "The project finally works.", "We found the lost dog safe at the park."]
EVAL_NEG = ["The flight was delayed and bumpy.", "My sister cancelled her visit this weekend.",
            "The project still does not work.", "We never found the lost dog."]


def main() -> None:
    bundle = ModelBundle.load("meta-llama/Llama-3.2-3B-Instruct")  # dtype chosen for your GPU, and said
    model, tok = bundle.model, bundle.tokenizer

    # Read the texts alone (no chat template), as the mean over each text's tokens, at every layer.
    pos, neg = read_texts(model, tok, TRAIN_POS, norms=True), read_texts(model, tok, TRAIN_NEG, norms=True)
    eval_set = EvalSet("eval", read_texts(model, tok, EVAL_POS), read_texts(model, tok, EVAL_NEG), paired=True)

    # The layer is chosen on half of the eval pairs and the numbers come from the other half, next to
    # what random directions and shuffled labels give on the same items.
    probe = ConceptProbe.fit(pos, neg, eval=eval_set, name="valence", model=bundle)
    m = probe.metrics
    print(f"layer {probe.layer}: d {m['d']:.2f} (random directions {m['null_d95']:.2f}), AUROC {m['auroc']:.2f}")

    print(probe.z(read_texts(model, tok, ["The storm passed and nobody was hurt."])))   # in within-pole SDs
    probe.score_texts(["The storm passed and nobody was hurt."])                        # every token, saved as HTML

    # Steering, against ten random directions of the same length, with coherence measured.
    ev = evaluate_steering(probe, ["Write a short story about a student getting exam results."],
                           doses=[-0.25, 0.25])
    print(ev.row(0.25)["effects"][probe.name], ev.largest_coherent_dose)


if __name__ == "__main__":
    main()
