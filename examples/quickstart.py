from concept_probe import ConceptSpec, ProbeWorkspace


def main() -> None:
    concept = ConceptSpec(
        name="calm_vs_urgent",
        pos_label="calm",
        neg_label="urgent",
        pos_system=(
            "You are a calm assistant. Use measured phrasing, steady pacing, "
            "and low-arousal framing."
        ),
        neg_system=(
            "You are an urgent assistant. Use high-energy phrasing, fast pacing, "
            "and immediate-action framing."
        ),
        eval_pos_texts=[
            "Take a breath, slow down, and handle one step at a time.",
            "A steady plan is better than a rushed reaction.",
        ],
        eval_neg_texts=[
            "Move now and fix the details later.",
            "There is no time to wait; act immediately.",
        ],
    )

    workspace = ProbeWorkspace(model_id="meta-llama/Llama-3.2-3B-Instruct")
    probe = workspace.train_concept(concept)

    probe.score_prompts(
        prompts=["Help me decide how to respond to a stressful deadline."],
        alphas=[0.0, 4.0, -4.0],
        alpha_unit="sigma",
    )


if __name__ == "__main__":
    main()
