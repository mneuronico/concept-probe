from concept_probe import train_concept_from_json


def main() -> None:
    train_concept_from_json("examples/concepts/bored_vs_interested.json")


if __name__ == "__main__":
    main()
