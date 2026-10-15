"""Append an unrelated re-mention of the target entity to every context (Experiment 3).

Deterministic given the base files (seed 42). Running it on the shipped
data/factual.json and data/sentiment.json reproduces
data/factual_remention.json and data/sentiment_remention.json.

    python data/add_remention.py --data_dir data
"""

import argparse
import json
import random
from pathlib import Path

SEED = 42

HOBBIES = [
    "painting", "gardening", "hiking", "reading", "cooking", "swimming", "cycling", "photography", "chess",
    "pottery", "knitting", "archery", "fishing", "running", "sketching", "origami", "birdwatching",
    "journaling", "quilting", "fencing",
]
OBJECT_LOCATIONS = ["attic", "garden", "basement", "closet", "garage", "shelf", "cupboard", "studio", "warehouse", "porch"]
EVENTS = [
    "a wedding", "a picnic", "a concert", "a meeting", "a party",
    "a ceremony", "a festival", "a banquet", "an exhibition", "a parade",
]
SENTIMENT_LOCATIONS = ["hallway", "courtyard", "lobby", "patio", "balcony", "corridor", "alcove", "foyer", "terrace", "veranda"]

FACTUAL_FORMATS = ["structured_attr_first", "structured_entity_first", "natural_attr_first", "natural_entity_first"]
SENTIMENT_FORMATS = ["structured_adj_first", "structured_entity_first", "natural_adj_first", "natural_entity_first"]


def factual_anchor(sample: dict, rng: random.Random) -> str:
    target, domain = sample["target_pair"], sample["domain"]
    if domain == "person_occupation":
        return f"{target['name']} also enjoys {rng.choice(HOBBIES)}."
    if domain == "object_material":
        return f"The {target['object']} is kept in the {rng.choice(OBJECT_LOCATIONS)}."
    if domain == "color_apparel":
        return f"The {target['apparel']} was worn to {rng.choice(EVENTS)}."
    raise ValueError(domain)


def sentiment_anchor(sample: dict, rng: random.Random) -> str:
    return f"The {sample['target_pair']['noun']} is near the {rng.choice(SENTIMENT_LOCATIONS)}."


def add_anchor(in_path: Path, out_path: Path, make_anchor, formats: list[str]) -> None:
    rng = random.Random(SEED)
    with open(in_path) as f:
        dataset = json.load(f)
    out = []
    for sample in dataset:
        anchor = make_anchor(sample, rng)
        new = dict(sample, anchor_sentence=anchor)
        for fmt in formats:
            new[f"context_{fmt}_with_anchor"] = sample[f"context_{fmt}"].rstrip() + " " + anchor
        out.append(new)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(out)} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data", help="Directory with factual.json and sentiment.json")
    parser.add_argument("--out_dir", default=None, help="Defaults to --data_dir")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir or args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    add_anchor(data_dir / "factual.json", out_dir / "factual_remention.json", factual_anchor, FACTUAL_FORMATS)
    add_anchor(data_dir / "sentiment.json", out_dir / "sentiment_remention.json", sentiment_anchor, SENTIMENT_FORMATS)


if __name__ == "__main__":
    main()
