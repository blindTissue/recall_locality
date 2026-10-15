"""Generate the Synthetic Associations datasets (factual and sentiment).

The exact files used in the paper are shipped as data/factual.json and
data/sentiment.json. They were drawn without a fixed seed, so this script
produces new samples from the same distribution rather than identical files.

    python data/make_synthetic.py --out_dir data/regenerated --seed 0
    python data/add_remention.py --data_dir data/regenerated

Each sample holds 20 (entity, attribute) pairs drawn without replacement and
randomly matched; one of the first 19 is the target. Every sample is rendered
in four formats: {structured (enumerated), natural (atomic)} x
{attribute-first, entity-first}.
"""

import argparse
import json
import random
from pathlib import Path

NUM_PAIRS = 20

# --- factual word pools ------------------------------------------------------

NAMES = [
    "Alice", "Boris", "Carol", "Derek", "Emma", "Felix", "Grace", "Henry", "Iris", "Julia",
    "Kate", "Leo", "Mona", "Noah", "Olivia", "Paul", "Quinn", "Rosa", "Sam", "Tina",
    "Ursula", "Victor", "Wendy", "Xavier", "Yara", "Zack", "Marcus", "Hugo", "Stella", "Vera",
    "Nadia", "Clark", "Greta", "Petra", "Oscar", "Lydia", "Ingrid", "Thea", "Bruno", "Elise",
    "Jorge", "Cleo", "Ewan", "Helga",
]
OCCUPATIONS = [
    "doctor", "lawyer", "teacher", "nurse", "chef", "pilot", "artist", "singer", "dancer", "writer",
    "farmer", "baker", "tailor", "mason", "plumber", "clerk", "judge", "guard", "miner", "welder",
    "barber", "brewer", "dentist", "florist", "surgeon", "butcher", "chemist", "janitor", "cashier", "merchant",
    "sculptor", "detective", "carpenter", "mechanic", "librarian", "botanist", "physicist", "surveyor",
    "novelist", "drummer",
]
OBJECTS = [
    "ring", "lamp", "throne", "vase", "clock", "mirror", "bowl", "shield", "crown", "chalice",
    "goblet", "pendant", "dagger", "helmet", "anchor", "compass", "lantern", "scepter", "medallion", "trumpet",
    "violin", "harp", "flute", "drum", "bell", "chest", "urn", "prism", "anvil", "comb",
    "fork", "spoon", "knife", "plate", "bucket", "funnel", "wrench", "locket", "bridle", "pestle",
]
MATERIALS = [
    "golden", "silver", "wooden", "copper", "marble", "crystal", "ceramic", "iron", "steel", "bronze",
    "granite", "leather", "silken", "velvet", "rubber", "bamboo", "plastic", "titanium", "ivory", "pewter",
    "chrome", "nickel", "cobalt", "onyx", "jade", "coral", "amber", "quartz", "basalt", "obsidian",
    "porcelain", "mahogany", "walnut", "cedar", "ebony", "maple", "teak", "birch", "wicker", "linen",
]
COLORS = [
    "crimson", "navy", "teal", "olive", "maroon", "indigo", "gold", "coral", "turquoise", "violet",
    "salmon", "orchid", "tan", "slategray", "plum", "khaki", "chocolate", "peru", "lavender", "tomato",
    "seagreen", "darkcyan", "steelblue", "firebrick", "mediumvioletred", "rosybrown", "cadetblue", "chartreuse",
    "dimgray", "cornflowerblue", "forestgreen", "sienna", "deeppink", "midnightblue", "royalblue", "hotpink",
    "burlywood", "palevioletred", "dodgerblue", "ivory",
]
APPAREL = [
    "t-shirt", "polo", "dress shirt", "henley", "tank top", "sweater", "hoodie", "cardigan", "blazer",
    "suit jacket", "bomber jacket", "denim jacket", "leather jacket", "overcoat", "parka", "windbreaker",
    "peacoat", "raincoat", "waistcoat", "jeans", "chinos", "cargo pants", "corduroy pants", "shorts",
    "joggers", "tracksuit", "tie", "bow tie", "belt", "scarf", "beanie", "cap", "gloves", "socks", "boots",
    "oxfords", "loafers", "sandals", "slippers", "watch", "undershirt", "boxers", "swim trunks",
]

FACTUAL_DOMAINS = [
    dict(name="person_occupation", entities=NAMES, attributes=OCCUPATIONS,
         entity_key="name", attribute_key="occupation", location="office",
         structured_attr_first="{attribute} {entity}", structured_entity_first="{entity} the {attribute}",
         natural_attr_first="The {attribute} is {entity}.", natural_entity_first="{entity} is a {attribute}.",
         question_by_entity="What is {entity}'s occupation?", question_by_attribute="Who is the {attribute}?"),
    dict(name="object_material", entities=OBJECTS, attributes=MATERIALS,
         entity_key="object", attribute_key="material", location="workshop",
         structured_attr_first="{attribute} {entity}", structured_entity_first="{entity} that is {attribute}",
         natural_attr_first="The {attribute} one is a {entity}.", natural_entity_first="The {entity} is {attribute}.",
         question_by_entity="What material is the {entity}?", question_by_attribute="What object is {attribute}?"),
    dict(name="color_apparel", entities=APPAREL, attributes=COLORS,
         entity_key="apparel", attribute_key="color", location="closet",
         structured_attr_first="{attribute} {entity}", structured_entity_first="{entity} of color {attribute}",
         natural_attr_first="The {attribute} one is a {entity}.", natural_entity_first="The {entity} is {attribute}.",
         question_by_entity="What is the color of the {entity}?",
         question_by_attribute="What is the apparel of color {attribute}?"),
]

# --- sentiment word pools ----------------------------------------------------

POSITIVE_ADJECTIVES = [
    "delightful", "wonderful", "magnificent", "brilliant", "exquisite", "superb", "splendid", "marvelous",
    "elegant", "gorgeous", "charming", "lovely", "pristine", "flawless", "stunning", "graceful", "radiant",
    "dazzling", "enchanting", "immaculate",
]
NEGATIVE_ADJECTIVES = [
    "terrible", "hideous", "appalling", "ghastly", "horrendous", "disgusting", "repulsive", "atrocious",
    "revolting", "abominable", "grotesque", "shabby", "decrepit", "awful", "tattered", "bleak", "dingy",
    "foul", "grim", "harsh",
]
# Sentiment-neutral everyday objects and places.
NOUNS = [
    "sofa", "desk", "shelf", "stool", "bench", "dresser", "cabinet", "counter", "table", "chair",
    "plate", "bowl", "mug", "fork", "spoon", "kettle", "bucket", "tray", "basket", "crate",
    "cafeteria", "warehouse", "lobby", "hallway", "garage", "terminal", "hangar", "pantry", "booth", "shed",
    "sidewalk", "crosswalk", "driveway", "mailbox", "faucet", "dumpster", "railing", "gutter", "awning", "vent",
    "clipboard", "folder", "planner", "marker", "eraser", "envelope", "notepad", "lanyard", "podium", "locker",
    "doorknob", "zipper", "coaster", "funnel", "anchor", "hinge", "lever", "bracket", "wedge", "pebble",
]
SCENE_LOCATIONS = ["story", "scene", "account", "review", "description"]
SENTIMENT_TEMPLATES = {
    "structured_adj_first": "{attribute} {entity}",
    "structured_entity_first": "{entity} that was {attribute}",
    "natural_adj_first": "The {attribute} one was the {entity}.",
    "natural_entity_first": "The {entity} was {attribute}.",
}


def _join(items: list[str]) -> str:
    return ", ".join(items[:-1]) + ", and " + items[-1]


def render(pairs: list[tuple[str, str]], template: str, structured: bool, location: str) -> str:
    rendered = [template.format(entity=e, attribute=a) for e, a in pairs]
    return f"In the {location}, there are {_join(rendered)}." if structured else " ".join(rendered)


def make_factual(samples_per_domain: int) -> list[dict]:
    dataset = []
    for d in FACTUAL_DOMAINS:
        for _ in range(samples_per_domain):
            entities = random.sample(d["entities"], NUM_PAIRS)
            attributes = random.sample(d["attributes"], NUM_PAIRS)
            pairs = [{d["entity_key"]: e, d["attribute_key"]: a} for e, a in zip(entities, attributes)]
            target_index = random.randint(0, NUM_PAIRS - 2)
            target = pairs[target_index]
            ea = list(zip(entities, attributes))
            sample = {"domain": d["name"]}
            for fmt in ["structured_attr_first", "structured_entity_first", "natural_attr_first", "natural_entity_first"]:
                sample[f"context_{fmt}"] = render(ea, d[fmt], fmt.startswith("structured"), d["location"])
            sample.update({
                "target_pair": target,
                "pair_list": pairs,
                "target_index": target_index,
                "total_pair_count": NUM_PAIRS,
                "question_by_entity": d["question_by_entity"].format(entity=target[d["entity_key"]]),
                "question_by_attribute": d["question_by_attribute"].format(attribute=target[d["attribute_key"]]),
                "answer_entity": target[d["entity_key"]],
                "answer_attribute": target[d["attribute_key"]],
            })
            dataset.append(sample)
    return dataset


def make_sentiment(num_samples: int) -> list[dict]:
    dataset = []
    for _ in range(num_samples):
        nouns = random.sample(NOUNS, NUM_PAIRS)
        n_pos = NUM_PAIRS // 2  # every list is balanced 10 positive / 10 negative
        labeled = ([(a, "positive") for a in random.sample(POSITIVE_ADJECTIVES, n_pos)]
                   + [(a, "negative") for a in random.sample(NEGATIVE_ADJECTIVES, NUM_PAIRS - n_pos)])
        random.shuffle(labeled)
        pairs = [{"noun": n, "adjective": a, "sentiment": s} for n, (a, s) in zip(nouns, labeled)]
        target_index = random.randint(0, NUM_PAIRS - 2)
        target = pairs[target_index]
        location = random.choice(SCENE_LOCATIONS)
        ea = [(p["noun"], p["adjective"]) for p in pairs]
        sample = {}
        for fmt, template in SENTIMENT_TEMPLATES.items():
            sample[f"context_{fmt}"] = render(ea, template, fmt.startswith("structured"), location)
        sample.update({
            "scene_location": location,
            "target_pair": target,
            "pair_list": pairs,
            "target_index": target_index,
            "total_pair_count": NUM_PAIRS,
            "question": (f"What is the sentiment of {target['noun']}? "
                         "Respond with exactly one word: positive or negative."),
            "answer": target["sentiment"],
        })
        dataset.append(sample)
    return dataset


def check_first_token_overlap(model_names: list[str]) -> None:
    """Warn if two words in a pool share a first token under any tokenizer."""
    from transformers import AutoTokenizer

    pools = {f"{d['name']}/entities": d["entities"] for d in FACTUAL_DOMAINS}
    pools |= {f"{d['name']}/attributes": d["attributes"] for d in FACTUAL_DOMAINS}
    pools |= {"nouns": NOUNS, "adjectives": POSITIVE_ADJECTIVES + NEGATIVE_ADJECTIVES}
    for model_name in model_names:
        tok = AutoTokenizer.from_pretrained(model_name)
        for pool_name, words in pools.items():
            seen = set()
            for w in words:
                firsts = {tok.encode(w, add_special_tokens=False)[0], tok.encode(" " + w, add_special_tokens=False)[0]}
                if firsts & seen:
                    print(f"[{model_name}] {pool_name}: first-token overlap at {w!r}")
                    break
                seen |= firsts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="data/regenerated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--factual_per_domain", type=int, default=300)
    parser.add_argument("--sentiment_samples", type=int, default=1000)
    parser.add_argument("--check_tokenizers", nargs="*", default=None,
                        help="Optional model ids whose tokenizers are checked for first-token overlap")
    args = parser.parse_args()

    if args.check_tokenizers:
        check_first_token_overlap(args.check_tokenizers)

    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("factual", make_factual(args.factual_per_domain)),
                       ("sentiment", make_sentiment(args.sentiment_samples))]:
        with open(out / f"{name}.json", "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(data)} samples to {out / f'{name}.json'}")


if __name__ == "__main__":
    main()
