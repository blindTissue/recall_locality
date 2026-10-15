"""Build the Needle-in-a-Haystack re-mention grid (Experiment 3).

  * Haystack: 5 evenly spaced, non-overlapping 4,096-token windows of the
    Paul Graham essay corpus (``alessiodevoto/paul_graham_essays``), one per
    "haystack chunk".
  * 20 fixed factual and 20 fixed sentiment needle pairs (listed below; none
    of the names / nouns / adjectives occur in the five chunks).
  * Needle depth 10%..100% x re-mention gap {0 (adjacent), 5, 10, 20, 30}%,
    skipping cells with depth + gap > 100% -> 43 cells.
  * Insertions are snapped to the next sentence boundary.

Total: 2 tasks x 20 pairs x 43 cells x 5 chunks = 8,600 samples (~330 MB),
so the file is generated rather than shipped. Output is deterministic.

    python data/make_niah.py --out data/niah.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# Tokenizer used to cut windows and place insertions. The contexts are stored
# as decoded text, so every evaluated model re-tokenizes the same strings.
TOKENIZER = "meta-llama/Llama-3.2-3B-Instruct"
WINDOW_TOKENS = 4096
NUM_CHUNKS = 5
DEPTHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
GAPS = [0, 5, 10, 20, 30]

FACTUAL = {
    "needle": "{name} works as a {occupation} at the city hospital.",
    "anchor": "{name} grew up in a small town near the coast.",
    "question": "What is {name}'s occupation?",
    "pairs": [
        ("Alice", "doctor"), ("Boris", "pilot"), ("Carol", "engineer"), ("Derek", "chef"),
        ("Emma", "architect"), ("Felix", "accountant"), ("Grace", "pharmacist"), ("Henry", "veterinarian"),
        ("Iris", "journalist"), ("Julia", "photographer"), ("Kevin", "lawyer"), ("Linda", "florist"),
        ("Marco", "geologist"), ("Nora", "librarian"), ("Owen", "biologist"), ("Pamela", "editor"),
        ("Ravi", "nurse"), ("Sara", "plumber"), ("Tom", "electrician"), ("Uma", "carpenter"),
    ],
}
SENTIMENT = {
    "needle": "The {noun} was absolutely {adjective}.",
    "anchor": "The {noun} was scheduled to begin at 3 PM.",
    "question": "What is the sentiment of the {noun}? Respond with exactly one word: positive or negative.",
    "pairs": [
        ("concert", "magnificent", "positive"), ("workshop", "superb", "positive"),
        ("festival", "splendid", "positive"), ("seminar", "wonderful", "positive"),
        ("gala", "outstanding", "positive"), ("restaurant", "disappointing", "negative"),
        ("exhibit", "dreadful", "negative"), ("ceremony", "appalling", "negative"),
        ("luncheon", "terrible", "negative"), ("parade", "atrocious", "negative"),
        ("reception", "delightful", "positive"), ("banquet", "excellent", "positive"),
        ("retreat", "marvelous", "positive"), ("brunch", "lovely", "positive"),
        ("reunion", "fantastic", "positive"), ("picnic", "awful", "negative"),
        ("conference", "miserable", "negative"), ("fundraiser", "ghastly", "negative"),
        ("mixer", "abysmal", "negative"), ("recital", "horrendous", "negative"),
    ],
}

# .!? (optionally followed by a closing quote/bracket), whitespace, then a
# capital letter or opening quote/bracket.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=["\'(\[]?[A-Z])')


def haystack_chunks(tokenizer) -> list[tuple[list[int], str, list[tuple[int, int]]]]:
    """(token ids, text, offsets relative to text) for each haystack window."""
    raw = load_dataset("alessiodevoto/paul_graham_essays", split="test")[0]["context"]
    enc = tokenizer(raw, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    stride = (len(ids) - WINDOW_TOKENS) // (NUM_CHUNKS - 1)
    chunks = []
    for start in (i * stride for i in range(NUM_CHUNKS)):
        chunk_ids = ids[start:start + WINDOW_TOKENS]
        chunk_offsets = offsets[start:start + WINDOW_TOKENS]
        first, last = chunk_offsets[0][0], chunk_offsets[-1][1]
        chunks.append((chunk_ids, raw[first:last], [(s - first, e - first) for s, e in chunk_offsets]))
    return chunks


def sentence_starts(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    """Indices of the tokens that begin a new sentence."""
    starts, j = [], 0
    for char in sorted({m.end() for m in _SENTENCE_BOUNDARY.finditer(text)}):
        while j < len(offsets) and offsets[j][1] <= char:
            j += 1
        if j < len(offsets) and (not starts or starts[-1] != j):
            starts.append(j)
    return starts


def snap(target: int, starts: list[int], n_tokens: int) -> int:
    """First sentence start at or after ``target`` (end of haystack if none)."""
    return next((b for b in starts if b >= target), n_tokens)


def insert(tokenizer, haystack: list[int], starts: list[int], needle: str, anchor: str,
           depth: int, gap: int) -> tuple[list[int], list[int], float, float]:
    """Return (tokens with needle only, tokens with needle + anchor, actual needle %, actual anchor %)."""
    n = len(haystack)
    needle_idx = snap(int(n * depth / 100), starts, n)
    needle_pct = round(needle_idx / n * 100, 2)
    needle_ids = tokenizer.encode(f" {needle} ", add_special_tokens=False)
    with_needle = haystack[:needle_idx] + needle_ids + haystack[needle_idx:]

    if gap == 0:  # adjacent: needle and anchor inserted as one block
        block = tokenizer.encode(f" {needle} {anchor} ", add_special_tokens=False)
        return with_needle, haystack[:needle_idx] + block + haystack[needle_idx:], needle_pct, needle_pct

    anchor_idx = snap(int(n * (depth + gap) / 100), starts, n)
    if anchor_idx <= needle_idx:
        anchor_idx = next((b for b in starts if b > needle_idx), n)
    anchor_ids = tokenizer.encode(f" {anchor} ", add_special_tokens=False)
    at = anchor_idx + len(needle_ids)
    return with_needle, with_needle[:at] + anchor_ids + with_needle[at:], needle_pct, round(anchor_idx / n * 100, 2)


def build() -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    chunks = [(ids, sentence_starts(text, offs)) for ids, text, offs in haystack_chunks(tokenizer)]

    def needles():
        for i, (name, occupation) in enumerate(FACTUAL["pairs"]):
            yield ("factual", i, FACTUAL["needle"].format(name=name, occupation=occupation),
                   FACTUAL["anchor"].format(name=name), FACTUAL["question"].format(name=name),
                   occupation, name, occupation)
        for i, (noun, adjective, sentiment) in enumerate(SENTIMENT["pairs"]):
            yield ("sentiment", i, SENTIMENT["needle"].format(noun=noun, adjective=adjective),
                   SENTIMENT["anchor"].format(noun=noun), SENTIMENT["question"].format(noun=noun),
                   sentiment, noun, adjective)

    dataset = []
    for depth in DEPTHS:
        for gap in GAPS:
            if depth + gap > 100:
                continue
            for chunk_idx, (haystack, starts) in enumerate(chunks):
                # Order: all factual pairs, then all sentiment pairs, per chunk.
                for task, pair_idx, needle, anchor, question, answer, entity, attribute in needles():
                    no_anchor, with_anchor, needle_pct, anchor_pct = insert(
                        tokenizer, haystack, starts, needle, anchor, depth, gap)
                    dataset.append({
                        "dataset_type": task,
                        "needle_depth": depth,
                        "anchor_depth": depth + gap,
                        "actual_needle_depth_pct": needle_pct,
                        "actual_anchor_depth_pct": anchor_pct,
                        "anchor_gap": gap,
                        "needle_text": needle,
                        "anchor_text": anchor,
                        "context_with_anchor": tokenizer.decode(with_anchor, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                        "context_no_anchor": tokenizer.decode(no_anchor, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                        "question": question,
                        "answer": answer,
                        "target_entity": entity,
                        "target_attribute": attribute,
                        "pair_idx": pair_idx,
                        "haystack_chunk_idx": chunk_idx,
                    })
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/niah.json")
    args = parser.parse_args()

    dataset = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    cells = {(d["needle_depth"], d["anchor_gap"]) for d in dataset}
    print(f"Saved {len(dataset)} samples ({len(cells)} cells) to {out}")
    print("  per task:", dict(Counter(d["dataset_type"] for d in dataset)))


if __name__ == "__main__":
    main()
