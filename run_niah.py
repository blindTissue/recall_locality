"""Experiment 3 (Needle-in-a-Haystack): re-mention rescue in ~4K-token natural prose.

A needle sentence and a later unrelated re-mention (the "anchor") are placed
in a Paul Graham essay haystack over a (depth x gap) grid (see
data/make_niah.py). Conditions:

  no_eviction              baseline (needle and anchor present)
  evict_needle             evict the needle; the anchor stays in cache
  evict_needle_and_anchor  evict the needle and the anchor
  evict_needle_no_anchor   context built without the anchor; evict the needle

Example:
    python run_niah.py --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import logging

from recall_locality.common import ResultWriter, load_json, model_dir
from recall_locality.eviction import find_token_positions
from recall_locality.inference import answer_token_ids, apply_chat_template, grade, load_model, prefill, run_condition

CONDITIONS = ["no_eviction", "evict_needle", "evict_needle_and_anchor", "evict_needle_no_anchor"]
FIELDS = [
    "model", "dataset_type", "needle_depth", "anchor_gap", "haystack_chunk", "pair_idx", "condition",
    "target_entity", "target_attribute", "correct_answer", "greedy_answer",
    "accuracy", "abstained", "correct_prob",
]

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--data", default="data/niah.json", help="NIAH JSON from data/make_niah.py")
    parser.add_argument("--out", default="results/niah", help="Output root")
    parser.add_argument("--tasks", nargs="+", default=["factual", "sentiment"], choices=["factual", "sentiment"])
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N samples")
    parser.add_argument("--chat_date", default="10 May 2026",
                        help="Date in the Llama-3 system prompt (default: day of the paper run)")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    dataset = [s for s in load_json(args.data) if s["dataset_type"] in args.tasks][:args.limit]
    writer = ResultWriter(f"{args.out}/{model_dir(args.model)}", FIELDS)

    for idx, sample in enumerate(dataset):
        task = sample["dataset_type"]
        answer_prefix = "The sentiment is" if task == "sentiment" else "The answer is"
        correct_ids = answer_token_ids(tokenizer, sample["answer"])

        with_anchor, suffix = apply_chat_template(sample["context_with_anchor"], tokenizer, args.chat_date)
        no_anchor, _ = apply_chat_template(sample["context_no_anchor"], tokenizer, args.chat_date)
        needle_pos = find_token_positions(tokenizer, with_anchor, sample["needle_text"])
        anchor_pos = find_token_positions(tokenizer, with_anchor, sample["anchor_text"])
        cache_with_anchor = prefill(model, tokenizer, with_anchor)
        cache_no_anchor = prefill(model, tokenizer, no_anchor)
        runs = {
            "no_eviction": (cache_with_anchor, []),
            "evict_needle": (cache_with_anchor, needle_pos),
            "evict_needle_and_anchor": (cache_with_anchor, sorted(set(needle_pos) | set(anchor_pos))),
            "evict_needle_no_anchor": (cache_no_anchor, find_token_positions(tokenizer, no_anchor, sample["needle_text"])),
        }

        for condition in CONDITIONS:
            cache, positions = runs[condition]
            result = run_condition(model, tokenizer, cache, positions,
                                   sample["question"] + suffix + answer_prefix, correct_ids, args.max_new_tokens)
            accuracy, abstained = grade(result["greedy_answer"], sample["answer"], task)
            writer.write({
                "model": args.model, "dataset_type": task,
                "needle_depth": sample["needle_depth"], "anchor_gap": sample["anchor_gap"],
                "haystack_chunk": sample["haystack_chunk_idx"], "pair_idx": sample["pair_idx"],
                "condition": condition,
                "target_entity": sample["target_entity"], "target_attribute": sample["target_attribute"],
                "correct_answer": sample["answer"], "greedy_answer": result["greedy_answer"],
                "accuracy": accuracy, "abstained": abstained,
                "correct_prob": f"{result['correct_prob']:.6f}",
            })
        if idx % 50 == 0:
            logger.info("[%d/%d] %s depth=%s gap=%s", idx, len(dataset), task,
                        sample["needle_depth"], sample["anchor_gap"])
    writer.close()


if __name__ == "__main__":
    main()
