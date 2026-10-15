"""Experiment 3 (dense controlled contexts): can a later re-mention rescue an evicted pair?

Each context ends with an unrelated sentence that re-mentions the target
entity (the "anchor", e.g. "Alice also enjoys reading."). Conditions:

  no_eviction             baseline (anchor present)
  evict_pair              evict the target pair; the anchor stays in cache
  evict_pair_and_anchor   evict the target pair and the anchor sentence

Example:
    python run_remention.py --model meta-llama/Llama-3.2-1B-Instruct --task sentiment
"""

import argparse
import logging

from recall_locality.common import ResultWriter, load_json, model_dir
from recall_locality.eviction import find_token_positions
from recall_locality.formats import entity_attribute, formats_for, pair_template, render_pair
from recall_locality.inference import answer_token_ids, apply_chat_template, grade, load_model, prefill, run_condition
from run_synthetic import FIELDS, questions

CONDITIONS = ["no_eviction", "evict_pair", "evict_pair_and_anchor"]
DEFAULT_DATA = {"factual": "data/factual_remention.json", "sentiment": "data/sentiment_remention.json"}

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--task", required=True, choices=["factual", "sentiment"])
    parser.add_argument("--data", default=None, help="Dataset JSON (default: data/<task>_remention.json)")
    parser.add_argument("--out", default="results/remention", help="Output root")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N samples")
    parser.add_argument("--chat_date", default="22 Apr 2026",
                        help="Date in the Llama-3 system prompt (default: day of the paper run)")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    dataset = load_json(args.data or DEFAULT_DATA[args.task], args.limit)
    writer = ResultWriter(f"{args.out}/{model_dir(args.model)}/{args.task}", FIELDS)

    for idx, sample in enumerate(dataset):
        entity, attribute = entity_attribute(sample, args.task)
        for direction, question, answer, answer_prefix in questions(sample, args.task, entity, attribute):
            correct_ids = answer_token_ids(tokenizer, answer)
            for fmt in formats_for(args.task):
                context, suffix = apply_chat_template(sample[f"context_{fmt}_with_anchor"], tokenizer, args.chat_date)
                pair_text = render_pair(entity, attribute, pair_template(sample, args.task, fmt))
                pair_pos = find_token_positions(tokenizer, context, pair_text)
                anchor_pos = find_token_positions(tokenizer, context, sample["anchor_sentence"])
                evict = {
                    "no_eviction": [],
                    "evict_pair": pair_pos,
                    "evict_pair_and_anchor": sorted(set(pair_pos) | set(anchor_pos)),
                }
                base_cache = prefill(model, tokenizer, context)

                for condition in CONDITIONS:
                    result = run_condition(model, tokenizer, base_cache, evict[condition],
                                           question + suffix + answer_prefix, correct_ids, args.max_new_tokens)
                    accuracy, abstained = grade(result["greedy_answer"], answer, args.task)
                    writer.write({
                        "model": args.model, "domain": sample.get("domain", "sentiment"),
                        "context_format": fmt, "question_direction": direction,
                        "condition": condition, "sample_idx": idx,
                        "target_entity": entity, "target_attribute": attribute,
                        "correct_answer": answer, "greedy_answer": result["greedy_answer"],
                        "accuracy": accuracy, "abstained": abstained,
                        "correct_prob": f"{result['correct_prob']:.6f}",
                    })
        if idx % 50 == 0:
            logger.info("[%d/%d] %s", idx, len(dataset), args.task)
    writer.close()


if __name__ == "__main__":
    main()
