"""Experiments 1 and 2: full and partial eviction on Synthetic Associations.

For every sample, context format, and question direction, the context is
prefilled once and each condition evicts a different span from a copy of
that cache:

  no_eviction            baseline
  removed_from_context   control: context re-rendered without the target pair
  evict_both             Exp. 1: evict entity + attribute (the whole pair)
  evict_both_keep_punct  Appendix: as evict_both, keeping trailing punctuation
  evict_attribute        Exp. 2: evict only the attribute
  evict_entity           Exp. 2: evict only the entity

Example:
    python run_synthetic.py --model meta-llama/Llama-3.2-1B-Instruct --task factual
"""

import argparse
import logging

from recall_locality.common import ResultWriter, load_json, model_dir
from recall_locality.eviction import find_token_positions
from recall_locality.formats import (
    attribute_span, context_without_target, entity_attribute, formats_for,
    is_attr_first, pair_template, render_pair,
)
from recall_locality.inference import (
    answer_token_ids, apply_chat_template, generate_and_score, grade, load_model, prefill, run_condition,
)

CONDITIONS = [
    "no_eviction", "removed_from_context", "evict_both",
    "evict_both_keep_punct", "evict_attribute", "evict_entity",
]
FIELDS = [
    "model", "domain", "context_format", "question_direction", "condition", "sample_idx",
    "target_entity", "target_attribute", "correct_answer", "greedy_answer",
    "accuracy", "abstained", "correct_prob",
]
DEFAULT_DATA = {"factual": "data/factual.json", "sentiment": "data/sentiment.json"}

logger = logging.getLogger(__name__)


def eviction_positions(tokenizer, context: str, condition: str, pair_text: str,
                       entity: str, attribute: str, attr_first: bool) -> list[int]:
    if condition == "evict_both":
        return find_token_positions(tokenizer, context, pair_text)
    if condition == "evict_both_keep_punct":
        return find_token_positions(tokenizer, context, pair_text, include_trailing_punct=False)
    if condition == "evict_attribute":
        return find_token_positions(tokenizer, context, attribute_span(pair_text, entity, attribute, attr_first))
    if condition == "evict_entity":
        return find_token_positions(tokenizer, context, entity)
    return []


def questions(sample: dict, task: str, entity: str, attribute: str):
    """Yield (direction, question, correct answer, answer prefix)."""
    if task == "sentiment":
        yield "by_entity", sample["question"], sample["answer"], "The sentiment is"
    else:
        yield "by_entity", sample["question_by_entity"], attribute, "The answer is"
        yield "by_attribute", sample["question_by_attribute"], entity, "The answer is"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face model id")
    parser.add_argument("--task", required=True, choices=["factual", "sentiment"])
    parser.add_argument("--data", default=None, help="Dataset JSON (default: data/<task>.json)")
    parser.add_argument("--out", default="results/synthetic", help="Output root")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N samples")
    parser.add_argument("--chat_date", default="21 Apr 2026",
                        help="Date in the Llama-3 system prompt (default: day of the paper run)")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    dataset = load_json(args.data or DEFAULT_DATA[args.task], args.limit)
    writer = ResultWriter(f"{args.out}/{model_dir(args.model)}/{args.task}", FIELDS)

    for idx, sample in enumerate(dataset):
        entity, attribute = entity_attribute(sample, args.task)
        for direction, question, answer, answer_prefix in questions(sample, args.task, entity, attribute):
            correct_ids = answer_token_ids(tokenizer, answer)
            for fmt in formats_for(args.task):
                context, suffix = apply_chat_template(sample[f"context_{fmt}"], tokenizer, args.chat_date)
                full_question = question + suffix + answer_prefix
                pair_text = render_pair(entity, attribute, pair_template(sample, args.task, fmt))
                base_cache = prefill(model, tokenizer, context)

                for condition in args.conditions:
                    if condition == "removed_from_context":
                        ctrl_context, ctrl_suffix = apply_chat_template(
                            context_without_target(sample, args.task, fmt), tokenizer, args.chat_date)
                        result = generate_and_score(
                            model, tokenizer, prefill(model, tokenizer, ctrl_context),
                            question + ctrl_suffix + answer_prefix, correct_ids, args.max_new_tokens)
                    else:
                        positions = eviction_positions(
                            tokenizer, context, condition, pair_text, entity, attribute, is_attr_first(fmt))
                        result = run_condition(model, tokenizer, base_cache, positions,
                                               full_question, correct_ids, args.max_new_tokens)

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
