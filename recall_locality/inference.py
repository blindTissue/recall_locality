"""Model loading, prefill, greedy decoding and answer grading."""

import copy
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from recall_locality.eviction import evict_and_rerotate


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def _input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        first = next(iter(model.hf_device_map.values()))
        return torch.device(f"cuda:{first}" if isinstance(first, int) else first)
    return model.device


def apply_chat_template(context: str, tokenizer, date_string: str | None = None) -> tuple[str, str]:
    """Split the chat-formatted prompt into (context part, suffix after the user turn).

    The context part is prefilled and edited in cache; the question is later
    fed as ``question + suffix + answer_prefix``. Thinking mode is disabled for
    models that support it (Qwen3).

    Llama-3 templates put a date in the system prompt (today's date unless
    ``date_string`` is given), so pin ``date_string`` for reproducible runs.
    Other templates ignore it.
    """
    if tokenizer.chat_template is None:
        return (tokenizer.bos_token or "") + context, "\n"

    separator = "#" * (len(context) + 10)
    messages = [{"role": "user", "content": context + separator}]
    kwargs = dict(add_generation_prompt=True, tokenize=False)
    if date_string:
        kwargs["date_string"] = date_string
    try:
        formatted = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, **kwargs)
    context_part, suffix = formatted.split(separator)
    return context_part, suffix


@torch.no_grad()
def prefill(model, tokenizer, context: str) -> DynamicCache:
    ids = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False).to(_input_device(model))
    cache = DynamicCache()
    model.model(input_ids=ids, past_key_values=cache)
    return cache


def answer_token_ids(tokenizer, answer: str) -> list[int]:
    """First-token ids of plausible surface forms of ``answer`` (bare / spaced / capitalized)."""
    variants = {answer, answer.capitalize(), answer.lower()}
    variants |= {" " + v for v in variants}
    ids = {tokenizer.encode(v, add_special_tokens=False)[0] for v in variants if v}
    return sorted(ids)


@torch.no_grad()
def generate_and_score(model, tokenizer, cache: DynamicCache, question: str,
                       correct_ids: list[int], max_new_tokens: int) -> dict:
    """Greedy-decode an answer after ``question`` on top of ``cache``.

    Returns the decoded answer and ``correct_prob``: the max, over generated
    positions, of the probability assigned to any of ``correct_ids`` (so that
    leading formatting tokens such as ``**`` do not hide the answer).
    """
    device = _input_device(model)
    question_ids = tokenizer.encode(question, return_tensors="pt", add_special_tokens=False).to(device)
    start = cache.get_seq_length()
    position_ids = torch.arange(start, start + question_ids.shape[1], device=device).unsqueeze(0)

    logits = model(input_ids=question_ids, past_key_values=cache, position_ids=position_ids).logits[0, -1].float()
    stop_ids = model.generation_config.eos_token_id
    stop_ids = stop_ids if isinstance(stop_ids, list) else [stop_ids] if stop_ids is not None else []

    generated, correct_probs = [], []
    next_pos = position_ids[:, -1:] + 1
    for step in range(max_new_tokens):
        probs = torch.softmax(logits, dim=-1)
        correct_probs.append(max(probs[t].item() for t in correct_ids))
        token = logits.argmax().item()
        generated.append(token)
        if step == max_new_tokens - 1 or (step > 0 and token in stop_ids):
            break
        logits = model(
            input_ids=torch.tensor([[token]], device=device),
            past_key_values=cache,
            position_ids=next_pos + step,
        ).logits[0, -1].float()

    return {
        "greedy_answer": tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "correct_prob": max(correct_probs),
    }


def run_condition(model, tokenizer, base_cache: DynamicCache, evict_positions: list[int],
                  question: str, correct_ids: list[int], max_new_tokens: int) -> dict:
    """Evict ``evict_positions`` from a copy of ``base_cache`` and answer ``question``."""
    if evict_positions:
        cache = evict_and_rerotate(base_cache, evict_positions, model)
    else:
        cache = copy.deepcopy(base_cache)
    return generate_and_score(model, tokenizer, cache, question, correct_ids, max_new_tokens)


_FORMAT_RE = re.compile(r"[*_`\"'#\[\](){}]")
_REFUSAL_PHRASES = [
    "not present", "not mentioned", "not provided", "not listed",
    "not found", "not specified", "not given", "not stated",
    "not included", "unknown", "cannot determine", "n/a",
    "no information", "is not in", "not explicitly", "not directly",
]


def grade(greedy_answer: str, correct_answer: str, task: str) -> tuple[int, int]:
    """Return (accuracy, abstained) for a generated answer.

    Correct = the answer string appears in the generation after stripping
    markdown. Sentiment answers that name neither label, and factual answers
    containing a refusal phrase, count as abstentions.
    """
    text = _FORMAT_RE.sub("", greedy_answer).lower()
    accuracy = int(correct_answer.lower() in text)
    if task == "sentiment":
        abstained = "positive" not in text and "negative" not in text
    else:
        abstained = any(p in text for p in _REFUSAL_PHRASES)
    return accuracy, int(abstained)
