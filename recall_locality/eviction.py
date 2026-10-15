"""Post-prefill KV-cache eviction with RoPE re-rotation.

The re-rotation math (``_rerotate_cos_sin``) is adapted from
``KeyRerotationPress`` in NVIDIA kvpress (https://github.com/NVIDIA/kvpress),
Apache License 2.0.
"""

import copy
import logging

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import rotate_half

logger = logging.getLogger(__name__)

_TRAILING_PUNCT = set(".,;:!?'\")")


def find_token_positions(
    tokenizer: PreTrainedTokenizerBase,
    context: str,
    target: str,
    include_trailing_punct: bool = True,
) -> list[int]:
    """Token indices in ``context`` that overlap the first occurrence of ``target``.

    Uses the tokenizer's offset mapping. With ``include_trailing_punct`` (the
    default, used for every condition except ``evict_both_keep_punct``), any
    punctuation-only tokens directly after the target are also returned.
    """
    char_start = context.find(target)
    if char_start == -1:
        logger.warning("Target %r not found in context", target)
        return []
    char_end = char_start + len(target)

    offsets = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False).offset_mapping
    positions = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]

    if include_trailing_punct and positions:
        for j in range(positions[-1] + 1, len(offsets)):
            s, e = offsets[j]
            text = context[s:e].strip()
            if text and all(c in _TRAILING_PUNCT for c in text):
                positions.append(j)
            else:
                break
    return positions


def _language_model(model: PreTrainedModel):
    """Inner decoder stack (Gemma-3 multimodal checkpoints wrap it once more)."""
    if hasattr(model.model, "language_model"):
        return model.model.language_model
    return model.model


def _inv_freq(model: PreTrainedModel, language_model, layer) -> torch.Tensor:
    rotary = language_model.rotary_emb
    if type(model).__name__.startswith("Gemma3"):
        # Gemma-3 uses separate RoPE bases for sliding-window and global layers.
        if getattr(layer.self_attn, "is_sliding", False):
            return rotary.sliding_attention_inv_freq
        return rotary.full_attention_inv_freq
    return rotary.inv_freq


def _rerotate_cos_sin(x, inv_freq, selected_positions):
    """cos/sin that move each kept key from its original position to its new index."""
    bsz, num_heads, n_kept = selected_positions.shape
    device = selected_positions.device
    device_type = x.device.type if x.device.type != "mps" else "cpu"

    idx = torch.arange(0, n_kept, device=device).unsqueeze(0)
    inv_freq = inv_freq.to(device)[None, None, :, None].float().expand(bsz, num_heads, -1, 1)
    idx = idx[:, None, :].float().expand(bsz, num_heads, n_kept)
    delta_pos = (idx - selected_positions).unsqueeze(2)

    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (delta_pos.float() * inv_freq.float()).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().contiguous(), emb.sin().contiguous()
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def evict_and_rerotate(
    cache: DynamicCache,
    positions: list[int],
    model: PreTrainedModel,
    rerotate: bool = True,
) -> DynamicCache:
    """Return a copy of ``cache`` with ``positions`` removed from every layer.

    The remaining keys are re-rotated so their RoPE angles are contiguous
    (``0 .. n_kept-1``), as if the evicted tokens had never been there.
    The input cache is left untouched.
    """
    cache = copy.deepcopy(cache)
    if not positions:
        return cache

    seq_len = cache.get_seq_length()
    evict = set(positions)
    keep_cpu = torch.tensor([p for p in range(seq_len) if p not in evict], dtype=torch.long)

    for layer in cache.layers:
        keep = keep_cpu.to(layer.keys.device)
        layer.keys = layer.keys.index_select(2, keep).contiguous()
        layer.values = layer.values.index_select(2, keep).contiguous()

    if rerotate:
        language_model = _language_model(model)
        for layer_idx, layer_module in enumerate(language_model.layers):
            keys = cache.layers[layer_idx].keys  # (batch, heads, seq, head_dim)
            bsz, num_heads = keys.shape[:2]
            selected = keep_cpu.to(keys.device).view(1, 1, -1).expand(bsz, num_heads, -1)
            cos, sin = _rerotate_cos_sin(keys, _inv_freq(model, language_model, layer_module), selected)
            cache.layers[layer_idx].keys = keys * cos + rotate_half(keys) * sin
    return cache
