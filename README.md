# What Survives Eviction? Information Recall Locality in Transformer KV Caches

Code for Experiments 1–3 of the TMLR paper
*What Survives Eviction? Information Recall Locality in Transformer KV Caches*

Each experiment does three things: prefill a context into the KV cache, remove
the cache entries of specific tokens, and then ask the model to recall what
those tokens said. Remaining keys are re-rotated so that RoPE positions stay
contiguous after eviction.

| Paper section | Question | Script |
|---|---|---|
| §4.1, Exp. 1 | Does information survive **full** eviction of an entity–attribute pair? | `run_synthetic.py` |
| §4.2, Exp. 2 | Does it survive **partial** eviction (only the entity or only the attribute)? | `run_synthetic.py` |
| §4.3.1, Exp. 3 | Can a later, unrelated **re-mention** of the entity rescue it (controlled lists)? | `run_remention.py` |
| §4.3.2–4.3.3, Exp. 3 | Same question in ~4K-token **Needle-in-a-Haystack** prose, over a depth × gap grid | `run_niah.py` |

## Setup

Python ≥ 3.10 and one GPU. The paper used one 48 GB GPU per run; every model
here is ≤ 8B parameters in bf16.

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the library versions used for the paper. The Llama-3
and Gemma-3 checkpoints are gated on Hugging Face, so accept their licenses
and run `huggingface-cli login` before you start.

## Data

| File | Contents | Used by |
|---|---|---|
| `data/factual.json` | 900 factual lists (300 each: person–occupation, object–material, color–apparel) | Exp. 1–2 |
| `data/sentiment.json` | 1,000 sentiment lists (binary adjective–noun) | Exp. 1–2 |
| `data/factual_remention.json`, `data/sentiment_remention.json` | The same lists, each with a re-mention sentence appended | Exp. 3 (controlled) |
| `data/niah.json` | 8,600 NIAH samples (not shipped, ~330 MB; built below) | Exp. 3 (NIAH) |

The four shipped files are the exact samples used in the paper. Build the
NIAH grid from the Paul Graham essays (`alessiodevoto/paul_graham_essays` on
HuggingFace). The output is deterministic and matches the paper's dataset:

```bash
python data/make_niah.py --out data/niah.json
```

The other generators document how the shipped files were made:

* `data/add_remention.py` rebuilds the two `*_remention.json` files
  byte-for-byte from `factual.json` and `sentiment.json`.
* `data/make_synthetic.py --seed N` draws *new* lists from the same word pools
  and templates. The paper's lists were drawn without a fixed seed, so use the
  shipped files to reproduce the paper.

In the data files and result CSVs, `structured` is the paper's *enumerated*
format ("In the office, there are doctor Alice, …"), and `natural` is the
*atomic* format ("The doctor is Alice. …"). `attr_first` / `adj_first` and
`entity_first` give the token order within each pair.

## Running

Example (Llama-3.2-1B):

```bash
# Experiments 1-2: full and partial eviction
python run_synthetic.py --model meta-llama/Llama-3.2-1B-Instruct --task factual
python run_synthetic.py --model meta-llama/Llama-3.2-1B-Instruct --task sentiment

# Experiment 3: re-mention in controlled contexts
python run_remention.py --model meta-llama/Llama-3.2-1B-Instruct --task factual
python run_remention.py --model meta-llama/Llama-3.2-1B-Instruct --task sentiment

# Experiment 3: Needle-in-a-Haystack
python run_niah.py --model meta-llama/Llama-3.1-8B-Instruct

# Tables
python summarize.py --synthetic results/synthetic --remention results/remention --niah results/niah
```

To run all eight models and print every table, use `bash run_all.sh`. Set
`LIMIT=20` for a quick pass over the first 20 samples of each dataset. Every
script also accepts `--limit N`.

Results are written to `results/<experiment>/<model>/[<task>/]results.csv`,
one row per (sample, format, question direction, condition), with these columns:

* `greedy_answer`: the greedy-decoded answer.
* `accuracy`: 1 if the gold answer appears in the answer.
* `abstained`: 1 if the model declined to answer.
* `correct_prob`: the probability of the gold answer's first token, maximized over decoding steps (the paper's P(c)).

`summarize.py` averages within each model, then across models, and prints the
numbers behind Figures 2–7 (main text).

### Conditions

| Script | Condition | KV cache after prefill |
|---|---|---|
| `run_synthetic.py` | `no_eviction` | unchanged |
| | `removed_from_context` | control: target pair never in the input |
| | `evict_both` | entity and attribute evicted (Exp. 1) |
| | `evict_both_keep_punct` | as `evict_both`, trailing punctuation kept (Appendix B.4) |
| | `evict_attribute`, `evict_entity` | one side evicted (Exp. 2) |
| `run_remention.py` | `no_eviction`, `evict_pair`, `evict_pair_and_anchor` | pair evicted; re-mention ("anchor") kept or also evicted |
| `run_niah.py` | `no_eviction`, `evict_needle`, `evict_needle_and_anchor`, `evict_needle_no_anchor` | needle evicted; re-mention kept, evicted, or never inserted |

Token spans are located with the tokenizer's offset mapping. Punctuation-only
tokens directly after a span are evicted along with it.

## Reproducibility notes

* **Models.** Exp. 1–3 (controlled): Llama-3.2-1B/3B-Instruct,
  Llama-3.1-8B-Instruct, Qwen3-0.6B/4B/8B, and Gemma-3-1B/4B-IT. NIAH: the four
  4B+ models, Llama-3.1-8B, Qwen3-4B, Qwen3-8B, and Gemma-3-4B (`run_all.sh`).
  Qwen3 runs with thinking disabled.
* **Llama-3 chat template date.** The Llama-3 chat template writes the current
  date into its system prompt, so outputs change slightly from day to day. Each
  runner pins the date with `--chat_date`. The defaults are the dates of the
  paper runs (21 Apr 2026, 22 Apr 2026, and 10 May 2026). Other model families
  ignore this option.
* **Decoding.** Greedy decoding with `max_new_tokens` 10 (Exp. 1–2, NIAH) and
  5 (Exp. 3, controlled), matching the paper.
* **Numerical agreement.** bf16 kernels differ across GPUs and library
  versions. In our checks this code reproduced the per-row correctness of the
  paper runs on 98–100% of rows. Generated text occasionally differs on
  borderline cases, which does not change the aggregate results.

## Code layout

```
recall_locality/
  eviction.py    find token spans; evict them from a DynamicCache; RoPE re-rotation
  inference.py   model loading, chat template, prefill, greedy decoding + P(correct), grading
  formats.py     Synthetic Associations templates and the removed_from_context rebuild
  common.py      dataset loading and CSV output
run_synthetic.py, run_remention.py, run_niah.py, summarize.py, run_all.sh
data/            datasets and the scripts that build them
```

The RoPE re-rotation is adapted from `KeyRerotationPress` in
[NVIDIA kvpress](https://github.com/NVIDIA/kvpress) (Apache-2.0).

