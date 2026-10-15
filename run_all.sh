#!/usr/bin/env bash
# Reproduce Experiments 1-3 for every model in the paper, then print the summary tables.
#
#   bash run_all.sh            # full run
#   LIMIT=20 bash run_all.sh   # quick check on the first 20 samples of each dataset
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

MODELS=(
  meta-llama/Llama-3.2-1B-Instruct
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
  Qwen/Qwen3-0.6B
  Qwen/Qwen3-4B
  Qwen/Qwen3-8B
  google/gemma-3-1b-it
  google/gemma-3-4b-it
)
# The NIAH experiment uses the four 4B+ models (see Section 4.3.2).
NIAH_MODELS=(
  meta-llama/Llama-3.1-8B-Instruct
  Qwen/Qwen3-4B
  Qwen/Qwen3-8B
  google/gemma-3-4b-it
)

[ -f data/niah.json ] || "$PYTHON" data/make_niah.py --out data/niah.json

for model in "${MODELS[@]}"; do
  for task in factual sentiment; do
    "$PYTHON" run_synthetic.py --model "$model" --task "$task" $LIMIT_ARG   # Exp. 1-2
    "$PYTHON" run_remention.py --model "$model" --task "$task" $LIMIT_ARG   # Exp. 3, controlled
  done
done
for model in "${NIAH_MODELS[@]}"; do
  "$PYTHON" run_niah.py --model "$model" $LIMIT_ARG                         # Exp. 3, NIAH
done

"$PYTHON" summarize.py --synthetic results/synthetic --remention results/remention --niah results/niah \
  | tee results/summary.txt
