#!/usr/bin/env bash
# Retrain the V4-B+ intent model exactly as released (1 GPU, ~36 min on an A100 80GB).
# Output: runs/intent_v4b_plus_<timestamp>/ with best.pt, config.json, test/holdout metrics.
#   bash scripts/train_intent_v4b_plus.sh            # seed 23101 (released model)
#   SEED=7 bash scripts/train_intent_v4b_plus.sh     # another seed
#   bash scripts/train_intent_v4b_plus.sh --smoke    # 2-epoch CPU pipeline check
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUTPUT:-runs/intent_v4b_plus_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$$"
python -u -m vcm.v4b_plus_training --data dataset/ai231 --output "$OUT" --seed "${SEED:-23101}" \
  --epochs 120 --batch-size 128 --lr 2e-3 --weight-decay 0.05 --label-smoothing 0.1 \
  --mixup-alpha 0.3 --mixup-prob 0.5 --ema-decay 0.999 --patience 30 --workers 6 --cpu-threads 2 "$@"
echo "Done: $OUT"
