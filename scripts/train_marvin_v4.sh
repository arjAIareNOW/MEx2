#!/usr/bin/env bash
# Retrain the MARVIN V4 wake word exactly as released (1 GPU, ~35 min on an A100 80GB).
# Output: runs/marvin_v4_<timestamp>/ with best.pt, marvin_v4.ts, thresholds and test metrics.
#   bash scripts/train_marvin_v4.sh                  # seed 1337 (released model)
#   bash scripts/train_marvin_v4.sh --smoke --device cpu   # 2-epoch CPU pipeline check
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUTPUT:-runs/marvin_v4_$(date +%Y%m%d_%H%M%S)}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-$$"
python -u -m wakeword_v4.train --data dataset/speech_commands \
  --extra-noise-dir dataset/noise/demand/wav16k --output "$OUT" \
  --epochs 100 --warmup-epochs 3 --batch-size 128 --learning-rate 3e-3 \
  --weight-decay 0.01 --label-smoothing 0.05 --negative-ratio 4 --noise-fraction 0.25 \
  --ema-decay 0.998 --patience 25 --workers 6 --cpu-threads 2 --seed 1337 \
  --validation-far-cap 0.0025 --device cuda "$@"
echo "Done: $OUT"
