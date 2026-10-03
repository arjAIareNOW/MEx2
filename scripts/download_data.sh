#!/usr/bin/env bash
# Download every dataset used by the released models, pinned to the exact revisions used
# for training, into ./dataset, then verify SHA-256 checksums (data_checksums.sha256).
#
#   bash scripts/download_data.sh            # all data (~4.5 GB download)
#   SKIP_DEMAND=1 bash scripts/download_data.sh   # intent model only (no wake-word noise)
#
# Needs: python with requirements.txt installed (for `hf` and soundfile), curl, tar, sha256sum.
set -euo pipefail
cd "$(dirname "$0")/.."

AI231_REPO=airimonda/ai231-me2-voice-commands
AI231_REV=5abbe539a46b9b26ff24e73d2860d1698d44e81f      # 2026-10-02, used for training
DEMAND_REPO=philgzl/demand
DEMAND_REV=29f46e613920cc05db198fd4ad7992e874e7f60f
SPEECH_COMMANDS_URL=http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

mkdir -p dataset

echo "== ai231 voice commands (intent model) =="
hf download "$AI231_REPO" --repo-type dataset --revision "$AI231_REV" \
  --include "data/train-*" "data/test-*" "data/holdout-*" "supplemental_fil/*" "README.md" \
  --local-dir dataset/ai231

echo "== TensorFlow Speech Commands v0.02 (wake word + background noise for both models) =="
if [[ ! -f dataset/speech_commands/testing_list.txt ]]; then
  mkdir -p dataset/speech_commands
  curl -L --fail "$SPEECH_COMMANDS_URL" | tar -xz -C dataset/speech_commands
fi

if [[ -z "${SKIP_DEMAND:-}" ]]; then
  echo "== DEMAND noise, 16 kHz (wake-word augmentation) =="
  hf download "$DEMAND_REPO" --repo-type dataset --revision "$DEMAND_REV" \
    --include "data/16k-*" --local-dir dataset/noise/demand
  python scripts/prepare_demand_noise.py \
    --parquet-dir dataset/noise/demand/data --output dataset/noise/demand/wav16k
fi

echo "== Verifying checksums =="
if [[ -n "${SKIP_DEMAND:-}" ]]; then
  grep -v demand data_checksums.sha256 | sha256sum -c -
else
  sha256sum -c data_checksums.sha256
fi
echo "Data ready in ./dataset"
