# MEx2: "Marvin" wake word + voice-command intent model

Two small on-device speech models for a Raspberry Pi voice assistant:

1. **MARVIN V4**: a wake-word detector that listens for the word *"marvin"*.
2. **V4-B+**: an intent classifier that maps the spoken command after the wake word to one of **19 intents** (e.g. `TIMER`, `LIGHT_ON`, `WEATHER`).

The intent model only runs after the wake word fires.

> **Status: partial release (2026-10-03).** Both models are trained and evaluated offline. Still to come: the **Raspberry Pi 5 on-device benchmark** (latency, RTF), the **ONNX export** of V4-B+, and an **out-of-scope reject threshold** for the intent model. Every number below was measured on the training server, not on a Pi.

| | Wake word: MARVIN V4 | Intent: V4-B+ |
|---|---|---|
| Task | binary: "marvin" vs. anything else | 19-class intent |
| Parameters | 65,293 (0.065 M) | 203,155 (0.203 M) |
| Weights | `models/marvin_v4/marvin_v4.ts` (TorchScript, 0.34 MB) + `best.pt` (0.30 MB) | `models/intent_v4b_plus/best.pt` (0.87 MB) |
| Test result | F1 **0.961**, recall 0.944, false-accept rate **0.037 %** (4 / 10,810) | accuracy **0.957**, macro-F1 **0.940** |
| Training data | TensorFlow Speech Commands v0.02 | ai231 voice-commands dataset |

---

## Contents

- [Quick start: run the released models](#quick-start-run-the-released-models)
- [Reproduce the training](#reproduce-the-training)
- [Model card](#model-card)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)
- [Licences and citation](#licences-and-citation)

---

## Quick start: run the released models

No dataset download or GPU is needed for this step.

```bash
git clone --recursive https://github.com/arjAIareNOW/MEx2.git
cd MEx2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

sha256sum -c models/SHA256SUMS                       # check the weights

python scripts/predict.py wake   my_clip.wav          # does it contain "marvin"?
python scripts/predict.py intent my_command.wav       # which of the 19 intents?
```

**Input:** a 16 kHz, mono, 16-bit PCM WAV file (≤ 5 s for the intent model). To convert any recording:
`ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 my_clip.wav`

Example output:

```json
{ "wake_word": "marvin", "detected": true, "score": 0.9867, "threshold": 0.9094, "window_end_s": 0.7 }
{ "intent": "TIMER", "top": [{"intent": "TIMER", "p": 0.97}, ...] }
```

`--recursive` also fetches the `vcm-benchmark` submodule. The intent code reads the official list of 19 intents from it. If you cloned without that flag, run `git submodule update --init`.

---

## Reproduce the training

Both models were trained on **one NVIDIA A100 80GB PCIe**, each taking about 35 minutes. Other CUDA GPUs should also work, only more slowly (not tested).

### 1. Install

```bash
git clone --recursive https://github.com/arjAIareNOW/MEx2.git && cd MEx2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 2. Download the data (pinned revisions, checksum-verified)

```bash
bash scripts/download_data.sh
```

This puts everything in `./dataset/` (about 4.5 GB download) and checks it against `data_checksums.sha256`:

| Dataset | Source (exact revision) | Used for |
|---|---|---|
| ai231 voice commands | HF [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) @ `5abbe53` (files `data/train-*`, `data/test-*`, `data/holdout-*`, `supplemental_fil/*`) | intent model: train / validation / test / holdout |
| Speech Commands v0.02 | `download.tensorflow.org/data/speech_commands_v0.02.tar.gz` (official `validation_list.txt` / `testing_list.txt`) | wake word: train / validation / test; `_background_noise_` augmentation for **both** models |
| DEMAND (16 kHz) | HF [`philgzl/demand`](https://huggingface.co/datasets/philgzl/demand) @ `29f46e6`, decoded once to WAV by `scripts/prepare_demand_noise.py` | wake-word noise augmentation |

### 3. Train

```bash
bash scripts/train_marvin_v4.sh          # wake word      -> runs/marvin_v4_<time>/
bash scripts/train_intent_v4b_plus.sh    # intent model   -> runs/intent_v4b_plus_<time>/
```

On a Slurm cluster, submit from the repo root. Edit `--partition` and `--gres` for your site first.

```bash
sbatch scripts/slurm/train_marvin_v4.slurm
sbatch scripts/slurm/train_intent_v4b_plus.slurm
```

**Check the pipeline first (CPU, a few minutes):**
`bash scripts/train_marvin_v4.sh --smoke --device cpu` and `bash scripts/train_intent_v4b_plus.sh --smoke`.

Each run writes `best.pt`, `config.json` (full hyper-parameters, environment and source hashes), `training_history.csv` and the test metrics. The intent run also scores the **holdout** set. The wake-word run also writes the thresholds (`deployment_config.json`), the per-word false-accept table and the TorchScript export.

**Seeds and determinism:** the released runs used seed **1337** (wake word) and **23101** (intent; this seed also fixes the speaker split). cuDNN autotuning is on for speed, so a re-run reproduces the results closely but not bit for bit. For reference, an earlier run of the same intent recipe with a different checkpoint-selection rule reached test macro-F1 0.945.

### 4. Optional: ONNX export and Pi benchmark (in progress)

```bash
python scripts/export_v4b_plus_onnx.py models/intent_v4b_plus            # -> v4b_plus.onnx
python scripts/select_v4b_plus_reject_threshold.py models/intent_v4b_plus --data dataset/ai231
# on the Raspberry Pi 5 (ONNX Runtime, NumPy front end, no torch needed):
python scripts/benchmark_v4b_plus_pi.py models/intent_v4b_plus \
       --parquet dataset/ai231/data/holdout-00000-of-00001.parquet --threads 4
```

---

## Model card

### Project

| | |
|---|---|
| Author / team | arjAIareNOW |
| Models | MARVIN V4 (wake word), then V4-B+ (intent) |
| Datasets | ai231 voice commands (intent); Speech Commands v0.02 (wake word) |
| Training | 1× NVIDIA A100 80GB PCIe, Slurm |
| On-device validation target | Raspberry Pi 5 (8 GB): **pending** |

### Model

| | Wake word: MARVIN V4 | Intent: V4-B+ |
|---|---|---|
| Mic input | 16 kHz mono PCM16 | 16 kHz mono PCM16 |
| Audio window | 1 s | up to 5 s (whole command) |
| Log-mel | 40 mels, 25 ms window / 10 ms hop, 20–7600 Hz, natural log, per-window mean/std normalised | 64 mels, 25 ms window / 10 ms hop, 20–8000 Hz, dB (80 dB range), per-clip mean/std normalised |
| Encoder | Conv stem (48 ch, stride 2), then residual depthwise-separable blocks with squeeze-excitation | Conv stem (64 ch, stride 2), then depthwise-separable CNN 64→128→128, then 6 dilated temporal blocks (dilation 1, 2, 4, 8, 16, 32; 128 ch) with squeeze-excitation |
| Causal / KV-cache | Not causal; streaming by sliding window (1 s window, 100 ms step, **2 consecutive hits** to fire, 1 s cooldown) | Not causal, no KV-cache: runs once on the command clip |
| Head | 1 logit, sigmoid | attentive statistics pooling (mean + std), then linear 256 → 19 |
| Slot head / actuator | none | none (intent only; slots are not predicted) |
| Parameters | 65,293 | 203,155 |
| Weights size | 0.30 MB (`best.pt`), 0.34 MB (TorchScript) | 0.87 MB (`best.pt`) |
| Decision thresholds | single clip **0.8335**; streaming **0.9094** (both picked on validation only, false-accept cap 0.25 %) | arg-max over 19 classes (reject threshold pending) |

**19 intents:** ALARM, BRIGHTNESS, CALL, COLOR, CREATE_REMINDER, LIGHT_OFF, LIGHT_ON, LIST_REMINDERS, MESSAGE, NEXT, PAUSE, PLAY_MUSIC, STOP, TEMPERATURE, TIME, TIMER, VOLUME_DOWN, VOLUME_UP, WEATHER.

### Dataset

**Intent (ai231 @ `5abbe53`).** In-scope clips only; out-of-scope clips are excluded from these counts. Validation is 15 % of the *training speakers*, held out by speaker (seed 23101).

| Split | Clips | Hours | Speakers | Human / synthetic |
|---|---|---|---|---|
| train (main + `supplemental_fil`) | 23,105 | 13.8 | 252 | 1,853 / 21,252 |
| validation | 1,478 | 0.8 | 43 | 305 / 1,173 |
| test (official) | 4,367 | 2.4 | 104 | 777 / 3,590 |
| holdout (Pi 5 set) | 186 | 0.16 | 4 | 86 / 100 |

The splits were checked for leakage: no speaker overlap and no identical audio between train, validation and test.

**Wake word (Speech Commands v0.02, official lists).**

| Split | Clips | "marvin" | Speakers |
|---|---|---|---|
| train | 84,843 | 1,710 | 2,112 |
| validation | 9,981 | 195 | 256 |
| test | 11,005 | 195 | 250 |

No speaker overlap between splits. All 34 other words act as negatives, including near-misses such as *backward*, *sheila* and *nine*.

### Training on the A100

| | Wake word: MARVIN V4 | Intent: V4-B+ |
|---|---|---|
| Hardware | 1× A100 80GB PCIe, node `uperdfi-a100gpu0` | same |
| Objective | binary cross-entropy (label smoothing 0.05) | cross-entropy (label smoothing 0.1) + mixup (α 0.3, p 0.5), class-balanced weights |
| Optimiser | AdamW, lr 3e-3, wd 0.01, 3-epoch warm-up, then cosine; EMA 0.998 | AdamW, lr 2e-3, wd 0.05, 5-epoch warm-up, then cosine to 1e-5; EMA 0.999; bf16 |
| Augmentation | speed 0.9–1.1, ±100 ms shift, ±6 dB gain, noise at 0–20 dB SNR (Speech Commands + DEMAND), SpecAugment, noise-only negatives | speed 0.9–1.1, ±100 ms shift, ±6 dB gain, noise at 5–25 dB SNR (Speech Commands), SpecAugment |
| Batch / steps | 128; 71 steps per epoch (1 marvin : 4 others); 88 epochs, best at epoch 63 | 128; ~181 steps per epoch; 120 epochs (~21.7k steps), best at epoch 104 |
| Checkpoint choice | validation F1 at the validation-chosen threshold | validation macro-F1 (better of raw / EMA weights each epoch) |
| Final validation loss | 0.034 | 0.334 |
| Wall-clock | 26 min training (35 min job) | 36 min |
| Seed | 1337 | 23101 |

### Validation (offline; Raspberry Pi 5 numbers pending)

| | Wake word: MARVIN V4 | Intent: V4-B+ |
|---|---|---|
| Accuracy | F1 0.961, recall 0.944, precision 0.979 | test: acc 0.957, macro-F1 0.940; holdout: acc 0.909, macro-F1 0.890 |
| False-accept rate | single clip: **0.037 %** (4 / 10,810); streaming replay: 0.032 %, recall 0.915 | out-of-scope rejection: **pending** (no reject threshold yet) |
| Latency p95 / RTF | pending (Pi 5) | pending (Pi 5) |
| Runtime | TorchScript CPU (ONNX Runtime planned) | PyTorch CPU now; ONNX Runtime, 4 threads planned |

---

## Results

### Wake word vs. smaller baselines (same test set, 11,005 clips)

| Model | Params | F1 | Recall | Precision | False accepts | False rejects |
|---|---|---|---|---|---|---|
| MARVIN V2 | 3.4 K | 0.836 | 0.800 | 0.876 | 22 | 39 |
| MARVIN V3 | 6.4 K | 0.890 | 0.851 | 0.933 | 12 | 29 |
| **MARVIN V4** | **65 K** | **0.961** | **0.944** | **0.979** | **4** | **11** |

Per-word false accepts are in `models/marvin_v4/hard_negative_word_results.csv`. The V2 and V3 baselines are not shipped. The training script still runs without them; it just skips the comparison.

### Intent vs. baseline (same test and holdout sets)

| Model | Params | Test acc / macro-F1 | Holdout acc / macro-F1 |
|---|---|---|---|
| V4-B (baseline) | 69 K | 0.890 / 0.858 | 0.731 / 0.712 |
| **V4-B+** | **203 K** | **0.957 / 0.940** | **0.909 / 0.890** |

The V4-B baseline is about a third of V4-B+'s size. The comparison in `models/intent_v4b_plus/comparison_vs_v4b.json` was scored in the same run. Per-class F1 and confusion matrices are in `test_metrics_v4b_plus.json` and `holdout_metrics_v4b_plus.json`. The weakest classes are VOLUME_UP (test F1 0.85), LIGHT_OFF (0.89) and STOP (0.90). On the holdout they are LIGHT_OFF and LIGHT_ON (about 0.62).

**How the final intent model was chosen:** two V4-B+ runs were trained; they differed only in the checkpoint-selection rule. The one with the higher **validation** macro-F1 (0.9453 vs. 0.9435) was released. Test and holdout were not used to choose.

---

## Repository layout

```
models/
  intent_v4b_plus/   best.pt, config.json, test/holdout metrics, comparison_vs_v4b.json,
                     split_manifest.json (exact train/validation indices), training_history.csv, training_log.txt
  marvin_v4/         best.pt, marvin_v4.ts, deployment_config.json (features + thresholds), threshold_selection.json,
                     metrics.json, comparison.csv, hard_negative_word_results.csv, training_history.csv, plots, training_log.txt
  SHA256SUMS         checksums of the weights
vcm/                 intent model, data split and training code (V4-B+ = tc_dscnn_v4b_plus.py, v4b_plus_training.py)
wakeword/, wakeword_v3/, wakeword_v4/   wake-word data, features, streaming rule and training (V4 = wakeword_v4/)
scripts/
  predict.py                     run either model on a WAV file
  download_data.sh               fetch + verify all datasets
  train_*.sh, slurm/*.slurm      retrain the released models
  prepare_demand_noise.py        decode DEMAND once to 16 kHz WAV
  export_v4b_plus_onnx.py, v4b_plus_frontend.py, benchmark_v4b_plus_pi.py, select_v4b_plus_reject_threshold.py
vcm-benchmark/       git submodule (airimonda/vcm-benchmark @ ab39857): intent list and Pi benchmark
data_checksums.sha256
```

In the saved JSON files, paths on the training machine appear as `<PROJECT>` (this repo) and `<AI231>` (`dataset/ai231`).

---

## Limitations

- **No on-device numbers yet.** Latency, RTF and memory on the Raspberry Pi 5 are still to be measured.
- **The intent model has no out-of-scope rejection yet.** Any speech after "marvin" becomes one of the 19 intents. A validation-chosen confidence threshold is the next step.
- **The intent data is mostly synthetic** (about 92 % of training clips are TTS). The drop from test (0.940) to the 4-speaker holdout (0.890) suggests real voices and rooms are harder.
- **The wake-word streaming numbers come from replaying test clips padded with silence.** They are not a real-world false-accepts-per-hour rate.
- English only, 16 kHz input only. If your mic records at 48 kHz, resample to 16 kHz before feeding the models.

---

## Licences and citation

- **Code and model weights:** MIT (see `LICENSE`).
- **Datasets keep their own licences. They are not redistributed here; the scripts download them.**
  - ai231 voice commands: mixed sources, each with its own terms; research and education use (see its dataset card).
  - Speech Commands v0.02: CC BY 4.0. Warden, P. (2018). *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.* arXiv:1804.03209.
  - DEMAND: CC BY 4.0. Thiemann, J., Ito, N., Vincent, E. (2013). *The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND).* doi:10.1121/1.4799597.
- `vcm-benchmark` is a separate repository, included as a submodule. It is not covered by this licence.
