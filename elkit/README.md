# ELKIT wake word ("Hi Elkit" / "Hello Elkit")

ELKIT is the wake word of the deployed appliance (E.L.K.I.T., *Embedded Listening Keyword-based Intent Terminal*).
It listens all the time; when it hears "Hi Elkit" or "Hello Elkit", the command after it goes to the
**V4-B+ intent model** and the slot model (see the main [README](../README.md)).

**The ELKIT weights are not in this repository.** ELKIT was trained only on generated voices (Piper and Meta MMS),
whose licences are non-commercial. This repository is MIT, which would allow commercial use, so the model is kept
out. What is here: how it was built, how it was tested, and the test script, so anyone with the model can check the
numbers. **MARVIN V4** (the wake word *"marvin"*, MIT, in `models/marvin_v4/`) is the shareable alternative and runs
in the same pipeline.

## Model

| | ELKIT |
|---|---|
| Phrases | "Hi Elkit", "Hello Elkit" (one model for both) |
| Training | UPERDFI A100 server, job 65357 |
| Training data | generated voices only (Piper + Meta MMS) |
| Front end | openWakeWord v0.5.1 mel spectrogram + speech embedding, unchanged (Apache-2.0) |
| Head (trained) | 16 embeddings [N,16,96] -> 1 logit |
| Streaming | last 2 s of audio, scored every 80 ms |
| Runtime | numpy + onnxruntime (CPU) |

## Results

**Hearing the wake phrase** (from the model's own test report, unseen speakers):

| | threshold 0.7575 | threshold 0.88 |
|---|---|---|
| Generated "Hi/Hello Elkit" voices (2,832) | 97.3 % | 95.2 % |
| "Hi/Hello Elkit" stitched from real speakers' words (800) | **91.1 %** | 86.4 % |
| Real single words (6,194): false wakes | 1 | 1 |
| Real ai231 commands (89): false wakes | 8 | 1 |

**False wakes on real conversational speech** (our test, 2026-10-12, `scripts/elkit_false_accept.py`):
2,000 random SLURP *slurp_real* utterances (seed 23101, 1.63 h of speech, none of them say the wake phrase).
A setting fires when *k* scores in a row reach the threshold.

| Threshold | 1 reading | 2 readings in a row |
|---|---|---|
| 0.7575 (model default) | 23.6 % (472) | 12.8 % (256) |
| 0.88 | 16.8 % (335) | 8.9 % (177) |
| 0.92 | 13.7 % (273) | 6.7 % (134) |
| 0.93 | 12.4 % (248) | 6.1 % (122) |
| **0.94** | 10.8 % (215) | **5.0 % (99)** |
| 0.95 | 8.9 % (177) | 4.0 % (79) |

**The appliance uses 0.94 with 2 readings in a row** (5.0 % of utterances; about 61 per hour of non-stop talking).
Two readings matter more than the threshold: 0.92 × 2 already beats 0.95 × 1. Padding the clips with room noise
instead of silence, or not padding them, gave similar rates (checked on 100 clips), so the false wakes are not an
artifact of the test.

## Why so many false wakes on real speech

- The model never heard real voices during training, only generated ones.
- Its own false-wake tests used single words and 89 short commands. SLURP is long, natural home-assistant speech,
  full of "hey", "hello", "kitchen", "el-" sounds, and a 4 s sentence gives the detector about 50 chances to fire.
- The highest-scoring false wakes reach 0.98-0.99, so no threshold removes them all.

The next step is retraining with real speech as negatives (for example the SLURP clips that falsely woke it).

## Reproduce the false-wake test

```bash
pip install numpy onnxruntime soundfile
python scripts/elkit_false_accept.py --elkit-dir path/to/elkit --slurp-dir path/to/slurp_real --n 2000
```
`--elkit-dir` is the ELKIT bundle (`elkit_wakeword.py` + `models/`). The script prints the table above and saves
every clip's score sequence, so other settings can be checked without re-running the model.
