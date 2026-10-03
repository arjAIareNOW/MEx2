#!/usr/bin/env python3
"""Run the released models on a 16 kHz mono PCM16 WAV file (CPU only).

  python scripts/predict.py wake   clip.wav   # MARVIN V4 wake word (streaming rule)
  python scripts/predict.py intent clip.wav   # V4-B+ intent, top-3 of 19 classes

In the full system the intent model only runs after the wake word fires. The intent
model has no out-of-scope rejection yet, so every clip is mapped to one of 19 intents.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from v4b_plus_frontend import decode_wav, log_mel  # noqa: E402
from vcm.tc_dscnn_v4b_plus import IntentTCDSCNNV4BPlus  # noqa: E402
from wakeword_v3.streaming import WindowFeatures  # noqa: E402

WAKE_DIR = ROOT / 'models/marvin_v4'
INTENT_DIR = ROOT / 'models/intent_v4b_plus'
SR, WINDOW, STRIDE = 16000, 16000, 1600  # 1 s window, 100 ms stride


def wake(waveform: np.ndarray) -> dict:
    cfg = json.loads((WAKE_DIR / 'deployment_config.json').read_text())
    model = torch.jit.load(str(WAKE_DIR / 'marvin_v4.ts'), map_location='cpu').eval()
    stream = torch.from_numpy(waveform)
    if len(stream) < WINDOW:
        stream = F.pad(stream, (0, WINDOW - len(stream)))
    # One second of leading silence, as in the evaluation replay, so a word at the start can fire.
    windows = F.pad(stream, (WINDOW, 0)).unfold(0, WINDOW, STRIDE)[1:].contiguous()
    with torch.no_grad():
        probs = model(WindowFeatures()(windows).unsqueeze(1)).sigmoid().flatten()
    # Two consecutive windows must both pass the threshold.
    pair = torch.minimum(probs[:-1], probs[1:]) if len(probs) > 1 else probs
    best = int(pair.argmax())
    score = float(pair[best])
    return {'wake_word': 'marvin', 'detected': score >= cfg['streaming_threshold'], 'score': round(score, 4),
            'threshold': round(cfg['streaming_threshold'], 4), 'window_end_s': round((best + 1) * STRIDE / SR, 2)}


def intent(waveform: np.ndarray, top: int = 3) -> dict:
    ck = torch.load(INTENT_DIR / 'best.pt', map_location='cpu', weights_only=True)
    cfg = ck['model_config']
    model = IntentTCDSCNNV4BPlus(cfg['num_classes'], cfg['dropout'], cfg['width']).eval()
    model.load_state_dict(ck['model_state'])
    classes = sorted(ck['class_to_index'], key=ck['class_to_index'].get)
    features = torch.from_numpy(log_mel(waveform))
    with torch.no_grad():
        p = model(features, torch.tensor([features.shape[-1]])).softmax(-1)[0]
    values, indices = p.topk(top)
    return {'intent': classes[int(indices[0])],
            'top': [{'intent': classes[int(i)], 'p': round(float(v), 4)} for v, i in zip(values, indices)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task', choices=['wake', 'intent'])
    ap.add_argument('wav', type=Path)
    a = ap.parse_args()
    torch.set_num_threads(1)
    waveform = decode_wav(a.wav.read_bytes())  # raises unless mono 16 kHz PCM16
    print(json.dumps(wake(waveform) if a.task == 'wake' else intent(waveform), indent=2))


if __name__ == '__main__':
    main()
