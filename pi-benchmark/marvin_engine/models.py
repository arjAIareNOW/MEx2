"""The three final models, run with ONNX Runtime on CPU.

WakeWord           MARVIN V4 (job 64883)          1 s window every 100 ms; fires after 2 windows in a row >= threshold
CommandRecognizer  intent V4-B+ (job 64884)        19 intents, optional out-of-scope reject threshold
                   slot V0 + DEMAND (job 65162_0)  18 slot values; only the 3 values of the predicted intent compete
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import WAKE_SAMPLES, command_features, wake_features

BUNDLE = Path(__file__).resolve().parents[1]
CONFIG = json.loads((BUNDLE / "models" / "config.json").read_text())


def _session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class WakeWord:
    """Streaming MARVIN detector. Feed 16 kHz float32 audio of any chunk size with push()."""

    def __init__(self, threshold: float | None = None, consecutive: int | None = None, cooldown_s: float | None = None,
                 threads: int = 2):
        c = CONFIG["wake"]
        self.session = _session(BUNDLE / "models" / c["onnx"], threads)
        self.input = self.session.get_inputs()[0].name
        self.threshold = float(threshold if threshold is not None else c["streaming_threshold"])
        self.consecutive = int(consecutive or c["consecutive_hits"])
        self.cooldown = int(16000 * (cooldown_s if cooldown_s is not None else c["cooldown_ms"] / 1000))
        self.stride = int(16000 * c["stride_ms"] / 1000)
        self.reset()

    def reset(self):
        self.buffer = np.zeros(WAKE_SAMPLES, np.float32)  # silence prefill: scoring starts immediately
        self.pending = np.zeros(0, np.float32)
        self.samples, self.hits, self.last_event = 0, 0, -10 ** 9

    def probability(self, windows: np.ndarray) -> np.ndarray:
        """MARVIN probability for one or more exact 1 s windows ([16000] or [N, 16000])."""
        logits = self.session.run(None, {self.input: wake_features(windows)})[0]
        return 1 / (1 + np.exp(-logits.reshape(-1)))

    def push(self, audio: np.ndarray) -> list[dict]:
        """Returns a list of wake events (usually empty). Each: {"time_s", "probability", "infer_ms"}."""
        self.pending = np.concatenate([self.pending, np.asarray(audio, np.float32).reshape(-1)])
        events = []
        while self.pending.size >= self.stride:
            chunk, self.pending = self.pending[:self.stride], self.pending[self.stride:]
            self.buffer = np.concatenate([self.buffer[self.stride:], chunk]); self.samples += self.stride
            t = time.perf_counter(); p = float(self.probability(self.buffer)[0]); ms = (time.perf_counter() - t) * 1000
            self.last_probability, self.last_infer_ms = p, ms
            self.hits = self.hits + 1 if p >= self.threshold else 0
            if self.hits >= self.consecutive and self.samples - self.last_event >= self.cooldown:
                self.last_event, self.hits = self.samples, 0
                events.append({"time_s": self.samples / 16000, "probability": p, "infer_ms": ms})
        return events

    def score_clip(self, audio: np.ndarray) -> float:
        """Offline score of a short recording with the streaming rule: max over adjacent windows of min(p).
        The clip is padded with 0.5 s of silence on both sides (as in the bundled benchmark)."""
        x = np.concatenate([np.zeros(8000, np.float32), np.asarray(audio, np.float32), np.zeros(8000, np.float32)])
        if x.size < WAKE_SAMPLES:
            x = np.pad(x, (0, WAKE_SAMPLES - x.size))
        starts = range(0, x.size - WAKE_SAMPLES + 1, self.stride)
        p = self.probability(np.stack([x[s:s + WAKE_SAMPLES] for s in starts]))
        return float(np.max(np.minimum(p[:-1], p[1:]))) if p.size > 1 else float(p[0])


class CommandRecognizer:
    """Intent + slot from the audio after the wake word (16 kHz float32, up to 5 s; longer audio is cut at 5 s)."""

    def __init__(self, reject_threshold: float | None = None, threads: int = 4):
        ci, cs = CONFIG["intent"], CONFIG["slot"]
        self.intent = _session(BUNDLE / "models" / ci["onnx"], threads)
        self.slot = _session(BUNDLE / "models" / cs["onnx"], threads)
        self.intents, self.slot_classes = ci["classes"], cs["classes"]
        self.slot_names = cs["slot_names"]
        self.reject_threshold = reject_threshold if reject_threshold is not None else ci.get("reject_threshold")
        self.groups = {}
        for i, c in enumerate(self.slot_classes):
            self.groups.setdefault(c.split("|")[0], []).append(i)

    def _run(self, session, feats):
        return session.run(None, {"log_mel": feats, "frame_lengths": np.array([feats.shape[-1]], np.int64)})[0][0]

    def predict(self, audio: np.ndarray) -> dict:
        t0 = time.perf_counter()
        feats = command_features(audio)
        t1 = time.perf_counter()
        p = _softmax(self._run(self.intent, feats))
        t2 = time.perf_counter()
        k = int(p.argmax()); intent = self.intents[k]
        out = {"intent": intent, "confidence": float(p[k]),
               "accepted": bool(self.reject_threshold is None or p[k] >= self.reject_threshold),
               "top3": [(self.intents[i], float(p[i])) for i in np.argsort(-p)[:3]],
               "slot_name": None, "slot": None, "slot_confidence": None,
               "audio_s": round(min(len(audio), 80000) / 16000, 3),
               "features_ms": (t1 - t0) * 1000, "intent_ms": (t2 - t1) * 1000, "slot_ms": 0.0}
        if intent in self.groups:  # slotted intent: pick one of its 3 values
            g = self.groups[intent]; z = self._run(self.slot, feats)[g]; q = _softmax(z); j = int(q.argmax())
            out.update(slot_name=self.slot_names[intent], slot=self.slot_classes[g[j]].split("|", 1)[1],
                       slot_confidence=float(q[j]), slot_ms=(time.perf_counter() - t2) * 1000)
        out["total_ms"] = (time.perf_counter() - t0) * 1000
        return out
