"""Wake word -> command capture -> intent/slot, as one event stream for a user interface.

    from marvin_engine import Assistant
    a = Assistant(on_event=print)      # any callable taking one dict
    a.run_microphone()                 # blocks; call a.stop() from another thread (or Ctrl+C)

Events (dicts, JSON-serialisable; "type" is always present, "t" = time.time()):
    {"type": "ready", "mic_rate": 16000, "device": ...}
    {"type": "level", "db": -42.1, "wake_probability": 0.03}         every 100 ms while idle (for a VU meter)
    {"type": "wake", "probability": 0.97, "infer_ms": 1.8}            MARVIN heard -> now listening for the command
    {"type": "listening_timeout"}                                     no speech within 4 s after the wake word
    {"type": "command", "intent": "TIMER", "confidence": 0.98, "accepted": true, "slot_name": "duration",
     "slot": "30 seconds", "slot_confidence": 0.99, "top3": [...], "audio_s": 2.1, "total_ms": 21.0,
     "endpoint_to_result_ms": 23.4, "audio_wav": b"..." (only if keep_audio=True)}
    {"type": "error", "message": "..."}
The engine only recognises; carrying out the command (lights, timers, music, replies) is the interface's job.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

from .audio import SR, Endpointer, level_db, resample, to_wav_bytes
from .models import CONFIG, CommandRecognizer, WakeWord


class Assistant:
    def __init__(self, on_event: Callable[[dict], None] = print, mic_rate: int | None = None, device=None,
                 wake_threshold: float | None = None, reject_threshold: float | None = None,
                 vad_rms: float | None = None, vad_silence: float | None = None, keep_audio: bool = False,
                 threads: int = 2):
        r = CONFIG["runtime"]
        self.on_event, self.device, self.keep_audio = on_event, device, keep_audio
        self.mic_rate = mic_rate
        self.wake = WakeWord(threshold=wake_threshold, threads=threads)
        self.commands = CommandRecognizer(reject_threshold=reject_threshold, threads=max(threads, 2))
        self.endpointer = Endpointer(vad_rms if vad_rms is not None else r["vad_rms"],
                                     vad_silence if vad_silence is not None else r["vad_silence_s"], r["max_command_s"])
        self.guard_s = r["wake_guard_s"]
        self._stop = threading.Event()
        self.state = "idle"

    # ---- pure processing (no microphone): usable from tests, files, websockets, etc. ----
    def _emit(self, event: dict):
        event.setdefault("t", time.time())
        try:
            self.on_event(event)
        except Exception as exc:  # never let a UI callback kill the audio loop
            print("on_event error:", exc)

    def reset(self):
        self.wake.reset(); self.endpointer.reset(); self.state = "idle"; self._guard = 0

    def feed(self, audio16k: np.ndarray):
        """Feed 16 kHz float32 audio (any length). Emits events through on_event."""
        audio16k = np.asarray(audio16k, np.float32).reshape(-1)
        if self.state == "idle":
            events = self.wake.push(audio16k)
            if events:
                self._emit({"type": "wake", **events[-1]})
                self.state, self._guard = "listening", int(self.guard_s * SR)
                self.endpointer.reset()
            return
        if self._guard > 0:  # skip the tail of "marvin" itself
            n = min(self._guard, audio16k.size); self._guard -= n; audio16k = audio16k[n:]
            if not audio16k.size:
                return
        status, cmd = self.endpointer.push(audio16k)
        if status == "timeout":
            self._emit({"type": "listening_timeout"}); self.wake.reset(); self.state = "idle"
        elif status == "done":
            t = time.perf_counter(); res = self.commands.predict(cmd)
            res["endpoint_to_result_ms"] = (time.perf_counter() - t) * 1000
            if self.keep_audio:
                res["audio_wav"] = to_wav_bytes(cmd)
            self._emit({"type": "command", **res}); self.wake.reset(); self.state = "idle"

    def recognize_command(self, audio16k: np.ndarray) -> dict:
        """Skip the wake word: intent + slot for a recorded command (e.g. a push-to-talk button)."""
        return self.commands.predict(np.asarray(audio16k, np.float32))

    # ---- live microphone (needs `pip install sounddevice`) ----
    def stop(self):
        self._stop.set()

    def run_microphone(self):
        import sounddevice as sd
        rate = self.mic_rate or int(sd.query_devices(self.device, "input")["default_samplerate"])
        try:  # prefer 16 kHz directly when the device supports it
            if self.mic_rate is None:
                sd.check_input_settings(device=self.device, samplerate=SR, channels=1, dtype="float32"); rate = SR
        except Exception:
            pass
        q: queue.Queue = queue.Queue()
        block = int(rate * 0.02)
        self.reset(); self._stop.clear()

        def callback(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        with sd.InputStream(samplerate=rate, channels=1, dtype="float32", blocksize=block, device=self.device,
                            callback=callback):
            self._emit({"type": "ready", "mic_rate": rate, "device": str(self.device)})
            acc, last_level = [], 0.0
            while not self._stop.is_set():
                try:
                    x = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                acc.append(x)
                if sum(a.size for a in acc) < int(rate * 0.1):  # process in 100 ms pieces
                    continue
                raw = np.concatenate(acc); acc = []
                try:
                    self.feed(resample(raw, rate, SR))
                except Exception as exc:
                    self._emit({"type": "error", "message": repr(exc)}); self.reset()
                if self.state == "idle" and time.time() - last_level >= 0.1:
                    last_level = time.time()
                    self._emit({"type": "level", "db": level_db(raw),
                                "wake_probability": float(getattr(self.wake, "last_probability", 0.0))})
