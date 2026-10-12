"""Audio helpers: WAV I/O, resampling to 16 kHz, and the voice-activity endpointer used after the wake word.

Everything works on float32 mono arrays scaled to [-1, 1]. Only numpy is needed.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

SR = 16000


def _lowpass_taps(factor: float, half: int = 31) -> np.ndarray:
    """Blackman-windowed sinc low-pass at the 16 kHz output's Nyquist (8 kHz) for an input `factor` x faster."""
    n = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.sinc(n / factor) / factor * np.blackman(n.size)
    return taps / taps.sum()


def resample(x: np.ndarray, input_rate: int, output_rate: int = SR) -> np.ndarray:
    """Anti-aliased resampling to `output_rate`.

    Integer ratios (48 kHz -> 16 kHz, the Pi microphone) use the exact filter + decimation the Pi app was tested with.
    Other rates (44.1 kHz laptop microphones) low-pass first, then interpolate linearly.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if input_rate == output_rate or x.size == 0:
        return x
    if input_rate < output_rate:  # upsampling (e.g. an 8 kHz file): interpolation only
        t = np.arange(int(round(x.size * output_rate / input_rate))) * (input_rate / output_rate)
        return np.interp(t, np.arange(x.size), x).astype(np.float32)
    factor = input_rate / output_rate
    taps = _lowpass_taps(factor)
    pad = min(len(taps) // 2, max(1, x.size - 1))
    filtered = np.convolve(np.pad(x, (pad, pad), mode="reflect"), taps, mode="valid")
    if float(factor).is_integer():
        return filtered[:: int(factor)][: (x.size + int(factor) - 1) // int(factor)].astype(np.float32)
    t = np.arange(int(x.size / factor)) * factor
    return np.interp(t, np.arange(filtered.size), filtered).astype(np.float32)


def read_wav(path_or_bytes) -> np.ndarray:
    """Read a PCM16/PCM32/float WAV (any rate, any channel count) -> float32 mono 16 kHz."""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(Path(path_or_bytes))
    with wave.open(src, "rb") as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return resample(x, rate, SR)


def to_wav_bytes(x: np.ndarray, rate: int = SR) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return out.getvalue()


def level_db(block: np.ndarray) -> float:
    return float(20 * np.log10(max(float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0, 1e-5)))


class Endpointer:
    """Collects the command after the wake word and decides when it ended (same rule as the Pi app).

    Speech starts when a 20 ms block has RMS >= `vad_rms`; the command ends after `vad_silence` seconds of quieter
    blocks once speech was heard, or after `max_seconds` in total (the intent model reads at most 5 s).
    Feed 16 kHz float32 audio with push(); it returns the finished command audio once, else None.
    """

    def __init__(self, vad_rms: float = 0.012, vad_silence: float = 0.55, max_seconds: float = 5.0,
                 no_speech_timeout: float = 4.0):
        self.vad_rms, self.vad_silence, self.max_seconds, self.no_speech_timeout = vad_rms, vad_silence, max_seconds, no_speech_timeout
        self.reset()

    def reset(self):
        self.chunks, self.pending = [], np.zeros(0, np.float32)
        self.speech, self.silence, self.age = False, 0.0, 0.0

    def push(self, x: np.ndarray):
        """Returns (status, audio): status in {"listening", "done", "timeout"}; audio only when done."""
        self.pending = np.concatenate([self.pending, np.asarray(x, np.float32)])
        while self.pending.size >= 320:  # 20 ms blocks
            block, self.pending = self.pending[:320], self.pending[320:]
            self.chunks.append(block); self.age += 0.02
            if float(np.sqrt(np.mean(block * block))) >= self.vad_rms:
                self.speech, self.silence = True, 0.0
            elif self.speech:
                self.silence += 0.02
            if (self.speech and self.silence >= self.vad_silence) or self.age >= self.max_seconds:
                audio = np.concatenate(self.chunks); self.reset()
                return "done", audio
            if not self.speech and self.age >= self.no_speech_timeout:
                self.reset()
                return "timeout", None
        return "listening", None
