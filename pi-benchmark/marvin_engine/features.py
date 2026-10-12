"""NumPy replicas of the two training front ends (checked against the PyTorch originals, see tests/verify.py).

wake_features : MARVIN V4  — 1 s window, 40 log-mels (n_fft 512, win 400, hop 160, no centre padding), per-window
                mean / std normalisation -> float32 [N, 1, 40, 97]
command_features : intent + slot models — up to 5 s, 64 mels 20-8000 Hz, centre/reflect padding, dB with top 80 dB,
                per-utterance mean / std -> float32 [1, 1, 64, T]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SR, N_FFT, WIN, HOP = 16000, 512, 400, 160
_ASSETS = Path(__file__).resolve().parent / "assets"
_wake = np.load(_ASSETS / "wake_frontend.npz")
_WAKE_WINDOW = _wake["window"].astype(np.float64)   # [512] periodic Hann(400) zero-padded to 512, centred
_WAKE_BANK = _wake["mel_bank"].astype(np.float64)   # [40, 257] (exported from the training code)
WAKE_SAMPLES = 16000


def wake_features(windows: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    assert x.shape[1] == WAKE_SAMPLES, "wake windows must be exactly 16000 samples (1 s at 16 kHz)"
    frames = 1 + (WAKE_SAMPLES - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    power = np.abs(np.fft.rfft(x[:, idx] * _WAKE_WINDOW, n=N_FFT)) ** 2       # [N, T, 257]
    f = np.log(np.maximum(power @ _WAKE_BANK.T, 1e-10)).transpose(0, 2, 1)  # [N, 40, T]
    mean = f.mean(axis=(1, 2), keepdims=True)
    std = np.maximum(f.std(axis=(1, 2), ddof=1, keepdims=True), 1e-5)
    return ((f - mean) / std)[:, None].astype(np.float32)


# ---- intent / slot front end (torchaudio MelSpectrogram + AmplitudeToDB replica) ----
N_MELS, F_MIN, F_MAX, MAX_SAMPLES = 64, 20.0, 8000.0, 80000


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_bank():
    freqs = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    points = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))
    diff = np.diff(points)
    slopes = points[None, :] - freqs[:, None]
    return np.maximum(0.0, np.minimum(-slopes[:, :-2] / diff[:-1], slopes[:, 2:] / diff[1:]))  # [257, 64]


_CMD_BANK = _mel_bank()
_CMD_WINDOW = np.zeros(N_FFT)
_CMD_WINDOW[(N_FFT - WIN) // 2:(N_FFT - WIN) // 2 + WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)


def command_features(waveform: np.ndarray) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)[:MAX_SAMPLES]
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * _CMD_WINDOW, n=N_FFT)) ** 2
    db = 10.0 * np.log10(np.maximum(power @ _CMD_BANK, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    db = (db - db.mean()) / max(db.std(ddof=1), 1e-5)
    return db.T[None, None].astype(np.float32)
