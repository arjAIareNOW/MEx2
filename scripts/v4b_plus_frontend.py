"""NumPy-only replica of the V2/V4-B/V4-B+ training front end (no torch needed on the Pi).

torchaudio MelSpectrogram(16k, n_fft 512, win 400, hop 160, 20-8000 Hz, 64 mels, power 2,
center=True reflect pad, periodic Hann zero-padded to n_fft, HTK mel, no norm)
-> AmplitudeToDB(power, top_db 80) -> per-utterance mean / unbiased std (floor 1e-5).
Input: mono PCM16 16 kHz bytes; at most 80,000 samples; short clips padded to 400.
Output: float32 [1, 1, 64, T].
"""
import io
import wave

import numpy as np

SR, N_FFT, WIN, HOP, N_MELS, F_MIN, F_MAX, MAX_SAMPLES = 16000, 512, 400, 160, 64, 20.0, 8000.0, 80000


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_bank():
    freqs = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    points = _mel_to_hz(np.linspace(_hz_to_mel(F_MIN), _hz_to_mel(F_MAX), N_MELS + 2))
    diff = np.diff(points)
    slopes = points[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float64)  # [freq, mels]


_BANK = _mel_bank()
_WINDOW = np.zeros(N_FFT)
_WINDOW[(N_FFT - WIN) // 2:(N_FFT - WIN) // 2 + WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)


def decode_wav(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            raise ValueError('Expected mono 16-kHz PCM16 WAV')
        pcm = np.frombuffer(w.readframes(min(w.getnframes(), MAX_SAMPLES)), dtype='<i2')
    return pcm.astype(np.float32) / 32768.0


def log_mel(waveform: np.ndarray) -> np.ndarray:
    x = waveform[:MAX_SAMPLES].astype(np.float64)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode='reflect')
    frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * _WINDOW, n=N_FFT)) ** 2  # [T, freq]
    mel = power @ _BANK  # [T, mels]
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    db = (db - db.mean()) / max(db.std(ddof=1), 1e-5)
    return db.T[None, None].astype(np.float32)  # [1,1,64,T]
