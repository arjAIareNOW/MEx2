from __future__ import annotations

import math
import random
import wave
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, Sampler


SAMPLE_RATE = 16_000
CLIP_SAMPLES = SAMPLE_RATE


def _read_split_file(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required official split file is missing: {path}")
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def discover_split(root: Path, split: str) -> list[tuple[Path, int]]:
    """Return (wav path, binary label) without writing into the dataset."""
    root = root.resolve()
    validation = _read_split_file(root / "validation_list.txt")
    testing = _read_split_file(root / "testing_list.txt")
    overlap = validation & testing
    if overlap:
        raise ValueError(f"Official split files overlap ({len(overlap)} entries)")

    wavs = sorted(
        path
        for path in root.glob("*/*.wav")
        if path.parent.name != "_background_noise_"
    )
    if not wavs:
        raise FileNotFoundError(f"No command WAV files found under {root}")

    items: list[tuple[Path, int]] = []
    for path in wavs:
        relative = path.relative_to(root).as_posix()
        assigned = (
            "validation"
            if relative in validation
            else "test"
            if relative in testing
            else "train"
        )
        if assigned == split:
            items.append((path, int(path.parent.name.lower() == "marvin")))
    if not items:
        raise ValueError(f"Unknown or empty split: {split}")
    return items


def load_wav(path: Path) -> Tensor:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_rate != SAMPLE_RATE or channels != 1 or sample_width != 2:
        raise ValueError(
            f"Expected mono 16-bit {SAMPLE_RATE} Hz PCM, got "
            f"{channels} channel(s), {sample_width * 8}-bit, {sample_rate} Hz: {path}"
        )
    return torch.frombuffer(bytearray(frames), dtype=torch.int16).float().div_(32768.0)


def load_random_segment(path: Path, frame_count: int) -> Tensor:
    """Read only a random section of a long background-noise WAV."""
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        available = wav.getnframes()
        if sample_rate != SAMPLE_RATE or channels != 1 or sample_width != 2:
            raise ValueError(f"Unexpected background-noise WAV format: {path}")
        start = random.randint(0, max(0, available - frame_count))
        wav.setpos(start)
        frames = wav.readframes(min(frame_count, available))
    return torch.frombuffer(bytearray(frames), dtype=torch.int16).float().div_(32768.0)


def fit_length(waveform: Tensor, training: bool) -> Tensor:
    difference = CLIP_SAMPLES - waveform.numel()
    if difference >= 0:
        left = random.randint(0, difference) if training else difference // 2
        return F.pad(waveform, (left, difference - left))
    start = random.randint(0, -difference) if training else (-difference) // 2
    return waveform[start : start + CLIP_SAMPLES]


class LogMelFeatures(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 40,
        f_min: float = 20.0,
        f_max: float = 7_600.0,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer(
            "mel_bank",
            self._make_mel_bank(sample_rate, n_fft, n_mels, f_min, f_max),
            persistent=False,
        )

    @staticmethod
    def _make_mel_bank(
        sample_rate: int,
        n_fft: int,
        n_mels: int,
        f_min: float,
        f_max: float,
    ) -> Tensor:
        def hz_to_mel(value: float) -> float:
            return 2595.0 * math.log10(1.0 + value / 700.0)

        def mel_to_hz(value: Tensor) -> Tensor:
            return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)

        mel_points = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
        bins = torch.floor((n_fft + 1) * mel_to_hz(mel_points) / sample_rate).long()
        bins.clamp_(0, n_fft // 2)
        bank = torch.zeros(n_mels, n_fft // 2 + 1)
        for index in range(n_mels):
            left, center, right = bins[index : index + 3].tolist()
            if center > left:
                bank[index, left:center] = torch.arange(center - left) / (center - left)
            if right > center:
                bank[index, center:right] = torch.arange(right - center, 0, -1) / (right - center)
        return bank

    def forward(self, waveform: Tensor) -> Tensor:
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs().square()
        features = torch.matmul(self.mel_bank, spectrum).clamp_min_(1e-10).log()
        return (features - features.mean()) / features.std().clamp_min(1e-5)


class SpeechCommandsBinary(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        root: Path,
        split: str,
        augment: bool = False,
        extra_noise_dir: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.split = split
        self.augment = augment
        self.items = discover_split(self.root, split)
        self.features = LogMelFeatures()
        self.speech_commands_noise_paths = sorted((self.root / "_background_noise_").glob("*.wav"))
        self.extra_noise_paths: list[Path] = []
        if extra_noise_dir is not None:
            extra_noise_dir = extra_noise_dir.resolve()
            if not extra_noise_dir.is_dir():
                raise FileNotFoundError(f"Extra noise directory is missing: {extra_noise_dir}")
            self.extra_noise_paths = sorted(extra_noise_dir.rglob("*.wav"))
            if not self.extra_noise_paths:
                raise FileNotFoundError(f"No WAV noise files found under {extra_noise_dir}")
        self.noise_paths = self.speech_commands_noise_paths + self.extra_noise_paths

    @property
    def noise_source_counts(self) -> dict[str, int]:
        return {
            "speech_commands": len(self.speech_commands_noise_paths),
            "extra": len(self.extra_noise_paths),
            "combined": len(self.noise_paths),
        }

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, waveform: Tensor) -> Tensor:
        if random.random() < 0.5:
            max_shift = int(0.1 * SAMPLE_RATE)
            shift = random.randint(-max_shift, max_shift)
            waveform = torch.roll(waveform, shift)
            if shift > 0:
                waveform[:shift] = 0
            elif shift < 0:
                waveform[shift:] = 0

        if self.noise_paths and random.random() < 0.8:
            noise = load_random_segment(random.choice(self.noise_paths), CLIP_SAMPLES)
            noise = fit_length(noise, training=True)
            signal_rms = waveform.square().mean().sqrt().clamp_min(1e-4)
            noise_rms = noise.square().mean().sqrt().clamp_min(1e-4)
            snr_db = random.uniform(5.0, 20.0)
            waveform = waveform + noise * (signal_rms / noise_rms) * (10.0 ** (-snr_db / 20.0))
        return waveform.clamp_(-1.0, 1.0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, label = self.items[index]
        waveform = fit_length(load_wav(path), training=self.augment)
        if self.augment:
            waveform = self._augment(waveform)
        features = self.features(waveform).unsqueeze(0)
        return features, torch.tensor(float(label))


class BalancedBinaryBatchSampler(Sampler[list[int]]):
    """Build 1:N batches using all positives and rotating negative samples."""

    def __init__(
        self,
        labels: Sequence[int],
        negatives_per_positive: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.positive = [index for index, label in enumerate(labels) if label == 1]
        self.negative = [index for index, label in enumerate(labels) if label == 0]
        if negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be at least 1")
        if batch_size < negatives_per_positive + 1:
            raise ValueError(f"batch_size must be at least {negatives_per_positive + 1}")
        self.negatives_per_positive = negatives_per_positive
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        if not self.positive or not self.negative:
            raise ValueError("Both positive and negative examples are required")
        if self.negative_count > len(self.negative):
            raise ValueError(
                f"Requested {self.negative_count} negatives per epoch, but only "
                f"{len(self.negative)} are available"
            )

    @property
    def positive_count(self) -> int:
        return len(self.positive)

    @property
    def negative_count(self) -> int:
        return self.negatives_per_positive * self.positive_count

    @property
    def effective_counts(self) -> dict[str, int]:
        return {
            "marvin": self.positive_count,
            "non_marvin": self.negative_count,
            "total": self.positive_count + self.negative_count,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        positive_order = torch.randperm(self.positive_count, generator=generator).tolist()
        negative_order = torch.randperm(len(self.negative), generator=generator)[: self.negative_count].tolist()
        shuffled_positive = [self.positive[index] for index in positive_order]
        shuffled_negative = [self.negative[index] for index in negative_order]

        batches: list[list[int]] = []
        total_count = self.positive_count + self.negative_count
        used_positive = 0
        used_negative = 0
        for start in range(0, total_count, self.batch_size):
            end = min(start + self.batch_size, total_count)
            # Rounded cumulative quotas keep each batch close to the requested
            # ratio while preserving the exact epoch-level class counts.
            target_positive = round(end / (self.negatives_per_positive + 1))
            positive_count = target_positive - used_positive
            batch_count = end - start
            negative_count = batch_count - positive_count
            positives = shuffled_positive[used_positive : used_positive + positive_count]
            negatives = shuffled_negative[used_negative : used_negative + negative_count]
            batch = positives + negatives
            batch_order = torch.randperm(len(batch), generator=generator).tolist()
            batches.append([batch[index] for index in batch_order])
            used_positive += positive_count
            used_negative += negative_count
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil((self.positive_count + self.negative_count) / self.batch_size)
