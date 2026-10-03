from __future__ import annotations

import hashlib
import io
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


EXPECTED_SPLIT_COUNTS = {
    "train": 10_682,
    "test": 4_418,
    "holdout": 196,
    "numerals": 66_390,
}


def split_files(dataset_dir: Path, split: str) -> list[Path]:
    files = sorted((dataset_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found for split {split!r} in {dataset_dir / 'data'}")
    return files


def parquet_row_count(files: Sequence[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def validate_source_splits(dataset_dir: Path) -> dict[str, int]:
    counts = {name: parquet_row_count(split_files(dataset_dir, name)) for name in EXPECTED_SPLIT_COUNTS}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected source split counts: expected {EXPECTED_SPLIT_COUNTS}, got {counts}")
    return counts


def read_columns(files: Sequence[Path], columns: Sequence[str]) -> pa.Table:
    tables = [pq.read_table(path, columns=list(columns), memory_map=True) for path in files]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


@dataclass(frozen=True)
class SpeakerSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_speakers: frozenset[str]
    validation_speakers: frozenset[str]
    train_class_counts: dict[str, int]
    validation_class_counts: dict[str, int]


def make_speaker_disjoint_split(
    commands: Sequence[str],
    speaker_ids: Sequence[str],
    validation_fraction: float,
    seed: int,
    trials: int = 4096,
) -> SpeakerSplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(commands) != len(speaker_ids):
        raise ValueError("commands and speaker_ids must have equal length")

    classes = sorted(set(commands))
    class_to_index = {name: index for index, name in enumerate(classes)}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, speaker_id in enumerate(speaker_ids):
        key = speaker_id if speaker_id else "__MISSING_SPEAKER__"
        grouped[key].append(index)

    speakers = sorted(grouped)
    group_counts = np.zeros((len(speakers), len(classes)), dtype=np.int64)
    group_sizes = np.zeros(len(speakers), dtype=np.int64)
    for group_index, speaker in enumerate(speakers):
        indices = grouped[speaker]
        group_sizes[group_index] = len(indices)
        for row_index in indices:
            group_counts[group_index, class_to_index[commands[row_index]]] += 1

    total_counts = group_counts.sum(axis=0)
    target_counts = total_counts * validation_fraction
    target_size = len(commands) * validation_fraction
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_score = float("inf")
    for _ in range(trials):
        mask = rng.random(len(speakers)) < validation_fraction
        if not mask.any() or mask.all():
            continue
        validation_counts = group_counts[mask].sum(axis=0)
        train_counts = total_counts - validation_counts
        if np.any(validation_counts == 0) or np.any(train_counts == 0):
            continue
        validation_size = int(group_sizes[mask].sum())
        class_error = np.mean(np.abs(validation_counts - target_counts) / np.maximum(target_counts, 1.0))
        size_error = abs(validation_size - target_size) / target_size
        score = float(class_error + 0.5 * size_error)
        if score < best_score:
            best_score = score
            best_mask = mask.copy()

    if best_mask is None:
        raise RuntimeError("Could not construct a speaker-disjoint split containing every class")

    validation_speakers = frozenset(speaker for speaker, selected in zip(speakers, best_mask) if selected)
    train_speakers = frozenset(speakers) - validation_speakers
    train_indices = np.asarray(
        [i for i, speaker in enumerate(speaker_ids) if (speaker or "__MISSING_SPEAKER__") in train_speakers],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [i for i, speaker in enumerate(speaker_ids) if (speaker or "__MISSING_SPEAKER__") in validation_speakers],
        dtype=np.int64,
    )
    if train_speakers & validation_speakers:
        raise AssertionError("Speaker overlap detected")
    if len(train_indices) + len(validation_indices) != len(commands):
        raise AssertionError("Speaker split lost rows")

    train_class_counts = dict(Counter(commands[i] for i in train_indices))
    validation_class_counts = dict(Counter(commands[i] for i in validation_indices))
    return SpeakerSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_speakers=train_speakers,
        validation_speakers=validation_speakers,
        train_class_counts={name: train_class_counts[name] for name in classes},
        validation_class_counts={name: validation_class_counts[name] for name in classes},
    )


def split_from_validation_speakers(
    commands: Sequence[str], speaker_ids: Sequence[str], validation_speakers: frozenset[str]
) -> SpeakerSplit:
    """Reconstruct an exact split from a frozen validation-speaker assignment."""
    normalized_speakers = [speaker_id or "__MISSING_SPEAKER__" for speaker_id in speaker_ids]
    all_speakers = frozenset(normalized_speakers)
    unknown = validation_speakers - all_speakers
    if unknown:
        raise ValueError(f"Frozen split contains unknown speaker IDs: {sorted(unknown)}")
    train_speakers = all_speakers - validation_speakers
    train_indices = np.asarray(
        [index for index, speaker in enumerate(normalized_speakers) if speaker in train_speakers],
        dtype=np.int64,
    )
    validation_indices = np.asarray(
        [index for index, speaker in enumerate(normalized_speakers) if speaker in validation_speakers],
        dtype=np.int64,
    )
    if train_speakers & validation_speakers:
        raise AssertionError("Speaker overlap detected in frozen split")
    classes = sorted(set(commands))
    train_counts = Counter(commands[index] for index in train_indices)
    validation_counts = Counter(commands[index] for index in validation_indices)
    if set(train_counts) != set(classes) or set(validation_counts) != set(classes):
        raise ValueError("Frozen split does not contain every command class in both partitions")
    return SpeakerSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_speakers=train_speakers,
        validation_speakers=validation_speakers,
        train_class_counts={name: train_counts[name] for name in classes},
        validation_class_counts={name: validation_counts[name] for name in classes},
    )


class LogMelExtractor:
    def __init__(
        self,
        sample_rate: int = 16_000,
        max_seconds: float = 5.0,
        n_mels: int = 40,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_samples = round(sample_rate * max_seconds)
        self.win_length = win_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=20.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def __call__(self, wav_bytes: bytes) -> Tensor:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
                raise ValueError("Expected uncompressed 16-bit PCM WAV audio")
            if wav.getframerate() != self.sample_rate:
                raise ValueError(f"Expected {self.sample_rate} Hz audio, got {wav.getframerate()}")
            if wav.getnchannels() != 1:
                raise ValueError(f"Expected mono audio, got {wav.getnchannels()} channels")
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
        waveform = torch.from_numpy(pcm).to(torch.float32).div_(32768.0).unsqueeze(0)
        waveform = waveform[:, : self.max_samples]
        if waveform.shape[1] < self.win_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.win_length - waveform.shape[1]))
        features = self.to_db(self.mel(waveform))
        return (features - features.mean()) / features.std().clamp_min(1e-5)


class ParquetIntentDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        table: pa.Table,
        indices: Sequence[int],
        class_to_index: dict[str, int],
        extractor: LogMelExtractor,
    ) -> None:
        self.audio = table.column("audio")
        self.commands = table.column("command")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.class_to_index = class_to_index
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, int]:
        row = int(self.indices[item])
        audio = self.audio[row].as_py()
        command = self.commands[row].as_py()
        if audio is None or audio.get("bytes") is None:
            raise ValueError(f"Missing embedded audio bytes at row {row}")
        return self.extractor(audio["bytes"]), self.class_to_index[command]


def collate_variable_audio(batch: Sequence[tuple[Tensor, int]]) -> tuple[Tensor, Tensor, Tensor]:
    features, labels = zip(*batch)
    lengths = torch.tensor([feature.shape[-1] for feature in features], dtype=torch.long)
    time_major = [feature.squeeze(0).transpose(0, 1) for feature in features]
    padded = pad_sequence(time_major, batch_first=True).transpose(1, 2).unsqueeze(1)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def split_fingerprint(files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}\n".encode())
    return digest.hexdigest()
