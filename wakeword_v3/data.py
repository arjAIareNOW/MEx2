"""Only Speech Commands v0.02; official lists and binary mapping unchanged."""
import hashlib
import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Sampler
from wakeword.data import (
    SpeechCommandsBinary, discover_split, fit_length, load_random_segment,
    CLIP_SAMPLES, SAMPLE_RATE,
)

AUGMENTATION = {
    "time_shift_probability": 0.5, "time_shift_ms": [-80, 80],
    "time_shift_mode": "zero-filled non-circular",
    "gain_probability": 0.5, "gain_db": [-3.0, 3.0],
    "noise_probability": 0.6, "snr_db": [10.0, 25.0],
    "noise_source": "Speech Commands v0.02/_background_noise_ only",
    "training_only": True,
}
FEATURES = {
    "sample_rate": 16000, "mono": True, "pcm_bits": 16,
    "clip_samples": 16000, "n_mels": 40, "n_fft": 512,
    "win_length": 400, "hop_length": 160, "f_min": 20, "f_max": 7600,
    "window": "periodic Hann", "stft_center": False, "power": 2,
    "mel_bank": "V2 discrete-bin HTK triangular bank",
    "log": "natural logarithm, floor 1e-10",
    "normalization": "per-window mean / sample standard deviation, floor 1e-5",
    "evaluation_fit": "center pad/crop to 16000; no augmentation",
    "training_fit": "random pad/crop to 16000, as V2",
}


def inventory(root: Path):
    if not (root / "README.md").read_text().startswith("# Speech Commands Data Set v0.02"):
        raise ValueError("Expected Speech Commands v0.02")
    groups, speakers, report = {}, {}, {}
    expected = {"train": (84843, 1710, 2112), "validation": (9981, 195, 256),
                "test": (11005, 195, 250)}
    for split in expected:
        items = discover_split(root, split)
        if any("_nohash_" not in p.name for p, _ in items):
            raise ValueError("Speaker ID cannot be verified")
        groups[split] = items
        speakers[split] = {p.name.split("_nohash_")[0] for p, _ in items}
        counts = (len(items), sum(y for _, y in items), len(speakers[split]))
        if counts != expected[split]:
            raise ValueError(f"Frozen V2 membership mismatch: {split}: {counts}")
        payload = "\n".join(f"{p.relative_to(root)}\t{y}" for p, y in items)
        report[split] = dict(examples=counts[0], marvin=counts[1],
                             non_marvin=counts[0]-counts[1], speakers=counts[2],
                             membership_sha256=hashlib.sha256(payload.encode()).hexdigest(),
                             words=dict(sorted(Counter(p.parent.name for p, _ in items).items())))
    overlap = {f"{a}/{b}": len(speakers[a] & speakers[b]) for a, b in
               [("train", "validation"), ("train", "test"), ("validation", "test")]}
    if any(overlap.values()):
        raise ValueError(f"Speaker leakage: {overlap}")
    return {"dataset": "TensorFlow Speech Commands v0.02", "splits": report,
            "speaker_overlap": overlap,
            "official_list_sha256": {n: hashlib.sha256((root/n).read_bytes()).hexdigest()
                                     for n in ["validation_list.txt", "testing_list.txt"]},
            "test_metadata_only_before_final_selection": True}


class TrainingData(SpeechCommandsBinary):
    def __init__(self, root):
        super().__init__(root, "train", augment=True)
        if len(self.noise_paths) != 6 or self.extra_noise_paths:
            raise ValueError("Expected six built-in noise WAVs, no external noise")

    def _augment(self, waveform):
        if random.random() < 0.5:
            shift = random.randint(-1280, 1280)
            waveform = waveform.roll(shift)
            if shift > 0:
                waveform[:shift] = 0
            elif shift < 0:
                waveform[shift:] = 0
        if random.random() < 0.5:
            waveform = waveform * (10 ** (random.uniform(-3, 3) / 20))
        if random.random() < 0.6:
            noise = fit_length(load_random_segment(random.choice(self.noise_paths), CLIP_SAMPLES), True)
            gain = waveform.square().mean().sqrt().clamp_min(1e-4)
            gain /= noise.square().mean().sqrt().clamp_min(1e-4)
            waveform = waveform + noise * gain * 10 ** (-random.uniform(10, 25) / 20)
        return waveform.clamp(-1, 1)


class CyclingSampler(Sampler):
    """Every MARVIN once/epoch; 2 negatives each; ALL words remain eligible.

    Cycle a seeded negative permutation before repetition rather than repeatedly
    sampling a random subset. Complete negative coverage takes 25 epochs here.
    No validation/test examples or model-dependent hard-negative selection.
    """
    def __init__(self, labels, seed=1337, batch_size=64):
        self.positives = [i for i, y in enumerate(labels) if y]
        negatives = [i for i, y in enumerate(labels) if not y]
        g = torch.Generator().manual_seed(seed)
        self.negatives = [negatives[i] for i in torch.randperm(len(negatives), generator=g).tolist()]
        self.seed, self.batch_size, self.epoch = seed, batch_size, 0
        if not self.positives or len(self.negatives) < 2*len(self.positives):
            raise ValueError("Insufficient binary training data")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return math.ceil(3*len(self.positives)/self.batch_size)

    def __iter__(self):
        count = 2*len(self.positives)
        negatives = [self.negatives[(self.epoch*count+i) % len(self.negatives)] for i in range(count)]
        g = torch.Generator().manual_seed(self.seed+self.epoch)
        positives = [self.positives[i] for i in torch.randperm(len(self.positives), generator=g).tolist()]
        # Interleave 1:2, preserving balanced batches rather than epoch-only balance.
        items = [i for k, p in enumerate(positives) for i in (p, negatives[2*k], negatives[2*k+1])]
        for start in range(0, len(items), self.batch_size):
            batch = items[start:start+self.batch_size]
            yield [batch[i] for i in torch.randperm(len(batch), generator=g).tolist()]
