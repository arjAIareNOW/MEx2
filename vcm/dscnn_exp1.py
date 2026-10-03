from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pyarrow
import torch
import torchaudio
from torch import Tensor, nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import LogMelExtractor, ParquetIntentDataset, read_columns, split_files, split_fingerprint
from .metrics import classification_report, selected_confusions, write_class_metrics
from .model import parameter_count
from .train import (
    class_weights,
    evaluate,
    make_loader,
    prepare_split,
    resolve_device,
    set_reproducible,
    split_report,
    train_epoch,
    write_history,
)


EXPERIMENT = "Main VCM DS-CNN Exp1"
MODEL_NAME = "IntentDSCNN"
CRNN_EXP2_PARAMETERS = 984_308
EXPECTED_CLASSES = (
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF",
    "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT", "OUT_OF_SCOPE", "PAUSE",
    "PLAY_MUSIC", "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN",
    "VOLUME_UP", "WEATHER",
)


def conv_output_lengths(
    lengths: Tensor, kernel_size: int, stride: int, padding: int, dilation: int = 1
) -> Tensor:
    return torch.div(
        lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
        stride,
        rounding_mode="floor",
    ) + 1


def temporal_mask(features: Tensor, lengths: Tensor) -> Tensor:
    steps = torch.arange(features.shape[-1], device=features.device).unsqueeze(0)
    mask = (steps < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
    return features * mask.to(features.dtype)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int = 64, stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.depthwise_activation = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.BatchNorm2d(channels)
        self.pointwise_activation = nn.ReLU(inplace=True)
        self.temporal_stride = stride[1]

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        features = self.depthwise_activation(self.depthwise_norm(self.depthwise(features)))
        lengths = conv_output_lengths(lengths, 3, self.temporal_stride, 1)
        features = temporal_mask(features, lengths)
        features = self.pointwise_activation(self.pointwise_norm(self.pointwise(features)))
        return temporal_mask(features, lengths), lengths


class IntentDSCNN(nn.Module):
    """Keyword-spotting-style depthwise-separable CNN with masked pooling."""

    def __init__(self, num_classes: int = 20, dropout: float = 0.20) -> None:
        super().__init__()
        self.initial_conv = nn.Conv2d(
            1, 64, kernel_size=3, stride=(2, 2), padding=1, bias=False
        )
        self.initial_norm = nn.BatchNorm2d(64)
        self.initial_activation = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList(
            DepthwiseSeparableBlock(64, stride=(1, 1)) for _ in range(4)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def masked_global_average_pool(self, features: Tensor, lengths: Tensor) -> Tensor:
        masked = temporal_mask(features, lengths)
        valid_elements = (features.shape[2] * lengths).unsqueeze(1).to(features.dtype)
        return masked.sum(dim=(2, 3)) / valid_elements

    def forward(self, features: Tensor, frame_lengths: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1:3] != (1, 64):
            raise ValueError(f"Expected [batch, 1, 64, time], got {tuple(features.shape)}")
        x = self.initial_activation(self.initial_norm(self.initial_conv(features)))
        lengths = conv_output_lengths(frame_lengths, 3, 2, 1)
        x = temporal_mask(x, lengths)
        for block in self.blocks:
            x, lengths = block(x, lengths)
        pooled = self.masked_global_average_pool(x, lengths)
        return self.classifier(self.dropout(pooled))


def architecture_config() -> dict[str, Any]:
    return {
        "family": "Depthwise Separable CNN (established keyword-spotting design pattern)",
        "original_architecture_claimed": False,
        "initial_convolution": {
            "in_channels": 1, "out_channels": 64, "kernel": [3, 3],
            "stride": [2, 2], "padding": [1, 1], "bias": False,
        },
        "initial_normalization": "BatchNorm2d(64)",
        "depthwise_separable_blocks": 4,
        "depthwise": {
            "in_channels": 64, "out_channels": 64, "kernel": [3, 3],
            "stride": [1, 1], "padding": [1, 1], "groups": 64, "bias": False,
        },
        "pointwise": {
            "in_channels": 64, "out_channels": 64, "kernel": [1, 1],
            "stride": [1, 1], "padding": [0, 0], "bias": False,
        },
        "block_normalization": "BatchNorm2d(64) after depthwise and pointwise convolutions",
        "activation": "ReLU",
        "pooling": "length-aware masked global average over frequency and valid time only",
        "dropout": 0.20,
        "classifier": "Linear(64,20)",
        "output": "20 raw logits",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main VCM DS-CNN Exp1")
    parser.add_argument("--data", type=Path, default=Path("dataset/ai231-me2-voice-commands"))
    parser.add_argument("--output", type=Path, default=Path("models/vcm_dscnn_exp1"))
    parser.add_argument("--split-manifest", type=Path, default=Path("splits/main_vcm_speaker_split_v1.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23101)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    expected = {
        "epochs": 50, "batch_size": 64, "learning_rate": 0.001,
        "weight_decay": 0.0001, "patience": 8, "dropout": 0.20,
        "max_seconds": 5.0, "n_mels": 64, "workers": 0,
        "cpu_threads": 4, "seed": 23101,
    }
    changed = {key: (getattr(args, key), value) for key, value in expected.items() if getattr(args, key) != value}
    if changed:
        raise ValueError(f"DS-CNN Exp1 fixed configuration changed: {changed}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return f"not measured: {error}"
    output = (result.stdout or result.stderr).strip()
    return output if output else f"not measured: command exited {result.returncode}"


def collect_environment(device: torch.device) -> dict[str, Any]:
    gpu: dict[str, Any] = {
        "model": None, "count": torch.cuda.device_count(), "memory_bytes": None,
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu |= {"model": properties.name, "memory_bytes": properties.total_memory}
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "gpu": gpu,
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "torchaudio_version": str(torchaudio.__version__),
        "pyarrow_version": str(pyarrow.__version__),
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
    }


def write_environment_files(output: Path, environment: dict[str, Any]) -> None:
    (output / "environment.txt").write_text(json.dumps(environment, indent=2) + "\n")
    (output / "pip_freeze.txt").write_text(
        command_output([sys.executable, "-m", "pip", "freeze"]) + "\n"
    )
    (output / "gpu_info.txt").write_text(
        command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]) + "\n"
    )
    slurm = {key: value for key, value in sorted(os.environ.items()) if key.startswith("SLURM_")}
    (output / "slurm_info.txt").write_text(json.dumps(slurm, indent=2) + "\n")


def save_confusion_matrix(path: Path, matrix: list[list[int]], classes: list[str]) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    positions = np.arange(len(classes))
    axis.set(
        xticks=positions, yticks=positions, xticklabels=classes, yticklabels=classes,
        xlabel="Predicted command", ylabel="True command",
        title="Main VCM DS-CNN Exp1 — test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.inference_mode()
def evaluate_final_test(
    model: nn.Module,
    dataset: ParquetIntentDataset,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[Any, float, dict[str, Any], list[int], list[int], list[float]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    timings: list[float] = []
    loss_sum = 0.0
    warmup = min(50, len(dataset))
    for index in range(len(dataset)):
        features, label = dataset[index]
        features = features.unsqueeze(0).to(device)
        lengths = torch.tensor([features.shape[-1]], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        logits = model(features, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        if index >= warmup:
            timings.append(elapsed_ms)
        label_tensor = torch.tensor([label], dtype=torch.long, device=device)
        loss_sum += float(criterion(logits, label_tensor))
        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        targets.append(label)
        predictions.append(int(prediction.item()))
        confidences.append(float(confidence.item()))
    values = np.asarray(timings, dtype=np.float64)
    latency = {
        "label": "A100 GPU inference latency",
        "device": str(device),
        "batch_size": 1,
        "includes_audio_decode_or_log_mel": False,
        "warmup_iterations": warmup,
        "measured_iterations": len(timings),
        "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "minimum_ms": float(values.min()), "maximum_ms": float(values.max()),
    }
    return (
        classification_report(targets, predictions, class_names),
        loss_sum / len(dataset), latency, targets, predictions, confidences,
    )


def build_oos_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    confidences: Sequence[float],
    classes: Sequence[str],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    oos_index = classes.index("OUT_OF_SCOPE")
    report = classification_report(targets, predictions, classes)
    oos_class = next(row for row in report.per_class if row["class"] == "OUT_OF_SCOPE")
    false_rows: list[dict[str, Any]] = []
    breakdown: Counter[str] = Counter()
    correct = 0
    for index, (target, prediction, confidence) in enumerate(zip(targets, predictions, confidences)):
        if target != oos_index:
            continue
        if prediction == oos_index:
            correct += 1
            continue
        command = classes[prediction]
        breakdown[command] += 1
        false_rows.append({
            "test_row_index": index,
            "filename": records[index].get("file", ""),
            "speaker_id": records[index].get("speaker_id", ""),
            "source": records[index].get("source", ""),
            "transcript": records[index].get("transcript", ""),
            "predicted_command": command,
            "softmax_confidence": confidence,
        })
    total = sum(target == oos_index for target in targets)
    return {
        "oos_test_sample_count": total,
        "correct_oos_rejections": correct,
        "oos_precision": oos_class["precision"],
        "oos_recall": oos_class["recall"],
        "oos_f1": oos_class["f1"],
        "oos_falsely_classified_as_valid_commands": len(false_rows),
        "false_oos_breakdown_by_predicted_command": dict(sorted(breakdown.items())),
        "false_oos_samples": false_rows,
    }


def write_oos_csv(path: Path, oos: dict[str, Any]) -> None:
    fields = (
        "test_row_index", "filename", "speaker_id", "source", "transcript",
        "predicted_command", "softmax_confidence",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(oos["false_oos_samples"])


def indexed_counts(values: Sequence[str], indices: Sequence[int]) -> dict[str, int]:
    counts = Counter(values[int(index)] for index in indices)
    return dict(sorted(counts.items()))


def dataset_summary(
    train_metadata: Any,
    test_table: Any,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    train_speakers: set[str] | frozenset[str],
    validation_speakers: set[str] | frozenset[str],
    split_manifest: Path,
    train_files: Sequence[Path],
) -> dict[str, Any]:
    train_commands = train_metadata.column("command").to_pylist()
    train_sources = train_metadata.column("source").to_pylist()
    train_durations = train_metadata.column("duration_s").to_pylist()
    train_speaker_column = train_metadata.column("speaker_id").to_pylist()
    test_commands = test_table.column("command").to_pylist()
    test_sources = test_table.column("source").to_pylist()
    test_durations = test_table.column("duration_s").to_pylist()
    test_speakers = {value or "__MISSING_SPEAKER__" for value in test_table.column("speaker_id").to_pylist()}
    source_train_indices = np.arange(len(train_commands))
    oos_train = [i for i, value in enumerate(train_commands) if value == "OUT_OF_SCOPE"]
    oos_test = [i for i, value in enumerate(test_commands) if value == "OUT_OF_SCOPE"]
    all_speakers = {value or "__MISSING_SPEAKER__" for value in train_speaker_column} | test_speakers
    return {
        "scope": "source train plus final test; holdout and numerals excluded",
        "dataset_sources": sorted(set(train_sources) | set(test_sources)),
        "total_utterance_count": len(train_commands) + len(test_commands),
        "total_audio_duration_hours": float((sum(train_durations) + sum(test_durations)) / 3600.0),
        "unique_speaker_count": len(all_speakers),
        "class_count": len(set(train_commands)),
        "class_names": sorted(set(train_commands)),
        "class_distribution": {
            "source_train": indexed_counts(train_commands, source_train_indices),
            "experiment_train": indexed_counts(train_commands, train_indices),
            "validation": indexed_counts(train_commands, validation_indices),
            "test": indexed_counts(test_commands, np.arange(len(test_commands))),
        },
        "sample_counts": {
            "source_train": len(train_commands), "experiment_train": len(train_indices),
            "validation": len(validation_indices), "test": len(test_commands),
        },
        "speaker_counts": {
            "train": len(train_speakers), "validation": len(validation_speakers),
            "test": len(test_speakers),
        },
        "train_validation_speaker_overlap": len(set(train_speakers) & set(validation_speakers)),
        "oos_counts": {
            "source_train": len(oos_train),
            "experiment_train": sum(train_commands[int(i)] == "OUT_OF_SCOPE" for i in train_indices),
            "validation": sum(train_commands[int(i)] == "OUT_OF_SCOPE" for i in validation_indices),
            "test": len(oos_test),
        },
        "oos_source_counts": {
            "source_train": indexed_counts(train_sources, oos_train),
            "test": indexed_counts(test_sources, oos_test),
        },
        "frozen_split": {
            "manifest": str(split_manifest),
            "manifest_sha256": sha256_file(split_manifest),
            "train_parquet_fingerprint": split_fingerprint(train_files),
        },
    }


def summary_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2) + "\n"


def run_preflight(args: argparse.Namespace) -> None:
    source_counts, classes, mapping, split, train_files, special_counts = prepare_split(args)
    model = IntentDSCNN()
    extractor = LogMelExtractor(
        sample_rate=16_000, max_seconds=5.0, n_mels=64,
        n_fft=512, win_length=400, hop_length=160,
    )
    table = read_columns(train_files, ("audio", "command"))
    dataset = ParquetIntentDataset(table, split.train_indices[:4], mapping, extractor)
    from torch.utils.data import DataLoader
    from .data import collate_variable_audio
    features, lengths, labels = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_variable_audio)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer.zero_grad(set_to_none=True)
    logits = model(features, lengths)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    print(json.dumps(split_report(source_counts, classes, split, special_counts) | {
        "mode": "CPU preflight", "actual_features": list(features.shape),
        "actual_lengths": lengths.tolist(), "logits": list(logits.shape),
        "loss": float(loss.detach()), "trainable_parameters": parameter_count(model),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "output_directory_absent": not args.output.exists(),
        "test_opened": False, "holdout_opened": False, "numerals_opened": False,
    }, indent=2))


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_reproducible(args.seed, args.cpu_threads)
    if args.preflight:
        if args.output.exists():
            raise FileExistsError(f"Refusing existing output during preflight: {args.output}")
        run_preflight(args)
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment output: {args.output}")
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Full DS-CNN Exp1 training requires CUDA")
    if "A100" not in torch.cuda.get_device_properties(device).name:
        raise RuntimeError("DS-CNN Exp1 requires the requested NVIDIA A100 GPU")

    source_counts, classes, mapping, split, train_files, special_counts = prepare_split(args)
    if tuple(classes) != EXPECTED_CLASSES:
        raise ValueError("Frozen 20-class mapping changed")
    if (len(split.train_indices), len(split.validation_indices)) != (9_119, 1_563):
        raise ValueError("Frozen split is not 9,119 train / 1,563 validation")
    if (len(split.train_speakers), len(split.validation_speakers)) != (270, 47):
        raise ValueError("Frozen speaker counts changed")
    if split.train_speakers & split.validation_speakers:
        raise ValueError("Frozen split has speaker overlap")

    start_utc = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = collect_environment(device)
    write_environment_files(args.output, environment)
    split_details = split_report(source_counts, classes, split, special_counts)
    model = IntentDSCNN(len(classes), args.dropout).to(device)
    trainable = parameter_count(model)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    reduction = CRNN_EXP2_PARAMETERS - trainable
    configuration = {
        "experiment": EXPERIMENT,
        "model": MODEL_NAME,
        "architecture": architecture_config(),
        "audio": {
            "sample_rate": 16_000, "channels": 1, "max_seconds": 5.0,
            "variable_duration": True, "n_mels": 64, "n_fft": 512,
            "window_length_samples": 400, "window_length_ms": 25,
            "hop_length_samples": 160, "hop_length_ms": 10,
            "frequency_range_hz": [20.0, 8000.0],
            "feature_normalization": "per-utterance mean/std over log-mel spectrogram",
        },
        "training": {
            "optimizer": "AdamW", "initial_learning_rate": 0.001,
            "weight_decay": 0.0001, "batch_size": 64, "maximum_epochs": 50,
            "early_stopping_patience": 8, "scheduler": "CosineAnnealingLR",
            "seed": 23101, "workers": 0, "cpu_threads": 4,
            "loss": "training-only inverse-frequency weighted CrossEntropyLoss",
            "checkpoint_selection": "validation macro-F1", "augmentation": "none",
        },
        "split": split_details,
        "train_speaker_ids": sorted(split.train_speakers),
        "validation_speaker_ids": sorted(split.validation_speakers),
        "frozen_split_manifest": str(args.split_manifest),
        "frozen_split_manifest_sha256": sha256_file(args.split_manifest),
        "train_parquet_fingerprint": split_fingerprint(train_files),
        "efficiency": {
            "trainable_parameters": trainable, "total_parameters": total_parameters,
            "macs": "not measured", "flops": "not measured",
            "comparison_model": "CRNN Exp2", "comparison_parameters": CRNN_EXP2_PARAMETERS,
            "absolute_parameter_reduction": reduction,
            "percentage_parameter_reduction": reduction / CRNN_EXP2_PARAMETERS * 100.0,
            "raspberry_pi_speed_claim": "not measured; requires Raspberry Pi 5 benchmark",
        },
        "environment_at_start": environment,
    }
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (args.output / "class_to_index.json").write_text(json.dumps(mapping, indent=2) + "\n")

    extractor = LogMelExtractor(
        sample_rate=16_000, max_seconds=5.0, n_mels=64,
        n_fft=512, win_length=400, hop_length=160,
    )
    # Only source train is opened before best.pt selection and reload.
    train_table = read_columns(train_files, ("audio", "command"))
    train_set = ParquetIntentDataset(train_table, split.train_indices, mapping, extractor)
    validation_set = ParquetIntentDataset(train_table, split.validation_indices, mapping, extractor)
    train_loader = make_loader(train_set, args, device, shuffle=True)
    validation_loader = make_loader(validation_set, args, device, shuffle=False)
    weights = class_weights(classes, split.train_class_counts, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    checkpoint_path = args.output / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        validation_report, validation_loss = evaluate(
            model, validation_loader, criterion, device, classes
        )
        row = {
            "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss, "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            **{f"validation_{key}": value for key, value in validation_report.summary.items()},
        }
        history.append(row)
        write_history(args.output / "training_history.csv", history)
        print(json.dumps(row), flush=True)
        macro_f1 = validation_report.summary["macro_f1"]
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(), "model_name": MODEL_NAME,
                "model_config": {"num_classes": len(classes), "dropout": args.dropout},
                "class_to_index": mapping, "epoch": epoch,
                "validation_report": {
                    **validation_report.summary, "loss": validation_loss,
                    "per_class": validation_report.per_class,
                    "confusion_matrix": validation_report.confusion_matrix,
                },
                "configuration": configuration,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            stopped_early = True
            break

    training_end_utc = datetime.now(timezone.utc)
    training_wall_seconds = time.perf_counter() - wall_start
    training_peak_allocated = torch.cuda.max_memory_allocated(device)
    training_peak_reserved = torch.cuda.max_memory_reserved(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # First access to test: after validation selection and authoritative best.pt reload.
    test_table = read_columns(
        split_files(args.data, "test"),
        ("audio", "command", "file", "speaker_id", "source", "transcript", "duration_s"),
    )
    test_commands = test_table.column("command").to_pylist()
    if len(test_table) != 4_418 or test_commands.count("OUT_OF_SCOPE") != 47:
        raise ValueError("Expected updated test with 4,418 samples including 47 OUT_OF_SCOPE")
    if set(test_commands) - set(classes):
        raise ValueError("Test contains a label outside the frozen mapping")
    test_set = ParquetIntentDataset(test_table, np.arange(len(test_table)), mapping, extractor)
    test_report, test_loss, latency, targets, predictions, confidences = evaluate_final_test(
        model, test_set, criterion, device, classes
    )
    records = [{
        "file": test_table.column("file")[index].as_py(),
        "speaker_id": test_table.column("speaker_id")[index].as_py(),
        "source": test_table.column("source")[index].as_py(),
        "transcript": test_table.column("transcript")[index].as_py(),
    } for index in range(len(test_table))]
    oos = build_oos_metrics(targets, predictions, confidences, classes, records)
    write_oos_csv(args.output / "oos_false_classifications.csv", oos)
    write_class_metrics(args.output / "class_metrics.csv", {"test": test_report})
    save_confusion_matrix(args.output / "confusion_matrix.png", test_report.confusion_matrix, classes)

    checkpoint_bytes = checkpoint_path.stat().st_size
    train_metadata = read_columns(train_files, ("command", "speaker_id", "source", "duration_s"))
    data_summary = dataset_summary(
        train_metadata, test_table, split.train_indices, split.validation_indices,
        split.train_speakers, split.validation_speakers, args.split_manifest, train_files,
    )
    training_result = {
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_accuracy": checkpoint["validation_report"]["accuracy"],
        "best_validation_macro_f1": checkpoint["validation_report"]["macro_f1"],
        "start_timestamp_utc": start_utc.isoformat(),
        "end_timestamp_utc": training_end_utc.isoformat(),
        "total_training_wall_clock_seconds": training_wall_seconds,
        "peak_cuda_memory_allocated_bytes": training_peak_allocated,
        "peak_cuda_memory_reserved_bytes": training_peak_reserved,
        "stopped_early": stopped_early,
    }
    efficiency = configuration["efficiency"] | {
        "checkpoint_size_bytes": checkpoint_bytes,
        "checkpoint_size_mb": checkpoint_bytes / 1_000_000.0,
    }
    final_test = {
        **test_report.summary, "loss": test_loss, "per_class": test_report.per_class,
        "confusion_matrix": test_report.confusion_matrix,
        "selected_confusions": selected_confusions(test_report.confusion_matrix, mapping),
        "oos": oos, "a100_gpu_inference_latency": latency,
    }
    metrics = {
        "experiment": EXPERIMENT, "configuration": configuration,
        "training_result": training_result, "best_validation": checkpoint["validation_report"],
        "final_test": final_test, "efficiency": efficiency, "dataset": data_summary,
        "holdout_used": False, "numerals_used": False,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    submission = {
        "model": {
            "family": "DS-CNN", "experiment_name": EXPERIMENT,
            "architecture": architecture_config(), "sample_rate": 16_000,
            "log_mel": configuration["audio"], "channels": 64, "ds_block_count": 4,
            "dropout": 0.20, "number_of_classes": 20, **efficiency,
        },
        "dataset": data_summary,
        "training_environment": collect_environment(device),
        "training_configuration": configuration["training"],
        "training_result": training_result,
        "final_test": final_test,
        "onnx": {
            "authoritative_checkpoint": "best.pt",
            "export_utility": "python -m vcm.export_dscnn_exp1_onnx",
            "status": "not exported during training; optional onnx/onnxruntime dependencies required",
        },
        "raspberry_pi": {
            "benchmark_script": "scripts/benchmark_vcm_dscnn_exp1_rpi.py",
            "performance_claim": "not measured",
        },
    }
    (args.output / "submission_summary.json").write_text(json.dumps(submission, indent=2) + "\n")
    (args.output / "submission_summary.txt").write_text(summary_text(submission))
    (args.output / "COMPLETED").write_text("completed without holdout or numerals evaluation\n")
    print(json.dumps({"completed": True, "best_epoch": best_epoch, "test": test_report.summary}))


if __name__ == "__main__":
    main()
