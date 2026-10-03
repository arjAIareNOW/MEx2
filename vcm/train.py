from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow
import torch
import torchaudio
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import (
    EXPECTED_SPLIT_COUNTS,
    LogMelExtractor,
    ParquetIntentDataset,
    collate_variable_audio,
    parquet_row_count,
    read_columns,
    split_from_validation_speakers,
    split_files,
    split_fingerprint,
)
from .metrics import (
    ClassificationReport,
    classification_report,
    save_confusion_matrix,
    selected_confusions,
    write_class_metrics,
)
from .model import IntentCNN, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main VCM CNN Experiment 1: command/intent recognition")
    parser.add_argument("--data", type=Path, default=Path("dataset/ai231-me2-voice-commands"))
    parser.add_argument("--output", type=Path, default=Path("models/vcm_cnn_exp1"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("splits/main_vcm_speaker_split_v1.json"),
        help="Frozen train/validation speaker assignment shared by CNN/RNN/CRNN experiments",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=23101)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight", action="store_true", help="Validate data/split and run one CPU/GPU step only")
    parser.add_argument("--smoke-samples", type=int, default=4)
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def set_reproducible(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def json_ready_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def prepare_split(args: argparse.Namespace) -> tuple[dict[str, int], list[str], dict[str, int], Any, Any, dict[str, Any]]:
    source_counts = dict(EXPECTED_SPLIT_COUNTS)
    train_files = split_files(args.data, "train")
    train_count = parquet_row_count(train_files)
    if train_count != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['train']} train rows, got {train_count}")
    metadata = read_columns(train_files, ("command", "speaker_id", "bucket"))
    commands = metadata.column("command").to_pylist()
    speakers = metadata.column("speaker_id").to_pylist()
    buckets = metadata.column("bucket").to_pylist()
    class_names = sorted(set(commands))
    if len(class_names) != 20 or "OUT_OF_SCOPE" not in class_names:
        raise ValueError(f"Expected 20 command classes including OUT_OF_SCOPE, got {class_names}")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    frozen_manifest = json.loads(args.split_manifest.read_text())
    expected_fingerprint = split_fingerprint(train_files)
    if frozen_manifest.get("train_parquet_fingerprint") != expected_fingerprint:
        raise ValueError("Frozen split manifest does not match the current train Parquet files")
    speaker_split = split_from_validation_speakers(
        commands, speakers, frozenset(frozen_manifest["validation_speaker_ids"])
    )
    if (
        len(speaker_split.train_indices) != frozen_manifest["train_sample_count"]
        or len(speaker_split.validation_indices) != frozen_manifest["validation_sample_count"]
    ):
        raise ValueError("Frozen split sample counts do not match the reconstructed split")
    overlap = speaker_split.train_speakers & speaker_split.validation_speakers
    if overlap:
        raise AssertionError(f"Speaker overlap: {sorted(overlap)}")
    out_of_scope = np.asarray([command == "OUT_OF_SCOPE" for command in commands])
    off_schema_slot = np.asarray([bucket.endswith("(other slot value)") for bucket in buckets])
    if int(out_of_scope.sum()) != 201 or int(off_schema_slot.sum()) != 158:
        raise ValueError(
            "Expected 201 OUT_OF_SCOPE and 158 off-schema-slot train rows, got "
            f"{int(out_of_scope.sum())} and {int(off_schema_slot.sum())}"
        )
    special_counts = {
        "source_train": {"out_of_scope": 201, "off_schema_slot": 158},
        "experiment_train": {
            "out_of_scope": int(out_of_scope[speaker_split.train_indices].sum()),
            "off_schema_slot": int(off_schema_slot[speaker_split.train_indices].sum()),
        },
        "validation": {
            "out_of_scope": int(out_of_scope[speaker_split.validation_indices].sum()),
            "off_schema_slot": int(off_schema_slot[speaker_split.validation_indices].sum()),
        },
    }
    return source_counts, class_names, class_to_index, speaker_split, train_files, special_counts


def split_report(
    source_counts: dict[str, int],
    class_names: list[str],
    speaker_split: Any,
    special_counts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_dataset_counts": source_counts,
        "experiment_counts": {
            "train": len(speaker_split.train_indices),
            "validation": len(speaker_split.validation_indices),
            "test": source_counts["test"],
        },
        "excluded": {"holdout": source_counts["holdout"], "numerals": source_counts["numerals"]},
        "num_classes": len(class_names),
        "classes": class_names,
        "train_speakers": len(speaker_split.train_speakers),
        "validation_speakers": len(speaker_split.validation_speakers),
        "speaker_overlap": len(speaker_split.train_speakers & speaker_split.validation_speakers),
        "train_class_counts": speaker_split.train_class_counts,
        "validation_class_counts": speaker_split.validation_class_counts,
        "retained_special_rows": special_counts,
    }


def make_loader(dataset: ParquetIntentDataset, args: argparse.Namespace, device: torch.device, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_variable_audio,
        generator=generator,
    )


def class_weights(class_names: list[str], counts: dict[str, int], device: torch.device) -> Tensor:
    raw = torch.tensor([1.0 / counts[name] for name in class_names], dtype=torch.float32)
    return (raw / raw.mean()).to(device)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[ClassificationReport, float]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    loss_sum = 0.0
    examples = 0
    for features, lengths, labels in loader:
        features = features.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        device_labels = labels.to(device, non_blocking=True)
        logits = model(features, lengths)
        loss_sum += float(criterion(logits, device_labels)) * labels.numel()
        examples += labels.numel()
        targets.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return classification_report(targets, predictions, class_names), loss_sum / examples


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for features, lengths, labels in loader:
        features = features.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum())
        examples += labels.numel()
    return loss_sum / examples, correct / examples


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fieldnames = (
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_precision",
        "validation_macro_recall",
        "validation_macro_f1",
        "validation_weighted_f1",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


@torch.inference_mode()
def evaluate_test_once(
    model: nn.Module,
    dataset: ParquetIntentDataset,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> tuple[ClassificationReport, float, dict[str, Any]]:
    """Compute final test metrics and batch-one forward latency in one test pass."""
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    timings: list[float] = []
    loss_sum = 0.0
    warmup = min(20, len(dataset))
    for index in range(len(dataset)):
        features, label = dataset[index]
        features = features.unsqueeze(0).to(device)
        lengths = torch.tensor([features.shape[-1]], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        logits = model(features, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        if index >= warmup:
            timings.append(elapsed_ms)
        device_label = torch.tensor([label], dtype=torch.long, device=device)
        loss_sum += float(criterion(logits, device_label))
        targets.append(label)
        predictions.append(int(logits.argmax(dim=1).item()))
    values = np.asarray(timings, dtype=np.float64)
    latency = {
        "label": "model forward-pass latency on the training/test machine",
        "device": str(device),
        "batch_size": 1,
        "samples_timed": len(timings),
        "warmup_samples": warmup,
        "includes_audio_decode_or_log_mel": False,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }
    return classification_report(targets, predictions, class_names), loss_sum / len(dataset), latency


def run_preflight(args: argparse.Namespace, device: torch.device) -> None:
    source_counts, class_names, class_to_index, speaker_split, train_files, special_counts = prepare_split(args)
    model = IntentCNN(len(class_names), args.dropout).to(device)
    extractor = LogMelExtractor(max_seconds=args.max_seconds, n_mels=args.n_mels)
    table = read_columns(train_files, ("audio", "command"))
    smoke_indices = speaker_split.train_indices[: args.smoke_samples]
    smoke_set = ParquetIntentDataset(table, smoke_indices, class_to_index, extractor)
    loader = DataLoader(smoke_set, batch_size=len(smoke_set), collate_fn=collate_variable_audio)
    features, lengths, labels = next(iter(loader))
    features, lengths, labels = features.to(device), lengths.to(device), labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    logits = model(features, lengths)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    checkpoint_buffer = io.BytesIO()
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {"num_classes": len(class_names), "dropout": args.dropout},
            "class_to_index": class_to_index,
        },
        checkpoint_buffer,
    )
    checkpoint_buffer.seek(0)
    torch.load(checkpoint_buffer, map_location=device, weights_only=True)
    report = split_report(source_counts, class_names, speaker_split, special_counts) | {
        "mode": "preflight_smoke_test",
        "device": str(device),
        "smoke_batch_shape": list(features.shape),
        "smoke_logits_shape": list(logits.shape),
        "smoke_loss": float(loss),
        "checkpoint_round_trip": True,
        "trainable_parameters": parameter_count(model),
        "source_split_files": {name: [str(path) for path in split_files(args.data, name)] for name in EXPECTED_SPLIT_COUNTS},
        "holdout_loaded": False,
        "numerals_loaded": False,
        "test_loaded": False,
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience < 0 or args.cpu_threads <= 0:
        raise ValueError("epochs, batch_size, and cpu_threads must be positive; patience must be non-negative")
    set_reproducible(args.seed, args.cpu_threads)
    device = resolve_device(args.device)
    if args.preflight:
        run_preflight(args, device)
        return
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {args.output}")

    source_counts, class_names, class_to_index, speaker_split, train_files, special_counts = prepare_split(args)
    args.output.mkdir(parents=True, exist_ok=False)
    split_details = split_report(source_counts, class_names, speaker_split, special_counts)
    (args.output / "class_to_index.json").write_text(json.dumps(class_to_index, indent=2) + "\n")
    split_manifest = split_details | {
        "seed": args.seed,
        "validation_fraction_target": args.validation_fraction,
        "frozen_split_source": str(args.split_manifest),
        "train_speaker_ids": sorted(speaker_split.train_speakers),
        "validation_speaker_ids": sorted(speaker_split.validation_speakers),
        "train_parquet_fingerprint": split_fingerprint(train_files),
    }
    (args.output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")

    extractor = LogMelExtractor(max_seconds=args.max_seconds, n_mels=args.n_mels)
    train_table = read_columns(train_files, ("audio", "command"))
    train_set = ParquetIntentDataset(train_table, speaker_split.train_indices, class_to_index, extractor)
    validation_set = ParquetIntentDataset(train_table, speaker_split.validation_indices, class_to_index, extractor)
    train_loader = make_loader(train_set, args, device, shuffle=True)
    validation_loader = make_loader(validation_set, args, device, shuffle=False)

    model = IntentCNN(len(class_names), args.dropout).to(device)
    parameters = parameter_count(model)
    weights = class_weights(class_names, speaker_split.train_class_counts, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint_path = args.output / "best.pt"
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    configuration = {
        "experiment": "Main VCM CNN Experiment 1",
        "task": "20-class command/intent recognition",
        "model": "IntentCNN",
        "architecture": "Conv2D(32)-BN-ReLU-Pool -> Conv2D(64)-BN-ReLU-Pool -> Conv2D(128)-BN-ReLU-Pool -> masked global average pool -> dropout -> linear(20)",
        "class_imbalance": "inverse-frequency weighted CrossEntropyLoss from the speaker-disjoint training subset",
        "audio": {
            "sample_rate": 16_000,
            "channels": 1,
            "format": "16-bit PCM WAV embedded in Parquet audio.bytes",
            "max_seconds": args.max_seconds,
            "n_mels": args.n_mels,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "variable_duration": True,
            "batch_padding": "right padded with temporal mask used by global pooling",
        },
        "arguments": json_ready_args(args),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torchaudio": str(torchaudio.__version__),
            "pyarrow": str(pyarrow.__version__),
            "numpy": str(np.__version__),
        },
        "device": str(device),
        "trainable_parameters": parameters,
        "class_weights": {name: float(weights[index].cpu()) for index, name in enumerate(class_names)},
        "split": split_details,
    }
    (args.output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    print(json.dumps(configuration), flush=True)

    for epoch in range(1, args.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        validation_report, validation_loss = evaluate(model, validation_loader, criterion, device, class_names)
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
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
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": "IntentCNN",
                    "model_config": {"num_classes": len(class_names), "dropout": args.dropout},
                    "class_to_index": class_to_index,
                    "epoch": epoch,
                    "validation_report": {
                        **validation_report.summary,
                        "per_class": validation_report.per_class,
                        "confusion_matrix": validation_report.confusion_matrix,
                        "loss": validation_loss,
                    },
                    "configuration": configuration,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if args.patience and epochs_without_improvement >= args.patience:
            stopped_early = True
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    # The untouched test split is opened only after validation has selected the checkpoint.
    test_files = split_files(args.data, "test")
    test_table = read_columns(test_files, ("audio", "command"))
    test_commands = test_table.column("command").to_pylist()
    if set(test_commands) - set(class_names):
        raise ValueError("Test contains command labels absent from training")
    test_set = ParquetIntentDataset(test_table, np.arange(len(test_table)), class_to_index, extractor)
    validation_report, validation_loss = evaluate(model, validation_loader, criterion, device, class_names)
    test_report, test_loss, latency = evaluate_test_once(
        model, test_set, criterion, device, class_names
    )
    model_size = checkpoint_path.stat().st_size
    write_class_metrics(
        args.output / "class_metrics.csv",
        {"validation": validation_report, "test": test_report},
    )
    save_confusion_matrix(args.output / "confusion_matrix.png", test_report.confusion_matrix, class_names)

    metrics = {
        "experiment": configuration["experiment"],
        "configuration": configuration,
        "dataset": split_details,
        "selection_policy": "best checkpoint selected only by validation macro-F1; test evaluated once afterward",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "stopped_early": stopped_early,
        "validation": {
            **validation_report.summary,
            "loss": validation_loss,
            "per_class": validation_report.per_class,
            "confusion_matrix": validation_report.confusion_matrix,
        },
        "test": {
            **test_report.summary,
            "loss": test_loss,
            "per_class": test_report.per_class,
            "confusion_matrix": test_report.confusion_matrix,
            "selected_confusions": selected_confusions(test_report.confusion_matrix, class_to_index),
        },
        "trainable_parameters": parameters,
        "saved_model_size_bytes": model_size,
        "inference_latency": latency,
        "holdout_used": False,
        "numerals_used": False,
        "artifacts": {
            "checkpoint": "best.pt",
            "metrics": "metrics.json",
            "history": "training_history.csv",
            "class_metrics": "class_metrics.csv",
            "confusion_matrix": "confusion_matrix.png",
            "class_mapping": "class_to_index.json",
            "configuration": "config.json",
            "split_manifest": "split_manifest.json",
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"completed": True, "output": str(args.output), "test": test_report.summary}), flush=True)


if __name__ == "__main__":
    main()
