from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import BalancedBinaryBatchSampler, SpeechCommandsBinary
from .metrics import best_f1_threshold, compute_metrics
from .model import MarvinDSCNN, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight MARVIN wake-word detector")
    parser.add_argument("--data", type=Path, default=Path("dataset/speech_commands"))
    parser.add_argument("--output", type=Path, default=Path("models/marvin"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument(
        "--extra-noise-dir",
        type=Path,
        default=None,
        help="Optional recursive WAV directory used only for training noise augmentation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation-F1 improvement; 0 disables early stopping",
    )
    parser.add_argument("--workers", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cpu-threads", type=int, choices=(1,), default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor, float]:
    model.eval()
    probabilities: list[Tensor] = []
    labels: list[Tensor] = []
    loss_sum = 0.0
    for features, batch_labels in loader:
        device_labels = batch_labels.to(device, non_blocking=True)
        logits = model(features.to(device, non_blocking=True))
        loss_sum += float(F.binary_cross_entropy_with_logits(logits, device_labels, reduction="sum"))
        probabilities.append(logits.sigmoid().cpu())
        labels.append(batch_labels.cpu())
    all_probabilities = torch.cat(probabilities)
    all_labels = torch.cat(labels)
    return all_probabilities, all_labels, loss_sum / all_labels.numel()


def main() -> None:
    args = parse_args()
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    train_set = SpeechCommandsBinary(
        args.data,
        "train",
        augment=True,
        extra_noise_dir=args.extra_noise_dir,
    )
    validation_set = SpeechCommandsBinary(args.data, "validation")
    test_set = SpeechCommandsBinary(args.data, "test")
    batch_sampler = BalancedBinaryBatchSampler(
        train_set.labels,
        negatives_per_positive=args.negative_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    model = MarvinDSCNN().to(device)
    model_parameters = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    best_validation_f1 = -1.0
    epochs_without_improvement = 0
    stopped_early = False
    checkpoint_path = args.output / "best.pt"
    history: list[dict[str, object]] = []

    counts = {
        name: {
            "examples": len(dataset),
            "marvin": sum(dataset.labels),
            "other": len(dataset) - sum(dataset.labels),
        }
        for name, dataset in (
            ("train", train_set),
            ("validation", validation_set),
            ("test", test_set),
        )
    }
    print(
        json.dumps(
            {
                "device": str(device),
                "parameters": model_parameters,
                "splits": counts,
                "training_noise_sources": train_set.noise_source_counts,
                "effective_training_examples_per_epoch": batch_sampler.effective_counts,
            }
        ),
        flush=True,
    )

    for epoch in range(args.epochs):
        batch_sampler.set_epoch(epoch)
        train_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        model.train()
        loss_sum = 0.0
        examples = 0
        marvin_examples = 0
        correct = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * labels.numel()
            examples += labels.numel()
            marvin_examples += int(labels.sum().item())
            correct += int(((logits >= 0) == labels.bool()).sum().item())
        scheduler.step()

        validation_probabilities, validation_labels, validation_loss = predict(model, validation_loader, device)
        threshold = best_f1_threshold(validation_probabilities, validation_labels)
        validation_metrics = compute_metrics(validation_probabilities, validation_labels, threshold)
        fixed_validation_metrics = compute_metrics(validation_probabilities, validation_labels, 0.5)
        epoch_report = {
            "epoch": epoch + 1,
            "train": {
                "loss": loss_sum / examples,
                "accuracy": correct / examples,
                "effective_counts": {
                    "marvin": marvin_examples,
                    "non_marvin": examples - marvin_examples,
                    "total": examples,
                },
            },
            "validation": {
                "loss": validation_loss,
                "fixed_threshold_0_5": fixed_validation_metrics.to_dict(),
                "selected_threshold": validation_metrics.to_dict(),
            },
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report), flush=True)
        if validation_metrics.f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "threshold": threshold,
                    "parameters": model_parameters,
                    "epoch": epoch + 1,
                    "validation_metrics": validation_metrics.to_dict(),
                    "history": history,
                    "args": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                stopped_early = True
                print(
                    json.dumps(
                        {
                            "early_stopping": True,
                            "epoch": epoch + 1,
                            "patience": args.patience,
                            "best_validation_f1": best_validation_f1,
                        }
                    ),
                    flush=True,
                )
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["threshold"])
    test_probabilities, test_labels, test_loss = predict(model, test_loader, device)
    test_metrics = compute_metrics(test_probabilities, test_labels, threshold)
    report = {
        "model": "MarvinDSCNN",
        "parameters": model_parameters,
        "checkpoint_epoch": checkpoint["epoch"],
        "epochs_completed": len(history),
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "stopped_early": stopped_early,
        "best_validation_f1": checkpoint["validation_metrics"]["f1"],
        "selected_validation_threshold": threshold,
        "effective_training_examples_per_epoch": batch_sampler.effective_counts,
        "validation": checkpoint["validation_metrics"],
        "test": test_metrics.to_dict() | {"loss": test_loss},
        "split_counts": counts,
        "training_noise_sources": train_set.noise_source_counts,
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    from .plots import save_training_plots

    plot_paths = save_training_plots(report, args.output)

    model = model.cpu().eval()
    scripted = torch.jit.script(model)
    scripted.save(str(args.output / "marvin_dscnn.ts"))
    print(json.dumps({"final_report": report, "plots": plot_paths, "artifacts": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
