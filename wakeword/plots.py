from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _finish_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_training_plots(report: dict[str, Any], output: Path) -> list[str]:
    history = report["history"]
    epochs = [row["epoch"] for row in history]
    paths: list[Path] = []

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train"]["loss"] for row in history], marker="o", label="Training")
    plt.plot(epochs, [row["validation"]["loss"] for row in history], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.title("MARVIN training and validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    paths.append(output / "training_validation_loss.png")
    _finish_figure(paths[-1])

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train"]["accuracy"] for row in history], marker="o", label="Training")
    plt.plot(
        epochs,
        [row["validation"]["fixed_threshold_0_5"]["accuracy"] for row in history],
        marker="o",
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy at threshold 0.5")
    plt.ylim(0.0, 1.01)
    plt.title("MARVIN training and validation accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    paths.append(output / "training_validation_accuracy.png")
    _finish_figure(paths[-1])

    plt.figure(figsize=(7, 4.5))
    selected = [row["validation"]["selected_threshold"] for row in history]
    for metric in ("precision", "recall", "f1"):
        plt.plot(epochs, [row[metric] for row in selected], marker="o", label=metric.capitalize())
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.01)
    plt.title("Validation precision, recall, and F1")
    plt.grid(alpha=0.25)
    plt.legend()
    paths.append(output / "validation_precision_recall_f1.png")
    _finish_figure(paths[-1])

    test = report["test"]
    matrix = [
        [test["true_negatives"], test["false_accepts"]],
        [test["false_rejects"], test["true_positives"]],
    ]
    plt.figure(figsize=(5.5, 5))
    image = plt.imshow(matrix, cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks((0, 1), ("non-MARVIN", "MARVIN"))
    plt.yticks((0, 1), ("non-MARVIN", "MARVIN"))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Final test confusion matrix")
    cutoff = max(max(row) for row in matrix) / 2
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            plt.text(
                column_index,
                row_index,
                f"{value:,}",
                ha="center",
                va="center",
                color="white" if value > cutoff else "black",
                fontsize=12,
            )
    paths.append(output / "test_confusion_matrix.png")
    _finish_figure(paths[-1])
    return [str(path) for path in paths]
