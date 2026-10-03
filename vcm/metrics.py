from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ClassificationReport:
    summary: dict[str, float]
    per_class: list[dict[str, float | int | str]]
    confusion_matrix: list[list[int]]


def classification_report(
    targets: Sequence[int], predictions: Sequence[int], class_names: Sequence[str]
) -> ClassificationReport:
    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    size = len(class_names)
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (targets_array, predictions_array), 1)
    tp = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    total = int(matrix.sum())
    summary = {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / support.sum()) if support.sum() else 0.0,
    }
    per_class = [
        {
            "class": name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    ]
    return ClassificationReport(summary, per_class, matrix.tolist())


def write_class_metrics(path: Path, reports: dict[str, ClassificationReport]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", "class", "precision", "recall", "f1", "support"))
        writer.writeheader()
        for split, report in reports.items():
            for row in report.per_class:
                writer.writerow({"split": split, **row})


def save_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]], class_names: Sequence[str]) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    positions = np.arange(len(class_names))
    axis.set(
        xticks=positions,
        yticks=positions,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted command",
        ylabel="True command",
        title="VCM CNN Experiment 1 — test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def selected_confusions(matrix: Sequence[Sequence[int]], class_to_index: dict[str, int]) -> dict[str, object]:
    values = np.asarray(matrix)
    pairs = (("LIGHT_ON", "LIGHT_OFF"), ("VOLUME_UP", "VOLUME_DOWN"), ("TIMER", "ALARM"))
    result: dict[str, object] = {}
    for left, right in pairs:
        i, j = class_to_index[left], class_to_index[right]
        result[f"{left}_vs_{right}"] = {
            f"{left}_predicted_as_{right}": int(values[i, j]),
            f"{right}_predicted_as_{left}": int(values[j, i]),
        }
    oos = class_to_index["OUT_OF_SCOPE"]
    result["OUT_OF_SCOPE_vs_valid"] = {
        "out_of_scope_predicted_as_valid": int(values[oos].sum() - values[oos, oos]),
        "valid_predicted_as_out_of_scope": int(values[:, oos].sum() - values[oos, oos]),
    }
    return result

