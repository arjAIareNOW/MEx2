from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    examples: int
    positives: int
    negatives: int
    true_positives: int
    true_negatives: int
    false_accepts: int
    false_rejects: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_accept_rate: float
    false_reject_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def compute_metrics(probabilities: Tensor, labels: Tensor, threshold: float) -> BinaryMetrics:
    probabilities = probabilities.float().cpu()
    labels = labels.long().cpu()
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int((predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    total = labels.numel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return BinaryMetrics(
        threshold=float(threshold),
        examples=total,
        positives=tp + fn,
        negatives=tn + fp,
        true_positives=tp,
        true_negatives=tn,
        false_accepts=fp,
        false_rejects=fn,
        accuracy=(tp + tn) / total if total else 0.0,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        false_accept_rate=fp / (fp + tn) if fp + tn else 0.0,
        false_reject_rate=fn / (fn + tp) if fn + tp else 0.0,
    )


def best_f1_threshold(probabilities: Tensor, labels: Tensor) -> float:
    """Find the exact validation-set threshold maximizing F1."""
    probabilities = probabilities.float().cpu()
    labels = labels.long().cpu()
    order = torch.argsort(probabilities, descending=True)
    scores = probabilities[order]
    sorted_labels = labels[order]
    tp = torch.cumsum(sorted_labels, dim=0).float()
    fp = torch.arange(1, labels.numel() + 1).float() - tp
    positives = float(labels.sum())
    f1 = 2 * tp / (2 * tp + fp + (positives - tp)).clamp_min(1.0)
    distinct_end = torch.ones_like(scores, dtype=torch.bool)
    distinct_end[:-1] = scores[:-1] != scores[1:]
    candidate_indices = torch.nonzero(distinct_end).squeeze(1)
    best_index = candidate_indices[torch.argmax(f1[candidate_indices])]
    return float(scores[best_index])
