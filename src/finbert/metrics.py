"""
Classification metrics for NanoFinBERT — pure Python/torch, no sklearn dependency.

Accuracy alone hides class-imbalance problems: a model that always predicts
the majority class still scores well on accuracy. Macro-F1 and per-class
precision/recall surface that failure mode, which matters here because the
full Financial PhraseBank dataset is imbalanced (negative is under-represented
relative to neutral/positive).
"""

from __future__ import annotations

import torch


def confusion_matrix(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """
    Build a (num_classes, num_classes) confusion matrix, rows=true, cols=predicted.

    Vectorized via bincount on a flattened (true, predicted) index — this runs
    every eval epoch during training, so it avoids a per-sample Python loop
    and the device sync that labels.tolist()/predictions.tolist() would cause.
    """
    flat_index = labels * num_classes + predictions
    counts = torch.bincount(flat_index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def classification_report(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute accuracy, macro-F1, and per-class precision/recall/F1/support.

    Args:
        predictions: (N,) predicted class indices.
        labels:      (N,) true class indices.
        num_classes: Number of classes.
        class_names: Optional display names, length num_classes.

    Returns:
        {
            "accuracy": float,
            "macro_f1": float,
            "per_class": {name: {"precision", "recall", "f1", "support"}, ...},
            "confusion_matrix": list[list[int]],
        }
    """
    names = class_names or [str(i) for i in range(num_classes)]
    cm = confusion_matrix(predictions, labels, num_classes)

    total = cm.sum().item()
    correct = cm.diagonal().sum().item()
    accuracy = correct / total if total else 0.0

    per_class = {}
    f1_scores = []
    for i, name in enumerate(names):
        tp = cm[i, i].item()
        support = cm[i, :].sum().item()  # true count for this class
        predicted_count = cm[:, i].sum().item()

        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_report(report: dict) -> str:
    """Render a classification_report() dict as a readable text table."""
    lines = [
        f"{'class':<10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for name, pc in report["per_class"].items():
        lines.append(
            f"{name:<10}{pc['precision']:>10.3f}{pc['recall']:>10.3f}"
            f"{pc['f1']:>10.3f}{pc['support']:>10d}"
        )
    lines.append("")
    lines.append(f"accuracy: {report['accuracy']:.3f}   macro_f1: {report['macro_f1']:.3f}")
    return "\n".join(lines)
