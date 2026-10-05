"""Tests for finbert.metrics — macro-F1 and per-class precision/recall."""

from __future__ import annotations

import torch

from finbert.metrics import classification_report, confusion_matrix


def test_confusion_matrix_perfect_predictions():
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = labels.clone()
    cm = confusion_matrix(preds, labels, num_classes=3)
    assert cm.diagonal().sum().item() == 6
    assert cm.sum().item() == 6


def test_confusion_matrix_counts_off_diagonal():
    labels = torch.tensor([0, 0, 1])
    preds = torch.tensor([0, 1, 1])  # one negative misclassified as neutral
    cm = confusion_matrix(preds, labels, num_classes=3)
    assert cm[0, 1].item() == 1  # true=0, predicted=1
    assert cm[0, 0].item() == 1
    assert cm[1, 1].item() == 1


def test_classification_report_perfect_predictions_gives_f1_one():
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = labels.clone()
    report = classification_report(preds, labels, num_classes=3, class_names=["neg", "neu", "pos"])
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    for name in ["neg", "neu", "pos"]:
        assert report["per_class"][name]["precision"] == 1.0
        assert report["per_class"][name]["recall"] == 1.0
        assert report["per_class"][name]["f1"] == 1.0


def test_classification_report_detects_minority_class_failure():
    # A model that always predicts the majority class ("pos") scores high
    # accuracy but zero recall on the minority class — macro-F1 must expose
    # this even though accuracy alone would look fine.
    labels = torch.tensor([2] * 8 + [0] * 2)  # 80% majority class, 20% minority
    preds = torch.tensor([2] * 10)  # always predicts majority class

    report = classification_report(preds, labels, num_classes=3, class_names=["neg", "neu", "pos"])
    assert report["accuracy"] == 0.8
    # Minority class recall/precision/f1 are all zero.
    assert report["per_class"]["neg"]["recall"] == 0.0
    assert report["per_class"]["neg"]["f1"] == 0.0
    # Macro-F1 averages in that zero, so it's well below accuracy.
    assert report["macro_f1"] < report["accuracy"]


def test_classification_report_support_matches_label_counts():
    labels = torch.tensor([0, 0, 1, 2, 2, 2])
    preds = torch.tensor([0, 1, 1, 2, 2, 0])
    report = classification_report(preds, labels, num_classes=3, class_names=["neg", "neu", "pos"])
    assert report["per_class"]["neg"]["support"] == 2
    assert report["per_class"]["neu"]["support"] == 1
    assert report["per_class"]["pos"]["support"] == 3


def test_classification_report_handles_zero_predicted_for_a_class():
    # If a class is never predicted, precision for it is defined as 0.0
    # (not a division-by-zero crash).
    labels = torch.tensor([0, 1, 2])
    preds = torch.tensor([1, 1, 1])  # class 0 and 2 never predicted
    report = classification_report(preds, labels, num_classes=3, class_names=["neg", "neu", "pos"])
    assert report["per_class"]["neg"]["precision"] == 0.0
    assert report["per_class"]["pos"]["precision"] == 0.0
