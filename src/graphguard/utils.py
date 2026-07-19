"""Shared utility functions for GraphGuard."""

import numpy as np


def best_f1_threshold(probs, labels):
    """Pick the decision threshold that maximizes F1 on a validation split.

    The illicit class is a ~2% minority and training uses pos_weight /
    scale_pos_weight, so the model's predicted probabilities are NOT centered
    at 0.5 — the default 0.5 cutoff systematically understates F1. We sweep
    the precision-recall curve (which already evaluates candidate thresholds)
    and return the F1-maximizing one. Falls back to 0.5 when there is no
    positive label in the split.
    """
    from sklearn.metrics import precision_recall_curve

    if labels.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # precision_recall_curve returns thresholds of length n-1 vs. p/r of length n;
    # the last (p, r) pair has no corresponding threshold. F1 is only defined
    # where a threshold exists, so align with thresholds.
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])
