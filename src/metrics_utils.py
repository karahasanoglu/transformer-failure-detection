"""Shared evaluation and threshold-tuning utilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.20,
    beta: float = 2.0,
) -> tuple[float, dict]:
    """Pick a threshold that preserves anomaly recall without collapsing specificity."""
    candidates = np.unique(np.round(y_prob, 4))
    candidates = np.concatenate(([0.01], candidates, [0.99]))

    best_threshold = 0.5
    best_metrics: dict | None = None
    best_score = -1.0

    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # Recall alt limite yetişmeyen eşikler cezalandırılır.
        score = ((1 + beta**2) * prec * rec / ((beta**2 * prec) + rec + 1e-12))
        if rec < min_recall:
            score *= 0.5
        score += 0.25 * bal_acc + 0.10 * spec

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                'precision': prec,
                'recall': rec,
                'specificity': spec,
                'f1': f1,
                'balanced_accuracy': bal_acc,
            }

    return best_threshold, best_metrics or {}


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'balanced_accuracy': bal_acc,
        'auc': auc,
        'average_precision': ap,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'report': report,
        'confusion_matrix': cm,
        'threshold': threshold,
    }


def print_evaluation(title: str, results: dict) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title} (Eşik: {results['threshold']:.3f})")
    print(f"{'=' * 55}")
    print(f"  Doğruluk   (Accuracy)  : {results['accuracy'] * 100:.2f}%")
    print(f"  F1-Skoru               : {results['f1']:.4f}")
    print(f"  Kesinlik   (Precision) : {results['precision']:.4f}")
    print(f"  Duyarlılık (Recall)    : {results['recall']:.4f}")
    print(f"  Özgüllük   (Specific.) : {results['specificity']:.4f}")
    print(f"  Dengeli Doğruluk       : {results['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC                : {results['auc']:.4f}")
    print(f"  PR-AUC (Avg Precision) : {results['average_precision']:.4f}")
    print(f"\n  Karmaşıklık Matrisi:\n{results['confusion_matrix']}")
    print("\n  Classification Report:")
    print(results['report'])
