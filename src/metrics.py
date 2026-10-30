"""
Evaluation metrics for 'Name That LLM'.
Calculates official balanced accuracy, per-class recall, and confusion diagnostics.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.config import MODEL_TO_LAB, VALID_18_MODELS


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Computes full evaluation report:
    - Balanced accuracy across 18 models
    - Lab-level attribution accuracy
    - Per-class recall
    - Top confused sibling pairs
    """
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    # Lab-level accuracy
    lab_true = np.array([MODEL_TO_LAB.get(y, "unknown") for y in y_true])
    lab_pred = np.array([MODEL_TO_LAB.get(y, "unknown") for y in y_pred])
    lab_acc = (lab_true == lab_pred).mean()
    
    # Per-class recall
    cm = confusion_matrix(y_true, y_pred, labels=VALID_18_MODELS)
    per_class_recall = {}
    for i, model in enumerate(VALID_18_MODELS):
        total = cm[i].sum()
        correct = cm[i, i]
        recall = correct / total if total > 0 else 0.0
        per_class_recall[model] = recall

    # Identify most confused pairs (where true != pred)
    confusion_pairs = []
    for i, true_m in enumerate(VALID_18_MODELS):
        for j, pred_m in enumerate(VALID_18_MODELS):
            if i != j and cm[i, j] > 0:
                is_sibling = MODEL_TO_LAB.get(true_m) == MODEL_TO_LAB.get(pred_m)
                confusion_pairs.append({
                    "true": true_m,
                    "pred": pred_m,
                    "count": int(cm[i, j]),
                    "is_sibling": is_sibling,
                    "lab": MODEL_TO_LAB.get(true_m),
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    
    # Sibling error percentage
    total_errors = (y_true != y_pred).sum()
    sibling_errors = sum(p["count"] for p in confusion_pairs if p["is_sibling"])
    sibling_error_pct = (sibling_errors / total_errors * 100) if total_errors > 0 else 0.0

    return {
        "balanced_accuracy": float(bacc),
        "lab_accuracy": float(lab_acc),
        "sibling_error_pct": float(sibling_error_pct),
        "per_class_recall": per_class_recall,
        "top_confusions": confusion_pairs[:10],
    }


def format_evaluation_summary(report: Dict) -> str:
    """Format evaluation results into a clean markdown table."""
    lines = [
        f"### Balanced Accuracy: {report['balanced_accuracy']:.4f}",
        f"- **Lab Attribution Accuracy**: {report['lab_accuracy']:.4f}",
        f"- **Sibling Error Share**: {report['sibling_error_pct']:.1f}% of errors are within the same lab\n",
        "#### Per-Class Recall (18 Models):",
        "| Model | Lab | Recall |",
        "| :--- | :--- | :--- |",
    ]
    for model, recall in report["per_class_recall"].items():
        lab = MODEL_TO_LAB.get(model, "")
        lines.append(f"| `{model}` | {lab} | {recall:.4f} |")

    lines.append("\n#### Top Confused Pairs:")
    lines.append("| True Model | Predicted Model | Errors | Sibling? |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for p in report["top_confusions"]:
        lines.append(f"| `{p['true']}` | `{p['pred']}` | {p['count']} | {'Yes' if p['is_sibling'] else 'No'} |")

    return "\n".join(lines)
