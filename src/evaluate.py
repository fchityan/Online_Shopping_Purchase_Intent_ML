from typing import Dict

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_predictions(y_true, proba, threshold: float) -> Dict[str, object]:
    pred = (proba >= threshold).astype(int)
    return {
        'auc': float(roc_auc_score(y_true, proba)),
        'f1': float(f1_score(y_true, pred)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred)),
        'accuracy': float(accuracy_score(y_true, pred)),
        'cm': confusion_matrix(y_true, pred).tolist(),
    }
