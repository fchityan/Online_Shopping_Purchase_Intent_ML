import pytest

from src.evaluate import evaluate_predictions


def test_evaluate_predictions_returns_expected_metrics():
    y_true = [0, 1, 0, 1]
    proba = [0.1, 0.9, 0.6, 0.4]

    metrics = evaluate_predictions(y_true, proba, threshold=0.5)

    assert metrics['auc'] == pytest.approx(0.75)
    assert metrics['f1'] == pytest.approx(0.5)
    assert metrics['precision'] == pytest.approx(0.5)
    assert metrics['recall'] == pytest.approx(0.5)
    assert metrics['accuracy'] == pytest.approx(0.5)
    assert metrics['cm'] == [[1, 1], [1, 1]]
