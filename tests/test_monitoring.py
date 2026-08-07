import pandas as pd

from src.monitoring import (
    build_data_quality_alerts,
    build_drift_report,
    build_weekly_performance_summary,
)


def test_build_data_quality_alerts_flags_missing_and_invalid_values():
    df = pd.DataFrame(
        {
            'BounceRate': [1.2, 0.2, None],
            'ExitRate': [0.1, 1.5, 0.3],
            'CustomerType': ['new', 'new', None],
        }
    )

    alerts = build_data_quality_alerts(df)

    assert any(alert['check'] == 'missing_values' for alert in alerts)
    assert any(alert['check'] == 'invalid_bounds' for alert in alerts)
    assert any(alert['column'] == 'CustomerType' and alert['check'] == 'missing_values' for alert in alerts)


def test_build_drift_report_returns_feature_summaries():
    reference = pd.DataFrame(
        {
            'PageValue': [10, 20, 30],
            'CustomerType': ['new', 'returning', 'new'],
        }
    )
    current = pd.DataFrame(
        {
            'PageValue': [12, 22, 32],
            'CustomerType': ['new', 'returning', 'new'],
        }
    )

    report = build_drift_report(reference, current, numeric_features=['PageValue'], categorical_features=['CustomerType'])

    assert 'numeric' in report
    assert 'categorical' in report
    assert 'PageValue' in report['numeric']
    assert 'CustomerType' in report['categorical']


def test_build_weekly_performance_summary_uses_metrics_rows():
    metrics = pd.DataFrame(
        [
            {'week': '2026-W01', 'accuracy': 0.91, 'precision': 0.82, 'recall': 0.74, 'f1': 0.78},
        ]
    )

    summary = build_weekly_performance_summary(metrics)

    assert summary.iloc[0]['week'] == '2026-W01'
    assert summary.iloc[0]['f1'] == 0.78
