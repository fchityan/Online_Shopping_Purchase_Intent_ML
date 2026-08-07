from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def build_data_quality_alerts(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Create a simple set of data quality alerts for monitoring."""
    alerts: List[Dict[str, object]] = []

    missing = df.isna().sum()
    for column, count in missing[missing > 0].items():
        alerts.append({
            'check': 'missing_values',
            'column': column,
            'value': int(count),
            'severity': 'warning',
        })

    for column in ['BounceRate', 'ExitRate', 'SpecialDayProximity']:
        if column in df.columns:
            invalid = ((df[column] < 0) | (df[column] > 1)).sum()
            if invalid > 0:
                alerts.append({
                    'check': 'invalid_bounds',
                    'column': column,
                    'value': int(invalid),
                    'severity': 'warning',
                })

    return alerts


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, object]:
    """Create a basic drift summary for numeric and categorical features."""
    report: Dict[str, object] = {'numeric': {}, 'categorical': {}}

    for feature in numeric_features:
        if feature in reference.columns and feature in current.columns:
            ref_values = pd.to_numeric(reference[feature], errors='coerce').dropna()
            cur_values = pd.to_numeric(current[feature], errors='coerce').dropna()
            if ref_values.empty or cur_values.empty:
                drift_score = np.nan
            else:
                ref_mean = float(ref_values.mean())
                cur_mean = float(cur_values.mean())
                drift_score = abs(cur_mean - ref_mean) / max(abs(ref_mean), 1e-6)
            report['numeric'][feature] = {
                'reference_mean': float(ref_values.mean()) if not ref_values.empty else np.nan,
                'current_mean': float(cur_values.mean()) if not cur_values.empty else np.nan,
                'drift_score': float(drift_score) if not np.isnan(drift_score) else np.nan,
            }

    for feature in categorical_features:
        if feature in reference.columns and feature in current.columns:
            ref_counts = reference[feature].fillna('missing').value_counts(normalize=True)
            cur_counts = current[feature].fillna('missing').value_counts(normalize=True)
            categories = sorted(set(ref_counts.index) | set(cur_counts.index))
            drift_score = 0.0
            for category in categories:
                ref_pct = float(ref_counts.get(category, 0.0))
                cur_pct = float(cur_counts.get(category, 0.0))
                drift_score += abs(cur_pct - ref_pct)
            report['categorical'][feature] = {
                'drift_score': float(drift_score),
                'reference_distribution': ref_counts.to_dict(),
                'current_distribution': cur_counts.to_dict(),
            }

    return report


def build_weekly_performance_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create a simple weekly performance summary table."""
    if metrics.empty:
        return pd.DataFrame(columns=['week', 'accuracy', 'precision', 'recall', 'f1'])
    return metrics[['week', 'accuracy', 'precision', 'recall', 'f1']].copy()


def build_business_impact_table(
    predictions: pd.DataFrame,
    labels: pd.Series | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Create a simple business-impact summary from predictions and outcomes."""
    if labels is None:
        labels = pd.Series([0] * len(predictions), index=predictions.index)

    scored = predictions.copy()
    scored['predicted_positive'] = scored['score'] >= threshold
    scored['actual_positive'] = labels.reset_index(drop=True).astype(int).to_numpy()

    segment = scored.groupby('predicted_positive').agg(
        targeted_sessions=('predicted_positive', 'size'),
        actual_positive=('actual_positive', 'sum'),
        total_sessions=('predicted_positive', 'size'),
    )
    segment['conversion_rate'] = segment['actual_positive'] / segment['total_sessions']
    segment['segment_name'] = ['non_targeted', 'targeted']
    return segment[['segment_name', 'targeted_sessions', 'actual_positive', 'conversion_rate']].reset_index(drop=True)
