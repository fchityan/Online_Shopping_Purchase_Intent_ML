from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import joblib
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    'CustomerType',
    'SpecialDayProximity',
    'ExitRate',
    'PageValue',
    'TrafficSource',
    'GeographicRegion',
    'BounceRate',
    'ProductPageTime',
]


def prepare_prediction_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of record dictionaries into the model input frame."""
    frame = pd.DataFrame(records)
    missing = [col for col in EXPECTED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    return frame[EXPECTED_COLUMNS].copy()


def predict_records(records: Sequence[dict[str, Any]], model_path: str | Path = 'model.joblib') -> List[float]:
    """Load a persisted sklearn-style pipeline and score a batch of records."""
    model_path = Path(model_path)
    pipeline = joblib.load(model_path)
    frame = prepare_prediction_frame(records)
    probabilities = pipeline.predict_proba(frame)[:, 1]
    return [float(score) for score in probabilities]


def build_prediction_response(records: Sequence[dict[str, Any]], model_path: str | Path = 'model.joblib') -> List[dict[str, Any]]:
    """Return a production-friendly payload with score and label."""
    scores = predict_records(records, model_path=model_path)
    return [
        {
            'score': score,
            'prediction': int(score >= 0.5),
        }
        for score in scores
    ]
