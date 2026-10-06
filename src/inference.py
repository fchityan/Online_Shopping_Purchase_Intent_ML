from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "CustomerType",
    "SpecialDayProximity",
    "ExitRate",
    "PageValue",
    "TrafficSource",
    "GeographicRegion",
    "BounceRate",
    "ProductPageTime",
]


def load_model_bundle(model_path: str | Path = "outputs/model_bundle.joblib") -> dict[str, Any]:
    """Load the production bundle, while supporting the legacy raw-pipeline artifact."""
    artifact = joblib.load(Path(model_path))
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {
        "model": artifact,
        "model_name": "legacy_pipeline",
        "model_display_name": "Legacy Pipeline",
        "model_version": "legacy",
        "threshold": 0.5,
        "feature_columns": EXPECTED_COLUMNS,
    }


def prepare_prediction_frame(
    records: Sequence[dict[str, Any]],
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
) -> pd.DataFrame:
    if not records:
        raise ValueError("At least one record is required.")
    frame = pd.DataFrame(records)
    missing = [column for column in expected_columns if column not in frame.columns]
    unexpected = sorted(set(frame.columns) - set(expected_columns))
    if missing or unexpected:
        raise ValueError(f"Schema mismatch. missing={missing}, unexpected={unexpected}")

    for column in ["SpecialDayProximity", "ExitRate", "BounceRate"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any():
            raise ValueError(f"Column '{column}' must contain finite numeric values.")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"Column '{column}' must be between 0 and 1.")
        frame[column] = values

    for column in ["PageValue", "ProductPageTime", "TrafficSource", "GeographicRegion"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any():
            raise ValueError(f"Column '{column}' must contain finite numeric values.")
        frame[column] = values

    frame["CustomerType"] = frame["CustomerType"].astype(str)
    return frame[list(expected_columns)].copy()


def predict_records(
    records: Sequence[dict[str, Any]],
    model_path: str | Path = "outputs/model_bundle.joblib",
    bundle: dict[str, Any] | None = None,
) -> list[float]:
    resolved_bundle = bundle or load_model_bundle(model_path)
    frame = prepare_prediction_frame(records, resolved_bundle.get("feature_columns", EXPECTED_COLUMNS))
    probabilities = resolved_bundle["model"].predict_proba(frame)[:, 1]
    return [float(score) for score in probabilities]


def build_prediction_response(
    records: Sequence[dict[str, Any]],
    model_path: str | Path = "outputs/model_bundle.joblib",
    bundle: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    resolved_bundle = bundle or load_model_bundle(model_path)
    scores = predict_records(records, model_path=model_path, bundle=resolved_bundle)
    threshold = float(resolved_bundle.get("threshold", 0.5))
    return [
        {
            "score": score,
            "prediction": int(score >= threshold),
            "threshold": threshold,
            "model_name": resolved_bundle.get("model_name", "unknown"),
            "model_version": resolved_bundle.get("model_version", "unknown"),
        }
        for score in scores
    ]
