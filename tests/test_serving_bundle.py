import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.inference import build_prediction_response


def test_serving_bundle_uses_promoted_threshold():
    columns = [
        "CustomerType",
        "SpecialDayProximity",
        "ExitRate",
        "PageValue",
        "TrafficSource",
        "GeographicRegion",
        "BounceRate",
        "ProductPageTime",
    ]
    frame = pd.DataFrame(
        [
            ["returning", 0.0, 0.1, 0.0, 1, 1, 0.05, 10.0],
            ["new", 0.4, 0.5, 12.0, 2, 2, 0.2, 100.0],
            ["returning", 0.1, 0.2, 4.0, 3, 1, 0.1, 50.0],
            ["new", 0.5, 0.6, 20.0, 4, 3, 0.3, 150.0],
        ],
        columns=columns,
    )
    y = [0, 1, 0, 1]
    categorical = ["CustomerType"]
    numeric = [column for column in columns if column not in categorical]
    preprocessor = ColumnTransformer([("num", "passthrough", numeric), ("cat", "drop", categorical)])
    pipeline = Pipeline([("preprocess", preprocessor), ("model", LogisticRegression())])
    pipeline.fit(frame, y)

    bundle = {
        "model": pipeline,
        "model_name": "test_model",
        "model_version": "v1",
        "threshold": 0.25,
        "feature_columns": columns,
    }
    response = build_prediction_response([frame.iloc[0].to_dict()], bundle=bundle)

    assert response[0]["threshold"] == 0.25
    assert response[0]["model_version"] == "v1"
    assert 0.0 <= response[0]["score"] <= 1.0
