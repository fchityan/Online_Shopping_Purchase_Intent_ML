import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.inference import prepare_prediction_frame, predict_records


def test_prepare_prediction_frame_returns_expected_columns():
    records = [
        {
            'CustomerType': 'Returning_Visitor',
            'SpecialDayProximity': 0.0,
            'ExitRate': 0.2,
            'PageValue': 0.0,
            'TrafficSource': 1.0,
            'GeographicRegion': 1,
            'BounceRate': 0.2,
            'ProductPageTime': 3.95e-07,
        }
    ]

    prepared = prepare_prediction_frame(records)

    assert list(prepared.columns) == [
        'CustomerType',
        'SpecialDayProximity',
        'ExitRate',
        'PageValue',
        'TrafficSource',
        'GeographicRegion',
        'BounceRate',
        'ProductPageTime',
    ]


def test_predict_records_uses_saved_pipeline(tmp_path):
    pipeline = Pipeline(
        [
            (
                'preprocess',
                ColumnTransformer(
                    [
                        ('num', SimpleImputer(strategy='median'), ['PageValue', 'ProductPageTime']),
                        ('cat', Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['CustomerType']),
                    ]
                ),
            ),
            ('model', LogisticRegression(max_iter=2000)),
        ]
    )

    X = pd.DataFrame(
        {
            'PageValue': [0.0, 5.0],
            'ProductPageTime': [1.0, 2.0],
            'CustomerType': ['Returning_Visitor', 'New_Visitor'],
        }
    )
    y = [0, 1]
    pipeline.fit(X, y)

    model_path = tmp_path / 'model.joblib'
    joblib.dump(pipeline, model_path)

    predictions = predict_records(
        [
            {
                'CustomerType': 'Returning_Visitor',
                'SpecialDayProximity': 0.0,
                'ExitRate': 0.2,
                'PageValue': 0.0,
                'TrafficSource': 1.0,
                'GeographicRegion': 1,
                'BounceRate': 0.2,
                'ProductPageTime': 3.95e-07,
            }
        ],
        model_path=model_path,
    )

    assert len(predictions) == 1
    assert all(0.0 <= score <= 1.0 for score in predictions)
