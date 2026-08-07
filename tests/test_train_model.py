import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.train_model import get_feature_importance, get_models, parse_args, tune_threshold_for_f1


def test_tune_threshold_for_f1_returns_best_threshold_and_sorted_results():
    y_true = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.61, 0.9]

    best_threshold, threshold_df = tune_threshold_for_f1(y_true, proba, start=0.2, stop=0.81, step=0.2)

    assert best_threshold == 0.6000000000000001
    assert threshold_df.iloc[0]['f1'] >= threshold_df.iloc[-1]['f1']
    assert set(threshold_df.columns) == {'threshold', 'f1', 'precision', 'recall', 'accuracy'}


def test_get_models_includes_lightgbm_classifier():
    models = get_models(random_state=42)

    assert 'lgbm_baseline' in models
    assert isinstance(models['lgbm_baseline'], LGBMClassifier)


def test_parse_args_includes_mlflow_options():
    args = parse_args([])

    assert args.mlflow_tracking_uri == 'file:./mlruns'
    assert args.experiment_name == 'purchase-intent-lightgbm'


def test_get_feature_importance_returns_empty_frame_for_models_without_importances():
    clf = Pipeline(
        [
            ('preprocess', ColumnTransformer([('num', 'passthrough', ['feature'])])),
            ('model', LogisticRegression()),
        ]
    )
    clf.fit(pd.DataFrame({'feature': [0.0, 1.0, 2.0, 3.0]}), [0, 0, 1, 1])

    feature_importance = get_feature_importance(clf, num_cols=['feature'], cat_cols=[])

    assert feature_importance.empty
    assert list(feature_importance.columns) == ['feature', 'importance']


def test_get_feature_importance_returns_ranked_importances_for_tree_models():
    X = pd.DataFrame(
        {
            'num_feature': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'cat_feature': ['a', 'a', 'b', 'b', 'c', 'c'],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = Pipeline(
        [
            (
                'preprocess',
                ColumnTransformer(
                    [
                        ('num', 'passthrough', ['num_feature']),
                        ('cat', Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))]), ['cat_feature']),
                    ]
                ),
            ),
            ('model', RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
    clf.fit(X, y)

    feature_importance = get_feature_importance(clf, num_cols=['num_feature'], cat_cols=['cat_feature'])

    assert not feature_importance.empty
    assert feature_importance.columns.tolist() == ['feature', 'importance']
    assert feature_importance['feature'].tolist()[0] in {'num_feature', 'cat_feature_a', 'cat_feature_b', 'cat_feature_c'}