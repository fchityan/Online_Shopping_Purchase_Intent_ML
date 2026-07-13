import numpy as np
import pandas as pd

from src.preprocess import build_preprocessor, domain_clean, get_feature_types, split_features_target


def test_domain_clean_normalizes_and_bounds_values():
    raw = pd.DataFrame(
        {
            'BounceRate': [0.2, 1.5],
            'ExitRate': [-0.1, 0.8],
            'SpecialDayProximity': [0.5, 2.0],
            'CustomerType': [' Returning_Visitor ', ' New_Visitor '],
            'TrafficSource': [1, 2.0],
            'GeographicRegion': [3, np.nan],
        }
    )

    cleaned = domain_clean(raw)

    assert pd.isna(cleaned.loc[1, 'BounceRate'])
    assert pd.isna(cleaned.loc[0, 'ExitRate'])
    assert pd.isna(cleaned.loc[1, 'SpecialDayProximity'])
    assert cleaned.loc[0, 'CustomerType'] == 'returning_visitor'
    assert cleaned.loc[1, 'CustomerType'] == 'new_visitor'
    assert cleaned.loc[0, 'TrafficSource'] == 'TS_1'
    assert cleaned.loc[0, 'GeographicRegion'] == 'GR_3'
    assert pd.isna(cleaned.loc[1, 'GeographicRegion'])


def test_split_features_target_separates_target_as_integers():
    frame = pd.DataFrame(
        {
            'feature': [1, 2],
            'PurchaseCompleted': [True, False],
        }
    )

    features, target = split_features_target(frame, 'PurchaseCompleted')

    assert list(features.columns) == ['feature']
    assert target.tolist() == [1, 0]


def test_build_preprocessor_handles_numeric_and_categorical_columns():
    frame = pd.DataFrame(
        {
            'num_feature': [1.0, np.nan, 3.0],
            'cat_feature': ['a', np.nan, 'b'],
        }
    )

    cat_cols, num_cols = get_feature_types(frame)
    preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
    transformed = preprocessor.fit_transform(frame)

    assert cat_cols == ['cat_feature']
    assert num_cols == ['num_feature']
    assert transformed.shape == (3, 3)
