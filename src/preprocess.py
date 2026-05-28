from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BOUNDED_RATE_COLUMNS = ['BounceRate', 'ExitRate']


def domain_clean(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    for col in BOUNDED_RATE_COLUMNS:
        data.loc[(data[col] < 0) | (data[col] > 1), col] = np.nan

    data.loc[
        (data['SpecialDayProximity'] < 0) | (data['SpecialDayProximity'] > 1),
        'SpecialDayProximity',
    ] = np.nan

    data['CustomerType'] = data['CustomerType'].str.strip().str.lower()

    for col, prefix in [('TrafficSource', 'TS'), ('GeographicRegion', 'GR')]:
        tmp = pd.to_numeric(data[col], errors='coerce').round().astype('Int64')
        data[col] = tmp.apply(lambda v: f'{prefix}_{int(v)}' if pd.notna(v) else np.nan).astype('object')

    return data


def split_features_target(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def get_feature_types(X_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    return cat_cols, num_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ])
