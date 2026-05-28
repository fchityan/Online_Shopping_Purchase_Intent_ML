from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        na_values=['NULL', 'nan', 'None', ''],
        keep_default_na=True,
    )
