from pathlib import Path

import pandas as pd

from src.data_loader import load_raw_data


def test_load_raw_data_parses_configured_missing_values(tmp_path: Path):
    csv_path = tmp_path / 'sample.csv'
    csv_path.write_text(
        'a,b,c\n1,NULL,text\n2,None,more\n3,,done\n',
        encoding='utf-8',
    )

    loaded = load_raw_data(csv_path)

    assert loaded.shape == (3, 3)
    assert pd.isna(loaded.loc[0, 'b'])
    assert pd.isna(loaded.loc[1, 'b'])
    assert pd.isna(loaded.loc[2, 'b'])
