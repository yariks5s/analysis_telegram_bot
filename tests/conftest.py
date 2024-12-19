import pytest
import pandas as pd

@pytest.fixture
def sample_dataframe():
    data = {
        "Open": [29000, 29500],
        "High": [30000, 31000],
        "Low": [28000, 29000],
        "Close": [29500, 30000],
        "Volume": [1000, 1500],
    }
    dates = ["2021-01-01 00:00:00", "2021-01-01 01:00:00"]
    return pd.DataFrame(data, index=pd.to_datetime(dates))

