import pytest  # type: ignore
import requests_mock  # type: ignore
import pandas as pd  # type: ignore

import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)
from src.api.data_fetcher import fetch_ohlc_data, analyze_data


@pytest.fixture
def mock_kline_data():
    return {
        "retCode": 0,
        "result": {
            "list": [
                [1609459200000, "29000", "30000", "28000", "29500", "1000"],
                [1609462800000, "29500", "31000", "29000", "30000", "1500"],
            ]
        },
    }


def test_fetch_ohlc_data(mock_kline_data):
    url = "https://api.bybit.com/v5/market/kline"
    with requests_mock.Mocker() as m:
        m.get(url, json=mock_kline_data)

        df = fetch_ohlc_data("BTCUSDT", 2, "1h")
        assert df is not None
        assert len(df) == 2
        assert "Open" in df.columns


def test_analyze_data(sample_dataframe, sample_preferences_all):
    indicators = analyze_data(
        sample_dataframe, sample_preferences_all, liq_lev_tolerance=0.05
    )
    assert indicators.order_blocks is not None
    assert indicators.liquidity_levels is not None
