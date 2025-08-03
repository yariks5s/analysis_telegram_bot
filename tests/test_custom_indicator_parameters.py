"""
Tests for custom indicator parameters functionality.
"""

import pandas as pd
from unittest.mock import patch, MagicMock

from src.core.preferences import INDICATOR_PARAMS, create_default_preferences
from src.database.operations import get_user_preferences, update_user_preferences
from src.analysis.detection.indicators import compute_atr, detect_fvgs
from src.api.data_fetcher import analyze_data


def test_default_parameters_exist():
    """Test that default parameters are defined in the preferences module."""
    assert "atr_period" in INDICATOR_PARAMS
    assert "fvg_min_size" in INDICATOR_PARAMS

    assert INDICATOR_PARAMS["atr_period"]["default"] == 14
    assert INDICATOR_PARAMS["fvg_min_size"]["default"] == 0.0005

    assert "min" in INDICATOR_PARAMS["atr_period"]
    assert "max" in INDICATOR_PARAMS["atr_period"]
    assert "min" in INDICATOR_PARAMS["fvg_min_size"]
    assert "max" in INDICATOR_PARAMS["fvg_min_size"]


def test_default_preferences_include_parameters():
    """Test that default preferences include indicator parameters."""
    prefs = create_default_preferences()

    assert "atr_period" in prefs
    assert "fvg_min_size" in prefs

    assert prefs["atr_period"] == INDICATOR_PARAMS["atr_period"]["default"]
    assert prefs["fvg_min_size"] == INDICATOR_PARAMS["fvg_min_size"]["default"]


@patch("src.database.operations.sqlite3.connect")
def test_get_user_preferences_includes_parameters(mock_connect):
    """Test that get_user_preferences returns indicator parameters."""
    mock_cursor = MagicMock()
    mock_connect.return_value.cursor.return_value = mock_cursor

    mock_cursor.fetchone.return_value = [
        1,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        20,
        0.001,
    ]

    prefs = get_user_preferences(1)

    assert "atr_period" in prefs
    assert "fvg_min_size" in prefs
    assert prefs["atr_period"] == 20
    assert prefs["fvg_min_size"] == 0.001


@patch("src.database.operations.sqlite3.connect")
def test_update_user_preferences_updates_parameters(mock_connect):
    """Test that update_user_preferences saves indicator parameters."""
    mock_cursor = MagicMock()
    mock_connect.return_value.cursor.return_value = mock_cursor

    mock_cursor.fetchone.return_value = [1]

    custom_prefs = {
        "order_blocks": True,
        "fvgs": True,
        "liquidity_levels": False,
        "breaker_blocks": True,
        "show_legend": True,
        "show_volume": False,
        "liquidity_pools": True,
        "dark_mode": True,
        "atr_period": 21,
        "fvg_min_size": 0.0008,
    }

    update_user_preferences(1, custom_prefs)
    call_args = mock_cursor.execute.call_args_list[1][0]

    assert "atr_period = ?" in call_args[0]
    assert "fvg_min_size = ?" in call_args[0]
    assert 21 in call_args[1]
    assert 0.0008 in call_args[1]


def test_compute_atr_uses_custom_period():
    """Test that compute_atr uses the custom period parameter."""
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [95, 96, 97, 98, 99],
            "Close": [102, 103, 104, 105, 106],
        }
    )

    atr_default = compute_atr(df)
    atr_custom = compute_atr(df, period=3)

    assert atr_default.iloc[-1] != atr_custom.iloc[-1]


def test_detect_fvgs_uses_custom_min_ratio():
    """Test that detect_fvgs uses the custom min_fvg_ratio parameter."""
    # Create a DataFrame with a definite FVG
    # Creating a price series with 5 candles where a clear bullish FVG exists
    # between candle index 2 and 4 (where candle 4's low is greater than candle 2's high)
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 110],
            "High": [105, 106, 107, 109, 115],
            "Low": [
                95,
                96,
                97,
                98,
                108,
            ],  # Note: candle[4].low > candle[2].high (108 > 107) = bullish FVG
            "Close": [102, 103, 104, 105, 112],
        },
        index=pd.date_range(start="2023-01-01", periods=5),
    )

    fvgs_small_threshold = detect_fvgs(df, min_fvg_ratio=0.001)
    fvgs_high_threshold = detect_fvgs(df, min_fvg_ratio=0.05)

    assert len(fvgs_small_threshold.list) > 0
    assert len(fvgs_high_threshold.list) == 0


@patch("src.analysis.detection.indicators.detect_fvgs")
@patch("src.analysis.detection.indicators.detect_liquidity_levels")
@patch("src.analysis.detection.indicators.detect_liquidity_pools")
def test_analyze_data_passes_custom_parameters(
    mock_detect_liquidity_pools, mock_detect_liquidity_levels, mock_detect_fvgs
):
    """Test that analyze_data passes custom parameters to indicator functions."""
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [102, 103, 104],
        }
    )

    preferences = {
        "order_blocks": True,
        "fvgs": True,
        "liquidity_levels": True,
        "breaker_blocks": True,
        "liquidity_pools": True,
        "atr_period": 21,
        "fvg_min_size": 0.0008,
    }

    analyze_data(df, preferences, 0.05)

    mock_detect_fvgs.assert_called_once_with(df, min_fvg_ratio=0.0008)
    mock_detect_liquidity_levels.assert_called_once_with(
        df, window=len(df), stdev_multiplier=0.05, atr_period=21
    )
    mock_detect_liquidity_pools.assert_called_once_with(df, atr_period=21)
