import pytest  # type: ignore

from data_fetching_instruments import fetch_from_json
from indicators import detect_liquidity_levels
from IndicatorUtils.liquidity_level_utils import LiquidityLevels


@pytest.fixture
def setup_data(three_ll_input):
    """Fixture to provide test data."""
    args = three_ll_input
    update = {}

    return args, update


@pytest.fixture
def dataframe(setup_data):
    args, update = setup_data
    # Default values
    return fetch_from_json(args)


### The main motivation for this test is to check that the lower the multiplier, the more levels are detected.
def test_ll_default(dataframe):
    levels = detect_liquidity_levels(dataframe, {})
    assert isinstance(
        levels, LiquidityLevels
    ), "Support and resistance levels should return a LiquidityLevels instance."
    assert (
        str(levels)
        == """
LiquidityLevel(price=1568.385)
LiquidityLevel(price=1587.8733333333332)
LiquidityLevel(price=1601.4725)
LiquidityLevel(price=1612.0033333333333)"""
    )


def test_ll_pro_mode(dataframe):
    levels = detect_liquidity_levels(dataframe, stdev_multiplier=0.05)
    assert isinstance(
        levels, LiquidityLevels
    ), "Support and resistance levels should return a LiquidityLevels instance."
    assert (
        str(levels)
        == """
LiquidityLevel(price=1568.385)
LiquidityLevel(price=1586.935)
LiquidityLevel(price=1599.76)
LiquidityLevel(price=1603.185)
LiquidityLevel(price=1610.975)"""
    )


def test_ll_pro_mode_2(dataframe):
    levels = detect_liquidity_levels(dataframe, stdev_multiplier=0.3)
    assert isinstance(
        levels, LiquidityLevels
    ), "Support and resistance levels should return a LiquidityLevels instance."
    assert (
        str(levels)
        == """
LiquidityLevel(price=1568.385)
LiquidityLevel(price=1600.3646153846153)"""
    )


def test_ll_comparison(dataframe):
    # Detect levels with different multipliers
    levels_no_multiplier = detect_liquidity_levels(dataframe, {})
    levels_0_05 = detect_liquidity_levels(dataframe, stdev_multiplier=0.05)
    levels_0_3 = detect_liquidity_levels(dataframe, stdev_multiplier=0.3)

    # Verify the number of levels
    assert len(levels_0_3.list) < len(
        levels_no_multiplier.list
    ), "Expected fewer levels with multiplier 0.3 than without multiplier."
    assert len(levels_no_multiplier.list) < len(
        levels_0_05.list
    ), "Expected fewer levels without multiplier than with multiplier 0.05."
