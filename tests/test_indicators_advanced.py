import pytest  # type: ignore
from indicators import (
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity_levels,
    detect_breaker_blocks,
)
from IndicatorUtils.breaker_block_utils import BreakerBlocks
from IndicatorUtils.fvg_utils import FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevels
from IndicatorUtils.order_block_utils import OrderBlocks

from data_fetching_instruments import fetch_from_json


@pytest.fixture
def setup_data():
    """Fixture to provide test data."""
    args = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "category": "spot",
            "symbol": "ADAUSDT",
            "list": [
                [
                    "1734717600000",
                    "0.8864",
                    "0.8918",
                    "0.8802",
                    "0.8806",
                    "1718006.72",
                    "1524108.952454",
                ],
                [
                    "1734714000000",
                    "0.8784",
                    "0.8905",
                    "0.8748",
                    "0.8864",
                    "2357034.61",
                    "2080101.00459",
                ],
                [
                    "1734710400000",
                    "0.8759",
                    "0.896",
                    "0.8694",
                    "0.8784",
                    "4540774.46",
                    "3995051.10712",
                ],
                [
                    "1734706800000",
                    "0.8548",
                    "0.8912",
                    "0.8529",
                    "0.8759",
                    "6785736.93",
                    "5915563.667385",
                ],
                [
                    "1734703200000",
                    "0.861",
                    "0.8649",
                    "0.8444",
                    "0.8548",
                    "3392823.91",
                    "2902433.142977",
                ],
                [
                    "1734699600000",
                    "0.8309",
                    "0.8662",
                    "0.7987",
                    "0.861",
                    "13772176.91",
                    "11572106.340974",
                ],
                [
                    "1734696000000",
                    "0.7623",
                    "0.8349",
                    "0.7618",
                    "0.8309",
                    "10831041.14",
                    "8668030.270388",
                ],
                [
                    "1734692400000",
                    "0.7941",
                    "0.8097",
                    "0.7623",
                    "0.7623",
                    "11119622.19",
                    "8708597.281615",
                ],
                [
                    "1734688800000",
                    "0.8184",
                    "0.8236",
                    "0.7846",
                    "0.7941",
                    "9528974.5",
                    "7623319.707574",
                ],
                [
                    "1734685200000",
                    "0.8421",
                    "0.8503",
                    "0.8184",
                    "0.8184",
                    "8772707.09",
                    "7307436.993643",
                ],
                [
                    "1734681600000",
                    "0.8889",
                    "0.8897",
                    "0.8362",
                    "0.8421",
                    "5302445.37",
                    "4531357.077279",
                ],
                [
                    "1734678000000",
                    "0.8785",
                    "0.906",
                    "0.8784",
                    "0.8889",
                    "2915921.25",
                    "2615134.590824",
                ],
                [
                    "1734674400000",
                    "0.8847",
                    "0.8947",
                    "0.871",
                    "0.8785",
                    "1791737.69",
                    "1582550.93245",
                ],
                [
                    "1734670800000",
                    "0.9",
                    "0.9041",
                    "0.8839",
                    "0.8847",
                    "1306534.55",
                    "1168168.901247",
                ],
                [
                    "1734667200000",
                    "0.9047",
                    "0.9089",
                    "0.8983",
                    "0.9",
                    "2209536.75",
                    "1997962.670153",
                ],
            ],
        },
        "retExtInfo": {},
        "time": 1734719712626,
    }
    update = {}

    return args, update


@pytest.fixture
def setup_ll(sample_response_for_ll):
    """Fixture to provide test data."""
    args = sample_response_for_ll
    update = {}

    return args, update


@pytest.fixture
def dataframe(setup_data):
    args, update = setup_data
    # Default values
    return fetch_from_json(args)


@pytest.fixture
def dataframe_ll(setup_ll):
    args, _ = setup_ll
    return fetch_from_json(args)


def test_detect_order_blocks(dataframe):

    order_blocks = detect_order_blocks(dataframe)
    assert isinstance(
        order_blocks, OrderBlocks
    ), "Order blocks should return an OrderBlocks instance."
    assert (
        str(order_blocks)
        == """
OrderBlock(type=bullish, index=7, high=0.8097, low=0.7623)"""
    )


def test_detect_fvgs(dataframe):
    """Test detect_fvgs function."""
    fvgs = detect_fvgs(dataframe)
    assert isinstance(fvgs, FVGs), "FVGs should return a FVGs instance."
    assert (
        str(fvgs)
        == """
FVG(type=bullish, start_index=8, end_index=10, start_price=0.8349, end_price=0.8444)
FVG(type=bullish, start_index=10, end_index=12, start_price=0.8649, end_price=0.8694)"""
    )


def test_detect_liquidity_levels(dataframe_ll):
    """Test detect_liquidity_levels function."""

    levels = detect_liquidity_levels(dataframe_ll, {})
    assert isinstance(
        levels, LiquidityLevels
    ), "Support and resistance levels should return a LiquidityLevels instance."
    assert (
        str(levels)
        == """
LiquidityLevel(price=7.195e-06)"""
    )


def test_detect_breaker_blocks(dataframe_ll):
    """Test detect_breaker_blocks function."""

    levels = detect_liquidity_levels(dataframe_ll, {})
    breaker_blocks = detect_breaker_blocks(dataframe_ll, levels)
    assert isinstance(
        breaker_blocks, BreakerBlocks
    ), "Breaker blocks should return a BreakerBlocks instance."
    assert (
        str(breaker_blocks)
        == """
BreakerBlock(type=bullish, index=20, zone=(7.02e-06, 7.24e-06))
BreakerBlock(type=bullish, index=23, zone=(7.04e-06, 7.22e-06))
BreakerBlock(type=bullish, index=25, zone=(7.18e-06, 7.27e-06))
BreakerBlock(type=bullish, index=26, zone=(7.18e-06, 7.25e-06))
BreakerBlock(type=bullish, index=33, zone=(7.17e-06, 7.27e-06))
BreakerBlock(type=bullish, index=35, zone=(7.19e-06, 7.25e-06))
BreakerBlock(type=bullish, index=36, zone=(7.18e-06, 7.25e-06))"""
    )
