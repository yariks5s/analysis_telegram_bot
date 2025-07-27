import pytest  # type: ignore
# Updated imports for the new structure
import src.analysis.detection.indicators as ind_module  # Import the indicators module

# Import functions from the indicators module
from src.analysis.detection.indicators import (
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity_levels,
    detect_breaker_blocks,
    detect_liquidity_pools,
)

# Import utility classes for indicators
from src.analysis.utils.breaker_block_utils import BreakerBlocks
from src.analysis.utils.fvg_utils import FVGs
from src.analysis.utils.liquidity_level_utils import LiquidityLevels
from src.analysis.utils.order_block_utils import OrderBlocks
from src.analysis.utils.liquidity_pool_utils import LiquidityPools

from src.api.data_fetcher import fetch_from_json


@pytest.fixture
def setup_ll(sample_response_for_ll):
    """Fixture to provide test data."""
    args = sample_response_for_ll
    update = {}

    return args, update


@pytest.fixture
def dataframe(advanced_indicators):
    args = advanced_indicators
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


def test_detect_liquidity_pools(dataframe):
    """Test detect_breaker_blocks function."""

    liquidity_pools = detect_liquidity_pools(dataframe)
    assert isinstance(
        liquidity_pools, LiquidityPools
    ), "Breaker blocks should return a BreakerBlocks instance."
    assert (
        str(liquidity_pools)
        == """
LiquidityPool(price=0.8063, volume=29421303.78, strength=1.0)
LiquidityPool(price=0.8140000000000001, volume=27996041.96, strength=1.0)"""
    )
