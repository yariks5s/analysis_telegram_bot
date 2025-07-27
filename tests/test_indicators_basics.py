import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

# Import indicator detection functions from new location
from src.analysis.detection.indicators import (
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity_levels,
    detect_breaker_blocks,
    detect_liquidity_pools,
)

# Import indicator utility classes from new location
from src.analysis.utils.breaker_block_utils import BreakerBlocks
from src.analysis.utils.fvg_utils import FVGs
from src.analysis.utils.order_block_utils import OrderBlocks
from src.analysis.utils.liquidity_level_utils import LiquidityLevels
from src.analysis.utils.liquidity_pool_utils import LiquidityPools


def test_detect_order_blocks(sample_dataframe):
    order_blocks = detect_order_blocks(sample_dataframe)
    assert isinstance(order_blocks, OrderBlocks)


def test_detect_fvgs(sample_dataframe):
    fvgs = detect_fvgs(sample_dataframe)
    assert isinstance(fvgs, FVGs)


def test_detect_liquidity_levels(sample_dataframe):
    levels = detect_liquidity_levels(sample_dataframe, stdev_multiplier={})
    assert isinstance(levels, LiquidityLevels)


def test_detect_breaker_blocks(sample_dataframe):
    levels = detect_liquidity_levels(sample_dataframe, stdev_multiplier={})
    breaker_blocks = detect_breaker_blocks(sample_dataframe, levels)
    assert isinstance(breaker_blocks, BreakerBlocks)


def test_detect_liquidity_pools(sample_dataframe):
    liquidity_pools = detect_liquidity_pools(sample_dataframe)
    assert isinstance(liquidity_pools, LiquidityPools)
