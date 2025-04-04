import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)
from indicators import (
    detect_order_blocks,
    detect_fvgs,
    detect_liquidity_levels,
    detect_breaker_blocks,
)
from IndicatorUtils.breaker_block_utils import BreakerBlocks


def test_detect_order_blocks(sample_dataframe):
    order_blocks = detect_order_blocks(sample_dataframe)
    assert isinstance(order_blocks.list, list)


def test_detect_fvgs(sample_dataframe):
    fvgs = detect_fvgs(sample_dataframe)
    assert isinstance(fvgs.list, list)


def test_detect_liquidity_levels(sample_dataframe):
    levels = detect_liquidity_levels(sample_dataframe, window=2, tolerance=0.1)
    assert isinstance(levels.list, list)


def test_detect_breaker_blocks(sample_dataframe):
    levels = detect_liquidity_levels(sample_dataframe)
    breaker_blocks = detect_breaker_blocks(sample_dataframe, levels)
    assert isinstance(breaker_blocks, BreakerBlocks)
