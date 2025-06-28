import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from plot_build_helpers import plot_price_chart
from IndicatorUtils.fvg_utils import FVGs
from IndicatorUtils.breaker_block_utils import BreakerBlocks
from IndicatorUtils.liquidity_level_utils import LiquidityLevels
from IndicatorUtils.order_block_utils import OrderBlocks
from IndicatorUtils.liquidity_pool_utils import LiquidityPools


from unittest.mock import Mock


def test_plot_price_chart_light_mode(sample_dataframe):
    """Test chart generation with light mode (default)"""
    mock_indicators = Mock()
    mock_indicators.fvgs = FVGs()
    mock_indicators.order_blocks = OrderBlocks()
    mock_indicators.liquidity_levels = LiquidityLevels()
    mock_indicators.breaker_blocks = BreakerBlocks()
    mock_indicators.liquidity_pools = LiquidityPools()

    chart_path = plot_price_chart(
        sample_dataframe,
        mock_indicators,
        show_legend=True,
        show_volume=True,
        dark_mode=False,
    )
    assert chart_path == "crypto_chart.png"


def test_plot_price_chart_dark_mode(sample_dataframe):
    """Test chart generation with dark mode enabled"""
    mock_indicators = Mock()
    mock_indicators.fvgs = FVGs()
    mock_indicators.order_blocks = OrderBlocks()
    mock_indicators.liquidity_levels = LiquidityLevels()
    mock_indicators.breaker_blocks = BreakerBlocks()
    mock_indicators.liquidity_pools = LiquidityPools()

    chart_path = plot_price_chart(
        sample_dataframe,
        mock_indicators,
        show_legend=True,
        show_volume=True,
        dark_mode=True,
    )
    assert chart_path == "crypto_chart.png"


def test_plot_price_chart_with_preferences(sample_dataframe):
    """Test chart generation with various preference combinations"""
    mock_indicators = Mock()
    mock_indicators.fvgs = FVGs()
    mock_indicators.order_blocks = OrderBlocks()
    mock_indicators.liquidity_levels = LiquidityLevels()
    mock_indicators.breaker_blocks = BreakerBlocks()
    mock_indicators.liquidity_pools = LiquidityPools()

    # Test with legend off and dark mode on
    chart_path = plot_price_chart(
        sample_dataframe,
        mock_indicators,
        show_legend=False,
        show_volume=True,
        dark_mode=True,
    )
    assert chart_path == "crypto_chart.png"

    # Test with volume off and light mode on
    chart_path = plot_price_chart(
        sample_dataframe,
        mock_indicators,
        show_legend=True,
        show_volume=False,
        dark_mode=False,
    )
    assert chart_path == "crypto_chart.png"
