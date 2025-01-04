import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from plot_build_helpers import plot_price_chart
from IndicatorUtils.fvg_utils import FVGs
from IndicatorUtils.breaker_block_utils import BreakerBlocks
from IndicatorUtils.liquidity_level_utils import LiquidityLevels
from IndicatorUtils.order_block_utils import OrderBlocks


from unittest.mock import Mock

def test_plot_price_chart(sample_dataframe):
    mock_indicators = Mock()
    mock_indicators.fvgs = FVGs()
    mock_indicators.order_blocks = OrderBlocks()
    mock_indicators.liquidity_levels = LiquidityLevels()
    mock_indicators.breaker_blocks = BreakerBlocks()

    chart_path = plot_price_chart(sample_dataframe, mock_indicators)
    assert chart_path == "crypto_chart.png"
