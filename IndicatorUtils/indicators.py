from IndicatorUtils.order_block_utils import OrderBlocks
from IndicatorUtils.fvg_utils import FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevels
from IndicatorUtils.breaker_block_utils import BreakerBlocks

class Indicators:
    def __init__(self):
        """
        Initialize an Indicators object to store various technical analysis features.
        """
        self.order_blocks = OrderBlocks()
        self.fvgs = FVGs()
        self.liquidity_levels = LiquidityLevels()
        self.breaker_blocks = BreakerBlocks()

    def __str__(self):
        """
        String representation of all indicators.
        """
        return (
            f"**Indicators:**\n"
            f"**Order Blocks:**{self.order_blocks}\n\n"
            f"**Fair Value Gaps (FVGs):**{self.fvgs.non_verbose_str()}\n\n"
            f"**Liquidity Levels:**{self.liquidity_levels}\n\n"
            f"**Breaker Blocks:**{self.breaker_blocks}\n"
        )
