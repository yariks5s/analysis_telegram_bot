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

    def filter(self, selected):
        """
        Filter indicators based on user selection.
        """
        filtered = Indicators()
        if selected.get("order_blocks"):
            filtered.order_blocks = self.order_blocks
        if selected.get("fvgs"):
            filtered.fvgs = self.fvgs
        if selected.get("liquidity_levels"):
            filtered.liquidity_levels = self.liquidity_levels
        if selected.get("breaker_blocks"):
            filtered.breaker_blocks = self.breaker_blocks
        return filtered
