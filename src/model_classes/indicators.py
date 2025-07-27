"""
Indicator model classes for CryptoBot.

This module contains the base indicator models for storing technical analysis results.
These classes serve as containers for various trading indicators detected in price action.
"""

# These imports will need to be updated after we reorganize the IndicatorUtils modules
from IndicatorUtils.order_block_utils import OrderBlocks
from IndicatorUtils.fvg_utils import FVGs
from IndicatorUtils.liquidity_level_utils import LiquidityLevels
from IndicatorUtils.breaker_block_utils import BreakerBlocks
from IndicatorUtils.liquidity_pool_utils import LiquidityPools


class Indicators:
    """
    Container class for all technical analysis indicators.
    
    Provides methods to store, filter, and represent various indicator types
    including order blocks, FVGs, liquidity levels, breaker blocks, and liquidity pools.
    """
    
    def __init__(self):
        """
        Initialize an Indicators object to store various technical analysis features.
        """
        self.order_blocks = OrderBlocks()
        self.fvgs = FVGs()
        self.liquidity_levels = LiquidityLevels()
        self.breaker_blocks = BreakerBlocks()
        self.liquidity_pools = LiquidityPools()

    def __str__(self):
        """
        String representation of all indicators.
        
        Returns:
            str: Formatted string representation of all indicator data
        """
        return (
            f"**Indicators:**\n"
            f"**Order Blocks:**{self.order_blocks}\n\n"
            f"**Fair Value Gaps (FVGs):**{self.fvgs.non_verbose_str()}\n\n"
            f"**Liquidity Levels:**{self.liquidity_levels}\n\n"
            f"**Breaker Blocks:**{self.breaker_blocks}\n\n"
            f"**Liquidity Pools:**{self.liquidity_pools}\n"
        )

    def filter(self, selected):
        """
        Filter indicators based on user selection.
        
        Args:
            selected (dict): Dictionary with indicator types as keys and boolean values
            
        Returns:
            Indicators: A new Indicators object with only the selected indicators
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
        if selected.get("liquidity_pools"):
            filtered.liquidity_pools = self.liquidity_pools
        return filtered
