"""
Breaker Block utilities for CryptoBot.

This module defines classes for representing breaker blocks in price action analysis.
"""


class BreakerBlock:
    """
    Class representing a single Breaker Block instance.
    
    A Breaker Block occurs when price sweeps a significant liquidity level and then reverses.
    These often indicate potential reversal zones.
    """
    
    def __init__(self, block_type, index, zone):
        """
        Initialize a BreakerBlock instance.
        
        Args:
            block_type (str): Type of breaker block - "bullish" or "bearish"
            index (int): Candle index in the DataFrame where the breaker block was detected
            zone (tuple): (lower_bound, upper_bound) price range of the breaker block
        """
        self.block_type = block_type
        self.index = index
        self.zone = zone
        
    def __str__(self):
        return f"\nBreakerBlock(type={self.block_type}, index={self.index}, zone={self.zone})"
    
    def __repr__(self):
        return self.__str__()


class BreakerBlocks:
    """
    Container class for managing multiple BreakerBlock instances.
    """
    
    def __init__(self):
        """Initialize an empty list of BreakerBlock instances."""
        self.list = []
        
    def add(self, breaker_block):
        """
        Add a BreakerBlock to the container.
        
        Args:
            breaker_block (BreakerBlock): The BreakerBlock instance to add
        """
        self.list.append(breaker_block)
        
    def __str__(self):
        """String representation of all contained BreakerBlocks."""
        if not self.list:
            return "\nNone."
        return "".join(str(bb) for bb in self.list)
    
    def __repr__(self):
        return self.__str__()
