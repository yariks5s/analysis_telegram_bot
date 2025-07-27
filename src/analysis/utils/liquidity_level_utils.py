"""
Liquidity Level utilities for CryptoBot.

This module defines classes for representing liquidity levels in price action analysis.
"""


class LiquidityLevel:
    """
    Class representing a single Liquidity Level instance.
    
    Liquidity levels are price zones where significant buying or selling interest exists.
    These levels often act as support or resistance zones where price tends to react.
    """
    
    def __init__(self, price):
        """
        Initialize a LiquidityLevel instance.
        
        Args:
            price (float): Price level of the liquidity zone
        """
        self.price = price
        
    def __str__(self):
        return f"\nLiquidityLevel(price={self.price})"
    
    def __repr__(self):
        return self.__str__()


class LiquidityLevels:
    """
    Container class for managing multiple LiquidityLevel instances.
    """
    
    def __init__(self):
        """Initialize an empty list of LiquidityLevel instances."""
        self.list = []
        
    def add(self, level):
        """
        Add a LiquidityLevel to the container.
        
        Args:
            level (LiquidityLevel): The LiquidityLevel instance to add
        """
        self.list.append(level)
        
    def __str__(self):
        """String representation of all contained LiquidityLevels."""
        if not self.list:
            return "\nNone."
        return "".join(str(level) for level in self.list)
    
    def __repr__(self):
        return self.__str__()
    
    def __bool__(self):
        return bool(self.list)
