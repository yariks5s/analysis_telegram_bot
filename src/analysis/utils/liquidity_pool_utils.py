"""
Liquidity Pool utilities for CryptoBot.

This module defines classes for representing liquidity pools in price action analysis.
"""


class LiquidityPool:
    """
    Class representing a single Liquidity Pool instance.

    A Liquidity Pool is a price zone with concentrated volume where price tends to
    hover or trade within a tight range. These areas often represent potential support
    or resistance levels.
    """

    def __init__(self, price, volume, strength):
        """
        Initialize a LiquidityPool instance.

        Args:
            price (float): The central price of the liquidity pool
            volume (float): The volume observed in this pool
            strength (float): The strength of the pool (0.0 to 1.0)
        """
        self.price = price
        self.volume = volume
        self.strength = strength

    def __str__(self):
        return f"\nLiquidityPool(price={self.price}, volume={self.volume}, strength={self.strength})"

    def __repr__(self):
        return self.__str__()


class LiquidityPools:
    """
    Container class for managing multiple LiquidityPool instances.
    """

    def __init__(self):
        """Initialize an empty list of LiquidityPool instances."""
        self.list = []

    def add(self, liquidity_pool):
        """
        Add a LiquidityPool to the container.

        Args:
            liquidity_pool (LiquidityPool): The LiquidityPool instance to add
        """
        self.list.append(liquidity_pool)

    def __str__(self):
        """String representation of all contained LiquidityPools."""
        if not self.list:
            return "\nNone."
        return "".join(str(lp) for lp in self.list)

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        return bool(self.list)
