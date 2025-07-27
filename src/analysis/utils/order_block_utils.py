"""
Order Block utilities for CryptoBot.

This module defines classes for representing order blocks in price action analysis.
"""


class OrderBlock:
    """
    Class representing a single Order Block instance.

    An Order Block is a type of supply/demand zone that precedes a strong price move.
    Bullish order blocks represent potential support zones, while bearish order blocks
    represent potential resistance zones.
    """

    def __init__(self, block_type, index, high, low):
        """
        Initialize an OrderBlock instance.

        Args:
            block_type (str): Type of order block - "bullish" or "bearish"
            index (int): Candle index in the DataFrame where this order block was detected
            high (float): Upper price boundary of the order block
            low (float): Lower price boundary of the order block
        """
        self.block_type = block_type
        self.index = index
        self.high = high
        self.low = low
        self.pos = None  # Optional position index for internal use

    def __str__(self):
        return f"\nOrderBlock(type={self.block_type}, index={self.index}, high={self.high}, low={self.low})"

    def __repr__(self):
        return self.__str__()

    def is_bullish(self):
        return self.block_type == "bullish"

    def is_bearish(self):
        return self.block_type == "bearish"


class OrderBlocks:
    """
    Container class for managing multiple OrderBlock instances.
    """

    def __init__(self):
        """Initialize an empty list of OrderBlock instances."""
        self.list = []

    def add(self, order_block):
        """
        Add an OrderBlock to the container.

        Args:
            order_block (OrderBlock): The OrderBlock instance to add
        """
        self.list.append(order_block)

    def __str__(self):
        """String representation of all contained OrderBlocks."""
        if not self.list:
            return "\nNone."
        return "".join(str(ob) for ob in self.list)

    def __repr__(self):
        return self.__str__()
