"""
Fair Value Gap (FVG) utilities for CryptoBot.

This module defines classes for representing Fair Value Gaps in price action analysis.
"""


class FVG:
    """
    Class representing a single Fair Value Gap (FVG) instance.

    A Fair Value Gap is a price inefficiency that occurs when price moves too quickly
    in one direction, leaving an unfilled gap. Bullish FVGs represent potential support zones,
    while bearish FVGs represent potential resistance zones.
    """

    def __init__(
        self,
        start_index=None,
        end_index=None,
        start_price=None,
        end_price=None,
        fvg_type="bullish",
    ):
        """
        Initialize an FVG instance.

        Args:
            start_index (int): Starting candle index in the DataFrame
            end_index (int): Ending candle index in the DataFrame
            start_price (float): Starting price of the gap
            end_price (float): Ending price of the gap
            fvg_type (str): Type of FVG - "bullish" or "bearish"
        """
        self.fvg_type = fvg_type
        self.start_index = start_index
        self.end_index = end_index
        self.start_price = start_price
        self.end_price = end_price

    def __str__(self):
        return f"\nFVG(type={self.fvg_type}, start_index={self.start_index}, end_index={self.end_index}, start_price={self.start_price}, end_price={self.end_price})"

    def __repr__(self):
        return self.__str__()


class FVGs:
    """
    Container class for managing multiple FVG instances.
    """

    def __init__(self):
        """Initialize an empty list of FVG instances."""
        self.list = []

    def add(self, fvg):
        """
        Add an FVG to the container.

        Args:
            fvg (FVG): The FVG instance to add
        """
        self.list.append(fvg)

    def non_verbose_str(self):
        if self.__bool__():
            return "\n" + "\n".join(str(fvg) for fvg in self.list)
        else:
            return "\nNone."

    def __str__(self):
        """String representation of all contained FVGs."""
        if not self.list:
            return "\nNone."
        return "".join(str(fvg) for fvg in self.list)

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        if not self.list:
            return False
        for fvg in self.list:
            return True
        return False
