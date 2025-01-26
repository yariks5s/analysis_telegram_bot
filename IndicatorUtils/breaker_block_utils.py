class BreakerBlock:
    def __init__(self, block_type, index, zone):
        """
        Initialize a BreakerBlock object.

        Parameters:
            block_type (str): 'bullish' or 'bearish' to indicate the block type.
            index (pd.Timestamp): The index (timestamp) of the breaker block's candle.
            zone (tuple): A tuple representing the price range of the breaker block (low, high).
            covered (bool): Whether the breaker block has been revisited or retested.
        """
        self.block_type = block_type
        self.index = index
        self.zone = zone

    def __str__(self):
        """
        String representation of the breaker block.
        """
        return (
            f"BreakerBlock(type={self.block_type}, index={self.index}, "
            f"zone={self.zone})"
        )

    def is_bullish(self):
        """
        Check if the breaker block is bullish.
        """
        return self.block_type == "bullish"

    def is_bearish(self):
        """
        Check if the breaker block is bearish.
        """
        return self.block_type == "bearish"


class BreakerBlocks:
    def __init__(self):
        self.list = []

    def add(self, breaker_block):
        if isinstance(breaker_block, BreakerBlock):
            self.list.append(breaker_block)

    def __str__(self):
        if self.list:
            return "\n" + "\n".join(str(block) for block in self.list)
        else:
            return "\nNone."

    def __bool__(self):
        return bool(self.list)
