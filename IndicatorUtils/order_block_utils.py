class OrderBlock:
    def __init__(self, block_type, index, high, low):
        """
        Initialize an OrderBlock object.

        Parameters:
            block_type (str): 'bullish' or 'bearish' to indicate the block type.
            index (pd.Timestamp): The index (timestamp) of the order block's candle.
            high (float): The high of the order block.
            low (float): The low of the order block.
        """
        self.block_type = block_type
        self.index = index
        self.high = high
        self.low = low

    def __str__(self):
        return f"OrderBlock(type={self.block_type}, index={self.index}, high={self.high}, low={self.low})"

    def is_bullish(self):
        return self.block_type == "bullish"

    def is_bearish(self):
        return self.block_type == "bearish"


class OrderBlocks:
    def __init__(self):
        self.list = []

    def add(self, order_block):
        if isinstance(order_block, OrderBlock):
            self.list.append(order_block)

    def __str__(self):
        if self.list:
            return "\n" + "\n".join(str(block) for block in self.list)
        else:
            return "\nNone."

    def __bool__(self):
        return bool(self.list)
