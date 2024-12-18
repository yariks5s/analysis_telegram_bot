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
        return self.block_type == 'bullish'

    def is_bearish(self):
        return self.block_type == 'bearish'
    
    def get_index(self):
        return self.index
    
    def get_high(self):
        return self.high
    
    def get_low(self):
        return self.low


class FVG:
    def __init__(self, start_index, end_index, start_price, end_price, fvg_type, covered=False):
        """
        Initialize a Fair Value Gap (FVG) object.

        Parameters:
            start_index (pd.Timestamp): The start of the gap.
            end_index (pd.Timestamp): The end of the gap.
            start_price (float): Price at the start of the gap.
            end_price (float): Price at the end of the gap.
            fvg_type (str): 'bullish' or 'bearish' to indicate the gap type.
            covered (bool): Whether the gap has been covered.
        """
        self.start_index = start_index
        self.end_index = end_index
        self.start_price = start_price
        self.end_price = end_price
        self.fvg_type = fvg_type
        self.covered = covered

    def __str__(self):
        return (f"FVG(type={self.fvg_type}, start_index={self.start_index}, "
                f"end_index={self.end_index}, start_price={self.start_price}, "
                f"end_price={self.end_price}, covered={self.covered})")

    def is_covered(self):
        return self.covered
    
    def get_start_index(self):
        return self.start_index
    
    def get_end_index(self):
        return self.end_index
    
    def get_start_price(self):
        return self.start_price
    
    def get_end_price(self):
        return self.end_price


class LiquidityLevel:
    def __init__(self, level_type, price):
        """
        Initialize a LiquidityLevel object.

        Parameters:
            level_type (str): 'support' or 'resistance'.
            price (float): The price of the liquidity level.
        """
        self.level_type = level_type
        self.price = price

    def __str__(self):
        return f"LiquidityLevel(type={self.level_type}, price={self.price})"

    def is_support(self):
        return self.level_type == 'support'

    def is_resistance(self):
        return self.level_type == 'resistance'
    
    def get_price(self):
        return self.price


class BreakerBlock:
    def __init__(self, block_type, index, zone, covered=False):
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
        self.covered = covered

    def __str__(self):
        """
        String representation of the breaker block.
        """
        return (f"BreakerBlock(type={self.block_type}, index={self.index}, "
                f"zone={self.zone}, covered={self.covered})")

    def is_bullish(self):
        """
        Check if the breaker block is bullish.
        """
        return self.block_type == 'bullish'

    def is_bearish(self):
        """
        Check if the breaker block is bearish.
        """
        return self.block_type == 'bearish'
    
    def get_index(self):
        return self.index
    
    def get_zone(self):
        return self.zone
    
    def is_covered(self):
        return self.is_covered
    
