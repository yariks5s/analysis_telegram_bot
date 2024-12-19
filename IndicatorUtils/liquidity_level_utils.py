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


class LiquidityLevels:
    def __init__(self):
        self.list = []

    def __bool__(self):
        return bool(self.list)

    def add(self, level):
        if isinstance(level, LiquidityLevel):
            self.list.append(level)

    def __str__(self):
        if (self.list):
            return "\n" + "\n".join(str(level) for level in self.list)
        else:
            return "\nNone."

    def __bool__(self):
        return bool(self.list)
