class LiquidityLevel:
    def __init__(self, price):
        """
        Initialize a unified LiquidityLevel object.

        Parameters:
            price (float): The price of the liquidity level.
        """
        self.price = price

    def __str__(self):
        return f"LiquidityLevel(price={self.price})"


class LiquidityLevels:
    def __init__(self):
        self.list = []

    def __bool__(self):
        return bool(self.list)

    def add(self, level):
        if isinstance(level, LiquidityLevel):
            self.list.append(level)

    def __str__(self):
        if self.list:
            return "\n" + "\n".join(str(level) for level in self.list)
        else:
            return "\nNone."

    def __bool__(self):
        return bool(self.list)
