class FVG:
    def __init__(
        self, start_index, end_index, start_price, end_price, fvg_type
    ):
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

    def __str__(self):
        return (
            f"FVG(type={self.fvg_type}, start_index={self.start_index}, "
            f"end_index={self.end_index}, start_price={self.start_price}, "
            f"end_price={self.end_price}"
        )


class FVGs:
    def __init__(self):
        self.list = []

    def add(self, fvg):
        if isinstance(fvg, FVG):
            self.list.append(fvg)

    def __str__(self):
        if self.list:
            return "\n" + "\n".join(str(fvg) for fvg in self.list)
        else:
            return "\nNone."

    def non_verbose_str(self):
        if self.__bool__():
            return "\n" + "\n".join(str(fvg) for fvg in self.list)
        else:
            return "\nNone."

    def __bool__(self):
        if not self.list:
            return False
        for fvg in self.list:
            return True
        return False
