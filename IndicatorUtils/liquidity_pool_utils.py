import pandas as pd
import numpy as np
from typing import List, Optional


class LiquidityPool:
    def __init__(self, price, volume, strength):
        """
        Initialize a LiquidityPool object.

        Parameters:
            price (float): The price level of the liquidity pool.
            volume (float): The trading volume at this level.
            strength (float): A measure of the pool's strength (0-1).
        """
        self.price = price
        self.volume = volume
        self.strength = strength

    def __str__(self):
        return f"LiquidityPool(price={self.price}, volume={self.volume}, strength={self.strength})"


class LiquidityPools:
    def __init__(self):
        self.list = []

    def __bool__(self):
        return bool(self.list)

    def add(self, pool):
        if isinstance(pool, LiquidityPool):
            self.list.append(pool)

    def __str__(self):
        if self.list:
            return "\n" + "\n".join(str(pool) for pool in self.list)
        else:
            return "\nNone."


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute the Average True Range (ATR) over a specified period.
    """
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def detect_liquidity_pools(
    df: pd.DataFrame,
    window: int = 20,  # Window for local high/low detection
    consolidation_window: int = 50,  # Window for consolidation detection
    round_numbers: Optional[List[float]] = None,  # List of round numbers to check
    atr_period: int = 14,  # Period for ATR calculation
    atr_multiplier: float = 0.5,  # Multiplier for ATR to determine price range
    min_touches: int = 3,  # Minimum number of touches to consider a level valid
) -> LiquidityPools:
    """
    Detect liquidity pools at key price levels where stop losses are likely to be clustered.

    The function looks for liquidity pools:
    1. Above local highs (short stop losses)
    2. Below local lows (long stop losses)
    3. At the boundaries of consolidation/flat ranges
    4. Near round numbers (if specified)

    Parameters:
        df (pd.DataFrame): DataFrame with OHLCV data
        window (int): Window size for local high/low detection
        consolidation_window (int): Window size for consolidation detection
        round_numbers (List[float]): List of round numbers to check for liquidity
        atr_period (int): Period for ATR calculation
        atr_multiplier (float): Multiplier for ATR to determine price range
        min_touches (int): Minimum number of touches to consider a level valid

    Returns:
        LiquidityPools: Container with detected liquidity pools
    """
    pools = LiquidityPools()

    # Calculate ATR for volatility-based thresholds
    atr = compute_atr(df, period=atr_period)
    price_range = atr * atr_multiplier

    # 1. Detect liquidity above local highs (short stop losses)
    for i in range(window, len(df) - window):
        # Check if current high is a local high
        if all(df["High"].iloc[i] > df["High"].iloc[i - window : i]) and all(
            df["High"].iloc[i] > df["High"].iloc[i + 1 : i + window + 1]
        ):
            # Look for liquidity pool above the high
            pool_high = df["High"].iloc[i] + price_range.iloc[i]
            pool_low = df["High"].iloc[i]

            # Count touches within the pool range
            touches = 0
            for j in range(i + 1, min(i + window * 2, len(df))):
                if df["Low"].iloc[j] <= pool_high and df["High"].iloc[j] >= pool_low:
                    touches += 1

            if touches >= min_touches:
                pools.add(
                    LiquidityPool(
                        price=(pool_high + pool_low) / 2,
                        volume=df["Volume"].iloc[i],
                        strength=min(1.0, touches / min_touches),
                    )
                )

    # 2. Detect liquidity below local lows (long stop losses)
    for i in range(window, len(df) - window):
        # Check if current low is a local low
        if all(df["Low"].iloc[i] < df["Low"].iloc[i - window : i]) and all(
            df["Low"].iloc[i] < df["Low"].iloc[i + 1 : i + window + 1]
        ):
            # Look for liquidity pool below the low
            pool_high = df["Low"].iloc[i]
            pool_low = df["Low"].iloc[i] - price_range.iloc[i]

            # Count touches within the pool range
            touches = 0
            for j in range(i + 1, min(i + window * 2, len(df))):
                if df["Low"].iloc[j] <= pool_high and df["High"].iloc[j] >= pool_low:
                    touches += 1

            if touches >= min_touches:
                pools.add(
                    LiquidityPool(
                        price=(pool_high + pool_low) / 2,
                        volume=df["Volume"].iloc[i],
                        strength=min(1.0, touches / min_touches),
                    )
                )

    # 3. Detect liquidity at consolidation boundaries
    for i in range(consolidation_window, len(df)):
        # Calculate price range of consolidation
        high_range = df["High"].iloc[i - consolidation_window : i].max()
        low_range = df["Low"].iloc[i - consolidation_window : i].min()
        range_size = high_range - low_range

        # If range is relatively small compared to ATR, it's a consolidation
        if range_size <= atr.iloc[i] * 2:
            # Check for liquidity at the boundaries
            upper_touches = 0
            lower_touches = 0

            for j in range(i - consolidation_window, i):
                if abs(df["High"].iloc[j] - high_range) <= price_range.iloc[i]:
                    upper_touches += 1
                if abs(df["Low"].iloc[j] - low_range) <= price_range.iloc[i]:
                    lower_touches += 1

            if upper_touches >= min_touches:
                pools.add(
                    LiquidityPool(
                        price=high_range,
                        volume=df["Volume"].iloc[i - consolidation_window : i].mean(),
                        strength=min(1.0, upper_touches / min_touches),
                    )
                )

            if lower_touches >= min_touches:
                pools.add(
                    LiquidityPool(
                        price=low_range,
                        volume=df["Volume"].iloc[i - consolidation_window : i].mean(),
                        strength=min(1.0, lower_touches / min_touches),
                    )
                )

    # 4. Check for liquidity near round numbers
    if round_numbers:
        for round_num in round_numbers:
            touches = 0
            for i in range(len(df)):
                if (
                    abs(df["High"].iloc[i] - round_num) <= price_range.iloc[i]
                    or abs(df["Low"].iloc[i] - round_num) <= price_range.iloc[i]
                ):
                    touches += 1

            if touches >= min_touches:
                pools.add(
                    LiquidityPool(
                        price=round_num,
                        volume=df["Volume"].iloc[-window:].mean(),
                        strength=min(1.0, touches / min_touches),
                    )
                )

    return pools
