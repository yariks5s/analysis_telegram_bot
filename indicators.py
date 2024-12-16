from helpers import logger

# def detect_order_blocks(df):
#     """
#     Detect order blocks based on bullish or bearish breakouts.

#     Parameters:
#         df (pd.DataFrame): A DataFrame containing OHLCV data with columns ['Open', 'High', 'Low', 'Close'].

#     Returns:
#         list: A list of tuples representing order blocks, where each tuple is:
#               (index, high, low, type)
#               - index: Index of the order block candle
#               - high: High of the candle
#               - low: Low of the candle
#               - type: 'bullish' or 'bearish'
#     """
#     order_blocks = []

#     for i in range(1, len(df) - 2):  # Avoid the first and last two candles
#         # Bearish Order Block
#         if df['Close'][i] < df['Open'][i]:  # Bearish candle
#             if df['Close'][i + 1] > df['High'][i]:  # Bullish breakout
#                 order_blocks.append((i, df['High'][i], df['Low'][i], 'bearish'))

#         # Bullish Order Block
#         elif df['Close'][i] > df['Open'][i]:  # Bullish candle
#             if df['Close'][i + 1] < df['Low'][i]:  # Bearish breakout
#                 order_blocks.append((i, df['High'][i], df['Low'][i], 'bullish'))

#     return order_blocks

def detect_order_blocks(df, volume_threshold=1.5, body_percentage=0.5, breakout_factor=1.01):
    """
    Detect less sensitive order blocks based on bullish or bearish breakouts.

    Parameters:
        df (pd.DataFrame): A DataFrame containing OHLCV data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].
        volume_threshold (float): Multiplier to determine significant volume (e.g., 1.5x the average).
        body_percentage (float): Minimum body size as a percentage of the candle's range to be considered significant.
        breakout_factor (float): Multiplier to determine the strength of the breakout (e.g., 1.01 for 1% breakout).

    Returns:
        list: A list of tuples representing order blocks, where each tuple is:
              (index, high, low, type)
              - index: Index of the order block candle
              - high: High of the candle
              - low: Low of the candle
              - type: 'bullish' or 'bearish'
    """
    order_blocks = []
    avg_volume = df['Volume'].mean()

    for i in range(1, len(df) - 3):
        high = df['High'][i]
        low = df['Low'][i]
        close = df['Close'][i]
        open_price = df['Open'][i]
        volume = df['Volume'][i]

        body_size = abs(close - open_price)
        range_size = high - low

        # Skip small candles
        if range_size == 0 or body_size / range_size < body_percentage:
            continue

        # Skip low-volume candles
        if volume < volume_threshold * avg_volume:
            continue

        # Detect order blocks
        if close < open_price and df['Close'][i + 1] > high * breakout_factor:
            order_blocks.append((i, high, low, 'bearish'))
        elif close > open_price and df['Close'][i + 1] < low * breakout_factor:
            order_blocks.append((i, high, low, 'bullish'))

    return order_blocks

def detect_fvgs(df):
    """
    Detect Fair Value Gaps (FVGs) in price action and check if they are covered later.

    Parameters:
        df (pd.DataFrame): A DataFrame containing OHLCV data with columns ['Open', 'High', 'Low', 'Close'].

    Returns:
        list: A list of tuples representing FVGs, where each tuple is:
              (start_index, end_index, start_price, end_price, type, covered)
              - start_index: Start of the gap
              - end_index: End of the gap
              - start_price: Price at the start of the gap
              - end_price: Price at the end of the gap
              - type: 'bullish' or 'bearish'
              - covered: Boolean indicating whether the FVG was later covered
    """
    fvgs = []

    for i in range(2, len(df)):
        logger.info(f"df['Low'][i]: {df['Low'][i]}, df['High'][i - 1]: {df['High'][i - 2]}")

        # Bullish FVG
        if df['Low'][i] > df['High'][i - 2]:  # Gap up
            fvgs.append((i - 2, i, df['High'][i - 2], df['Low'][i], 'bullish', False))

        # Bearish FVG
        elif df['High'][i] < df['Low'][i - 2]:  # Gap down
            fvgs.append((i - 2, i, df['Low'][i - 2], df['High'][i], 'bearish', False))

    # Check if FVGs are covered
    for idx, (start_idx, end_idx, start_price, end_price, fvg_type, _) in enumerate(fvgs):
        for j in range(end_idx + 1, len(df)):
            # Check if the gap is covered
            if fvg_type == 'bullish' and df['Low'][j] <= start_price:
                fvgs[idx] = (start_idx, end_idx, start_price, end_price, fvg_type, True)
                break
            elif fvg_type == 'bearish' and df['High'][j] >= end_price:
                fvgs[idx] = (start_idx, end_idx, start_price, end_price, fvg_type, True)
                break

    return fvgs