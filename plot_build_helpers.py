import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import mplfinance as mpf
import pandas as pd

from helpers import logger
from indicators import detect_order_blocks, detect_fvgs, detect_support_resistance_levels, detect_breaker_blocks


def plot_price_chart(df: pd.DataFrame, liq_lev_tolerance: float):
    """
    Generate a candlestick chart with detected order blocks, FVGs, and support/resistance levels as horizontal lines.
    """

    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvgs(df)

    if not liq_lev_tolerance:
        liq_lev_tolerance = 0.02

    liquidity_levels = detect_support_resistance_levels(df, window=len(df), tolerance=liq_lev_tolerance)

    breaker_blocks = detect_breaker_blocks(df, liquidity_levels)

    # Log detected features for debugging
    logger.info(f"Detected Order Blocks: {order_blocks}")
    logger.info(f"Detected FVGs: {fvgs}")
    logger.info(f"Detected Liquidity Levels: {liquidity_levels}")
    logger.info(f"Detected Breaker Blocks: {breaker_blocks}")

    fig, ax = mpf.plot(
        df,
        type='candle',
        style='yahoo',
        volume=True,
        returnfig=True,
        ylabel='Price (USDT)',
        ylabel_lower='Volume',
        tight_layout=True,
        figsize=(12, 8)
    )

    add_fvgs(ax, fvgs)
    add_order_blocks(ax, order_blocks, df)
    add_liquidity_levels(ax, liquidity_levels)
    add_breaker_blocks(ax, breaker_blocks, df)

    add_legend(ax)

    fig.subplots_adjust(left=0.05, right=0.95)
    fig.savefig('crypto_chart.png', bbox_inches='tight', pad_inches=0.1)

    return 'crypto_chart.png'

def add_legend(ax):
    red_patch = mlines.Line2D([], [], color='blue', marker='o', markersize=10, label='Bearish Order Block', linestyle='None')
    green_patch = mlines.Line2D([], [], color='purple', marker='o', markersize=10, label='Bullish Order Block', linestyle='None')

    purple_patch = mpatches.Patch(color='purple', alpha=0.3, label='FVG')

    light_green_patch = mpatches.Patch(color='green', alpha=0.05, label='Bullish Breaker Block')
    light_red_patch = mpatches.Patch(color='red', alpha=0.05, label='Bearish Breaker Block')

    red_dashed_line = mlines.Line2D([], [], color='red', linestyle='--', markersize=10, label='Resistance level')
    green_dashed_line = mlines.Line2D([], [], color='green', linestyle='--', markersize=10, label='Support level')

    ax[0].legend(handles=[red_patch, green_patch, purple_patch, light_green_patch, light_red_patch, red_dashed_line, green_dashed_line], loc='upper left')

def add_order_blocks(ax, order_blocks, df):
    for block in order_blocks:
        idx = block.get_index()
        low = block.get_low()

        if idx < 0 or idx >= len(df):
            logger.warning(f"Invalid order block index: {idx}")
            continue

        circle_y_position = low * 0.995  # Slightly above the high for visibility

        ax[0].scatter(
            idx,
            circle_y_position,
            color='purple' if block.is_bullish() else 'blue',
            s=20,
            zorder=5,
            label="Order Block"
        )

def add_fvgs(ax, fvgs):
    for fvg in fvgs:
        start_idx = fvg.get_start_index()
        end_idx = fvg.get_end_index()
        start_price = fvg.get_start_price()
        end_price = fvg.get_end_price()
        is_covered = fvg.is_covered()

        if not is_covered:
            # Plot the FVG as a horizontal rectangle
            ax[0].fill_betweenx(
                y=[start_price, end_price],
                x1=start_idx,
                x2=end_idx,
                # color='blue' if start_price < end_price else 'orange',
                color='blue',
                alpha=0.2,
                label="FVG",
            )

def add_liquidity_levels(ax, liquidity_levels):
    for level in liquidity_levels:
        ax[0].axhline(
            y=level.get_price(),
            color='green' if level.is_support() else 'red',
            linestyle='--',
            linewidth=1,
        )

def add_breaker_blocks(ax, breaker_blocks, df):
    """
    Add breaker blocks as shaded zones on the chart.

    Parameters:
        ax (list): Axes object for the mplfinance chart.
        breaker_blocks (list): List of breaker blocks.
        df (pd.DataFrame): DataFrame containing OHLC data with DateTime index.
    """
    for block in breaker_blocks:
        # Start time and end time (extend the block forward by 1-3 candles)
        start_index = block.get_index()
        end_index = start_index + 2  # Add ~2 candles forward

        # Breaker block price range
        y1, y2 = block.get_zone()

        # Choose color based on type
        color = 'green' if block.is_bullish() else 'red'

        # Fill between x-axis and y-axis values
        ax[0].fill_betweenx(
            y=[y1, y2],                     # Price range (y-axis)
            x1=start_index,                 # Start time of the block
            x2=end_index,                   # End time of the block
            color=color,
            alpha=0.05,
        )
