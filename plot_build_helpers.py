import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import mplfinance as mpf
import pandas as pd

from utils import logger

def plot_price_chart(df: pd.DataFrame, indicators):
    """
    Generate a candlestick chart with detected order blocks, FVGs, and support/resistance levels as horizontal lines.
    """

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

    add_fvgs(ax, indicators.fvgs)
    add_order_blocks(ax, indicators.order_blocks, df)
    add_liquidity_levels(ax, indicators.liquidity_levels)
    add_breaker_blocks(ax, indicators.breaker_blocks)

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
    for block in order_blocks.list:
        idx = block.index
        low = block.low

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
    for fvg in fvgs.list:
        start_idx = fvg.start_index
        end_idx = fvg.end_index
        start_price = fvg.start_price
        end_price = fvg.end_price

        if not fvg.covered:
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
    for level in liquidity_levels.list:
        ax[0].axhline(
            y=level.price,
            color='green' if level.is_support() else 'red',
            linestyle='--',
            linewidth=1,
        )

def add_breaker_blocks(ax, breaker_blocks):
    for block in breaker_blocks.list:
        # Start time and end time (extend the block forward by 1-3 candles)
        start_index = block.index
        end_index = start_index + 2  # Add ~2 candles forward

        # Breaker block price range
        y1, y2 = block.zone

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
