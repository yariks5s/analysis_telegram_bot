import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import mplfinance as mpf
import pandas as pd

from helpers import logger
from indicators import detect_order_blocks, detect_fvgs, detect_support_resistance_levels


def plot_price_chart(df: pd.DataFrame, liq_lev_tolerance: float):
    """
    Generate a candlestick chart with detected order blocks, FVGs, and support/resistance levels as horizontal rectangles.
    """

    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvgs(df)

    if not liq_lev_tolerance:
        liq_lev_tolerance = 0.02

    liquidity_levels = detect_support_resistance_levels(df, window=len(df), tolerance=liq_lev_tolerance)

    # Log detected features for debugging
    logger.info(f"Detected Order Blocks: {order_blocks}")
    logger.info(f"Detected FVGs: {fvgs}")
    logger.info(f"Detected Liquidity Levels: {liquidity_levels}")

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
    add_liquidity_levels(ax, liquidity_levels, df)

    add_legend(ax)

    fig.subplots_adjust(left=0.05, right=0.95)
    fig.savefig('crypto_chart.png', bbox_inches='tight', pad_inches=0.1)

    return 'crypto_chart.png'

def add_legend(ax):
    red_patch = mlines.Line2D([], [], color='blue', marker='o', markersize=10, label='Bearish Order Block', linestyle='None')
    green_patch = mlines.Line2D([], [], color='purple', marker='o', markersize=10, label='Bullish Order Block', linestyle='None')
    purple_patch = mpatches.Patch(color='purple', alpha=0.3, label='FVG')

    red_dashed_line = mlines.Line2D([], [], color='red', linestyle='--', markersize=10, label='Resistance level')
    green_dashed_line = mlines.Line2D([], [], color='green', linestyle='--', markersize=10, label='Support level')

    ax[0].legend(handles=[red_patch, green_patch, purple_patch, red_dashed_line, green_dashed_line], loc='upper left')

def add_order_blocks(ax, order_blocks, df):
    for idx, high, low, ob_type in order_blocks:
        if idx < 0 or idx >= len(df):
            logger.warning(f"Invalid order block index: {idx}")
            continue

        circle_y_position = low * 0.995  # Slightly above the high for visibility

        ax[0].scatter(
            idx,
            circle_y_position,
            color='purple' if ob_type == 'bullish' else 'blue',
            s=20,
            zorder=5,
            label="Order Block"
        )

def add_fvgs(ax, fvgs):
    for fvg in fvgs:
        start_idx, end_idx, start_price, end_price, _, is_covered = fvg

        if not is_covered:
            # Plot the FVG as a horizontal rectangle
            ax[0].fill_betweenx(
                y=[start_price, end_price],
                x1=start_idx,
                x2=end_idx,
                color='blue' if start_price < end_price else 'orange',
                alpha=0.2,
                label="FVG",
            )

def add_liquidity_levels(ax, liquidity_levels, df):
    for level in liquidity_levels[0]:  # Support levels
        ax[0].axhline(
            y=level,
            color='green',
            linestyle='--',
            linewidth=1,
            label="Support Level"
        )

    for level in liquidity_levels[1]:  # Resistance levels
        ax[0].axhline(
            y=level,
            color='red',
            linestyle='--',
            linewidth=1,
            label="Resistance Level"
        )
