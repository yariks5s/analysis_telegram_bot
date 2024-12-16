import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import mplfinance as mpf
import pandas as pd

from helpers import logger
from indicators import detect_order_blocks, detect_fvgs


def plot_price_chart(df: pd.DataFrame, symbol: str):
    """
    Generate a candlestick chart with detected order blocks and FVGs as horizontal rectangles.
    """
    # Detect order blocks and FVGs
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvgs(df)

    # Log detected features for debugging
    logger.info(f"Detected Order Blocks: {order_blocks}")
    logger.info(f"Detected FVGs: {fvgs}")

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

    add_legend(ax)

    fig.subplots_adjust(left=0.05, right=0.95)
    fig.savefig('crypto_chart.png', bbox_inches='tight', pad_inches=0.1)
    return 'crypto_chart.png'

def add_legend(ax):
    red_patch = mlines.Line2D([], [], color='blue', marker='o', markersize=10, label='Bearish Order Block', linestyle='None')
    green_patch = mlines.Line2D([], [], color='purple', marker='o', markersize=10, label='Bullish Order Block', linestyle='None')
    purple_patch = mpatches.Patch(color='purple', alpha=0.3, label='FVG')

    ax[0].legend(handles=[red_patch, green_patch, purple_patch], loc='upper left')

def add_order_blocks(ax, order_blocks, df):
    for idx, high, low, ob_type in order_blocks:
        if idx < 0 or idx >= len(df):
            logger.warning(f"Invalid order block index: {idx}")
            continue

        # Convert datetime index to mplfinance-compatible x-coordinates
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
