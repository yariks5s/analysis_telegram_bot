"""
Chart styling module for CryptoBot.

This module contains all the theme-specific styling constants and functions
used for rendering charts with different themes (light/dark mode).
"""

import mplfinance as mpf  # type: ignore


class ChartTheme:
    """Class containing styling configurations for chart themes."""
    
    @staticmethod
    def get_mpf_style(dark_mode=False):
        """
        Get the appropriate mplfinance style based on the theme.
        
        Args:
            dark_mode: Whether to use dark mode styling
            
        Returns:
            MPF style object or string
        """
        if dark_mode:
            mc = mpf.make_marketcolors(
                up="#8AFF80",           # Bright green for up candles
                down="#FF5555",         # Bright red for down candles
                edge="inherit",
                wick={"up": "#8AFF80", "down": "#FF5555"},
                volume={"up": "#8AFF80", "down": "#FF5555"}
            )
            return mpf.make_mpf_style(
                base_mpf_style="nightclouds",
                marketcolors=mc,
                gridstyle=":",
                gridcolor="#44475A",
                facecolor="#282A36",
                edgecolor="#44475A",
                figcolor="#282A36",
                y_on_right=False
            )
        else:
            # Default light style
            return "yahoo"
    
    # Order blocks styling
    @staticmethod
    def get_order_block_style(dark_mode=False, is_bullish=True):
        """Get style for order blocks based on theme and type."""
        if dark_mode:
            return {
                "bullish": {
                    "color": "#FF79C6",  # Bright pink for dark mode
                    "marker_size": 30     # Larger marker for better visibility
                },
                "bearish": {
                    "color": "#8BE9FD",  # Bright cyan for dark mode
                    "marker_size": 30     # Larger marker for better visibility
                }
            }["bullish" if is_bullish else "bearish"]
        else:
            return {
                "bullish": {
                    "color": "purple",    # Original colors for light mode
                    "marker_size": 20     # Original size
                },
                "bearish": {
                    "color": "blue",      # Original colors for light mode
                    "marker_size": 20     # Original size
                }
            }["bullish" if is_bullish else "bearish"]
    
    # FVG styling
    @staticmethod
    def get_fvg_style(dark_mode=False):
        """Get style for Fair Value Gaps based on theme."""
        if dark_mode:
            return {
                "color": "#BD93F9",  # Bright purple for dark mode
                "alpha": 0.5         # Higher opacity for better visibility
            }
        else:
            return {
                "color": "blue",     # Original color for light mode
                "alpha": 0.2         # Original opacity
            }
    
    # Liquidity levels styling
    @staticmethod
    def get_liquidity_level_style(dark_mode=False):
        """Get style for liquidity levels based on theme."""
        if dark_mode:
            return {
                "color": "#FFB86C",  # Bright orange for dark mode
                "linewidth": 1.5,    # Thicker line for better visibility
                "linestyle": "--"
            }
        else:
            return {
                "color": "orange",   # Original color for light mode
                "linewidth": 1,      # Original linewidth
                "linestyle": "--"
            }
    
    # Breaker blocks styling
    @staticmethod
    def get_breaker_block_style(dark_mode=False, is_bullish=True):
        """Get style for breaker blocks based on theme and type."""
        if dark_mode:
            return {
                "bullish": {
                    "color": "#50FA7B",  # Bright green for dark mode
                    "alpha": 0.3         # Higher opacity for dark mode
                },
                "bearish": {
                    "color": "#FF5555",  # Bright red for dark mode
                    "alpha": 0.3         # Higher opacity for dark mode
                }
            }["bullish" if is_bullish else "bearish"]
        else:
            return {
                "bullish": {
                    "color": "green",    # Original color for light mode
                    "alpha": 0.05        # Original opacity
                },
                "bearish": {
                    "color": "red",      # Original color for light mode
                    "alpha": 0.05        # Original opacity
                }
            }["bullish" if is_bullish else "bearish"]
    
    # Liquidity pools styling
    @staticmethod
    def get_liquidity_pool_style(dark_mode=False):
        """Get style for liquidity pools based on theme."""
        if dark_mode:
            return {
                "color": "#8BE9FD",     # Bright cyan for dark mode
                "base_alpha": 0.3,      # Higher base opacity for dark mode
                "line_alpha": 0.7,      # Higher line opacity for dark mode
                "linewidth": 1.5,       # Thicker line for better visibility
                "linestyle": ":"
            }
        else:
            return {
                "color": "cyan",        # Original color for light mode
                "base_alpha": 0.2,      # Original base opacity
                "line_alpha": 0.5,      # Original line opacity
                "linewidth": 1,         # Original linewidth
                "linestyle": ":"
            }
