"""
Preferences-related constants and utilities for the CryptoBot.
"""

from typing import Dict, List, Any


PREFERENCE_DISPLAY_NAMES = {
    "order_blocks": "Order Blocks",
    "fvgs": "FVGs",
    "liquidity_levels": "Liquidity Levels",
    "breaker_blocks": "Breaker Blocks",
    "show_legend": "Show Legend",
    "show_volume": "Show Volume",
    "liquidity_pools": "Liquidity Pools",
    "dark_mode": "Dark Mode",
}


def get_pretty_preference_name(key: str, value=None) -> str:
    """
    Get a human-readable name for a preference.
    For dark_mode, returns "Dark Mode" or "Light Mode" based on the value.

    Args:
        key: The preference key
        value: The preference value (for special handling like dark_mode)

    Returns:
        A human-readable name for the preference
    """
    if key == "dark_mode" and value is not None:
        return "Dark Mode" if value else "Light Mode"

    return PREFERENCE_DISPLAY_NAMES.get(key, key)


def get_formatted_preferences(preferences_dict: Dict[str, Any]) -> List[str]:
    """
    Format a preferences dictionary into a list of human-readable strings.

    Args:
        preferences_dict: Dictionary of preference key-value pairs

    Returns:
        List of formatted preference strings
    """
    formatted = []

    for key, val in preferences_dict.items():
        if val or key == "dark_mode":
            formatted.append(get_pretty_preference_name(key, val))

    return formatted


def create_default_preferences() -> Dict[str, bool]:
    """
    Create a default preferences dictionary with all indicators enabled.

    Returns:
        dict: Default preferences dictionary
    """
    return {
        "order_blocks": True,
        "fvgs": True,
        "liquidity_levels": True,
        "breaker_blocks": True,
        "show_legend": True,
        "show_volume": True,
        "liquidity_pools": True,
        "dark_mode": False,
    }
