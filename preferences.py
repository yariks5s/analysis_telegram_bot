"""
Preferences-related constants and utilities for the CryptoBot.
"""

# Dictionary mapping preference keys to their display names
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


def get_formatted_preferences(preferences_dict):
    """
    Format a preferences dictionary into a list of human-readable strings.

    Args:
        preferences_dict: Dictionary of preference key-value pairs

    Returns:
        List of formatted preference strings
    """
    formatted = []

    for key, val in preferences_dict.items():
        # Include dark_mode regardless of value (showing either Dark or Light mode)
        # Include other preferences only if they're enabled (True)
        if val or key == "dark_mode":
            formatted.append(get_pretty_preference_name(key, val))

    return formatted
