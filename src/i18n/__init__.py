"""
Internationalization (i18n) module for CryptoBot.

This module provides multi-language support with:
- Translation loading from YAML files
- Language preference per user
- Placeholder interpolation
- Fallback to English for missing translations
"""

import os
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

import yaml

logger = logging.getLogger(__name__)

# Supported languages with their native names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ru": "Русский",
    "uk": "Українська",
    "de": "Deutsch",
    "fr": "Français",
    "nl": "Nederlands",
}

# Default language
DEFAULT_LANGUAGE = "en"

# Cache for loaded translations
_translations: Dict[str, Dict[str, Any]] = {}

# Path to locales directory
LOCALES_DIR = os.path.join(os.path.dirname(__file__), "locales")


def load_translations() -> None:
    """
    Load all translation files from the locales directory.
    Should be called once at application startup.
    """
    global _translations
    _translations = {}
    
    for lang_code in SUPPORTED_LANGUAGES.keys():
        file_path = os.path.join(LOCALES_DIR, f"{lang_code}.yaml")
        try:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    _translations[lang_code] = yaml.safe_load(f) or {}
                logger.info(f"Loaded translations for {lang_code}")
            else:
                logger.warning(f"Translation file not found: {file_path}")
                _translations[lang_code] = {}
        except Exception as e:
            logger.error(f"Error loading translations for {lang_code}: {e}")
            _translations[lang_code] = {}
    
    # Ensure English is always available as fallback
    if "en" not in _translations:
        _translations["en"] = {}


def _get_nested_value(data: Dict[str, Any], key: str) -> Optional[str]:
    """
    Get a nested value from a dictionary using dot notation.
    
    Args:
        data: The dictionary to search
        key: The key in dot notation (e.g., "tutorial.welcome.title")
    
    Returns:
        The value if found, None otherwise
    """
    keys = key.split(".")
    value = data
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    
    return value if isinstance(value, str) else None


def t(key: str, lang: str = DEFAULT_LANGUAGE, **kwargs) -> str:
    """
    Get a translated string for the given key and language.
    
    Args:
        key: Translation key in dot notation (e.g., "tutorial.welcome.title")
        lang: Language code (e.g., "en", "ru", "uk", "de", "fr", "nl")
        **kwargs: Placeholder values to interpolate
    
    Returns:
        Translated string with placeholders replaced, or the key if not found
    
    Example:
        t("common.welcome", "ru", name="Иван")
        # Returns "Привет, Иван!" if translation is "Привет, {name}!"
    """
    # Ensure translations are loaded
    if not _translations:
        load_translations()
    
    # Validate language code
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE
    
    # Try to get translation in requested language
    translation = _get_nested_value(_translations.get(lang, {}), key)
    
    # Fallback to English if not found
    if translation is None and lang != DEFAULT_LANGUAGE:
        translation = _get_nested_value(_translations.get(DEFAULT_LANGUAGE, {}), key)
    
    # If still not found, return the key
    if translation is None:
        logger.warning(f"Missing translation for key '{key}' in language '{lang}'")
        return key
    
    # Interpolate placeholders
    try:
        return translation.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing placeholder {e} for key '{key}'")
        return translation


def get_user_language(user_id: int) -> str:
    """
    Get the language preference for a user from the database.
    
    Args:
        user_id: Telegram user ID
    
    Returns:
        Language code (defaults to English if not set)
    """
    # Lazy import to avoid circular dependency
    from src.database.operations import get_user_preferences
    
    try:
        prefs = get_user_preferences(user_id)
        lang = prefs.get("language", DEFAULT_LANGUAGE)
        return lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
    except Exception as e:
        logger.error(f"Error getting user language: {e}")
        return DEFAULT_LANGUAGE


def set_user_language(user_id: int, lang: str) -> bool:
    """
    Set the language preference for a user.
    
    Args:
        user_id: Telegram user ID
        lang: Language code
    
    Returns:
        True if successful, False otherwise
    """
    if lang not in SUPPORTED_LANGUAGES:
        return False
    
    # Lazy import to avoid circular dependency
    from src.database.operations import update_user_language
    
    try:
        update_user_language(user_id, lang)
        return True
    except Exception as e:
        logger.error(f"Error setting user language: {e}")
        return False


def get_language_keyboard_data() -> list:
    """
    Get data for building a language selection keyboard.
    
    Returns:
        List of tuples (lang_code, native_name, flag_emoji)
    """
    flags = {
        "en": "🇬🇧",
        "ru": "🇷🇺",
        "uk": "🇺🇦",
        "de": "🇩🇪",
        "fr": "🇫🇷",
        "nl": "🇳🇱",
    }
    
    return [
        (code, name, flags.get(code, "🌐"))
        for code, name in SUPPORTED_LANGUAGES.items()
    ]


# Convenience function for handler usage
def tr(update, key: str, **kwargs) -> str:
    """
    Convenience function to get translation based on update's user.
    
    Args:
        update: Telegram Update object
        key: Translation key
        **kwargs: Placeholder values
    
    Returns:
        Translated string
    """
    user_id = update.effective_user.id if update.effective_user else 0
    lang = get_user_language(user_id)
    return t(key, lang, **kwargs)


# Export public API
__all__ = [
    "SUPPORTED_LANGUAGES",
    "DEFAULT_LANGUAGE",
    "load_translations",
    "t",
    "tr",
    "get_user_language",
    "set_user_language",
    "get_language_keyboard_data",
]

