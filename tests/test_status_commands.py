"""
Tests for status commands including rate limit information.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Update, User, Message, Chat
from telegram.ext import ContextTypes

from src.telegram.commands.status_commands import rate_limit_command


@pytest.fixture
def mock_update():
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.message = MagicMock(spec=Message)
    update.message.reply_text = AsyncMock()
    update.message.chat = MagicMock(spec=Chat)
    return update


@pytest.fixture
def mock_context():
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)


@pytest.mark.asyncio
async def test_rate_limit_command_normal_user(mock_update, mock_context):
    with patch(
        "src.telegram.commands.status_commands.get_rate_limit_stats",
        return_value={
            "remaining": 15,
            "limit": 20,
            "reset_seconds": 30,
            "is_suspicious": False,
        },
    ):
        await rate_limit_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        message = mock_update.message.reply_text.call_args[0][0]

        assert "✅" in message
        assert "15/20" in message
        assert "30 seconds" in message  # Should show reset time
        assert "reduced due to suspicious activity" not in message


@pytest.mark.asyncio
async def test_rate_limit_command_suspicious_user(mock_update, mock_context):
    with patch(
        "src.telegram.commands.status_commands.get_rate_limit_stats",
        return_value={
            "remaining": 2,
            "limit": 4,
            "reset_seconds": 45,
            "is_suspicious": True,
        },
    ):
        await rate_limit_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        message = mock_update.message.reply_text.call_args[0][0]

        assert "⚠️" in message
        assert "2/4" in message
        assert "45 seconds" in message
        assert "reduced due to suspicious activity" in message
        assert "temporarily reduced" in message


@pytest.mark.asyncio
async def test_rate_limit_command_low_remaining(mock_update, mock_context):
    with patch(
        "src.telegram.commands.status_commands.get_rate_limit_stats",
        return_value={
            "remaining": 3,
            "limit": 20,
            "reset_seconds": 25,
            "is_suspicious": False,
        },
    ):
        await rate_limit_command(mock_update, mock_context)

        message = mock_update.message.reply_text.call_args[0][0]
        assert "⚠️" in message
