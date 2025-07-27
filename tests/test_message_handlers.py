import pytest  # type: ignore
import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)
from src.telegram.handlers import handle_indicator_selection, select_indicators


@pytest.mark.asyncio
async def test_select_indicators(mocker):
    update = mocker.Mock()
    update.effective_user.id = 123
    update.message.reply_text = mocker.AsyncMock()

    context = mocker.Mock()
    await select_indicators(update, context)
    update.message.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_handle_indicator_selection(mocker):
    query = mocker.Mock()
    query.from_user.id = 123
    query.data = "indicator_order_blocks"
    query.answer = mocker.AsyncMock()
    query.edit_message_reply_markup = mocker.AsyncMock()

    update = mocker.Mock()
    update.callback_query = query
    context = mocker.Mock()

    await handle_indicator_selection(update, context)
    query.edit_message_reply_markup.assert_called_once()


@pytest.mark.asyncio
async def test_handle_dark_mode_toggle(mocker):
    """Test that toggling dark mode works correctly"""
    # Mock the database functions
    mocker.patch(
        "src.telegram.handlers.get_user_preferences",
        return_value={
            "order_blocks": False,
            "fvgs": False,
            "liquidity_levels": False,
            "breaker_blocks": False,
            "show_legend": True,
            "show_volume": True,
            "liquidity_pools": True,
            "dark_mode": False,  # Start with light mode
        },
    )
    update_preferences_mock = mocker.patch("src.telegram.handlers.update_user_preferences")

    # Mock the query object
    query = mocker.Mock()
    query.from_user.id = 123
    query.data = "indicator_dark_mode"  # Toggle dark mode
    query.answer = mocker.AsyncMock()
    query.edit_message_reply_markup = mocker.AsyncMock()
    query.message.reply_markup = mocker.Mock()
    query.message.reply_markup.to_dict.return_value = {"different": "value"}

    # Mock the update object
    update = mocker.Mock()
    update.callback_query = query
    context = mocker.Mock()

    # Call the handler function
    await handle_indicator_selection(update, context)

    # Verify update_user_preferences was called with dark_mode=True
    update_preferences_mock.assert_called_once()
    called_args = update_preferences_mock.call_args[0]
    assert called_args[0] == 123  # user_id
    preferences = called_args[1]
    assert preferences["dark_mode"] is True  # Dark mode should be toggled on


@pytest.mark.asyncio
async def test_preferences_done_message_formatting(mocker):
    """Test that the 'You selected' message is properly formatted with Dark/Light Mode"""
    # Mock user preferences with dark mode enabled
    preferences = {
        "order_blocks": True,
        "fvgs": False,
        "liquidity_levels": True,
        "breaker_blocks": False,
        "show_legend": True,
        "show_volume": True,
        "liquidity_pools": False,
        "dark_mode": True,  # Dark mode enabled
    }

    # Mock the get_user_preferences function
    mocker.patch("src.telegram.handlers.get_user_preferences", return_value=preferences)
    mocker.patch("src.telegram.handlers.update_user_preferences")

    # Mock the query object
    query = mocker.Mock()
    query.from_user.id = 123
    query.data = "indicator_done"  # User clicked Done
    query.answer = mocker.AsyncMock()
    query.edit_message_text = mocker.AsyncMock()

    # Mock the update object
    update = mocker.Mock()
    update.callback_query = query
    context = mocker.Mock()

    # Call the handler function
    await handle_indicator_selection(update, context)

    # Verify the message contains "Dark Mode" not "dark_mode"
    query.edit_message_text.assert_called_once()
    message_text = query.edit_message_text.call_args[0][0]
    assert "Dark Mode" in message_text
    assert "dark_mode" not in message_text
    assert "Order Blocks" in message_text
    assert "Liquidity Levels" in message_text


@pytest.mark.asyncio
async def test_preferences_done_message_with_light_mode(mocker):
    """Test that the 'You selected' message shows Light Mode when dark_mode is False"""
    # Mock user preferences with dark mode disabled
    preferences = {
        "order_blocks": True,
        "dark_mode": False,  # Light mode
    }

    # Mock the get_user_preferences function
    mocker.patch("src.telegram.handlers.get_user_preferences", return_value=preferences)
    mocker.patch("src.telegram.handlers.update_user_preferences")

    # Mock the query object
    query = mocker.Mock()
    query.from_user.id = 123
    query.data = "indicator_done"  # User clicked Done
    query.answer = mocker.AsyncMock()
    query.edit_message_text = mocker.AsyncMock()

    # Mock the update object
    update = mocker.Mock()
    update.callback_query = query
    context = mocker.Mock()

    # Call the handler function
    await handle_indicator_selection(update, context)

    # Verify the message contains "Light Mode"
    query.edit_message_text.assert_called_once()
    message_text = query.edit_message_text.call_args[0][0]
    assert "Light Mode" in message_text
    assert "dark_mode" not in message_text
