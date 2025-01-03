import pytest
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)
from message_handlers import handle_indicator_selection, select_indicators

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
