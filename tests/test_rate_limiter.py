"""
Tests for the rate limiting middleware functionality.
"""

import pytest
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

from telegram import Update, User
from telegram.ext import ContextTypes

from src.core.rate_limiter import (
    rate_limit,
    get_rate_limit_stats,
    RateLimitExceeded,
    BotRateLimiter,
    USER_RATE_LIMIT,
    BURST_THRESHOLD,
    BURST_WINDOW,
    SUSPICIOUS_THRESHOLD,
    SUSPICIOUS_PENALTY
)


@pytest.fixture
def mock_update():
    """Create a mock Update object with a user ID."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_context():
    return MagicMock(spec=ContextTypes.DEFAULT_TYPE)


@pytest.fixture
def fresh_rate_limiter():
    return BotRateLimiter()


def test_rate_limiter_initialization():
    limiter = BotRateLimiter()
    assert isinstance(limiter.user_requests, dict)
    assert isinstance(limiter.global_requests, list)
    assert isinstance(limiter.suspicious_users, dict)
    assert isinstance(limiter.burst_tracking, dict)


def test_clean_old_requests():
    limiter = BotRateLimiter()
    
    current_time = time.time()
    old_time = current_time - 120  # 2 minutes ago
    recent_time = current_time - 30  # 30 seconds ago
    
    test_requests = [old_time, recent_time, current_time]
    cleaned = limiter._clean_old_requests(test_requests, window=60)
    
    assert len(cleaned) == 2
    assert old_time not in cleaned
    assert recent_time in cleaned
    assert current_time in cleaned


def test_is_suspicious():
    limiter = BotRateLimiter()
    user_id = 54321
    
    assert not limiter.is_suspicious(user_id)
    
    current_time = time.time()
    limiter.suspicious_users[user_id] = [
        current_time - 100,  # Still within detection window
        current_time - 50,
        current_time
    ]
    
    assert limiter.is_suspicious(user_id)
    
    limiter.suspicious_users[user_id] = []
    assert not limiter.is_suspicious(user_id)
    
    limiter.suspicious_users[user_id] = [
        current_time - 1000,  # Outside detection window
        current_time - 800
    ]
    
    # Should not be suspicious as activities are old
    assert not limiter.is_suspicious(user_id)


def test_check_limits():
    limiter = BotRateLimiter()
    user_id = 98765
    
    assert limiter.check_limits(user_id)
    
    current_time = time.time()
    limiter.user_requests[user_id] = [current_time] * USER_RATE_LIMIT
    
    with pytest.raises(RateLimitExceeded):
        limiter.check_limits(user_id)


def test_burst_detection():
    limiter = BotRateLimiter()
    user_id = 24680
    
    limiter.burst_tracking[user_id] = deque(maxlen=100)
    limiter.suspicious_users[user_id] = []
    limiter.user_requests[user_id] = []  # Ensure we don't hit rate limits
    
    current_time = time.time()
    for _ in range(BURST_THRESHOLD - 1):
        limiter.burst_tracking[user_id].append(current_time)
        
    recent_requests = [ts for ts in limiter.burst_tracking[user_id] if current_time - ts <= BURST_WINDOW]
    assert len(recent_requests) < BURST_THRESHOLD
    assert len(limiter.suspicious_users.get(user_id, [])) == 0
    
    limiter.burst_tracking[user_id].append(current_time)
    
    recent_requests = [ts for ts in limiter.burst_tracking[user_id] if current_time - ts <= BURST_WINDOW]
    if len(recent_requests) >= BURST_THRESHOLD:
        limiter.suspicious_users[user_id].append(current_time)
        
    assert len(limiter.suspicious_users.get(user_id, [])) == 1


def test_reduced_limit_for_suspicious(fresh_rate_limiter):
    limiter = fresh_rate_limiter
    user_id = 13579
    
    # Make the user suspicious
    current_time = time.time()
    limiter.suspicious_users[user_id] = [current_time] * SUSPICIOUS_THRESHOLD
    
    quota = limiter.get_user_quota(user_id)
    
    assert quota["limit"] == USER_RATE_LIMIT // SUSPICIOUS_PENALTY
    assert quota["is_suspicious"] is True


def test_get_user_quota():
    limiter = BotRateLimiter()
    user_id = 11223
    
    quota = limiter.get_user_quota(user_id)
    
    assert quota["limit"] == USER_RATE_LIMIT
    assert quota["remaining"] == USER_RATE_LIMIT
    assert quota["used"] == 0
    assert "reset_seconds" in quota
    assert not quota["is_suspicious"]
    
    current_time = time.time()
    limiter.user_requests[user_id] = [current_time] * 5
    
    quota = limiter.get_user_quota(user_id)
    assert quota["used"] == 5
    assert quota["remaining"] == USER_RATE_LIMIT - 5


@pytest.mark.asyncio
async def test_rate_limit_decorator(mock_update, mock_context):
    @rate_limit(cost=1)
    async def test_handler(update, context):
        return "success"
    
    result = await test_handler(mock_update, mock_context)
    assert result == "success"
    
    with patch('src.core.rate_limiter.rate_limiter.check_limits', side_effect=RateLimitExceeded("Test limit exceeded")):
        result = await test_handler(mock_update, mock_context)
        assert result is None
        mock_update.message.reply_text.assert_called_once()
        assert "limit exceeded" in str(mock_update.message.reply_text.call_args[0][0]).lower()


def test_get_rate_limit_stats():
    user_id = 99887
    
    stats = get_rate_limit_stats(user_id)
    
    assert "limit" in stats
    assert "remaining" in stats
    assert "used" in stats
    assert "reset_seconds" in stats
    assert "is_suspicious" in stats
