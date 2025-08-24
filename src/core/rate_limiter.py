"""
Rate limiting middleware for Telegram bot commands.

This module provides a centralized rate limiting system that can be applied
to all command handlers in the bot to prevent abuse and ensure fair usage.
"""

import time
import logging
from collections import defaultdict, deque
from functools import wraps
from typing import Dict, List, Any

from telegram import Update
from telegram.ext import ContextTypes

# Rate limiting configuration
USER_RATE_LIMIT = 20        # Max requests per minute per user
GLOBAL_RATE_LIMIT = 500     # Max requests per minute globally
BURST_THRESHOLD = 10        # Number of requests in a short period to mark as suspicious
BURST_WINDOW = 10           # Time window (in seconds) for burst detection
SUSPICIOUS_PENALTY = 5      # Multiplier for suspicious users' rate limits
DETECTION_WINDOW = 300      # Time window for suspicious activity detection (5 minutes)
SUSPICIOUS_THRESHOLD = 3    # Number of bursts to mark a user as suspicious

logger = logging.getLogger(__name__)

class RateLimitExceeded(Exception):
    pass

class BotRateLimiter:
    def __init__(self):
        self.user_requests: Dict[int, List[float]] = defaultdict(list)
        self.global_requests: List[float] = []
        self.suspicious_users: Dict[int, List[float]] = defaultdict(list)
        self.burst_tracking: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def _clean_old_requests(self, request_list: List[float], window: int = 60) -> List[float]:
        current_time = time.time()
        return [ts for ts in request_list if current_time - ts < window]
    
    def is_suspicious(self, user_id: int) -> bool:
        current_time = time.time()
        
        # Clean up old suspicious activities
        self.suspicious_users[user_id] = [
            ts for ts in self.suspicious_users.get(user_id, [])
            if current_time - ts < DETECTION_WINDOW
        ]
        
        # Check if there are enough recent suspicious activities
        return len(self.suspicious_users.get(user_id, [])) >= SUSPICIOUS_THRESHOLD
    
    def check_limits(self, user_id: int) -> bool:
        """
        Check if the user or global rate limits have been exceeded.
        
        Args:
            user_id: The ID of the user making the request
            
        Returns:
            bool: True if the request is allowed, False if rate limited
            
        Raises:
            RateLimitExceeded: If the user or global limit is exceeded
        """
        current_time = time.time()
        
        self.burst_tracking[user_id].append(current_time)
        
        recent_requests = [
            ts for ts in self.burst_tracking[user_id]
            if current_time - ts <= BURST_WINDOW
        ]
        
        if len(recent_requests) >= BURST_THRESHOLD:
            logger.warning(f"Burst activity detected for user {user_id}")
            self.suspicious_users[user_id].append(current_time)
        
        self.user_requests[user_id] = self._clean_old_requests(self.user_requests[user_id])
        self.global_requests = self._clean_old_requests(self.global_requests)
        
        user_limit = USER_RATE_LIMIT
        if self.is_suspicious(user_id):
            user_limit = USER_RATE_LIMIT // SUSPICIOUS_PENALTY
            logger.warning(f"Using reduced rate limit for suspicious user {user_id}")
        
        if len(self.user_requests[user_id]) >= user_limit:
            raise RateLimitExceeded(f"User rate limit exceeded. Try again later.")
        
        if len(self.global_requests) >= GLOBAL_RATE_LIMIT:
            raise RateLimitExceeded(f"Global rate limit exceeded. Try again later.")
        
        self.user_requests[user_id].append(current_time)
        self.global_requests.append(current_time)
        
        return True
    
    def get_user_quota(self, user_id: int) -> Dict[str, Any]:
        """
        Get the remaining quota for a user.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dict containing quota information
        """
        current_time = time.time()
        
        self.user_requests[user_id] = self._clean_old_requests(self.user_requests[user_id])
        
        user_limit = USER_RATE_LIMIT
        if self.is_suspicious(user_id):
            user_limit = USER_RATE_LIMIT // SUSPICIOUS_PENALTY
        
        used = len(self.user_requests[user_id])
        remaining = max(0, user_limit - used)
        
        reset_time = 60
        if self.user_requests[user_id]:
            oldest = min(self.user_requests[user_id])
            reset_time = max(0, (oldest + 60) - current_time)
        
        return {
            "limit": user_limit,
            "remaining": remaining,
            "used": used,
            "reset_seconds": int(reset_time),
            "is_suspicious": self.is_suspicious(user_id)
        }

rate_limiter = BotRateLimiter()

def rate_limit(cost: int = 1):
    """
    Decorator for rate limiting Telegram command handlers.
    
    Args:
        cost: The cost of this command in rate limit tokens
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            # Skip rate limiting for admin users if needed
            # if update.effective_user and update.effective_user.id in ADMIN_IDS:
            #    return await func(update, context)
            
            user_id = update.effective_user.id if update.effective_user else 0
            
            try:
                rate_limiter.check_limits(user_id)
                
                # If we get here, the rate limit hasn't been exceeded
                return await func(update, context)
                
            except RateLimitExceeded as e:
                quota = rate_limiter.get_user_quota(user_id)
                await update.message.reply_text(
                    f"⚠️ {str(e)}\n"
                    f"Reset in {quota['reset_seconds']} seconds.\n"
                    f"Limit: {quota['limit']} requests per minute."
                )
                logger.warning(f"Rate limit exceeded for user {user_id}: {str(e)}")
                return None
                
        return wrapper
    
    return decorator

def get_rate_limit_stats(user_id: int) -> Dict[str, Any]:
    """Get rate limiting statistics for a user."""
    return rate_limiter.get_user_quota(user_id)
