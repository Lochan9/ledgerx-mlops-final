"""
LedgerX - Aggressive Rate Limiting for GCP Free Tier
====================================================

Protects your $300 credit from abuse while maintaining usability.

FREE TIER BUDGET:
- $300 credit
- Target: Last 6+ months
- Max spend: $50/month ($1.67/day)

RATE LIMITS:
- Free tier users: 100 requests/day
- Authenticated users: 1000 requests/day
- Enterprise users: 10,000 requests/day
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import Request, HTTPException
from collections import defaultdict
import time

# ============================================================================
# IN-MEMORY RATE LIMITER (No Redis needed for free tier)
# ============================================================================

class CostOptimizedRateLimiter:
    """
    Memory-efficient rate limiter for GCP free tier
    
    Features:
    - Per-IP limiting
    - Per-user limiting  
    - Sliding window
    - Auto-cleanup old entries
    """
    
    def __init__(self):
        # {ip_address: [(timestamp, count), ...]}
        self.ip_requests = defaultdict(list)
        # {username: [(timestamp, count), ...]}
        self.user_requests = defaultdict(list)
        
        # Limits
        self.IP_LIMIT_PER_HOUR = 50  # Very conservative for free tier
        self.IP_LIMIT_PER_DAY = 200
        
        self.USER_FREE_LIMIT_PER_DAY = 100
        self.USER_PRO_LIMIT_PER_DAY = 1000
        self.USER_ENTERPRISE_LIMIT_PER_DAY = 10000
        
        # Cost tracking
        self.COST_PER_REQUEST = 0.00002400  # GCP Cloud Run cost
        self.total_requests_today = 0
        self.estimated_cost_today = 0.0
        self.daily_budget = 1.67  # $50/month ÷ 30 days
    
    def _cleanup_old_entries(self, request_list, window_hours=24):
        """Remove entries older than window"""
        cutoff = time.time() - (window_hours * 3600)
        return [entry for entry in request_list if entry[0] > cutoff]
    
    def check_ip_limit(self, ip: str) -> bool:
        """
        Check if IP is within rate limits
        
        Returns:
            True if allowed, False if limit exceeded
        """
        # Cleanup old entries
        self.ip_requests[ip] = self._cleanup_old_entries(self.ip_requests[ip])
        
        now = time.time()
        recent_requests = self.ip_requests[ip]
        
        # Check hourly limit
        hour_ago = now - 3600
        requests_last_hour = sum(1 for ts, _ in recent_requests if ts > hour_ago)
        
        if requests_last_hour >= self.IP_LIMIT_PER_HOUR:
            return False
        
        # Check daily limit
        day_ago = now - 86400
        requests_last_day = sum(1 for ts, _ in recent_requests if ts > day_ago)
        
        if requests_last_day >= self.IP_LIMIT_PER_DAY:
            return False
        
        return True
    
    def check_user_limit(self, username: str, tier: str = "free") -> bool:
        """
        Check if user is within their tier limit
        
        Args:
            username: Username
            tier: 'free', 'pro', or 'enterprise'
            
        Returns:
            True if allowed, False if exceeded
        """
        # Cleanup
        self.user_requests[username] = self._cleanup_old_entries(
            self.user_requests[username]
        )
        
        now = time.time()
        day_ago = now - 86400
        
        requests_last_day = sum(
            1 for ts, _ in self.user_requests[username] 
            if ts > day_ago
        )
        
        # Check tier limit
        limits = {
            "free": self.USER_FREE_LIMIT_PER_DAY,
            "pro": self.USER_PRO_LIMIT_PER_DAY,
            "enterprise": self.USER_ENTERPRISE_LIMIT_PER_DAY
        }
        
        limit = limits.get(tier, self.USER_FREE_LIMIT_PER_DAY)
        
        return requests_last_day < limit
    
    def record_request(self, ip: str, username: Optional[str] = None):
        """Record a request for rate limiting"""
        now = time.time()
        
        self.ip_requests[ip].append((now, 1))
        
        if username:
            self.user_requests[username].append((now, 1))
        
        # Track costs
        self.total_requests_today += 1
        self.estimated_cost_today += self.COST_PER_REQUEST
        
        # Alert if approaching daily budget
        if self.estimated_cost_today > self.daily_budget * 0.8:
            from src.utils.alerts import send_alert
            send_alert(
                f"⚠️ Cost Alert: ${self.estimated_cost_today:.2f} / ${self.daily_budget:.2f} daily budget used!",
                severity="warning"
            )
    
    def get_usage_stats(self, ip: str = None, username: str = None) -> dict:
        """Get usage statistics"""
        stats = {
            "total_requests_today": self.total_requests_today,
            "estimated_cost_today": round(self.estimated_cost_today, 4),
            "daily_budget": self.daily_budget,
            "budget_remaining": round(self.daily_budget - self.estimated_cost_today, 4)
        }
        
        if ip:
            day_ago = time.time() - 86400
            ip_requests = sum(
                1 for ts, _ in self.ip_requests[ip] 
                if ts > day_ago
            )
            stats["ip_requests_today"] = ip_requests
            stats["ip_limit_remaining"] = self.IP_LIMIT_PER_DAY - ip_requests
        
        if username:
            day_ago = time.time() - 86400
            user_requests = sum(
                1 for ts, _ in self.user_requests[username]
                if ts > day_ago
            )
            stats["user_requests_today"] = user_requests
        
        return stats


# Global instance
rate_limiter = CostOptimizedRateLimiter()


# ============================================================================
# FASTAPI DEPENDENCY
# ============================================================================

async def check_rate_limit(request: Request, current_user: Optional[str] = None):
    """
    FastAPI dependency to check rate limits
    
    Usage:
        @app.post("/predict")
        async def predict(
            features: InvoiceFeatures,
            _: None = Depends(check_rate_limit)
        ):
    """
    ip = request.client.host
    
    # Check IP limit first (prevents anonymous abuse)
    if not rate_limiter.check_ip_limit(ip):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many requests from your IP. Limit: 50/hour, 200/day",
                "retry_after": "1 hour"
            }
        )
    
    # Check user limit if authenticated
    if current_user:
        # Determine user tier (from database or JWT)
        user_tier = "free"  # Default, get from user profile
        
        if not rate_limiter.check_user_limit(current_user, user_tier):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Daily limit exceeded",
                    "message": f"Tier: {user_tier}, Limit: {rate_limiter.USER_FREE_LIMIT_PER_DAY}/day",
                    "upgrade_url": "https://ledgerx.com/pricing"
                }
            )
    
    # Record request
    rate_limiter.record_request(ip, current_user)
    
    # Check daily budget
    if rate_limiter.estimated_cost_today > rate_limiter.daily_budget:
        # EMERGENCY: Budget exceeded, shut down
        from src.utils.alerts import send_alert
        send_alert(
            f"🚨 EMERGENCY: Daily budget exceeded! ${rate_limiter.estimated_cost_today:.2f}",
            severity="critical"
        )
        
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporarily unavailable",
                "message": "Daily budget exceeded. Service will resume tomorrow.",
                "retry_after": "24 hours"
            }
        )


# ============================================================================
# COST MONITORING ENDPOINT
# ============================================================================

def get_cost_stats():
    """Get current cost and usage statistics"""
    return {
        "today": {
            "requests": rate_limiter.total_requests_today,
            "estimated_cost": round(rate_limiter.estimated_cost_today, 4),
            "budget": rate_limiter.daily_budget,
            "budget_used_pct": round(
                (rate_limiter.estimated_cost_today / rate_limiter.daily_budget) * 100, 2
            )
        },
        "limits": {
            "ip_per_hour": rate_limiter.IP_LIMIT_PER_HOUR,
            "ip_per_day": rate_limiter.IP_LIMIT_PER_DAY,
            "user_free_per_day": rate_limiter.USER_FREE_LIMIT_PER_DAY
        }
    }