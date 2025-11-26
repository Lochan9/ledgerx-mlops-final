"""
LedgerX - Prediction Caching for Cost Optimization
===================================================

Implements intelligent caching to reduce costs by 30-40%.

Features:
- In-memory LRU cache (no Redis needed for free tier)
- Automatic cache invalidation
- Cache hit/miss tracking
- Cost savings metrics
"""

import hashlib
import json
import logging
from functools import lru_cache
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("ledgerx_cache")

# ============================================================================
# PREDICTION CACHE
# ============================================================================

class PredictionCache:
    """
    LRU cache for predictions with cost tracking
    
    Features:
    - Caches up to 1000 predictions
    - Automatic eviction of old entries
    - Tracks cache hit rate
    - Calculates cost savings
    """
    
    def __init__(self, max_size=1000, ttl_hours=24):
        """
        Args:
            max_size: Maximum number of cached predictions
            ttl_hours: Time-to-live for cached entries (hours)
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        
        # Cache storage: {hash: (result, timestamp)}
        self.cache = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.cost_per_prediction = 0.00002400  # GCP Cloud Run cost
        
        logger.info(f"[CACHE] Initialized with max_size={max_size}, ttl={ttl_hours}h")
    
    def _generate_key(self, features: Dict) -> str:
        """
        Generate cache key from features
        
        Args:
            features: Invoice features dictionary
            
        Returns:
            MD5 hash of features
        """
        # Sort keys for consistent hashing
        features_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(features_str.encode()).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() - timestamp > self.ttl
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if self._is_expired(timestamp)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"[CACHE] Cleaned up {len(expired_keys)} expired entries")
    
    def _evict_oldest(self):
        """Evict oldest entry if cache is full"""
        if len(self.cache) >= self.max_size:
            # Find oldest entry
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )
            del self.cache[oldest_key]
            logger.debug(f"[CACHE] Evicted oldest entry")
    
    def get(self, features: Dict) -> Optional[Dict]:
        """
        Get cached prediction if exists
        
        Args:
            features: Invoice features
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(features)
        
        # Cleanup expired entries periodically
        if len(self.cache) % 100 == 0:
            self._cleanup_expired()
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check if expired
            if self._is_expired(timestamp):
                del self.cache[key]
                self.misses += 1
                return None
            
            # Cache hit!
            self.hits += 1
            logger.info(f"[CACHE] HIT - Returning cached prediction")
            return result
        
        # Cache miss
        self.misses += 1
        return None
    
    def set(self, features: Dict, result: Dict):
        """
        Cache a prediction result
        
        Args:
            features: Invoice features
            result: Prediction result
        """
        key = self._generate_key(features)
        
        # Evict if full
        self._evict_oldest()
        
        # Store with timestamp
        self.cache[key] = (result, datetime.utcnow())
        logger.debug(f"[CACHE] Stored prediction (cache size: {len(self.cache)})")
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics and cost savings
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate cost savings
        # Each cache hit saves one model prediction computation
        cost_saved = self.hits * self.cost_per_prediction
        potential_cost = total_requests * self.cost_per_prediction
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "total_requests": total_requests,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cost_saved_today": round(cost_saved, 6),
            "cost_without_cache": round(potential_cost, 6),
            "savings_percent": round(hit_rate, 2)  # Hit rate = savings
        }
    
    def clear(self):
        """Clear all cached entries"""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"[CACHE] Cleared {count} entries")
    
    def invalidate_old(self, hours: int = 24):
        """
        Invalidate entries older than specified hours
        
        Args:
            hours: Age threshold in hours
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        before_count = len(self.cache)
        
        self.cache = {
            k: v for k, v in self.cache.items()
            if v[1] > cutoff
        }
        
        removed = before_count - len(self.cache)
        logger.info(f"[CACHE] Invalidated {removed} entries older than {hours}h")


# Global cache instance
prediction_cache = PredictionCache(max_size=1000, ttl_hours=24)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cached_or_predict(features: Dict, predict_func) -> Dict:
    """
    Get prediction from cache or compute if not cached
    
    Args:
        features: Invoice features
        predict_func: Function to call if cache miss
        
    Returns:
        Prediction result (cached or fresh)
    """
    # Try cache first
    cached_result = prediction_cache.get(features)
    
    if cached_result is not None:
        # Add cache indicator
        cached_result['from_cache'] = True
        return cached_result
    
    # Cache miss - compute prediction
    result = predict_func(features)
    
    # Cache the result
    prediction_cache.set(features, result)
    
    # Add cache indicator
    result['from_cache'] = False
    
    return result


def get_cache_stats() -> Dict:
    """Get cache statistics"""
    return prediction_cache.get_stats()


def clear_cache():
    """Clear all cached predictions"""
    prediction_cache.clear()


# ============================================================================
# CACHE WARMING (Startup Optimization)
# ============================================================================

def warm_cache_with_common_patterns(predict_func):
    """
    Pre-warm cache with common invoice patterns
    
    Reduces cold start latency for typical requests
    """
    logger.info("[CACHE] Warming cache with common patterns...")
    
    common_patterns = [
        # Good quality invoice
        {
            "blur_score": 20.0,
            "contrast_score": 80.0,
            "ocr_confidence": 0.95,
            "file_size_kb": 250.0,
            "vendor_name": "Common Vendor",
            "vendor_freq": 0.5,
            "total_amount": 1000.0,
            "invoice_number": "INV-001",
            "invoice_date": "2024-01-15",
            "currency": "USD"
        },
        # Poor quality invoice
        {
            "blur_score": 80.0,
            "contrast_score": 20.0,
            "ocr_confidence": 0.3,
            "file_size_kb": 100.0,
            "vendor_name": "Unknown",
            "vendor_freq": 0.01,
            "total_amount": 100.0,
            "invoice_number": "TEST",
            "invoice_date": "2024-01-01",
            "currency": "USD"
        }
    ]
    
    for pattern in common_patterns:
        result = predict_func(pattern)
        prediction_cache.set(pattern, result)
    
    logger.info(f"[CACHE] Warmed {len(common_patterns)} common patterns")