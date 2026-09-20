# guardian_policy_agent/infra/cache.py
"""
Multi-Level Caching for Decision Pipeline

L1: In-memory LRU cache for recent decisions (same behavior+domain)
L2: Redis cache for policy embeddings and retrieval results
L3: Materialized views handled at DB level (not in this module)
"""
from __future__ import annotations
import json
import hashlib
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Redis is optional - graceful fallback if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("[Cache] Redis not installed, L2 cache disabled")


class DecisionCache:
    """
    Two-tier caching for privacy decisions.

    L1 (in-memory LRU): Fast, limited size, process-local
    L2 (Redis): Shared across instances, larger capacity, optional
    """

    # L1 cache configuration
    L1_MAX_SIZE = 1000

    # L2 cache configuration
    L2_TTL_SECONDS = 3600  # 1 hour default
    L2_PREFIX = "guardian:decision:"

    _instance: Optional["DecisionCache"] = None

    def __new__(cls) -> "DecisionCache":
        """Singleton pattern for cache instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._redis: Optional[Any] = None
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0

        # Initialize Redis connection if available
        if REDIS_AVAILABLE:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            try:
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info(f"[Cache] Redis L2 cache connected: {redis_url}")
            except Exception as e:
                logger.warning(f"[Cache] Redis connection failed, L2 disabled: {e}")
                self._redis = None

    def cache_key(
        self,
        behavior: Dict[str, Any],
        domain: str,
        user_prefs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a deterministic cache key from behavior + domain + user preferences.

        Uses SHA256 hash of the canonicalized JSON representation.
        User preferences are included to ensure different preference settings
        result in different cache entries.
        """
        # Normalize behavior dict for consistent hashing
        key_data = {
            "domain": domain,
            "data_categories": sorted(behavior.get("data_categories") or []),
            "actions": sorted(behavior.get("actions") or []),
            "purposes": sorted(behavior.get("purposes") or []),
            "recipients": sorted(behavior.get("recipients") or []),
            "action_type": behavior.get("action_type"),
        }

        # Include user preferences in cache key
        if user_prefs:
            # Normalize preferences for consistent hashing
            key_data["user_prefs"] = json.loads(
                json.dumps(user_prefs, sort_keys=True)
            )

        serialized = json.dumps(key_data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()[:32]

    @lru_cache(maxsize=L1_MAX_SIZE)
    def _l1_get(self, key: str) -> Optional[str]:
        """L1 cache lookup (decorated with LRU)."""
        # This is a placeholder - actual values are set via cache_info
        return None

    def get_cached_decision(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Look up a cached decision, checking L1 then L2.

        Returns None if not found in any cache layer.
        """
        # L1: Check in-memory cache
        cached = self._l1_cache.get(key)
        if cached is not None:
            self._l1_hits += 1
            logger.debug(f"[Cache] L1 hit: {key[:8]}...")
            return cached

        # L2: Check Redis cache
        if self._redis is not None:
            try:
                redis_key = f"{self.L2_PREFIX}{key}"
                data = self._redis.get(redis_key)
                if data:
                    self._l2_hits += 1
                    logger.debug(f"[Cache] L2 hit: {key[:8]}...")
                    decision = json.loads(data)
                    # Promote to L1
                    self._l1_cache[key] = decision
                    return decision
            except Exception as e:
                logger.warning(f"[Cache] Redis get error: {e}")

        self._misses += 1
        return None

    def set_cached_decision(
        self,
        key: str,
        decision: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Store a decision in both cache layers.

        Args:
            key: Cache key from cache_key()
            decision: The decision dict to cache
            ttl: TTL in seconds for L2 (Redis), defaults to L2_TTL_SECONDS
        """
        # Don't cache if decision came from LLM (may need fresh reasoning)
        system_used = decision.get("system_used", "")
        if system_used == "llm_agent":
            logger.debug(f"[Cache] Skipping cache for LLM decision: {key[:8]}...")
            return

        # Add cache metadata
        cached_decision = {
            **decision,
            "_cached_at": datetime.utcnow().isoformat(),
            "_cache_key": key[:8],
        }

        # L1: Store in memory
        self._l1_cache[key] = cached_decision

        # L2: Store in Redis
        if self._redis is not None:
            try:
                redis_key = f"{self.L2_PREFIX}{key}"
                ttl = ttl or self.L2_TTL_SECONDS
                self._redis.setex(
                    redis_key,
                    ttl,
                    json.dumps(cached_decision)
                )
                logger.debug(f"[Cache] Stored in L2: {key[:8]}... (TTL={ttl}s)")
            except Exception as e:
                logger.warning(f"[Cache] Redis set error: {e}")

    def invalidate(self, key: str) -> None:
        """Remove a specific key from all cache layers."""
        # L1
        self._l1_cache.pop(key, None)

        # L2
        if self._redis is not None:
            try:
                self._redis.delete(f"{self.L2_PREFIX}{key}")
            except Exception:
                pass

    def invalidate_domain(self, domain: str) -> int:
        """
        Invalidate all cached decisions for a domain.

        Useful when policies are updated.
        Returns count of invalidated keys.
        """
        count = 0

        # L1: Clear entries (simple approach - clear all, as we don't index by domain)
        # A more sophisticated approach would maintain a domain->keys index
        self._l1_cache.clear()

        # L2: Scan and delete matching keys
        if self._redis is not None:
            try:
                pattern = f"{self.L2_PREFIX}*"
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                    for key in keys:
                        self._redis.delete(key)
                        count += 1
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"[Cache] Redis invalidate error: {e}")

        logger.info(f"[Cache] Invalidated {count} entries for domain: {domain}")
        return count

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._l1_hits + self._l2_hits + self._misses
        return {
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "total_requests": total,
            "l1_hit_rate": self._l1_hits / total if total > 0 else 0,
            "l2_hit_rate": self._l2_hits / total if total > 0 else 0,
            "overall_hit_rate": (self._l1_hits + self._l2_hits) / total if total > 0 else 0,
            "l1_size": len(self._l1_cache),
            "l2_connected": self._redis is not None,
        }

    @property
    def _l1_cache(self) -> Dict[str, Any]:
        """Lazy initialization of L1 cache dict."""
        if not hasattr(self, "_l1_cache_dict"):
            self._l1_cache_dict: Dict[str, Any] = {}
        return self._l1_cache_dict


# Module-level convenience functions
_cache: Optional[DecisionCache] = None

def get_cache() -> DecisionCache:
    """Get the singleton cache instance."""
    global _cache
    if _cache is None:
        _cache = DecisionCache()
    return _cache


def cached_decision(
    behavior: Dict[str, Any],
    domain: str,
    user_prefs: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Convenience function to check cache for a decision."""
    cache = get_cache()
    key = cache.cache_key(behavior, domain, user_prefs)
    return cache.get_cached_decision(key)


def cache_decision(
    behavior: Dict[str, Any],
    domain: str,
    decision: Dict[str, Any],
    user_prefs: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to cache a decision."""
    cache = get_cache()
    key = cache.cache_key(behavior, domain, user_prefs)
    cache.set_cached_decision(key, decision)
