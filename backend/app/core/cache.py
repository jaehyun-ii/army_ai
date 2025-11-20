"""
Redis caching layer for performance optimization.
"""
import json
import pickle
import hashlib
import asyncio
from typing import Any, Optional, Union, Callable, TypeVar, cast
from functools import wraps
from datetime import timedelta

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class CacheBackend:
    """Abstract cache backend interface."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        raise NotImplementedError

    async def clear(self, pattern: str = "*") -> int:
        """Clear cache keys matching pattern."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close cache connection."""
        pass


class RedisCache(CacheBackend):
    """Redis cache backend implementation."""

    def __init__(
        self,
        url: Optional[str] = None,
        prefix: str = "army_backend",
        default_expire: int = 300
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis URL
            prefix: Key prefix
            default_expire: Default expiration in seconds
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis not available. Install with: pip install redis"
            )

        self.url = url or settings.REDIS_URL
        self.prefix = prefix
        self.default_expire = default_expire
        self._client: Optional[Redis] = None

    async def _get_client(self) -> Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = await redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=False
            )
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            client = await self._get_client()
            data = await client.get(self._make_key(key))
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            client = await self._get_client()
            data = pickle.dumps(value)
            expire = expire or self.default_expire

            return await client.setex(
                self._make_key(key),
                expire,
                data
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            client = await self._get_client()
            result = await client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            client = await self._get_client()
            return await client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False

    async def clear(self, pattern: str = "*") -> int:
        """Clear cache keys matching pattern."""
        try:
            client = await self._get_client()
            pattern_with_prefix = f"{self.prefix}:{pattern}"

            # Use SCAN for better performance with large datasets
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await client.scan(
                    cursor,
                    match=pattern_with_prefix,
                    count=100
                )
                if keys:
                    deleted += await client.delete(*keys)
                if cursor == 0:
                    break

            return deleted
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


class InMemoryCache(CacheBackend):
    """Simple in-memory cache for development/testing."""

    def __init__(self, default_expire: int = 300):
        """Initialize in-memory cache."""
        self._cache: dict = {}
        self._expires: dict = {}
        self.default_expire = default_expire

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._expires:
            if asyncio.get_event_loop().time() > self._expires[key]:
                del self._cache[key]
                del self._expires[key]
                return None
        return self._cache.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        self._cache[key] = value
        expire = expire or self.default_expire
        self._expires[key] = asyncio.get_event_loop().time() + expire
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._expires:
                del self._expires[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._cache

    async def clear(self, pattern: str = "*") -> int:
        """Clear cache keys matching pattern."""
        if pattern == "*":
            count = len(self._cache)
            self._cache.clear()
            self._expires.clear()
            return count

        # Simple pattern matching (only supports * at end)
        deleted = 0
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            keys_to_delete = [
                k for k in self._cache.keys()
                if k.startswith(prefix)
            ]
            for key in keys_to_delete:
                await self.delete(key)
                deleted += 1

        return deleted


class CacheManager:
    """Main cache manager."""

    def __init__(self):
        """Initialize cache manager."""
        self._backend: Optional[CacheBackend] = None

    async def initialize(self) -> None:
        """Initialize cache backend based on configuration."""
        if settings.CACHE_TYPE == "redis" and REDIS_AVAILABLE:
            try:
                self._backend = RedisCache(
                    url=settings.REDIS_URL,
                    default_expire=settings.CACHE_TTL
                )
                # Test connection
                await self._backend.set("_test", "test", expire=1)
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}. Using in-memory cache.")
                self._backend = InMemoryCache(default_expire=settings.CACHE_TTL)
        else:
            self._backend = InMemoryCache(default_expire=settings.CACHE_TTL)
            logger.info("In-memory cache initialized")

    @property
    def backend(self) -> CacheBackend:
        """Get cache backend."""
        if self._backend is None:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        return self._backend

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return await self.backend.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        return await self.backend.set(key, value, expire)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        return await self.backend.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.backend.exists(key)

    async def clear(self, pattern: str = "*") -> int:
        """Clear cache keys matching pattern."""
        return await self.backend.clear(pattern)

    async def close(self) -> None:
        """Close cache connection."""
        if self._backend:
            await self._backend.close()


# Global cache manager instance
cache_manager = CacheManager()


class CacheKeyBuilder:
    """
    Builds cache keys from function arguments.

    This class provides flexible cache key generation with support for:
    - Custom prefixes
    - Automatic hashing of long keys
    - Configurable max key length
    """

    def __init__(self, max_length: int = 200, use_md5_hash: bool = True):
        """
        Initialize cache key builder.

        Args:
            max_length: Maximum key length before hashing
            use_md5_hash: Whether to hash long keys
        """
        self.max_length = max_length
        self.use_md5_hash = use_md5_hash

    def build(self, *args, prefix: Optional[str] = None, **kwargs) -> str:
        """
        Build cache key from arguments.

        Args:
            *args: Positional arguments
            prefix: Optional key prefix
            **kwargs: Keyword arguments

        Returns:
            Cache key string
        """
        key_parts = []

        if prefix:
            key_parts.append(prefix)

        # Add positional arguments
        key_parts.extend(str(arg) for arg in args)

        # Add keyword arguments (sorted for consistency)
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))

        key_string = ":".join(key_parts)

        # Hash if too long
        if self.use_md5_hash and len(key_string) > self.max_length:
            return hashlib.md5(key_string.encode()).hexdigest()

        return key_string

    def build_from_func(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        prefix: Optional[str] = None
    ) -> str:
        """
        Build cache key from function and its arguments.

        Args:
            func: Function being cached
            args: Function positional arguments
            kwargs: Function keyword arguments
            prefix: Optional key prefix (defaults to function name)

        Returns:
            Cache key string
        """
        func_name = func.__name__
        key_prefix = prefix or func_name
        return self.build(*args, prefix=key_prefix, **kwargs)


# Default key builder instance
default_key_builder = CacheKeyBuilder()


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from arguments (legacy function).

    This function is maintained for backward compatibility.
    Consider using CacheKeyBuilder directly for more control.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    return default_key_builder.build(*args, **kwargs)


def cached(
    expire: Optional[int] = None,
    key_prefix: Optional[str] = None,
    key_builder: Optional[Callable] = None,
    cache_key_builder: Optional[CacheKeyBuilder] = None
):
    """
    Decorator for caching function results.

    Args:
        expire: Cache expiration in seconds
        key_prefix: Prefix for cache key
        key_builder: Custom key builder function (legacy, for backward compatibility)
        cache_key_builder: CacheKeyBuilder instance for advanced key generation

    Usage:
        # Simple usage
        @cached(expire=300)
        async def get_user(user_id: int):
            return await fetch_user_from_db(user_id)

        # Advanced usage with custom key builder
        custom_builder = CacheKeyBuilder(max_length=100)
        @cached(expire=300, cache_key_builder=custom_builder)
        async def get_data(param1, param2):
            return await fetch_data(param1, param2)
    """
    # Use provided builder or default
    builder = cache_key_builder or default_key_builder

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                # Legacy custom key builder function
                cache_key_str = key_builder(*args, **kwargs)
            else:
                # Use CacheKeyBuilder
                func_name = func.__name__
                prefix = key_prefix or func_name
                cache_key_str = builder.build(*args, prefix=prefix, **kwargs)

            # Try to get from cache
            cached_value = await cache_manager.get(cache_key_str)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key_str}")
                return cached_value

            # Call function and cache result
            logger.debug(f"Cache miss: {cache_key_str}")
            result = await func(*args, **kwargs)

            # Cache the result
            await cache_manager.set(cache_key_str, result, expire)

            return result

        # Add cache control methods
        async def invalidate(*args, **kwargs):
            """Invalidate cache for these arguments."""
            if key_builder:
                cache_key_str = key_builder(*args, **kwargs)
            else:
                func_name = func.__name__
                prefix = key_prefix or func_name
                cache_key_str = builder.build(*args, prefix=prefix, **kwargs)

            return await cache_manager.delete(cache_key_str)

        wrapper.invalidate = invalidate
        wrapper.cache_key = cache_key
        wrapper.key_builder = builder

        return wrapper
    return decorator


class CacheTag:
    """Cache tagging for invalidation groups."""

    def __init__(self, tag: str):
        """Initialize cache tag."""
        self.tag = tag
        self._keys: set = set()

    async def add(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Add item to cache with tag."""
        # Store in cache
        result = await cache_manager.set(key, value, expire)

        if result:
            # Track key for this tag
            self._keys.add(key)

            # Store tag mapping
            tag_key = f"tag:{self.tag}"
            tags = await cache_manager.get(tag_key) or set()
            tags.add(key)
            await cache_manager.set(tag_key, tags, expire=86400)  # 1 day

        return result

    async def invalidate(self) -> int:
        """Invalidate all keys with this tag."""
        tag_key = f"tag:{self.tag}"
        keys = await cache_manager.get(tag_key) or set()

        deleted = 0
        for key in keys:
            if await cache_manager.delete(key):
                deleted += 1

        # Clear tag mapping
        await cache_manager.delete(tag_key)
        self._keys.clear()

        return deleted