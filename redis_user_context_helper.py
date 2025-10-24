"""
Redis Helper for UserContext

Shows how to store and retrieve UserContext in Redis.

Usage:
    # Store context
    await store_user_context(redis_client, user_id, context)
    
    # Retrieve context
    context = await get_user_context(redis_client, user_id)
"""

import redis.asyncio as redis
from typing import Optional
from loguru import logger

from user_context import UserContext


async def store_user_context(
    redis_client: redis.Redis,
    user_id: str,
    context: UserContext,
    ttl_seconds: int = 3600  # 1 hour default
) -> bool:
    """
    Store UserContext in Redis.
    
    Args:
        redis_client: Redis async client
        user_id: User identifier (used as Redis key)
        context: UserContext to store
        ttl_seconds: Time to live in seconds (default 1 hour)
        
    Returns:
        bool: True if stored successfully
    """
    try:
        key = f"user_context:{user_id}"
        
        # Serialize context to bytes
        serialized = context.to_redis_value()
        
        # Store in Redis with TTL
        await redis_client.setex(key, ttl_seconds, serialized)
        
        logger.info(f"✅ Stored UserContext for {user_id} in Redis (TTL: {ttl_seconds}s)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to store UserContext in Redis: {e}")
        return False


async def get_user_context(
    redis_client: redis.Redis,
    user_id: str
) -> Optional[UserContext]:
    """
    Retrieve UserContext from Redis.
    
    Args:
        redis_client: Redis async client
        user_id: User identifier
        
    Returns:
        UserContext if found, None otherwise
    """
    try:
        key = f"user_context:{user_id}"
        
        # Get from Redis
        serialized = await redis_client.get(key)
        
        if serialized is None:
            logger.warning(f"⚠️ No UserContext found in Redis for {user_id}")
            return None
        
        # Deserialize
        context = UserContext.from_redis_value(serialized)
        
        logger.info(f"✅ Retrieved UserContext for {user_id} from Redis")
        return context
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve UserContext from Redis: {e}")
        return None


async def delete_user_context(
    redis_client: redis.Redis,
    user_id: str
) -> bool:
    """
    Delete UserContext from Redis.
    
    Args:
        redis_client: Redis async client
        user_id: User identifier
        
    Returns:
        bool: True if deleted successfully
    """
    try:
        key = f"user_context:{user_id}"
        await redis_client.delete(key)
        
        logger.info(f"✅ Deleted UserContext for {user_id} from Redis")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete UserContext from Redis: {e}")
        return False


# Example usage in WebSocket server:
"""
# Initialize Redis client
redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)

# In handle_init():
# After loading schema, store in Redis
await store_user_context(redis_client, user_id, context, ttl_seconds=3600)

# In handle_query():
# Try to get from Redis first
context = await get_user_context(redis_client, user_id)
if context is None:
    # Schema not in Redis, need to reload
    await websocket.send(json.dumps({
        "success": False,
        "error": "Session expired. Please reinitialize."
    }))
    return
"""
