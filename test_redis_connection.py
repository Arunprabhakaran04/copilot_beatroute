import redis
import os
from dotenv import load_dotenv

load_dotenv()

try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=6379,
        decode_responses=True,
        socket_connect_timeout=5
    )
    
    r.ping()
    print("✅ Redis connection successful!")
    
    r.set("test_key", "Hello Redis!")
    value = r.get("test_key")
    print(f"✅ Redis read/write test: {value}")
    
    r.delete("test_key")
    print("✅ Redis is ready to use!")
    
except redis.ConnectionError:
    print("❌ Cannot connect to Redis.")
    print("Make sure Redis is running:")
    print("  - Docker: docker run -d -p 6379:6379 redis")
    print("  - WSL2: sudo service redis-server start")
    print("  - Windows: Start Redis service")
except Exception as e:
    print(f"❌ Error: {e}")
