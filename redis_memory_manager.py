import time
import re
import json
import logging
from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
import redis
import os
from dotenv import load_dotenv
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, date

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_SESSION_ID = "cli_session"
DEFAULT_USER_ID = "default_user"
DEFAULT_TTL = 3600


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle DataFrames, Series, and other non-serializable types"""
    
    def default(self, obj):
        # Handle pandas DataFrame
        if isinstance(obj, pd.DataFrame):
            return {
                "_type": "dataframe",
                "data": obj.to_dict(orient="records"),
                "columns": list(obj.columns),
                "index": list(obj.index)
            }
        
        # Handle pandas Series
        elif isinstance(obj, pd.Series):
            return {
                "_type": "series",
                "data": obj.to_dict(),
                "index": list(obj.index),
                "name": obj.name
            }
        
        # Handle numpy types
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle datetime objects
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle sets
        elif isinstance(obj, set):
            return list(obj)
        
        # For any other type, convert to string as fallback
        else:
            try:
                return super().default(obj)
            except TypeError:
                return str(obj)


def get_redis_connection():
    """Get Redis connection with error handling"""
    try:
        r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        r.ping()
        return r
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.error(f"Redis connection failed: {e}")
        return None


class RedisSessionManager:
    """Redis-backed session management for conversation memory"""
    
    def __init__(self):
        self.redis_client = get_redis_connection()
        self.ttl = DEFAULT_TTL
        
        if self.redis_client:
            logger.info("Redis session manager initialized successfully")
        else:
            logger.warning("Redis unavailable - operating in degraded mode")
    
    def _get_key(self, user_id: str, session_id: str, suffix: str) -> str:
        """Generate Redis key with user and session namespacing"""
        return f"user:{user_id}:session:{session_id}:{suffix}"
    
    def _refresh_ttl(self, user_id: str, session_id: str):
        """Refresh TTL for all session keys to prevent expiration during active use"""
        if not self.redis_client:
            return
        
        try:
            pattern = f"user:{user_id}:session:{session_id}:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                self.redis_client.expire(key, self.ttl)
            
            if keys:
                logger.debug(f"Refreshed TTL for {len(keys)} keys in session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to refresh session TTL: {e}")
    
    def set_data(self, user_id: str, session_id: str, key: str, value: Any, ttl: Optional[int] = None):
        """Store data in Redis with session namespacing"""
        if not self.redis_client:
            logger.warning(f"Redis unavailable - cannot store {key}")
            return False
        
        try:
            redis_key = self._get_key(user_id, session_id, key)
            # Use custom serializer to handle DataFrames and other non-JSON types
            serialized = json.dumps(value, cls=CustomJSONEncoder, default=str)
            self.redis_client.setex(redis_key, ttl or self.ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Failed to set Redis data for key {key}: {e}")
            return False
    
    def get_data(self, user_id: str, session_id: str, key: str) -> Optional[Any]:
        """Retrieve data from Redis"""
        if not self.redis_client:
            return None
        
        try:
            redis_key = self._get_key(user_id, session_id, key)
            data = self.redis_client.get(redis_key)
            
            if data:
                self._refresh_ttl(user_id, session_id)
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get Redis data for key {key}: {e}")
            return None
    
    def delete_data(self, user_id: str, session_id: str, key: str) -> bool:
        """Delete specific key from Redis"""
        if not self.redis_client:
            return False
        
        try:
            redis_key = self._get_key(user_id, session_id, key)
            self.redis_client.delete(redis_key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete Redis key {key}: {e}")
            return False
    
    def clear_session(self, user_id: str, session_id: str) -> bool:
        """Clear all data for a specific session"""
        if not self.redis_client:
            return False
        
        try:
            pattern = f"user:{user_id}:session:{session_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {e}")
            return False
    
    def session_exists(self, user_id: str, session_id: str) -> bool:
        """Check if session has any data"""
        if not self.redis_client:
            return False
        
        try:
            pattern = f"user:{user_id}:session:{session_id}:*"
            keys = self.redis_client.keys(pattern)
            return len(keys) > 0
        except Exception as e:
            logger.error(f"Failed to check session existence: {e}")
            return False


class RedisClassificationValidator:
    """Redis-backed classification validator"""
    
    def __init__(self, session_manager: RedisSessionManager):
        self.session_manager = session_manager
        self.max_history = 100
    
    def validate_classification(
        self,
        user_id: str,
        session_id: str,
        query: str,
        predicted_agent: str,
        actual_result: str
    ) -> bool:
        """Validate and store classification result"""
        is_correct = actual_result == "completed"
        
        history = self.session_manager.get_data(user_id, session_id, "classification_history") or []
        
        entry = {
            "query": query,
            "predicted": predicted_agent,
            "correct": is_correct,
            "timestamp": time.time()
        }
        
        history.append(entry)
        
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        
        self.session_manager.set_data(user_id, session_id, "classification_history", history)
        
        if not is_correct:
            self._record_error_pattern(user_id, session_id, query, predicted_agent)
        
        return is_correct
    
    def _record_error_pattern(self, user_id: str, session_id: str, query: str, wrong_agent: str):
        """Record patterns that lead to misclassification"""
        error_patterns = self.session_manager.get_data(user_id, session_id, "error_patterns") or {}
        
        key_phrases = re.findall(r'\b\w+\b', query.lower())
        for phrase in key_phrases:
            if phrase not in error_patterns:
                error_patterns[phrase] = {}
            if wrong_agent not in error_patterns[phrase]:
                error_patterns[phrase][wrong_agent] = 0
            error_patterns[phrase][wrong_agent] += 1
        
        self.session_manager.set_data(user_id, session_id, "error_patterns", error_patterns)
    
    def get_classification_accuracy(self, user_id: str, session_id: str) -> float:
        """Calculate recent classification accuracy"""
        history = self.session_manager.get_data(user_id, session_id, "classification_history") or []
        
        if not history:
            return 0.0
        
        correct = sum(1 for entry in history if entry["correct"])
        return correct / len(history)
    
    def get_problematic_patterns(self, user_id: str, session_id: str) -> List[str]:
        """Get patterns that frequently lead to misclassification"""
        error_patterns = self.session_manager.get_data(user_id, session_id, "error_patterns") or {}
        
        problematic = []
        for phrase, agents in error_patterns.items():
            total_errors = sum(agents.values())
            if total_errors >= 3:
                problematic.append(phrase)
        return problematic


class RedisMemoryManager:
    """Redis-backed conversation memory manager"""
    
    def __init__(self, llm, max_history: int = 3):
        self.llm = llm
        self.max_history = max_history
        self.session_manager = RedisSessionManager()
        self.enrichment_cache = {}
        self.failed_patterns = set()
        self.enrichment_timeout = 10
    
    def add_entry(
        self,
        session_id: str,
        original_query: str,
        enriched_query: str,
        agent_type: str,
        result: Dict[str, Any],
        user_id: str = DEFAULT_USER_ID
    ):
        """Add conversation entry to Redis"""
        # Serialize result to make it JSON-compatible (handle DataFrames, etc.)
        serialized_result = self._serialize_result(result)
        
        entry = {
            "original": original_query,
            "enriched": enriched_query,
            "signature": self.compute_query_signature(original_query, result.get("meta")),
            "agent_type": agent_type,
            "result": serialized_result,
            "timestamp": time.time()
        }
        
        history = self.session_manager.get_data(user_id, session_id, "conversation_history") or []
        history.append(entry)
        
        if len(history) > self.max_history:
            history = history[-self.max_history:]
        
        success = self.session_manager.set_data(user_id, session_id, "conversation_history", history)
        
        if success:
            logger.info(f"Added memory entry: {agent_type} - {original_query[:50]}...")
        else:
            logger.warning(f"Failed to add memory entry to Redis")
    
    def _serialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert result to JSON-serializable format (handle DataFrames, etc.)"""
        try:
            import pandas as pd
            serialized = {}
            
            for key, value in result.items():
                if isinstance(value, pd.DataFrame):
                    # Convert DataFrame to dict with records format for easy reconstruction
                    serialized[key] = {
                        "_type": "dataframe",
                        "data": value.to_dict(orient="records"),
                        "columns": list(value.columns)
                    }
                elif isinstance(value, pd.Series):
                    serialized[key] = {
                        "_type": "series",
                        "data": value.to_dict()
                    }
                elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
                    serialized[key] = value
                else:
                    # For any other non-serializable type, convert to string
                    serialized[key] = str(value)
            
            return serialized
        except ImportError:
            # pandas not available, return as-is
            return result
        except Exception as e:
            logger.warning(f"Failed to serialize result: {e}. Storing string representation.")
            return {k: str(v) if not isinstance(v, (list, dict, str, int, float, bool, type(None))) else v 
                    for k, v in result.items()}
    
    def _deserialize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct DataFrames and other objects from serialized format"""
        try:
            deserialized = {}
            
            for key, value in result.items():
                if isinstance(value, dict):
                    # Check if it's a serialized DataFrame
                    if value.get("_type") == "dataframe":
                        df = pd.DataFrame(value.get("data", []))
                        if "columns" in value:
                            df.columns = value["columns"]
                        deserialized[key] = df
                        logger.debug(f"Deserialized DataFrame with {len(df)} rows for key '{key}'")
                    
                    # Check if it's a serialized Series
                    elif value.get("_type") == "series":
                        deserialized[key] = pd.Series(value.get("data", {}))
                        logger.debug(f"Deserialized Series for key '{key}'")
                    
                    else:
                        deserialized[key] = value
                else:
                    deserialized[key] = value
            
            return deserialized
        except Exception as e:
            logger.warning(f"Failed to deserialize result: {e}. Returning as-is.")
            return result
    
    def get_recent(self, session_id: str, n: int = 3, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        """Get recent conversation entries with deserialized DataFrames"""
        history = self.session_manager.get_data(user_id, session_id, "conversation_history") or []
        
        # Deserialize DataFrames in results
        for entry in history:
            if "result" in entry and isinstance(entry["result"], dict):
                entry["result"] = self._deserialize_result(entry["result"])
        
        recent_entries = history[-n:] if history else []
        return sorted(recent_entries, key=lambda x: x.get('timestamp', 0), reverse=True)

    def compute_query_signature(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Compute a canonical signature for a query with improved semantic normalization.

        - Normalizes whitespace and lowercases
        - Replaces numeric tokens with a placeholder
        - Normalizes month names and time references
        - Removes filler words and variations
        - Optionally includes structured params (dates, ranges) to make signature precise
        Returns a short hex digest to use as a cache key.
        """
        if not isinstance(query, str):
            query = str(query)

        normalized = query.lower().strip()
        
        # Remove punctuation at end
        normalized = normalized.rstrip('?!.')
        
        # Normalize greetings and pleasantries (they don't change the query intent)
        normalized = re.sub(r"^(great|good|thanks|thank you|ok|okay|nice|excellent|perfect),?\s*", "", normalized)
        normalized = re.sub(r"^(now|then|also)\s+", "", normalized)
        
        # Remove politeness markers FIRST (before question word normalization)
        normalized = re.sub(r"^(please|can you|could you|would you|tell me)\s+", "", normalized)
        
        # Normalize question words and imperative variations
        normalized = re.sub(r"^(what\s+(are|is)\s+the\s+|what\s+are\s+|what\s+is\s+)", "get ", normalized)
        normalized = re.sub(r"^(give me|show me|get me|display|fetch|retrieve)\s+", "get ", normalized)
        
        # Normalize "top N" queries (important for SKU queries)
        normalized = re.sub(r"\b(the\s+)?top\s+", "top-", normalized)
        
        # Normalize "for" prepositions
        normalized = re.sub(r"\b(for\s+the\s+month\s+of|for\s+month\s+of|for\s+the\s+month|for\s+month|for)\s+", "for ", normalized)
        
        # Normalize month names to consistent format BEFORE number replacement
        month_map = {
            "january": "month-01", "jan": "month-01",
            "february": "month-02", "feb": "month-02",
            "march": "month-03", "mar": "month-03",
            "april": "month-04", "apr": "month-04",
            "may": "month-05",
            "june": "month-06", "jun": "month-06",
            "july": "month-07", "jul": "month-07",
            "august": "month-08", "aug": "month-08",
            "september": "month-09", "september": "month-09", "sept": "month-09", "sep": "month-09",
            "october": "month-10", "oct": "month-10",
            "november": "month-11", "nov": "month-11",
            "december": "month-12", "dec": "month-12"
        }
        for month_name, month_code in month_map.items():
            normalized = re.sub(r"\b" + month_name + r"\b", month_code, normalized)
        
        # Replace dates with placeholders (but preserve numbers in queries)
        # Dates like "10/20/2024" or "2024-09-01" become DATE
        normalized = re.sub(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", "DATE", normalized)
        normalized = re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", normalized)
        
        # IMPORTANT: DO NOT replace simple numbers like "top 10", "last 3", etc.
        # These numbers are semantically meaningful and should be preserved
        # Only replace numbers in non-semantic contexts (e.g., IDs, codes)
        # For now, KEEP numbers to ensure cache accuracy
        # Future: Use smarter context-aware replacement
        
        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # If structured params provided, include them in signature deterministically
        params_part = ""
        if params and isinstance(params, dict):
            try:
                # sorted keys to keep stable order
                params_part = json.dumps({k: params[k] for k in sorted(params.keys())}, sort_keys=True)
            except Exception:
                params_part = str(params)

        digest_source = normalized + "|" + params_part
        h = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()
        # Keep a shorter key for readability
        return h[:16]

    def get_cached_result_by_signature(self, session_id: str, signature: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        """Retrieve a cached conversation entry that has the given signature with deserialized DataFrames."""
        history = self.session_manager.get_data(user_id, session_id, "conversation_history") or []
        for entry in reversed(history):
            if entry.get("signature") == signature:
                result = entry.get("result")
                # Deserialize DataFrames if present
                if result and isinstance(result, dict):
                    result = self._deserialize_result(result)
                return result
        return None

    def get_cached_result_for_query(self, session_id: str, query: str, user_id: str = DEFAULT_USER_ID, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Compute the signature for a query and return cached result if present with deserialized DataFrames."""
        sig = self.compute_query_signature(query, params)
        cached = self.get_cached_result_by_signature(session_id, sig, user_id)
        
        if cached:
            logger.info(f"✅ Cache HIT for query signature: {sig}")
        else:
            logger.debug(f"❌ Cache MISS for query signature: {sig}")
        
        return cached
    
    def clear_memory(self, session_id: str, user_id: str = DEFAULT_USER_ID):
        """Clear all memory for a session"""
        success = self.session_manager.clear_session(user_id, session_id)
        if success:
            logger.info(f"Memory cleared for session {session_id}")
        else:
            logger.warning(f"Failed to clear memory for session {session_id}")
    
    def get_memory_summary(self, session_id: str, user_id: str = DEFAULT_USER_ID) -> str:
        """Get a summary of current memory state"""
        history = self.session_manager.get_data(user_id, session_id, "conversation_history") or []
        
        if not history:
            return "No memory entries"
        
        summary = f"Memory: {len(history)} entries\n"
        for i, entry in enumerate(history):
            summary += f"  {i+1}. {entry['agent_type']}: {entry['original'][:30]}...\n"
        return summary
    
    def _heuristic_expand(self, query: str, history: List[Dict[str, Any]]) -> Optional[str]:
        """Simple heuristic-based query expansion - handles 90% of follow-ups"""
        lowered = query.lower()
        
        # Pattern 1: References to meeting dates
        if any(token in lowered for token in ["that date", "that day", "that time", "that slot"]):
            for entry in reversed(history):
                if entry.get("agent_type") == "meeting":
                    meeting_date = entry.get("result", {}).get("meeting_date") or entry.get("result", {}).get("parsed_date")
                    if meeting_date:
                        return re.sub(r"(?i)that date|that day|that time|that slot", meeting_date, query)
        
        # Pattern 2: "Send confirmation" - inject most recent meeting/action details
        if "send confirmation" in lowered or "send a confirmation" in lowered:
            email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', query)
            if email_match:
                email = email_match.group(0)
                # Find most recent meeting
                for entry in reversed(history):
                    if entry.get("agent_type") == "meeting":
                        result = entry.get("result", {})
                        user_id = result.get("user_id", "")
                        meeting_date = result.get("meeting_date") or result.get("parsed_date", "")
                        start_time = result.get("start_time", "")
                        if user_id and meeting_date:
                            enriched = f"Send confirmation email to {email} about meeting with User {user_id} on {meeting_date}"
                            if start_time:
                                enriched += f" at {start_time}"
                            logger.info(f"Heuristic: send confirmation → meeting details injected")
                            return enriched
        
        # Pattern 3: "Also send details about the meeting" - critical for optimization
        if any(pattern in lowered for pattern in ["also send", "send the details", "send details", "email the details"]):
            if any(word in lowered for word in ["meeting", "appointment", "schedule"]):
                # Find most recent meeting
                for entry in reversed(history):
                    if entry.get("agent_type") == "meeting":
                        result = entry.get("result", {})
                        user_id = result.get("user_id", "")
                        meeting_date = result.get("meeting_date") or result.get("parsed_date", "")
                        
                        # Extract email from current query
                        email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', query)
                        if email_match and user_id and meeting_date:
                            email = email_match.group(0)
                            # Create enriched query that DIRECTLY uses the cached details
                            enriched = f"Send email to {email} with details about meeting scheduled with User {user_id} on {meeting_date}"
                            logger.info(f"Heuristic expansion: using cached meeting details (User {user_id}, {meeting_date})")
                            return enriched
        
        # Pattern 4: "Also cc" or "cc someone on that email"
        if ("also cc" in lowered or "cc" in lowered) and "@" in query:
            cc_email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', query)
            if cc_email_match:
                cc_email = cc_email_match.group(0)
                # Find most recent entry to determine context
                if history:
                    last_entry = history[-1]
                    agent_type = last_entry.get("agent_type")
                    result = last_entry.get("result", {})
                    
                    if agent_type == "db_query":
                        enriched = f"Send email to {cc_email} (cc) with the database query results from previous query"
                        logger.info(f"Heuristic: cc → DB query results")
                        return enriched
                    elif agent_type == "meeting":
                        user_id = result.get("user_id", "")
                        meeting_date = result.get("meeting_date") or result.get("parsed_date", "")
                        enriched = f"Send email to {cc_email} (cc) with meeting details: User {user_id} on {meeting_date}"
                        logger.info(f"Heuristic: cc → meeting details")
                        return enriched
        
        # Pattern 5: "Send that to..." or "send it to..." or "email that to..." or "send this data"
        if any(p in lowered for p in ["send that to", "send it to", "email that to", "send that information to", 
                                       "send this data", "send this to", "send the data", "email this to"]):
            email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', query)
            if email_match:
                email = email_match.group(0)
                # Find most recent entry with data to send
                for entry in reversed(history):
                    agent_type = entry.get("agent_type")
                    result = entry.get("result", {})
                    
                    if agent_type == "db_query" and ("query_results" in result or "query_data" in result or "data" in result):
                        # Check if we have structured data
                        query_data = result.get("query_results") or result.get("query_data") or result.get("data")
                        rows = result.get("rows_returned", 0)
                        
                        enriched = f"Send email to {email} with the database query results from the previous query ({rows} rows)"
                        logger.info(f"Heuristic expansion: 'send this data' → using cached DB query results ({rows} rows)")
                        return enriched
                    elif agent_type == "meeting" and "user_id" in result:
                        user_id = result.get("user_id", "")
                        meeting_date = result.get("meeting_date") or result.get("parsed_date", "")
                        enriched = f"Send email to {email} with details about meeting scheduled with User {user_id} on {meeting_date}"
                        logger.info(f"Heuristic expansion: using cached meeting details")
                        return enriched
        
        # Pattern 6: "What was the result?" - summarize most recent action
        if any(p in lowered for p in ["what was the result", "what was the outcome", "what happened"]):
            if history:
                last_entry = history[-1]
                agent_type = last_entry.get("agent_type")
                result = last_entry.get("result", {})
                
                if agent_type == "db_query":
                    rows = result.get("rows_returned", 0)
                    enriched = f"Summarize the database query result which returned {rows} rows"
                    logger.info(f"Heuristic: 'what was result' → DB summary")
                    return enriched
                elif agent_type == "meeting":
                    user_id = result.get("user_id", "")
                    meeting_date = result.get("meeting_date") or result.get("parsed_date", "")
                    enriched = f"Summarize meeting scheduled with User {user_id} on {meeting_date}"
                    logger.info(f"Heuristic: 'what was result' → meeting summary")
                    return enriched
        
        # Pattern 7: Multi-step complex references - let LLM handle these
        # Examples: "send metrics from step 1 and mention meeting"
        if re.search(r'\b(step \d|from step|and mention|combine|both)\b', lowered):
            logger.debug("Complex multi-reference detected - will use LLM")
            return None  # LLM will handle complex multi-step references
        
        # Original ambiguous patterns - return None to use LLM enrichment
        ambiguous_patterns = [
            "confirm the", "update the", "change the", "modify the",
            "for the meeting", "for that meeting", "to the meeting",
            "the same", "same meeting", "same time", "same date"
        ]
        
        if any(pattern in lowered for pattern in ambiguous_patterns):
            return None
        
        return None
    
    def enrich_query(self, session_id: str, original_query: str, user_id: str = DEFAULT_USER_ID) -> str:
        """Enrich query with context from conversation history"""
        history = self.session_manager.get_data(user_id, session_id, "conversation_history") or []

        if not history:
            return original_query

        # First, use existing lightweight heuristics (fast, deterministic)
        heur = self._heuristic_expand(original_query, history)
        if heur:
            return heur

        lowered = original_query.lower()
        needs_enrichment = any([
            "also" in lowered,
            "the meeting" in lowered,
            "that" in lowered and ("date" in lowered or "time" in lowered or "meeting" in lowered),
            "same" in lowered,
            "confirm" in lowered and "location" in lowered,
            "invite" in lowered and "@" in original_query,
            "cc" in lowered and "@" in original_query,
            "same subject" in lowered,
            "same email" in lowered,
            "add to" in lowered and ("email" in lowered or "@" in original_query),
            "send to" in lowered and "also" in lowered,
        ])

        if not needs_enrichment:
            return original_query

        recent = self.get_recent(session_id, self.max_history, user_id)

        # Build recent context for LLM enrichment
        context_lines = []
        for i, e in enumerate(recent):
            small_result = {k: v for k, v in (e.get("result") or {}).items() if isinstance(v, (str, int, float))}
            priority_marker = "[MOST RECENT]" if i == 0 else f"[{len(recent)-i} ago]"
            context_lines.append(f"{priority_marker} User: {e['original']} --> Agent: {e.get('agent_type')} --> Result: {small_result}")

        context_text = "\n".join(context_lines)

        enrich_prompt = ChatPromptTemplate.from_template("""
        You are a context-aware query rewriter for a multi-agent system. Your job is to combine RECENT CONTEXT with the CURRENT USER QUERY to create a complete, unambiguous instruction.

        RECENT INTERACTIONS (ordered by recency, MOST RECENT first):
        {recent_context}

        CURRENT USER QUERY (the new instruction): {query}

        CRITICAL RULES FOR COMBINING CONTEXT + CURRENT QUERY:
        1. PRIORITIZE THE [MOST RECENT] ENTRY - This is the most relevant context
        2. The CURRENT QUERY contains NEW INSTRUCTIONS that must be followed
        3. Use context ONLY to fill in missing details (subjects, dates, recipients, meeting info)
        4. IMPORTANT: When current query is a follow-up to an already completed task (e.g., "also send details about the meeting") - USE the cached details from the most recent relevant context and DO NOT re-run the original agent.
        5. PRESERVE all new information from current query (new emails, new dates, new instructions)
        6. AVOID DUPLICATION: Never ask to re-execute a task that was already completed in recent context

        Respond in this EXACT format (no extra commentary):
        ENRICHED_QUERY: [the complete instruction combining context + current query]
        """)

        try:
            messages = enrich_prompt.format_messages(recent_context=context_text, query=original_query)
            response = self.llm.invoke(messages)
            content = response.content.strip()
            match = re.search(r'ENRICHED_QUERY:\s*(.+)', content, re.DOTALL)
            if match:
                enriched = match.group(1).strip()
                if enriched and self._validate_enriched_query(original_query, enriched):
                    logger.info(f"Query enriched: '{original_query}' → '{enriched}'")
                    return enriched
        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}")

        # If LLM enrichment failed or returned something ambiguous, fall back to deterministic cache-based enrichment
        try:
            # Look for recent meeting or db_query results we can reuse
            for entry in reversed(recent):
                agent_type = entry.get("agent_type")
                result = entry.get("result") or {}

                # Follow-ups that request sending details should reuse meeting/db info
                if agent_type == "meeting" and any(tok in lowered for tok in ["send", "also send", "send details", "email"]):
                    meeting_date = result.get("meeting_date") or result.get("parsed_date")
                    meeting_user = result.get("user_id") or result.get("participant") or result.get("user")
                    email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', original_query)
                    email = email_match.group(0) if email_match else result.get("email") or result.get("organizer_email")

                    if meeting_user and meeting_date and email:
                        enriched = f"Send email to {email} with details about meeting scheduled with User {meeting_user} on {meeting_date}"
                        logger.info("Fallback enrichment: using cached meeting details to avoid re-execution")
                        return enriched

                if agent_type == "db_query" and any(tok in lowered for tok in ["send", "also send", "send details", "email"]) :
                    email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', original_query)
                    email = email_match.group(0) if email_match else None
                    if email:
                        enriched = f"Send email to {email} with the database query results from the previous query"
                        logger.info("Fallback enrichment: using cached DB query results to avoid re-execution")
                        return enriched
        except Exception as e:
            logger.debug(f"Fallback enrichment failed: {e}")

        # As a last resort, return the original query (system will re-classify/execute)
        return original_query
    
    def _validate_enriched_query(self, original: str, enriched: str) -> bool:
        """Validate that enriched query makes sense"""
        if not enriched or len(enriched.strip()) == 0:
            return False
        
        if len(enriched) > len(original) * 4:
            return False
        
        original_emails = re.findall(r'\b\w+@[\w.-]+\.\w+\b', original)
        enriched_emails = re.findall(r'\b\w+@[\w.-]+\.\w+\b', enriched)
        
        for email in original_emails:
            if email not in enriched:
                return False
        
        return True


def get_last_db_query_result(session_id: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent DB query result from Redis"""
    session_manager = RedisSessionManager()
    history = session_manager.get_data(user_id, session_id, "conversation_history") or []
    
    for entry in reversed(history):
        if entry.get("agent_type") == "db_query":
            result = entry.get("result", {})
            
            # Handle nested structure from main.py (result -> final_result -> query_data)
            # Check if result has the nested "final_result" structure
            if "final_result" in result:
                result = result["final_result"]
            
            # Check for various result field names (query_results, query_data, data)
            if any(key in result for key in ["query_results", "query_data", "data", "rows_returned"]):
                logger.info(f"Retrieved cached DB query result from session {session_id}")
                return result
    
    logger.info(f"No DB query result found in session {session_id}")
    return None


def initialize_session(user_id: str, session_id: str):
    """Initialize a new session in Redis"""
    session_manager = RedisSessionManager()
    
    if not session_manager.session_exists(user_id, session_id):
        session_manager.set_data(user_id, session_id, "conversation_history", [])
        session_manager.set_data(user_id, session_id, "classification_history", [])
        session_manager.set_data(user_id, session_id, "error_patterns", {})
        logger.info(f"Initialized new session: {session_id} for user: {user_id}")
    else:
        logger.info(f"Session {session_id} already exists for user {user_id}")
