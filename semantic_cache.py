"""
Semantic cache implementation for query result caching based on embedding similarity.

This module provides semantic caching functionality that complements the exact cache
by matching queries based on semantic similarity rather than exact string matching.
"""

import redis
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from redis_memory_manager import get_redis_connection
import time
import logging
import os
from openai import OpenAI
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

try:
    from normalized_semantic_cache import (
        normalize_question_struct,
        canonical_struct_string,
        normalize_and_canonicalize
    )
    NORMALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("normalized_semantic_cache module not available - using RAW-only mode")
    NORMALIZATION_AVAILABLE = False

# Constants
RAW_SIMILARITY_THRESHOLD = 0.80      # Lower threshold for raw text matching
NORM_SIMILARITY_THRESHOLD = 0.90     # Higher threshold for normalized matching
SIMILARITY_THRESHOLD = 0.90          # Backward compatibility (use NORM_SIMILARITY_THRESHOLD)
CACHE_EXPIRATION = 3600              # 1 hour

# Initialize Redis connection
r = get_redis_connection()

def generate_query_embedding(query: str) -> Optional[np.ndarray]:
    """
    Generate embedding vector for a query using OpenAI API.
    
    Args:
        query: The user query to generate embedding for
        
    Returns:
        numpy array of embedding vector, or None if generation fails
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return None
            
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        
        embedding = np.array(response.data[0].embedding)
        logger.debug(f"Generated embedding for query: {query[:50]}... (dimension: {len(embedding)})")
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to generate embedding for query '{query[:50]}...': {e}")
        return None


def add_to_semantic_cache(
    original_question: str,
    complete_question: str,
    raw_embedding: np.ndarray,
    sql: Optional[str] = None,
    query_results: Optional[Dict[str, Any]] = None,
    focused_schema: Optional[str] = None,
    normalized_struct: Optional[Dict[str, Any]] = None,
    normalized_embedding: Optional[np.ndarray] = None
) -> bool:
    """
    Add a question to semantic cache with DUAL-MODE support (RAW + NORMALIZED).
    
    Args:
        original_question: The raw user question
        complete_question: The enriched/complete question after processing
        raw_embedding: The embedding vector for the original question text
        sql: The generated SQL query (optional)
        query_results: The query execution results (optional)
        focused_schema: The focused schema from EnrichAgent (optional) - saves 4.2s on cache hits
        normalized_struct: The normalized structure dict (optional) - for better semantic matching
        normalized_embedding: The embedding of normalized structure (optional)
        
    Returns:
        True if successfully added, False otherwise
    """
    if raw_embedding is None or r is None:
        logger.warning("Cannot add to semantic cache: raw_embedding or Redis connection unavailable")
        return False
    
    try:
        cache_key = f"semantic_cache:question:{original_question}"
        
        cache_data = {
            "original_question": original_question,
            "complete_question": complete_question,
            "raw_embedding": json.dumps(raw_embedding.tolist()),
            "timestamp": str(time.time())
        }
        
        if sql:
            cache_data["sql"] = sql
            
        if query_results:
            cache_data["query_results"] = json.dumps(query_results)
        
        # ⚡ OPTIMIZATION: Store focused schema to skip EnrichAgent on cache hit
        if focused_schema:
            cache_data["focused_schema"] = focused_schema
            logger.debug(f"Stored focused schema in cache ({len(focused_schema)} chars)")
        
        # ⚡ NEW: Store normalized representation for better semantic matching
        if normalized_struct:
            cache_data["normalized_struct"] = json.dumps(normalized_struct)
            cache_data["normalized_struct_str"] = canonical_struct_string(normalized_struct)
            logger.debug(f"Stored normalized structure: {cache_data['normalized_struct_str']}")
        
        if normalized_embedding is not None:
            cache_data["normalized_embedding"] = json.dumps(normalized_embedding.tolist())
            logger.debug(f"Stored normalized embedding (dimension: {len(normalized_embedding)})")
        
        # Store in Redis hash with expiration
        r.hset(cache_key, mapping=cache_data)
        r.expire(cache_key, CACHE_EXPIRATION)
        
        logger.info(f"Added to semantic cache: '{original_question[:50]}...'")
        return True
        
    except Exception as e:
        logger.error(f"Failed to add to semantic cache: {e}")
        return False


def get_from_semantic_cache_dual(
    question: str,
    raw_threshold: float = RAW_SIMILARITY_THRESHOLD,
    norm_threshold: float = NORM_SIMILARITY_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """
    DUAL-MODE semantic cache lookup: Try NORMALIZED first, fall back to RAW.
    
    This provides better cache hit rates by:
    1. Normalizing the question to structured form (higher precision, 0.90 threshold)
    2. Falling back to raw text matching if normalized doesn't hit (0.80 threshold)
    
    Args:
        question: The user query to search for
        raw_threshold: Similarity threshold for RAW mode (default: 0.80)
        norm_threshold: Similarity threshold for NORMALIZED mode (default: 0.90)
        
    Returns:
        Cached data dictionary with additional 'cache_mode' field ('normalized' or 'raw'),
        or None if no match found
        
    Example:
        >>> result = get_from_semantic_cache_dual("total sales last month")
        >>> if result:
        ...     print(f"Cache hit via {result['cache_mode']} mode")
    """
    if r is None:
        logger.warning("Cannot search semantic cache: Redis connection unavailable")
        return None
    
    start_time = time.perf_counter()
    
    try:
        # ============================================================
        # PHASE 1: Try NORMALIZED mode (higher confidence)
        # ============================================================
        if NORMALIZATION_AVAILABLE:
            try:
                logger.info("🔍 PHASE 1: Checking NORMALIZED semantic cache...")
                
                # Normalize the question
                normalized_struct, normalized_str = normalize_and_canonicalize(question)
                logger.debug(f"Normalized structure: {normalized_str}")
                
                # Generate embedding for normalized structure
                normalized_embedding = generate_query_embedding(normalized_str)
                
                if normalized_embedding is not None:
                    # Search cache using normalized embedding
                    cache_keys = r.keys("semantic_cache:question:*")
                    
                    best_norm_score = -1.0
                    best_norm_data = None
                    
                    for key in cache_keys:
                        cached_data = r.hgetall(key)
                        
                        if cached_data and "normalized_embedding" in cached_data:
                            try:
                                cached_norm_emb_list = json.loads(cached_data["normalized_embedding"])
                                cached_norm_emb = np.array(cached_norm_emb_list)
                                
                                similarity = cosine_similarity(
                                    normalized_embedding.reshape(1, -1),
                                    cached_norm_emb.reshape(1, -1)
                                )[0][0]
                                
                                if similarity > best_norm_score:
                                    best_norm_score = similarity
                                    best_norm_data = cached_data
                                    
                            except Exception as e:
                                logger.debug(f"Error processing normalized embedding: {e}")
                                continue
                    
                    if best_norm_score >= norm_threshold and best_norm_data:
                        elapsed = time.perf_counter() - start_time
                        cached_question = best_norm_data.get('original_question', 'N/A')
                        
                        logger.info(f"✅ NORMALIZED CACHE HIT! (score={best_norm_score:.4f} >= {norm_threshold:.4f})")
                        logger.info(f"   User query: '{question}'")
                        logger.info(f"   Cached query: '{cached_question[:60]}...'")
                        logger.info(f"   Time: {elapsed:.4f}s")
                        
                        # Add metadata
                        best_norm_data['cache_mode'] = 'normalized'
                        best_norm_data['similarity'] = best_norm_score
                        best_norm_data['question'] = cached_question
                        
                        return best_norm_data
                    else:
                        logger.info(f"❌ NORMALIZED mode: No match (best={best_norm_score:.4f} < {norm_threshold:.4f})")
                else:
                    logger.warning("Failed to generate normalized embedding")
                    
            except Exception as norm_err:
                logger.warning(f"NORMALIZED mode failed: {norm_err}, falling back to RAW")
        else:
            logger.debug("Normalization not available, skipping to RAW mode")
        
        # ============================================================
        # PHASE 2: Fall back to RAW mode (lower threshold)
        # ============================================================
        logger.info("🔍 PHASE 2: Checking RAW semantic cache...")
        
        # Generate raw embedding
        raw_embedding = generate_query_embedding(question)
        
        if raw_embedding is None:
            logger.warning("Failed to generate raw embedding")
            return None
        
        # Search cache using raw embedding
        cache_keys = r.keys("semantic_cache:question:*")
        
        best_raw_score = -1.0
        best_raw_data = None
        
        for key in cache_keys:
            cached_data = r.hgetall(key)
            
            if cached_data and "raw_embedding" in cached_data:
                try:
                    cached_raw_emb_list = json.loads(cached_data["raw_embedding"])
                    cached_raw_emb = np.array(cached_raw_emb_list)
                    
                    similarity = cosine_similarity(
                        raw_embedding.reshape(1, -1),
                        cached_raw_emb.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_raw_score:
                        best_raw_score = similarity
                        best_raw_data = cached_data
                        
                except Exception as e:
                    logger.debug(f"Error processing raw embedding: {e}")
                    continue
        
        elapsed = time.perf_counter() - start_time
        
        if best_raw_score >= raw_threshold and best_raw_data:
            cached_question = best_raw_data.get('original_question', 'N/A')
            
            logger.info(f"✅ RAW CACHE HIT! (score={best_raw_score:.4f} >= {raw_threshold:.4f})")
            logger.info(f"   User query: '{question}'")
            logger.info(f"   Cached query: '{cached_question[:60]}...'")
            logger.info(f"   Time: {elapsed:.4f}s")
            
            # Add metadata
            best_raw_data['cache_mode'] = 'raw'
            best_raw_data['similarity'] = best_raw_score
            best_raw_data['question'] = cached_question
            
            return best_raw_data
        else:
            logger.info(f"❌ RAW mode: No match (best={best_raw_score:.4f} < {raw_threshold:.4f})")
            logger.info(f"   Total cache check time: {elapsed:.4f}s")
            return None
            
    except Exception as e:
        logger.error(f"Error in dual-mode semantic cache search: {e}")
        return None


def get_from_semantic_cache(
    question_embedding: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD
) -> Optional[Dict[str, Any]]:
    """
    LEGACY: Search semantic cache using RAW embedding only.
    
    Kept for backward compatibility. New code should use get_from_semantic_cache_dual().
    
    Args:
        question_embedding: The embedding vector of the new question
        threshold: Minimum cosine similarity score to consider a match (default: 0.90)
        
    Returns:
        Cached data dictionary if a match is found, None otherwise
    """
    if question_embedding is None or r is None:
        logger.warning("Cannot search semantic cache: embedding or Redis connection unavailable")
        return None
    
    start_time = time.perf_counter()
    
    try:
        # Get all cached question keys
        cache_keys = r.keys("semantic_cache:question:*")
        
        logger.info(f"Semantic cache size: {len(cache_keys)} items")
        
        if not cache_keys:
            elapsed = time.perf_counter() - start_time
            logger.info(f"Semantic cache check: {elapsed:.4f}s (empty cache)")
            return None
        
        best_match_score = -1.0
        best_match_data = None
        
        # Iterate through cached embeddings to find best match
        for key in cache_keys:
            cached_data = r.hgetall(key)
            
            if cached_data and "embedding" in cached_data:
                try:
                    cached_embedding_list = json.loads(cached_data["embedding"])
                    cached_embedding = np.array(cached_embedding_list)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        question_embedding.reshape(1, -1),
                        cached_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_data = cached_data
                        
                except Exception as e:
                    logger.warning(f"Error processing cached embedding for key {key}: {e}")
                    continue
        
        elapsed = time.perf_counter() - start_time
        
        # Log top 3 candidates with similarity scores
        all_candidates = []
        for key in cache_keys:
            cached_data = r.hgetall(key)
            if cached_data and "embedding" in cached_data:
                try:
                    cached_embedding_list = json.loads(cached_data["embedding"])
                    cached_embedding = np.array(cached_embedding_list)
                    similarity = cosine_similarity(
                        question_embedding.reshape(1, -1),
                        cached_embedding.reshape(1, -1)
                    )[0][0]
                    question_text = cached_data.get('original_question', 'N/A')
                    all_candidates.append((similarity, question_text))
                except:
                    pass
        
        # Sort and log top 3
        all_candidates.sort(reverse=True)
        logger.info(f"🔍 Semantic cache search results (threshold: {threshold:.3f}):")
        for i, (score, question_text) in enumerate(all_candidates[:3], 1):
            status = "✅ HIT" if score >= threshold else "❌ MISS"
            logger.info(f"   [{i}] {status} similarity={score:.3f}: '{question_text[:70]}...'")
        
        # Log best candidate regardless of hit/miss
        if best_match_data:
            best_question = best_match_data.get('original_question', 'N/A')
            logger.info(f"Best cache candidate: '{best_question[:60]}...'")
            logger.info(f"Similarity score: {best_match_score:.4f}")
        else:
            logger.info("No candidates found in semantic cache")
        
        # Check if best match meets threshold
        if best_match_score >= threshold:
            logger.info(
                f"SEMANTIC CACHE HIT! (score={best_match_score:.4f} >= threshold={threshold:.4f}) "
                f"Time: {elapsed:.4f}s"
            )
            return best_match_data
        else:
            logger.info(
                f"SEMANTIC CACHE MISS! (score={best_match_score:.4f} < threshold={threshold:.4f}) "
                f"Time: {elapsed:.4f}s"
            )
            return None
            
    except Exception as e:
        logger.error(f"Error searching semantic cache: {e}")
        return None


def _add_to_semantic_cache_if_not_present(
    question: str,
    sql: str,
    raw_embedding: np.ndarray,
    normalized_struct: Optional[Dict[str, Any]] = None,
    normalized_embedding: Optional[np.ndarray] = None,
    query_results: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Check semantic cache first, only add if no similar question exists.
    Prevents duplicate entries for semantically similar questions.
    
    Args:
        question: The user question
        sql: The generated SQL query
        raw_embedding: The raw question embedding vector
        normalized_struct: The normalized question structure (optional)
        normalized_embedding: The normalized embedding vector (optional)
        query_results: The query execution results (optional)
        
    Returns:
        True if added, False if similar entry already exists or on error
    """
    try:
        # Use dual-mode cache check
        cached_data = get_from_semantic_cache_dual(
            question=question,
            raw_threshold=RAW_SIMILARITY_THRESHOLD,
            norm_threshold=NORM_SIMILARITY_THRESHOLD
        )
        
        if not cached_data:
            # No similar question found, safe to add
            return add_to_semantic_cache(
                original_question=question,
                complete_question=question,
                raw_embedding=raw_embedding,
                normalized_struct=normalized_struct,
                normalized_embedding=normalized_embedding,
                sql=sql,
                query_results=query_results
            )
        else:
            logger.info(f"Similar question already cached, skipping duplicate entry")
            return False
            
    except Exception as e:
        logger.error(f"Error checking/adding to semantic cache for question '{question[:50]}...': {e}")
        return False


def _add_to_semantic_cache(
    question: str,
    sql: str,
    embedding: np.ndarray,
    query_results: Optional[Dict[str, Any]] = None,
    focused_schema: Optional[str] = None
) -> bool:
    """
    Add question to semantic cache without checking for duplicates.
    Use this after successful query execution to cache results.
    
    This function now generates BOTH raw and normalized embeddings for dual-mode caching.
    
    Args:
        question: The user question
        sql: The generated SQL query
        embedding: The RAW question embedding vector (already generated)
        query_results: The query execution results (optional)
        focused_schema: The focused schema from EnrichAgent (optional) - saves 4.2s on cache hits
        
    Returns:
        True if successfully added, False otherwise
    """
    try:
        # Generate normalized embedding
        normalized_struct = None
        normalized_embedding = None
        
        if NORMALIZATION_AVAILABLE:
            try:
                # Normalize the question to structured format
                normalized_struct, canonical_string = normalize_and_canonicalize(question)
                
                if canonical_string:
                    # Generate embedding for the canonical string
                    normalized_embedding = generate_query_embedding(canonical_string)
                    logger.debug(f"Generated normalized embedding for: {canonical_string[:100]}")
                else:
                    logger.warning("Failed to generate canonical string for normalization")
            except Exception as e:
                logger.warning(f"Error during normalization (will use raw-only mode): {e}")
        
        # Add to cache with both embeddings
        return add_to_semantic_cache(
            original_question=question,
            complete_question=question,
            raw_embedding=embedding,
            normalized_struct=normalized_struct,
            normalized_embedding=normalized_embedding,
            sql=sql,
            query_results=query_results,
            focused_schema=focused_schema
        )
    except Exception as e:
        logger.error(f"Error adding to semantic cache: {e}")
        return False


def clear_semantic_cache() -> bool:
    """
    Clear all entries from the semantic cache.
    Useful for maintenance or testing purposes.
    
    Returns:
        True if successfully cleared, False otherwise
    """
    if r is None:
        logger.warning("Cannot clear semantic cache: Redis connection unavailable")
        return False
    
    try:
        cache_keys = r.keys("semantic_cache:question:*")
        
        if cache_keys:
            r.delete(*cache_keys)
            logger.info(f"Cleared {len(cache_keys)} entries from semantic cache")
            return True
        else:
            logger.info("Semantic cache already empty")
            return True
            
    except Exception as e:
        logger.error(f"Failed to clear semantic cache: {e}")
        return False


def get_semantic_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the semantic cache.
    
    Returns:
        Dictionary containing cache statistics
    """
    if r is None:
        return {
            "available": False,
            "error": "Redis connection unavailable"
        }
    
    try:
        cache_keys = r.keys("semantic_cache:question:*")
        
        stats = {
            "available": True,
            "total_entries": len(cache_keys),
            "threshold": SIMILARITY_THRESHOLD,
            "expiration_seconds": CACHE_EXPIRATION
        }
        
        # Get sample of questions (first 5)
        sample_questions = []
        for key in cache_keys[:5]:
            cached_data = r.hgetall(key)
            if cached_data and "original_question" in cached_data:
                sample_questions.append(cached_data["original_question"])
        
        stats["sample_questions"] = sample_questions
        
        return stats
        
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }
