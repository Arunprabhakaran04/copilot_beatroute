"""
Normalized Semantic Cache - Query Normalization Module

This module provides LLM-based query normalization to convert natural language
questions into structured representations for more consistent semantic caching.

Example:
    "show me total sales last month" → {"metric": "sales", "timeframe": "last month", ...}
    "what is the total sales for last month" → {"metric": "sales", "timeframe": "last month", ...}
    
Both queries produce the same normalized structure, leading to better cache hits.
"""

import json
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = None

def _get_openai_client() -> OpenAI:
    """Lazy initialization of OpenAI client."""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        client = OpenAI(api_key=api_key)
    return client


# ============================================================
#                 LLM NORMALIZATION PROMPT
# ============================================================

NORMALIZER_SYSTEM_PROMPT = """
You are a data understanding assistant for a sales analytics system.

Given a natural language question about orders, sales, visits, teams, users, customers, SKUs, brands, campaigns, etc.,
you MUST convert it into a structured JSON with exactly these keys:

- "metric": what is being measured 
    Examples: "order value", "sales", "sales value", "number of visits", "unique customers", "SKU count", 
              "campaign responses", "return rate", "distributor sales", "customer count", "visit count"
              
- "timeframe": the time window mentioned 
    Examples: "last month", "this month", "last 3 months", "last quarter", "current month", "last week", 
              "previous month", "last 6 months", "year to date", "last 12 months"
    If not mentioned, use null.
    
- "entity": the primary entity the question is about 
    Examples: "overall", "customer", "team", "sales rep", "user", "sku", "brand", "route", "distributor",
              "campaign", "outlet", "order", "invoice", "return"
              
- "breakdown": what dimension to group by or analyze separately, if any 
    Examples: "month", "user", "sales rep", "sku", "customer", "brand", "team", "route", "distributor",
              "week", "day", "quarter", "category"
    If not applicable, use null.
    
- "filters": free-text description of any additional filters or conditions
    Examples: "only my team", "only outlets in Delhi", "only campaign XYZ", "top 10", "top 3", 
              "excluding returns", "only active customers", "SKU similar to ABC"
    If none, use null.
    
- "aggregation": type of aggregation if specified
    Examples: "sum", "count", "average", "max", "min", "top", "bottom", "trend", "comparison"
    If not specified, use null.

Rules:
- Always return a single JSON OBJECT, not an array.
- Use EXACTLY these keys: ["metric", "timeframe", "entity", "breakdown", "filters", "aggregation"].
- Use double quotes for all keys and string values.
- If something is unknown or not specified, set it to null.
- Normalize synonyms: "previous month" → "last month", "past 3 months" → "last 3 months"
- Extract the core meaning, ignore filler words like "show me", "what is", "can you tell me"
- Do NOT add any commentary, explanation, or additional keys.
- The output MUST be valid JSON that can be parsed by json.loads().

Examples:
Input: "show me total sales last month"
Output: {"metric": "sales", "timeframe": "last month", "entity": "overall", "breakdown": null, "filters": null, "aggregation": "sum"}

Input: "what is the order value for the last 3 months by customer"
Output: {"metric": "order value", "timeframe": "last 3 months", "entity": "customer", "breakdown": "customer", "filters": null, "aggregation": "sum"}

Input: "top 5 SKUs by sales this month"
Output: {"metric": "sales", "timeframe": "this month", "entity": "sku", "breakdown": "sku", "filters": "top 5", "aggregation": "top"}

Input: "customer visits in Mumbai last week"
Output: {"metric": "visit count", "timeframe": "last week", "entity": "customer", "breakdown": null, "filters": "Mumbai", "aggregation": "count"}
"""


def normalize_question_struct(
    question: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Convert natural language question into structured JSON representation using LLM.
    
    Args:
        question: The natural language query to normalize
        model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        temperature: LLM temperature (0 for deterministic output)
        
    Returns:
        Dictionary with keys: metric, timeframe, entity, breakdown, filters, aggregation
        
    Example:
        >>> normalize_question_struct("show me total sales last month")
        {
            "metric": "sales",
            "timeframe": "last month",
            "entity": "overall",
            "breakdown": null,
            "filters": null,
            "aggregation": "sum"
        }
    """
    try:
        client = _get_openai_client()
        
        logger.debug(f"Normalizing question: {question[:80]}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": NORMALIZER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=temperature,
        )
        
        content = response.choices[0].message.content.strip()
        logger.debug(f"Normalization response: {content}")
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse normalization JSON: {content[:200]}")
            # Defensive fallback: keep raw text in filters
            data = {
                "metric": None,
                "timeframe": None,
                "entity": None,
                "breakdown": None,
                "filters": content,  # Store unparsed response
                "aggregation": None,
            }
        
        # Ensure all required keys exist
        for key in ["metric", "timeframe", "entity", "breakdown", "filters", "aggregation"]:
            if key not in data:
                data[key] = None
        
        logger.info(f"✅ Normalized question structure: {json.dumps(data, ensure_ascii=False)}")
        return data
        
    except Exception as e:
        logger.error(f"Error normalizing question '{question[:50]}...': {e}")
        # Return defensive fallback
        return {
            "metric": None,
            "timeframe": None,
            "entity": None,
            "breakdown": None,
            "filters": question,  # Store original question
            "aggregation": None,
        }


def canonical_struct_string(struct: Dict[str, Any]) -> str:
    """
    Convert structured dict to canonical JSON string for stable embedding.
    
    Keys are sorted alphabetically and output is minified for consistency.
    
    Args:
        struct: Normalized question structure
        
    Returns:
        Canonical JSON string representation
        
    Example:
        >>> struct = {"metric": "sales", "timeframe": "last month", "entity": "overall"}
        >>> canonical_struct_string(struct)
        '{"entity":"overall","metric":"sales","timeframe":"last month"}'
    """
    return json.dumps(struct, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def normalize_and_canonicalize(question: str) -> tuple[Dict[str, Any], str]:
    """
    Convenience function to normalize question and get canonical string in one call.
    
    Args:
        question: Natural language query
        
    Returns:
        Tuple of (normalized_struct, canonical_string)
        
    Example:
        >>> struct, canonical = normalize_and_canonicalize("total sales last month")
        >>> print(canonical)
        '{"aggregation":"sum","breakdown":null,"entity":"overall","filters":null,"metric":"sales","timeframe":"last month"}'
    """
    struct = normalize_question_struct(question)
    canonical = canonical_struct_string(struct)
    return struct, canonical


# ============================================================
#                     SIMILARITY HELPERS
# ============================================================

def compare_normalized_structs(struct1: Dict[str, Any], struct2: Dict[str, Any]) -> float:
    """
    Simple structural similarity between two normalized questions.
    Returns 1.0 if all non-null fields match, 0.0 if no overlap.
    
    This is a fast heuristic check before embedding comparison.
    
    Args:
        struct1: First normalized structure
        struct2: Second normalized structure
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    keys = ["metric", "timeframe", "entity", "breakdown", "aggregation"]
    
    matches = 0
    comparisons = 0
    
    for key in keys:
        val1 = struct1.get(key)
        val2 = struct2.get(key)
        
        # Skip if both are null
        if val1 is None and val2 is None:
            continue
            
        comparisons += 1
        
        # Normalize strings for comparison
        if isinstance(val1, str) and isinstance(val2, str):
            if val1.lower().strip() == val2.lower().strip():
                matches += 1
        elif val1 == val2:
            matches += 1
    
    if comparisons == 0:
        return 0.0
    
    return matches / comparisons


# ============================================================
#                     TESTING / DEBUG
# ============================================================

if __name__ == "__main__":
    # Test normalization with example queries
    test_queries = [
        "show me total sales last month",
        "what is the total sales for last month",
        "order value previous month",
        "top 5 customers by sales this quarter",
        "customer visits in Delhi last week",
        "compare sales of this month with last 3 months",
        "how many unique SKUs sold last month",
    ]
    
    print("Testing Query Normalization")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        struct, canonical = normalize_and_canonicalize(query)
        print(f"Canonical: {canonical}")
        print(f"Struct: {json.dumps(struct, indent=2)}")
