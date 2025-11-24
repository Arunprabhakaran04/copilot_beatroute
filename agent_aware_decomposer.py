"""
Agent-Aware Query Decomposer
This decomposer understands agent capabilities and creates optimal task sequences
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from token_tracker import track_llm_call
from agent_registry import (
    AGENT_REGISTRY, 
    format_agent_context_for_llm,
    get_optimal_agent_sequence
)

logger = logging.getLogger(__name__)


class AgentAwareDecomposer:
    """
    Intelligent decomposer that understands what each agent can do
    and creates optimal, non-redundant task sequences
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.agent_context = format_agent_context_for_llm()
    
    def analyze_and_decompose(self, query: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze query with full knowledge of agent capabilities
        Strategy: Fast heuristics first → LLM analysis if no match
        """
        try:
            # STEP 1: Fast heuristic optimization with conversation history
            logger.debug(f" Query: {query}")
            logger.debug(f" History entries: {len(conversation_history) if conversation_history else 0}")
            
            if conversation_history:
                heuristic_result = self._optimize_with_history(query, conversation_history)
                if heuristic_result:
                    logger.info(f" Heuristic match: {heuristic_result.get('reasoning')}")
                    logger.info(f" Method: {heuristic_result.get('decomposition_method')} (fast, no LLM)")
                    return heuristic_result
                else:
                    logger.debug(f" No heuristic pattern matched, falling back to LLM analysis")
            else:
                logger.debug(f" No conversation history available, skipping heuristics")
            
            # STEP 2: LLM-based analysis (if heuristics didn't match)
            logger.debug(f" Using LLM for query decomposition")
            suggested_sequence = get_optimal_agent_sequence(query)
            analysis = self._agent_aware_llm_analysis(query, suggested_sequence)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Agent-aware decomposition failed: {e}")
            return self._create_fallback(query, str(e))
    
    def _optimize_with_history(self, query: str, history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Fast heuristic patterns for common followup queries
        Returns optimized decomposition if pattern matched, None otherwise (triggers LLM)
        
        Common patterns:
        1. "send that/this/the [data/list/results] to [email]" → Use cached result + email
        2. "show me that data/result" → Return cached result
        3. "visualize that/this" → Use cached data + visualization
        4. "summarize that/this" → Use cached data + summary
        """
        if not history:
            logger.debug(" Heuristic check: No history provided")
            return None
            
        lowered = query.lower()
        logger.debug(f" Checking heuristic patterns for: '{query}'")
        logger.debug(f" History has {len(history)} entries")
        
        # PATTERN 1: Email followup (send/email + cached data)

        send_patterns = [
            "send that", "send this", "send it", "send the",
            "email that", "email this", "email it", "email the",
            "forward that", "forward this", "forward it",
            "share that", "share this", "share it"
        ]
        
        pattern_matched = any(pattern in lowered for pattern in send_patterns)
        logger.debug(f"Email pattern match: {pattern_matched}")
        
        if pattern_matched:
            email_match = re.search(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', query)
            logger.debug(f" Email address found: {email_match.group(0) if email_match else 'None'}")
            
            if email_match:
                logger.debug(f" Searching {len(history)} history entries for cached data...")
                for i, entry in enumerate(reversed(history)):
                    agent_type = entry.get("agent_type")
                    result = entry.get("result", {})
                    logger.debug(f"  Entry {i+1}: agent_type={agent_type}, result_keys={list(result.keys()) if result else 'None'}")
                    
                    # Check if result has data (handle nested structure)
                    actual_result = result.get("final_result", result)  # Handle nested result structure
                    has_data = (
                        actual_result.get("query_data") is not None or
                        actual_result.get("data") is not None or
                        actual_result.get("rows") is not None or
                        actual_result.get("meeting_date") is not None
                    )
                    logger.debug(f"    → has_data={has_data}, actual_result_keys={list(actual_result.keys()) if actual_result else 'None'}")
                    
                    if agent_type in ["db_query", "meeting", "summary", "visualization"] and has_data:
                        logger.info(f" Heuristic: Email followup detected (previous {agent_type})")
                        return {
                            "is_multi_step": False,
                            "confidence": 0.95,
                            "reasoning": f"{agent_type} already completed in previous query. Only email agent needed.",
                            "task_count": 1,
                            "tasks": [{
                                "step": 1,
                                "agent": "email",
                                "description": f"Send email to {email_match.group(0)} with results from previous {agent_type} task",
                                "reasoning": f"{agent_type} agent already completed. Using cached results."
                            }],
                            "decomposed_tasks": [f"Send email with {agent_type} results to {email_match.group(0)}"],
                            "original_query": query,
                            "decomposition_method": "heuristic_email_followup",
                            "optimization_notes": f"Detected email followup. Using cached {agent_type} result."
                        }
                logger.debug(f" No cached data found in history for email followup")
        
        # PATTERN 2: Show/display followup (return cached data)
        show_patterns = [
            "show me that", "show that", "show it", "show the",
            "display that", "display it", "display the",
            "give me that", "give me the",
            "what was that", "what were the"
        ]
        
        if any(pattern in lowered for pattern in show_patterns):
            # Find most recent db_query or summary result
            for entry in reversed(history):
                agent_type = entry.get("agent_type")
                result = entry.get("result", {})
                
                if agent_type in ["db_query", "summary"] and result:
                    logger.info(f"🎯 Heuristic: Display followup detected (previous {agent_type})")
                    return {
                        "is_multi_step": False,
                        "confidence": 0.90,
                        "reasoning": f"Returning cached {agent_type} result from previous query.",
                        "task_count": 1,
                        "tasks": [{
                            "step": 1,
                            "agent": agent_type,
                            "description": f"Return cached {agent_type} result",
                            "reasoning": "User wants to see previous result again."
                        }],
                        "decomposed_tasks": [f"Display previous {agent_type} result"],
                        "original_query": query,
                        "decomposition_method": "heuristic_display_followup",
                        "optimization_notes": f"User requesting previous data. Returning cached {agent_type} result."
                    }
        
        # PATTERN 3: Visualize followup (cached data + visualization)
        # CRITICAL: Only match if query is TRULY a followup ("that", "this", "it")
        # DO NOT match if query contains data retrieval keywords ("get", "show", "top", "list", "sales", "trend")
        viz_followup_patterns = [
            "visualize that", "visualize this", "visualize it",
            "chart that", "chart this", "graph that", "graph this",
            "plot that", "plot this"
        ]
        
        # Keywords indicating NEW data retrieval is needed (NOT a simple followup)
        data_retrieval_keywords = [
            "give me", "show me", "get me", "find", "list",
            "top", "best", "highest", "lowest", "most",
            "sales", "trend", "total", "count", "sum",
            "last month", "this month", "last year", "this year",
            "customers", "products", "skus", "orders", "users"
        ]
        
        has_viz_followup = any(pattern in lowered for pattern in viz_followup_patterns)
        has_data_retrieval = any(keyword in lowered for keyword in data_retrieval_keywords)
        
        logger.debug(f" Viz followup pattern: {has_viz_followup}, Data retrieval: {has_data_retrieval}")
        
        # Only treat as followup if it matches followup pattern AND does NOT have data retrieval keywords
        if has_viz_followup and not has_data_retrieval:
            # Find most recent db_query result
            for entry in reversed(history):
                if entry.get("agent_type") == "db_query" and entry.get("result"):
                    logger.info(f" Heuristic: Visualization followup detected (no new data needed)")
                    return {
                        "is_multi_step": False,
                        "confidence": 0.90,
                        "reasoning": "Using cached db_query result for visualization.",
                        "task_count": 1,
                        "tasks": [{
                            "step": 1,
                            "agent": "visualization",
                            "description": "Create visualization from previous query result",
                            "reasoning": "db_query already completed. Using cached data."
                        }],
                        "decomposed_tasks": ["Create visualization from cached db_query result"],
                        "original_query": query,
                        "decomposition_method": "heuristic_viz_followup",
                        "optimization_notes": "Using cached data for visualization."
                    }
        
        # If query has "and visualize" or "and also visualize" with data retrieval keywords,
        # it's a multi-step query requiring db_query FIRST
        if ("and visualize" in lowered or "also visualize" in lowered) and has_data_retrieval:
            logger.info(f" Heuristic: Multi-step query detected (data + visualization)")
            logger.debug(f"  → Query requires NEW data retrieval + visualization")
            # Return None to trigger LLM decomposition for proper multi-step handling
            return None
        
        # PATTERN 4: Summarize followup (cached data + summary)
        summary_patterns = [
            "summarize that", "summarize this", "summarize it",
            "explain that", "explain this", "what does that mean",
            "tell me about that",
            "prepare a summary", "prepare summary", "create a summary", "create summary",
            "make a summary", "make summary", "generate a summary", "generate summary"
        ]
        
        if any(pattern in lowered for pattern in summary_patterns):
            # Find most recent db_query result
            for entry in reversed(history):
                if entry.get("agent_type") == "db_query" and entry.get("result"):
                    logger.info(f" Heuristic: Summary followup detected")
                    return {
                        "is_multi_step": False,
                        "confidence": 0.85,
                        "reasoning": "Using cached db_query result for summary.",
                        "task_count": 1,
                        "tasks": [{
                            "step": 1,
                            "agent": "summary",
                            "description": "Summarize previous query result",
                            "reasoning": "db_query already completed. Using cached data."
                        }],
                        "decomposed_tasks": ["Summarize cached db_query result"],
                        "original_query": query,
                        "decomposition_method": "heuristic_summary_followup",
                        "optimization_notes": "Using cached data for summary."
                    }
        
        # No heuristic pattern matched - return None to trigger LLM
        logger.debug(f"No heuristic pattern matched for query: {query[:50]}...")
        return None
    
    def _agent_aware_llm_analysis(self, query: str, suggested_sequence: List[str]) -> Dict[str, Any]:
        """
        LLM analysis with full agent capability context
        """
        
        prompt = ChatPromptTemplate.from_template("""
You are an expert query decomposer with COMPLETE KNOWLEDGE of all available agents.

{agent_context}

# CRITICAL DECOMPOSITION RULES

## Rule 1: MAXIMIZE AGENT CAPABILITIES (with exceptions)
✅ **DO:** Use each agent to its full potential WHEN POSSIBLE
   - If db_query can do aggregation + grouping + sorting in ONE query, DON'T split it
   - If visualization needs data, get ALL data needed in ONE db_query step

✅ **DO:** Combine operations that belong to the same agent WHEN PRACTICAL
   - "Get sales by city and calculate totals" = ONE db_query task (use GROUP BY)
   - "Show customers in Mumbai" = ONE db_query task (use WHERE)

❌ **DON'T:** Split operations that one agent can handle IN A SIMPLE WAY
   - DON'T: Task 1: Get data, Task 2: Group data ← Both are db_query, combine!
   - DON'T: Task 1: Filter customers, Task 2: Sort customers ← One SQL query!

⚠️ **EXCEPTION - DO SPLIT when query requires dependent filtering:**
   - "Top 3 SKUs of September AND their 12-month trend" = TWO db_query tasks:
     * Task 1: Get top 3 SKU names for September (simple SELECT with LIMIT 3)
     * Task 2: Get 12-month trend for those 3 SKUs using results from Task 1
   - Reason: Step 2 needs to filter by specific SKU names from Step 1
   
⚠️ **EXCEPTION - DO SPLIT when SQL becomes too complex:**
   - Avoid window functions (ROW_NUMBER, RANK) in Cube.js - they often fail
   - Avoid nested CTEs with complex JOINs - split into simpler steps
   - If SQL has >3 CROSS JOINs or complex subqueries, consider splitting

## Rule 2: RESPECT AGENT BOUNDARIES
✅ **DO:** Split when crossing agent boundaries
   - db_query → summary (different agents, must split)
   - db_query → visualization (different agents, must split)
   - db_query → email (different agents, must split)

✅ **DO:** Ensure data dependencies are satisfied
   - summary agent REQUIRES data from db_query first
   - visualization agent REQUIRES data from db_query first
   - If Step 2 needs results from Step 1, they MUST be separate tasks

⚠️ **CRITICAL PATTERN:** "get/show/give [data] and visualize"
   - This ALWAYS requires TWO steps:
     * Step 1: db_query to get the data
     * Step 2: visualization using results from Step 1
   - Example: "show sales trend and visualize" = db_query → visualization
   - Example: "top 3 SKUs and also visualize" = db_query → visualization
   - NEVER route directly to visualization without db_query first!

## Rule 3: AVOID REDUNDANT STEPS
❌ **NEVER** create redundant database queries:
   - If Step 1 gets "sales data", Step 2 should NOT query database again
   - If Step 1 gets "top customers", Step 2 should NOT re-query for same data
   - Step 2 should ALWAYS use {{{{RESULT_FROM_STEP_1}}}} if it needs Step 1's data

# USER QUERY TO DECOMPOSE

QUERY: "{query}"

SUGGESTED AGENT SEQUENCE (from pattern analysis): {suggested_sequence}

# YOUR TASK

Analyze this query and determine:
1. Is this single-step or multi-step?
2. If multi-step, what is the OPTIMAL task sequence?

**OPTIMIZATION PRINCIPLES:**
- Each task should be assigned to the agent that can handle it MOST COMPLETELY
- Minimize task count by combining operations within agent capabilities
- Avoid redundant data retrieval
- Ensure proper data flow between agents

# EXAMPLES

## BAD Decomposition ❌ (Too few steps for complex query)
Query: "Show sales trend for the last 12 months for the top 3 SKUs of September"
Tasks:
  1. Get top 3 SKUs of September and their 12-month trend in one query (db_query) 
     ← BAD! This requires window functions and complex CTEs that often fail

## GOOD Decomposition ✅ (Proper separation)
Query: "Show sales trend for the last 12 months for the top 3 SKUs of September"
Tasks:
  1. Get top 3 SKU names by sales for September 2025 (db_query)
     - Simple query: SELECT Sku.name, SUM(sales) ... WHERE month = '2025-09-01' GROUP BY Sku.name ORDER BY SUM(sales) DESC LIMIT 3
  2. Get monthly sales trend for these 3 SKUs over last 12 months (db_query)
     - Uses SKU names from step 1: SELECT month, Sku.name, SUM(sales) ... WHERE Sku.name IN ({{{{RESULT_FROM_STEP_1}}}}) AND date >= 12_months_ago GROUP BY month, Sku.name
  3. Analyze the sales trend and provide insights (summary)
     - Uses {{{{RESULT_FROM_STEP_2}}}} for analysis

Reasoning: Splitting into 3 steps avoids complex window functions and CTEs. Each step is simple and reliable.

## Another BAD Example ❌
Query: "Show Q1 2025 sales by city, identify best/worst regions, recommend strategies"
Tasks:
  1. Get Q1 2025 sales data (db_query)
  2. Group by city and calculate totals (db_query) ← REDUNDANT! Should be in Task 1
  3. Sort by performance (db_query) ← REDUNDANT! Should be in Task 1
  4. Analyze and recommend (summary)

## GOOD Decomposition ✅
Query: "Show Q1 2025 sales by city, identify best/worst regions, recommend strategies"
Tasks:
  1. Get Q1 2025 sales by city with totals, sorted by performance (db_query)
     - Single SQL query: SELECT city, SUM(sales) FROM ... WHERE date >= '2025-01-01' ... GROUP BY city ORDER BY SUM(sales) DESC
  2. Analyze performance patterns and recommend resource allocation strategies (summary)
     - Uses {{{{RESULT_FROM_STEP_1}}}} for analysis

Reasoning: db_query agent can do filtering, grouping, aggregation, and sorting in ONE SIMPLE query.
Only split when moving to summary agent for business insights.

## Another Example - Meeting + Email

Query: "Schedule a meeting with user 7 on 30th October about cost per click and send email to arun@example.com"
Tasks:
  1. Schedule meeting with user 7 on 30th October (meeting)
  2. Send email to arun@example.com with meeting confirmation details from {{{{RESULT_FROM_STEP_1}}}} (email)

Reasoning: meeting agent schedules first, then email agent sends confirmation using meeting details from step 1.

## Another Example - Data + Visualization

Query: "Get top 5 products and create a bar chart"
Tasks:
  1. Get top 5 products by sales with detailed metrics (db_query)
  2. Create bar chart visualization from {{{{RESULT_FROM_STEP_1}}}} (visualization)

Reasoning: db_query gets ALL needed data in one query. Visualization creates chart from that data.

# OUTPUT FORMAT

Respond with this EXACT JSON structure:
{{
  "is_multi_step": true/false,
  "confidence": 0.85,
  "reasoning": "clear explanation of why this decomposition is optimal",
  "task_count": 2,
  "tasks": [
    {{
      "step": 1,
      "agent": "db_query",
      "description": "detailed task description",
      "reasoning": "why this agent and scope"
    }},
    {{
      "step": 2,
      "agent": "summary",
      "description": "task description using {{{{RESULT_FROM_STEP_1}}}}",
      "reasoning": "why this needs to be separate from step 1",
      "depends_on": [1]
    }}
  ],
  "optimization_notes": "how tasks were optimized to avoid redundancy"
}}
""")
        
        try:
            messages = prompt.format_messages(
                query=query,
                agent_context=self.agent_context,
                suggested_sequence=suggested_sequence
            )
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            track_llm_call(
                input_prompt=messages,
                output=content,
                agent_type="agent_aware_decomposer",
                operation="decomposition",
                model_name="gpt-4.1-mini"
            )
            
            analysis = self._safe_json_parse(content)
            
            if not analysis:
                raise ValueError("Failed to parse LLM response")
            
            analysis = self._validate_and_enhance_analysis(query, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Agent-aware LLM analysis failed: {e}")
            raise
    
    def _validate_and_enhance_analysis(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the analysis and add enhancements
        """
        # Ensure required fields
        analysis.setdefault("is_multi_step", False)
        analysis.setdefault("confidence", 0.75)
        analysis.setdefault("task_count", 1)
        analysis.setdefault("reasoning", "Analysis completed")
        
        if "tasks" in analysis:
            task_descriptions = []
            for task in analysis["tasks"]:
                if isinstance(task, dict):
                    task_descriptions.append(task.get("description", query))
                else:
                    task_descriptions.append(str(task))
            
            analysis["decomposed_tasks"] = task_descriptions
        else:
            analysis["decomposed_tasks"] = [query]
        
        # Add metadata
        analysis["original_query"] = query
        analysis["decomposition_method"] = "agent_aware"
        
        # Validation checks
        if analysis["is_multi_step"]:
            if len(analysis.get("decomposed_tasks", [])) <= 1:
                logger.warning("Multi-step flagged but only 1 task - correcting")
                analysis["is_multi_step"] = False
                analysis["confidence"] *= 0.8
        
        return analysis
    
    def _safe_json_parse(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Robust JSON parsing with multiple strategies
        """
        # Remove markdown code blocks
        content = re.sub(r'```(?:json)?\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        try:
            # Try direct parsing
            return json.loads(content.strip())
        except:
            pass
        
        try:
            # Extract JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
        
        return None
    
    def _create_fallback(self, query: str, error_reason: str) -> Dict[str, Any]:
        """
        Safe fallback when analysis fails
        """
        # Simple heuristic
        query_lower = query.lower()
        
        is_multi = any(keyword in query_lower for keyword in [
            "and then", "followed by", " and visualize", " and also visualize",
            " and send", " and email", " and create", " and show"
        ])
        
        # Additional check: if query mentions visualization + data retrieval, it's multi-step
        has_viz = any(kw in query_lower for kw in ["visualize", "chart", "graph", "plot"])
        has_data = any(kw in query_lower for kw in ["give me", "show me", "get", "top", "sales", "trend"])
        
        if has_viz and has_data:
            is_multi = True
        
        return {
            "is_multi_step": is_multi,
            "confidence": 0.6,
            "reasoning": f"Fallback: {error_reason}",
            "decomposed_tasks": [query],
            "original_query": query,
            "decomposition_method": "fallback",
            "error": error_reason
        }
