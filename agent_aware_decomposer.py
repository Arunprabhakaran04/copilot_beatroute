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
    
    def analyze_and_decompose(self, query: str) -> Dict[str, Any]:
        """
        Analyze query with full knowledge of agent capabilities
        """
        try:
            # Quick check: suggest optimal sequence
            suggested_sequence = get_optimal_agent_sequence(query)
            
            # Comprehensive LLM analysis with agent context
            analysis = self._agent_aware_llm_analysis(query, suggested_sequence)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Agent-aware decomposition failed: {e}")
            return self._create_fallback(query, str(e))
    
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
            
            # Track tokens
            track_llm_call(
                input_prompt=messages,
                output=content,
                agent_type="agent_aware_decomposer",
                operation="decomposition",
                model_name="gpt-4o"
            )
            
            # Parse JSON
            analysis = self._safe_json_parse(content)
            
            if not analysis:
                raise ValueError("Failed to parse LLM response")
            
            # Validate and enhance
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
        
        # Extract task descriptions for backward compatibility
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
            "and then", "followed by", " and visualize", " and send", " and email"
        ])
        
        return {
            "is_multi_step": is_multi,
            "confidence": 0.6,
            "reasoning": f"Fallback: {error_reason}",
            "decomposed_tasks": [query],
            "original_query": query,
            "decomposition_method": "fallback",
            "error": error_reason
        }
