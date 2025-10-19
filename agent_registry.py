"""
Agent Registry - Defines capabilities and limitations of all agents
This helps the decomposer make intelligent task assignments
"""

import re
from typing import Dict, Any, List

AGENT_REGISTRY = {
    "db_query": {
        "name": "Database Query Agent",
        "capabilities": [
            "Execute complex SQL queries with JOINs",
            "Aggregate data (SUM, COUNT, AVG, GROUP BY)",
            "Filter by date ranges and conditions",
            "Sort and rank results (ORDER BY, LIMIT)",
            "Calculate comparisons between time periods",
            "Group data by multiple dimensions (city, product, customer)",
            "Perform window functions and CTEs",
            "Identify top/bottom N items in a single query",
            "Calculate growth/degrowth percentages"
        ],
        "limitations": [
            "Cannot provide business insights or recommendations",
            "No narrative analysis or interpretation",
            "Returns structured data only (no visualizations)",
            "Cannot send emails or schedule meetings",
            "No predictive analytics or forecasting"
        ],
        "best_for": [
            "Retrieving data with aggregations",
            "Combining multiple data sources",
            "Time-series queries",
            "Ranking and filtering data",
            "Complex analytical queries in single step"
        ],
        "output_format": "Structured data (JSON/table)",
        "can_combine": [
            "Data retrieval + aggregation",
            "Filtering + grouping + sorting",
            "Multiple joins in one query"
        ]
    },
    
    "summary": {
        "name": "Summary/Analysis Agent",
        "capabilities": [
            "Analyze data patterns and trends",
            "Identify outliers and anomalies",
            "Generate business insights",
            "Compare performance metrics",
            "Provide recommendations",
            "Explain data significance",
            "Statistical analysis",
            "Root cause analysis"
        ],
        "limitations": [
            "REQUIRES data from previous step (db_query)",
            "Cannot query database directly",
            "Cannot send emails or schedule meetings",
            "No data visualization capabilities"
        ],
        "best_for": [
            "Interpreting query results",
            "Business recommendations",
            "Performance analysis",
            "Trend identification"
        ],
        "output_format": "Narrative text with insights",
        "requires_input_from": ["db_query"]
    },
    
    "visualization": {
        "name": "Visualization Agent",
        "capabilities": [
            "Create bar charts, line charts, pie charts",
            "Generate scatter plots and heatmaps",
            "Multi-series comparisons",
            "Time-series visualizations",
            "Geographical plots",
            "Save charts as HTML files"
        ],
        "limitations": [
            "REQUIRES structured data from previous step",
            "Cannot query database directly",
            "No data transformation or aggregation",
            "Cannot send emails or schedule meetings"
        ],
        "best_for": [
            "Presenting trends visually",
            "Comparative analysis charts",
            "Time-series visualization"
        ],
        "output_format": "Interactive HTML charts",
        "requires_input_from": ["db_query"]
    },
    
    "email": {
        "name": "Email Agent",
        "capabilities": [
            "Send emails to specified addresses",
            "Format email content",
            "Include data from previous steps",
            "Add subject lines"
        ],
        "limitations": [
            "Cannot query database",
            "No data analysis",
            "Requires email addresses in query"
        ],
        "best_for": [
            "Sending reports",
            "Notifying stakeholders",
            "Sharing analysis results"
        ],
        "output_format": "Email confirmation",
        "can_use_data_from": ["db_query", "summary", "visualization"]
    },
    
    "meeting": {
        "name": "Meeting Scheduler Agent",
        "capabilities": [
            "Schedule meetings with users",
            "Book appointments",
            "Set meeting times and dates"
        ],
        "limitations": [
            "Cannot query database",
            "No data analysis",
            "Requires user IDs"
        ],
        "best_for": [
            "Scheduling follow-ups",
            "Booking demos",
            "Arranging calls"
        ],
        "output_format": "Meeting confirmation"
    },
    
    "campaign": {
        "name": "Campaign Agent",
        "capabilities": [
            "Retrieve campaign information",
            "Query campaign metrics",
            "Campaign data enrichment"
        ],
        "limitations": [
            "Limited to campaign-related queries",
            "May redirect to db_query for complex queries"
        ],
        "best_for": [
            "Campaign-specific queries",
            "Marketing campaign data"
        ],
        "output_format": "Campaign data or redirect to db_query"
    }
}


# Task combination rules
COMBINATION_RULES = {
    "data_retrieval_and_grouping": {
        "pattern": r"(get|show|retrieve).*and.*(group|calculate|aggregate)",
        "recommendation": "Combine into single db_query with GROUP BY",
        "agents": ["db_query"]
    },
    
    "ranking_and_filtering": {
        "pattern": r"(top|bottom|best|worst).*\d+.*(and|with).*(filter|where)",
        "recommendation": "Combine into single db_query with ORDER BY and LIMIT",
        "agents": ["db_query"]
    },
    
    "data_and_visualization": {
        "pattern": r"(get|show).*and.*(visualize|chart|graph)",
        "recommendation": "Split into: 1) db_query for data, 2) visualization",
        "agents": ["db_query", "visualization"]
    },
    
    "data_and_summary": {
        "pattern": r"(get|show).*and.*(analyze|summarize|insights)",
        "recommendation": "Split into: 1) db_query for data, 2) summary for analysis",
        "agents": ["db_query", "summary"]
    },
    
    "data_and_email": {
        "pattern": r"(get|show).*and.*(send|email)",
        "recommendation": "Split into: 1) db_query for data, 2) email to send",
        "agents": ["db_query", "email"]
    }
}


def get_agent_capabilities(agent_type: str) -> Dict[str, Any]:
    """Get capabilities for a specific agent"""
    return AGENT_REGISTRY.get(agent_type, {})


def can_agent_handle(agent_type: str, task_description: str) -> bool:
    """Check if an agent can handle a specific task"""
    agent_info = AGENT_REGISTRY.get(agent_type, {})
    capabilities = agent_info.get("capabilities", [])
    
    # Simple keyword matching (can be enhanced with embeddings)
    task_lower = task_description.lower()
    
    for capability in capabilities:
        capability_keywords = capability.lower().split()
        if any(keyword in task_lower for keyword in capability_keywords[:3]):
            return True
    
    return False


def get_optimal_agent_sequence(query: str) -> List[str]:
    """Suggest optimal agent sequence for a query"""
    query_lower = query.lower()
    sequence = []
    
    # Check combination rules
    for rule_name, rule_info in COMBINATION_RULES.items():
        if re.search(rule_info["pattern"], query_lower):
            return rule_info["agents"]
    
    # Fallback: default sequence based on query characteristics
    needs_data = any(word in query_lower for word in ["show", "get", "find", "list", "display"])
    needs_analysis = any(word in query_lower for word in ["analyze", "summarize", "insights", "explain"])
    needs_viz = any(word in query_lower for word in ["visualize", "chart", "graph", "plot"])
    needs_email = any(word in query_lower for word in ["send", "email", "notify"])
    
    if needs_data:
        sequence.append("db_query")
    if needs_analysis:
        sequence.append("summary")
    if needs_viz:
        sequence.append("visualization")
    if needs_email:
        sequence.append("email")
    
    return sequence if sequence else ["db_query"]


def format_agent_context_for_llm() -> str:
    """Format agent registry as context for LLM prompts"""
    context = "# AVAILABLE AGENTS AND THEIR CAPABILITIES\n\n"
    
    for agent_type, info in AGENT_REGISTRY.items():
        context += f"## {info['name']} ({agent_type})\n"
        context += f"**Output Format:** {info['output_format']}\n\n"
        
        context += "**Capabilities:**\n"
        for cap in info['capabilities']:
            context += f"  ✓ {cap}\n"
        
        context += "\n**Limitations:**\n"
        for lim in info['limitations']:
            context += f"  ✗ {lim}\n"
        
        context += f"\n**Best For:** {', '.join(info['best_for'])}\n"
        
        if 'can_combine' in info:
            context += f"\n**Can Combine:** {', '.join(info['can_combine'])}\n"
        
        if 'requires_input_from' in info:
            context += f"\n**Requires Input From:** {', '.join(info['requires_input_from'])}\n"
        
        context += "\n" + "="*70 + "\n\n"
    
    return context
