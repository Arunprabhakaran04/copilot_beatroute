import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TokenUsage:
    """Represents token usage for a single LLM call"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_type: str = ""
    operation: str = ""

@dataclass 
class AgentCostSummary:
    """Summary of token usage for an agent"""
    agent_type: str
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_estimate: float
    operations: List[str]

class TokenTracker:
    """Global token usage tracker for all agents"""
    
    def __init__(self):
        self.usage_history: List[TokenUsage] = []
        self.session_start = datetime.now()
        
    def add_usage(self, input_tokens: int, output_tokens: int, 
                  agent_type: str = "", operation: str = "", cost_estimate: float = 0.0):
        """Add token usage record"""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_estimate=cost_estimate,
            agent_type=agent_type,
            operation=operation
        )
        self.usage_history.append(usage)
        
        logger.info(f"Token usage recorded - {agent_type}:{operation} - "
                   f"Input: {input_tokens}, Output: {output_tokens}, "
                   f"Total: {usage.total_tokens}, Cost: ${cost_estimate:.4f}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of token usage for current session"""
        if not self.usage_history:
            return {
                "total_tokens": 0,
                "total_cost": 0.0,
                "total_calls": 0,
                "agents": {},
                "session_duration": 0
            }
        
        total_input = sum(u.input_tokens for u in self.usage_history)
        total_output = sum(u.output_tokens for u in self.usage_history)
        total_tokens = sum(u.total_tokens for u in self.usage_history)
        total_cost = sum(u.cost_estimate for u in self.usage_history)
        
        # Group by agent
        agent_stats = {}
        for usage in self.usage_history:
            agent_type = usage.agent_type or "unknown"
            if agent_type not in agent_stats:
                agent_stats[agent_type] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "operations": set()
                }
            
            stats = agent_stats[agent_type]
            stats["calls"] += 1
            stats["input_tokens"] += usage.input_tokens
            stats["output_tokens"] += usage.output_tokens
            stats["total_tokens"] += usage.total_tokens
            stats["cost"] += usage.cost_estimate
            stats["operations"].add(usage.operation)
        
        # Convert sets to lists for serialization
        for agent_type in agent_stats:
            agent_stats[agent_type]["operations"] = list(agent_stats[agent_type]["operations"])
        
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "total_calls": len(self.usage_history),
            "agents": agent_stats,
            "session_duration": session_duration,
            "session_start": self.session_start.isoformat()
        }
    
    def clear_session(self):
        """Clear current session data"""
        self.usage_history = []
        self.session_start = datetime.now()
        logger.info("Token tracking session cleared")

# Global token tracker instance
_token_tracker = None

def get_token_tracker() -> TokenTracker:
    """Get singleton token tracker instance"""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker

def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count based on text length.
    More accurate counting would require the specific tokenizer.
    """
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def get_cost(input_prompt: Any, output: str, model_name: str = "gpt-4o") -> Dict[str, Any]:
    """
    Estimate cost of LLM call based on input and output.
    
    Args:
        input_prompt: Input prompt (can be string or list of messages)
        output: Output text from LLM
        model_name: Name of the model used
        
    Returns:
        Dictionary with token counts and cost estimate
    """
    # Convert input to text for token counting
    if isinstance(input_prompt, list):
        input_text = " ".join([
            msg.get("content", "") if isinstance(msg, dict) 
            else str(msg) for msg in input_prompt
        ])
    else:
        input_text = str(input_prompt)
    
    # Estimate token counts
    input_tokens = estimate_token_count(input_text)
    output_tokens = estimate_token_count(output)
    total_tokens = input_tokens + output_tokens
    
    # Cost estimation (rough rates as of 2024)
    cost_per_1k_tokens = {
        "gpt-4o": {"input": 0.0025, "output": 0.010},  # $2.5/$10 per 1M tokens
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "llama-3.1-8b-instant": {"input": 0.0001, "output": 0.0001},  # Groq pricing
    }
    
    model_costs = cost_per_1k_tokens.get(model_name, cost_per_1k_tokens["gpt-4o"])
    
    input_cost = (input_tokens / 1000) * model_costs["input"]
    output_cost = (output_tokens / 1000) * model_costs["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model_name
    }

def track_llm_call(input_prompt: Any, output: str, agent_type: str = "", 
                   operation: str = "", model_name: str = "gpt-4o"):
    """
    Convenience function to track an LLM call and add it to global tracker.
    
    Args:
        input_prompt: Input to the LLM
        output: Output from the LLM
        agent_type: Type of agent making the call
        operation: Specific operation being performed
        model_name: Model used for the call
    """
    cost_info = get_cost(input_prompt, output, model_name)
    tracker = get_token_tracker()
    
    tracker.add_usage(
        input_tokens=cost_info["input_tokens"],
        output_tokens=cost_info["output_tokens"],
        agent_type=agent_type,
        operation=operation,
        cost_estimate=cost_info["total_cost"]
    )
    
    return cost_info