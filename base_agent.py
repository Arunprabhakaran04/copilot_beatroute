from abc import ABC, abstractmethod
from typing import TypedDict, Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseAgentState(TypedDict):
    query: str
    agent_type: str
    user_id: str
    status: str
    error_message: str
    success_message: str
    result: Dict[str, Any]
    start_time: float
    end_time: float
    execution_time: float
    classification_confidence: Optional[float]
    redirect_count: Optional[int]
    # Multi-step processing fields
    original_query: str
    remaining_tasks: List[str]
    completed_steps: List[Dict[str, Any]]
    current_step: int
    is_multi_step: bool
    intermediate_results: Dict[str, Any]
    # Session management
    session_id: str
    # Conversation history for enrichment
    conversation_history: List[Dict[str, Any]]
    # Table callback for immediate streaming
    table_callback: Optional[Any]

class MeetingAgentState(BaseAgentState):
    meeting_date: str
    parsed_date: str
    file_path: str

class DBAgentState(BaseAgentState):
    sql_query: str
    query_type: str

class EmailAgentState(BaseAgentState):
    email_to: str
    email_subject: str
    email_content: str

class BaseAgent(ABC):
    def __init__(self, llm):
        self.llm = llm
    
    @abstractmethod
    def process(self, state: BaseAgentState) -> BaseAgentState:
        pass
    
    @abstractmethod
    def get_agent_type(self) -> str:
        pass
    
    def get_cached_db_result(self, state: BaseAgentState) -> Optional[Dict[str, Any]]:
        """
        Retrieve the last database query result from cache.
        All agents can use this method to access previous query results.
        
        Args:
            state: Current agent state containing session_id and user_id
            
        Returns:
            Dict containing cached query results, or None if not found
        """
        try:
            if not state.get("session_id") or not state.get("user_id"):
                logger.warning("Missing session_id or user_id - cannot retrieve cached data")
                return None
            
            from redis_memory_manager import get_last_db_query_result
            
            cached_result = get_last_db_query_result(
                state["session_id"],
                state.get("user_id")
            )
            
            if cached_result:
                logger.info(f"Retrieved cached DB result for {self.get_agent_type()} agent")
                return cached_result
            else:
                logger.debug(f"No cached DB result found for session {state['session_id']}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving cached data in {self.get_agent_type()} agent: {str(e)}")
            return None
    
    def should_use_cached_data(self, query: str) -> bool:
        """
        Determine if the query references previous data/results.
        
        Args:
            query: The user's query string
            
        Returns:
            True if query references previous data, False otherwise
        """
        query_lower = query.lower()
        
        # Phrases that indicate reference to previous results
        reference_phrases = [
            "that data", "this data", "the data", "those results", "these results",
            "previous query", "last query", "above data", "same data",
            "database results", "query results", "that result", "this result",
            "visualize that", "visualize this", "visualize it", "chart that",
            "summarize that", "summarize this", "summarize it",
            "from that", "from this", "from the", "using that", "using this",
            "based on that", "based on this", "for that", "for this",
            # Generic visualization/summary requests without explicit data source
            "provide a visualization", "create a visualization", "make a visualization",
            "provide a chart", "create a chart", "make a chart",
            "provide a summary", "create a summary", "make a summary", "give me a summary",
            "provide a graph", "create a graph", "make a graph",
            "show me a visualization", "show me a chart", "show me a summary"
        ]
        
        return any(phrase in query_lower for phrase in reference_phrases)