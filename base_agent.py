from abc import ABC, abstractmethod
from typing import TypedDict, Dict, Any, List, Optional

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