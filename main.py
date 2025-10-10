import os
import re
import time
from typing import Literal, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging

# Import all agent classes and state definitions
from base_agent import BaseAgent, BaseAgentState
from db_query_agent import DBQueryAgent
from email_agent import EmailAgent
from campaign_agent import CampaignAgent
from meeting_scheduler_agent import MeetingSchedulerAgent
from sql_retriever_agent import SQLRetrieverAgent
from summary_agent import SummaryAgent
from visualization_agent import VisualizationAgent
from memory_manager import ClassificationValidator, EnhancedMemoryManager
from token_tracker import get_token_tracker, track_llm_call

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CentralOrchestrator:
    def __init__(self, files_directory: str = "./user_files", schema_file_path: str = "schema"):
        load_dotenv()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        self.db_llm = ChatOpenAI(
            model="gpt-4o", 
            api_key=openai_api_key,
            temperature=0.1,
            max_tokens=2000
        )
        
        self.agents = {
            "db_query": DBQueryAgent(self.db_llm, schema_file_path),
            "email": EmailAgent(self.llm),
            "meeting": MeetingSchedulerAgent(self.llm, files_directory),
            "campaign": CampaignAgent(self.llm),
            "sql_retriever": SQLRetrieverAgent(self.db_llm, "embeddings.pkl"),
            "summary": SummaryAgent(self.db_llm, "gpt-4o"),
            "visualization": VisualizationAgent(self.db_llm, "gpt-4o")
        }
        
        logger.info("Initialized CentralOrchestrator with:")
        logger.info(f"  - DB Query Agent: OpenAI GPT-4o (orchestrator with multi-step capability)")
        logger.info(f"  - Email Agent: Groq Llama-3.1-8b-instant")
        logger.info(f"  - Meeting Agent: Groq Llama-3.1-8b-instant")
        logger.info(f"  - SQL Retriever Agent: OpenAI GPT-4o (for embeddings and retrieval)")
        logger.info(f"  - Summary Agent: OpenAI GPT-4o (for data summarization)")
        logger.info(f"  - Visualization Agent: OpenAI GPT-4o (for data visualization)")
        
        self.classification_keywords = {
            "db_query": [
                "database", "query", "sql", "insert", "select", "update", "delete",
                "add to cart", "product", "order", "table", "record", "data",
                "create", "remove", "modify", "store", "retrieve", "cart", "user profile",
                "show", "find", "get", "list", "search", "display", "customers", "brands"
            ],
            "email": [
                "email", "mail", "send", "notify", "message", "contact",
                "inform", "alert", "notification", "compose", "write to", "@",
                "cc", "bcc", "subject", "recipient", "sender", "reply"
            ],
            "meeting": [
                "meeting", "schedule", "appointment", "book", "calendar",
                "tomorrow", "today", "next week", "date", "time", "meet",
                "demo", "call", "session", "conference", "invite", "attendee"
            ],
            "campaign": [
                "campaign", "holiday sale", "summer promo", "black friday",
                "campaign id", "campaign name", "campaign data", "campaign info",
                "active campaigns", "campaign metrics", "campaign response"
            ],
            "summary": [
                "summarize", "summary", "analyze", "analysis", "insights", "overview",
                "explain", "breakdown", "interpret", "describe", "what does this mean",
                "tell me about", "explain the data", "data insights", "trends"
            ],
            "visualization": [
                "visualize", "chart", "graph", "plot", "show", "display", "draw",
                "bar chart", "line chart", "pie chart", "scatter plot", "histogram",
                "visualization", "visual", "graphical", "dashboard", "plot it",
                "create chart", "make graph", "draw chart", "show chart", "render"
            ]
        }
        
        self.classification_patterns = {
            "email": [
                (r'\b\w+@[\w.-]+\.\w+\b', 3.0),  # Email addresses
                (r'\b(send|compose|email|mail)\b.*@', 2.5),
                (r'\bcc\s+\w+@', 2.0),
                (r'\bsubject\s*:', 2.0),
            ],
            "meeting": [
                (r'\b(schedule|book|arrange)\s+(meeting|appointment|demo|call)', 3.0),
                (r'\bmeet\s+with\s+user\s+\d+', 2.5),
                (r'\b(tomorrow|today|next\s+\w+day)\b', 1.5),
                (r'\b\d{1,2}[:/]\d{2}(\s*(am|pm))?\b', 1.0),
            ],
            "db_query": [
                (r'\b(show|find|get|list|display)\s+(all|from)', 2.5),
                (r'\b(insert|create|add)\s+.*\s+(to|into|in)\b', 2.0),
                (r'\b(update|modify|change|edit)\b', 2.0),
                (r'\btable\b.*\b(records?|data|entries)\b', 2.0),
            ],
            "campaign": [
                (r'\b(holiday\s+sale|summer\s+promo|black\s+friday)\b', 3.0),
                (r'\bcampaign\s+(id|name|data|info|metrics)\b', 2.5),
                (r'\bactive\s+campaigns\b', 2.0),
            ],
            "summary": [
                (r'\b(summarize|summary|analyze|analysis)\b', 3.0),
                (r'\b(explain|describe|interpret)\s+.*\b(data|results|trends)\b', 2.5),
                (r'\b(what\s+does\s+this\s+mean|tell\s+me\s+about|insights)\b', 2.0),
                (r'\b(breakdown|overview)\s+', 1.5),
            ],
            "visualization": [
                (r'\b(visualize|chart|graph|plot|draw)\b', 3.0),
                (r'\b(bar\s+chart|line\s+chart|pie\s+chart|scatter\s+plot)\b', 3.5),
                (r'\b(show|display|render)\s+.*\b(chart|graph|plot|visual)\b', 2.5),
                (r'\b(create|make|generate)\s+.*\b(chart|graph|visualization)\b', 2.5),
                (r'\bplot\s+it\b', 2.0),
            ]
        }
        
        self.classification_validator = ClassificationValidator()
        self.graph = self._build_orchestrator_graph()
        self.memory = EnhancedMemoryManager(self.llm, max_history=3)
    
    def clear_session_memory(self):
        """Clear memory and reset session state"""
        self.memory.clear_memory()
        self.classification_validator = ClassificationValidator()  # Reset validation history
        
        # Clear token tracking session
        token_tracker = get_token_tracker()
        token_tracker.clear_session()
        
        logger.info("Session memory, validation history, and token tracking cleared")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            "memory_entries": len(self.memory.history),
            "classification_accuracy": self.classification_validator.get_classification_accuracy(),
            "memory_summary": self.memory.get_memory_summary()
        }
    
    def _build_orchestrator_graph(self) -> StateGraph:
        workflow = StateGraph(BaseAgentState)
        
        workflow.add_node("thinking_agent", self._thinking_agent)
        workflow.add_node("route_to_agent", self._route_to_agent)
        workflow.add_node("handle_error", self._handle_error)
        
        workflow.set_entry_point("thinking_agent")
        
        workflow.add_conditional_edges(
            "thinking_agent",
            self._should_continue_or_complete,
            {
                "continue": "route_to_agent",
                "complete": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "route_to_agent",
            self._should_return_to_thinking_or_error,
            {
                "thinking": "thinking_agent",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _thinking_agent(self, state: BaseAgentState) -> BaseAgentState:
        """Central thinking agent that orchestrates multi-step queries"""
        try:
            if state.get("current_step", 0) == 0:
                state["original_query"] = state["query"]
                state["completed_steps"] = []
                state["intermediate_results"] = {}
                state["current_step"] = 1
                
                # Check if this is a multi-step query and decompose it
                if self._is_multi_step_query(state["query"]):
                    state["is_multi_step"] = True
                    state["remaining_tasks"] = self._decompose_query(state["query"])
                    logger.info(f"Multi-step query detected. Tasks: {state['remaining_tasks']}")
                else:
                    state["is_multi_step"] = False
                    state["remaining_tasks"] = [state["query"]]
                    logger.info("Single-step query detected")
            
            # Check if we have more tasks to process
            if not state["remaining_tasks"]:
                state["status"] = "completed"
                state["success_message"] = self._generate_final_message(state)
                return state
            
            current_task = state["remaining_tasks"].pop(0)
            
            enhanced_task = self._enhance_task_with_context(current_task, state["intermediate_results"])
            
            # Classify the current task
            agent_type = self._classify_single_task(enhanced_task)
            
            state["query"] = enhanced_task
            state["agent_type"] = agent_type
            state["status"] = "ready_for_agent"
            
            logger.info(f"Step {state['current_step']}: Routing '{enhanced_task}' to {agent_type} agent")
            
            return state
            
        except Exception as e:
            state["error_message"] = f"Thinking agent error: {str(e)}"
            state["status"] = "failed"
            logger.error(f"Thinking agent error: {e}")
            return state
    
    def _pattern_classification(self, query: str) -> Dict[str, float]:
        """Enhanced pattern matching with weighted scoring"""
        scores = {}
        
        for agent_type, patterns in self.classification_patterns.items():
            total_score = 0
            for pattern, weight in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                total_score += matches * weight
            scores[agent_type] = total_score
        
        return scores
    
    def _keyword_classification(self, query: str) -> Dict[str, float]:
        """Keyword-based classification with scoring"""
        scores = {}
        query_words = set(query.lower().split())
        
        for agent_type, keywords in self.classification_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword in query)
            scores[agent_type] = keyword_matches * 0.5 
        
        return scores
    
    def _llm_classify_with_context(self, state: BaseAgentState, scores: Dict[str, float]) -> BaseAgentState:
        """LLM classification with context from pattern/keyword scores and multi-step detection"""
        try:
            # First, detect if this is a multi-step query
            if self._is_multi_step_query(state["query"]):
                return self._handle_multi_step_query(state, scores)
            
            scores_text = ", ".join([f"{k}: {v:.1f}" for k, v in scores.items()])
            
            classify_prompt = ChatPromptTemplate.from_template("""
            You are a production query classifier with access to preliminary scoring analysis.
            
            Query: {query}
            Preliminary Scores: {scores}
            
            Categories:
            1. campaign - Questions about campaigns, campaign IDs, campaign names, campaign data, Holiday Sale, Summer Promo, Black Friday, active campaigns, campaign metrics, etc.
            2. db_query - Database operations (show, find, insert, select, update, delete, list, display, retrieve data, etc.)
            3. email - Email operations (send, compose, cc, bcc, notify, @ symbols, email addresses, etc.)  
            4. meeting - Meeting/scheduling (schedule, book, appointment, meet, demo, call, dates, times, etc.)
            5. summary - Data summarization and analysis (summarize, analyze, explain, insights, overview, breakdown, interpret, describe data)
            6. visualization - Data visualization (visualize, chart, graph, plot, show chart, create graph, bar chart, line chart, pie chart, etc.)
            
            IMPORTANT: For multi-step queries, focus on the PRIMARY action (usually the first verb).
            
            EXAMPLES:
            "what is the holiday sale campaign id" → campaign
            "show me all campaigns" → campaign
            "send email to john@x.com" → email
            "schedule meeting with user 2" → meeting
            "show all customers" → db_query
            "find customers in Mumbai" → db_query
            "summarize customer data" → summary
            "analyze sales trends" → summary
            "create a chart of sales data" → visualization
            "visualize customer trends" → visualization
            "show all campaigns and email results to team@company.com" → campaign (PRIMARY action is "show campaigns")
            "get customer data and send to manager@company.com" → db_query (PRIMARY action is "get data")
            "show top customers and summarize" → db_query (PRIMARY action is "show customers")
            "show customers and create chart" → db_query (PRIMARY action is "show customers")
            
            Focus on the FIRST/PRIMARY action in compound queries.
            
            Respond with ONLY one word: campaign, db_query, email, meeting, summary, or visualization
            """)
            
            messages = classify_prompt.format_messages(query=state["query"], scores=scores_text)
            response = self.llm.invoke(messages)
            agent_type = response.content.strip().lower()
            
            valid_agents = ["campaign", "db_query", "email", "meeting", "summary", "visualization"]
            if agent_type in valid_agents:
                state["agent_type"] = agent_type
                state["status"] = "classified"
                state["classification_confidence"] = scores.get(agent_type, 1.0)
                logger.info(f"LLM Classification: {agent_type}")
                return state
            else:
                best_agent = max(scores, key=scores.get) if scores else "db_query"
                state["agent_type"] = best_agent
                state["status"] = "classified"
                state["classification_confidence"] = scores.get(best_agent, 0.5)
                logger.warning(f"LLM gave invalid response '{agent_type}', using highest score: {best_agent}")
                return state
                
        except Exception as e:
            best_agent = max(scores, key=scores.get) if scores else "db_query"
            state["agent_type"] = best_agent
            state["status"] = "classified"
            state["classification_confidence"] = 0.1  # Low confidence
            logger.error(f"LLM classification failed: {e}, using fallback: {best_agent}")
            return state
    
    def _is_multi_step_query(self, query: str) -> bool:
        """Detect if query contains multiple actions requiring different agents"""
        query_lower = query.lower()
        
        # Multi-step indicators
        multi_step_patterns = [
            r'\band\s+(email|send|notify)',  # "show data and email results"
            r'\band\s+(schedule|book|meet)',  # "get info and schedule meeting"  
            r'(show|get|find).*and.*(email|send)',  # "show X and send to Y"
            r'(email|send).*and.*(schedule|meet)',  # "email X and schedule Y"
            r'(show|get).*then.*(email|send|schedule)',  # "show X then send Y"
            r'\bthen\s+(email|send|schedule|book)',  # "do X then Y"
            r'(show|get|find).*and.*(summarize|analyze|explain)',  # "show X and summarize"
            r'\band\s+(summarize|analyze|summary)',  # "get data and summarize"
            r'(show|get).*then.*(summarize|analyze)',  # "show X then summarize"
            r'(show|get|find).*and.*(visualize|chart|plot|graph)',  # "show X and visualize"
            r'\band\s+(visualize|chart|plot|graph)',  # "get data and visualize"
            r'(show|get).*then.*(visualize|chart|plot)',  # "show X then visualize"
            r'(summarize|analyze).*and.*(visualize|chart|plot)',  # "summarize and visualize"
            r'(visualize|chart|plot).*and.*(summarize|analyze)',  # "visualize and summarize"
        ]
        
        return any(re.search(pattern, query_lower) for pattern in multi_step_patterns)
    
    def _is_db_with_summary_query(self, query: str) -> bool:
        """Detect if query requires both DB query and summarization"""
        query_lower = query.lower()
        
        # Check for DB + Summary/Visualization patterns
        db_summary_patterns = [
            r'\b(show|get|find|list).*and.*(summarize|analyze|explain)',
            r'\b(summarize|analyze|explain).*\b(customers|sales|data|orders|products)',
            r'\b(what|how).*\b(customers|sales|data|orders|products)',
            r'\btell\s+me\s+about.*\b(customers|sales|data|orders)',
            r'\bexplain.*\b(data|results|trends)',
            r'\b(show|get|find|list).*and.*(visualize|chart|plot|graph)',
            r'\b(visualize|chart|plot|graph).*\b(customers|sales|data|orders|products)',
            r'\b(create|make|generate).*\b(chart|graph|visualization).*\b(of|for).*\b(customers|sales|data)',
            r'\bplot.*\b(customers|sales|data|orders|products)',
        ]
        
        return any(re.search(pattern, query_lower) for pattern in db_summary_patterns)
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose multi-step query into individual tasks"""
        try:
            decompose_prompt = ChatPromptTemplate.from_template("""
            You are a query decomposer. Break down this multi-step query into individual, sequential tasks.
            
            Query: {query}
            
            Rules:
            1. Each task should be actionable by a single agent type (database, email, meeting, campaign, summary, visualization)
            2. Preserve all specific details (emails, dates, conditions, etc.)
            3. Tasks should be in logical execution order
            4. Use {{RESULT_FROM_STEP_N}} placeholder for data from previous steps
            5. For queries requiring data analysis/summarization/visualization, first get the data, then analyze/visualize
            
            Examples:
            Query: "find top 10 customers in Mumbai and send the list to xyz@gmail.com"
            Output:
            TASK_1: find top 10 customers in Mumbai
            TASK_2: send email to xyz@gmail.com with subject "Top 10 Customers in Mumbai" and content "{{RESULT_FROM_STEP_1}}"
            
            Query: "get campaign data for Holiday Sale and schedule meeting with user 5 tomorrow"
            Output:
            TASK_1: get campaign data for Holiday Sale
            TASK_2: schedule meeting with user 5 tomorrow
            
            Query: "show me top customers and summarize their behavior"
            Output:
            TASK_1: show me top customers
            TASK_2: summarize the customer behavior data from {{RESULT_FROM_STEP_1}}
            
            Query: "analyze sales trends for top products"
            Output:
            TASK_1: get sales data for top products
            TASK_2: analyze and summarize the sales trends from {{RESULT_FROM_STEP_1}}
            
            Query: "show top customers and create a chart"
            Output:
            TASK_1: show top customers
            TASK_2: create a visualization chart from {{RESULT_FROM_STEP_1}}
            
            Query: "get sales data, summarize it and visualize the trends"
            Output:
            TASK_1: get sales data
            TASK_2: summarize the sales data from {{RESULT_FROM_STEP_1}}
            TASK_3: create a visualization chart from {{RESULT_FROM_STEP_1}}
            
            Respond with each task on a new line starting with TASK_N:
            """)
            
            messages = decompose_prompt.format_messages(query=query)
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Track token usage
            track_llm_call(
                input_prompt=messages,
                output=content,
                agent_type="orchestrator", 
                operation="query_decomposition",
                model_name="llama-3.1-8b-instant"
            )
            
            # Parse tasks
            tasks = []
            for line in content.split('\n'):
                if line.strip().startswith('TASK_'):
                    task = line.split(':', 1)[1].strip()
                    tasks.append(task)
            
            return tasks if tasks else [query]  # Fallback to original query
            
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]  
    
    def _enhance_task_with_context(self, task: str, intermediate_results: Dict[str, Any]) -> str:
        """Enhance current task with data from previous steps"""
        enhanced_task = task
        
        for step_num, result in intermediate_results.items():
            placeholder = f"{{RESULT_FROM_STEP_{step_num}}}"
            if placeholder in enhanced_task:
                if isinstance(result, dict):
                    if "sql_query" in result: 
                        result_text = f"Database query result: {result.get('explanation', 'Query executed successfully')}"
                    elif "email_to" in result:  
                        result_text = f"Email sent to: {result['email_to']}"
                    elif "user_id" in result: 
                        result_text = f"Meeting scheduled with user {result['user_id']}"
                    else:
                        result_text = str(result)
                else:
                    result_text = str(result)
                
                enhanced_task = enhanced_task.replace(placeholder, result_text)
        
        return enhanced_task
    
    def _classify_single_task(self, task: str) -> str:
        """Classify a single task to determine which agent should handle it"""
        query_lower = task.lower()
        
        pattern_scores = self._pattern_classification(query_lower)
        keyword_scores = self._keyword_classification(query_lower)
        
        combined_scores = {}
        for agent_type in self.agents.keys():
            pattern_score = pattern_scores.get(agent_type, 0)
            keyword_score = keyword_scores.get(agent_type, 0)
            combined_scores[agent_type] = (pattern_score * 0.6) + (keyword_score * 0.4)
        
        max_score = max(combined_scores.values()) if combined_scores else 0
        best_agent = max(combined_scores, key=combined_scores.get) if combined_scores else "db_query"
        
        if max_score >= 1.5: 
            return best_agent
        
        # Fallback to LLM classification
        return self._llm_classify_single_task(task, combined_scores)
    
    def _llm_classify_single_task(self, task: str, scores: Dict[str, float]) -> str:
        """Use LLM to classify individual task"""
        try:
            classify_prompt = ChatPromptTemplate.from_template("""
            Classify this single task to determine which agent should handle it.
            
            Task: {task}
            
            Agent Types:
            - campaign: Campaign-related queries (campaign data, campaign IDs, Holiday Sale, etc.)
            - db_query: Database operations (show, find, get, list, insert, update, delete)
            - email: Email operations (send, compose, notify, email addresses with @)
            - meeting: Meeting/scheduling (schedule, book, appointment, meet with user)
            - summary: Data summarization (summarize, analyze, explain, insights, overview)
            - visualization: Data visualization (visualize, chart, graph, plot, create chart)
            
            Respond with ONLY one word: campaign, db_query, email, meeting, summary, or visualization
            """)
            
            messages = classify_prompt.format_messages(task=task)
            response = self.llm.invoke(messages)
            agent_type = response.content.strip().lower()
            
            if agent_type in ["campaign", "db_query", "email", "meeting", "summary", "visualization"]:
                return agent_type
            else:
                return "db_query"  # Default fallback
                
        except Exception as e:
            logger.error(f"LLM task classification failed: {e}")
            return "db_query"  # Default fallback
    
    def _handle_multi_step_query(self, state: BaseAgentState, scores: Dict[str, float]) -> BaseAgentState:
        """Handle multi-step queries by identifying primary action"""
        query = state["query"].lower()
        
        # Priority order for multi-step queries (primary action wins)
        priority_patterns = [
            ("campaign", [r'\b(show|get|find|list|display)\s+.*campaigns?', r'\bcampaign\s+(data|info|id|name)']),
            ("db_query", [r'\b(show|get|find|list|display)\s+.*(customers?|data|records?|users?|products?)', r'\b(insert|update|delete|create)\s+']),
            ("meeting", [r'\b(schedule|book|arrange)\s+.*(meeting|appointment|demo|call)', r'\bmeet\s+with\s+user\s+\d+']),
            ("email", [r'^\s*(send|compose|email|mail)\s+', r'^.*email\s+to\s+\w+@'])  # Only if email is the primary action
        ]
        
        for agent_type, patterns in priority_patterns:
            for pattern in patterns:
                if re.search(pattern, query):
                    logger.info(f"Multi-step query detected. Primary action: {agent_type}")
                    state["agent_type"] = agent_type
                    state["status"] = "classified"
                    state["classification_confidence"] = scores.get(agent_type, 1.5) 
                    
                    # Store the full query for potential chaining
                    state["original_multi_step_query"] = state["query"]
                    return state
        
        # Fallback to highest scoring agent
        best_agent = max(scores, key=scores.get) if scores else "db_query"
        state["agent_type"] = best_agent
        state["status"] = "classified" 
        state["classification_confidence"] = scores.get(best_agent, 0.8)
        logger.info(f"Multi-step query fallback to: {best_agent}")
        return state
    
    def _route_to_agent(self, state: BaseAgentState) -> BaseAgentState:
        """Route to specific agent and capture results for multi-step processing"""
        try:
            agent_type = state["agent_type"]
            if agent_type not in self.agents:
                return self._handle_routing_error(state, f"Unknown agent type: {agent_type}")
            
            logger.info(f"Executing step {state['current_step']}: {agent_type} agent")
            agent = self.agents[agent_type]
            
            if agent_type == "campaign":
                result_state = self._handle_campaign_routing(state, agent)
            else:
                result_state = self._handle_direct_routing(state, agent)
            
            # If agent execution was successful, capture the result
            if result_state["status"] == "completed":
                # Store the result for future steps
                step_key = str(state["current_step"])
                result_state["intermediate_results"][step_key] = result_state["result"]
                
                # Record completed step
                completed_step = {
                    "step": state["current_step"],
                    "agent_type": agent_type,
                    "task": state["query"],
                    "result": result_state["result"],
                    "success_message": result_state["success_message"]
                }
                result_state["completed_steps"].append(completed_step)
                
                # Increment step counter
                result_state["current_step"] = state["current_step"] + 1
                
                # Set status to return to thinking agent
                result_state["status"] = "step_completed"
                
                logger.info(f"Step {state['current_step']} completed successfully")
                
            return result_state
                
        except Exception as e:
            return self._handle_routing_error(state, f"Routing error: {str(e)}")
    
    def _handle_campaign_routing(self, state: BaseAgentState, agent: BaseAgent) -> BaseAgentState:
        """Handle campaign agent routing with proper state management"""
        max_redirects = 2  
        redirect_count = state.get("redirect_count", 0)
        
        if redirect_count >= max_redirects:
            logger.error(f"Maximum redirects exceeded for campaign routing")
            return self._handle_routing_error(state, "Campaign routing exceeded maximum redirects")
        
        campaign_result = agent.process(state)
        
        if campaign_result["status"] == "completed":
            logger.info("Campaign agent provided direct answer")
            return campaign_result
        
        if campaign_result["status"] == "campaign_processed":
            enriched_query = campaign_result.get("result", {}).get("enriched_query")
            
            if not enriched_query:
                logger.warning("Campaign agent processed but no enriched query found")
                return self._fallback_to_db_query(state)
            
            return self._reclassify_enriched_query(state, enriched_query, redirect_count + 1)
        
        logger.warning(f"Unexpected campaign agent status: {campaign_result['status']}")
        return self._fallback_to_db_query(state)
    
    def _handle_direct_routing(self, state: BaseAgentState, agent: BaseAgent) -> BaseAgentState:
        """Handle direct agent routing with validation"""
        try:
            if not self._validate_agent_capability(state["agent_type"], state["query"]):
                logger.warning(f"Agent {state['agent_type']} may not be suitable for query: {state['query']}")
            
            # If this is a db_query agent, first retrieve similar SQL examples
            if state["agent_type"] == "db_query":
                try:
                    sql_retriever = self.agents["sql_retriever"]
                    similar_sqls = sql_retriever.retrieve_similar_sql(state["query"], k=5)
                    
                    if similar_sqls:
                        state["retrieved_sql_context"] = similar_sqls
                        logger.info(f"Retrieved {len(similar_sqls)} similar SQL examples for db_query agent")
                    else:
                        logger.info("No similar SQL examples found for db_query agent")
                        
                except Exception as e:
                    logger.warning(f"SQL retrieval failed, proceeding without examples: {e}")
                    # Continue without SQL examples if retrieval fails
            
            result_state = agent.process(state)
            
            self.classification_validator.validate_classification(
                state["query"], 
                state["agent_type"], 
                result_state["status"]
            )
            
            return result_state
        except Exception as e:
            logger.error(f"Direct routing failed for {state['agent_type']}: {e}")
            return self._handle_agent_failure(state, state["agent_type"])
    
    def _reclassify_enriched_query(self, state: BaseAgentState, enriched_query: str, redirect_count: int) -> BaseAgentState:
        """Reclassify enriched query with error handling"""
        try:
            reclassify_prompt = ChatPromptTemplate.from_template("""
            Classify this enriched query into exactly ONE category.
            
            Query: {query}
            
            Categories:
            1. db_query - Database operations
            2. email - Email operations  
            3. meeting - Meeting/scheduling
            
            Respond with ONLY one word: db_query, email, or meeting
            """)
            
            messages = reclassify_prompt.format_messages(query=enriched_query)
            response = self.llm.invoke(messages)
            new_agent_type = response.content.strip().lower()
            
            if new_agent_type in ["db_query", "email", "meeting"]:
                new_state = dict(state)
                new_state["agent_type"] = new_agent_type
                new_state["query"] = enriched_query
                new_state["redirect_count"] = redirect_count
                
                logger.info(f"Reclassified enriched query as: {new_agent_type}")
                return self.agents[new_agent_type].process(new_state)
            else:
                return self._fallback_to_db_query(state)
                
        except Exception as e:
            logger.error(f"Reclassification failed: {e}")
            return self._fallback_to_db_query(state)
    
    def _validate_agent_capability(self, agent_type: str, query: str) -> bool:
        """Validate if agent can handle the specific query"""
        validation_rules = {
            "email": lambda q: bool(re.search(r'\b\w+@[\w.-]+\.\w+\b', q)),
            "meeting": lambda q: bool(re.search(r'\buser\s+\d+\b', q)),
            "db_query": lambda q: len(q.split()) >= 3,  
            "campaign": lambda q: any(term in q.lower() for term in ["campaign", "holiday", "promo", "sale"])
        }
        
        return validation_rules.get(agent_type, lambda q: True)(query)
    
    def _fallback_to_db_query(self, state: BaseAgentState) -> BaseAgentState:
        """Fallback to db_query agent"""
        logger.info("Falling back to db_query agent")
        state["agent_type"] = "db_query"
        return self.agents["db_query"].process(state)
    
    def _handle_routing_error(self, state: BaseAgentState, error_message: str) -> BaseAgentState:
        """Centralized error handling for routing"""
        state["error_message"] = error_message
        state["status"] = "routing_error"
        logger.error(error_message)
        return state
    
    def _handle_agent_failure(self, state: BaseAgentState, failed_agent: str) -> BaseAgentState:
        """Handle agent failures with graceful degradation"""
        fallback_order = {
            "campaign": ["db_query"],
            "email": ["db_query"],
            "meeting": ["db_query"],
            "db_query": []  
        }
        
        fallbacks = fallback_order.get(failed_agent, [])
        
        for fallback_agent in fallbacks:
            try:
                logger.info(f"Attempting fallback from {failed_agent} to {fallback_agent}")
                state["agent_type"] = fallback_agent
                return self.agents[fallback_agent].process(state)
            except Exception as e:
                logger.error(f"Fallback to {fallback_agent} also failed: {e}")
                continue
        
        state["error_message"] = f"All agents failed for query type: {failed_agent}"
        state["status"] = "failed"
        return state
    
    def _handle_error(self, state: BaseAgentState) -> BaseAgentState:
        state["status"] = "failed"
        logger.error(f"Error state: {state.get('error_message', 'Unknown error')}")
        return state
    
    def _should_continue_or_complete(self, state: BaseAgentState) -> Literal["continue", "complete", "error"]:
        """Determine if thinking agent should continue, complete, or error"""
        if state["status"] == "ready_for_agent":
            return "continue"
        elif state["status"] == "completed":
            return "complete"
        else:
            return "error"
    
    def _should_return_to_thinking_or_error(self, state: BaseAgentState) -> Literal["thinking", "error"]:
        """Determine if we should return to thinking agent or error"""
        if state["status"] == "step_completed":
            return "thinking"
        else:
            return "error"
    
    def _generate_final_message(self, state: BaseAgentState) -> str:
        """Generate final success message summarizing all completed steps"""
        if not state.get("is_multi_step", False):
            # Single step - use the last step's success message
            if state["completed_steps"]:
                return state["completed_steps"][-1]["success_message"]
            else:
                return "Task completed successfully"
        
        # Multi-step - summarize all steps
        step_count = len(state["completed_steps"])
        summary = f"Multi-step query completed successfully ({step_count} steps):\n"
        
        for step in state["completed_steps"]:
            summary += f"  Step {step['step']}: {step['success_message']}\n"
        
        return summary.strip()
    
    def add_agent(self, agent_type: str, agent: BaseAgent, keywords: List[str]):
        self.agents[agent_type] = agent
        self.classification_keywords[agent_type] = keywords
        print(f"Added new agent: {agent_type}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        enriched_query = self.memory.enrich_query(query)

        initial_state = BaseAgentState(
            query=enriched_query,
            agent_type="",
            user_id="",
            status="",
            error_message="",
            success_message="",
            result={},
            start_time=time.time(),
            end_time=0.0,
            execution_time=0.0,
            classification_confidence=None,
            redirect_count=0,
            # Multi-step fields
            original_query=query,
            remaining_tasks=[],
            completed_steps=[],
            current_step=0,
            is_multi_step=False,
            intermediate_results={}
        )
        
        print(f"\nProcessing Query: '{query}'")
        if enriched_query != query:
            print(f"Enriched Query: '{enriched_query}'")
        
        start_time = time.time()
        result = self.graph.invoke(initial_state)
        end_time = time.time()
        execution_time = end_time - start_time
        
        result["start_time"] = start_time
        result["end_time"] = end_time
        result["execution_time"] = execution_time
        
        print(f"Execution Time: {execution_time:.4f} seconds")
        
        # Get token usage summary
        token_tracker = get_token_tracker()
        token_summary = token_tracker.get_session_summary()
        print(f"Token Usage: {token_summary['total_tokens']} tokens (${token_summary['total_cost']:.4f})")
        
        if result.get("classification_confidence"):
            print(f"Classification Confidence: {result['classification_confidence']:.2f}")
        
        accuracy = self.classification_validator.get_classification_accuracy()
        if accuracy > 0:
            print(f"Recent Classification Accuracy: {accuracy:.1%}")
        
        if result["status"] == "completed":
            print(f"Success: {result['success_message']}")
            
            if result.get("is_multi_step", False) and result.get("completed_steps"):
                print("\nStep Details:")
                for step in result["completed_steps"]:
                    print(f"  Step {step['step']} ({step['agent_type']}): {step['success_message']}")
            
            try:
                final_agent_type = "multi_step" if result.get("is_multi_step", False) else result.get("agent_type", "unknown")
                final_result = {
                    "steps": result.get("completed_steps", []),
                    "final_result": result.get("result", {}),
                    "is_multi_step": result.get("is_multi_step", False)
                }
                self.memory.add_entry(query, enriched_query, final_agent_type, final_result)
            except Exception as e:
                logger.warning(f"Failed to add memory entry: {e}")
            
            return {
                "success": True,
                "message": result["success_message"],
                "agent_type": result.get("agent_type", "multi_step"),
                "result": result.get("result", {}),
                "completed_steps": result.get("completed_steps", []),
                "is_multi_step": result.get("is_multi_step", False),
                "execution_time": execution_time,
                "classification_confidence": result.get("classification_confidence"),
                "token_usage": token_summary
            }
        else:
            print(f"Error: {result['error_message']}")
            return {
                "success": False,
                "error": result.get("error_message", "Unknown error occurred"),
                "status": result["status"],
                "agent_type": result.get("agent_type", "unknown"),
                "execution_time": execution_time,
                "classification_confidence": result.get("classification_confidence"),
                "token_usage": token_summary
            }



def main():
    try:
        orchestrator = CentralOrchestrator(files_directory="./user_files", schema_file_path="schema")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check your GROQ API key configuration.")
        return
    
    print("Multi-Agent Orchestrator System")
    print("=" * 70)
    print("Commands:")
    print("  - 'quit'/'exit' - Exit the system and clear memory")
    print("  - 'clear' - Clear session memory")
    print("  - 'stats' - Show session statistics")
    print("=" * 70)
    print("Example queries:")
    print("  - 'Show all brands'")
    print("  - 'Find customers in Mumbai'")
    print("  - 'Get campaign responses from last week'")
    print("  - 'Show user performance data'")
    print("  - 'Send email to john@example.com about campaign results'")
    print("  - 'Schedule meeting with user 3 tomorrow'")
    print("  - 'Show top 5 customers and summarize their behavior'")
    print("  - 'Analyze sales trends for Mumbai customers'")
    print("  - 'Get customer data and explain the key insights'")
    print("  - 'Create a chart of top 10 customers by sales'")
    print("  - 'Visualize sales trends for the last quarter'")
    print("  - 'Show customer data, summarize it and create a chart'")
    print("=" * 70)
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            orchestrator.clear_session_memory()
            print("Session memory cleared. Goodbye!")
            break
        
        if query.lower() == 'clear':
            orchestrator.clear_session_memory()
            print("Session memory cleared.")
            continue
            
        if query.lower() == 'stats':
            stats = orchestrator.get_session_stats()
            print(f"\nSession Statistics:")
            print(f"Memory Entries: {stats['memory_entries']}")
            print(f"Classification Accuracy: {stats['classification_accuracy']:.1%}")
            print(stats['memory_summary'])
            continue
        
        if query == '':
            continue
        
        result = orchestrator.process_query(query)
        
        if not result["success"]:
            print(f"Status: {result.get('status', 'unknown')}")

if __name__ == "__main__":
    main()