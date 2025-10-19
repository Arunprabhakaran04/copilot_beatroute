import os
import re
import time
from typing import Literal, Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
from clean_logging import setup_logging, AgentLogger, SQLLogger, TokenLogger, OrchestrationLogger
from loguru import logger

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
from improved_sql_generator import ImprovedSQLGenerator
from temp_files.improved_multi_step_handler import create_improved_decomposition
from temp_files.llm_multi_step_analyzer import EnhancedMultiStepHandler, create_enhanced_llm_decomposition
from temp_files.enhanced_ultra_analyzer import EnhancedUltraAnalyzer
from agent_aware_decomposer import AgentAwareDecomposer

# Setup clean logging
setup_logging()


class CentralOrchestrator:
    def __init__(self, files_directory: str = "./user_files", schema_file_path: str = "schema"):
        load_dotenv()
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=openai_api_key,
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
            "visualization": VisualizationAgent(self.db_llm, "gpt-4o"),
            "improved_sql_generator": ImprovedSQLGenerator(self.db_llm, schema_file_path)
        }
        
        logger.info("Initialized CentralOrchestrator with:")
        logger.info(f"  - DB Query Agent: OpenAI GPT-4o (orchestrator with multi-step capability)")
        logger.info(f"  - Email Agent: OpenAI GPT-4o")
        logger.info(f"  - Meeting Agent: OpenAI GPT-4o")
        logger.info(f"  - SQL Retriever Agent: OpenAI GPT-4o (for embeddings and retrieval)")
        logger.info(f"  - Summary Agent: OpenAI GPT-4o (for data summarization)")
        logger.info(f"  - Visualization Agent: OpenAI GPT-4o (for data visualization)")
        
        self.classification_keywords = {
            "db_query": [
                "database", "query", "sql", "insert", "select", "update", "delete",
                "add to cart", "product", "order", "table", "record", "data",
                "create", "remove", "modify", "store", "retrieve", "cart", "user profile",
                "show", "find", "get", "list", "search", "display", "customers", "brands",
                "top", "top 3", "top 5", "top 10", "best", "highest", "lowest", "most",
                "skus", "sku", "products", "sales", "orders", "users", "month", "week", 
                "year", "september", "january", "february", "march", "april", "may", 
                "june", "july", "august", "october", "november", "december"
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
                "tell me about", "explain the data", "data insights", "trends",
                "behavior", "pattern", "meaning"
            ],
            "visualization": [
                "visualize", "chart", "graph", "plot", "draw",
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
                (r'\b(top|best|highest|lowest|most)\s+\d+', 3.5),  # "top 3", "best 5", etc.
                (r'\b(give|show|find|get)\s+.*\b(top|best|highest|lowest)', 3.0),
                (r'\b(skus?|products?|customers?|users?|orders?|sales)\b', 2.0),
                (r'\bfor\s+the\s+(month|week|year|day)\b', 2.0),  # "for the month"
                (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 1.5),
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
                (r'\b(behavior|pattern|meaning|trend)\b', 1.0),
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
        self.enhanced_multi_step_handler = EnhancedMultiStepHandler(self.db_llm)
        self.enhanced_ultra_analyzer = EnhancedUltraAnalyzer(self.db_llm)
        self.agent_aware_decomposer = AgentAwareDecomposer(self.db_llm)
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
                
                # Use NEW Agent-Aware Decomposer with full knowledge of agent capabilities
                try:
                    analysis_result = self.agent_aware_decomposer.analyze_and_decompose(state["query"])
                    
                    if analysis_result.get("is_multi_step", False):
                        state["is_multi_step"] = True
                        # Get decomposed tasks from the analysis result
                        if "decomposed_tasks" in analysis_result:
                            state["remaining_tasks"] = analysis_result["decomposed_tasks"]
                        else:
                            # Fallback to simple decomposition based on the analysis
                            state["remaining_tasks"] = self._decompose_query(state["query"])
                        
                        OrchestrationLogger.multi_step_detected(state['remaining_tasks'])
                        
                        logger.info(f"🤖 Agent-Aware Analysis: Multi-step detected (confidence: {analysis_result.get('confidence', 0):.2f})")
                        logger.info(f"📋 Method: {analysis_result.get('decomposition_method', 'unknown')}")
                        logger.info(f"💡 Reasoning: {analysis_result.get('reasoning', 'No reasoning provided')}")
                        
                        # Enhanced logging for task details if available
                        if "tasks" in analysis_result and isinstance(analysis_result["tasks"], list):
                            logger.info(f"✅ Optimized into {len(analysis_result['tasks'])} tasks:")
                            for i, task in enumerate(analysis_result["tasks"], 1):
                                if isinstance(task, dict):
                                    agent = task.get('agent', 'unknown')
                                    desc = task.get('description', task)
                                    logger.info(f"  Task {i} ({agent}): {desc}")
                                else:
                                    logger.info(f"  Task {i}: {task}")
                            
                            # Show optimization notes if available
                            if "optimization_notes" in analysis_result:
                                logger.info(f"🔧 Optimization: {analysis_result['optimization_notes']}")
                        else:
                            logger.info(f"✅ Decomposed into {len(state['remaining_tasks'])} tasks:")
                            for i, task in enumerate(state['remaining_tasks'], 1):
                                logger.info(f"  Task {i}: {task}")
                    else:
                        state["is_multi_step"] = False
                        state["remaining_tasks"] = [state["query"]]
                        logger.info(f"🤖 Agent-Aware Analysis: Single-step query detected (confidence: {analysis_result.get('confidence', 0):.2f})")
                        logger.info(f"💡 Reasoning: {analysis_result.get('reasoning', 'No reasoning provided')}")
                except Exception as e:
                    logger.warning(f"Agent-aware analysis failed, falling back to enhanced ultra analyzer: {e}")
                    # Fallback to enhanced ultra analyzer
                    try:
                        analysis_result = self.enhanced_ultra_analyzer.analyze_and_decompose(state["query"])
                        
                        if analysis_result.get("is_multi_step", False):
                            state["is_multi_step"] = True
                            if "decomposed_tasks" in analysis_result:
                                state["remaining_tasks"] = analysis_result["decomposed_tasks"]
                            else:
                                state["remaining_tasks"] = self._decompose_query(state["query"])
                            OrchestrationLogger.multi_step_detected(state['remaining_tasks'])
                            logger.info(f"Enhanced Ultra Fallback: decomposed into {len(state['remaining_tasks'])} tasks")
                        else:
                            state["is_multi_step"] = False
                            state["remaining_tasks"] = [state["query"]]
                            logger.info("Enhanced Ultra Fallback: Single-step query detected")
                    except Exception as e2:
                        logger.warning(f"Enhanced Ultra analysis also failed, using standard LLM: {e2}")
                        # Fallback to standard LLM method
                        try:
                            if self.enhanced_multi_step_handler.should_decompose_query(state["query"]):
                                state["is_multi_step"] = True
                                state["remaining_tasks"] = self.enhanced_multi_step_handler.decompose_query(state["query"])
                                OrchestrationLogger.multi_step_detected(state['remaining_tasks'])
                                logger.info(f"Standard LLM fallback: decomposed into {len(state['remaining_tasks'])} tasks")
                            else:
                                state["is_multi_step"] = False
                                state["remaining_tasks"] = [state["query"]]
                                logger.info("Standard LLM fallback: Single-step query detected")
                        except Exception as e3:
                            logger.warning(f"Standard LLM analysis also failed, using heuristic: {e3}")
                            # Final fallback to heuristic method
                            if self._is_multi_step_query(state["query"]):
                                state["is_multi_step"] = True
                                state["remaining_tasks"] = self._decompose_query(state["query"])
                                OrchestrationLogger.multi_step_detected(state['remaining_tasks'])
                                logger.info(f"Heuristic fallback: Multi-step query decomposed into {len(state['remaining_tasks'])} tasks")
                            else:
                                state["is_multi_step"] = False
                                state["remaining_tasks"] = [state["query"]]
                                logger.info("Heuristic fallback: Single-step query detected")
            
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
            
            OrchestrationLogger.step_routing(state['current_step'], enhanced_task, agent_type)
            
            # ENHANCED: Log intermediate results context for debugging
            if state.get("intermediate_results"):
                logger.info(f"Intermediate results available from {len(state['intermediate_results'])} previous step(s)")
                for step_key, step_result in state["intermediate_results"].items():
                    if isinstance(step_result, dict):
                        result_info = f"Step {step_key}: "
                        if "query_results" in step_result:
                            data_count = len(step_result["query_results"].get("data", []))
                            result_info += f"DB result with {data_count} rows"
                        elif "summary" in step_result:
                            result_info += "Summary result"
                        elif "visualization" in step_result:
                            result_info += "Visualization result"
                        else:
                            result_info += f"Result with keys: {list(step_result.keys())}"
                        logger.info(result_info)
            
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
            
            CRITICAL RULE: If the query requests specific data that needs to be retrieved from database 
            (like "top 3 customers", "sales for September", "show products", "give top skus", etc.), 
            classify as "db_query" FIRST, regardless of summary/analysis words present.
            
            Summary and visualization should only be used when data is already provided or when the 
            primary intent is to analyze existing data.
            
            IMPORTANT: For multi-step queries, focus on the PRIMARY action (usually the first verb).
            
            EXAMPLES:
            "what is the holiday sale campaign id" → campaign
            "show me all campaigns" → campaign
            "send email to john@x.com" → email
            "schedule meeting with user 2" → meeting
            "show all customers" → db_query
            "find customers in Mumbai" → db_query
            "give the top 3 skus for the month september" → db_query (needs data retrieval)
            "top 5 products by sales" → db_query (needs data retrieval)
            "summarize customer data" → summary (only if data already available)
            "analyze sales trends" → summary (only if data already available)
            "create a chart of sales data" → visualization (only if data already available)
            "visualize customer trends" → visualization (only if data already available)
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
        """Detect if query contains multiple actions requiring different agents or complex dependencies"""
        query_lower = query.lower()
        
        # DEBUG: Add detailed logging for troubleshooting
        logger.info(f"🔍 MULTI-STEP DETECTION DEBUG:")
        logger.info(f"   Query: {query}")
        logger.info(f"   Query (lower): {query_lower}")
        
        # ENHANCED: Always treat visualization and summarization queries as multi-step
        # if they don't explicitly contain database operations
        forced_result = self._requires_forced_multi_step_decomposition(query_lower)
        logger.info(f"   Forced multi-step result: {forced_result}")
        if forced_result:
            logger.info("   ✅ DETECTED as multi-step via forced decomposition")
            return True
        
        # Explicit multi-step indicators (explicit connectives)
        explicit_multi_step_patterns = [
            r'\band\s+(email|send|notify)',  # "show data and email results"
            r'\band\s+(schedule|book|meet)',  # "get info and schedule meeting"  
            r'(show|get|find).*and.*(email|send)',  # "show X and send to Y"
            r'(email|send).*and.*(schedule|meet)',  # "email X and schedule Y"
            r'(show|get).*then.*(email|send|schedule)',  # "show X then send Y"
            r'\bthen\s+(email|send|schedule|book)',  # "do X then Y"
            r'(show|get|find|give|top).*and.*(summarize|analyze|explain)',  # "show X and summarize"
            r'\band\s+(summarize|analyze|summary)',  # "get data and summarize"
            r'\band\s+give\s+me\s+a?\s*(summary|summarize|analysis)',  # "X and give me a summary"
            r'(show|get|give|top).*then.*(summarize|analyze)',  # "show X then summarize"
            r'(show|get|find|give|top).*and.*(visualize|chart|plot|graph)',  # "show X and visualize"
            r'\band\s+(visualize|chart|plot|graph)',  # "get data and visualize"
            r'(show|get|give|top).*then.*(visualize|chart|plot)',  # "show X then visualize"
            r'(summarize|analyze).*and.*(visualize|chart|plot)',  # "summarize and visualize"
            r'(visualize|chart|plot).*and.*(summarize|analyze)',  # "visualize and summarize"
            r'and\s+(provide|create|generate)\s+a?\s*(summary|analysis|report)',  # "X and provide summary"
        ]
        
        # Implicit multi-step indicators (complex dependency patterns)
        implicit_multi_step_patterns = [
            # Trend analysis for top items - requires finding top items first, then analyzing trends
            r'(trend|trends)\s+for.*top\s+\d+',  # "trend for top 3"
            r'(sales\s+trend|trend\s+analysis).*top\s+\d+',  # "sales trend for top 3"
            r'(top\s+\d+).*\b(trend|trends|pattern|analysis|over\s+time)',  # "top 3 ... trend"
            r'(trend|pattern|analysis).*\b(top\s+\d+)',  # "trend of top 3"
            
            # Complex time-based analysis requiring sub-queries
            r'(trend|analysis|pattern).*\b(last|past)\s+\d+\s+(months?|weeks?|days?)',  # trend over time period
            r'(performance|analysis).*\bfor.*\btop\s+\d+',  # "performance for top X"
            r'(compare|comparison).*\btop\s+\d+',  # "compare top X"
            
            # Complex aggregations that need sub-queries
            r'(breakdown|detailed?\s+analysis).*\bof.*\btop\s+\d+',
            r'(monthly|weekly|daily).*\b(trend|analysis|breakdown).*\btop\s+\d+',
            r'\b(top\s+\d+).*\b(monthly|weekly|daily|over\s+time)',
            
            # Analysis requiring filtering by dynamic results
            r'(analysis|breakdown|trend).*\bfor.*\b(best|top|highest)\s+\w+',
            r'(historical|time\s+series).*\bfor.*\btop\s+\d+',
        ]
        
        # Check explicit patterns first
        explicit_match = any(re.search(pattern, query_lower) for pattern in explicit_multi_step_patterns)
        logger.info(f"   Explicit pattern match: {explicit_match}")
        if explicit_match:
            logger.info("   ✅ DETECTED as multi-step via explicit patterns")
            return True
        
        # Check implicit patterns
        implicit_match = any(re.search(pattern, query_lower) for pattern in implicit_multi_step_patterns)
        logger.info(f"   Implicit pattern match: {implicit_match}")
        if implicit_match:
            logger.info("   ✅ DETECTED as multi-step via implicit patterns")
            return True
        
        logger.info("   ❌ NOT DETECTED as multi-step")
        return False
    
    def _requires_forced_multi_step_decomposition(self, query: str) -> bool:
        """
        Detect if query requires visualization or summarization and should be forced
        into multi-step decomposition (DB Query -> Visualization/Summary)
        """
        
        # DEBUG: Add detailed logging
        logger.debug(f"🔍 FORCED DECOMPOSITION CHECK:")
        logger.debug(f"   Query: {query}")
        
        # Patterns that indicate visualization needs
        visualization_patterns = [
            r'\b(visualize|chart|graph|plot|draw)\b',
            r'\b(bar\s+chart|line\s+chart|pie\s+chart|scatter\s+plot|histogram)\b',
            r'\b(create|make|generate|show).*\b(chart|graph|visualization|plot)\b',
            r'\bplot\s+(it|this|that|the\s+data)\b',
            r'\b(show|display).*\b(chart|graph|visual)\b',
            r'\bvisualization\s+(of|for|showing)\b'
        ]
        
        # Patterns that indicate summarization needs  
        summarization_patterns = [
            r'\b(summarize|summary|analyze|analysis)\b',
            r'\b(explain|describe|interpret).*\b(data|results|trends|behavior|patterns)\b',
            r'\b(what\s+does\s+this\s+mean|tell\s+me\s+about|insights)\b',
            r'\b(breakdown|overview)\s+(of|for)\b',
            r'\b(trends|patterns|behavior)\s+(in|of|for)\b',
            r'\b(analyze|analysis)\s+(the|this)\s+(data|results)\b',
            r'\b(insights|key\s+findings|observations)\b'
        ]
        
        # Check if query contains visualization keywords
        has_visualization = any(re.search(pattern, query, re.IGNORECASE) for pattern in visualization_patterns)
        logger.debug(f"   Has visualization: {has_visualization}")
        
        # Check if query contains summarization keywords  
        has_summarization = any(re.search(pattern, query, re.IGNORECASE) for pattern in summarization_patterns)
        logger.debug(f"   Has summarization: {has_summarization}")
        
        # If query has viz/summary keywords, check if it already has explicit data retrieval
        if has_visualization or has_summarization:
            
            # For visualization queries, be more strict about what counts as "explicit DB"
            # Only allow very clear data retrieval + visualization combos to skip forced decomposition
            if has_visualization:
                # Very specific patterns that explicitly request data AND visualization in one go
                explicit_viz_db_patterns = [
                    r'\b(show|get|find|list|display)\s+.*\b(data|sales|orders|customers)\s+and\s+(visualize|chart|graph|plot)\b',
                    r'\b(show|display)\s+.*\b(chart|graph|visualization)\s+of\s+.*\b(data|sales|orders)\b',
                    r'\b(get|show|find)\s+.*\bdata\s+.*\band\s+(chart|graph|plot|visualize)\b',
                    r'\b(get|show|find|list|display)\s+.*\b(data|sales|orders|customers)\s+and\s+(create|make|generate)\s+.*\b(chart|graph|plot)\b'
                ]
                
                has_explicit_viz_db = any(re.search(pattern, query, re.IGNORECASE) for pattern in explicit_viz_db_patterns)
                logger.debug(f"   Has explicit viz+DB: {has_explicit_viz_db}")
                
                if not has_explicit_viz_db:
                    logger.debug(f"   ✅ FORCED: Visualization query requires multi-step decomposition")
                    return True
            
            # For summarization, use the original logic
            elif has_summarization:
                # Look for explicit database operations that show data is being requested
                explicit_db_patterns = [
                    r'\b(show|get|find|list|select|display).*\b(all|from|where)\b',
                    r'\b(show|get|find|list)\s+.*\b(customers|products|sales|orders|data|users|campaigns)\b',
                    r'\b(top|best|highest|lowest)\s+\d+',
                    r'\b(give|show|find)\s+.*\b(customers|products|sales|orders|data)\b',
                    r'\b(show|get|list|find|display)\s+.*\bfor\s+the\s+(month|week|year|day)\b',
                    r'\b(show|get|list|find|display)\s+.*\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
                    r'\band\s+(get|show|find|list|display)\b'  # "X and get/show Y"
                ]
                
                has_explicit_db = any(re.search(pattern, query, re.IGNORECASE) for pattern in explicit_db_patterns)
                logger.debug(f"   Has explicit DB: {has_explicit_db}")
                
                # If no explicit DB operation found, force multi-step
                if not has_explicit_db:
                    logger.debug(f"   ✅ FORCED: Requires multi-step decomposition")
                    return True
                else:
                    logger.debug(f"   ❌ BLOCKED: Has explicit DB operations, not forcing")
        
        logger.debug(f"   ❌ NO FORCED: Does not require forced decomposition")
        return False

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
        """Decompose multi-step query into individual tasks using improved handler"""
        try:
            # Use the improved multi-step handler
            improved_tasks = create_improved_decomposition(query)
            if len(improved_tasks) > 1:
                logger.info(f"Using improved decomposition: {len(improved_tasks)} tasks")
                # Ensure any visualization/summarization tasks have a preceding DB retrieval
                fixed_tasks = self._ensure_db_before_viz(improved_tasks, query)
                return fixed_tasks
            
            # Fallback to original logic if needed
            if self._requires_forced_multi_step_decomposition(query.lower()):
                return self._create_forced_decomposition(query)
            
            decompose_prompt = ChatPromptTemplate.from_template("""
            You are an expert query decomposer specializing in business intelligence queries. Break down complex queries into logical, sequential steps.
            
            Query: {query}
            
            CRITICAL RULES:
            1. For ANY query requesting visualization or summarization WITHOUT explicit data retrieval, ALWAYS start with database step
            2. Identify data dependencies - if a query needs "top X items" and then analyzes them, first get the top items, then analyze
            3. Each task must be actionable by a single specialized agent (database, email, meeting, campaign, summary, visualization)
            4. Preserve ALL specific details (emails, dates, conditions, time periods, numbers)
            5. Use {{RESULT_FROM_STEP_N}} placeholder for referencing data from previous steps
            6. For complex analytics requiring base data + analysis, ALWAYS split into: get data first, then analyze
            7. Time-based analysis often requires multi-step: identify items first, then get their time-series data
            
            ENHANCED EXAMPLES:
            
            Query: "visualize customer acquisition trends"
            Analysis: No explicit data request, needs DB first
            Output:
            TASK_1: get customer acquisition data from database
            TASK_2: create visualization showing customer acquisition trends from {{RESULT_FROM_STEP_1}}
            
            Query: "summarize sales performance"  
            Analysis: No explicit data request, needs DB first
            Output:
            TASK_1: get sales performance data from database
            TASK_2: summarize and analyze the sales performance data from {{RESULT_FROM_STEP_1}}
            
            Query: "chart customer distribution by region"
            Analysis: No explicit data request, needs DB first
            Output: 
            TASK_1: get customer data with regional information from database
            TASK_2: create chart visualization showing customer distribution by region from {{RESULT_FROM_STEP_1}}
            
            Query: "give sales trend for the last 12 months for the top 3 skus of this month"
            Analysis: This needs top 3 SKUs first, then their 12-month trends
            Output:
            TASK_1: get the top 3 skus by sales for this month
            TASK_2: get sales trend data for the last 12 months for the SKUs from {{RESULT_FROM_STEP_1}}
            
            Query: "show performance analysis of best selling products over time"
            Analysis: Need to identify best sellers, then analyze their performance over time  
            Output:
            TASK_1: get the best selling products from database
            TASK_2: analyze performance trends over time for products from {{RESULT_FROM_STEP_1}}
            
            Query: "find top 10 customers and email their details to manager@company.com"
            Output:
            TASK_1: find top 10 customers by sales or value
            TASK_2: send email to manager@company.com with subject "Top 10 Customer Details" and content "{{RESULT_FROM_STEP_1}}"
            
            Query: "get monthly sales breakdown for our top performing SKUs and create a visualization"
            Output: 
            TASK_1: get top performing SKUs from database
            TASK_2: get monthly sales breakdown for SKUs from {{RESULT_FROM_STEP_1}}
            TASK_3: create chart visualization showing the sales breakdown data from {{RESULT_FROM_STEP_2}}
            
            Query: "give me the sales trend for the last 12 months for the top 3 skus of this month and also visualize the result with a good graph"
            Output:
            TASK_1: get the top 3 skus by sales for this month
            TASK_2: get sales trend data for the last 12 months for the SKUs from {{RESULT_FROM_STEP_1}}
            TASK_3: create line chart visualization showing sales trends from {{RESULT_FROM_STEP_2}}
            
            Query: "analyze customer behavior trends and summarize insights"
            Output:
            TASK_1: get customer behavior data from database
            TASK_2: analyze trends and summarize insights from {{RESULT_FROM_STEP_1}}
            
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
                model_name="gpt-4o"
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
    
    def _create_forced_decomposition(self, query: str) -> List[str]:
        """
        Create forced two-step decomposition for visualization/summarization queries
        that don't explicitly request data retrieval
        """
        query_lower = query.lower()
        
        # Check if it's a visualization query
        visualization_keywords = ['visualize', 'chart', 'graph', 'plot', 'draw']
        is_visualization = any(keyword in query_lower for keyword in visualization_keywords)
        
        # Check if it's a summarization query  
        summarization_keywords = ['summarize', 'summary', 'analyze', 'analysis', 'explain', 'describe', 'interpret']
        is_summarization = any(keyword in query_lower for keyword in summarization_keywords)
        
        tasks = []
        
        if is_visualization and is_summarization:
            # Both visualization and summarization requested
            db_task = self._generate_db_task_from_viz_summary_query(query)
            tasks = [
                db_task,
                f"analyze and summarize the key insights and patterns from {{{{RESULT_FROM_STEP_1}}}}",  
                f"create chart visualization showing the data trends from {{{{RESULT_FROM_STEP_1}}}}"
            ]
            logger.info("Created forced 3-step decomposition: DB -> Summary -> Visualization")
        
        elif is_visualization:
            # Only visualization requested
            db_task = self._generate_db_task_from_viz_query(query)
            viz_task = f"create chart visualization showing {self._extract_viz_intent(query)} from {{{{RESULT_FROM_STEP_1}}}}"
            tasks = [db_task, viz_task]
            logger.info("Created forced 2-step decomposition: DB -> Visualization")
        
        elif is_summarization:
            # Only summarization requested
            db_task = self._generate_db_task_from_summary_query(query)
            summary_task = f"analyze and summarize {self._extract_summary_intent(query)} from {{{{RESULT_FROM_STEP_1}}}}"
            tasks = [db_task, summary_task]
            logger.info("Created forced 2-step decomposition: DB -> Summarization")
        
        else:
            # Fallback - shouldn't reach here but handle gracefully
            tasks = [query]
            logger.warning("Forced decomposition called but no viz/summary keywords found")
        
        return tasks

    def _ensure_db_before_viz(self, tasks: List[str], original_query: str) -> List[str]:
        """
        Ensure that visualization/summary tasks have proper DB retrieval.
        Smart approach: Only insert ONE DB task at the beginning if needed.
        """
        if not tasks:
            return tasks
            
        # Check if any task is viz/summary
        has_viz_or_summary = False
        has_db_task = False
        
        for task in tasks:
            task_lower = task.lower()
            if any(k in task_lower for k in ['visualize', 'visualization', 'chart', 'graph', 'plot', 'summarize', 'summary']):
                has_viz_or_summary = True
            if any(k in task_lower for k in ['get ', 'fetch ', 'retrieve ', 'query ', 'select ', 'show ', 'find ']):
                has_db_task = True
                
        # If we have viz/summary but no DB task, insert ONE at the beginning
        if has_viz_or_summary and not has_db_task:
            # Generate a single comprehensive DB task
            db_task = f"retrieve monthly sales data for the last quarter from database"
            logger.info(f"Inserting single DB task: {db_task}")
            return [db_task] + tasks
            
        return tasks
    
    def _generate_db_task_from_viz_query(self, query: str) -> str:
        """Generate simple, specific database task for visualization query"""
        query_lower = query.lower()
        
        # Keep it simple to avoid overly complex SQL generation
        if 'monthly' in query_lower and 'sales' in query_lower:
            return "show monthly sales for the last quarter"
        elif 'quarterly' in query_lower and 'sales' in query_lower:
            return "show quarterly sales totals"
        elif 'yearly' in query_lower and 'sales' in query_lower:
            return "show yearly sales data"
        elif 'customer' in query_lower and 'sales' in query_lower:
            return "show customer sales data"
        elif 'product' in query_lower or 'sku' in query_lower:
            return "show product sales data"
        elif 'sales' in query_lower:
            return "show sales data for the requested period"
        else:
            return "show relevant business data"
    
    def _generate_db_task_from_summary_query(self, query: str) -> str:
        """Generate appropriate database task for summarization query"""
        query_lower = query.lower()
        
        # Extract domain entities and analysis type
        if 'customer' in query_lower:
            if 'behavior' in query_lower:
                return "get customer behavior and activity data from database"
            elif 'performance' in query_lower:
                return "get customer performance metrics from database"
            else:
                return "get customer data from database"
        elif 'sales' in query_lower:
            if 'performance' in query_lower:
                return "get sales performance data from database" 
            elif 'trend' in query_lower:
                return "get sales trend data from database"
            else:
                return "get sales data from database"
        elif 'product' in query_lower or 'sku' in query_lower:
            return "get product performance and sales data from database"
        elif 'campaign' in query_lower:
            return "get campaign performance data from database"
        else:
            # Generic fallback
            return "get relevant business data from database for analysis"
    
    def _generate_db_task_from_viz_summary_query(self, query: str) -> str:
        """Generate database task for queries requesting both viz and summary"""
        # Use the more comprehensive approach - prioritize getting rich data
        query_lower = query.lower()
        
        if 'customer' in query_lower:
            return "get comprehensive customer data with performance metrics from database"
        elif 'sales' in query_lower:
            return "get comprehensive sales data with trends and breakdowns from database"
        elif 'product' in query_lower or 'sku' in query_lower:
            return "get comprehensive product performance and sales data from database"
        else:
            return "get comprehensive business data from database for analysis and visualization"
    
    def _extract_viz_intent(self, query: str) -> str:
        """Extract visualization intent from query for better task description"""
        query_lower = query.lower()
        
        if 'trend' in query_lower:
            return "trends and patterns"
        elif 'distribution' in query_lower:
            return "data distribution"
        elif 'comparison' in query_lower or 'compare' in query_lower:
            return "data comparison"
        elif 'breakdown' in query_lower:
            return "detailed breakdown"
        else:
            return "the data patterns"
    
    def _extract_summary_intent(self, query: str) -> str:
        """Extract summarization intent from query for better task description"""
        query_lower = query.lower()
        
        if 'behavior' in query_lower:
            return "behavior patterns and insights"
        elif 'performance' in query_lower:
            return "performance metrics and trends"
        elif 'trend' in query_lower:
            return "trends and patterns"
        else:
            return "key insights and patterns"  
    
    def _enhance_task_with_context(self, task: str, intermediate_results: Dict[str, Any]) -> str:
        """Enhance current task with data from previous steps"""
        enhanced_task = task
        
        for step_num, result in intermediate_results.items():
            placeholder = f"{{RESULT_FROM_STEP_{step_num}}}"
            if placeholder in enhanced_task:
                if isinstance(result, dict):
                    # For DB query results, extract the formatted output or JSON data
                    if "formatted_output" in result:
                        result_text = result["formatted_output"]
                    elif "json_results" in result:
                        result_text = result["json_results"]
                    elif "query_results" in result and "data" in result["query_results"]:
                        # Extract just the data portion
                        data = result["query_results"]["data"]
                        if isinstance(data, list) and len(data) > 0:
                            result_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(data)])
                        else:
                            result_text = str(data)
                    elif "email_to" in result:  
                        result_text = f"Email sent to: {result['email_to']}"
                    elif "user_id" in result and "meeting_date" in result:
                        # Meeting scheduling result - create descriptive text
                        result_text = f"Meeting scheduled with User {result['user_id']} on {result['meeting_date']}"
                    else:
                        result_text = str(result)
                else:
                    result_text = str(result)
                
                enhanced_task = enhanced_task.replace(placeholder, result_text)
        
        return enhanced_task
    
    def _classify_single_task(self, task: str) -> str:
        """Classify a single task to determine which agent should handle it"""
        # Clean the task for better classification - remove result placeholders
        clean_task = re.sub(r'\{[^}]*result[^}]*\}', '', task, flags=re.IGNORECASE)
        clean_task = re.sub(r'\{[^}]*\}', '', clean_task)  # Remove any other placeholders
        
        # If task is very long (likely contains context), focus on first part
        if len(clean_task) > 500:
            sentences = clean_task.split('.')
            clean_task = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else clean_task[:500]
        
        query_lower = clean_task.lower().strip()
        
        # ENHANCED: Strong classification patterns for forced decomposition scenarios
        # ORDER MATTERS: Check specific patterns before generic data retrieval
        
        # Strong meeting detection - FIRST priority to avoid DB misclassification
        meeting_patterns = [
            r'\b(schedule|book|arrange|set\s+up)\s+.*\b(meeting|appointment|demo|call|meet)\b',
            r'\b(schedule|book|arrange)\s+.*\bwith\s+user\s+\d+',
            r'\bmeet\s+with\s+user\s+\d+',
            r'\b(meeting|appointment|demo)\s+.*\bon\s+\d+',
            r'\bschedule.*\buser\s+\d+',
            r'^schedule\s+a\s+(meet|meeting|call|demo)',
            r'\b(book|arrange)\s+.*\b(meeting|call|demo|appointment)\b'
        ]
        
        for pattern in meeting_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return "meeting"
        
        # Strong email detection - override other classifications
        email_patterns = [
            r'\b(send|email|mail|notify).*@[\w.-]+\.\w+',
            r'\b(send|email|mail)\s+.*\s+to\s+',
            r'^send\s+email\s+to\s+',
            r'send.*email.*with\s+(subject|content)',
            r'email.*with\s+subject.*and\s+content'
        ]
        
        for pattern in email_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return "email"
        
        # ENHANCED: Strong visualization detection with better patterns
        visualization_patterns = [
            r'\b(create|generate|make|build)\s+.*\b(visualization|chart|graph|plot)',
            r'\b(visualize|chart|graph|plot|draw)\b',
            r'\b(bar\s+chart|line\s+chart|pie\s+chart|scatter\s+plot|histogram)\b',
            r'\b(create|make|generate)\s+.*\b(chart|graph|visualization)\b',
            r'\bvisualization\s+(for|showing|of|from)',
            r'\bcreate.*visualization',
            r'\b(chart|graph)\s+visualization',
            r'\b(line|bar|pie)\s+chart\s+visualization',
            r'showing.*from.*{{RESULT_FROM_STEP',
            r'create.*chart.*from.*{{RESULT_FROM_STEP',
            r'visualization.*from.*{{RESULT_FROM_STEP'
        ]
        
        for pattern in visualization_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return "visualization"
        
        # ENHANCED: Strong summarization detection - MUST BE BEFORE data retrieval check!
        summarization_patterns = [
            r'\b(summarize|summary)\b',
            r'\b(analyze|analysis)\b.*\b(data|results|findings)\b',
            r'\b(explain|describe|interpret)\s+.*\b(data|results|trends|behavior|patterns)\b',
            r'\b(analyze|analysis)\s+.*using.*{{RESULT_FROM_STEP',
            r'\bsummarize.*using.*{{RESULT_FROM_STEP',
            r'\bsummarize.*findings',
            r'\b(insights|patterns|trends)\s+from\b',
            r'\b(key\s+)?(insights|findings|observations)\b',
            r'summarize\s+(your\s+)?findings',
            r'analyze.*and\s+summarize',
            r'summarize.*the\s+(data|results|findings)'
        ]
        
        for pattern in summarization_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return "summary"
        
        # NOW check if this is a data retrieval query (after checking summary/viz)
        if self._is_data_retrieval_query(query_lower):
            return "db_query"
        
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
    
    def _is_data_retrieval_query(self, query: str) -> bool:
        """Detect if query is requesting data retrieval that should go to db_query first"""
        # First check if this is clearly an email task (more comprehensive patterns)
        email_patterns = [
            r'\b(send|email|mail|notify).*@[\w.-]+\.\w+',
            r'\b(send|email|mail)\s+.*\s+to\s+',
            r'\bemail\s+(to|the\s+list\s+to)',
            r'^send\s+email\s+to\s+',
            r'send.*email.*with\s+(subject|content)',
            r'email.*with\s+subject.*and\s+content'
        ]
        
        for pattern in email_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False
            
        # Check for meeting tasks
        if re.search(r'\b(schedule|book|meet\s+with)', query, re.IGNORECASE):
            return False
            
        # Now check for data retrieval patterns
        data_retrieval_patterns = [
            r'\b(give|show|find|get|list|display)\s+.*\b(top|best|highest|lowest|most)\s+\d+',
            r'\b(top|best|highest|lowest|most)\s+\d+\s+.*\b(skus?|products?|customers?|users?|orders?|sales|items?)',
            r'\b(give|show|find|get)\s+.*\b(skus?|products?|customers?|users?|orders?|sales)',
            r'\bfor\s+the\s+(month|week|year|day)\s+(of\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(sales|data|orders|customers)',
            r'\b(all|show|list|get)\s+.*\b(from|in|where)\b',
            r'\b(data|information|records)\s+(for|from|about)\b'
        ]
        
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in data_retrieval_patterns)
    
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
            
            CRITICAL RULE: If the task requests specific data that needs to be retrieved from database 
            (like "top 3 customers", "sales for September", "show products", etc.), it should be 
            classified as "db_query" FIRST, regardless of whether it contains summary/analysis words.
            
            Summary and visualization agents should only be used when data is already available or 
            when the primary intent is to analyze existing data.
            
            Examples:
            - "give the top 3 skus for the month september" → db_query (needs data retrieval)
            - "show all customers" → db_query (needs data retrieval)
            - "top 5 products" → db_query (needs data retrieval)
            - "summarize this customer data: [data provided]" → summary (data already available)
            - "create a chart from this data: [data provided]" → visualization (data already available)
            
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
            elif agent_type == "visualization":
                result_state = self._handle_visualization_routing(state, agent)
            elif agent_type == "summary":
                result_state = self._handle_summary_routing(state, agent)
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
                
                OrchestrationLogger.step_complete(state['current_step'], agent_type)
                
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
            
            # Special handling for visualization agent - needs structured data from previous steps
            if state["agent_type"] == "visualization":
                return self._handle_visualization_routing(state, agent)
            
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
    
    def _handle_visualization_routing(self, state: BaseAgentState, agent: BaseAgent) -> BaseAgentState:
        """Special handling for visualization agent - provides structured data from previous steps"""
        try:
            # ENHANCED: Check if this is step 1 of multi-step and no data available
            if state.get("current_step", 1) == 1 and not state.get("intermediate_results"):
                return {
                    **state,
                    "status": "failed", 
                    "error_message": "Error: Visualization requires data from a database query step. No data query was executed. Please first retrieve the data you want to visualize."
                }
            
            # Find the most recent step that has query data (usually from db_query)
            structured_data = None
            data_source_step = None
            
            # Debug: Print intermediate results structure
            logger.info(f"Available intermediate results: {list(state.get('intermediate_results', {}).keys())}")
            
            for step_key in reversed(sorted(state.get("intermediate_results", {}).keys())):
                step_result = state["intermediate_results"][step_key]
                logger.info(f"Checking step {step_key}, result type: {type(step_result)}, keys: {step_result.keys() if isinstance(step_result, dict) else 'N/A'}")
                
                # Look for database query results - check multiple possible structures
                if isinstance(step_result, dict):
                    # Check if it's directly the query results structure
                    if "data" in step_result and "columns" in step_result:
                        structured_data = step_result
                        data_source_step = step_key
                        logger.info(f"Found structured data (direct) from step {step_key} for visualization")
                        break
                    
                    # ENHANCED: Check for db_connection.py format - results.data structure
                    elif "result" in step_result and isinstance(step_result["result"], dict):
                        result_dict = step_result["result"]
                        # Check for execute_query_with_results_formatting format
                        if "results" in result_dict and isinstance(result_dict["results"], dict):
                            results = result_dict["results"]
                            if "data" in results and "summary" in results:
                                columns = results["summary"].get("columns", [])
                                data = results["data"]
                                if data and columns:
                                    structured_data = {
                                        "data": data,
                                        "columns": columns
                                    }
                                    data_source_step = step_key
                                    logger.info(f"Found structured data (db_connection format) from step {step_key} for visualization")
                                    break
                    
                    # Check if it's nested in query_results
                    elif "query_results" in step_result:
                        query_results = step_result["query_results"]
                        logger.info(f"Found query_results in step {step_key}, type: {type(query_results)}")
                        if isinstance(query_results, dict):
                            # Check for direct data/columns structure
                            if "data" in query_results and "columns" in query_results:
                                structured_data = query_results
                                data_source_step = step_key
                                logger.info(f"Found structured data (nested) from step {step_key} for visualization")
                                break
                            # Check for results.data structure
                            elif "data" in query_results and "summary" in query_results:
                                columns = query_results["summary"].get("columns", [])
                                data = query_results["data"]
                                if data and columns:
                                    structured_data = {
                                        "data": data,
                                        "columns": columns
                                    }
                                    data_source_step = step_key
                                    logger.info(f"Found structured data (query_results format) from step {step_key} for visualization")
                                    break
                    
                    # Check if there's query_data key (alternative structure)
                    elif "query_data" in step_result:
                        query_data = step_result["query_data"]
                        logger.info(f"Found query_data in step {step_key}, type: {type(query_data)}")
                        # If it's already structured data with columns and data
                        if isinstance(query_data, dict) and "data" in query_data and "columns" in query_data:
                            structured_data = query_data
                            data_source_step = step_key
                            logger.info(f"Found structured data (query_data) from step {step_key} for visualization")
                            break
                        # If query_data is a list of dictionaries, convert it
                        elif isinstance(query_data, list) and len(query_data) > 0:
                            structured_data = {
                                "data": query_data,
                                "columns": list(query_data[0].keys()) if query_data else []
                            }
                            data_source_step = step_key
                            logger.info(f"Converted query_data list to structured data from step {step_key}")
                            break
                        # If query_data is a pandas DataFrame, convert it
                        elif hasattr(query_data, 'to_dict') and hasattr(query_data, 'columns'):
                            # This is a pandas DataFrame
                            data_records = query_data.to_dict('records')
                            columns = list(query_data.columns)
                            structured_data = {
                                "data": data_records,
                                "columns": columns
                            }
                            data_source_step = step_key
                            logger.info(f"Converted pandas DataFrame to structured data from step {step_key}")
                            break
            
            # ENHANCED: Comprehensive data validation
            if not structured_data:
                logger.warning("No structured data available for visualization from previous steps")
                return {
                    **state,
                    "status": "failed",
                    "error_message": "Error: No data available for visualization. The database query step must execute successfully and return data before visualization can be performed. Please check that your data query returns results."
                }
            
            # Check if data is empty
            data_rows = structured_data.get('data', [])
            if not data_rows or len(data_rows) == 0:
                logger.warning("Data available but empty for visualization")
                return {
                    **state,
                    "status": "failed", 
                    "error_message": "Error: The database query returned no data to visualize. Please modify your query to return data or check if the requested data exists."
                }
            
            # Create a new state for visualization with the structured data
            viz_state = {**state}
            
            # Pass the entire step result to leverage visualization agent's robust data extraction
            step_result = state["intermediate_results"][data_source_step]
            viz_state["result"] = step_result
            
            # Also add the extracted structured data for additional context
            viz_state["query_data"] = structured_data
            
            row_count = len(data_rows)
            logger.info(f"Passing {row_count} rows to visualization agent from step {data_source_step}")
            result_state = agent.process(viz_state)
            
            # If visualization succeeds, save the chart file
            if result_state["status"] == "completed" and "result" in result_state:
                viz_result = result_state["result"]
                if "visualization" in viz_result and "html" in viz_result["visualization"]:
                    chart_path = self._save_visualization_chart(viz_result["visualization"])
                    viz_result["visualization"]["saved_path"] = chart_path
                    logger.info(f"Visualization saved to: {chart_path}")
            
            return result_state
                
        except Exception as e:
            logger.error(f"Visualization routing failed: {e}")
            return self._handle_agent_failure(state, "visualization")
    
    def _handle_summary_routing(self, state: BaseAgentState, agent: BaseAgent) -> BaseAgentState:
        """Special handling for summary agent - ensures data is available from previous steps"""
        try:
            # ENHANCED: Check if this is step 1 of multi-step and no data available
            if state.get("current_step", 1) == 1 and not state.get("intermediate_results"):
                return {
                    **state,
                    "status": "failed",
                    "error_message": "Error: Summarization requires data from a database query step. No data query was executed. Please first retrieve the data you want to summarize."
                }
            
            # Find the most recent step that has query data (usually from db_query)
            structured_data = None
            data_source_step = None
            
            logger.info(f"Available intermediate results for summary: {list(state.get('intermediate_results', {}).keys())}")
            
            for step_key in reversed(sorted(state.get("intermediate_results", {}).keys())):
                step_result = state["intermediate_results"][step_key]
                logger.info(f"Checking step {step_key} for summary, result type: {type(step_result)}")
                
                # Look for database query results - check multiple possible structures
                if isinstance(step_result, dict):
                    # Check if it has query data directly or nested
                    if "data" in step_result and "columns" in step_result:
                        structured_data = step_result
                        data_source_step = step_key
                        logger.info(f"Found structured data for summary from step {step_key}")
                        break
                    elif "query_results" in step_result:
                        query_results = step_result["query_results"]
                        if isinstance(query_results, dict) and "data" in query_results:
                            structured_data = query_results
                            data_source_step = step_key
                            logger.info(f"Found nested structured data for summary from step {step_key}")
                            break
                    elif "query_data" in step_result:
                        query_data = step_result["query_data"]
                        if isinstance(query_data, (list, dict)):
                            structured_data = step_result  # Pass entire result to summary agent
                            data_source_step = step_key
                            logger.info(f"Found query_data for summary from step {step_key}")
                            break
            
            # ENHANCED: Comprehensive data validation
            if not structured_data:
                logger.warning("No structured data available for summarization from previous steps")
                return {
                    **state,
                    "status": "failed",
                    "error_message": "Error: No data available for summarization. The database query step must execute successfully and return data before summarization can be performed. Please check that your data query returns results."
                }
            
            # Check if data is empty (handle different data structures)
            data_rows = []
            if "data" in structured_data:
                data_rows = structured_data.get("data", [])
            elif "query_data" in structured_data:
                query_data = structured_data["query_data"]
                if isinstance(query_data, list):
                    data_rows = query_data
                elif isinstance(query_data, dict) and "data" in query_data:
                    data_rows = query_data["data"]
            
            if not data_rows or len(data_rows) == 0:
                logger.warning("Data available but empty for summarization")
                return {
                    **state,
                    "status": "failed",
                    "error_message": "Error: The database query returned no data to summarize. Please modify your query to return data or check if the requested data exists."
                }
            
            # Create a new state for summarization with the structured data
            summary_state = {**state}
            
            # Pass the entire step result to leverage summary agent's robust data extraction
            step_result = state["intermediate_results"][data_source_step]
            summary_state["result"] = step_result
            
            row_count = len(data_rows)
            logger.info(f"Passing {row_count} rows to summary agent from step {data_source_step}")
            result_state = agent.process(summary_state)
            
            return result_state
                
        except Exception as e:
            logger.error(f"Summary routing failed: {e}")
            return self._handle_agent_failure(state, "summary")
    
    def _save_visualization_chart(self, visualization_data: Dict[str, Any]) -> str:
        """Save visualization chart to file and return file path"""
        try:
            import os
            from datetime import datetime
            
            # Create charts directory if it doesn't exist
            charts_dir = "charts"
            if not os.path.exists(charts_dir):
                os.makedirs(charts_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_type = visualization_data.get("type", "chart")
            filename = f"{chart_type}_{timestamp}.html"
            filepath = os.path.join(charts_dir, filename)
            
            # Save HTML content to file
            html_content = visualization_data.get("html", "")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return os.path.abspath(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save visualization chart: {e}")
            return "chart_save_failed.html"
    
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
    
    def _execute_with_resilience(self, initial_state: BaseAgentState) -> Dict[str, Any]:
        """Execute workflow with resilience mechanisms"""
        max_retries = 3
        timeout_seconds = 300  # 5 minutes
        
        for attempt in range(max_retries):
            try:
                # Create a copy of state for each attempt
                state_copy = {**initial_state}
                
                # Execute with timeout
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Query execution timed out")
                
                # Set timeout for non-Windows systems
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)
                
                try:
                    result = self.graph.invoke(state_copy)
                    
                    # Clear timeout
                    if hasattr(signal, 'SIGALRM'):
                        signal.alarm(0)
                    
                    return result
                    
                except TimeoutError:
                    logger.warning(f"Query execution timed out on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        return {
                            "status": "failed",
                            "error_message": "Query execution timed out",
                            "agent_type": "timeout"
                        }
                    continue
                    
            except Exception as e:
                logger.warning(f"Execution attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return {
                        "status": "failed", 
                        "error_message": f"All retry attempts failed: {str(e)}",
                        "agent_type": "error"
                    }
                
                # Brief delay before retry
                import time
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return {
            "status": "failed",
            "error_message": "Maximum retries exceeded",
            "agent_type": "error"
        }
    
    def _validate_sql_query(self, sql: str) -> Dict[str, Any]:
        """Validate SQL query before execution"""
        validation_result = {"valid": True, "issues": [], "severity": "none"}
        
        if not sql or not sql.strip():
            validation_result["valid"] = False
            validation_result["issues"].append("Empty SQL query")
            validation_result["severity"] = "critical"
            return validation_result
        
        sql_lower = sql.lower().strip()
        
        # Check for dangerous operations
        dangerous_operations = [
            "drop", "delete", "truncate", "alter", "create", "insert", "update", 
            "grant", "revoke", "exec", "execute", "sp_", "xp_"
        ]
        
        for op in dangerous_operations:
            if f" {op} " in f" {sql_lower} ":
                validation_result["valid"] = False
                validation_result["issues"].append(f"Dangerous operation detected: {op}")
                validation_result["severity"] = "critical"
        
        # Check query length
        if len(sql) > 50000:
            validation_result["valid"] = False
            validation_result["issues"].append("Query too long")
            validation_result["severity"] = "critical"
        
        # Check for basic SQL structure
        if not any(keyword in sql_lower for keyword in ["select", "with"]):
            validation_result["issues"].append("Query doesn't appear to be a valid SELECT statement")
            validation_result["severity"] = "warning"
        
        return validation_result
    
    def _validate_result_data(self, result: Dict[str, Any]) -> bool:
        """Validate query result data"""
        if not isinstance(result, dict):
            return False
        
        # Check for required fields in query results
        if "query_results" in result:
            query_results = result["query_results"]
            required_fields = ["summary", "data"]
            
            if not all(field in query_results for field in required_fields):
                logger.warning(f"Missing required fields in query results: {required_fields}")
                return False
        
        return True
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove potentially dangerous characters for logging/processing
        sanitized = re.sub(r'[<>"\']', '', text)
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000] + "..."
        
        return sanitized
    
    def add_agent(self, agent_type: str, agent: BaseAgent, keywords: List[str]):
        self.agents[agent_type] = agent
        self.classification_keywords[agent_type] = keywords
        print(f"Added new agent: {agent_type}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Production-grade query processing with comprehensive error handling"""
        # Input validation
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Empty query provided",
                "status": "failed",
                "agent_type": "unknown",
                "execution_time": 0.0
            }
        
        # Sanitize query
        query = query.strip()
        if len(query) > 10000:  # Prevent extremely long queries
            return {
                "success": False,
                "error": "Query too long (maximum 10,000 characters)",
                "status": "failed",
                "agent_type": "unknown", 
                "execution_time": 0.0
            }
        
        start_time = time.time()
        
        try:
            # Memory enrichment with error handling
            try:
                enriched_query = self.memory.enrich_query(query)
            except Exception as e:
                logger.warning(f"Memory enrichment failed: {e}. Using original query.")
                enriched_query = query

            initial_state = BaseAgentState(
                query=enriched_query,
                agent_type="",
                user_id="",
                status="",
                error_message="",
                success_message="",
                result={},
                start_time=start_time,
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
            
            logger.info(f"PROCESSING QUERY: {query}")
            if enriched_query != query:
                logger.info(f"ENRICHED QUERY: {enriched_query}")
            
            # Execute workflow with timeout and retries
            result = self._execute_with_resilience(initial_state)
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"Critical error in process_query: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"System error: {str(e)}",
                "status": "failed",
                "agent_type": "unknown",
                "execution_time": execution_time
            }
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        result["start_time"] = start_time
        result["end_time"] = end_time
        result["execution_time"] = execution_time
        
        # Get token usage summary
        token_tracker = get_token_tracker()
        token_summary = token_tracker.get_session_summary()
        TokenLogger.session_summary(
            token_summary['total_tokens'], 
            token_summary['total_cost'], 
            token_summary['total_calls']
        )
        
        if result.get("classification_confidence"):
            print(f"Classification Confidence: {result['classification_confidence']:.2f}")
        
        accuracy = self.classification_validator.get_classification_accuracy()
        if accuracy > 0:
            print(f"Recent Classification Accuracy: {accuracy:.1%}")
        
        if result["status"] == "completed":
            if result.get("is_multi_step", False):
                OrchestrationLogger.workflow_complete(
                    len(result.get("completed_steps", [])), 
                    execution_time
                )
            else:
                logger.success(f"SUCCESS: {result['success_message']}")
            
            if result.get("is_multi_step", False) and result.get("completed_steps"):
                logger.info("STEP SUMMARY:")
                for step in result["completed_steps"]:
                    logger.info(f"   Step {step['step']} ({step['agent_type']}): {step['success_message']}")
            
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
            logger.error(f"ERROR: {result['error_message']}")
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
        print("Please check your OPENAI_API_KEY configuration.")
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
        else:
            # Display successful results
            print("\n" + "="*70)
            print("QUERY RESULTS:")
            print("="*70)
            
            if result.get("is_multi_step", False):
                print(f"Multi-step workflow completed with {len(result.get('completed_steps', []))} steps")
                
                # Display each step's result
                for i, step in enumerate(result.get('completed_steps', []), 1):
                    print(f"\nStep {i} ({step.get('agent_type', 'unknown')}): {step.get('success_message', 'No message')}")
                    
                    # If it's a summary step, display the actual summary
                    if step.get('agent_type') == 'summary' and 'result' in step and 'summary' in step['result']:
                        summary_data = step['result']['summary']
                        print("\n📊 SUMMARY:")
                        if 'html' in summary_data:
                            # Convert HTML to readable text for console display
                            import re
                            html_content = summary_data['html']
                            # Remove HTML tags but preserve structure
                            text_content = html_content.replace('<ul>', '').replace('</ul>', '')
                            text_content = text_content.replace('<li>', '  • ').replace('</li>', '')
                            text_content = text_content.replace('<strong>', '').replace('</strong>', '')
                            text_content = text_content.replace('&nbsp;', ' ')
                            # Clean up extra whitespace while preserving structure
                            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                            text_content = '\n'.join(lines)
                            print(text_content)
                        print(f"\n   📊 Rows analyzed: {summary_data.get('row_count', 'N/A')}")
                        print(f"   📋 Columns: {', '.join(summary_data.get('columns', []))}")
                    
                    # Display DB query results
                    elif step.get('agent_type') == 'db_query' and 'result' in step:
                        db_result = step['result']
                        if 'formatted_output' in db_result:
                            print(f"\n📋 DATABASE RESULTS:")
                            print(db_result['formatted_output'])
                        elif 'query_data' in db_result:
                            print(f"\n📋 DATABASE RESULTS: {len(db_result.get('query_data', []))} rows returned")
            else:
                # Single-step result
                agent_type = result.get('agent_type', 'unknown')
                print(f"Single-step query processed by {agent_type} agent")
                
                if 'result' in result and result['result']:
                    final_result = result['result']
                    if 'formatted_output' in final_result:
                        print(f"\n📋 RESULTS:")
                        print(final_result['formatted_output'])
                    elif 'summary' in final_result:
                        summary_data = final_result['summary']
                        print("\n📊 SUMMARY:")
                        if 'html' in summary_data:
                            import re
                            html_content = summary_data['html']
                            # Remove HTML tags but preserve structure
                            text_content = html_content.replace('<ul>', '').replace('</ul>', '')
                            text_content = text_content.replace('<li>', '  • ').replace('</li>', '')
                            text_content = text_content.replace('<strong>', '').replace('</strong>', '')
                            text_content = text_content.replace('&nbsp;', ' ')
                            # Clean up extra whitespace while preserving structure
                            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
                            text_content = '\n'.join(lines)
                            print(text_content)
                        print(f"\n   📊 Rows analyzed: {summary_data.get('row_count', 'N/A')}")
                        print(f"   📋 Columns: {', '.join(summary_data.get('columns', []))}")
            
            print("\n" + "="*70)

if __name__ == "__main__":
    main()