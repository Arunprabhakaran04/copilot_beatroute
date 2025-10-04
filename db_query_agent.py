import re
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from base_agent import BaseAgent, BaseAgentState, DBAgentState
from sql_query_decomposer import SQLQueryDecomposer
from sql_generator_agent import SQLGeneratorAgent

logger = logging.getLogger(__name__)

class DBQueryAgent(BaseAgent):
    """
    DB Query Agent now works as an orchestrator for handling both simple and complex multi-step SQL queries.
    It coordinates between the SQL Query Decomposer and SQL Generator agents.
    """
    
    def __init__(self, llm, schema_file_path: str = "schema"):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.conversation_history = []  # Store conversation history
        
        # Initialize sub-agents
        self.decomposer = SQLQueryDecomposer(llm)
        self.sql_generator = SQLGeneratorAgent(llm, schema_file_path)
        
        logger.info("DBQueryAgent initialized as orchestrator with sub-agents:")
        logger.info("  - SQL Query Decomposer: for multi-step analysis")
        logger.info("  - SQL Generator: for individual query generation")
    
    def get_agent_type(self) -> str:
        return "db_query"
    
    def _add_to_conversation_history(self, user_query: str, assistant_response: str):
        """Add user query and assistant response to conversation history"""
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # Keep only last 10 messages to prevent context overflow
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_schema_info(self) -> str:
        """Return the schema information from SQL generator."""
        return self.sql_generator.get_schema_info()
    
    def process(self, state: BaseAgentState) -> DBAgentState:
        """
        Main orchestrator method that handles both simple and multi-step SQL queries.
        
        Flow:
        1. Analyze query complexity using SQLQueryDecomposer
        2. If single-step: generate SQL directly using SQLGeneratorAgent
        3. If multi-step: execute each step sequentially, passing results between steps
        """
        db_state = DBAgentState(**state)
        start_time = time.time()
        
        try:
            logger.info(f"DBQueryAgent processing: {state['query']}")
            
            # Step 1: Analyze query complexity
            decomposition_result = self._analyze_query_complexity(state)
            
            if not decomposition_result["analysis_successful"]:
                db_state["error_message"] = decomposition_result.get("error", "Query analysis failed")
                db_state["status"] = "failed"
                return db_state
            
            # Step 2: Log decomposition results
            logger.info(f"🔍 QUERY COMPLEXITY ANALYSIS COMPLETED:")
            logger.info(f"Original Query: {state['query']}")
            logger.info(f"Is Multi-step: {decomposition_result['is_multi_step']}")
            logger.info(f"Question Count: {decomposition_result['question_count']}")
            
            if decomposition_result["is_multi_step"]:
                logger.info(f"📝 DECOMPOSED QUESTIONS:")
                for i, question in enumerate(decomposition_result["decomposed_questions"], 1):
                    logger.info(f"  {i}. {question}")
                logger.info(f"="*80)
                return self._handle_multi_step_query(state, decomposition_result, db_state)
            else:
                logger.info(f"⚡ Single-step query - proceeding directly to SQL generation")
                logger.info(f"="*80)
                return self._handle_single_step_query(state, db_state)
                
        except Exception as e:
            logger.error(f"DBQueryAgent orchestration error: {e}")
            db_state["error_message"] = f"DB query orchestration error: {str(e)}"
            db_state["status"] = "failed"
            db_state["execution_time"] = time.time() - start_time
            return db_state
    
    def _analyze_query_complexity(self, state: BaseAgentState) -> Dict[str, Any]:
        """Analyze if the query requires multiple steps"""
        try:
            decomposer_state = BaseAgentState(**state)
            result_state = self.decomposer.process(decomposer_state)
            
            if result_state["status"] == "completed":
                return result_state["result"]
            else:
                logger.error(f"Query decomposition failed: {result_state.get('error_message')}")
                return {
                    "analysis_successful": False,
                    "error": result_state.get("error_message", "Decomposition failed")
                }
        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            return {
                "analysis_successful": False,
                "error": f"Analysis error: {str(e)}"
            }
    
    def _handle_single_step_query(self, state: BaseAgentState, db_state: DBAgentState) -> DBAgentState:
        """Handle simple single-step queries"""
        try:
            logger.info("Processing single-step query")
            
            # Prepare state for SQL generator
            generator_state = BaseAgentState(**state)
            result_state = self.sql_generator.process(generator_state)
            
            if result_state["status"] == "completed":
                # Copy results to db_state
                db_state["query_type"] = result_state["query_type"]
                db_state["sql_query"] = result_state["sql_query"]
                db_state["status"] = "completed"
                db_state["success_message"] = "Single-step SQL query generated successfully"
                db_state["result"] = result_state["result"]
                db_state["result"]["is_multi_step"] = False
                db_state["result"]["step_count"] = 1
                
                # Log the SQL query
                logger.info(f"📋 SINGLE-STEP SQL QUERY GENERATED:")
                logger.info(f"Question: {state['query']}")
                logger.info(f"SQL: {result_state['sql_query']}")
                logger.info(f"Query Type: {result_state['query_type']}")
                logger.info(f"="*80)
                
                # Also print to console for user visibility
                print(f"\n⚡ SINGLE-STEP QUERY EXECUTION:")
                print(f"Query: {state['query']}")
                print(f"Generated SQL: {result_state['sql_query']}")
                print(f"Query Type: {result_state['query_type']}")
                print("="*80)
                
                # Add to conversation history
                self._add_to_conversation_history(state["query"], result_state["sql_query"])
                
                logger.info(f"Single-step query completed: {result_state['sql_query']}")
                
            else:
                db_state["error_message"] = result_state.get("error_message", "SQL generation failed")
                db_state["status"] = "failed"
                db_state["result"] = result_state.get("result", {})
                logger.error(f"❌ Single-step query failed: {db_state['error_message']}")
            
            return db_state
            
        except Exception as e:
            logger.error(f"Single-step query error: {e}")
            db_state["error_message"] = f"Single-step query error: {str(e)}"
            db_state["status"] = "failed"
            return db_state
    
    def _handle_multi_step_query(self, state: BaseAgentState, decomposition_result: Dict[str, Any], 
                                  db_state: DBAgentState) -> DBAgentState:
        """Handle complex multi-step queries"""
        try:
            steps = decomposition_result["decomposed_questions"]
            step_count = len(steps)
            logger.info(f"Processing multi-step query with {step_count} steps")
            
            executed_steps = []
            previous_results = {}
            all_sql_queries = []
            
            # Execute each step sequentially
            for step_idx, step_question in enumerate(steps, 1):
                logger.info(f"🔄 EXECUTING MULTI-STEP QUERY - STEP {step_idx}/{step_count}")
                logger.info(f"Step {step_idx} Question: {step_question}")
                
                step_result = self._execute_single_step(
                    step_question, 
                    state.get("retrieved_sql_context", []),
                    previous_results
                )
                
                if not step_result["success"]:
                    logger.error(f"❌ Step {step_idx} failed: {step_result['error']}")
                    db_state["error_message"] = f"Step {step_idx} failed: {step_result['error']}"
                    db_state["status"] = "failed"
                    db_state["result"] = {
                        "is_multi_step": True,
                        "step_count": step_count,
                        "completed_steps": step_idx - 1,
                        "failed_step": step_idx,
                        "executed_steps": executed_steps,
                        "error_details": step_result
                    }
                    return db_state
                
                # Log the generated SQL for this step
                logger.info(f"📋 STEP {step_idx} SQL GENERATED:")
                logger.info(f"Question: {step_question}")
                logger.info(f"SQL: {step_result['sql']}")
                logger.info(f"Execution Time: {step_result.get('execution_time', 0):.3f}s")
                logger.info(f"Used Previous Results: {step_result.get('used_previous_results', False)}")
                
                # Store step results
                step_info = {
                    "step_number": step_idx,
                    "question": step_question,
                    "sql_query": step_result["sql"],
                    "explanation": step_result.get("explanation", ""),
                    "execution_time": step_result.get("execution_time", 0)
                }
                
                executed_steps.append(step_info)
                all_sql_queries.append(step_result["sql"])
                
                # Note: In a real implementation, you would execute the SQL and store results
                # For now, we simulate this by storing the SQL query as the "result"
                previous_results[f"step_{step_idx}"] = {
                    "sql": step_result["sql"],
                    "question": step_question,
                    "step_number": step_idx
                }
                
                logger.info(f"✅ Step {step_idx} completed successfully")
                logger.info("" + "-" * 80)
            
            # All steps completed successfully
            final_sql = all_sql_queries[-1]  # Last query is usually the final answer
            
            # Log comprehensive multi-step completion summary
            logger.info(f"🎉 MULTI-STEP QUERY COMPLETED SUCCESSFULLY")
            logger.info(f"Original Question: {state['query']}")
            logger.info(f"Total Steps: {step_count}")
            logger.info(f"="*80)
            
            # Log all intermediate queries
            logger.info(f"📚 ALL GENERATED SQL QUERIES:")
            for i, (question, sql_query) in enumerate(zip(steps, all_sql_queries), 1):
                logger.info(f"Step {i} Question: {question}")
                logger.info(f"Step {i} SQL: {sql_query}")
                logger.info(f"-" * 60)
            
            # Log final SQL prominently
            logger.info(f"🏆 FINAL SQL QUERY (Step {step_count}):")
            logger.info(f"SQL: {final_sql}")
            logger.info(f"="*80)
            
            # Also print to console for user visibility
            print(f"\n🎯 MULTI-STEP QUERY EXECUTION SUMMARY:")
            print(f"Query: {state['query']}")
            print(f"Steps Executed: {step_count}")
            print(f"\n📋 Generated SQL Queries:")
            for i, (question, sql_query) in enumerate(zip(steps, all_sql_queries), 1):
                print(f"Step {i}: {question}")
                print(f"SQL: {sql_query}")
                print("-" * 60)
            print(f"\n🏆 Final SQL Query:")
            print(f"{final_sql}")
            print("="*80)
            
            db_state["query_type"] = "SELECT"  # Multi-step queries are typically SELECT operations
            db_state["sql_query"] = final_sql
            db_state["status"] = "completed"
            db_state["success_message"] = f"Multi-step query completed successfully ({step_count} steps)"
            
            db_state["result"] = {
                "is_multi_step": True,
                "step_count": step_count,
                "completed_steps": step_count,
                "executed_steps": executed_steps,
                "final_sql": final_sql,
                "all_sql_queries": all_sql_queries,
                "original_question": state["query"],
                "decomposed_questions": steps,
                "format": "multi_step_cube_js_api"
            }
            
            # Add to conversation history
            summary = f"Multi-step query with {step_count} steps completed. Final SQL: {final_sql}"
            self._add_to_conversation_history(state["query"], summary)
            
            logger.info(f"Multi-step query completed successfully: {step_count} steps")
            return db_state
            
        except Exception as e:
            logger.error(f"Multi-step query error: {e}")
            db_state["error_message"] = f"Multi-step query error: {str(e)}"
            db_state["status"] = "failed"
            db_state["result"] = {
                "is_multi_step": True,
                "error": str(e),
                "original_question": state["query"]
            }
            return db_state
    
    def _execute_single_step(self, question: str, similar_sqls: List[str], 
                           previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in a multi-step query"""
        try:
            step_start_time = time.time()
            
            # Generate SQL for this step
            result = self.sql_generator.generate_sql(
                question=question,
                similar_sqls=similar_sqls,
                previous_results=previous_results
            )
            
            step_execution_time = time.time() - step_start_time
            result["execution_time"] = step_execution_time
            
            return result
            
        except Exception as e:
            logger.error(f"Step execution error: {e}")
            return {
                "success": False,
                "error": f"Step execution error: {str(e)}",
                "type": "step_execution_error"
            }
    
    def get_multi_step_capabilities(self) -> Dict[str, Any]:
        """Return information about multi-step query capabilities"""
        return {
            "supports_multi_step": True,
            "max_steps": 5,  # Reasonable limit to prevent infinite loops
            "supported_patterns": [
                "top_n_then_details",  # "Show trend for top 3 SKUs"
                "filter_then_analyze",  # "Find degrowth customers, then their purchases"
                "temporal_comparison",  # "Last month vs this month"
                "sequential_filtering"  # "Customers who ordered both periods"
            ],
            "decomposer_agent": "SQLQueryDecomposer",
            "generator_agent": "SQLGeneratorAgent"
        }
    
    def test_multi_step_examples(self) -> List[Dict[str, Any]]:
        """Test the multi-step functionality with example queries"""
        test_queries = [
            "Show me the sales trend of the top 3 SKUs in the last month",
            "Find customers who placed orders both last month and this month",
            "Show me customers who had degrowth in the last 3 months, and their top 3 purchased SKUs",
            "Get the top 5 SKUs by sales this month"  # Should be single-step
        ]
        
        results = []
        for query in test_queries:
            try:
                test_state = BaseAgentState(
                    query=query,
                    agent_type="db_query",
                    user_id="test_user",
                    status="",
                    error_message="",
                    success_message="",
                    result={},
                    start_time=time.time(),
                    end_time=0.0,
                    execution_time=0.0,
                    classification_confidence=None,
                    redirect_count=0,
                    original_query=query,
                    remaining_tasks=[],
                    completed_steps=[],
                    current_step=0,
                    is_multi_step=False,
                    intermediate_results={}
                )
                
                result_state = self.process(test_state)
                
                results.append({
                    "query": query,
                    "is_multi_step": result_state["result"].get("is_multi_step", False),
                    "step_count": result_state["result"].get("step_count", 1),
                    "status": result_state["status"],
                    "final_sql": result_state.get("sql_query", ""),
                    "success": result_state["status"] == "completed"
                })
                
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation history"""
        return {
            "total_conversations": len(self.conversation_history) // 2,  # User + Assistant pairs
            "recent_queries": [
                msg["content"] for msg in self.conversation_history[-10:] 
                if msg["role"] == "user"
            ],
            "schema_info_available": bool(self.get_schema_info()),
            "sub_agents_initialized": {
                "decomposer": self.decomposer is not None,
                "sql_generator": self.sql_generator is not None
            }
        }