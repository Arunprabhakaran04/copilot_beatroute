import re
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from langchain.prompts import ChatPromptTemplate
from base_agent import BaseAgent, BaseAgentState, DBAgentState
from sql_query_decomposer import SQLQueryDecomposer
from sql_generator_agent import SQLGeneratorAgent
from sql_exception_agent import SQLExceptionAgent
from db_connection import get_database_connection, execute_sql
from token_tracker import track_llm_call, get_token_tracker

class DBQueryAgent(BaseAgent):
    """
    DB Query Agent now works as an orchestrator for handling both simple and complex multi-step SQL queries.
    It coordinates between the SQL Query Decomposer and SQL Generator agents.
    """
    
    def __init__(self, llm, schema_file_path: str = "schema"):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.conversation_history = []  
        
        self.decomposer = SQLQueryDecomposer(llm)
        self.sql_generator = SQLGeneratorAgent(llm, schema_file_path)
        self.exception_agent = SQLExceptionAgent(llm, schema_file_path, max_iterations=5)
        
        # Add improved SQL generator
        try:
            from improved_sql_generator import ImprovedSQLGenerator
            self.improved_sql_generator = ImprovedSQLGenerator(llm, schema_file_path)
            logger.info("  - Improved SQL Generator: ENABLED (for better accuracy)")
        except ImportError:
            logger.warning("  - Improved SQL Generator: NOT AVAILABLE (using fallback)")
            self.improved_sql_generator = None
        
        try:
            from sql_retriever_agent import SQLRetrieverAgent
            self.sql_retriever = SQLRetrieverAgent(llm, "embeddings.pkl")
            logger.info("DBQueryAgent initialized as orchestrator with sub-agents:")
            logger.info("  - SQL Query Decomposer: for multi-step analysis")
            logger.info("  - SQL Generator: for individual query generation")
            logger.info("  - SQL Retriever: for step-specific contecxt retrieval")
            logger.info("  - SQL Exception Agent: for error analysis and fixing")
        except Exception as e:
            logger.warning(f"Could not initialize SQL retriever: {e}")
            self.sql_retriever = None
            logger.info("DBQueryAgent initialized as orchestrator with sub-agents:")
            logger.info("  - SQL Query Decomposer: for multi-step analysis")
            logger.info("  - SQL Generator: for individual query generation")
            logger.info("  - SQL Retriever: NOT AVAILABLE (will use fallback)")
            logger.info("  - SQL Exception Agent: for error analysis and fixing")
    
    def get_agent_type(self) -> str:
        return "db_query"
    
    def _add_to_conversation_history(self, user_query: str, assistant_response: str):
        """Add user query and assistant response to conversation history"""
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
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
            try:
                from clean_logging import AgentLogger
                AgentLogger.query_start("db_query", state['query'])
            except ImportError:
                logger.info(f"DB_QUERY | Processing: {state['query']}")
            
            # Step 1: Input validation and sanitization
            if not state.get('query') or not state['query'].strip():
                db_state["error_message"] = "Empty or invalid query provided"
                db_state["status"] = "failed"
                return db_state
            
            # Sanitize query
            sanitized_query = re.sub(r'[^\w\s@.,?!\-:()&]', '', state['query'])
            if len(sanitized_query) != len(state['query']):
                logger.warning("Query contained potentially unsafe characters - sanitized")
                state['query'] = sanitized_query
            
            # ✅ IMPROVED: Check if agent-aware decomposer already determined single-step
            # The AgentAwareDecomposer runs BEFORE routing to db_query agent
            # If it says single-step with high confidence, trust it and skip redundant decomposition
            is_multi_step_flag = state.get("is_multi_step")
            confidence = state.get("classification_confidence")
            
            logger.info(f"🔍 DECOMPOSITION CHECK:")
            logger.info(f"   is_multi_step flag: {is_multi_step_flag}")
            logger.info(f"   confidence: {confidence}")
            logger.info(f"   State has is_multi_step: {'is_multi_step' in state}")
            
            is_confirmed_single_step = (
                is_multi_step_flag == False and  # Explicitly check == False (not just falsy)
                confidence is not None and
                confidence >= 0.85
            )
            
            if is_confirmed_single_step:
                logger.info("✅ Agent-aware decomposer confirmed single-step (skipping redundant analysis)")
                logger.info(f"   Confidence: {confidence:.2f}")
                
                # Treat as single-step directly
                result = self._handle_single_step_query(state, db_state)
                
                # Add timing
                execution_time = time.time() - start_time
                result["execution_time"] = execution_time
                
                try:
                    from clean_logging import AgentLogger
                    AgentLogger.query_complete("db_query", execution_time)
                except ImportError:
                    logger.success(f"DB_QUERY | Completed in {execution_time:.2f}s")
                
                return result
            
            # Step 2: Analyze query complexity with error handling (only if not already analyzed)
            try:
                decomposition_result = self._analyze_query_complexity(state)
                
                if not decomposition_result["analysis_successful"]:
                    db_state["error_message"] = decomposition_result.get("error", "Query analysis failed")
                    db_state["status"] = "failed"
                    return db_state
            except Exception as e:
                logger.error(f"Query analysis failed: {e}")
                db_state["error_message"] = f"Query analysis error: {str(e)}"
                db_state["status"] = "failed"
                return db_state
            
            # Step 2: Log decomposition results
            logger.info(f"QUERY COMPLEXITY ANALYSIS COMPLETED:")
            logger.info(f"Original Query: {state['query']}")
            logger.info(f"Is Multi-step: {decomposition_result['is_multi_step']}")
            logger.info(f"Question Count: {decomposition_result['question_count']}")
            
            if decomposition_result["is_multi_step"]:
                logger.info(f"DECOMPOSED QUESTIONS:")
                for i, question in enumerate(decomposition_result["decomposed_questions"], 1):
                    logger.info(f"  {i}. {question}")
                logger.info(f"="*80)
                result = self._handle_multi_step_query(state, decomposition_result, db_state)
                
                # Add timing
                execution_time = time.time() - start_time
                result["execution_time"] = execution_time
                
                try:
                    from clean_logging import AgentLogger
                    AgentLogger.query_complete("db_query", execution_time)
                except ImportError:
                    logger.success(f"DB_QUERY | Completed in {execution_time:.2f}s")
                
                return result
            else:
                logger.info(f"Single-step query - proceeding directly to SQL generation")
                logger.info(f"="*80)
                result = self._handle_single_step_query(state, db_state)
                
                # Add timing
                execution_time = time.time() - start_time
                result["execution_time"] = execution_time
                
                try:
                    from clean_logging import AgentLogger
                    AgentLogger.query_complete("db_query", execution_time)
                except ImportError:
                    logger.success(f"DB_QUERY | Completed in {execution_time:.2f}s")
                
                return result
                
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
            
            # ✅ FIX: Retrieve SQL examples for single-step queries too!
            # Previously only multi-step queries retrieved examples, causing single-step to generate wrong SQL
            logger.info("🔍 Retrieving SQL examples for single-step query...")
            retrieved_sqls = self._retrieve_step_specific_sqls(state["query"])
            
            # Check if this is part of a multi-step workflow (has intermediate_results from previous steps)
            intermediate_results = state.get("intermediate_results", {})
            if intermediate_results:
                logger.info(f" Single-step query has {len(intermediate_results)} previous step(s) - will pass to generator")
                # Explicitly add to generator state to ensure it's passed through
                generator_state = BaseAgentState(**state)
                generator_state["previous_step_results"] = intermediate_results
                generator_state["retrieved_sql_context"] = retrieved_sqls  # Add SQL examples
                logger.info(f"   Added intermediate_results as previous_step_results to generator state")
            else:
                # Prepare state for SQL generator
                generator_state = BaseAgentState(**state)
                generator_state["retrieved_sql_context"] = retrieved_sqls  # Add SQL examples
            
            logger.info(f"📚 Passing {len(retrieved_sqls)} SQL examples to generator")
            
            result_state = self.sql_generator.process(generator_state)
            
            if result_state["status"] == "completed":
                # Try to execute the SQL and handle any errors
                execution_result = self._execute_sql_with_error_handling(
                    sql_query=result_state["sql_query"],
                    original_question=state["query"],
                    similar_sqls=generator_state.get("retrieved_sql_context", [])
                )
                
                if execution_result["success"]:
                    # Convert query results to pandas DataFrame
                    query_data_df = self._convert_to_dataframe(execution_result.get("query_results", {}))
                    
                    db_state["query_type"] = result_state["query_type"]
                    db_state["sql_query"] = execution_result["final_sql"]
                    db_state["status"] = "completed"
                    db_state["success_message"] = execution_result["message"]
                    db_state["result"] = result_state["result"]
                    db_state["result"]["is_multi_step"] = False
                    db_state["result"]["step_count"] = 1
                    db_state["result"]["exception_handling"] = execution_result.get("exception_summary")
                    # Add database execution results
                    db_state["result"]["query_results"] = execution_result.get("query_results", {})
                    db_state["result"]["query_data"] = query_data_df  # Add DataFrame for summary agent
                    db_state["result"]["formatted_output"] = execution_result.get("formatted_output", "")
                    db_state["result"]["json_results"] = execution_result.get("json_results", "[]")
                    db_state["query_data"] = execution_result.get("query_results", {}).get("data", [])
                else:
                    # SQL execution failed even after exception handling
                    db_state["error_message"] = execution_result["error"]
                    db_state["status"] = "failed"
                    db_state["result"] = execution_result
                    logger.error(f"Single-step query failed after exception handling: {execution_result['error']}")
                    return db_state
                
                logger.info(f" SINGLE-STEP SQL QUERY GENERATED:")
                logger.info(f"Question: {state['query']}")
                logger.info(f"SQL: {result_state['sql_query']}")
                logger.info(f"Query Type: {result_state['query_type']}")
                logger.info(f"="*80)
                
                print(f"\n SINGLE-STEP QUERY EXECUTION:")
                print(f"Query: {state['query']}")
                print(f"Generated SQL: {result_state['sql_query']}")
                print(f"Query Type: {result_state['query_type']}")
                
                # Show database results
                if execution_result.get("query_results"):
                    results_summary = execution_result["query_results"]["summary"]
                    print(f"\nDATABASE RESULTS:")
                    print(f"Rows Retrieved: {results_summary['total_rows']}")
                    print(f"Columns: {results_summary['columns']}")
                    print(f"Execution Time: {results_summary['execution_time']}")
                    
                    if execution_result.get("formatted_output"):
                        print(f"\n SAMPLE RESULTS:")
                        print(execution_result["formatted_output"])
                
                print("="*80)
                
                self._add_to_conversation_history(state["query"], result_state["sql_query"])
                
                logger.info(f"Single-step query completed: {result_state['sql_query']}")
                
            else:
                db_state["error_message"] = result_state.get("error_message", "SQL generation failed")
                db_state["status"] = "failed"
                db_state["result"] = result_state.get("result", {})
                logger.error(f"Single-step query failed: {db_state['error_message']}")
            
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
            
            for step_idx, step_question in enumerate(steps, 1):
                logger.info(f" EXECUTING MULTI-STEP QUERY - STEP {step_idx}/{step_count}")
                logger.info(f"Step {step_idx} Question: {step_question}")
                
                # Retrieve SQL examples specific to this step
                step_specific_sqls = self._retrieve_step_specific_sqls(step_question)
                
                logger.info(f" Retrieved {len(step_specific_sqls)} SQL examples for step {step_idx}")
                if step_specific_sqls:
                    for i, sql_example in enumerate(step_specific_sqls[:3], 1): 
                        if isinstance(sql_example, dict) and "question" in sql_example:
                            logger.info(f"  Example {i}: {sql_example['question'][:60]}...")
                        else:
                            logger.info(f"  Example {i}: {str(sql_example)[:60]}...")
                
                step_result = self._execute_single_step(
                    step_question, 
                    step_specific_sqls,  
                    previous_results,
                    state
                )
                
                if not step_result["success"]:
                    logger.error(f"Step {step_idx} failed: {step_result['error']}")
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
                
                logger.info(f"STEP {step_idx} SQL GENERATED:")
                logger.info(f"Question: {step_question}")
                logger.info(f"SQL: {step_result['sql']}")
                logger.info(f"Context Examples Used: {step_result.get('step_specific_context_count', 0)}")
                logger.info(f"Context Quality: {step_result.get('context_quality', 'unknown')}")
                logger.info(f"Execution Time: {step_result.get('execution_time', 0):.3f}s")
                logger.info(f"Used Previous Results: {step_result.get('used_previous_results', False)}")
                
                # Store enhanced step results with retrieval metadata
                step_info = {
                    "step_number": step_idx,
                    "question": step_question,
                    "sql_query": step_result["sql"],
                    "explanation": step_result.get("explanation", ""),
                    "execution_time": step_result.get("execution_time", 0),
                    "context_examples_count": step_result.get("step_specific_context_count", 0),
                    "context_quality": step_result.get("context_quality", "unknown"),
                    "retrieved_examples": [
                        ex.get("question", "") if isinstance(ex, dict) else str(ex)[:60] 
                        for ex in step_specific_sqls[:3]
                    ]
                }
                
                executed_steps.append(step_info)
                all_sql_queries.append(step_result["sql"])
                
                # Store results for next step with enhanced metadata
                previous_results[f"step_{step_idx}"] = {
                    "sql": step_result["sql"],
                    "question": step_question,
                    "step_number": step_idx,
                    "context_count": len(step_specific_sqls),
                    "context_quality": step_result.get("context_quality", "unknown"),
                    "retrieval_quality": "high" if len(step_specific_sqls) >= 3 else "low"
                }
                
                logger.info(f"Step {step_idx} completed successfully")
                logger.info("" + "-" * 80)
            
            final_sql = all_sql_queries[-1]  # Last query is usually the final answer
            
            logger.info(f" MULTI-STEP QUERY COMPLETED SUCCESSFULLY")
            logger.info(f"Original Question: {state['query']}")
            logger.info(f"Total Steps: {step_count}")
            logger.info(f"="*80)
            
            logger.info(f" ALL GENERATED SQL QUERIES:")
            for i, (question, sql_query) in enumerate(zip(steps, all_sql_queries), 1):
                logger.info(f"Step {i} Question: {question}")
                logger.info(f"Step {i} SQL: {sql_query}")
                logger.info(f"-" * 60)
            
            logger.info(f" FINAL SQL QUERY (Step {step_count}):")
            logger.info(f"SQL: {final_sql}")
            logger.info(f"="*80)
            
            print(f"\n MULTI-STEP QUERY EXECUTION SUMMARY:")
            print(f"Query: {state['query']}")
            print(f"Steps Executed: {step_count}")
            print(f"\n Generated SQL Queries with Context:")
            for i, (question, sql_query, step_info) in enumerate(zip(steps, all_sql_queries, executed_steps), 1):
                print(f"Step {i}: {question}")
                print(f"Context Examples: {step_info.get('context_examples_count', 0)} | Quality: {step_info.get('context_quality', 'unknown')}")
                print(f"SQL: {sql_query}")
                
                # Show database results for each step
                if step_info.get('query_results'):
                    step_results = step_info['query_results']
                    if 'summary' in step_results:
                        print(f"Results: {step_results['summary']['total_rows']} rows | {step_results['summary']['execution_time']}")
                
                if step_info.get('retrieved_examples'):
                    print(f"Sample Context: {step_info['retrieved_examples'][0][:50]}..." if step_info['retrieved_examples'] else "No examples")
                print("-" * 60)
            
            print(f"\n Final SQL Query:")
            print(f"{final_sql}")
            
            final_step_results = executed_steps[-1] if executed_steps else {}
            
            if final_step_results.get("query_results"):
                final_summary = final_step_results["query_results"]["summary"]
                print(f"\n FINAL QUERY RESULTS:")
                print(f"Total Rows: {final_summary['total_rows']}")
                print(f"Columns: {final_summary['columns']}")
                print(f"Execution Time: {final_summary['execution_time']}")
                
                if final_step_results.get("formatted_output"):
                    print(f"\nSAMPLE RESULTS:")
                    print(final_step_results["formatted_output"])
            
            print("="*80)
            
            # Convert final results to DataFrame for summary agent
            final_query_results = final_step_results.get("query_results", {})
            query_data_df = self._convert_to_dataframe(final_query_results)
            
            db_state["query_type"] = "SELECT"  
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
                "format": "multi_step_cube_js_api_with_dataframe",
                # Add final database results
                "query_results": final_query_results,
                "query_data": query_data_df,  # Add DataFrame for summary agent
                "formatted_output": final_step_results.get("formatted_output", ""),
                "json_results": final_step_results.get("json_results", "[]")
            }
            
            # Add final query data for easy access by other agents
            db_state["query_data"] = final_query_results.get("data", [])
            
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
    
    def _retrieve_step_specific_sqls(self, step_question: str) -> List[str]:
        """Retrieve SQL queries specific to the current step question"""
        try:
            logger.info(f" RETRIEVING STEP-SPECIFIC SQL CONTEXT:")
            logger.info(f"Step Question: {step_question}")
            
            if not self.sql_retriever:
                logger.warning("SQL retriever not available, using empty context")
                return []
            
            # Create a temporary state for the step question
            from base_agent import BaseAgentState
            step_state = BaseAgentState(
                query=step_question,
                agent_type="sql_retriever",
                user_id="multi_step_user",
                status="",
                error_message="",
                success_message="",
                result={},
                start_time=time.time(),
                end_time=0.0,
                execution_time=0.0,
                classification_confidence=None,
                redirect_count=0,
                original_query=step_question,
                remaining_tasks=[],
                completed_steps=[],
                current_step=0,
                is_multi_step=False,
                intermediate_results={}
            )
            
            # Retrieve step-specific SQL examples
            result_state = self.sql_retriever.process(step_state)
            
            if result_state["status"] == "completed":
                similar_sqls = result_state["result"].get("similar_sqls", [])
                logger.info(f"Retrieved {len(similar_sqls)} step-specific SQL examples")
                return similar_sqls[:10]  # Limit to top 10 for efficiency
            else:
                logger.warning(f"Step-specific SQL retrieval failed for: {step_question}")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving step-specific SQLs: {e}")
            return []
    
    def _execute_single_step(self, question: str, similar_sqls: List[str], 
                           previous_results: Dict[str, Any], state: BaseAgentState = None) -> Dict[str, Any]:
        """Execute a single step in a multi-step query with enhanced context tracking and error handling"""
        try:
            step_start_time = time.time()
            
            logger.info(f" STEP-SPECIFIC SQL GENERATION:")
            logger.info(f"Question: {question}")
            logger.info(f"Available context examples: {len(similar_sqls)}")
            logger.info(f"Previous results available: {len(previous_results)}")
            
            # Get conversation history from state (if available)
            conversation_history = state.get("conversation_history", []) if state else []
            
            # Generate SQL for this step - use improved generator if available
            if hasattr(self, 'improved_sql_generator') and self.improved_sql_generator:
                original_query = state.get("original_query", question)
                entity_info = state.get("entity_info", None)  # Get entity info from state
                
                # ✅ FIXED: Better entity_info validation and fallback
                if entity_info is None:
                    # Try alternative state keys where entity info might be stored
                    entity_info = state.get("verified_entities", None)
                    if entity_info:
                        logger.info(f"✅ Found entity_info under 'verified_entities' key")
                    else:
                        logger.warning(f"⚠️ No entity_info in state. Available keys: {list(state.keys())}")
                        # Create empty entity_info structure to prevent downstream errors
                        entity_info = {
                            "entities": [],
                            "entity_types": [],
                            "entity_mapping": {},
                            "verified_entities": {}
                        }
                else:
                    logger.info(f"✅ Entity info available: {len(entity_info.get('entities', []))} entities")
                
                generation_result = self.improved_sql_generator.generate_sql(
                    question=question,
                    similar_sqls=similar_sqls,
                    previous_results=previous_results,
                    original_query=original_query,
                    entity_info=entity_info,  # Pass entity info to SQL generator
                    conversation_history=conversation_history  # Pass conversation history
                )
                logger.info(f"Using improved SQL generator (method: {generation_result.get('method', 'unknown')})")
            else:
                entity_info = state.get("entity_info", None)
                
                # ✅ FIXED: Same validation for standard generator
                if entity_info is None:
                    entity_info = state.get("verified_entities", None)
                    if entity_info:
                        logger.info(f"✅ Found entity_info under 'verified_entities' key (standard gen)")
                    else:
                        logger.warning(f"⚠️ No entity_info in state (standard gen)")
                        entity_info = {
                            "entities": [],
                            "entity_types": [],
                            "entity_mapping": {},
                            "verified_entities": {}
                        }
                
                generation_result = self.sql_generator.generate_sql(
                    question=question,
                    similar_sqls=similar_sqls,
                    previous_results=previous_results,
                    entity_info=entity_info  # Pass entity info here too
                )
                logger.info("Using standard SQL generator")
            
            if not generation_result["success"]:
                return {
                    "success": False,
                    "error": generation_result["error"],
                    "type": "sql_generation_error"
                }
            
            # Log the generated SQL
            try:
                from clean_logging import SQLLogger
                SQLLogger.generated_sql(generation_result["sql"])
            except ImportError:
                logger.info(f"Generated SQL:\n{generation_result['sql']}")
            
            # Execute the generated SQL with error handling
            execution_result = self._execute_sql_with_error_handling(
                sql_query=generation_result["sql"],
                original_question=question,
                similar_sqls=similar_sqls
            )
            
            step_execution_time = time.time() - step_start_time
            
            # Log step completion with timing
            try:
                from clean_logging import AgentLogger
                AgentLogger.query_complete("sql_generation_step", step_execution_time)
            except ImportError:
                logger.success(f"SQL_GENERATION_STEP | Completed in {step_execution_time:.2f}s")
            
            if execution_result["success"]:
                return {
                    "success": True,
                    "sql": execution_result["final_sql"],
                    "explanation": generation_result.get("explanation", ""),
                    "execution_time": step_execution_time,
                    "step_specific_context_count": len(similar_sqls),
                    "context_quality": "high" if len(similar_sqls) >= 3 else "medium" if len(similar_sqls) >= 1 else "low",
                    "used_previous_results": previous_results is not None and len(previous_results) > 0,
                    "exception_handling": execution_result.get("exception_summary"),
                    "sql_attempts": execution_result.get("attempts", 1),
                    "original_generated_sql": generation_result["sql"],
                    # Add database results
                    "query_results": execution_result.get("query_results", {}),
                    "formatted_output": execution_result.get("formatted_output", ""),
                    "json_results": execution_result.get("json_results", "[]")
                }
            else:
                return {
                    "success": False,
                    "error": execution_result["error"],
                    "type": "sql_execution_error",
                    "execution_time": step_execution_time,
                    "failed_sql": generation_result["sql"],
                    "exception_details": execution_result.get("exception_details", {}),
                    "attempts": execution_result.get("attempts", 1)
                }
            
        except Exception as e:
            logger.error(f"Step execution error: {e}")
            return {
                "success": False,
                "error": f"Step execution error: {str(e)}",
                "type": "step_execution_error"
            }
    
    def _execute_sql_in_database(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute SQL query in the actual Cube.js database and return results.
        
        Returns:
            Dict with success status, results, and any error information
        """
        try:
            logger.info(f"EXECUTING SQL IN CUBE.JS DATABASE:")
            logger.info(f"SQL: {sql_query}")
            
            # Execute query using database connection
            db_result = execute_sql(sql_query)
            
            if db_result['success']:
                logger.info(f" DATABASE EXECUTION SUCCESSFUL:")
                logger.info(f"Rows returned: {db_result['results']['summary']['total_rows']}")
                logger.info(f"Execution time: {db_result['results']['summary']['execution_time']}")
                logger.info(f"Columns: {db_result['results']['summary']['columns']}")
                
                return {
                    "success": True,
                    "message": "SQL executed successfully in database",
                    "executed_sql": sql_query,
                    "results": db_result['results'],
                    "formatted_output": db_result['formatted_output'],
                    "json_results": db_result['json_results'],
                    "execution_metadata": db_result['execution_metadata']
                }
            else:
                logger.error(f"DATABASE EXECUTION FAILED:")
                logger.error(f"Error: {db_result['error']}")
                
                return {
                    "success": False,
                    "error": db_result['error'],
                    "failed_sql": sql_query,
                    "error_type": "database_execution_error",
                    "execution_metadata": db_result.get('execution_metadata', {})
                }
                
        except Exception as e:
            logger.error(f"SQL EXECUTION ERROR: {e}")
            return {
                "success": False,
                "error": f"Database execution error: {str(e)}",
                "failed_sql": sql_query,
                "error_type": "execution_system_error"
            }
    
    def _execute_sql_with_error_handling(self, sql_query: str, original_question: str, 
                                       similar_sqls: List[str] = None) -> Dict[str, Any]:
        """
        Execute SQL with comprehensive error handling and automatic fixing.
        
        This method:
        1. Simulates SQL execution 
        2. If an error occurs, uses the Exception Agent to analyze and fix
        3. Retries execution with the corrected SQL
        4. Returns the final result or failure after max attempts
        """
        try:
            logger.info(f"EXECUTING SQL WITH ERROR HANDLING:")
            logger.info(f"Original Question: {original_question}")
            logger.info(f"SQL to Execute: {sql_query}")
            
            execution_result = self._execute_sql_in_database(sql_query)
            
            if execution_result["success"]:
                logger.info(f"SQL EXECUTED SUCCESSFULLY ON FIRST ATTEMPT")
                logger.info(f"Retrieved {execution_result.get('results', {}).get('summary', {}).get('total_rows', 0)} rows")
                
                return {
                    "success": True,
                    "final_sql": sql_query,
                    "message": "SQL executed successfully on first attempt",
                    "attempts": 1,
                    "exception_summary": None,
                    "query_results": execution_result.get("results", {}),
                    "formatted_output": execution_result.get("formatted_output", ""),
                    "json_results": execution_result.get("json_results", "[]")
                }
            
            logger.info(f" SQL EXECUTION FAILED - INVOKING EXCEPTION AGENT")
            logger.info(f"Error: {execution_result['error']}")
            
            fix_result = self.exception_agent.iterative_fix_sql(
                original_question=original_question,
                failed_sql=sql_query,
                error_message=execution_result["error"],
                similar_sqls=similar_sqls or []
            )
            
            if fix_result["success"]:
                corrected_sql = fix_result["final_sql"]
                logger.info(f" RETRYING WITH CORRECTED SQL:")
                logger.info(f"Corrected SQL: {corrected_sql}")
                
                retry_execution = self._execute_sql_in_database(corrected_sql)
                
                if retry_execution["success"]:
                    logger.info(f" SQL EXECUTION SUCCESSFUL AFTER EXCEPTION HANDLING")
                    logger.info(f" Retrieved {retry_execution.get('results', {}).get('summary', {}).get('total_rows', 0)} rows")
                    
                    return {
                        "success": True,
                        "final_sql": corrected_sql,
                        "message": f"SQL fixed and executed successfully after {fix_result['total_iterations']} iterations",
                        "attempts": fix_result["total_iterations"] + 1,
                        "query_results": retry_execution.get("results", {}),
                        "formatted_output": retry_execution.get("formatted_output", ""),
                        "json_results": retry_execution.get("json_results", "[]"),
                        "exception_summary": {
                            "original_error": execution_result["error"],
                            "fix_iterations": fix_result["total_iterations"],
                            "fix_type": fix_result["fix_summary"]["fix_type"],
                            "root_cause": fix_result["fix_summary"]["root_cause"],
                            "learning_points": fix_result["fix_summary"]["learning_points"]
                        },
                        "original_sql": sql_query,
                        "exception_details": fix_result
                    }
                else:
                    # Even corrected SQL failed
                    logger.error(f" CORRECTED SQL ALSO FAILED:")
                    logger.error(f"New Error: {retry_execution['error']}")
                    
                    return {
                        "success": False,
                        "error": f"Corrected SQL also failed: {retry_execution['error']}",
                        "final_sql": corrected_sql,
                        "attempts": fix_result["total_iterations"] + 1,
                        "exception_summary": {
                            "original_error": execution_result["error"],
                            "fix_attempts": fix_result["total_iterations"],
                            "final_error": retry_execution["error"],
                            "status": "fix_attempted_but_still_failing"
                        },
                        "original_sql": sql_query,
                        "exception_details": fix_result
                    }
            
            else:
                # Exception agent couldn't fix the SQL
                logger.error(f" EXCEPTION AGENT FAILED TO FIX SQL:")
                logger.error(f"Failure Reason: {fix_result['error']}")
                
                return {
                    "success": False,
                    "error": f"SQL execution failed and could not be automatically fixed: {fix_result['error']}",
                    "final_sql": sql_query,
                    "attempts": fix_result["total_iterations"],
                    "exception_summary": {
                        "original_error": execution_result["error"],
                        "fix_attempts": fix_result["total_iterations"],
                        "fix_status": "failed",
                        "recommendation": fix_result.get("recommendation", "Manual review required")
                    },
                    "original_sql": sql_query,
                    "exception_details": fix_result
                }
                
        except Exception as e:
            logger.error(f"Error in SQL execution with error handling: {e}")
            return {
                "success": False,
                "error": f"Error handling system failure: {str(e)}",
                "final_sql": sql_query,
                "attempts": 0,
                "exception_summary": {
                    "system_error": str(e),
                    "status": "error_handling_system_failed"
                }
            }
    
    def get_multi_step_capabilities(self) -> Dict[str, Any]:
        """Return information about multi-step query capabilities"""
        return {
            "supports_multi_step": True,
            "max_steps": 5, #max number of nested queries.
            "supported_patterns": [
                "top_n_then_details",  
                "filter_then_analyze", 
                "temporal_comparison",
                "sequential_filtering"
            ],
            "decomposer_agent": "SQLQueryDecomposer",
            "generator_agent": "SQLGeneratorAgent",
            "exception_agent": "SQLExceptionAgent",
            "error_handling": {
                "enabled": True,
                "max_fix_iterations": 3,
                "supported_error_types": [
                    "syntax_error", "column_error", "table_error", 
                    "join_error", "function_error", "date_error", "cube_js_error"
                ],
                "learning_enabled": True,
                "automatic_retry": True
            }
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation history"""
        return {
            "total_conversations": len(self.conversation_history) // 2,
            "recent_queries": [
                msg["content"] for msg in self.conversation_history[-10:] 
                if msg["role"] == "user"
            ],
            "schema_info_available": bool(self.get_schema_info()),
            "sub_agents_initialized": {
                "decomposer": self.decomposer is not None,
                "sql_generator": self.sql_generator is not None,
                "sql_retriever": self.sql_retriever is not None,
                "exception_agent": self.exception_agent is not None
            },
            "per_step_retrieval_enabled": self.sql_retriever is not None
        }
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about the per-step retrieval system"""
        return {
            "retrieval_system_active": self.sql_retriever is not None,
            "retrieval_approach": "per_step" if self.sql_retriever else "main_query_only",
            "benefits": [
                "Higher accuracy for each step",
                "Step-specific context matching", 
                "Better SQL quality",
                "Reduced noise from irrelevant examples"
            ] if self.sql_retriever else ["Limited context accuracy"],
            "estimated_accuracy_improvement": "25-35%" if self.sql_retriever else "0%"
        }
    
    def _convert_to_dataframe(self, query_results: Dict[str, Any]):
        """
        Convert query results to pandas DataFrame for summary agent consumption.
        
        Args:
            query_results: Dictionary containing query results with 'data' key
            
        Returns:
            pandas.DataFrame or None if conversion fails
        """
        try:
            # Import pandas here to avoid global import issues
            import pandas as pd
            
            if not query_results or 'data' not in query_results:
                logger.warning("No data found in query_results for DataFrame conversion")
                return pd.DataFrame() 
            
            data = query_results['data']
            
            if not data:
                logger.info("Empty data list, returning empty DataFrame")
                return pd.DataFrame()
            
            if not isinstance(data, list):
                logger.warning(f"Data is not a list, type: {type(data)}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"Successfully converted {len(df)} rows to DataFrame with columns: {list(df.columns)}")
            return df
            
        except ImportError:
            logger.error("Pandas not available for DataFrame conversion")
            return None
        except Exception as e:
            logger.error(f"Error converting to DataFrame: {e}")
            return None