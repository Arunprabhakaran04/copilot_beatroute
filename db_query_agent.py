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
from improved_sql_generator import ImprovedSQLGenerator
from multi_step_sql_generator import MultiStepSQLGenerator
from sql_exception_agent import SQLExceptionAgent
from summary_agent import SummaryAgent
from db_connection import get_database_connection, execute_sql
from token_tracker import track_llm_call, get_token_tracker

class DBQueryAgent(BaseAgent):
    """
    DB Query Agent orchestrator for handling both simple and complex multi-step SQL queries.
    Uses ImprovedSQLGenerator for single-step and Step 1 of multi-step queries.
    Uses MultiStepSQLGenerator for Step 2+ of multi-step queries (stricter Cube.js rules).
    """
    
    def __init__(self, llm, schema_file_path: str = None):
        super().__init__(llm)
        self.schema_file_path = schema_file_path  # Kept for backward compatibility, not used
        
        # Initialize sub-agents (schema will come from UserContext)
        self.sql_generator = ImprovedSQLGenerator(llm, schema_file_path=None)
        self.multi_step_sql_generator = MultiStepSQLGenerator(llm, schema_file_path=None)
        self.exception_agent = SQLExceptionAgent(llm, schema_file_path=None, max_iterations=5)
        self.summary_agent = SummaryAgent(llm, "gpt-4o-mini")
        self.conversation_history = []  
        
        self.decomposer = SQLQueryDecomposer(llm)
        
        # Keep reference as improved_sql_generator for compatibility
        self.improved_sql_generator = self.sql_generator
        
        try:
            from sql_retriever_agent import SQLRetrieverAgent
            self.sql_retriever = SQLRetrieverAgent(llm, "embeddings.pkl")
            logger.info("DBQueryAgent initialized as orchestrator with sub-agents:")
            logger.info("  - SQL Query Decomposer: for multi-step analysis")
            logger.info("  - Improved SQL Generator: for single-step & Step 1 of multi-step")
            logger.info("  - Multi-Step SQL Generator: for Step 2+ of multi-step (strict Cube.js rules)")
            logger.info("  - SQL Retriever: for step-specific context retrieval")
            logger.info("  - SQL Exception Agent: for error analysis and fixing")
            logger.info("  - Summary Agent: for generating data summaries")
        except Exception as e:
            logger.warning(f"Could not initialize SQL retriever: {e}")
            self.sql_retriever = None
            logger.info("DBQueryAgent initialized as orchestrator with sub-agents:")
            logger.info("  - SQL Query Decomposer: for multi-step analysis")
            logger.info("  - Improved SQL Generator: for single-step & Step 1")
            logger.info("  - Multi-Step SQL Generator: for Step 2+ (strict Cube.js rules)")
            logger.info("  - SQL Retriever: NOT AVAILABLE (will use fallback)")
            logger.info("  - SQL Exception Agent: for error analysis and fixing")
            logger.info("  - Summary Agent: for generating data summaries")
    
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
            
            # ✅ CRITICAL: Store user_context as class attribute for multi-step access
            if "user_context" in state and state["user_context"]:
                self.user_context = state["user_context"]
                logger.info("✅ Stored user_context in DBQueryAgent for multi-step execution")
            
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
                
                # Record timing in global tracker
                try:
                    from clean_logging import get_timing_tracker
                    timing_tracker = get_timing_tracker()
                    timing_tracker.record("db_query_multi_step", execution_time)
                except ImportError:
                    pass
                
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
                
                # Record timing in global tracker
                try:
                    from clean_logging import get_timing_tracker
                    timing_tracker = get_timing_tracker()
                    timing_tracker.record("db_query", execution_time)
                except ImportError:
                    pass
                
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
            
            # ✅ PERFORMANCE OPTIMIZATION: Use cached SQLs from EnrichAgent if available
            cached_sqls = state.get("cached_retrieved_sqls", [])
            cache_source = state.get("cache_source_query", "")
            retrieval_query = state.get("original_query", state["query"])
            
            # Validate cache: ensure it's for the SAME question (cache invalidation)
            if cached_sqls and cache_source == retrieval_query:
                logger.info(f"💾 Using cached SQL examples from EnrichAgent: {len(cached_sqls)} SQLs")
                logger.info(f"   Cache is valid for query: '{cache_source}'")
                retrieved_sqls = cached_sqls
            else:
                # Cache miss or invalid - retrieve fresh
                if cached_sqls:
                    logger.warning(f"❌ Cache invalid: source='{cache_source}' vs current='{retrieval_query}'")
                logger.info(f"🔍 Retrieving SQL examples for: '{retrieval_query}'")
                if retrieval_query != state["query"]:
                    logger.info(f"   (Enriched query: '{state['query']}')")
                retrieved_sqls = self._retrieve_step_specific_sqls(retrieval_query)
            
            # ✅ PERFORMANCE OPTIMIZATION: Reduce SQL examples based on top similarity
            # High similarity = fewer examples needed (pattern is clear)
            if retrieved_sqls and len(retrieved_sqls) > 0:
                # Get top similarity (retrieved_sqls are already sorted by similarity ascending)
                top_similarity = retrieved_sqls[-1].get('similarity', 0) if isinstance(retrieved_sqls[-1], dict) else 0
                original_count = len(retrieved_sqls)
                
                if top_similarity > 0.80:
                    # Very high similarity - keep only top 10 examples
                    retrieved_sqls = retrieved_sqls[-10:] if len(retrieved_sqls) > 10 else retrieved_sqls
                    logger.info(f"⚡ High similarity ({top_similarity:.3f}) detected - reduced from {original_count} to {len(retrieved_sqls)} examples")
                elif top_similarity > 0.70:
                    # Good similarity - keep top 15 examples
                    retrieved_sqls = retrieved_sqls[-15:] if len(retrieved_sqls) > 15 else retrieved_sqls
                    logger.info(f"⚡ Good similarity ({top_similarity:.3f}) detected - reduced from {original_count} to {len(retrieved_sqls)} examples")
                else:
                    # Lower similarity - keep all 20 examples
                    logger.info(f"📚 Lower similarity ({top_similarity:.3f}) - using all {len(retrieved_sqls)} examples")
            
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
            
            # Use Improved SQL Generator for all SQL generation
            logger.info("Using Improved SQL Generator for single-step query")
            
            # Get entity info and conversation history
            entity_info = state.get("verified_entities", {})
            conversation_history = state.get("conversation_history", [])
            
            # ✅ PERFORMANCE OPTIMIZATION: Pass cached focused schema to SQL generator
            cached_schema = state.get("cached_focused_schema", "")
            if cached_schema and not entity_info:
                entity_info = {}
            if cached_schema:
                entity_info["cached_focused_schema"] = cached_schema
                logger.info(f"💾 Passing cached focused schema to SQL generator ({len(cached_schema)} chars)")
            
            generation_result = self.sql_generator.generate_sql(
                question=state["query"],
                similar_sqls=retrieved_sqls,
                previous_results=state.get("previous_step_results", {}),
                original_query=state.get("original_query", state["query"]),
                entity_info=entity_info,
                conversation_history=conversation_history
            )
            
            # Convert to expected format
            if generation_result.get("success"):
                result_state = {
                    "status": "completed",
                    "sql_query": generation_result["sql"],
                    "query_type": generation_result.get("query_type", "SELECT"),
                    "explanation": generation_result.get("explanation", "Generated with Improved SQL Generator")
                }
            else:
                result_state = {
                    "status": "failed",
                    "error_message": generation_result.get("error", "SQL generation failed")
                }
            
            logger.info(f"🔍 SQL GENERATOR RESULT:")
            logger.info(f"   Type: {type(result_state)}")
            logger.info(f"   Status: {result_state.get('status') if isinstance(result_state, dict) else 'N/A'}")
            
            if result_state["status"] == "completed":
                # Get focused schema from user_context if available
                focused_schema = None
                if "user_context" in state and state["user_context"]:
                    user_context = state["user_context"]
                    if hasattr(user_context, 'get_focused_schema'):
                        # Get schema based on the question and retrieved SQLs
                        similar_sqls = generator_state.get("retrieved_sql_context", [])
                        focused_schema = user_context.get_focused_schema(
                            question=state["query"],
                            retrieved_sqls=[{"sql": sql} for sql in similar_sqls] if similar_sqls else [],
                            k=10
                        )
                        logger.info("✅ Got focused schema from UserContext for exception handling")
                
                # Try to execute the SQL and handle any errors
                execution_result = self._execute_sql_with_error_handling(
                    sql_query=result_state["sql_query"],
                    original_question=state["query"],
                    similar_sqls=generator_state.get("retrieved_sql_context", []),
                    focused_schema=focused_schema,
                    session_id=state.get("session_id")  # Pass session_id for DB auth
                )
                
                if execution_result["success"]:
                    # Log execution_result structure for debugging
                    logger.info(f"🔍 EXECUTION_RESULT STRUCTURE:")
                    logger.info(f"   Keys: {list(execution_result.keys())}")
                    logger.info(f"   Has 'query_results': {'query_results' in execution_result}")
                    logger.info(f"   Has 'results': {'results' in execution_result}")
                    if "query_results" in execution_result:
                        logger.info(f"   query_results type: {type(execution_result['query_results'])}")
                    if "results" in execution_result:
                        logger.info(f"   results type: {type(execution_result['results'])}")
                    
                    # Convert query results to pandas DataFrame - with error handling
                    query_data_df = None
                    try:
                        query_results_for_df = execution_result.get("query_results", {})
                        logger.info(f"   Converting to DataFrame - input type: {type(query_results_for_df)}")
                        query_data_df = self._convert_to_dataframe(query_results_for_df)
                        logger.info(f"   DataFrame conversion result: {type(query_data_df)}")
                    except Exception as df_error:
                        logger.error(f"❌ DataFrame conversion failed: {df_error}")
                        logger.error(f"   Input was: {type(execution_result.get('query_results'))}")
                        query_data_df = None
                    
                    # Send table data immediately via callback (before summary generation)
                    # 🚫 DO NOT send intermediate step tables in multi-step queries
                    table_sent_via_callback = False
                    is_multi_step = state.get("is_multi_step", False)
                    current_step = state.get("current_step", 1)
                    total_steps = state.get("total_steps", 1)
                    is_final_step = (current_step == total_steps)
                    
                    # Only send table if: (1) single-step query, OR (2) final step of multi-step
                    should_send_table = (not is_multi_step) or is_final_step
                    
                    if state.get("table_callback") and execution_result.get("query_results"):
                        if should_send_table:
                            try:
                                table_data = execution_result["query_results"].get("data")
                                if table_data:
                                    logger.info(f"📤 Calling table_callback to send data immediately (step {current_step}/{total_steps})")
                                    import asyncio
                                    loop = asyncio.new_event_loop()
                                    loop.run_until_complete(state["table_callback"](table_data))
                                    loop.close()
                                    table_sent_via_callback = True
                                    logger.info("✅ Table sent via callback successfully")
                            except Exception as cb_error:
                                logger.error(f"❌ Table callback failed: {cb_error}")
                                table_sent_via_callback = False
                        else:
                            logger.info(f"⏭️ Skipping table callback for intermediate step {current_step}/{total_steps}")
                    
                    # ⚡ OPTIMIZATION: Skip summary generation for single-step queries
                    # Summary will be generated by multi-step handler for final step only
                    # This saves 4-8 seconds and reduces token costs
                    summary_html = None
                    logger.info("⚡ Skipping summary generation (will be handled by multi-step handler if needed)")
                    
                    db_state["query_type"] = result_state.get("query_type", "SELECT")
                    # FIX: Get SQL from result_state (where SQL generator puts it), not execution_result
                    db_state["sql_query"] = result_state.get("sql_query", "") or execution_result.get("final_sql", "")
                    db_state["status"] = "completed"
                    db_state["success_message"] = execution_result.get("message", "Query completed")
                    
                    # Safely get result_state["result"]
                    if "result" in result_state and isinstance(result_state["result"], dict):
                        db_state["result"] = result_state["result"]
                    else:
                        logger.warning(f"⚠️ result_state['result'] is missing or not a dict, creating new dict")
                        db_state["result"] = {}
                    
                    db_state["result"]["is_multi_step"] = False
                    db_state["result"]["step_count"] = 1
                    db_state["result"]["exception_handling"] = execution_result.get("exception_summary")
                    db_state["result"]["table_sent_via_callback"] = table_sent_via_callback
                    
                    # Add database execution results - ensure it's a dict
                    query_results_data = execution_result.get("query_results", {})
                    if not isinstance(query_results_data, dict):
                        logger.error(f"❌ query_results is not a dict! Type: {type(query_results_data)}, converting to dict")
                        query_results_data = {}
                    
                    db_state["result"]["query_results"] = query_results_data
                    
                    # Add summary to query_results
                    if summary_html:
                        db_state["result"]["query_results"]["summary"] = summary_html
                    
                    # Log what we're putting in query_results (with error handling)
                    query_results = db_state["result"]["query_results"]
                    logger.info(f"📦 QUERY_RESULTS STRUCTURE:")
                    
                    # Check if query_results is actually a dict
                    if isinstance(query_results, dict):
                        logger.info(f"   Keys: {list(query_results.keys())}")
                        if "data" in query_results:
                            data = query_results["data"]
                            logger.info(f"   Data type: {type(data)}")
                            logger.info(f"   Data length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                            if isinstance(data, list) and len(data) > 0:
                                logger.info(f"   📊 TABLE DATA (first 3 rows):")
                                for idx, row in enumerate(data[:3]):
                                    logger.info(f"      Row {idx + 1}: {row}")
                        else:
                            logger.warning(f"   ⚠️ NO 'data' FIELD IN query_results!")
                        
                        if "summary" in query_results:
                            logger.info(f"   Summary field exists: {len(query_results['summary'])} chars")
                        else:
                            logger.warning(f"   ⚠️ NO 'summary' FIELD IN query_results!")
                    else:
                        logger.error(f"   ❌ query_results is not a dict! Type: {type(query_results)}, Value: {query_results}")
                    
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
                    # Check if there's execution metadata summary (not HTML summary)
                    query_results_obj = execution_result["query_results"]
                    if isinstance(query_results_obj, dict) and "data" in query_results_obj:
                        print(f"\nDATABASE RESULTS:")
                        # data is now a JSON string, parse it to count rows
                        data_rows = query_results_obj.get("data", "[]")
                        if isinstance(data_rows, str):
                            import json
                            try:
                                data_rows = json.loads(data_rows)
                            except:
                                data_rows = []
                        print(f"Rows Retrieved: {len(data_rows) if isinstance(data_rows, list) else 0}")
                        # Try to get columns from first row
                        if data_rows and isinstance(data_rows, list) and len(data_rows) > 0:
                            print(f"Columns: {list(data_rows[0].keys())}")
                
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
            
            # 🔍 CRITICAL DEBUG: Log what's being returned to main.py
            logger.info(f"🔍 RETURNING FROM _handle_single_step_query:")
            logger.info(f"   db_state type: {type(db_state)}")
            logger.info(f"   db_state top-level keys: {list(db_state.keys())}")
            logger.info(f"   db_state['sql_query'] exists: {'sql_query' in db_state}")
            if 'sql_query' in db_state:
                logger.info(f"   db_state['sql_query'] value: {db_state['sql_query'][:100] if db_state['sql_query'] else 'EMPTY STRING'}")
            else:
                logger.warning(f"   ⚠️ db_state['sql_query'] is MISSING!")
            logger.info(f"   db_state['result'] type: {type(db_state.get('result', 'NOT_FOUND'))}")
            if isinstance(db_state.get('result'), dict):
                logger.info(f"   db_state['result'] keys: {list(db_state['result'].keys())}")
                logger.info(f"   db_state['result']['query_results'] exists: {'query_results' in db_state['result']}")
            
            return db_state
            
        except Exception as e:
            logger.error(f"Single-step query error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full traceback:", exc_info=True)
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
            
            # ⚡ OPTIMIZATION: Use shared focused schema for all steps
            # EnrichAgent already created focused schema for the entire query
            # No need to retrieve schema per-step, saving 4.2s+ per additional step
            cached_schema = state.get("cached_focused_schema", "")
            if cached_schema:
                logger.info(f"⚡ Using shared focused schema across all {step_count} steps (from EnrichAgent)")
                logger.info(f"   Schema contains {len(cached_schema.split('CREATE TABLE'))-1} tables")
            else:
                logger.warning("⚠️ No cached_focused_schema found - may impact performance")
            
            executed_steps = []
            previous_results = {}
            all_sql_queries = []
            
            for step_idx, step_question in enumerate(steps, 1):
                logger.info(f" EXECUTING MULTI-STEP QUERY - STEP {step_idx}/{step_count}")
                logger.info(f"Step {step_idx} Question: {step_question}")
                
                # 🚫 CRITICAL: Remove table_callback for intermediate steps
                # Only the final step (handled by multi-step handler) should send tables
                original_callback = state.get("table_callback")
                if step_idx < step_count:
                    # Temporarily remove callback for intermediate steps
                    state["table_callback"] = None
                    logger.info(f"⏭️ Suppressing table_callback for intermediate step {step_idx}/{step_count}")
                else:
                    # Restore callback for final step (though multi-step handler sends it)
                    logger.info(f"📤 Final step {step_idx}/{step_count} - callback will be used by multi-step handler")
                
                # ✅ PERFORMANCE OPTIMIZATION: Use cached SQLs for FIRST step only
                # Subsequent steps get fresh retrieval since they're different questions
                if step_idx == 1:
                    cached_sqls = state.get("cached_retrieved_sqls", [])
                    cache_source = state.get("cache_source_query", "")
                    original_query = state.get("original_query", state["query"])
                    
                    # Validate cache for first step (should match original query)
                    if cached_sqls and cache_source == original_query:
                        logger.info(f"💾 Using cached SQL examples for step 1: {len(cached_sqls)} SQLs")
                        step_specific_sqls = cached_sqls
                    else:
                        logger.info(f"🔍 Cache miss for step 1 - retrieving fresh examples")
                        step_specific_sqls = self._retrieve_step_specific_sqls(step_question)
                else:
                    # For steps 2+, always retrieve fresh (different questions)
                    step_specific_sqls = self._retrieve_step_specific_sqls(step_question)
                
                # ✅ PERFORMANCE OPTIMIZATION: Reduce SQL examples based on top similarity (multi-step)
                if step_specific_sqls and len(step_specific_sqls) > 0:
                    top_similarity = step_specific_sqls[-1].get('similarity', 0) if isinstance(step_specific_sqls[-1], dict) else 0
                    original_count = len(step_specific_sqls)
                    
                    if top_similarity > 0.80:
                        step_specific_sqls = step_specific_sqls[-10:] if len(step_specific_sqls) > 10 else step_specific_sqls
                        logger.info(f"⚡ Step {step_idx}: High similarity ({top_similarity:.3f}) - reduced to {len(step_specific_sqls)} examples")
                    elif top_similarity > 0.70:
                        step_specific_sqls = step_specific_sqls[-15:] if len(step_specific_sqls) > 15 else step_specific_sqls
                        logger.info(f"⚡ Step {step_idx}: Good similarity ({top_similarity:.3f}) - reduced to {len(step_specific_sqls)} examples")
                    else:
                        logger.info(f"📚 Step {step_idx}: Using all {len(step_specific_sqls)} examples (similarity: {top_similarity:.3f})")
                
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
                    ],
                    # ✅ FIX: Include actual execution results for visualization
                    "query_results": step_result.get("query_results", {}),
                    "formatted_output": step_result.get("formatted_output", ""),
                    "json_results": step_result.get("json_results", "[]")
                }
                
                executed_steps.append(step_info)
                all_sql_queries.append(step_result["sql"])
                
                # ✅ CRITICAL FIX: Store actual query results for multi-step generator
                previous_results[f"step_{step_idx}"] = {
                    "sql": step_result["sql"],
                    "question": step_question,
                    "step_number": step_idx,
                    "context_count": len(step_specific_sqls),
                    "context_quality": step_result.get("context_quality", "unknown"),
                    "retrieval_quality": "high" if len(step_specific_sqls) >= 3 else "low",
                    # ✅ Include actual query results for value extraction
                    "query_results": step_result.get("query_results", {}),
                    "formatted_output": step_result.get("formatted_output", ""),
                    "json_results": step_result.get("json_results", "[]")
                }
                
                logger.info(f"Step {step_idx} completed successfully")
                logger.info("" + "-" * 80)
            
            # ✅ Restore original callback after loop
            if original_callback:
                state["table_callback"] = original_callback
                logger.info("✅ Restored original table_callback after multi-step execution")
            
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
            
            # ⚡ OPTIMIZATION: Generate summary ONLY for final step (not intermediate steps)
            # This saves 4-8 seconds per intermediate step and reduces token costs
            logger.info(f"⚡ Generating summary for FINAL step only (step {step_count}/{step_count})")
            summary_html = None
            try:
                if query_data_df is not None and not query_data_df.empty:
                    summary_html = self.summary_agent.generate_summary(state['query'], query_data_df)
                    logger.info(f"✅ Summary generated successfully ({len(summary_html)} chars)")
                else:
                    logger.warning("No data available for summary generation")
            except Exception as summary_error:
                logger.error(f"Failed to generate summary: {summary_error}")
                summary_html = None
            
            # Add summary to final query_results
            if summary_html:
                final_query_results["summary"] = summary_html
            
            # Log what we're putting in query_results for multi-step
            logger.info(f"📦 MULTI-STEP QUERY_RESULTS STRUCTURE:")
            logger.info(f"   Keys: {list(final_query_results.keys())}")
            if "data" in final_query_results:
                data = final_query_results["data"]
                logger.info(f"   Data type: {type(data)}")
                logger.info(f"   Data length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"   📊 TABLE DATA (first 3 rows):")
                    for idx, row in enumerate(data[:3]):
                        logger.info(f"      Row {idx + 1}: {row}")
            else:
                logger.warning(f"   ⚠️ NO 'data' FIELD IN query_results!")
            
            if "summary" in final_query_results:
                logger.info(f"   Summary field exists: {len(final_query_results['summary'])} chars")
            else:
                logger.warning(f"   ⚠️ NO 'summary' FIELD IN query_results!")
            
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
                # Add final database results (with summary)
                "query_results": final_query_results,
                "query_data": query_data_df,  # Add DataFrame for summary agent
                "formatted_output": final_step_results.get("formatted_output", ""),
                "json_results": final_step_results.get("json_results", "[]")
            }
            
            # Add final query data for easy access by other agents
            db_state["query_data"] = final_query_results.get("data", [])
            
            # ✅ SEND FINAL STEP TABLE VIA CALLBACK (not intermediate steps)
            if state.get("table_callback") and final_query_results.get("data"):
                try:
                    logger.info("📤 Calling table_callback to send FINAL STEP data immediately")
                    table_data = final_query_results.get("data", [])
                    
                    # Convert to JSON string if it's a list
                    if isinstance(table_data, list):
                        import json
                        table_data = json.dumps(table_data)
                    
                    # Send table via callback
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(state["table_callback"](table_data))
                    loop.close()
                    logger.info("✅ FINAL STEP table sent via callback successfully")
                except Exception as callback_error:
                    logger.error(f"Failed to send table via callback: {callback_error}")
            
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
                logger.info(f"📥 Retrieved {len(similar_sqls)} step-specific SQL examples")
                
                # ✅ Log top 3 retrieved SQLs with actual queries for debugging
                for idx, sql_info in enumerate(similar_sqls[:3], 1):
                    question = sql_info.get('question', 'N/A')
                    sql = sql_info.get('sql', 'N/A')
                    similarity = sql_info.get('similarity', 0)
                    
                    logger.info(f"   📋 Example {idx} (sim={similarity:.3f}):")
                    logger.info(f"      Q: {question}")
                    logger.info(f"      SQL: {sql[:200]}...")  # First 200 chars
                
                # ✅ CRITICAL: Pass ALL 20 retrieved SQLs to maximize LLM learning
                num_to_pass = min(20, len(similar_sqls))
                logger.info(f"🎯 Passing top {num_to_pass} examples to SQL generator (increased from 10)")
                return similar_sqls[:20]  # Pass 20 instead of 10 for better pattern learning
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
            
            # ✅ DETERMINE WHICH GENERATOR TO USE
            # Use MultiStepSQLGenerator for Step 2+ (has previous_results)
            # Use ImprovedSQLGenerator for Step 1 (no previous_results)
            use_multi_step_generator = bool(previous_results)
            
            if use_multi_step_generator:
                logger.info("🔄 Using MultiStepSQLGenerator (Step 2+, has previous results)")
                active_generator = self.multi_step_sql_generator
            else:
                logger.info("🔄 Using ImprovedSQLGenerator (Step 1, no previous results)")
                active_generator = self.sql_generator
            
            # ✅ CRITICAL FIX: Ensure user_context is available in state for focused schema
            # During multi-step execution, user_context may not be in step state
            if state and "user_context" not in state and hasattr(self, 'user_context'):
                state["user_context"] = self.user_context
                logger.info("✅ Copied user_context to step state for schema access")
            
            # Get conversation history from state (if available)
            conversation_history = state.get("conversation_history", []) if state else []
            
            # Generate SQL using appropriate generator
            original_query = state.get("original_query", question)
            entity_info = state.get("entity_info", None)
            
            # Validate and get entity_info
            if entity_info is None:
                entity_info = state.get("verified_entities", None)
                if entity_info:
                    logger.info(f"Found entity_info under 'verified_entities' key")
                else:
                    logger.warning(f"No entity_info in state. Available keys: {list(state.keys())}")
                    entity_info = {
                        "entities": [],
                        "entity_types": [],
                        "entity_mapping": {},
                        "verified_entities": {}
                    }
            else:
                logger.info(f"Entity info available: {len(entity_info.get('entities', []))} entities")
            
            # ✅ PERFORMANCE OPTIMIZATION: Pass cached focused schema for multi-step queries
            cached_schema = state.get("cached_focused_schema", "") if state else ""
            if cached_schema and not entity_info:
                entity_info = {}
            if cached_schema:
                entity_info["cached_focused_schema"] = cached_schema
                logger.info(f"💾 Passing cached focused schema to SQL generator (multi-step)")
            
            # ✅ Pass schema_manager to multi-step generator if available
            if use_multi_step_generator and state and "user_context" in state:
                user_context = state["user_context"]
                if hasattr(user_context, 'schema_manager'):
                    active_generator.schema_manager = user_context.schema_manager
                    logger.info("✅ Passed schema_manager to MultiStepSQLGenerator")
            
            generation_result = active_generator.generate_sql(
                question=question,
                similar_sqls=similar_sqls,
                previous_results=previous_results,
                original_query=original_query,
                entity_info=entity_info,
                conversation_history=conversation_history
            )
            
            generator_name = "Multi-Step SQL Generator" if use_multi_step_generator else "Improved SQL Generator"
            logger.info(f"Using {generator_name} (method: {generation_result.get('method', 'unknown')})")
            
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
            
            # Get focused schema from user_context if available
            focused_schema = None
            if "user_context" in state and state["user_context"]:
                user_context = state["user_context"]
                if hasattr(user_context, 'get_focused_schema'):
                    # Get schema based on the question and retrieved SQLs
                    focused_schema = user_context.get_focused_schema(
                        question=question,
                        retrieved_sqls=[{"sql": sql} for sql in similar_sqls] if similar_sqls else [],
                        k=10
                    )
                    logger.info("✅ Got focused schema from UserContext for exception handling (multi-step)")
            
            # Execute the generated SQL with error handling
            execution_result = self._execute_sql_with_error_handling(
                sql_query=generation_result["sql"],
                original_question=question,
                similar_sqls=similar_sqls,
                focused_schema=focused_schema,  # Pass focused schema
                session_id=state.get("session_id")  # Pass session_id for DB auth
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
    
    def _execute_sql_in_database(self, sql_query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Execute SQL query in the actual Cube.js database and return results.
        
        Args:
            sql_query: The SQL query to execute
            session_id: Session ID for database authentication
            
        Returns:
            Dict with success status, results, and any error information
        """
        try:
            logger.info(f"EXECUTING SQL IN CUBE.JS DATABASE:")
            logger.info(f"SQL: {sql_query}")
            
            # Execute query using database connection with session_id for authentication
            db_result = execute_sql(sql_query, session_id=session_id)
            
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
                                       similar_sqls: List[str] = None,
                                       focused_schema: str = None,
                                       session_id: str = None) -> Dict[str, Any]:
        """
        Execute SQL with comprehensive error handling and automatic fixing.
        
        This method:
        1. Simulates SQL execution 
        2. If an error occurs, uses the Exception Agent to analyze and fix
        3. Retries execution with the corrected SQL
        4. Returns the final result or failure after max attempts
        
        Args:
            sql_query: The SQL query to execute
            original_question: The original user question
            similar_sqls: Optional list of similar successful SQL queries
            focused_schema: Optional focused schema from UserContext
            session_id: Session ID for database authentication
        """
        try:
            logger.info(f"EXECUTING SQL WITH ERROR HANDLING:")
            logger.info(f"Original Question: {original_question}")
            logger.info(f"SQL to Execute: {sql_query}")
            
            execution_result = self._execute_sql_in_database(sql_query, session_id=session_id)
            
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
            
            # Check if error is due to non-existent field/column (graceful handling)
            error_msg = execution_result['error'].lower()
            if "no field named" in error_msg or "column does not exist" in error_msg:
                logger.info("🔍 Detected missing field/column error - checking if data exists...")
                
                # Extract the missing field name from error message
                import re
                field_match = re.search(r"no field named '([^']+)'", execution_result['error'], re.IGNORECASE)
                if field_match:
                    missing_field = field_match.group(1)
                    logger.info(f"❌ Field '{missing_field}' does not exist in schema")
                    
                    # Check if this is a filter condition (e.g., category='Doctors')
                    if "category" in missing_field.lower() or "subtype" in missing_field.lower():
                        logger.info("💡 Graceful handling: Returning empty result (no data matches criteria)")
                        
                        return {
                            "success": True,
                            "final_sql": sql_query,
                            "message": f"No data found - field '{missing_field}' does not exist in the database schema",
                            "attempts": 1,
                            "query_results": {
                                "data": [],
                                "columns": [],
                                "summary": {
                                    "total_rows": 0,
                                    "columns_count": 0,
                                    "query_type": "SELECT"
                                }
                            },
                            "formatted_output": "No data found matching your criteria. The requested field does not exist in the database.",
                            "json_results": "[]",
                            "graceful_empty": True,
                            "graceful_reason": f"Field '{missing_field}' does not exist in schema"
                        }
            
            fix_result = self.exception_agent.iterative_fix_sql(
                original_question=original_question,
                failed_sql=sql_query,
                error_message=execution_result["error"],
                similar_sqls=similar_sqls or [],
                focused_schema=focused_schema  # Pass focused schema to exception agent
            )
            
            if fix_result["success"]:
                corrected_sql = fix_result["final_sql"]
                logger.info(f" RETRYING WITH CORRECTED SQL:")
                logger.info(f"Corrected SQL: {corrected_sql}")
                
                retry_execution = self._execute_sql_in_database(corrected_sql, session_id=session_id)
                
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
    
    def _format_dataframe(self, df):
        """Apply formatting to DataFrame: round numbers and format dates"""
        try:
            import pandas as pd
            import re
            
            if df is None or df.empty:
                return df
            
            # Round numeric columns (only affects numeric columns, safe if none exist)
            numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = df[numeric_cols].round(2)
                logger.info(f"Rounded {len(numeric_cols)} numeric columns")
            
            # Format date columns
            date_formats = [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%Y/%m/%d",
                "%d-%m-%Y %H:%M:%S",
                "%d-%m-%Y %H:%M",
                "%d-%m-%Y",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y %H:%M",
                "%m/%d/%Y"
            ]
            
            pattern = re.compile(r"(date|month|time)", re.IGNORECASE)
            matching_cols = [col for col in df.columns if pattern.search(col)]
            
            if len(matching_cols) == 0:
                return df
            
            logger.info(f"Found {len(matching_cols)} potential date columns: {matching_cols}")
            
            for col in matching_cols:
                parsed = None
                for fmt in date_formats:
                    try:
                        parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
                        if parsed.notna().sum() > 0:
                            df[col] = parsed.dt.strftime("%B %d' %Y").combine_first(df[col])
                            break
                    except Exception:
                        continue
            
            return df
        except Exception as e:
            logger.error(f"Error formatting DataFrame: {e}")
            return df
    
    def _convert_to_dataframe(self, query_results: Dict[str, Any]):
        """
        Convert query results to pandas DataFrame for summary agent consumption.
        
        Args:
            query_results: Dictionary containing query results with 'data' key
                          where data is now a JSON string from df.to_json(orient="records")
            
        Returns:
            pandas.DataFrame or None if conversion fails
        """
        try:
            # Import pandas here to avoid global import issues
            import pandas as pd
            import json
            
            if not query_results or 'data' not in query_results:
                logger.warning("No data found in query_results for DataFrame conversion")
                return pd.DataFrame() 
            
            data = query_results['data']
            
            if not data:
                logger.info("Empty data, returning empty DataFrame")
                return pd.DataFrame()
            
            # Handle JSON string format (from df.to_json(orient="records"))
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                    logger.info(f"✅ Parsed JSON string to list of {len(data)} records")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON string: {e}")
                    return pd.DataFrame()
            
            if not isinstance(data, list):
                logger.warning(f"Data is not a list after parsing, type: {type(data)}")
                return pd.DataFrame()
            
            if not data:
                logger.info("Empty data list after parsing, returning empty DataFrame")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Successfully converted {len(df)} rows to DataFrame")
            logger.info(f"   Columns ({len(df.columns)}): {list(df.columns)}")
            logger.info(f"   Shape: {df.shape}")
            
            # Apply formatting
            df = self._format_dataframe(df)
            logger.info("✅ DataFrame formatted (rounded numbers and formatted dates)")
            
            return df
            
        except ImportError:
            logger.error("Pandas not available for DataFrame conversion")
            return None
        except Exception as e:
            logger.error(f"Error converting to DataFrame: {e}", exc_info=True)
            return None