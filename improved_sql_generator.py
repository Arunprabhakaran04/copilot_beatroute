import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from base_agent import BaseAgent, BaseAgentState, DBAgentState
from token_tracker import track_llm_call
from loguru import logger as loguru_logger

logger = logging.getLogger(__name__)

class ImprovedSQLGenerator(BaseAgent):
    """
    Improved SQL Generator with robust context handling and simplification logic.
    Focuses on generating accurate, simple queries with better multi-step support.
    """
    
    def __init__(self, llm, schema_file_path: str = None, schema_manager=None):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.schema_manager = schema_manager  # Optional SchemaManager for focused schema
        
        # Load full schema from file as fallback (only if path provided)
        self.schema_content = self._load_schema_file() if schema_file_path else ""
        
        self.core_patterns = {
            "monthly_sales": """
            SELECT
                DATE_TRUNC('month', CustomerInvoice.dispatchedDate) AS MonthYear,
                SUM(CustomerInvoice.dispatchedvalue) AS TotalSales
            FROM CustomerInvoice
            WHERE {date_filter}
            GROUP BY DATE_TRUNC('month', CustomerInvoice.dispatchedDate)
            ORDER BY MonthYear
            """,
            "total_sales": """
            SELECT SUM(CustomerInvoice.dispatchedvalue) AS TotalSales
            FROM CustomerInvoice
            WHERE {date_filter}
            """,
            "average_order_value": """
            SELECT AVG(Order.value) AS AverageOrderValue
            FROM Order
            WHERE {date_filter}
            """
        }
    
    def get_agent_type(self) -> str:
        return "improved_sql_generator"
    
    def _load_schema_file(self) -> str:
        """Load the database schema from the schema file."""
        if not self.schema_file_path:
            logger.info("No schema file path provided - will use UserContext schema")
            return ""
        
        try:
            with open(self.schema_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                content = content.replace('\\n', '\n')
                
                logger.info(f"Successfully loaded schema file: {self.schema_file_path}")
                return content
        except Exception as e:
            logger.warning(f"Could not load schema file: {e} - will use UserContext schema")
            return ""
    
    def _get_schema_context(self, focused_schema: str = None) -> str:
        """
        Get schema context for the prompt
        
        Args:
            focused_schema: Optional focused schema from SchemaManager
            
        Returns:
            Schema string to use in prompt
        """
        if focused_schema:
            logger.info("Using focused schema from SchemaManager")
            return focused_schema
        else:
            logger.info("Using full schema from file")
            return self.schema_content
    
    def _preserve_original_context(self, current_question: str, original_query: str) -> str:
        """
        Preserve the original query context when dealing with generic multi-step tasks.
        This prevents loss of specific details during decomposition.
        """
        current_lower = current_question.lower()
        original_lower = original_query.lower()
        
        if ("get" in current_lower or "retrieve" in current_lower) and len(original_lower) > len(current_lower):
            # Extract specific details from original query
            preserved_details = []
            
            time_matches = re.findall(r'(august|september|october|november|december|january|february|march|april|may|june|july)', original_lower)
            if time_matches:
                preserved_details.extend([f"month: {month}" for month in set(time_matches)])
            
            year_matches = re.findall(r'(202[0-9])', original_lower)
            if year_matches:
                preserved_details.extend([f"year: {year}" for year in set(year_matches)])
            
            if 'last 12 months' in original_lower or '12 months' in original_lower:
                preserved_details.append("time range: last 12 months")
            elif 'last 6 months' in original_lower or '6 months' in original_lower:
                preserved_details.append("time range: last 6 months")
            elif 'last 3 months' in original_lower or '3 months' in original_lower:
                preserved_details.append("time range: last 3 months")
            
            if 'sku' in original_lower:
                if 'top 3 sku' in original_lower or 'top 3 sku' in current_lower:
                    preserved_details.append("requirement: top 3 SKUs")
                elif 'top' in original_lower and any(char.isdigit() for char in original_lower):
                    top_n = re.search(r'top\s*(\d+)', original_lower)
                    if top_n:
                        preserved_details.append(f"requirement: top {top_n.group(1)} SKUs")
            
            specific_terms = re.findall(r'(separately|monthly|total|average|by month|month wise|trend|sales trend)', original_lower)
            if specific_terms:
                preserved_details.extend([f"grouping: {term}" for term in set(specific_terms)])
            
            if preserved_details:
                enhanced_question = f"{current_question} ({', '.join(preserved_details)})"
                logger.info(f"Enhanced generic question: '{current_question}' -> '{enhanced_question}'")
                return enhanced_question
        
        return current_question
    
    def _detect_query_intent(self, question: str) -> Dict[str, Any]:
        """
        Detect the primary intent of the query and return appropriate pattern and parameters.
        """
        question_lower = question.lower()
        
        # Default to SUM aggregation (will be changed to None for LIST queries)
        intent = {
            "pattern": None,
            "table": "CustomerInvoice",
            "date_column": "dispatchedDate", 
            "value_column": "dispatchedvalue",
            "aggregation": "SUM",
            "grouping": None,
            "time_filter": None,
            "is_simple": False,  
            "requires_sku_join": False,
            "requires_top_n": False,
            "top_n_value": None
        }
        
        # Detect LIST queries (no aggregation needed)
        list_indicators = ['list of', 'show me', 'give me', 'get me', 'find', 'which customers', 'which users', 'who']
        aggregation_indicators = ['total', 'sum', 'count', 'average', 'avg', 'maximum', 'minimum', 'max', 'min']
        
        # If query asks for a list AND doesn't ask for aggregation, set aggregation to None
        is_list_query = any(indicator in question_lower for indicator in list_indicators)
        has_aggregation = any(agg in question_lower for agg in aggregation_indicators)
        
        if is_list_query and not has_aggregation:
            intent["aggregation"] = None
            logger.info("🎯 Detected LIST query (no aggregation required)")
        
        # Detect SKU-related queries
        if 'sku' in question_lower:
            intent["requires_sku_join"] = True
            intent["is_simple"] = False  # SKU queries need LLM
            
            # Detect top N requirements
            if 'top' in question_lower:
                top_n_match = re.search(r'top\s*(\d+)', question_lower)
                if top_n_match:
                    intent["requires_top_n"] = True
                    intent["top_n_value"] = int(top_n_match.group(1))
        
        if any(month in question_lower for month in ['august', 'september', 'october']):
            months = []
            if 'august' in question_lower:
                months.append("august")
            if 'september' in question_lower:
                months.append("september")
            if 'october' in question_lower:
                months.append("october")
            
            if len(months) > 1 and ('separately' in question_lower or 'each' in question_lower):
                intent["pattern"] = "monthly_sales"
                intent["grouping"] = "month"
                intent["months_requested"] = months
            else:
                intent["pattern"] = "total_sales"
                intent["months_requested"] = months
        
        # Detect time ranges (last N months)
        time_range_match = re.search(r'last\s*(\d+)\s*months?', question_lower)
        if time_range_match:
            num_months = int(time_range_match.group(1))
            intent["time_range_months"] = num_months
            intent["grouping"] = "month" if 'trend' in question_lower or 'monthly' in question_lower else None
        
        # Detect order vs sales queries
        if any(term in question_lower for term in ['order', 'orders']):
            intent["table"] = "Order"
            intent["date_column"] = "datetime"
            intent["value_column"] = "value"
        
        # Detect aggregation type
        if 'average' in question_lower:
            intent["aggregation"] = "AVG"
            intent["pattern"] = "average_order_value" if intent["table"] == "Order" else "total_sales"
        
        return intent
    
    def generate_sql(self, question: str, similar_sqls: List[str] = None, 
                     previous_results: Dict[str, Any] = None, 
                     original_query: str = None,
                     entity_info: Dict[str, Any] = None,
                     conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SQL query with improved context handling.
        Always uses LLM generation for accuracy (simple generation disabled).
        
        Args:
            entity_info: Dictionary containing verified entity information from entity verification
                        Format: {'entities': ['name'], 'entity_types': ['ViewCustomer'], 
                                'entity_mapping': {'name': 'ViewCustomer'}}
            conversation_history: List of previous conversation entries from RedisMemoryManager
                        Format: [{'original': str, 'result': dict, 'timestamp': float}, ...]
        """
        try:
            enhanced_question = self._preserve_original_context(
                question, original_query or question
            )
            
            intent = self._detect_query_intent(enhanced_question)
            
            # ALWAYS use LLM generation 
            return self._generate_with_llm(enhanced_question, similar_sqls, previous_results, intent, entity_info, conversation_history)
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return {
                "success": False,
                "error": f"SQL generation error: {str(e)}",
                "type": "generation_error"
            }
    
    def _generate_with_llm(self, question: str, similar_sqls: List[Dict], 
                          previous_results: Dict[str, Any], intent: Dict[str, Any],
                          entity_info: Dict[str, Any] = None,
                          conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SQL using LLM with focused prompts and better validation.
        
        Args:
            entity_info: Dictionary containing verified entity information
            conversation_history: List of previous conversation entries for multi-turn context
        """
        import time
        start_time = time.time()
        
        # Log retrieved SQL queries (simplified - question and similarity only)
        if similar_sqls:
            logger.info(f"Retrieved {len(similar_sqls)} similar SQL examples:")
            for idx, sql_info in enumerate(similar_sqls, 1):
                similarity = sql_info.get('similarity', 0)
                question_text = sql_info.get('question', 'N/A')
                logger.info(f"  [{idx}] sim={similarity:.3f} | {question_text}")
        else:
            logger.info("No similar SQL queries retrieved")
        
        if previous_results:
            logger.info(f"Previous results available: {list(previous_results.keys())}")
        
        cleaned_previous_results = self._clean_previous_results(previous_results)
        
        # NOTE: We no longer filter examples - all 20 retrieved SQLs are injected into conversation
        # The LLM will naturally focus on the most similar ones (which are last in conversation)
        
        # Get focused schema if SchemaManager is available
        focused_schema = None
        focused_tables = []
        
        if self.schema_manager is not None:
            try:
                schema_start = time.time()
                focused_schema = self.schema_manager.get_schema_to_use_in_prompt(
                    current_question=question,
                    list_similar_question_sql_pair=similar_sqls or [],
                    k=10  # Top 10 similar tables
                )
                schema_time = time.time() - schema_start
                
                # Extract table names from focused schema for logging
                if focused_schema:
                    import re
                    focused_tables = re.findall(r'^Table: (.+)$', focused_schema, re.MULTILINE)
                    logger.info(f"Focused schema generated: {len(focused_tables)} tables ({schema_time:.2f}s)")
                else:
                    logger.warning("Focused schema is empty")
            except Exception as e:
                logger.warning(f"Failed to get focused schema: {e}. Using full schema.")
                focused_schema = None
        else:
            logger.info("No SchemaManager - using full schema from file")
        
        # Create focused prompt with previous results and entity context
        # Note: We pass similar_sqls instead of filtered examples since all SQLs are in conversation
        prompt = self._create_focused_prompt(
            question, 
            similar_sqls,  # Pass all retrieved SQLs (for year extraction)
            intent,  # Keep for logging purposes only
            cleaned_previous_results, 
            entity_info,
            focused_schema=focused_schema  # Pass focused schema to prompt
        )
        
        # Build user message with explicit entity names if available
        user_message = f"Generate SQL for: {question}"
        
        # Add entity names reminder in user message for visibility
        if entity_info and entity_info.get("entities"):
            entity_names = entity_info.get("entities", [])
            entity_types = entity_info.get("entity_types", [])
            
            user_message += "\n\nVerified entities found in database:\n"
            for entity_name, entity_type in zip(entity_names, entity_types):
                user_message += f"   - {entity_type}: '{entity_name}'\n"
            
            user_message += "\nNote: Use these exact values in your WHERE clause (case-sensitive)."
        
        # Build message log with conversation history AND retrieved SQLs
        # Retrieved SQLs are injected as fake conversation to make LLM pay attention
        message_log = self._build_message_log_with_history(
            system_prompt=prompt,
            current_question=user_message,
            conversation_history=conversation_history,
            retrieved_sqls=similar_sqls,  # Pass retrieved SQLs for injection
            max_history=5,  # Keep last 5 REAL conversations
            max_retrieved_sqls=20  # Inject up to 20 retrieved SQLs (increased from 10)
        )
        
        # ✅ CRITICAL: Add explicit instruction to follow retrieved SQL patterns
        # This is THE MOST IMPORTANT fix to make LLM use retrieved examples
        if similar_sqls and len(similar_sqls) > 0:
            num_examples = min(len(similar_sqls), 20)
            top_similarity = similar_sqls[0].get('similarity', 0) if similar_sqls else 0
            
            # Find the last user message (current question) to enhance it
            last_user_idx = None
            for i in range(len(message_log) - 1, -1, -1):
                if message_log[i]["role"] == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx is not None:
                original_question = message_log[last_user_idx]["content"]
                
                # ✅ Add EMPHATIC instruction to prioritize retrieved examples
                enhanced_question = f"""{original_question}

🎯 CRITICAL INSTRUCTIONS - FOLLOW RETRIEVED EXAMPLES:

The conversation history above contains {num_examples} real SQL examples from our database.
These examples are PROVEN to work with our exact schema and field names.

YOU MUST CAREFULLY STUDY THESE EXAMPLES FOR:
1. ✅ Exact field names to use (e.g., ViewCustomerActivity.__user, NOT ViewCustomer.name)
2. ✅ Table join patterns (single table vs CROSS JOIN)
3. ✅ WHERE clause syntax and date filtering
4. ✅ GROUP BY and aggregation patterns
5. ✅ Field availability in each table

⚠️ COMMON MISTAKES TO AVOID:
- DO NOT invent field names that don't exist in the examples
- DO NOT join tables if examples use single table approach
- DO NOT use fields from one table when selecting from another
- DO NOT ignore the exact syntax patterns shown in examples

🔥 HIGHEST PRIORITY: Match the patterns from the example with similarity {top_similarity:.3f}
This example is the MOST similar to your current question - follow its structure closely!

Now generate SQL following these proven patterns:"""
                
                message_log[last_user_idx]["content"] = enhanced_question
                logger.info(f"✅ Added explicit instruction to follow {num_examples} retrieved SQL patterns")
                logger.info(f"   Top similarity: {top_similarity:.3f} - LLM instructed to prioritize this example")
        
        # ADAPTIVE TEMPERATURE: Adjust creativity based on similarity
        # Note: No longer using filtered examples, just use similarity scores from retrieved SQLs
        adaptive_llm = self._get_adaptive_temperature_llm(similar_sqls)
        
        # Generate response with adaptive temperature
        response = adaptive_llm.invoke(message_log)
        content = response.content.strip()
        
        track_llm_call(
            input_prompt=message_log,
            output=content,
            agent_type="improved_sql_generator",
            operation="generate_sql",
            model_name="gpt-4.1"
        )
        
        # Log execution timing
        execution_time = time.time() - start_time
        logger.info(f"SQL generation completed in {execution_time:.2f}s")
        
        return self._parse_and_validate_response(content, question, intent)
    
    def _build_message_log_with_history(
        self,
        system_prompt: str,
        current_question: str,
        conversation_history: List[Dict[str, Any]] = None,
        retrieved_sqls: List[Dict[str, Any]] = None,
        max_history: int = 5,
        max_retrieved_sqls: int = 20
    ) -> List[Dict[str, str]]:
        """
        Build message log with conversation history AND retrieved SQLs for multi-turn context.
        
        Strategy: Inject retrieved SQLs as fake conversation history to make LLM pay attention.
        Retrieved SQLs are sorted in ASCENDING order of similarity (most similar last = most recent).
        This makes LLM think the most relevant example was just generated.
        
        Args:
            system_prompt: System prompt with instructions
            current_question: Current user question
            conversation_history: List of previous REAL conversation entries (from Redis)
            retrieved_sqls: List of retrieved SQL examples from embedding search
            max_history: Maximum number of previous conversations to include
            max_retrieved_sqls: Maximum number of retrieved SQLs to inject (default 20)
            
        Returns:
            Message log in format: [{"role": "system/user/assistant", "content": str}, ...]
        """
        message_log = [{"role": "system", "content": system_prompt}]
        
        # Step 1: Add REAL conversation history (if exists)
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
            
            logger.info(f"Adding {len(recent_history)} REAL conversation(s) from Redis")
            
            for idx, entry in enumerate(recent_history, 1):
                user_query = entry.get("original", "")
                message_log.append({"role": "user", "content": user_query})
                
                result = entry.get("result", {})
                assistant_response = self._format_assistant_response(result, max_rows=5)
                message_log.append({"role": "assistant", "content": assistant_response})
                
                logger.info(f"   Real conversation {idx}: '{user_query[:50]}...'")
        else:
            logger.info("No real conversation history")
        
        # Step 2: Inject RETRIEVED SQLs as fake conversation (ASCENDING similarity order)
        if retrieved_sqls and len(retrieved_sqls) > 0:
            # Sort by similarity ASCENDING (lowest first, highest last)
            sorted_sqls = sorted(retrieved_sqls[:max_retrieved_sqls], key=lambda x: x.get('similarity', 0))
            
            logger.info(f"🎯 Injecting {len(sorted_sqls)} retrieved SQLs as fake conversation (ASCENDING similarity)")
            logger.info(f"   Similarity range: {sorted_sqls[0].get('similarity', 0):.3f} (first) -> {sorted_sqls[-1].get('similarity', 0):.3f} (last/most similar)")
            
            # ✅ Log top 3 and bottom 1 to verify what LLM is seeing
            logger.info(f"   📥 Sample injected SQLs:")
            for idx, sql_info in enumerate(sorted_sqls[:3], 1):
                question = sql_info.get('question', '')
                sql = sql_info.get('sql', '')
                similarity = sql_info.get('similarity', 0)
                logger.info(f"      [{idx}] sim={similarity:.3f} Q: '{question[:60]}...'")
                logger.info(f"          SQL: {sql[:120]}...")
            
            # Log the MOST similar (last one)
            if len(sorted_sqls) > 3:
                last_sql = sorted_sqls[-1]
                logger.info(f"      [MOST SIMILAR] sim={last_sql.get('similarity', 0):.3f} Q: '{last_sql.get('question', '')[:60]}...'")
                logger.info(f"                     SQL: {last_sql.get('sql', '')[:120]}...")
            
            for idx, sql_info in enumerate(sorted_sqls, 1):
                # Add as user question
                question = sql_info.get('question', '')
                message_log.append({"role": "user", "content": question})
                
                # Add as assistant's SQL response (just the SQL, no explanation)
                sql = sql_info.get('sql', '')
                assistant_sql = f"```sql\n{sql}\n```"
                message_log.append({"role": "assistant", "content": assistant_sql})
        else:
            logger.info("No retrieved SQLs to inject")
        
        # Step 3: Add CURRENT question with explicit instruction to follow examples
        # ✅ CRITICAL: Add instruction to force LLM to follow retrieved SQL patterns
        if retrieved_sqls and len(retrieved_sqls) > 0:
            enhanced_question = f"""{current_question}

🎯 CRITICAL INSTRUCTIONS - You MUST follow these:
1. The {len(retrieved_sqls)} SQL examples above are REAL, WORKING queries from this exact database
2. Study their patterns carefully:
   - FIELD NAMES used (e.g., ViewCustomer.name requires CROSS JOIN ViewCustomer)
   - TABLE JOINS (CROSS JOIN patterns are mandatory when selecting fields from multiple tables)
   - WHERE clause date filtering patterns
   - WITH clause structures for multi-step queries
3. The MOST SIMILAR example (last one above) is the best pattern to follow
4. If example uses CROSS JOIN ViewCustomer to access ViewCustomer.name, YOU MUST DO THE SAME
5. DO NOT assume field names - use EXACT field names from the examples
6. If ViewCustomerActivity is used, remember it has: __user, activityTime, count, calls, etc.
7. If you need ViewCustomer.name, you MUST include: CROSS JOIN ViewCustomer

⚠️ COMMON MISTAKES TO AVOID:
- Using ViewCustomer.name without CROSS JOIN ViewCustomer (will fail!)
- Removing CROSS JOIN that was in the example pattern
- Assuming field names that don't match the examples
- Ignoring the table join patterns from similar examples"""
            
            message_log.append({"role": "user", "content": enhanced_question})
            logger.info("✅ Added explicit instruction to follow retrieved SQL patterns")
        else:
            message_log.append({"role": "user", "content": current_question})
        
        total_messages = len(message_log)
        real_conv_count = len(conversation_history) * 2 if conversation_history else 0
        injected_sql_count = len(retrieved_sqls) * 2 if retrieved_sqls else 0
        
        logger.info(f"FINAL message log: {total_messages} messages")
        logger.info(f"   = 1 system + {real_conv_count} real history + {injected_sql_count} injected SQLs + 1 current")
        
        return message_log
    
    def _format_assistant_response(self, result: Dict[str, Any], max_rows: int = 5) -> str:
        """
        Format assistant's previous response for conversation history.
        Includes SQL + summary, with optional data sample for small results.
        
        Args:
            result: Result dictionary from previous query
            max_rows: Maximum number of data rows to include
            
        Returns:
            Formatted response string
        """
        response_parts = []
        
        # SQL query (truncated if too long)
        if "sql" in result:
            sql = result["sql"]
            if len(sql) > 200:
                response_parts.append(f"Generated SQL: {sql[:200]}...")
            else:
                response_parts.append(f"Generated SQL: {sql}")
        
        # Result summary
        rows_returned = result.get("rows_returned", 0)
        if rows_returned > 0:
            response_parts.append(f"Result: {rows_returned} rows returned")
            
            # Include sample data if available and small enough
            data = result.get("data", [])
            if data and isinstance(data, list) and len(data) <= max_rows:
                response_parts.append(f"\nSample data (first {len(data)} rows):")
                response_parts.append(str(data))
            elif data and isinstance(data, list) and len(data) > max_rows:
                # Just show first few rows
                sample_data = data[:max_rows]
                response_parts.append(f"\nSample data (first {max_rows} of {len(data)} rows):")
                response_parts.append(str(sample_data))
        else:
            response_parts.append("Result: Query executed successfully (no rows returned)")
        
        # Error info if present
        if "error" in result:
            response_parts.append(f"Error: {result['error']}")
        
        return "\n".join(response_parts)
    
    def _get_adaptive_temperature_llm(self, similar_sqls: List[Dict]):
        """
        Create LLM instance with simple 2-level adaptive temperature.
        
        Temperature Strategy (simplified):
        - High similarity (≥0.70): Low temperature (0.1) - follow conversation patterns closely
        - Low similarity (<0.70): Medium temperature (0.2) - more creative exploration
        
        Since examples are in conversation history, the LLM naturally focuses on the most
        similar ones (which appear last). Temperature just adds a safety valve for low similarity.
        """
        try:
            from langchain_openai import ChatOpenAI
            import os
            
            # Get best similarity score from retrieved SQLs
            best_similarity = 0
            if similar_sqls and len(similar_sqls) > 0:
                best_similarity = similar_sqls[0].get('similarity', 0) if isinstance(similar_sqls[0], dict) else 0
            
            # Simple 2-level temperature
            if best_similarity >= 0.70:
                temperature = 0.1
                temp_reason = f"high similarity ({best_similarity:.3f}) - follow conversation patterns"
            else:
                temperature = 0.2
                temp_reason = f"low similarity ({best_similarity:.3f}) - allow more creativity"
            
            logger.info(f"Temperature: {temperature} ({temp_reason})")
            
            # Create adaptive LLM instance
            adaptive_llm = ChatOpenAI(
                model="gpt-4.1",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,  # Fixed at 0.1 for maximum accuracy
                max_tokens=2000
            )
            
            return adaptive_llm
            
        except Exception as e:
            logger.warning(f"Failed to create adaptive LLM: {e}, using original LLM")
            return self.llm
    
    # NOTE: _filter_relevant_examples() method removed
    # We no longer filter examples since ALL 20 retrieved SQLs are injected into conversation history.
    # The LLM naturally focuses on the most similar ones (which appear last in conversation).
    
    def _clean_previous_results(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean previous_results to remove non-serializable objects like DataFrames.
        Converts DataFrames to list of dicts for LLM consumption.
        """
        if not previous_results:
            return None
        
        import pandas as pd
        cleaned = {}
        
        logger.info(f" Cleaning previous_results: {len(previous_results)} step(s)")
        
        for step_key, step_data in previous_results.items():
            if isinstance(step_data, dict):
                cleaned_step = {}
                logger.info(f"   Cleaning {step_key}: {list(step_data.keys())}")
                for key, value in step_data.items():
                    # Convert DataFrame to list of dicts
                    if isinstance(value, pd.DataFrame):
                        converted = value.to_dict('records')
                        cleaned_step[key] = converted
                        logger.info(f"       Converted DataFrame '{key}' to {len(converted)} records")
                    elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        cleaned_step[key] = value
                    else:
                        try:
                            import json
                            json.dumps(value)
                            cleaned_step[key] = value
                        except (TypeError, ValueError):
                            # If it fails, convert to string representation
                            cleaned_step[key] = str(value)[:500]  
                            logger.info(f"      Converted non-serializable '{key}' to string")
                
                cleaned[step_key] = cleaned_step
            else:
                # If step_data itself is not a dict, try to keep it if serializable
                try:
                    import json
                    json.dumps(step_data)
                    cleaned[step_key] = step_data
                except (TypeError, ValueError):
                    cleaned[step_key] = str(step_data)[:500]
        
        logger.info(f" Cleaning complete: {len(cleaned)} step(s) cleaned")
        return cleaned
    
    def _extract_year_from_examples(self, examples: List[Dict]) -> int:
        """
        Extract the data year from example SQL queries.
        Looks for patterns like '2024-09-01' in WHERE clauses.
        Falls back to current year if not found.
        """
        from datetime import datetime
        
        if not examples:
            return datetime.now().year
        
        for ex in examples:
            sql = ex.get('sql', '')
            year_matches = re.findall(r"'(202[0-9])-\d{2}-\d{2}'", sql)
            if year_matches:
                return int(year_matches[0])
            
            timestamp_matches = re.findall(r"TIMESTAMP\s+'(202[0-9])-\d{2}-\d{2}'", sql, re.IGNORECASE)
            if timestamp_matches:
                return int(timestamp_matches[0])
        
        logger.info(f"No year found in examples, using current year: {datetime.now().year}")
        return datetime.now().year
    
    def _create_focused_prompt(self, question: str, examples: List[Dict], intent: Dict[str, Any], 
                              previous_results: Dict[str, Any] = None,
                              entity_info: Dict[str, Any] = None,
                              focused_schema: str = None) -> str:
        """
        Create a focused prompt that emphasizes accuracy and proper SQL generation.
        
        Args:
            entity_info: Dictionary containing verified entity information
            focused_schema: Optional focused schema containing only relevant tables
        """
        data_year = self._extract_year_from_examples(examples)
        
        # Note: Retrieved SQL examples are now injected into conversation history
        # (not in the system prompt). This makes LLM pay more attention to them.
        
        # Build previous context (if multi-step query)
        # Note: Query enrichment is handled by RedisMemoryManager.enrich_query()
        # This section just provides minimal context for reference
        previous_context = ""
        if previous_results:
            previous_context = "\n📋 PREVIOUS STEP RESULTS:\n"
            previous_context += "Note: The current query has already been enriched with context from previous steps.\n"
            previous_context += "If the query mentions specific values or filters, use them directly.\n\n"
            
            for step_key, step_data in previous_results.items():
                if isinstance(step_data, dict):
                    previous_context += f"{step_key.upper()}: {step_data.get('question', 'N/A')}\n"
                    if 'sql' in step_data:
                        prev_sql = step_data['sql'][:200]
                        previous_context += f"SQL: {prev_sql}...\n"
                    if 'rows_returned' in step_data:
                        previous_context += f"Rows returned: {step_data['rows_returned']}\n"
                    previous_context += "\n"
        
        # Add entity context from verification (if available)
        entity_context = ""
        if entity_info and entity_info.get("verified_entities"):
            entity_context = "\n" + "="*80 + "\n"
            entity_context += "🎯 VERIFIED ENTITY INFORMATION\n"
            entity_context += "="*80 + "\n"
            entity_context += "The entity verification agent has confirmed these entities exist in the database.\n"
            entity_context += "Use the EXACT names as shown below (they are case-sensitive in the database).\n\n"
            
            verified_entities = entity_info.get("verified_entities", {})
            
            for entity_name, entity_details in verified_entities.items():
                entity_type = entity_details.get("type", "")
                db_name = entity_details.get("db_name", entity_name)
                
                entity_context += f"ENTITY: '{db_name}'\n"
                entity_context += f"   Type: {entity_type}\n"
                entity_context += f"   WARNING: Use this EXACT name (case-sensitive): '{db_name}'\n\n"
                
                # Provide general guidance based on entity type
                if "ViewCustomer" in entity_type:
                    entity_context += "   Usage: Filter with WHERE ViewCustomer.name = '{db_name}'\n"
                    entity_context += "   Join: CROSS JOIN ViewCustomer with CustomerInvoice using ViewCustomer.external_id = CustomerInvoice.externalCode\n"
                elif "ViewDistributor" in entity_type:
                    entity_context += "   Usage: Filter with WHERE ViewDistributor.name = '{db_name}'\n"
                    entity_context += "   Join: CROSS JOIN ViewDistributor with relevant sales tables\n"
                elif "Sku" in entity_type:
                    entity_context += f"   Usage: Filter with WHERE Sku.name = '{db_name}'\n"
                else:
                    # Generic guidance for any other entity type
                    entity_context += f"   Usage: Filter using the appropriate WHERE clause with this exact value\n"
                
                entity_context += "\n"
            
            entity_context += "="*80 + "\n"
        
        # Note: Intent guidance removed - examples in conversation history show patterns better
        # Note: Retrieved SQL examples are now part of conversation history (not in prompt)
        # This makes LLM pay more attention to them as "recent successful queries"
        
        return f"""You are an expert SQL generator for Cube.js.

🎯 CRITICAL INSTRUCTION - READ THIS FIRST
================================================================================
The conversation history above contains REAL SQL examples from this database.
Examples are ordered by similarity score (ASCENDING order).
→ The MOST RECENT message in conversation = HIGHEST similarity = BEST pattern to follow

Your PRIMARY task: COPY the pattern from the MOST RECENT example (last message before current question).

⚠️ The MOST RECENT example has the HIGHEST similarity score.
⚠️ If similarity > 0.70, you MUST copy that example's structure exactly.

REQUIRED STEPS:
1. Check the MOST RECENT example (last message before current question)
2. What TABLE does it use? → YOU use the SAME table
3. Does it use CROSS JOIN ViewCustomer? → YOU use CROSS JOIN ViewCustomer
4. Copy the WHERE clause pattern, date logic, and structure
5. Change ONLY the specific date ranges for the current question

⛔ CRITICAL TABLE RULES:
- Question about "visits"? → Use ViewCustomerActivity (from examples)
- Question about "sales/orders"? → Use CustomerInvoice (from examples)
- Need customer names? → MUST use "CROSS JOIN ViewCustomer"
- DO NOT use tables not shown in examples above

⛔ CRITICAL CROSS JOIN RULE:
If example shows: "CROSS JOIN ViewCustomer" to get ViewCustomer.name
→ YOU MUST also use "CROSS JOIN ViewCustomer"
→ NEVER access ViewCustomer.name without CROSS JOIN

Example Pattern (COPY THIS):
```sql
SELECT DISTINCT ViewCustomer.name 
FROM ViewCustomerActivity 
CROSS JOIN ViewCustomer
WHERE ViewCustomerActivity.activityTime >= ...
```

YEAR RULE: Current data year is {data_year}. Use this year in date filters.

{entity_context}

DATABASE SCHEMA:
{self._get_schema_context(focused_schema)}

RETURN FORMAT: {{"sql": "YOUR_QUERY_HERE"}}

Now generate SQL by COPYING the most similar example's pattern."""
    
    def _parse_and_validate_response(self, content: str, question: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM response and validate the generated SQL.
        """
        try:
            # Extract SQL from JSON response - try multiple approaches
            # Approach 1: Try to find JSON block with proper escaping
            json_match = re.search(r'\{[^{}]*"sql"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    # If JSON parsing fails due to escape sequences, try extracting SQL directly
                    logger.warning(f"JSON parsing failed: {e}, attempting direct SQL extraction")
                    sql_match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', json_match.group(0), re.DOTALL)
                    if sql_match:
                        sql = sql_match.group(1)
                        # Unescape the SQL string
                        sql = sql.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                        parsed = {"sql": sql}
                    else:
                        # Try markdown code block extraction as fallback
                        sql_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
                        if sql_block_match:
                            sql = sql_block_match.group(1).strip()
                            parsed = {"sql": sql}
                        else:
                            raise
                
                if "sql" in parsed:
                    sql = parsed["sql"].strip()
                    
                    # Validate SQL is not a placeholder
                    if self._is_placeholder_sql(sql):
                        logger.error(f"Generated SQL is a placeholder: {sql}")
                        return {
                            "success": False,
                            "error": "LLM generated placeholder SQL with WHERE 1=1",
                            "type": "validation_error",
                            "invalid_sql": sql
                        }
                    
                    # Validate SQL matches intent
                    if self._validate_sql_intent(sql, intent):
                        return {
                            "success": True,
                            "sql": sql,
                            "query_type": "SELECT",
                            "explanation": "Generated with LLM and validated",
                            "method": "llm_validated",
                            "intent": intent
                        }
                    else:
                        logger.warning("Generated SQL doesn't match intent")
                        return {
                            "success": False,
                            "error": "Generated SQL doesn't match detected intent",
                            "type": "validation_error",
                            "invalid_sql": sql
                        }
            
            # Approach 2: Try to extract SQL from markdown code block
            sql_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
            if sql_block_match:
                sql = sql_block_match.group(1).strip()
                logger.info("Extracted SQL from markdown code block")
                if self._validate_sql_intent(sql, intent):
                    return {
                        "success": True,
                        "sql": sql,
                        "query_type": "SELECT",
                        "explanation": "Generated with LLM (extracted from markdown)",
                        "method": "llm_markdown_extraction",
                        "intent": intent
                    }
            
            # Approach 3: Check if content itself is raw SQL
            if content.upper().strip().startswith(('SELECT', 'WITH')):
                sql = content.strip()
                logger.info("Content appears to be raw SQL")
                if self._validate_sql_intent(sql, intent):
                    return {
                        "success": True,
                        "sql": sql,
                        "query_type": "SELECT",
                        "explanation": "Generated with LLM (raw SQL)",
                        "method": "llm_raw_sql",
                        "intent": intent
                    }
            
            # If all parsing attempts fail, return error
            logger.error("Failed to parse LLM response using all methods")
            return {
                "success": False,
                "error": "Failed to parse LLM response - no valid JSON, markdown, or raw SQL found",
                "type": "parsing_error",
                "response": content[:200]
            }
            
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return {
                "success": False,
                "error": f"Failed to parse response: {str(e)}",
                "type": "parsing_error",
                "response": content[:300]
            }
    
    def _is_placeholder_sql(self, sql: str) -> bool:
        """
        Check if the SQL is a useless placeholder query.
        Returns True if SQL is invalid/placeholder.
        """
        sql_lower = sql.lower().strip()
        
        # Check for WHERE 1=1 with minimal logic
        if 'where 1=1' in sql_lower or 'where 1 = 1' in sql_lower:
            # Check if there's any real filtering beyond WHERE 1=1
            where_clause = sql_lower.split('where')[-1]
            # If WHERE clause only has "1=1" and nothing else meaningful, it's a placeholder
            if where_clause.strip().replace('1=1', '').replace('=', '').replace('1', '').strip() == '':
                logger.warning("Detected placeholder SQL with WHERE 1=1 only")
                return True
        
        if 'sum(' in sql_lower and 'totalsum' in sql_lower and 'where 1=1' in sql_lower:
            logger.warning("Detected generic SUM query placeholder")
            return True
        
        return False
    
    def _validate_sql_intent(self, sql: str, intent: Dict[str, Any]) -> bool:
        """
        Basic SQL validation - checks for placeholders and obvious issues.
        Note: Intent-based validation removed - conversation history guides LLM better.
        """
        sql_lower = sql.lower()
        
        # Check for placeholder SQL
        if self._is_placeholder_sql(sql):
            logger.warning("SQL is a placeholder query")
            return False
        
        # Check for excessive CROSS JOINs (more than 6 is suspicious)
        cross_join_count = sql_lower.count('cross join')
        if cross_join_count > 6:
            logger.warning(f"SQL has too many CROSS JOINs: {cross_join_count}")
            return False
        
        # Basic sanity check: SQL should start with SELECT or WITH
        if not sql.strip().upper().startswith(('SELECT', 'WITH')):
            logger.warning(f"SQL doesn't start with SELECT or WITH")
            return False
        
        logger.info("SQL passed basic validation")
        return True
    
    def process(self, state: BaseAgentState) -> DBAgentState:
        """
        Process method required by BaseAgent interface.
        """
        db_state = DBAgentState(**state)
        
        try:
            similar_sqls = state.get("retrieved_sql_context", [])
            previous_results = state.get("previous_step_results", None)
            original_query = state.get("original_query", state["query"])
            
            result = self.generate_sql(
                question=state["query"],
                similar_sqls=similar_sqls,
                previous_results=previous_results,
                original_query=original_query
            )
            
            if result["success"]:
                db_state["query_type"] = result["query_type"]
                db_state["sql_query"] = result["sql"]
                db_state["status"] = "completed"
                db_state["success_message"] = result["explanation"]
                db_state["result"] = result
                
                logger.info(f"SQL Generated using {result.get('method', 'unknown')} method")
                logger.info(f"Generated SQL: {result['sql'][:100]}...")
                
            else:
                db_state["error_message"] = result["error"]
                db_state["status"] = "failed"
                db_state["result"] = result
                
        except Exception as e:
            db_state["error_message"] = f"Improved SQL generator error: {str(e)}"
            db_state["status"] = "failed"
            logger.error(f"ImprovedSQLGenerator process failed: {e}")
        
        return db_state