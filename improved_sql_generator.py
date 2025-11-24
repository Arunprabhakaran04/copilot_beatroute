import re
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from base_agent import BaseAgent, BaseAgentState, DBAgentState
from token_tracker import track_llm_call
from loguru import logger

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
        """Use ONLY focused schema if available, otherwise return empty"""
        if focused_schema:
            logger.info("Using focused schema from SchemaManager")
            return focused_schema
        
        logger.warning("No focused schema available")
        return ""
    
    def generate_sql(self, question: str, similar_sqls: List[str] = None, 
                     previous_results: Dict[str, Any] = None, 
                     original_query: str = None,
                     entity_info: Dict[str, Any] = None,
                     conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SQL query using LLM with retrieved examples as context.
        
        Args:
            entity_info: Dictionary containing verified entity information
            conversation_history: List of previous conversation entries from RedisMemoryManager
        """
        try:
            return self._generate_with_llm(question, similar_sqls, previous_results, entity_info, conversation_history)
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return {
                "success": False,
                "error": f"SQL generation error: {str(e)}",
                "type": "generation_error"
            }
    
    def _generate_with_llm(self, question: str, similar_sqls: List[Dict], 
                          previous_results: Dict[str, Any],
                          entity_info: Dict[str, Any] = None,
                          conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SQL using LLM with retrieved examples injected as conversation history.
        
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
        
        # ✅ PERFORMANCE OPTIMIZATION: Use cached focused schema if available
        # Check if we have a cached schema passed from EnrichAgent via state
        cached_schema = entity_info.get("cached_focused_schema") if entity_info else None
        focused_schema = None
        focused_tables = []
        
        if cached_schema:
            # Use cached schema from EnrichAgent (no regeneration needed)
            focused_schema = cached_schema
            import re
            focused_tables = re.findall(r'^Table: (.+)$', focused_schema, re.MULTILINE)
            logger.info(f"💾 Using cached focused schema from EnrichAgent: {len(focused_tables)} tables (0.00s)")
        elif self.schema_manager is not None:
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
        
        # Calculate top similarity for adaptive prompt
        top_similarity = 0.0
        if similar_sqls and len(similar_sqls) > 0:
            sorted_sqls = sorted(similar_sqls, key=lambda x: x.get('similarity', 0))
            top_similarity = sorted_sqls[-1].get('similarity', 0) if sorted_sqls else 0
        
        # Create focused prompt with previous results and entity context
        prompt = self._create_focused_prompt(
            question, 
            similar_sqls,
            previous_results=cleaned_previous_results, 
            entity_info=entity_info,
            focused_schema=focused_schema,
            top_similarity=top_similarity
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
        # Retrieved SQLs are injected as user-assistant pairs with JSON format
        message_log = self._build_message_log_with_history(
            system_prompt=prompt,
            current_question=user_message,
            conversation_history=conversation_history,
            retrieved_sqls=similar_sqls,  # Pass retrieved SQLs for injection
            max_history=5,  # Keep last 5 REAL conversations
            max_retrieved_sqls=20  # Inject up to 20 retrieved SQLs (increased from 10)
        )
        
        # Calculate top similarity and determine dynamic temperature
        top_similarity = 0.0
        if similar_sqls and len(similar_sqls) > 0:
            num_examples = min(len(similar_sqls), 20)
            sorted_sqls = sorted(similar_sqls[:num_examples], key=lambda x: x.get('similarity', 0))
            top_similarity = sorted_sqls[-1].get('similarity', 0) if sorted_sqls else 0
            
            logger.info(f"Passed {num_examples} examples to LLM as user-assistant pairs")
            logger.info(f"   Top similarity (last example): {top_similarity:.3f}")
        
        # 🎯 DYNAMIC TEMPERATURE based on similarity score
        if top_similarity > 0.80:
            temperature = 0.1  # High similarity - deterministic, copy pattern closely
            creativity_level = "LOW (COPY pattern)"
        elif top_similarity >= 0.60:
            temperature = 0.4  # Moderate similarity - moderate creativity, adapt pattern
            creativity_level = "MODERATE (ADAPT pattern)"
        else:
            temperature = 0.6  # Low similarity - high creativity, be inventive
            creativity_level = "HIGH (BE CREATIVE)"
        
        logger.info(f"🌡️ Dynamic temperature: {temperature} (similarity: {top_similarity:.3f}, creativity: {creativity_level})")
        
        # Generate SQL with dynamic temperature
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            seed=42,
            max_tokens=2000
        )
        
        response = llm.invoke(message_log)
        content = response.content.strip()
        
        track_llm_call(
            input_prompt=message_log,
            output=content,
            agent_type="improved_sql_generator",
            operation="generate_sql",
            model_name="gpt-4.1-mini"
        )
        
        # Log execution timing
        execution_time = time.time() - start_time
        logger.info(f"SQL generation completed in {execution_time:.2f}s")
        
        return self._parse_and_validate_response(content, question)
    
    def _build_message_log_with_history(
        self,
        system_prompt: str,
        current_question: str,
        conversation_history: List[Dict[str, Any]] = None,  # Kept for compatibility but not used
        retrieved_sqls: List[Dict[str, Any]] = None,
        max_history: int = 5,  # Not used anymore
        max_retrieved_sqls: int = 20
    ) -> List[Dict[str, str]]:
        """
        Build message log with ONLY retrieved SQLs (no real conversation history).
        Each query gets fresh context with only the 20 most similar examples.
        
        Strategy: Examples appear as real Q->A pairs with assistant returning JSON format.
        Retrieved SQLs are sorted in ASCENDING order of similarity (most similar last).
        
        Args:
            system_prompt: System prompt with instructions
            current_question: Current user question
            conversation_history: NOT USED - kept for compatibility
            retrieved_sqls: List of retrieved SQL examples from embedding search
            max_history: NOT USED - kept for compatibility
            max_retrieved_sqls: Maximum number of retrieved SQLs to inject (default 20)
            
        Returns:
            Message log in format: [{"role": "system/user/assistant", "content": str}, ...]
        """
        message_log = [{"role": "system", "content": system_prompt}]
        
        # Inject ONLY retrieved SQLs as user-assistant pairs (ASCENDING similarity order)
        # NO real conversation history - keeps context focused on similar patterns
        if retrieved_sqls and len(retrieved_sqls) > 0:
            # Sort by similarity ASCENDING (lowest first, highest last)
            sorted_sqls = sorted(retrieved_sqls[:max_retrieved_sqls], key=lambda x: x.get('similarity', 0))
            
            logger.info(f"Injecting {len(sorted_sqls)} retrieved SQLs as user-assistant pairs (ASCENDING similarity)")
            logger.info(f"   Similarity range: {sorted_sqls[0].get('similarity', 0):.3f} (first) -> {sorted_sqls[-1].get('similarity', 0):.3f} (last/most similar)")
            
            # Log top 3 and bottom 1 to verify what LLM is seeing
            logger.info(f"   Sample injected SQLs:")
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
                
                # Add as assistant's JSON response (CRITICAL: JSON format with sql key)
                sql = sql_info.get('sql', '')
                # Escape the SQL for JSON format
                escaped_sql = json.dumps(sql)
                assistant_json = f'{{"sql": {escaped_sql}}}'
                message_log.append({"role": "assistant", "content": assistant_json})
        else:
            logger.info("No retrieved SQLs to inject")
        
        # Add CURRENT question with explicit COPY instructions
        if retrieved_sqls and len(retrieved_sqls) > 0:
            sorted_sqls = sorted(retrieved_sqls[:max_retrieved_sqls], key=lambda x: x.get('similarity', 0))
            most_similar = sorted_sqls[-1]
            similarity = most_similar.get('similarity', 0)
            most_similar_question = most_similar.get('question', '')
            most_similar_sql = most_similar.get('sql', '')
            
            # Add STRICT copy instruction with FULL SQL pattern shown (no truncation)
            # Show complete SQL so LLM can copy exact structure
            
            enhanced_question = f"""{current_question}

🎯 CRITICAL: Your TEMPLATE is immediately above (similarity: {similarity:.3f})
Template Question: "{most_similar_question}"

Template SQL Pattern (COMPLETE - NO TRUNCATION):
```sql
{most_similar_sql}
```

⚠️ MANDATORY COPYING RULES:
1. ✅ COPY the WITH clause structure (how many CTEs, their names)
2. ✅ COPY the FROM + CROSS JOIN pattern (same tables, same order)  
3. ✅ COPY the WHERE clause structure (AND logic, date comparisons)
4. ✅ COPY the main SELECT logic (NOT IN, columns, grouping)
5. ✅ ONLY modify: specific dates (e.g., '2025-10-01' instead of CURRENT_DATE - INTERVAL)
6. ❌ Do NOT use OR conditions in WHERE if template uses AND
7. ❌ Do NOT combine multiple time periods in one CTE if template separates them
8. ❌ Do NOT add extra complexity (nested subqueries, extra joins)

🔴 CRITICAL: If query asks for "X but not Y", use TWO separate CTEs:
   - CTE 1: Filter for X only
   - CTE 2: Filter for Y only
   - Main SELECT: FROM CTE1 WHERE NOT IN (CTE2)
   
DO NOT use OR to combine time periods - this creates cartesian products!

The template SQL WORKS. Copy its structure. Change ONLY dates/values. Nothing else."""
            
            message_log.append({"role": "user", "content": enhanced_question})
        else:
            message_log.append({"role": "user", "content": current_question})
        
        total_messages = len(message_log)
        injected_sql_count = len(retrieved_sqls) * 2 if retrieved_sqls else 0
        
        logger.info(f"FINAL message log: {total_messages} messages")
        logger.info(f"   = 1 system + {injected_sql_count} injected SQLs + 1 current")
        logger.info(f"   (No real conversation history - fresh context per query)")
        
        # DEBUG: Log first 2 and last 2 injected examples to verify format
        if retrieved_sqls and len(retrieved_sqls) > 0:
            logger.info("DEBUG: Verifying injected example format:")
            logger.info("   FIRST 2 examples (lowest similarity):")
            example_count = 0
            for i, msg in enumerate(message_log):
                if msg["role"] == "user" and i > 0 and example_count < 2:
                    if i + 1 < len(message_log) and message_log[i + 1]["role"] == "assistant":
                        logger.info(f"      Example {example_count + 1}:")
                        logger.info(f"         USER: {msg['content'][:80]}...")
                        logger.info(f"         ASSISTANT: {message_log[i + 1]['content'][:150]}...")
                        example_count += 1
            
            logger.info("   LAST 2 examples (highest similarity - most important):")
            example_positions = []
            for i, msg in enumerate(message_log):
                if msg["role"] == "user" and i > 0 and i + 1 < len(message_log):
                    if message_log[i + 1]["role"] == "assistant":
                        example_positions.append(i)
            
            if len(example_positions) >= 2:
                for idx, pos in enumerate(example_positions[-2:], 1):
                    logger.info(f"      Example (last-{3-idx}):")
                    logger.info(f"         USER: {message_log[pos]['content'][:80]}...")
                    logger.info(f"         ASSISTANT: {message_log[pos + 1]['content'][:200]}...")
        
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
    
    def _create_focused_prompt(self, question: str, examples: List[Dict],
                              previous_results: Dict[str, Any] = None,
                              entity_info: Dict[str, Any] = None,
                              focused_schema: str = None,
                              top_similarity: float = 0.0) -> str:
        """
        Create a focused prompt that emphasizes accuracy and proper SQL generation.
        
        Args:
            entity_info: Dictionary containing verified entity information
            focused_schema: Optional focused schema containing only relevant tables
            top_similarity: Highest similarity score from retrieved examples (for adaptive instructions)
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
        
        schema_content = self._get_schema_context(focused_schema)
        schema_warning = ""
        if not schema_content or len(schema_content.strip()) == 0:
            schema_warning = "\n⚠️ WARNING: No schema available - you MUST copy field names EXACTLY from the examples above!\n"
        
        # Build few-shot examples section with top 3 most similar SQLs
        few_shot_examples = ""
        if examples and len(examples) > 0:
            # Get top 3 most similar examples (they're already sorted by similarity)
            top_examples = sorted(examples, key=lambda x: x.get('similarity', 0), reverse=True)[:3]
            
            few_shot_examples = "\n" + "="*80 + "\n"
            few_shot_examples += "📚 FEW-SHOT EXAMPLES - TOP 3 MOST SIMILAR PATTERNS\n"
            few_shot_examples += "="*80 + "\n"
            few_shot_examples += "These are PROVEN, TESTED SQL patterns that work correctly in CubeJS.\n"
            few_shot_examples += "YOU MUST FOLLOW THESE PATTERNS EXACTLY.\n\n"
            
            for idx, example in enumerate(top_examples, 1):
                question = example.get('question', 'N/A')
                sql = example.get('sql', 'N/A')
                similarity = example.get('similarity', 0)
                
                few_shot_examples += f"EXAMPLE {idx} (Similarity: {similarity:.3f}):\n"
                few_shot_examples += f"User Question: {question}\n"
                few_shot_examples += f"Correct SQL Pattern:\n{sql}\n"
                few_shot_examples += f"\nKey Pattern Elements to COPY:\n"
                
                # Extract pattern elements
                if 'CROSS JOIN' in sql:
                    cross_joins = sql.count('CROSS JOIN')
                    few_shot_examples += f"  ✓ Uses {cross_joins} CROSS JOIN(s) - YOU MUST USE SAME NUMBER\n"
                if 'WITH' in sql.upper():
                    few_shot_examples += f"  ✓ Uses WITH clause (CTE) - YOU CAN USE THIS PATTERN\n"
                if 'ViewCustomer' in sql:
                    few_shot_examples += f"  ✓ Uses ViewCustomer table - COPY THIS IF YOUR QUERY INVOLVES CUSTOMERS\n"
                if 'NOT IN' in sql.upper():
                    few_shot_examples += f"  ⚠️ Uses NOT IN - ONLY USE IF ABSOLUTELY NECESSARY\n"
                if 'GROUP BY' in sql.upper():
                    few_shot_examples += f"  ✓ Uses GROUP BY - COPY THIS PATTERN IF AGGREGATING\n"
                
                few_shot_examples += "\n"
            
            few_shot_examples += "="*80 + "\n\n"
        
        # 🎯 Generate adaptive instructions based on similarity score
        if top_similarity > 0.75:
            adaptive_instructions = """🔒 HIGH SIMILARITY (>0.75) - COPY MODE:
1. COPY the FROM clause WORD-FOR-WORD from the most similar example
2. COPY the JOIN structure CHARACTER-BY-CHARACTER from the example
3. COPY the WHERE clause structure from the example
4. Only modify: date filters, entity names, and minor adjustments
5. Do NOT add patterns the example doesn't show
6. TRUST the example - it's proven to work"""
        elif top_similarity >= 0.50:
            adaptive_instructions = """🔄 MODERATE SIMILARITY (0.50-0.75) - ADAPT MODE:
1. USE the example as a GUIDE, not strict template
2. ADAPT the logic to fit your specific question
3. Be creative with WHERE conditions if question logic differs
4. Keep the same table selection and JOIN pattern
5. Think critically: Does example logic match question logic?
6. If logic doesn't match, redesign the approach while keeping SQL style"""
        else:
            adaptive_instructions = """🚀 LOW SIMILARITY (<0.50) - CREATIVE MODE:
1. Examples are just REFERENCE - be creative and independent
2. Design SQL from scratch based on question requirements
3. Follow CubeJS rules (CROSS JOIN, no forbidden functions)
4. Think through the logic step-by-step
5. Use simple patterns: WITH clause + CROSS JOIN + clean WHERE
6. Don't force-fit an example pattern that doesn't match"""
        
        return f"""You are an expert Cube.js SQL generator.

CRITICAL: Must output JSON only: {{"sql": "..."}}

=== CUBE.JS SQL API STRICT RULES (FAILURE = QUERY REJECTED) ===

🚫 ABSOLUTE PROHIBITIONS (THESE WILL FAIL):
1. ❌ NO JOIN CONDITIONS: ONLY use CROSS JOIN, NO ON clause
   - WRONG: FROM Order JOIN ViewCustomer ON Order.customerId = ViewCustomer.id
   - RIGHT: FROM Order CROSS JOIN ViewCustomer
   - ⚠️ IMPORTANT: You CAN and MUST use WHERE filters on dates, entity names, etc!
   - RIGHT: FROM Order CROSS JOIN ViewCustomer WHERE Order.datetime >= '2024-08-01' AND ViewCustomer.name = 'ABC Corp'
   
2. ❌ NO SUBQUERIES IN WHERE CLAUSE OF MAIN SELECT: Especially NOT IN (SELECT ...) or IN (SELECT ...)
   - WRONG: SELECT * FROM ViewCustomer WHERE name IN (SELECT customer FROM Order WHERE ...)
   - WRONG: SELECT * FROM ViewCustomer WHERE name NOT IN (SELECT customer FROM Order WHERE ...)
   - RIGHT: Use WITH clause to create CTEs, then use NOT IN with CTE names
   - RIGHT: WITH cte AS (...) SELECT * FROM main_table WHERE field NOT IN (SELECT field FROM cte)
   
3. ❌ NO FORBIDDEN FUNCTIONS: TO_CHAR(), EXTRACT(), COALESCE(), GETDATE()
   
4. ❌ NO SELECT *: Must explicitly list columns
   
5. ❌ NO ALIASES IN WHERE: Cannot reference column aliases in WHERE clause
   
6. ❌ NO NESTED SUBQUERIES: Subqueries inside other subqueries in WHERE/HAVING

=== MANDATORY PATTERNS (ONLY THESE WORK) ===

✅ WITH Clause Pattern (for complex queries):
   - Define CTEs first with WITH clause
   - Use MEASURE() inside WITH clause for aggregations
   - Use SUM() outside WITH clause (MEASURE doesn't work outside)
   - Always assign aliases in WITH clause
   - Include GROUP BY inside WITH clause
   - ⚠️ DO NOT create 2 tables with WITH and then JOIN them - use UNION ALL instead
   
🔴 CRITICAL CTE SCOPE RULE:
   - Tables referenced INSIDE a CTE are NOT accessible OUTSIDE the CTE
   - Main SELECT can ONLY reference: (1) CTE name, (2) CTE output columns
   - WRONG: WITH cte AS (SELECT Table.col AS x FROM Table) SELECT Table.col FROM cte
   - RIGHT: WITH cte AS (SELECT Table.col AS x FROM Table) SELECT x FROM cte
   - This is a UNIVERSAL SQL rule - violating it causes "field not found" errors
   
✅ CROSS JOIN Pattern:
   - Use CROSS JOIN ONLY (no other join types)
   - NO ON clause ever
   - NO join conditions in WHERE clause
   - Tables are automatically related by CROSS JOIN
   
✅ Table Relationships (use CROSS JOIN for these):
   - Order ↔ ViewCustomer, ViewDistributor, ViewUser, OrderDetail
   - OrderDetail ↔ Sku
   - Sku ↔ Brand, Category
   - CustomerInvoice ↔ CustomerInvoiceDetail, ViewCustomer
   
✅ Name Column Patterns:
   - Customer names: CROSS JOIN ViewCustomer, use ViewCustomer.name
   - User names: CROSS JOIN ViewUser, use ViewUser.name
   - Distributor names: CROSS JOIN ViewDistributor, use ViewDistributor.name
   - SKU names: CROSS JOIN Sku, use Sku.name (NOT Order.skuName or CustomerInvoiceDetail.skuName)
   - Category names: CROSS JOIN Category with Sku, use Category.name
   - Brand names: CROSS JOIN Brand with Sku, use Brand.name
   
✅ Date Field Patterns:
   - Order dates: Order.datetime
   - Sales/Revenue/Dispatch dates: CustomerInvoice.dispatchedDate
   
🚨 CRITICAL: Hardcoded Date Handling (October, November, specific months):
   ❌ WRONG - String literals may fail due to timezone/format:
      WHERE Order.datetime >= '2025-10-01' AND Order.datetime < '2025-11-01'
   
   ✅ CORRECT - Use DATE_TRUNC for specific months/years:
      -- For October 2025:
      WHERE Order.datetime >= DATE_TRUNC('month', DATE '2025-10-01')
        AND Order.datetime < DATE_TRUNC('month', DATE '2025-10-01') + INTERVAL '1 month'
      
      -- For November 2025:
      WHERE Order.datetime >= DATE_TRUNC('month', DATE '2025-11-01')
        AND Order.datetime < DATE_TRUNC('month', DATE '2025-11-01') + INTERVAL '1 month'
   
   ⚠️ WHY: DATE_TRUNC ensures proper date boundaries and handles timezone correctly.
   String literals like '2025-10-01' may not match database datetime format.
   
   📋 PATTERN TO COPY:
      DATE_TRUNC('month', DATE 'YYYY-MM-DD') for start of month
      DATE_TRUNC('month', DATE 'YYYY-MM-DD') + INTERVAL '1 month' for end of month
   
✅ Count Patterns:
   - WRONG: COUNT(DISTINCT ViewCustomer.id)
   - RIGHT: MEASURE(ViewCustomer.count)

🎯 CRITICAL PATTERN: "Items in Period A but NOT in Period B"
This is YOUR CURRENT TASK TYPE - PAY CLOSE ATTENTION!

⚠️ CARTESIAN PRODUCT WARNING - THIS IS YOUR MAIN ERROR:
The most common failure is creating unfiltered CROSS JOINs that produce cartesian products.

❌ WRONG - UNFILTERED CROSS JOIN (CAUSES CARTESIAN PRODUCT):
WITH order_dates AS (
  SELECT ViewCustomer.name AS CustomerName, DATE_TRUNC('month', Order.datetime) AS OrderMonth
  FROM Order CROSS JOIN ViewCustomer  -- ❌ NO WHERE CLAUSE = EVERY ORDER × EVERY CUSTOMER!
  WHERE Order.datetime >= ... OR Order.datetime <= ...  -- ❌ Filters AFTER join
  GROUP BY ViewCustomer.name, DATE_TRUNC('month', Order.datetime)
)
-- This will crash: millions of rows in cartesian product before WHERE filters

❌ WRONG - OR COMBINING TIME PERIODS (CAUSES CARTESIAN PRODUCT):
WITH combined_dates AS (
  FROM Order CROSS JOIN ViewCustomer
  WHERE (Order.datetime >= '2025-10-01' AND Order.datetime < '2025-11-01')
     OR (Order.datetime >= '2025-11-01' AND Order.datetime < '2025-12-01')  -- ❌ OR creates BOTH periods together!
)
-- This creates: every order in Oct × every customer + every order in Nov × every customer = HUGE cartesian product

❌ WRONG - SUBQUERY IN MAIN WHERE CLAUSE:
SELECT name FROM ViewCustomer 
WHERE name NOT IN (SELECT name FROM Order CROSS JOIN ViewCustomer WHERE date >= ...)  -- ❌ SUBQUERY!

✅ CORRECT APPROACH (METHOD 1 - Two CTEs with NOT IN):
-- Example: Customers with orders in October 2025 but not November 2025
WITH october_orders AS (
  SELECT DISTINCT ViewCustomer.name AS customer_name
  FROM Order
  CROSS JOIN ViewCustomer
  WHERE Order.datetime >= DATE_TRUNC('month', DATE '2025-10-01')
    AND Order.datetime < DATE_TRUNC('month', DATE '2025-10-01') + INTERVAL '1 month'
), november_orders AS (
  SELECT DISTINCT ViewCustomer.name AS customer_name
  FROM Order
  CROSS JOIN ViewCustomer
  WHERE Order.datetime >= DATE_TRUNC('month', DATE '2025-11-01')
    AND Order.datetime < DATE_TRUNC('month', DATE '2025-11-01') + INTERVAL '1 month'
)
SELECT customer_name AS CustomerName  -- ✅ Use CTE column (NOT ViewCustomer.name!)
FROM october_orders
WHERE customer_name NOT IN (SELECT customer_name FROM november_orders)
ORDER BY CustomerName
LIMIT 1000;

KEY POINTS FOR METHOD 1:
- ✅ Each CTE filters by date range IN THE CTE WHERE CLAUSE (not in main SELECT)
- ✅ Main SELECT uses NOT IN with CTE names (NOT subqueries)
- ✅ CROSS JOIN is used but WITH proper WHERE filters on dates
- ✅ Main SELECT references CTE columns (customer_name), NOT original table columns (ViewCustomer.name)
- ❌ NEVER create unfiltered CROSS JOIN - always add WHERE for dates/entities
- ❌ NEVER reference original tables in main SELECT after FROM cte_name

✅ CORRECT APPROACH (METHOD 2 - LEFT JOIN with NULL check):
WITH last_month_orders AS (
  SELECT DISTINCT ViewCustomer.name AS customer_name
  FROM Order
  CROSS JOIN ViewCustomer
  WHERE Order.datetime >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
    AND Order.datetime < DATE_TRUNC('month', CURRENT_DATE)
), this_month_orders AS (
  SELECT DISTINCT ViewCustomer.name AS customer_name
  FROM Order
  CROSS JOIN ViewCustomer
  WHERE Order.datetime >= DATE_TRUNC('month', CURRENT_DATE)
)
SELECT lm.customer_name AS CustomerName
FROM last_month_orders lm
LEFT JOIN this_month_orders tm ON lm.customer_name = tm.customer_name
WHERE tm.customer_name IS NULL
ORDER BY CustomerName
LIMIT 1000;

KEY DIFFERENCES THAT PREVENT CARTESIAN PRODUCTS:
- ✅ Each CTE MUST have WHERE clause filtering dates INSIDE the CTE
- ✅ WHERE filters applied BEFORE CROSS JOIN result is materialized
- ✅ Main SELECT uses NOT IN with CTE names (NOT inline subqueries)
- ✅ Main SELECT references CTE columns ONLY (NOT original table columns)
- ❌ NEVER: CROSS JOIN without WHERE filters (creates cartesian product)
- ❌ NEVER: WHERE x NOT IN (SELECT ... FROM ... CROSS JOIN ... WHERE ...) in main SELECT
- ❌ NEVER: GROUP BY before filtering dates (groups cartesian product)
- ❌ NEVER: SELECT Table.column FROM cte_name (Table doesn't exist outside CTE!)

🔴 THE #1 MISTAKE: Unfiltered CROSS JOIN
If you write "FROM Order CROSS JOIN ViewCustomer" without WHERE filtering in the SAME CTE/query block,
you create a cartesian product of EVERY order × EVERY customer, which crashes the system.

🔴 THE #2 MISTAKE: Wrong Column Reference After CTE
If you write "WITH cte AS (SELECT Table.col AS alias ...) SELECT Table.col FROM cte",
you get "No field named 'Table.col'" error because Table only exists INSIDE the CTE.
Always use: SELECT alias FROM cte

SCHEMA:
{schema_content}
{schema_warning}

DATABASE YEAR: {data_year}
{entity_context}
{previous_context}

{few_shot_examples}

CRITICAL PATTERN MATCHING INSTRUCTIONS:
You will see example queries in the conversation history before the current question.
These examples are sorted by similarity - the MOST RECENT example (just before the current question) is the MOST SIMILAR.

🎯 ADAPTIVE STRATEGY (Based on Similarity Score: {top_similarity:.3f}):
{adaptive_instructions}

⚠️ LOGIC VALIDATION:
- ALWAYS verify your SQL makes logical sense
- NEVER create contradictory conditions (e.g., X NOT IN (...) AND X IN (...))
- If the question's logic differs from examples, ADAPT the pattern, don't blindly copy
- Think through: "Does this SQL actually answer the question?"

SIMPLICITY RULE:
Simpler SQL is better. If you can answer with one CTE and one CROSS JOIN (like examples), 
do NOT add more complexity. More complexity = higher chance of failure.

OUTPUT FORMAT: {{"sql": "YOUR_QUERY_HERE"}}"""
    
    def _parse_and_validate_response(self, content: str, question: str) -> Dict[str, Any]:
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
                    
                    # 🔴 CRITICAL: Validate no unfiltered CROSS JOINs and CTE scope violations
                    if not self._validate_sql_intent(sql, {}):
                        logger.error("SQL validation failed - structural issue detected")
                        return {
                            "success": False,
                            "error": "Generated SQL has structural issues (unfiltered CROSS JOIN or CTE scope violation)",
                            "type": "validation_error",
                            "invalid_sql": sql,
                            "fix_hint": "1) Add WHERE clause to filter CROSS JOIN before GROUP BY, OR 2) Use CTE column aliases (not original table.column) in main SELECT after FROM cte_name"
                        }
                    
                    return {
                        "success": True,
                        "sql": sql,
                        "query_type": "SELECT",
                        "explanation": "Generated with LLM",
                        "method": "llm_validated"
                    }
            
            # Approach 2: Try to extract SQL from markdown code block
            sql_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
            if sql_block_match:
                sql = sql_block_match.group(1).strip()
                logger.info("Extracted SQL from markdown code block")
                return {
                    "success": True,
                    "sql": sql,
                    "query_type": "SELECT",
                    "explanation": "Generated with LLM (extracted from markdown)",
                    "method": "llm_markdown_extraction"
                }
            
            # Approach 3: Check if content itself is raw SQL
            if content.upper().strip().startswith(('SELECT', 'WITH')):
                sql = content.strip()
                logger.info("Content appears to be raw SQL")
                return {
                    "success": True,
                    "sql": sql,
                    "query_type": "SELECT",
                    "explanation": "Generated with LLM (raw SQL)",
                    "method": "llm_raw_sql"
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
        
        # 🔴 CRITICAL: Check for unfiltered CROSS JOINs (cartesian product risk)
        # Pattern: CROSS JOIN followed by another CROSS JOIN or GROUP BY without WHERE clause in between
        if 'cross join' in sql_lower:
            # Split SQL into CTEs and main query
            parts = re.split(r'\bwith\b|\bselect\b', sql_lower, flags=re.IGNORECASE)
            
            for part in parts:
                if 'cross join' in part:
                    # Check if this CROSS JOIN block has WHERE clause before GROUP BY or end
                    # Extract the portion from CROSS JOIN to next major clause
                    cross_join_pos = part.find('cross join')
                    remaining = part[cross_join_pos:]
                    
                    # Find next major clause (GROUP BY, ORDER BY, or end)
                    group_by_pos = remaining.find('group by')
                    order_by_pos = remaining.find('order by')
                    next_clause_pos = min([p for p in [group_by_pos, order_by_pos, len(remaining)] if p > 0])
                    
                    cross_join_block = remaining[:next_clause_pos]
                    
                    # Check if WHERE exists in this block
                    if 'where' not in cross_join_block:
                        logger.error("🔴 CARTESIAN PRODUCT DETECTED: CROSS JOIN without WHERE clause!")
                        logger.error(f"   Problematic block: {cross_join_block[:200]}...")
                        logger.warning("SQL likely to cause 'Join Error: task panicked' - rejecting")
                        return False
                    
                    # 🔴 SMART OR CHECK: Only reject OR combining date ranges in CROSS JOIN
                    # Pattern: WHERE (date >= X AND date < Y) OR (date >= Z AND date < W)
                    # This is BAD because it creates cartesian product for BOTH time periods
                    # But OR for entity conditions (name = 'A' OR name = 'B') is OK
                    if 'where' in cross_join_block and ' or ' in cross_join_block:
                        # Extract WHERE clause
                        where_clause = cross_join_block[cross_join_block.find('where'):]
                        
                        # Check if OR is combining date range conditions (date >= X AND date < Y) OR (date >= Z)
                        # Look for pattern: parentheses with date comparisons separated by OR
                        # Match: (datetime >= 'X' AND datetime < 'Y') OR (datetime >= 'Z' AND datetime < 'W')
                        or_date_range_pattern = r'\([^)]*datetime[^)]*>=[^)]*AND[^)]*<[^)]*\)\s+OR\s+\([^)]*datetime[^)]*>=[^)]*'
                        
                        if re.search(or_date_range_pattern, where_clause, re.IGNORECASE):
                            logger.error("🔴 CARTESIAN PRODUCT: OR combining DATE RANGES in CROSS JOIN!")
                            logger.error(f"   Pattern: (date >= X AND date < Y) OR (date >= Z) creates cartesian product")
                            logger.error(f"   Use TWO separate CTEs instead: one per time range, then NOT IN")
                            logger.error(f"   Note: OR for entity filters (name = 'A' OR name = 'B') is OK, but not date ranges")
                            logger.warning("SQL likely to cause 'dimensions.includes' error - rejecting")
                            return False
                        else:
                            # OR exists but not with date ranges - this is OK (e.g., name = 'A' OR name = 'B')
                            logger.info("ℹ️ OR condition detected but not with date ranges - allowing (likely entity filter)")
                    
                    # 🔴 NEW CHECK: Hardcoded date strings without DATE_TRUNC (can cause format mismatch)
                    # Pattern: WHERE datetime >= '2025-10-01' (string literal)
                    # Should be: WHERE datetime >= DATE_TRUNC('month', DATE '2025-10-01')
                    if 'where' in cross_join_block:
                        where_clause = cross_join_block[cross_join_block.find('where'):]
                        # Look for datetime comparison with quoted date string (not using DATE_TRUNC)
                        hardcoded_date_pattern = r"datetime\s*[><=]+\s*'\d{4}-\d{2}-\d{2}"
                        if re.search(hardcoded_date_pattern, where_clause, re.IGNORECASE):
                            # Check if DATE_TRUNC is NOT used
                            if 'date_trunc' not in where_clause.lower():
                                logger.warning("⚠️ HARDCODED DATE STRING detected without DATE_TRUNC!")
                                logger.warning(f"   Pattern: datetime >= '2025-XX-XX' may cause format/timezone mismatch")
                                logger.warning(f"   Recommend: Use DATE_TRUNC('month', DATE '2025-XX-XX') instead")
                                logger.info("🔧 Allowing but flagging for potential issue (not rejecting)")
                                # Don't reject - just warn, as it might work in some databases
        
        # 🔴 CRITICAL: Check for CTE scope violations (referencing original tables after FROM cte_name)
        if 'with' in sql_lower and 'as' in sql_lower:
            # Extract CTE names
            cte_pattern = r'with\s+(\w+)\s+as\s*\('
            cte_names = re.findall(cte_pattern, sql_lower)
            
            if cte_names:
                # Find main SELECT (after all CTEs)
                # Split by ') SELECT' or ') \nSELECT' to get main query
                main_select_match = re.search(r'\)\s*select\s+', sql_lower)
                if main_select_match:
                    main_select_start = main_select_match.end()
                    main_select = sql_lower[main_select_start:]
                    
                    # Extract FROM clause in main SELECT
                    from_match = re.search(r'\bfrom\s+(\w+)', main_select)
                    if from_match:
                        from_table = from_match.group(1)
                        
                        # If FROM references a CTE, check if SELECT references original tables
                        if from_table in cte_names:
                            # Check for table.column patterns in SELECT clause (before FROM)
                            select_clause = main_select[:main_select.find('from')]
                            # Look for patterns like ViewCustomer.name, Order.value, etc.
                            table_ref_pattern = r'\b[A-Z]\w+\.[a-z_]+\b'
                            table_refs = re.findall(table_ref_pattern, select_clause)
                            
                            if table_refs:
                                logger.error(f"🔴 CTE SCOPE VIOLATION: Main SELECT references original table columns after FROM {from_table}")
                                logger.error(f"   Problematic references: {table_refs}")
                                logger.error(f"   FROM {from_table} means you can ONLY use columns defined in {from_table} CTE")
                                logger.warning("SQL likely to cause 'No field named' error - rejecting")
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