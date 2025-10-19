import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from base_agent import BaseAgent, BaseAgentState, DBAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

class ImprovedSQLGenerator(BaseAgent):
    """
    Improved SQL Generator with robust context handling and simplification logic.
    Focuses on generating accurate, simple queries with better multi-step support.
    """
    
    def __init__(self, llm, schema_file_path: str = "schema"):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.schema_content = self._load_schema_file()
        
        # Core query patterns for simple operations
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
        try:
            with open(self.schema_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                content = content.replace('\\n', '\n')
                
                logger.info(f"Successfully loaded schema file: {self.schema_file_path}")
                return content
        except Exception as e:
            logger.error(f"Error loading schema file: {e}")
            return "Schema file not found. Unable to load database schema."
    
    def _preserve_original_context(self, current_question: str, original_query: str) -> str:
        """
        Preserve the original query context when dealing with generic multi-step tasks.
        This prevents loss of specific details during decomposition.
        """
        current_lower = current_question.lower()
        original_lower = original_query.lower()
        
        # If current question is generic but original was specific, enhance it
        if ("get" in current_lower or "retrieve" in current_lower) and len(original_lower) > len(current_lower):
            # Extract specific details from original query
            preserved_details = []
            
            # Extract time periods (months)
            time_matches = re.findall(r'(august|september|october|november|december|january|february|march|april|may|june|july)', original_lower)
            if time_matches:
                preserved_details.extend([f"month: {month}" for month in set(time_matches)])
            
            # Extract years
            year_matches = re.findall(r'(202[0-9])', original_lower)
            if year_matches:
                preserved_details.extend([f"year: {year}" for year in set(year_matches)])
            
            # Extract time ranges
            if 'last 12 months' in original_lower or '12 months' in original_lower:
                preserved_details.append("time range: last 12 months")
            elif 'last 6 months' in original_lower or '6 months' in original_lower:
                preserved_details.append("time range: last 6 months")
            elif 'last 3 months' in original_lower or '3 months' in original_lower:
                preserved_details.append("time range: last 3 months")
            
            # Extract SKU references
            if 'sku' in original_lower:
                if 'top 3 sku' in original_lower or 'top 3 sku' in current_lower:
                    preserved_details.append("requirement: top 3 SKUs")
                elif 'top' in original_lower and any(char.isdigit() for char in original_lower):
                    top_n = re.search(r'top\s*(\d+)', original_lower)
                    if top_n:
                        preserved_details.append(f"requirement: top {top_n.group(1)} SKUs")
            
            # Extract specific terms like "separately", "monthly", "trend", etc.
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
        
        intent = {
            "pattern": None,
            "table": "CustomerInvoice",
            "date_column": "dispatchedDate", 
            "value_column": "dispatchedvalue",
            "aggregation": "SUM",
            "grouping": None,
            "time_filter": None,
            "is_simple": False,  # Changed to False - always use LLM
            "requires_sku_join": False,
            "requires_top_n": False,
            "top_n_value": None
        }
        
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
        
        # Detect time periods
        if any(month in question_lower for month in ['august', 'september', 'october']):
            months = []
            if 'august' in question_lower:
                months.append("'2025-08-01'")
            if 'september' in question_lower:
                months.append("'2025-09-01'")
            if 'october' in question_lower:
                months.append("'2025-10-01'")
            
            if len(months) > 1 and ('separately' in question_lower or 'each' in question_lower):
                # Multiple months separately - group by month
                intent["pattern"] = "monthly_sales"
                intent["grouping"] = "month"
                intent["time_filter"] = f"DATE_TRUNC('month', CustomerInvoice.dispatchedDate) IN ({', '.join(months)})"
            else:
                # Single period or combined
                intent["pattern"] = "total_sales"
                if len(months) == 1:
                    intent["time_filter"] = f"DATE_TRUNC('month', CustomerInvoice.dispatchedDate) = {months[0]}"
                else:
                    intent["time_filter"] = f"DATE_TRUNC('month', CustomerInvoice.dispatchedDate) IN ({', '.join(months)})"
        
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
    
    def _generate_simple_query(self, question: str, intent: Dict[str, Any]) -> str:
        """
        Generate a simple, focused SQL query based on detected intent.
        """
        if intent["pattern"] and intent["pattern"] in self.core_patterns:
            template = self.core_patterns[intent["pattern"]]
            return template.format(date_filter=intent["time_filter"] or "1=1")
        
        # Fallback: construct simple query
        table = intent["table"]
        date_col = f"{table}.{intent['date_column']}"
        value_col = f"{table}.{intent['value_column']}"
        aggregation = intent["aggregation"]
        
        if intent["grouping"] == "month":
            return f"""
            SELECT
                DATE_TRUNC('month', {date_col}) AS MonthYear,
                {aggregation}({value_col}) AS Total{aggregation}
            FROM {table}
            WHERE {intent['time_filter'] or '1=1'}
            GROUP BY DATE_TRUNC('month', {date_col})
            ORDER BY MonthYear
            """.strip()
        else:
            return f"""
            SELECT {aggregation}({value_col}) AS Total{aggregation}
            FROM {table}
            WHERE {intent['time_filter'] or '1=1'}
            """.strip()
    
    def _should_use_simple_generation(self, question: str, similar_sqls: List[Dict]) -> bool:
        """
        Determine if we should use simple pattern-based generation instead of LLM.
        DISABLED: Simple generation creates incorrect SQL with WHERE 1=1 fallbacks.
        Always use LLM generation for accuracy.
        """
        # DISABLED: Simple generation produces incorrect SQL for multi-step queries
        # Always return False to force LLM generation
        logger.info(f"Simple generation DISABLED - using LLM for all queries")
        return False
    
    def generate_sql(self, question: str, similar_sqls: List[str] = None, 
                     previous_results: Dict[str, Any] = None, 
                     original_query: str = None) -> Dict[str, Any]:
        """
        Generate SQL query with improved context handling.
        Always uses LLM generation for accuracy (simple generation disabled).
        """
        try:
            # Preserve original context for multi-step queries
            enhanced_question = self._preserve_original_context(
                question, original_query or question
            )
            
            # Detect query intent
            intent = self._detect_query_intent(enhanced_question)
            
            # ALWAYS use LLM generation (simple generation disabled due to accuracy issues)
            return self._generate_with_llm(enhanced_question, similar_sqls, previous_results, intent)
            
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return {
                "success": False,
                "error": f"SQL generation error: {str(e)}",
                "type": "generation_error"
            }
    
    def _generate_with_llm(self, question: str, similar_sqls: List[Dict], 
                          previous_results: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SQL using LLM with focused prompts and better validation.
        """
        # Build focused examples based on intent
        relevant_examples = self._filter_relevant_examples(similar_sqls, intent)
        
        # Create focused prompt with previous results context
        prompt = self._create_focused_prompt(question, relevant_examples, intent, previous_results)
        
        message_log = [{"role": "system", "content": prompt}]
        message_log.append({"role": "user", "content": f"Generate SQL for: {question}"})
        
        # ADAPTIVE TEMPERATURE: Adjust creativity based on similarity and relevance
        adaptive_llm = self._get_adaptive_temperature_llm(similar_sqls, relevant_examples)
        
        # Generate response with adaptive temperature
        response = adaptive_llm.invoke(message_log)
        content = response.content.strip()
        
        # Track token usage
        track_llm_call(
            input_prompt=message_log,
            output=content,
            agent_type="improved_sql_generator",
            operation="generate_sql",
            model_name="gpt-4o"
        )
        
        # Parse and validate response
        return self._parse_and_validate_response(content, question, intent)
    
    def _get_adaptive_temperature_llm(self, similar_sqls: List[Dict], relevant_examples: List[Dict]):
        """
        Create LLM instance with adaptive temperature based on similarity and relevance of examples.
        
        Temperature Strategy:
        - High relevance + high similarity: Low temperature (0.05) - stick to patterns  
        - Medium relevance: Medium temperature (0.15-0.2) - balanced approach
        - Low relevance + low similarity: High temperature (0.3-0.4) - creative exploration
        """
        try:
            from langchain_openai import ChatOpenAI
            import os
            
            # Calculate relevance score
            total_examples = len(similar_sqls) if similar_sqls else 0
            relevant_count = len(relevant_examples) if relevant_examples else 0
            relevance_ratio = relevant_count / total_examples if total_examples > 0 else 0
            
            # Get best similarity score
            best_similarity = 0
            if similar_sqls and len(similar_sqls) > 0:
                best_similarity = similar_sqls[0].get('similarity', 0) if isinstance(similar_sqls[0], dict) else 0
            
            # Determine temperature based on combined factors
            if best_similarity > 0.85 and relevance_ratio > 0.7:
                temperature = 0.05  # High similarity + high relevance = low temperature
                temp_reason = f"high sim ({best_similarity:.3f}) + high rel ({relevance_ratio:.2f}) - follow patterns"
            elif best_similarity > 0.75 and relevance_ratio > 0.5:
                temperature = 0.1   # Good similarity + decent relevance = low-medium temperature
                temp_reason = f"good sim ({best_similarity:.3f}) + decent rel ({relevance_ratio:.2f}) - guided creativity"
            elif best_similarity > 0.65 or relevance_ratio > 0.3:
                temperature = 0.2   # Medium similarity or some relevance = medium temperature
                temp_reason = f"medium factors (sim: {best_similarity:.3f}, rel: {relevance_ratio:.2f}) - balanced approach"
            elif best_similarity > 0.5:
                temperature = 0.3   # Low similarity = high temperature
                temp_reason = f"low similarity ({best_similarity:.3f}) - increased creativity"
            else:
                temperature = 0.4   # Very low similarity = maximum creativity
                temp_reason = f"very low similarity ({best_similarity:.3f}) - maximum creativity"
            
            logger.info(f"IMPROVED ADAPTIVE TEMP: {temperature} ({temp_reason})")
            
            # Create adaptive LLM instance
            adaptive_llm = ChatOpenAI(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temperature,
                max_tokens=2000
            )
            
            return adaptive_llm
            
        except Exception as e:
            logger.warning(f"Failed to create adaptive LLM: {e}, using original LLM")
            return self.llm
    
    def _filter_relevant_examples(self, similar_sqls: List[Dict], intent: Dict[str, Any]) -> List[Dict]:
        """
        Filter examples to only include those relevant to the detected intent.
        """
        if not similar_sqls:
            return []
        
        relevant = []
        for sql_info in similar_sqls[:5]:  # Only top 5
            sql_lower = sql_info.get('sql', '').lower()
            
            # Check if example matches intent
            matches_table = intent['table'].lower() in sql_lower
            matches_aggregation = intent['aggregation'].lower() in sql_lower
            
            if matches_table and sql_info.get('similarity', 0) > 0.7:
                relevant.append(sql_info)
        
        return relevant[:3]  # Max 3 relevant examples
    
    def _create_focused_prompt(self, question: str, examples: List[Dict], intent: Dict[str, Any], 
                              previous_results: Dict[str, Any] = None) -> str:
        """
        Create a focused prompt that emphasizes accuracy and proper SQL generation.
        """
        examples_text = ""
        if examples:
            examples_text = "RELEVANT EXAMPLES:\n"
            for i, ex in enumerate(examples, 1):
                examples_text += f"Example {i}: {ex['question']}\nSQL: {ex['sql']}\n\n"
        
        # Add previous results context for multi-step queries
        previous_context = ""
        if previous_results:
            previous_context = "\nPREVIOUS STEP RESULTS (for reference in multi-step queries):\n"
            for step_key, step_data in previous_results.items():
                if isinstance(step_data, dict):
                    previous_context += f"{step_key}: {step_data.get('question', 'N/A')}\n"
                    previous_context += f"SQL: {step_data.get('sql', 'N/A')[:200]}...\n"
        
        # Add intent-specific guidance
        intent_guidance = ""
        if intent.get("requires_sku_join"):
            intent_guidance += "\n- This query requires SKU data: JOIN with Sku table and use Sku.name\n"
        if intent.get("requires_top_n"):
            intent_guidance += f"- This query needs top {intent.get('top_n_value', 'N')} results: Use ORDER BY and LIMIT {intent.get('top_n_value', 'N')}\n"
        if intent.get("time_range_months"):
            intent_guidance += f"- This query needs data for last {intent.get('time_range_months')} months: Use appropriate date range filter\n"
        if intent.get("grouping") == "month":
            intent_guidance += "- This query needs monthly breakdown: Use DATE_TRUNC('month', ...) in SELECT and GROUP BY\n"
        
        return f"""You are an expert SQL generator for Cube.js. Generate ACCURATE, COMPLETE queries.

CRITICAL RULES:
1. NEVER generate SQL with WHERE 1=1 unless it's the complete, correct query
2. For SKU queries, ALWAYS join with Sku table and use Sku.name
3. For top N queries, use ORDER BY and LIMIT N
4. For date filtering, use DATE_TRUNC('month', ...) = 'YYYY-MM-01'
5. For multi-month ranges, use DATE_TRUNC('month', ...) BETWEEN 'start' AND 'end'
6. Use CROSS JOIN when necessary to connect tables
7. Use MEASURE() for aggregations in Cube.js
8. Return JSON format: {{"sql": "SQL_QUERY_HERE"}}

DETECTED INTENT:
- Primary table: {intent['table']}
- Aggregation: {intent['aggregation']}
- Grouping: {intent.get('grouping', 'none')}
- Requires SKU join: {intent.get('requires_sku_join', False)}
- Requires top N: {intent.get('requires_top_n', False)}
{intent_guidance}

{examples_text}
{previous_context}

SCHEMA (use relevant tables and columns):
{self.schema_content[:2000]}

IMPORTANT: Generate a COMPLETE, ACCURATE SQL query. Do NOT use WHERE 1=1 as a placeholder."""
    
    def _parse_and_validate_response(self, content: str, question: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse LLM response and validate the generated SQL.
        """
        try:
            # Extract SQL from JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
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
            
            # If parsing fails, return error
            logger.error("Failed to parse LLM response")
            return {
                "success": False,
                "error": "Failed to parse LLM response - no valid JSON found",
                "type": "parsing_error",
                "response": content[:200]
            }
            
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return {
                "success": False,
                "error": f"Failed to parse response: {str(e)}",
                "type": "parsing_error"
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
        
        # Check for generic SUM with no specifics
        if 'sum(' in sql_lower and 'totalsum' in sql_lower and 'where 1=1' in sql_lower:
            logger.warning("Detected generic SUM query placeholder")
            return True
        
        return False
    
    def _validate_sql_intent(self, sql: str, intent: Dict[str, Any]) -> bool:
        """
        Validate that generated SQL matches the detected intent.
        """
        sql_lower = sql.lower()
        
        # Check for placeholder SQL
        if self._is_placeholder_sql(sql):
            logger.warning("SQL is a placeholder query")
            return False
        
        # Check primary table is used (relaxed for multi-table queries)
        # Skip this check as many valid queries use multiple tables
        
        # Check for excessive CROSS JOINs (more than 5 is suspicious)
        cross_join_count = sql_lower.count('cross join')
        if cross_join_count > 5:
            logger.warning(f"SQL has too many CROSS JOINs: {cross_join_count}")
            return False
        
        # Check aggregation is present (if expected)
        if intent['aggregation'] and intent['aggregation'].lower() not in sql_lower:
            logger.warning(f"SQL missing expected aggregation: {intent['aggregation']}")
            return False
        
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