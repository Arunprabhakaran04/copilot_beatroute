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
                     original_query: str = None) -> Dict[str, Any]:
        """
        Generate SQL query with improved context handling.
        Always uses LLM generation for accuracy (simple generation disabled).
        """
        try:
            enhanced_question = self._preserve_original_context(
                question, original_query or question
            )
            
            intent = self._detect_query_intent(enhanced_question)
            
            # ALWAYS use LLM generation 
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
        if previous_results:
            logger.info(f" PREVIOUS_RESULTS received: {list(previous_results.keys())}")
            for key, val in previous_results.items():
                if isinstance(val, dict):
                    logger.info(f"   {key}: keys={list(val.keys())}")
                    if 'data' in val:
                        logger.info(f"      data type: {type(val['data'])}, sample: {str(val['data'])[:200]}")
        else:
            logger.info(" PREVIOUS_RESULTS: None")
        
        cleaned_previous_results = self._clean_previous_results(previous_results)
        
        relevant_examples = self._filter_relevant_examples(similar_sqls, intent)
        
        # Create focused prompt with previous results context
        prompt = self._create_focused_prompt(question, relevant_examples, intent, cleaned_previous_results)
        
        message_log = [{"role": "system", "content": prompt}]
        message_log.append({"role": "user", "content": f"Generate SQL for: {question}"})
        
        # ADAPTIVE TEMPERATURE: Adjust creativity based on similarity and relevance
        adaptive_llm = self._get_adaptive_temperature_llm(similar_sqls, relevant_examples)
        
        # Generate response with adaptive temperature
        response = adaptive_llm.invoke(message_log)
        content = response.content.strip()
        
        track_llm_call(
            input_prompt=message_log,
            output=content,
            agent_type="improved_sql_generator",
            operation="generate_sql",
            model_name="gpt-4o"
        )
        
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
            
            if best_similarity > 0.85 and relevance_ratio > 0.7:
                temperature = 0.05  #
                temp_reason = f"high sim ({best_similarity:.3f}) + high rel ({relevance_ratio:.2f}) - follow patterns"
            elif best_similarity > 0.75 and relevance_ratio > 0.5:
                temperature = 0.1   
                temp_reason = f"good sim ({best_similarity:.3f}) + decent rel ({relevance_ratio:.2f}) - guided creativity"
            elif best_similarity > 0.65 or relevance_ratio > 0.3:
                temperature = 0.2  
                temp_reason = f"medium factors (sim: {best_similarity:.3f}, rel: {relevance_ratio:.2f}) - balanced approach"
            elif best_similarity > 0.5:
                temperature = 0.3  
                temp_reason = f"low similarity ({best_similarity:.3f}) - increased creativity"
            else:
                temperature = 0.4  
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
            
            matches_table = intent['table'].lower() in sql_lower
            matches_aggregation = intent['aggregation'].lower() in sql_lower
            
            if matches_table and sql_info.get('similarity', 0) > 0.7:
                relevant.append(sql_info)
        
        return relevant[:3]  # Max 3 relevant examples
    
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
        Falls back to current year - 1 if not found.
        """
        from datetime import datetime
        
        if not examples:
            # Default to previous year (data is usually from previous year)
            return datetime.now().year - 1
        
        for ex in examples:
            sql = ex.get('sql', '')
            year_matches = re.findall(r"'(202[0-9])-\d{2}-\d{2}'", sql)
            if year_matches:
                return int(year_matches[0])
            
            timestamp_matches = re.findall(r"TIMESTAMP\s+'(202[0-9])-\d{2}-\d{2}'", sql, re.IGNORECASE)
            if timestamp_matches:
                return int(timestamp_matches[0])
        
        # Default to previous year if no year found in examples
        logger.warning("No year found in examples, defaulting to previous year")
        return datetime.now().year - 1
    
    def _create_focused_prompt(self, question: str, examples: List[Dict], intent: Dict[str, Any], 
                              previous_results: Dict[str, Any] = None) -> str:
        """
        Create a focused prompt that emphasizes accuracy and proper SQL generation.
        """
        data_year = self._extract_year_from_examples(examples)
        
        examples_text = ""
        if examples:
            examples_text = "\n🔍 RELEVANT EXAMPLES (Study these patterns - they show correct table choices and YEAR):\n\n"
            for i, ex in enumerate(examples, 1):
                examples_text += f"Example {i} (Similarity: {ex.get('similarity', 0):.3f}):\n"
                examples_text += f"Question: {ex['question']}\n"
                examples_text += f"SQL: {ex['sql'][:400]}...\n\n"
            
            examples_text += f" KEY LEARNING: Examples use year {data_year} - this is the DATA YEAR, use it!\n"
            examples_text += " KEY LEARNING: Notice which tables these examples use (CustomerInvoice vs DistributorSales)\n"
            examples_text += f" KEY LEARNING: For dates in {data_year}, use WHERE conditions with '{data_year}-MM-DD' format\n\n"
        
        # Get current date for reference only (NOT for queries)
        from datetime import datetime
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_year = datetime.now().year
        
        # Extract actual SKU values from previous results
        previous_context = ""
        extracted_skus = []
        if previous_results:
            previous_context = "\n PREVIOUS STEP RESULTS (MUST USE THESE IN YOUR QUERY):\n"
            for step_key, step_data in previous_results.items():
                if isinstance(step_data, dict):
                    previous_context += f"\n{step_key.upper()}: {step_data.get('question', 'N/A')}\n"
                    
                    if 'sql' in step_data:
                        prev_sql = step_data['sql'][:300]
                        previous_context += f"SQL: {prev_sql}...\n"
                    
                    # Extract SKU values from data if available
                    # Handle both 'data' (list) and 'query_data' (DataFrame) formats
                    data = None
                    if 'data' in step_data:
                        data = step_data['data']
                        logger.info(f"      Found 'data' field with type: {type(data)}")
                    elif 'query_data' in step_data:
                        # Convert DataFrame to list of dicts
                        import pandas as pd
                        query_data = step_data['query_data']
                        logger.info(f"      Found 'query_data' field with type: {type(query_data)}")
                        if isinstance(query_data, pd.DataFrame):
                            data = query_data.to_dict('records')
                            logger.info(f"      Converted DataFrame to {len(data)} records")
                        else:
                            data = query_data
                    else:
                        logger.info(f"      No 'data' or 'query_data' found. Available keys: {list(step_data.keys())}")
                    
                    if data and isinstance(data, list) and len(data) > 0:
                        logger.info(f"      Processing {len(data)} rows for SKU extraction")
                        # Try to find SKU column (skuName, sku_name, SKU, etc.)
                        for row in data:
                            if isinstance(row, dict):
                                for key, value in row.items():
                                    if 'sku' in key.lower() and value and str(value) not in extracted_skus:
                                        extracted_skus.append(str(value))
                                        logger.info(f"      Extracted SKU: {value} from field '{key}'")
                            
                            if extracted_skus:
                                sku_list = "', '".join(extracted_skus)
                                previous_context += f" EXTRACTED SKUs: {extracted_skus}\n"
                                previous_context += f"   YOU MUST FILTER: WHERE skuName IN ('{sku_list}')\n"
                    
            if extracted_skus:
                sku_list = "', '".join(extracted_skus)
                previous_context += f"\n CRITICAL: Filter by these exact SKUs: {extracted_skus}\n"
                previous_context += f"   Add to your query: WHERE skuName IN ('{sku_list}')\n"
            else:
                previous_context += "\n IMPORTANT: If the current question references 'step 1' or 'identified in step X', you MUST use those results.\n"
                previous_context += "For example: If step 1 identified SKUs ['SKU-A', 'SKU-B', 'SKU-C'], filter WHERE skuName IN ('SKU-A', 'SKU-B', 'SKU-C')\n"
        
        # Add intent-specific guidance
        intent_guidance = ""
        
        if intent.get("requires_sku_join"):
            intent_guidance += "\n- Query requires SKU data from CustomerInvoiceDetail table\n"
            intent_guidance += "- CustomerInvoiceDetail has: skuName, skuAmount, dispatch_date, base_quantity, sellingQuantity\n"
        
        if intent.get("requires_top_n"):
            intent_guidance += f"- Return top {intent.get('top_n_value', 'N')} results: Use ORDER BY and LIMIT {intent.get('top_n_value', 'N')}\n"
        
        if intent.get("time_range_months"):
            intent_guidance += f"- Data needed for last {intent.get('time_range_months')} months from {data_year}\n"
            intent_guidance += f"- Calculate range: DATE_TRUNC('month', dispatch_date) >= '{data_year}-01-01'\n"
        
        if intent.get("grouping") == "month":
            intent_guidance += "- Monthly breakdown needed: Use DATE_TRUNC('month', dispatch_date) in SELECT and GROUP BY\n"
            intent_guidance += f"- For CustomerInvoiceDetail: GROUP BY DATE_TRUNC('month', dispatch_date), skuName\n"
        
        if intent.get("months_requested"):
            months_str = ', '.join(intent['months_requested'])
            intent_guidance += f"- Specific months requested: {months_str} in year {data_year}\n"
        
        # Specific guidance for multi-step monthly trend queries
        if extracted_skus and intent.get("grouping") == "month":
            sku_list = "', '".join(extracted_skus)
            intent_guidance += f"\n🎯 MULTI-STEP MONTHLY TREND PATTERN:\n"
            intent_guidance += f"   SELECT\n"
            intent_guidance += f"     DATE_TRUNC('month', dispatch_date) AS month,\n"
            intent_guidance += f"     skuName,\n"
            intent_guidance += f"     MEASURE(SUM(skuAmount)) AS total_sales\n"
            intent_guidance += f"   FROM CustomerInvoiceDetail\n"
            intent_guidance += f"   WHERE skuName IN ('{sku_list}')\n"
            intent_guidance += f"     AND DATE_TRUNC('year', dispatch_date) = DATE_TRUNC('year', TIMESTAMP '{data_year}-01-01')\n"
            intent_guidance += f"   GROUP BY 1, 2\n"
            intent_guidance += f"   ORDER BY 1, 2\n"
        
        return f"""You are an expert SQL generator for Cube.js. Generate ACCURATE, COMPLETE queries.

📅 DATA YEAR: {data_year} (this is the year in your database, NOT current year {current_year})
📅 USE {data_year} for ALL date filters, NOT {current_year}!

CRITICAL RULES FOR CUBE.JS:
1. 🔍 Study the EXAMPLES carefully - they show correct table patterns and use year {data_year}
2. 📅 YEAR RULE: Your data is from {data_year}. Use dates like '{data_year}-09-01', NOT '{current_year}-09-01'
3. 📊 TABLE RULE for SKU queries:
   - Customer sales/invoices → CustomerInvoice + CustomerInvoiceDetail
   - Distributor sales → DistributorSales + DistributorSalesDetail
   - Monthly trends with SKUs → CustomerInvoiceDetail (has dispatch_date + skuName + skuAmount)
4. 🔗 JOIN RULE: Use CROSS JOIN only (never INNER/LEFT/RIGHT JOIN with ON)
5. 📐 AGGREGATION RULE: Use MEASURE(SUM(...)) inside WITH clause or direct SELECT
6. 📝 GROUP BY RULE: Can be inside or outside WITH clause, both work

QUERY REQUIREMENTS:
- Aggregation type: {intent.get('aggregation', 'N/A')}
- Time grouping: {intent.get('grouping', 'none')}
- Top N required: {intent.get('requires_top_n', False)}
{intent_guidance}

{examples_text}
{previous_context}

SCHEMA (relevant tables):
CustomerInvoiceDetail:
  - dispatch_date (timestamp) - USE THIS for date grouping
  - skuName (text) - USE THIS for SKU filtering
  - skuAmount (numeric) - USE THIS for sales amount
  - base_quantity, sellingQuantity (numeric)
  
CustomerInvoice:
  - dispatchedDate (timestamp)
  - dispatchedvalue (numeric)
  - year, monthYear, quarter

⚠️ CRITICAL CHECKLIST:
✓ Using year {data_year} in WHERE clause (NOT {current_year})
✓ Using correct table (CustomerInvoiceDetail for SKU monthly trends)
✓ If previous steps found SKUs, filtering WHERE skuName IN (...)
✓ For monthly grouping: DATE_TRUNC('month', dispatch_date)
✓ NO INNER/LEFT/RIGHT JOIN syntax (use CROSS JOIN if needed)

RETURN FORMAT: {{"sql": "YOUR_QUERY_HERE"}}

IMPORTANT: Generate a COMPLETE, ACCURATE SQL query. Do NOT use WHERE 1=1 as a placeholder."""
    
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