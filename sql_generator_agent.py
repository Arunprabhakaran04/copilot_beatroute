import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from base_agent import BaseAgent, BaseAgentState, DBAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

class SQLGeneratorAgent(BaseAgent):
    """
    Production-ready SQL generator for Cube.js API.
    Generates reliable, error-free SQL queries with strict validation.
    """
    
    def __init__(self, llm, schema_file_path: str = None):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.schema_content = self._load_schema_file() if schema_file_path else ""
        self.structured_schema = self._load_structured_schema()
        self.cube_js_rules = self._initialize_cube_js_rules()
    
    def get_agent_type(self) -> str:
        return "sql_generator"
    
    def _load_structured_schema(self) -> Dict[str, Any]:
        """Load enhanced schema with FK relationships and Cube.js rules."""
        try:
            import json
            with open('enhanced_schema.json', 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Count relationships
                total_relationships = sum(
                    len(table_data.get('relationships', {})) 
                    for table_data in data.get('tables', {}).values()
                )
                logger.info(f"Enhanced schema loaded: {data['metadata']['total_tables']} tables, {total_relationships} relationships")
                return data
        except FileNotFoundError:
            logger.warning("enhanced_schema.json not found, falling back to database_structure.json")
            try:
                with open('database_structure.json', 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    logger.info(f"Structured schema loaded: {data['metadata']['total_tables']} tables, {data['metadata']['total_columns']} columns")
                    return data
            except FileNotFoundError:
                logger.warning("database_structure.json not found, using text schema only")
                return {}
        except Exception as e:
            logger.error(f"Error loading structured schema: {e}")
            return {}
    
    def _initialize_cube_js_rules(self) -> Dict[str, Any]:
        """Initialize Cube.js specific validation rules."""
        return {
            "forbidden_patterns": [
                r"GROUP\s+BY.*MEASURE\(",
                r"MEASURE\(.*\).*GROUP\s+BY",
                r"INNER\s+JOIN",
                r"LEFT\s+JOIN",
                r"RIGHT\s+JOIN",
                r"FULL\s+JOIN",
                r"JOIN\s+\w+\s+ON",
                r"TO_CHAR\(",
                r"EXTRACT\(",
                r"COALESCE\(",
            ],
            "required_patterns": {
                "measure_in_with": r"WITH\s+\w+\s+AS\s*\([^)]*MEASURE\(",
                "cross_join_only": r"CROSS\s+JOIN",
            },
            "measure_fields": [
                "dispatchedvalue", "dispatchedqty", "invoiceamount", 
                "ordervalue", "quantity", "sku_value", "skuAmount"
            ]
        }
    
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
                
                logger.info(f"Schema loaded: {self.schema_file_path} ({len(content)} chars)")
                return content
        except FileNotFoundError:
            logger.warning(f"Schema file not found: {self.schema_file_path} - will use UserContext schema")
            return ""
        except Exception as e:
            logger.warning(f"Error loading schema: {e} - will use UserContext schema")
            return ""
    
    def get_schema_info(self) -> str:
        """Return the loaded schema information."""
        return self.schema_content
    
    def _clean_previous_results(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean previous_results to remove non-serializable objects like DataFrames.
        Converts DataFrames to list of dicts for JSON serialization.
        """
        if not previous_results:
            return None
        
        import pandas as pd
        cleaned = {}
        
        logger.info(f"🧹 SQL Generator cleaning previous_results: {len(previous_results)} step(s)")
        
        for step_key, step_data in previous_results.items():
            if isinstance(step_data, dict):
                cleaned_step = {}
                for key, value in step_data.items():
                    # Convert DataFrame to list of dicts
                    if isinstance(value, pd.DataFrame):
                        converted = value.to_dict('records')
                        cleaned_step[key] = converted
                        logger.info(f"      ✓ Converted DataFrame '{key}' to {len(converted)} records")
                    # Keep serializable types
                    elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        cleaned_step[key] = value
                    # For everything else, try to serialize or convert to string
                    else:
                        try:
                            json.dumps(value)
                            cleaned_step[key] = value
                        except (TypeError, ValueError):
                            cleaned_step[key] = str(value)[:500]
                            logger.info(f"      ⚠ Converted non-serializable '{key}' to string")
                
                cleaned[step_key] = cleaned_step
            else:
                try:
                    json.dumps(step_data)
                    cleaned[step_key] = step_data
                except (TypeError, ValueError):
                    cleaned[step_key] = str(step_data)[:500]
        
        return cleaned
    
    def _preserve_original_context(self, current_question: str, original_query: str) -> str:
        """
        Preserve context from original query when dealing with multi-step decomposition.
        Extracts time periods, SKU requirements, and grouping terms to prevent detail loss.
        Integrated from improved_sql_generator.py for better multi-step handling.
        """
        if not original_query or original_query == current_question:
            return current_question
        
        original_lower = original_query.lower()
        current_lower = current_question.lower()
        
        # Extract time period context
        time_periods = []
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']
        
        for month in months:
            if month in original_lower and month not in current_lower:
                time_periods.append(month)
        
        # Extract year context
        year_pattern = r'20\d{2}'
        original_years = re.findall(year_pattern, original_query)
        current_years = re.findall(year_pattern, current_question)
        missing_years = [y for y in original_years if y not in current_years]
        
        # Extract SKU requirements (top N)
        top_n_pattern = r'top\s+(\d+)'
        top_n_match = re.search(top_n_pattern, original_lower)
        current_top_n = re.search(top_n_pattern, current_lower)
        
        # Extract grouping requirements
        grouping_terms = ['separately', 'each month', 'monthly', 'by month', 'per month', 'trend']
        missing_grouping = [term for term in grouping_terms 
                           if term in original_lower and term not in current_lower]
        
        # Build enhanced question
        enhancements = []
        if time_periods:
            enhancements.append(f"month: {', '.join(time_periods)}")
        if missing_years:
            enhancements.append(f"year: {', '.join(missing_years)}")
        if top_n_match and not current_top_n:
            enhancements.append(f"requirement: top {top_n_match.group(1)} SKUs")
        if missing_grouping:
            enhancements.append(f"grouping: {', '.join(missing_grouping)}")
        
        if enhancements:
            enhanced_question = f"{current_question} ({'; '.join(enhancements)})"
            logger.info(f"Enhanced question with context: {enhanced_question}")
            return enhanced_question
        
        return current_question
    
    def _detect_query_intent(self, question: str) -> Dict[str, Any]:
        """
        Detect query intent for better pattern matching and validation.
        Returns intent dictionary with pattern, table, columns, and requirements.
        Integrated from improved_sql_generator.py for better pattern recognition.
        """
        question_lower = question.lower()
        
        intent = {
            "pattern": "basic",
            "table": "CustomerInvoice",
            "columns": [],
            "aggregation": "",
            "grouping": None,
            "requires_sku_join": False,
            "requires_top_n": False,
            "top_n_value": None,
            "time_range_months": None
        }
        
        # Detect SKU-related queries
        if any(term in question_lower for term in ['sku', 'product', 'item']):
            intent["requires_sku_join"] = True
            intent["columns"].append("Sku.name")
        
        # Detect top N requirements
        top_n_match = re.search(r'top\s+(\d+)', question_lower)
        if top_n_match:
            intent["requires_top_n"] = True
            intent["top_n_value"] = int(top_n_match.group(1))
        
        # Detect time periods
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                 'july', 'august', 'september', 'october', 'november', 'december']
        for month in months:
            if month in question_lower:
                intent["time_period"] = month
                break
        
        # Detect time range
        if any(term in question_lower for term in ['last', 'past', 'previous']):
            range_match = re.search(r'(?:last|past|previous)\s+(\d+)\s+months?', question_lower)
            if range_match:
                intent["time_range_months"] = int(range_match.group(1))
        
        # Detect aggregation type
        if 'sum' in question_lower or 'total' in question_lower:
            intent["aggregation"] = "SUM"
        elif 'average' in question_lower or 'avg' in question_lower:
            intent["aggregation"] = "AVG"
        elif 'count' in question_lower or 'number of' in question_lower:
            intent["aggregation"] = "COUNT"
        
        # Detect grouping requirements
        if any(term in question_lower for term in ['each month', 'monthly', 'by month', 'per month', 'separately']):
            intent["grouping"] = "month"
        elif any(term in question_lower for term in ['each year', 'yearly', 'by year', 'per year']):
            intent["grouping"] = "year"
        
        # Determine primary table
        if 'distributor' in question_lower or 'sales' in question_lower:
            intent["table"] = "DistributorSales"
        elif 'invoice' in question_lower or 'customer' in question_lower:
            intent["table"] = "CustomerInvoice"
        
        logger.info(f"Detected intent: {intent}")
        return intent
    
    def generate_sql(self, question: str, similar_sqls: List[str] = None, 
                     previous_results: Dict[str, Any] = None, 
                     original_query: str = None,
                     entity_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate production-ready SQL query for Cube.js.
        
        Args:
            question: User question to convert to SQL
            similar_sqls: List of similar SQL examples with similarity scores
            previous_results: Results from previous steps in multi-step workflow
            original_query: Original user query for context preservation in multi-step scenarios
            
        Returns:
            Dict with success status, sql query, and metadata
        """
        try:
            # Preserve context from original query if in multi-step scenario
            if original_query:
                question = self._preserve_original_context(question, original_query)
            
            # Detect query intent for better pattern matching
            intent = self._detect_query_intent(question)
            
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            generation_strategy = self._determine_generation_strategy(question, similar_sqls)
            
            # Filter examples by relevance to detected intent
            if similar_sqls and intent:
                similar_sqls = self._filter_by_intent(similar_sqls, intent)
            
            sql_examples_context = self._build_examples_context(similar_sqls, generation_strategy)
            
            # Clean previous_results to remove non-serializable objects like DataFrames
            cleaned_previous_results = self._clean_previous_results(previous_results) if previous_results else None
            
            previous_results_context = ""
            if cleaned_previous_results:
                previous_results_context = f"\n\nPREVIOUS STEP RESULTS:\n{json.dumps(cleaned_previous_results, indent=2)}\nUse this data to construct WHERE clauses or filters if the question references specific entities."
            
            # Build entity context if available
            entity_context = ""
            if entity_info and entity_info.get("entity_mapping"):
                entity_context = "\n\n" + "="*80 + "\n"
                entity_context += "🏷️ ENTITY VERIFICATION RESULTS - THESE ARE MANDATORY CONSTRAINTS:\n"
                entity_context += "="*80 + "\n"
                
                for entity_name, table_type in entity_info.get("entity_mapping", {}).items():
                    entity_context += f"✓ Entity '{entity_name}' was verified in table '{table_type}'\n"
                
                # Determine which main table to use based on entities
                if "ViewCustomer" in entity_info.get("entity_types", []):
                    entity_context += "\n🚨 CRITICAL REQUIREMENT - CUSTOMER QUERY:\n"
                    entity_context += "   • You MUST use ViewCustomer table for filtering the customer\n"
                    entity_context += "   • You MUST use CustomerInvoice for sales data\n"
                    entity_context += "   • Use EXACT entity name as verified (case-sensitive match)\n"
                    entity_context += "   • DO NOT change the case of entity names - use exactly as provided below\n"
                    entity_context += "   • DO NOT use CustomerInvoice.externalCode for filtering customers\n"
                    entity_context += "   • DO NOT use DistributorSales table - this is a CUSTOMER query!\n"
                    entity_context += "   • CROSS JOIN ViewCustomer with CustomerInvoice\n"
                    
                    # Add specific entity names for WHERE clause (using exact verified names)
                    entity_names = entity_info.get("entities", [])
                    if entity_names:
                        if len(entity_names) == 1:
                            # Single entity - use exact match with verified name
                            entity_exact = entity_names[0]  # Use exact name from verification
                            entity_context += f"\n   • WHERE clause MUST be: WHERE ViewCustomer.name = '{entity_exact}'\n"
                            entity_context += f"   • IMPORTANT: Use EXACT value '{entity_exact}' (from entity verification)\n"
                            entity_context += f"   • DO NOT capitalize or change case - database stores it as '{entity_exact}'\n"
                        else:
                            # Multiple entities - use IN with exact names
                            entity_list = ', '.join([f"'{e}'" for e in entity_names])
                            entity_context += f"\n   • WHERE clause MUST be: WHERE ViewCustomer.name IN ({entity_list})\n"
                            entity_context += f"   • Use EXACT values from entity verification (case-sensitive)\n"
                
                elif "ViewDistributor" in entity_info.get("entity_types", []):
                    entity_context += "\n🚨 CRITICAL REQUIREMENT - DISTRIBUTOR QUERY:\n"
                    entity_context += "   • You MUST use ViewDistributor table for filtering\n"
                    entity_context += "   • You MUST use DistributorSales for sales data\n"
                    entity_context += "   • Use EXACT entity name as verified (case-sensitive match)\n"
                    entity_context += "   • DO NOT change the case of entity names\n"
                    entity_context += "   • DO NOT use CustomerInvoice table - this is a DISTRIBUTOR query!\n"
                    
                    # Add specific entity names for WHERE clause (using exact verified names)
                    entity_names = entity_info.get("entities", [])
                    if entity_names:
                        if len(entity_names) == 1:
                            entity_exact = entity_names[0]  # Use exact name from verification
                            entity_context += f"\n   • WHERE clause: WHERE ViewDistributor.name = '{entity_exact}'\n"
                            entity_context += f"   • Use EXACT value '{entity_exact}' from verification\n"
                        else:
                            entity_list = ', '.join([f"'{e}'" for e in entity_names])
                            entity_context += f"\n   • WHERE clause: WHERE ViewDistributor.name IN ({entity_list})\n"
                    
                elif "Sku" in entity_info.get("entity_types", []):
                    entity_context += "\n🚨 CRITICAL REQUIREMENT - SKU QUERY:\n"
                    entity_context += "   • You MUST join with Sku table for filtering\n"
                    entity_context += "   • Use EXACT entity name as verified (case-sensitive match)\n"
                    entity_context += "   • DO NOT change the case of entity names\n"
                    
                    # Add specific entity names for WHERE clause (using exact verified names)
                    entity_names = entity_info.get("entities", [])
                    if entity_names:
                        if len(entity_names) == 1:
                            entity_exact = entity_names[0]  # Use exact name from verification
                            entity_context += f"\n   • WHERE clause: WHERE Sku.name = '{entity_exact}'\n"
                            entity_context += f"   • Use EXACT value '{entity_exact}' from verification\n"
                        else:
                            entity_list = ', '.join([f"'{e}'" for e in entity_names])
                            entity_context += f"\n   • WHERE clause: WHERE Sku.name IN ({entity_list})\n"
                
                entity_context += "\n" + "="*80 + "\n"
            
            system_prompt = self._build_system_prompt(
                current_date=current_date,
                schema_info=self.schema_content,
                sql_examples=sql_examples_context,
                previous_results=previous_results_context,
                entity_context=entity_context,
                strategy=generation_strategy
            )
            
            # Build user message with explicit entity names if available
            user_message = f"Generate SQL for: {question}"
            
            # Add EXPLICIT entity name reminder in user message for maximum visibility
            if entity_info and entity_info.get("entities"):
                entity_names = entity_info.get("entities", [])
                entity_types = entity_info.get("entity_types", [])
                
                user_message += "\n\n🚨 VERIFIED ENTITY NAMES (USE THESE EXACT VALUES):\n"
                for entity_name, entity_type in zip(entity_names, entity_types):
                    user_message += f"   • {entity_type}: '{entity_name}' (EXACT - do not change case or spelling)\n"
                
                user_message += "\n⚠️ CRITICAL: Use the EXACT entity names shown above in your WHERE clause."
                user_message += "\n   DO NOT capitalize, lowercase, or modify these names in any way!"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # For exact matches (similarity > 0.95), provide the example and instruct to use it
            if generation_strategy == "exact_match" and similar_sqls:
                exact_match_sql = similar_sqls[0]['sql']
                
                # Add instruction to use exact match
                messages.append({
                    "role": "system",
                    "content": """🎯 EXACT MATCH FOUND (similarity > 0.95)!

INSTRUCTION: A nearly identical query exists in the database. Use it DIRECTLY with minimal modifications.

ONLY modify if absolutely necessary:
- Update date values (e.g., change specific month to current month)
- Change LIMIT values if user specifies different number
- Keep all other patterns EXACTLY the same

DO NOT:
- Add unnecessary complexity (WITH clauses, CASE statements, etc.)
- Change the aggregation pattern
- Modify the table joins
- Alter the WHERE clause logic (except date values)

If the example already matches the user's request perfectly, return it AS IS."""
                })
                
                # Provide the exact match as one-shot example
                messages.append({
                    "role": "user", 
                    "content": similar_sqls[0]['question']
                })
                messages.append({
                    "role": "assistant", 
                    "content": json.dumps({"sql": exact_match_sql})
                })
            
            adaptive_llm = self._get_adaptive_llm(similar_sqls, generation_strategy)
            
            response = adaptive_llm.invoke(messages)
            content = response.content.strip()
            
            track_llm_call(
                input_prompt=messages,
                output=content,
                agent_type="sql_generator",
                operation="generate_sql",
                model_name="gpt-4o"
            )
            
            parsed_result = self._parse_and_validate_sql(content, question)
            
            if parsed_result["success"]:
                validated_sql = self._validate_cube_js_compliance(parsed_result["sql"])
                parsed_result["sql"] = validated_sql
                parsed_result["strategy_used"] = generation_strategy
            
            return parsed_result
                
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return {
                "success": False,
                "error": f"SQL generation error: {str(e)}",
                "type": "generation_error"
            }
    
    def _filter_by_intent(self, similar_sqls: List[Dict], intent: Dict[str, Any]) -> List[Dict]:
        """
        Filter examples to only include those relevant to the detected intent.
        Integrated from improved_sql_generator.py for better example selection.
        """
        if not similar_sqls:
            return []
        
        relevant = []
        for sql_info in similar_sqls[:5]:  # Only top 5
            sql_lower = sql_info.get('sql', '').lower()
            
            # Check if example matches intent
            matches_table = intent['table'].lower() in sql_lower
            matches_aggregation = intent['aggregation'].lower() in sql_lower if intent['aggregation'] else True
            
            if matches_table and sql_info.get('similarity', 0) > 0.7:
                relevant.append(sql_info)
        
        return relevant[:3]  # Max 3 relevant examples
    
    def _build_system_prompt(self, current_date: str, schema_info: str, 
                            sql_examples: str, previous_results: str, entity_context: str, strategy: str) -> str:
        """Build comprehensive system prompt for SQL generation."""
        
        # Use enhanced schema with FK relationships and metadata
        schema_context = ""
        if self.structured_schema and 'tables' in self.structured_schema:
            # Build focused schema for most commonly used tables with relationships
            important_tables = ['CustomerInvoice', 'CustomerInvoiceDetail', 'DistributorSales', 
                               'DistributorSalesDetail', 'Sku', 'Order', 'OrderDetail', 'OrderReturn', 'OrderReturnDetail']
            
            schema_parts = []
            for table_name in important_tables:
                if table_name in self.structured_schema['tables']:
                    table_data = self.structured_schema['tables'][table_name]
                    
                    # Build table section
                    table_part = f"\nTABLE: {table_name}"
                    
                    # Add fields (limit to first 15 most important fields)
                    if 'fields' in table_data:
                        fields_list = []
                        for field_name, field_type in list(table_data['fields'].items())[:15]:
                            if not field_name.startswith('__'):
                                fields_list.append(f"  - {field_name}: {field_type}")
                        if fields_list:
                            table_part += "\nFields:\n" + "\n".join(fields_list)
                    
                    # Add metadata (aggregatable fields, date fields)
                    if 'metadata' in table_data:
                        metadata = table_data['metadata']
                        
                        if metadata.get('is_detail_table'):
                            table_part += f"\n  [Detail table, parent: {metadata.get('parent_table')}]"
                        
                        if metadata.get('date_fields'):
                            table_part += f"\n  Date fields: {', '.join(metadata['date_fields'][:3])}"
                        
                        if metadata.get('aggregatable_fields'):
                            table_part += f"\n  Aggregatable (use with MEASURE): {', '.join(metadata['aggregatable_fields'][:5])}"
                    
                    # Add relationships
                    if 'relationships' in table_data and table_data['relationships']:
                        table_part += "\n  Relationships:"
                        for related_table, rel_info in list(table_data['relationships'].items())[:3]:
                            table_part += f"\n    → {related_table} (use {rel_info.get('cube_js_join', 'CROSS JOIN')})"
                    
                    schema_parts.append(table_part)
            
            schema_context = "\n\nDATABASE SCHEMA (with relationships and field metadata):" + "".join(schema_parts)
        else:
            # Fallback to text schema if enhanced JSON not available
            schema_context = f"\n\nDATABASE SCHEMA:\n{schema_info[:3000]}"
        
        # Enhanced previous results context
        previous_results_formatted = previous_results
        if previous_results and "PREVIOUS STEP RESULTS:" in previous_results:
            previous_results_formatted = previous_results + """

⚠️ CRITICAL FOR MULTI-STEP QUERIES:
- If question mentions "identified in step 1" or "from step X", you MUST filter using those specific results
- Example: If step 1 found SKUs ['A', 'B', 'C'], use WHERE skuName IN ('A', 'B', 'C')
- DO NOT ignore previous step results - they are the input for this query"""
        
        return f"""You are an expert SQL generator for Cube.js API. Generate reliable, production-ready SQL queries.

TODAY'S DATE: {current_date}
{schema_context}
{entity_context}

CUBE.JS CRITICAL RULES - STRICTLY FOLLOW THESE:

1. AGGREGATION PATTERN - WITH CLAUSE + MEASURE():
   For aggregation queries (top N, sum, count, avg), you MUST use WITH clause with MEASURE() inside.
   
   ✅ CORRECT PATTERN for Top N Aggregation:
   ```sql
   WITH aggregated AS (
       SELECT 
           CustomerInvoiceDetail.skuName,
           MEASURE(CustomerInvoiceDetail.skuAmount) AS TotalAmount
       FROM CustomerInvoice 
       CROSS JOIN CustomerInvoiceDetail
       WHERE DATE_TRUNC('month', CustomerInvoice.dispatchedDate) = '2024-09-01'
       GROUP BY CustomerInvoiceDetail.skuName
   )
   SELECT skuName, TotalAmount 
   FROM aggregated 
   ORDER BY TotalAmount DESC 
   LIMIT 3
   ```
   
   ⚠️ CRITICAL RULES:
   - MEASURE() MUST be inside WITH clause (NEVER outside)
   - GROUP BY is ALLOWED inside WITH clause (REQUIRED for aggregation)
   - Assign aliases to all columns in WITH clause
   - Use those aliases in outer SELECT
   - DO NOT use SUM/AVG/COUNT outside WITH clause

2. JOIN RULES - CROSS JOIN ONLY, NO CONDITIONS:
   - ALWAYS use CROSS JOIN (NEVER INNER/LEFT/RIGHT/FULL JOIN)
   - NEVER use ON clause
   - STRICTLY NO JOIN CONDITIONS in WHERE clause
   - ❌ WRONG: WHERE CustomerInvoice.id = CustomerInvoiceDetail.invoice_id
   - ✅ CORRECT: Just CROSS JOIN tables, Cube.js handles relationships internally
   
   Example:
   ```sql
   -- WRONG (has WHERE join condition):
   FROM CustomerInvoice CROSS JOIN CustomerInvoiceDetail 
   WHERE CustomerInvoice.id = CustomerInvoiceDetail.invoice_id
   
   -- CORRECT (no join conditions):
   FROM CustomerInvoice CROSS JOIN CustomerInvoiceDetail
   WHERE DATE_TRUNC('month', CustomerInvoice.dispatchedDate) = '2024-09-01'
   ```

3. WITH CLAUSE RULES:
   - Use WITH clause for any aggregation or subquery
   - MEASURE() MUST be inside WITH clause
   - GROUP BY is ALLOWED and REQUIRED inside WITH clause
   - DO NOT create two tables with WITH and JOIN them (use UNION ALL instead)
   - Always assign aliases to columns in WITH clause
4. DATE HANDLING - LEARN YEAR FROM EXAMPLES:
   - 🔍 Check the EXAMPLES below to see which YEAR is used (2024 or 2025)
   - 📅 ALWAYS use the SAME YEAR as shown in examples
   - Specific months: DATE_TRUNC('month', date_column) = 'YYYY-MM-01'
   - For September: If examples show '2024-09-01', use 2024 (not current year)
   - Use dispatchedDate for CustomerInvoice, datetime for Order
   - For Order table: Order.datetime
   - For CustomerInvoice: CustomerInvoice.dispatchedDate
   - For "last N months": Use INTERVAL arithmetic from CURRENT_DATE

5. FORBIDDEN PATTERNS:
   - NO TO_CHAR(), EXTRACT(), COALESCE(), GET_DATE() (not supported in Cube.js)
   - NO INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN (use CROSS JOIN only)
   - NO ON clause with joins
   - NO JOIN CONDITIONS in WHERE clause (Cube.js handles relationships)
   - NO SUM/AVG/COUNT outside WITH clause (use MEASURE inside WITH)
   - NO SELECT * (always specify columns)
   - NO subqueries in WHERE clause
   - NO alias usage in WHERE clause

6. ⚠️ CRITICAL: TABLE NAMES - STRICTLY FOLLOW EXAMPLES:
   - 🚫 NEVER INVENT OR GUESS TABLE NAMES (e.g., "CustomerVisit", "SalesData", "CallHistory")
   - ✅ ONLY use table names that appear in the PROVIDED SQL EXAMPLES below
   - 📋 Extract ALL table names from examples and use ONLY those tables
   - If examples show tables for visits/calls, USE THOSE EXACT TABLE NAMES
   - If you don't see a table in examples, CHECK THE SCHEMA before inventing
   - Common mistakes to AVOID:
     * ❌ Using "CustomerVisit" when examples show different table
     * ❌ Using "CallHistory" when examples use different table
     * ❌ Using "SalesTransaction" when examples use CustomerInvoice
   
7. 🎯 STRICT EXAMPLE-FOLLOWING RULES:
   - The 20 SQL EXAMPLES below are HIGHLY RELEVANT to your current question
   - These examples have 70%+ similarity - they are VERY CLOSE to what you need
   - ⚠️ PRIORITY ORDER when generating SQL:
     1. FIRST: Check if examples use WITH clause → Copy that pattern
     2. SECOND: Use EXACT table names from examples (don't invent new ones)
     3. THIRD: Copy field naming patterns (skuName vs sku_name, dispatchedDate vs dispatch_date)
     4. FOURTH: Copy date filtering patterns (DATE_TRUNC vs BETWEEN)
     5. FIFTH: Copy aggregation patterns (MEASURE location, GROUP BY placement)
   - 📝 IF EXAMPLES DON'T USE WITH CLAUSE: Don't add it unless absolutely needed for aggregation
   - 📝 IF EXAMPLES USE SIMPLE SELECT: Follow that simpler pattern
   - Most examples are PROVEN WORKING queries - don't overcomplicate!

8. FIELD USAGE NOTES - LEARN FROM EXAMPLES:
   - 🔍 Check the EXAMPLES below to see correct field naming patterns
   - For Customer name: CROSS JOIN ViewCustomer, use ViewCustomer.name
   - For User name: CROSS JOIN ViewUser, use ViewUser.name  
   - For Distributor name: CROSS JOIN ViewDistributor, use ViewDistributor.name
   - For SKU name: Follow the pattern from examples (CustomerInvoiceDetail.skuName vs Sku.name)
   - For Category name: CROSS JOIN Category with Sku, use Category.name
   - For Brand name: CROSS JOIN Brand with Sku, use Brand.name
   - For counts: Use MEASURE(table.count), NOT COUNT(DISTINCT table.id)

7. COMMON QUERY PATTERNS:

   Pattern A - Top N with Aggregation (CORRECT for all cubes):
   ```sql
   WITH aggregated AS (
       SELECT 
           CustomerInvoiceDetail.skuName,
           MEASURE(CustomerInvoiceDetail.skuAmount) AS TotalAmount
       FROM CustomerInvoice 
       CROSS JOIN CustomerInvoiceDetail
       WHERE DATE_TRUNC('month', CustomerInvoice.dispatchedDate) = '2024-09-01'
       GROUP BY CustomerInvoiceDetail.skuName
   )
   SELECT skuName, TotalAmount 
   FROM aggregated 
   ORDER BY TotalAmount DESC 
   LIMIT 3
   ```
   
   Pattern B - Top N with SKU Names (join Sku table):
   ```sql
   WITH aggregated AS (
       SELECT 
           Sku.name AS SkuName,
           MEASURE(DistributorSalesDetail.sku_value) AS TotalValue
       FROM DistributorSales 
       CROSS JOIN DistributorSalesDetail 
       CROSS JOIN Sku
       WHERE DATE_TRUNC('month', DistributorSales.erp_document_date) = '2024-09-01'
       GROUP BY Sku.name
   )
   SELECT SkuName, TotalValue 
   FROM aggregated 
   ORDER BY TotalValue DESC 
   LIMIT 10
   ```
   
   Pattern C - Time-based Aggregation:
   ```sql
   WITH monthly AS (
       SELECT 
           DATE_TRUNC('month', CustomerInvoice.dispatchedDate) AS Month,
           MEASURE(CustomerInvoice.dispatchedvalue) AS TotalSales
       FROM CustomerInvoice
       WHERE CustomerInvoice.dispatchedDate >= '2024-01-01'
       GROUP BY DATE_TRUNC('month', CustomerInvoice.dispatchedDate)
   )
   SELECT Month, TotalSales 
   FROM monthly 
   ORDER BY Month
   ```
   
   Pattern D - Simple Filter (no aggregation needed):
   ```sql
   SELECT 
       ViewCustomer.name AS CustomerName,
       CustomerInvoice.dispatchedvalue AS InvoiceValue
   FROM CustomerInvoice 
   CROSS JOIN ViewCustomer
   WHERE DATE_TRUNC('month', CustomerInvoice.dispatchedDate) = '2024-09-01'
   ORDER BY InvoiceValue DESC
   LIMIT 10
   ```

{sql_examples}

{previous_results}

RESPONSE FORMAT:
Return ONLY valid JSON with properly escaped SQL (use \\n for newlines): {{"sql": "YOUR_SQL_QUERY_HERE"}}
IMPORTANT: Escape all newlines, tabs, and special characters in the SQL string to create valid JSON.

CRITICAL CHECKLIST FOR ALL QUERIES:

✅ **Aggregation**: Use WITH clause + MEASURE() + GROUP BY inside WITH
✅ **Joins**: CROSS JOIN only (no INNER/LEFT/RIGHT/FULL)
✅ **No ON clause**: Never use ON with joins
✅ **No WHERE join conditions**: No table1.id = table2.fk in WHERE
✅ **MEASURE location**: Inside WITH clause only (not outside)
✅ **GROUP BY location**: Inside WITH clause only (not outside)
✅ **Aliases**: Assign to all WITH clause columns, reference in outer SELECT
✅ **No forbidden functions**: TO_CHAR, EXTRACT, COALESCE, GET_DATE
✅ **No SELECT ***: Always specify columns explicitly
✅ **Date fields**: dispatchedDate for CustomerInvoice, datetime for Order

Example for "top 3 SKUs for September" (CORRECT WITH AGGREGATION):
```sql
WITH aggregated AS (
    SELECT 
        CustomerInvoiceDetail.skuName,
        MEASURE(CustomerInvoiceDetail.skuAmount) AS TotalAmount
    FROM CustomerInvoice 
    CROSS JOIN CustomerInvoiceDetail
    WHERE DATE_TRUNC('month', CustomerInvoice.dispatchedDate) = '2024-09-01'
    GROUP BY CustomerInvoiceDetail.skuName
)
SELECT skuName, TotalAmount 
FROM aggregated 
ORDER BY TotalAmount DESC 
LIMIT 3
```"""
    
    def _build_examples_context(self, similar_sqls: Optional[List[Dict]], strategy: str) -> str:
        """Build SQL examples context based on strategy."""
        if not similar_sqls or strategy == "schema_based":
            return "No similar examples available. Generate based on schema and Cube.js best practices."
        
        examples_text = f"\nSTRATEGY: {strategy.upper()}\n\nSIMILAR SQL EXAMPLES:\n"
        
        max_examples = 3 if strategy == "exact_match" else 2 if strategy == "high_similarity" else 1
        
        for i, sql_data in enumerate(similar_sqls[:max_examples], 1):
            examples_text += f"\nExample {i} (Similarity: {sql_data.get('similarity', 0):.3f}):\n"
            examples_text += f"Question: {sql_data.get('question', 'N/A')}\n"
            examples_text += f"SQL: {sql_data.get('sql', 'N/A')}\n"
        
        return examples_text
    
    def _parse_and_validate_sql(self, content: str, question: str) -> Dict[str, Any]:
        """Parse LLM response and extract SQL with validation."""
        try:
            content = content.strip()
            
            json_match = re.search(r'\{[^{}]*"sql"[^{}]*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Try parsing as-is first
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # If parsing fails, try extracting SQL directly with regex
                    # This handles cases where SQL contains unescaped newlines
                    logger.warning(f"JSON parsing failed: {e}")
                    logger.info("Attempting direct SQL extraction from malformed JSON...")
                    
                    # More robust extraction: find "sql": " and get everything until closing " followed by } or ,
                    # This handles multi-line SQL with proper quote escaping
                    sql_pattern = r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"\s*[,}]'
                    sql_match = re.search(sql_pattern, json_str, re.DOTALL)
                    if sql_match:
                        sql = sql_match.group(1)
                        # Unescape any escaped characters (like \n, \t)
                        sql = sql.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                        sql = sql.strip()
                        logger.info(f"✅ Extracted SQL directly ({len(sql)} chars)")
                        
                        if sql.upper().startswith(('SELECT', 'WITH')):
                            return {
                                "success": True,
                                "sql": sql,
                                "query_type": "SELECT",
                                "explanation": "SQL extracted directly (malformed JSON)",
                                "format": "json_extracted"
                            }
                    
                    # If extraction failed, set data to None to try other methods
                    logger.error("❌ Failed to extract SQL from malformed JSON")
                    data = None
                
                if data and "sql" in data:
                    sql = data["sql"].strip()
                    
                    if sql.upper().startswith(('SELECT', 'WITH')):
                        return {
                            "success": True,
                            "sql": sql,
                            "query_type": "SELECT",
                            "explanation": "SQL generated successfully",
                            "format": "json"
                        }
                
                if data and "error" in data:
                    return {
                        "success": False,
                        "error": data["error"],
                        "type": "context_insufficient"
                    }
            
            sql_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
            if sql_block_match:
                sql = sql_block_match.group(1).strip()
                return {
                    "success": True,
                    "sql": sql,
                    "query_type": "SELECT",
                    "explanation": "SQL extracted from markdown",
                    "format": "markdown"
                }
            
            if content.upper().strip().startswith(('SELECT', 'WITH')):
                return {
                    "success": True,
                    "sql": content,
                    "query_type": "SELECT",
                    "explanation": "SQL extracted as raw text",
                    "format": "raw"
                }
            
            return {
                "success": False,
                "error": "Could not parse SQL from response",
                "type": "parsing_error",
                "raw_response": content[:200]
            }
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return {
                "success": False,
                "error": f"Parsing error: {str(e)}",
                "type": "parsing_error"
            }
    
    def _validate_cube_js_compliance(self, sql: str) -> str:
        """Validate and fix Cube.js compliance issues based on official guidelines."""
        try:
            sql_upper = sql.upper()
            
            # Fix 1: Replace INNER/LEFT/RIGHT/FULL JOIN with CROSS JOIN
            join_keywords = ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'LEFT OUTER JOIN', 'RIGHT OUTER JOIN']
            for keyword in join_keywords:
                if keyword in sql_upper:
                    logger.warning(f"🔧 FIX: Replaced {keyword} with CROSS JOIN")
                    sql = re.sub(
                        rf'{keyword}',
                        'CROSS JOIN',
                        sql,
                        flags=re.IGNORECASE
                    )
            
            # Fix 2: Remove JOIN...ON clauses (convert to CROSS JOIN without condition)
            if re.search(r'JOIN\s+\w+\s+ON\s+', sql, re.IGNORECASE):
                logger.warning("🔧 FIX: Removing ON clause from JOIN (Cube.js doesn't use ON)")
                sql = re.sub(
                    r'(CROSS\s+)?JOIN\s+(\w+)\s+ON\s+[^\n]+',
                    r'CROSS JOIN \2',
                    sql,
                    flags=re.IGNORECASE
                )
            
            # Fix 3: Remove WHERE join conditions (e.g., WHERE table1.id = table2.fk)
            # This is tricky - we need to identify join conditions vs filter conditions
            # Join conditions typically involve .id = .something_id pattern
            join_condition_pattern = r'AND\s+\w+\.\w*id\s*=\s*\w+\.\w+|WHERE\s+\w+\.\w*id\s*=\s*\w+\.\w+\s+AND'
            if re.search(join_condition_pattern, sql, re.IGNORECASE):
                logger.warning("🔧 FIX: Removing join conditions from WHERE clause (Cube.js handles relationships)")
                # Remove patterns like: WHERE table1.id = table2.fk AND
                sql = re.sub(
                    r'WHERE\s+\w+\.\w*id\s*=\s*\w+\.\w+\s+AND\s+',
                    'WHERE ',
                    sql,
                    flags=re.IGNORECASE
                )
                # Remove patterns like: AND table1.id = table2.fk
                sql = re.sub(
                    r'\s+AND\s+\w+\.\w*id\s*=\s*\w+\.\w+',
                    '',
                    sql,
                    flags=re.IGNORECASE
                )
            
            # Fix 4: Check if MEASURE() is outside WITH clause (should be inside)
            if 'MEASURE(' in sql_upper:
                # Check if MEASURE is in outer SELECT (not in WITH clause)
                if not re.search(r'WITH\s+\w+\s+AS\s*\([^)]*MEASURE\(', sql, re.IGNORECASE | re.DOTALL):
                    logger.warning("⚠️ WARNING: MEASURE() found outside WITH clause - should be inside WITH")
            
            # Fix 5: Check if GROUP BY is outside WITH clause
            if 'GROUP BY' in sql_upper and 'WITH' in sql_upper:
                # GROUP BY should be inside WITH clause
                # This is hard to fix automatically, so just warn
                if not re.search(r'WITH\s+\w+\s+AS\s*\([^)]*GROUP\s+BY', sql, re.IGNORECASE | re.DOTALL):
                    logger.warning("⚠️ WARNING: GROUP BY found outside WITH clause - should be inside WITH")
            
            # Fix 6: Replace SUM/AVG/COUNT outside WITH with MEASURE inside WITH
            # Only log warning - complex to fix automatically
            if re.search(r'SELECT.*?(SUM|AVG|COUNT)\s*\(', sql, re.IGNORECASE) and 'WITH' not in sql_upper:
                logger.warning("⚠️ WARNING: Using SUM/AVG/COUNT without WITH clause - should use WITH + MEASURE()")
            
            # Fix 7: Double SELECT statement
            if 'SELECTSELECT' in sql_upper:
                logger.warning("🔧 FIX: Removed double SELECT statement")
                sql = re.sub(r'SELECT\s*SELECT', 'SELECT', sql, flags=re.IGNORECASE)
            
            return sql
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return sql
    
    def _determine_generation_strategy(self, question: str, similar_sqls: Optional[List[Dict]]) -> str:
        """Determine SQL generation strategy based on similarity scores."""
        if not similar_sqls or len(similar_sqls) == 0:
            return "schema_based"
        
        best_similarity = similar_sqls[0].get('similarity', 0.0)
        
        if best_similarity > 0.95:
            return "exact_match"
        elif best_similarity > 0.80:
            return "high_similarity"
        elif best_similarity > 0.65:
            return "hybrid"
        else:
            return "schema_based"
    
    def _get_adaptive_llm(self, similar_sqls: Optional[List[Dict]], strategy: str):
        """
        Create LLM with adaptive temperature based on strategy and relevance.
        Enhanced with 5-level temperature strategy considering both similarity and relevance.
        Integrated from improved_sql_generator.py for more nuanced temperature control.
        """
        try:
            from langchain_openai import ChatOpenAI
            import os
            
            # Calculate both similarity and relevance ratio
            similarity = 0.0
            relevance_ratio = 0.0
            
            if similar_sqls and len(similar_sqls) > 0:
                similarity = similar_sqls[0].get('similarity', 0.0)
                
                # Calculate relevance ratio (relevant examples / total examples)
                relevant_count = sum(1 for sql in similar_sqls[:5] if sql.get('similarity', 0) > 0.7)
                total_count = min(len(similar_sqls), 5)
                relevance_ratio = relevant_count / total_count if total_count > 0 else 0
            
            # 5-level temperature strategy with relevance consideration
            if similarity > 0.95:
                # Exact match: use very low temperature for direct copying
                temp = 0.05
                reason = "exact match (>0.95) - use example directly"
            elif similarity > 0.85 and relevance_ratio > 0.7:
                # Very high similarity + high relevance: conservative but not too rigid
                temp = 0.1
                reason = "very high similarity + high relevance"
            elif similarity > 0.75 and relevance_ratio > 0.5:
                # Good similarity + decent relevance: moderate-conservative
                temp = 0.15
                reason = "good similarity + decent relevance"
            elif similarity > 0.65 or relevance_ratio > 0.3:
                # Medium similarity OR some relevance: moderate
                temp = 0.2
                reason = "medium similarity or some relevance"
            elif similarity > 0.5:
                # Low similarity: allow more creativity
                temp = 0.3
                reason = "low similarity"
            else:
                # Very low similarity: maximum creativity
                temp = 0.4
                reason = "very low similarity"
            
            logger.info(f"Adaptive temp: {temp} ({reason}, sim: {similarity:.3f}, rel_ratio: {relevance_ratio:.2f})")
            
            return ChatOpenAI(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temp,
                max_tokens=2000
            )
            
        except Exception as e:
            logger.warning(f"Failed to create adaptive LLM: {e}, using original")
            return self.llm
    
    def process(self, state: BaseAgentState) -> DBAgentState:
        """Process method required by BaseAgent interface."""
        db_state = DBAgentState(**state)
        
        try:
            similar_sqls = state.get("retrieved_sql_context", [])
            # Check for intermediate_results from multi-step workflow
            previous_results = state.get("previous_step_results", None) or state.get("intermediate_results", None)
            original_query = state.get("original_query", state["query"])
            entity_info = state.get("entity_info", None)
            
            # Log if we have previous results
            if previous_results:
                logger.info(f"🔗 SQL Generator received {len(previous_results)} previous step(s)")
            
            # Log if we have entity information
            if entity_info:
                logger.info(f"🏷️ SQL Generator received entity info: {entity_info}")
            
            result = self.generate_sql(
                question=state["query"],
                similar_sqls=similar_sqls,
                previous_results=previous_results,
                original_query=original_query,
                entity_info=entity_info
            )
            
            if result["success"]:
                db_state["query_type"] = result["query_type"]
                db_state["sql_query"] = result["sql"]
                db_state["status"] = "completed"
                db_state["success_message"] = "SQL generated successfully"
                db_state["result"] = result
                
                try:
                    from clean_logging import SQLLogger
                    SQLLogger.generation_complete(result['sql'], result['query_type'])
                except ImportError:
                    logger.info(f"SQL Generated ({result['query_type']}): {len(result['sql'])} chars)")
            else:
                db_state["error_message"] = result["error"]
                db_state["status"] = "failed"
                db_state["result"] = result
                
                logger.error(f"SQL generation failed:")
                logger.error(f"Question: {state['query']}")
                logger.error(f"Error: {result['error']}")
                
        except Exception as e:
            db_state["error_message"] = f"SQL generator error: {str(e)}"
            db_state["status"] = "failed"
            logger.error(f"SQLGeneratorAgent process failed: {e}")
        
        return db_state
