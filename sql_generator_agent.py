import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from base_agent import BaseAgent, BaseAgentState, DBAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

class SQLGeneratorAgent(BaseAgent):
    """
    Dedicated agent for generating individual SQL queries.
    This agent focuses on creating single, well-formed SQL queries based on schema and context.
    """
    
    def __init__(self, llm, schema_file_path: str = "schema"):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.schema_content = self._load_schema_file()
        
        self.query_templates = {
            "insert": "INSERT INTO {table} ({columns}) VALUES ({values})",
            "select": "SELECT {columns} FROM {table} WHERE {condition}",
            "update": "UPDATE {table} SET {updates} WHERE {condition}",
            "delete": "DELETE FROM {table} WHERE {condition}"
        }
    
    def get_agent_type(self) -> str:
        return "sql_generator"
    
    def _create_system_message(self, content: str) -> dict:
        """Create a system message for chat completions"""
        return {"role": "system", "content": content}
    
    def _create_user_message(self, content: str) -> dict:
        """Create a user message for chat completions"""
        return {"role": "user", "content": content}
    
    def _create_assistant_message(self, content: str) -> dict:
        """Create an assistant message for chat completions"""
        return {"role": "assistant", "content": content}
    
    def _load_schema_file(self) -> str:
        """Load the database schema from the schema file."""
        try:
            with open(self.schema_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                content = content.replace('\\n', '\n')
                
                logger.info(f"Successfully loaded schema file: {self.schema_file_path}")
                logger.info(f"Schema content length: {len(content)} characters")
                
                return content
        except FileNotFoundError:
            error_msg = f"Schema file not found: {self.schema_file_path}"
            logger.error(error_msg)
            return "Schema file not found. Unable to load database schema."
        except Exception as e:
            error_msg = f"Error loading schema file: {e}"
            logger.error(error_msg)
            return f"Error loading schema: {str(e)}"
    
    def get_schema_info(self) -> str:
        """Return the loaded schema information."""
        return self.schema_content
    
    def generate_sql(self, question: str, similar_sqls: List[str] = None, 
                     previous_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a single SQL query for the given question.
        
        Args:
            question: The specific question to generate SQL for
            similar_sqls: List of similar SQL queries for context (with similarity scores)
            previous_results: Results from previous steps (for multi-step queries)
            
        Returns:
            Dict containing the generated SQL and metadata
        """
        try:
            schema_info = self.get_schema_info()
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Analyze retrieved examples and determine generation strategy
            generation_strategy = self._determine_generation_strategy(question, similar_sqls)
            
            # Build context based on strategy
            sql_examples_text, strategy_instructions = self._build_strategy_context(
                question, similar_sqls, generation_strategy
            )
            
            context_from_previous = ""
            if previous_results:
                context_from_previous = f"""
                
                === PREVIOUS STEP RESULTS ===
                You have access to results from previous SQL executions:
                {json.dumps(previous_results, indent=2)}
                
                Use this data to inform your current query. If the question refers to specific entities
                (like "top 3 SKUs", "these customers", etc.), use the data from previous results
                to construct appropriate WHERE clauses or filters.
                """
            
            sql_generation_prompt = f"""You are an AI SQL assistant specialized in Cube.js SQL API. 
                            Your task is to generate a SINGLE, SIMPLE, and FOCUSED SQL query to answer the specific question provided.
                            
                            {strategy_instructions}
                            
                            === QUERY SIMPLICITY PRINCIPLE ===
                            **ALWAYS prioritize the simplest possible SQL that answers the question:**
                            1. For basic aggregations (average, sum, count), use simple SELECT with aggregation functions
                            2. Only use WITH clauses when complex grouping or multiple aggregations are needed
                            3. Only join tables when the question explicitly requires data from multiple tables
                            4. Avoid unnecessary complexity and user-wise breakdowns unless specifically requested
                            
                            === INTENT DETECTION RULES ===
                            **"Average order value for [time period]" = Overall average across all orders, NOT user-wise**
                            **"Sales for [time period]" = Total sales, NOT broken down by customer unless requested**
                            **"Orders this month" = Count/sum of orders, NOT per-user analysis**
                            
                            IMPORTANT: Generate SQL for THIS SPECIFIC QUESTION ONLY. Do not add extra analysis.
                            
                            STRICTLY follow the rules below and generate SQL in the format:
                            {{ "sql": "SQL_QUERY_HERE" }}
                            
                            === CRITICAL DATE AND TIME HANDLING ===
                            Today's date is {current_date}
                            
                            **SPECIFIC MONTH REFERENCES:**
                            - "September" / "month of September" = WHERE DATE_TRUNC('month', date_column) = '2024-09-01'
                            - "October" / "month of October" = WHERE DATE_TRUNC('month', date_column) = '2024-10-01'  
                            - "August" / "month of August" = WHERE DATE_TRUNC('month', date_column) = '2024-08-01'
                            
                            **TIME RANGE EXPRESSIONS:**
                            - "this month" = WHERE DATE_TRUNC('month', date_column) = DATE_TRUNC('month', CURRENT_DATE)
                            - "last month" = WHERE DATE_TRUNC('month', date_column) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
                            - "this year" = WHERE DATE_TRUNC('year', date_column) = DATE_TRUNC('year', CURRENT_DATE)
                            - "last quarter" = WHERE date_column >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months') AND date_column < DATE_TRUNC('quarter', CURRENT_DATE)
                            
                            **NEVER use just date_column < CURRENT_DATE for time range queries!**
                            
                            === MANDATORY DATA SOURCE SELECTION ===
                            **FOR ORDER QUERIES:**
                            - Primary table: Order
                            - Date field: Order.datetime  
                            - Value field: Order.value
                            - Simple average: SELECT AVG(Order.value) FROM Order WHERE [time_filter]
                            
                            **FOR SALES QUERIES:**
                            - Primary table: CustomerInvoice
                            - Date field: CustomerInvoice.dispatchedDate
                            - Value field: CustomerInvoice.dispatchedvalue
                            
                            === SIMPLE QUERY EXAMPLES ===
                            
                            **Example 1 - Average order value for specific month:**
                            ```sql
                            SELECT AVG(Order.value) AS AverageOrderValue
                            FROM Order
                            WHERE DATE_TRUNC('month', Order.datetime) = '2024-09-01'
                            ```
                            
                            **Example 2 - Total orders this month:**
                            ```sql
                            SELECT COUNT(*) AS TotalOrders
                            FROM Order
                            WHERE DATE_TRUNC('month', Order.datetime) = DATE_TRUNC('month', CURRENT_DATE)
                            ```
                            
                            **Example 3 - Monthly sales (only when breakdown needed):**
                            ```sql
                            SELECT 
                                DATE_TRUNC('month', CustomerInvoice.dispatchedDate) AS Month,
                                SUM(CustomerInvoice.dispatchedvalue) AS TotalSales
                            FROM CustomerInvoice
                            WHERE CustomerInvoice.dispatchedDate >= DATE_TRUNC('year', CURRENT_DATE)
                            GROUP BY DATE_TRUNC('month', CustomerInvoice.dispatchedDate)
                            ORDER BY Month
                            ```
                            
                            === TREND QUERY PATTERNS ===
                            **For "sales trend over last quarter" or "quarterly trends":**
                            ```sql
                            SELECT 
                                DATE_TRUNC('month', CustomerInvoice.dispatchedDate) AS Month,
                                SUM(CustomerInvoice.dispatchedvalue) AS TotalSales
                            FROM CustomerInvoice
                            WHERE CustomerInvoice.dispatchedDate >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')
                                AND CustomerInvoice.dispatchedDate < DATE_TRUNC('quarter', CURRENT_DATE)
                            GROUP BY DATE_TRUNC('month', CustomerInvoice.dispatchedDate)
                            ORDER BY Month
                            ```
                            
                            **For "display sales trend" or "show trends":**
                            - ALWAYS group by time periods (month, quarter, week)
                            - NEVER return individual SKU breakdowns for trend queries
                            - Use ORDER BY time_period for proper chronological ordering
                            - Focus on time-based aggregations, not item-level details
                            
                            === MANDATORY DATA SOURCE SELECTION ===
                            **CRITICAL: For any query about "SALES", "REVENUE", "DISPATCH" or "MONTHLY SALES":**
                            - MUST use CustomerInvoice as primary table
                            - MUST use CustomerInvoice.dispatchedDate for date filtering
                            - MUST use CustomerInvoice.dispatchedvalue for sales values
                            - NEVER use DistributorSales for general sales queries
                            - CROSS JOIN with CustomerInvoiceDetail only if item-level details needed
                            
                            **FOR ORDER QUERIES (different from sales):**
                            - Primary table: Order
                            - Date field: Order.datetime  
                            - Value field: Order.value
                            
                            ===Response Guidelines 
                            1. If the provided context is sufficient, generate a valid SQL query without explanations. Generate STRICTLY in format {{\"sql\" : \"SQL_QUERY_HERE\" }}.
                            2. If the provided context is insufficient, explain why it can't be generated in format {{\"error\" : \"ERROR_EXPLANATION_HERE\" }}.
                            3. If you need clarification, ask a follow-up question in format {{\"follow_up\" : \"FOLLOW_UP_QUESTION_HERE\" }}.
                            4. Always assume data is up-to-date unless explicitly stated otherwise.
                            5. Use the most relevant table(s) based on the schema.
                            6. Keep queries simple and focused on the specific question asked.
                            7. **MANDATORY**: For any time-based queries, use the precise date filtering rules above.
                            
                            ===SQL Generation Guidelines (Cube.js API Compatible)===
                            
                            **CUBE.JS SPECIFIC FUNCTIONS:**
                            - Use MEASURE() inside WITH clauses for aggregations (SUM, COUNT, etc.)
                            - Use DATE_TRUNC() for date grouping and filtering
                            - INTERVAL syntax: CURRENT_DATE - INTERVAL '12 months'
                            - Supported functions: DATE_TRUNC, CURRENT_DATE, INTERVAL, MEASURE
                            
                            **JOIN RULES:**
                            - STRICTLY ONLY USE CROSS JOIN for joining two tables 
                            - STRICTLY DO NOT USE ANY JOIN CONDITION EITHER WITH ON CLAUSE OR IN WHERE CLAUSE
                            - ONLY Use **CROSS JOIN** , DO NOT USE INNER JOIN or LEFT JOIN or RIGHT JOIN when joining tables.
                            - DO NOT CREATE TWO TABLES USING WITH CLAUSE AND JOIN THEM. 
                            - IF THERE IS NO DIRECT CROSS JOIN IN THE EXAMPLE BETWEEN TWO TABLES USE UNION ALL INSTEAD WITH PROPER CAST OF TYPE. 
                            
                            **UNSUPPORTED FUNCTIONS:**
                            - TO_CHAR() IS NOT SUPPORTED
                            - GET_DATE() IS NOT SUPPORTED  
                            - EXTRACT() IS NOT SUPPORTED
                            - COALESCE() IS NOT SUPPORTED
                            
                            === SQL Query Construction Rules ===
                            - DO NOT USE ALIAS IN WHERE CLAUSE
                            - DO NOT USE SELECT *.
                            - Use WITH Clause for sub queries:
                                - DO NOT USE MEASURE outside WITH CLAUSE Use SUM instead.
                                - STRICTLY USE MEASURE inside the WITH CLAUSE.
                                - STRICTLY Always assign alias to select inside the WITH CLAUSE and refer them outside.
                                - Try to also include GROUP BY CLAUSE inside the WITH CLAUSE
                                - DO NOT CREATE TWO TABLES USING WITH CLAUSE AND JOIN THEM.
                            - CROSS JOIN Rules:
                                - CROSS JOIN is sufficient there is no need of ON clause.
                                - WRONG: SELECT ViewCustomer.name FROM CustomerInvoice JOIN CustomerInvoiceDetail ON CustomerInvoice.id = CustomerInvoiceDetail.invoice_id
                                - CORRECT: SELECT ViewCustomer.name FROM CustomerInvoice CROSS JOIN CustomerInvoiceDetail CROSS JOIN ViewCustomer WHERE conditions
                            - Relationships Between Tables: 
                                - The **Order** table can be **cross joined** with **ViewCustomer**, **ViewDistributor**, **ViewUser**, and **OrderDetail** table.
                                - The **OrderDetails** table can be **cross joined** with the **Sku** table.
                                - The **Sku** table can be **cross joined** with **Brand** and **Category** tables.
                            - For filtering of Sku,Brand,Category CROSS JOIN and Filter by name column is sufficient.
                            - STRICTLY There should be no Subquery in a WHERE CLAUSE
                            - Names related queries:
                                - Customer's name: CROSS JOIN 'ViewCustomer' table and get ViewCustomer.name.
                                - User's name: CROSS JOIN 'ViewUser' table and get ViewUser.name.
                                - Distributor's name: CROSS JOIN 'ViewDistributor' table and get ViewDistributor.name.
                                - SKU's name: CROSS JOIN 'Sku' table and get Sku.name.
                                - Category's name: CROSS JOIN 'Category' table with 'Sku' table and get Category.name.
                                - Brand's name: CROSS JOIN 'Brand' table with 'Sku' table and get Brand.name.
                            - Date field related:
                                - Order related questions: use Order.datetime from Order table.
                                - Sales/revenue/dispatch related questions: use CustomerInvoice.dispatchedDate from CustomerInvoice table.
                            
                            **MANDATORY TIME-BASED QUERY EXAMPLES:**
                            
                            EXAMPLE 1 - Monthly sales trend for past year:
                            ```sql
                            WITH sales_data AS (
                                SELECT 
                                    DATE_TRUNC('month', CustomerInvoice.dispatchedDate) AS MonthYear,
                                    MEASURE(CustomerInvoice.dispatchedvalue) AS TotalSales
                                FROM CustomerInvoice 
                                CROSS JOIN CustomerInvoiceDetail
                                WHERE CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '12 months')
                                  AND CustomerInvoice.dispatchedDate < DATE_TRUNC('month', CURRENT_DATE)
                                GROUP BY DATE_TRUNC('month', CustomerInvoice.dispatchedDate)
                            )
                            SELECT MonthYear, TotalSales FROM sales_data ORDER BY MonthYear
                            ```
                            
                            EXAMPLE 2 - Sales for last 3 months:
                            ```sql
                            SELECT 
                                DATE_TRUNC('month', CustomerInvoice.dispatchedDate) AS MonthYear,
                                MEASURE(CustomerInvoice.dispatchedvalue) AS TotalSales
                            FROM CustomerInvoice 
                            CROSS JOIN CustomerInvoiceDetail
                            WHERE CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months')
                              AND CustomerInvoice.dispatchedDate < DATE_TRUNC('month', CURRENT_DATE)
                            GROUP BY DATE_TRUNC('month', CustomerInvoice.dispatchedDate)
                            ORDER BY MonthYear
                            ```
                            - Fields NOT to use:
                                - DO NOT USE Order.skuName, CustomerInvoiceDetail.skuName for sku names
                                - USE Sku.name from Sku table instead with proper CROSS JOIN
                                - DO NOT use 'id' for count.
                                - Example: USE MEASURE(ViewCustomer.count) instead of COUNT(DISTINCT ViewCustomer.id)
                            
                            DATABASE SCHEMA:
                            {schema_info}
                            
                            RELEVANT SQL EXAMPLES (for reference and pattern matching):
                            {sql_examples_text}
                            
                            {context_from_previous}
                            
                            QUESTION TO ANSWER: {question}
                            
                            **CRITICAL INSTRUCTIONS FOR THIS SPECIFIC QUESTION:**
                            1. Focus ONLY on the question asked: "{question}"
                            2. Do NOT mix patterns from examples unless they directly match the question requirements
                            3. For "sales trend" or "monthly sales" queries, use CustomerInvoice table, NOT DistributorSales
                            4. For "past year" queries, use exactly 12 months: INTERVAL '12 months' 
                            5. Do NOT add filters (like SKU names) that are not mentioned in the question
                            6. Keep the query simple and directly answer what is asked
                            
                            Generate a single SQL query to answer this specific question only."""
            
            message_log = [self._create_system_message(sql_generation_prompt)]
            message_log.append(self._create_user_message(f"Generate SQL for: {question}"))
            
            # Build intelligent conversation context based on strategy
            message_log = self._build_conversation_context(message_log, question, similar_sqls, generation_strategy)
            
            # ADAPTIVE TEMPERATURE: Adjust creativity based on similarity score
            adaptive_llm = self._get_adaptive_temperature_llm(similar_sqls, generation_strategy)
            
            # Generate response with adaptive temperature
            response = adaptive_llm.invoke(message_log)
            content = response.content.strip()
            
            # Validate and potentially simplify the generated SQL
            content = self._validate_and_simplify_sql(content, question)
            
            # Track token usage
            track_llm_call(
                input_prompt=message_log,
                output=content,
                agent_type="sql_generator",
                operation="generate_sql",
                model_name="gpt-4o"
            )
            
            try:
                # First try JSON format
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
                    parsed_response = json.loads(json_content)
                    
                    if "sql" in parsed_response:
                        return {
                            "success": True,
                            "sql": parsed_response["sql"],
                            "query_type": "SELECT",  # Default for Cube.js
                            "explanation": "Generated SQL query successfully",
                            "format": "cube_js_api",
                            "used_examples": len(similar_sqls) > 0 if similar_sqls else False,
                            "used_previous_results": previous_results is not None
                        }
                    
                    elif "error" in parsed_response:
                        return {
                            "success": False,
                            "error": parsed_response["error"],
                            "type": "context_insufficient"
                        }
                    
                    elif "follow_up" in parsed_response:
                        return {
                            "success": False,
                            "error": f"Follow-up question needed: {parsed_response['follow_up']}",
                            "type": "follow_up_required"
                        }
                
                # If JSON parsing fails, try extracting SQL from markdown code blocks
                sql_block_match = re.search(r'```(?:sql)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
                if sql_block_match:
                    sql_content = sql_block_match.group(1).strip()
                    return {
                        "success": True,
                        "sql": sql_content,
                        "query_type": "SELECT",  # Default for Cube.js
                        "explanation": "Generated SQL query successfully (extracted from markdown)",
                        "format": "markdown_extracted",
                        "used_examples": len(similar_sqls) > 0 if similar_sqls else False,
                        "used_previous_results": previous_results is not None
                    }
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, attempting fallback parsing")
                
                # Fallback parsing for non-JSON format
                query_type_match = re.search(r'QUERY_TYPE:\s*(\w+)', content)
                sql_match = re.search(r'SQL:\s*(.+?)(?=\nEXPLANATION:|$)', content, re.DOTALL)
                
                if sql_match:
                    return {
                        "success": True,
                        "sql": sql_match.group(1).strip(),
                        "query_type": query_type_match.group(1).upper() if query_type_match else "SELECT",
                        "explanation": "Generated SQL query using fallback parsing",
                        "format": "fallback_format",
                        "used_examples": len(similar_sqls) > 0 if similar_sqls else False,
                        "used_previous_results": previous_results is not None
                    }
            
            return {
                "success": False,
                "error": "Could not parse SQL query from response",
                "type": "parsing_error",
                "raw_response": content
            }
                
        except Exception as e:
            logger.error(f"SQL generation error: {e}")
            return {
                "success": False,
                "error": f"SQL generation error: {str(e)}",
                "type": "generation_error"
            }
    
    def _validate_and_simplify_sql(self, content: str, question: str) -> str:
        """
        Validate generated SQL and suggest simplifications for better performance and clarity.
        """
        try:
            # Extract SQL from response
            if content.startswith('{'):
                import json
                data = json.loads(content)
                sql = data.get("sql", "")
            else:
                sql = content
            
            sql_lower = sql.lower()
            question_lower = question.lower()
            
            # Check for unnecessary complexity patterns
            complexity_issues = []
            
            # Check 1: Unnecessary user/customer breakdown for simple aggregations
            if ("average" in question_lower or "total" in question_lower) and "user" not in question_lower and "customer" not in question_lower:
                if ("viewuser" in sql_lower or "viewcustomer" in sql_lower) and "group by" in sql_lower:
                    complexity_issues.append("Unnecessary user/customer breakdown for simple aggregation")
            
            # Check 2: Complex WITH clauses for simple operations
            if "with " in sql_lower and ("avg(" in question_lower or "count(" in question_lower or "sum(" in question_lower):
                if sql_lower.count("select") > 2:  # WITH clause + main SELECT + potential subqueries
                    complexity_issues.append("Complex WITH clause for simple aggregation")
            
            # Check 3: CASE statements for simple time filtering
            if "case when" in sql_lower and ("month" in question_lower or "quarter" in question_lower):
                if question_lower.count("month") == 1 or question_lower.count("quarter") == 1:
                    complexity_issues.append("Unnecessary CASE statement for single time period")
            
            # Check 4: Multiple CROSS JOINs when not needed
            cross_join_count = sql_lower.count("cross join")
            if cross_join_count > 1 and not any(breakdown in question_lower for breakdown in ["by customer", "by user", "by sku", "by category"]):
                complexity_issues.append("Multiple unnecessary table joins")
            
            # If issues found, suggest simplification
            if complexity_issues:
                logger.warning(f"SQL complexity issues detected: {complexity_issues}")
                simplified_sql = self._generate_simplified_sql(question, sql)
                if simplified_sql:
                    return simplified_sql
            
            return content
            
        except Exception as e:
            logger.error(f"SQL validation error: {e}")
            return content
    
    def _determine_generation_strategy(self, question: str, similar_sqls: List[Dict]) -> str:
        """
        Determine the best strategy for SQL generation based on retrieved examples.
        
        Returns:
            str: Strategy type - 'exact_match', 'high_similarity', 'schema_based', 'hybrid'
        """
        if not similar_sqls or len(similar_sqls) == 0:
            logger.info("No similar SQLs found - using schema-based generation")
            return 'schema_based'
        
        # Ensure similar_sqls is in the right format
        if isinstance(similar_sqls[0], dict) and 'similarity' in similar_sqls[0]:
            # Check similarity scores and question matching
            best_match = similar_sqls[0]
            best_similarity = best_match['similarity']
            
            # Exact match: Very high similarity (>0.95) or exact question match
            if best_similarity > 0.95:
                logger.info(f"Exact match found (similarity: {best_similarity:.3f}) - using exact match strategy")
                return 'exact_match'
            
            # High similarity: Good match (>0.85) 
            elif best_similarity > 0.85:
                logger.info(f"High similarity match found (similarity: {best_similarity:.3f}) - using high similarity strategy")
                return 'high_similarity'
            
            # Medium similarity: Some relevant examples (>0.7)
            elif best_similarity > 0.7:
                logger.info(f"Medium similarity match found (similarity: {best_similarity:.3f}) - using hybrid strategy")
                return 'hybrid'
            
            # Low similarity: Examples not very relevant
            else:
                logger.info(f"Low similarity match (similarity: {best_similarity:.3f}) - using schema-based generation")
                return 'schema_based'
        
        # Fallback if format is unexpected
        logger.info("Unexpected similar_sqls format - using hybrid strategy")
        return 'hybrid'
    
    def _build_strategy_context(self, question: str, similar_sqls: List[Dict], strategy: str) -> tuple:
        """
        Build context and instructions based on the determined strategy.
        
        Returns:
            tuple: (sql_examples_text, strategy_instructions)
        """
        if strategy == 'exact_match':
            return self._build_exact_match_context(question, similar_sqls)
        elif strategy == 'high_similarity':
            return self._build_high_similarity_context(question, similar_sqls)
        elif strategy == 'hybrid':
            return self._build_hybrid_context(question, similar_sqls)
        else:  # schema_based
            return self._build_schema_based_context(question)
    
    def _build_exact_match_context(self, question: str, similar_sqls: List[Dict]) -> tuple:
        """Build context for exact match strategy."""
        best_match = similar_sqls[0]
        
        sql_examples_text = f"""
        ⭐ EXACT MATCH FOUND ⭐
        Question: {best_match['question']}
        Similarity Score: {best_match['similarity']:.3f}
        
        PROVEN SQL QUERY:
        {best_match['sql']}
        
        Additional Reference Examples:
        """
        
        # Add 2-3 more examples for context
        for i, sql_data in enumerate(similar_sqls[1:4], 2):
            sql_examples_text += f"""
        Example {i} (Similarity: {sql_data['similarity']:.3f}):
        Question: {sql_data['question']}
        SQL: {sql_data['sql']}
        """
        
        strategy_instructions = f"""
        === EXACT MATCH STRATEGY ===
        🎯 PRIORITY: You have found an EXACT or NEAR-EXACT match for this question!
        
        **INSTRUCTIONS:**
        1. **USE THE PROVEN SQL QUERY** from the exact match as your primary reference
        2. **MAKE MINIMAL MODIFICATIONS** only if absolutely necessary (e.g., different time periods)
        3. **PRESERVE THE CORE STRUCTURE** and table relationships from the proven query
        4. **DO NOT DEVIATE** from the established pattern unless the question explicitly requires it
        5. **TRUST THE PROVEN QUERY** - it has been validated and works correctly
        
        **MODIFICATION GUIDELINES:**
        - Only change date filters if different time periods are mentioned
        - Only change column names if different fields are requested
        - Keep the same tables, joins, and general structure
        - If the question is identical, return the exact same SQL
        
        **CRITICAL:** The proven query represents the CORRECT way to answer this type of question.
        Use it as your template and make only necessary adjustments.
        """
        
        return sql_examples_text, strategy_instructions
    
    def _build_high_similarity_context(self, question: str, similar_sqls: List[Dict]) -> tuple:
        """Build context for high similarity strategy."""
        sql_examples_text = f"""
        🔥 HIGH SIMILARITY MATCHES FOUND 🔥
        Your question has strong similarity to these proven queries:
        
        """
        
        # Add top 3-4 examples
        for i, sql_data in enumerate(similar_sqls[:4], 1):
            sql_examples_text += f"""
        Example {i} (Similarity: {sql_data['similarity']:.3f}):
        Question: {sql_data['question']}
        SQL: {sql_data['sql']}
        
        """
        
        strategy_instructions = f"""
        === HIGH SIMILARITY STRATEGY ===
        🎯 PRIORITY: Leverage the PROVEN PATTERNS from high-similarity matches!
        
        **INSTRUCTIONS:**
        1. **ANALYZE THE PATTERNS** from the high-similarity examples above
        2. **IDENTIFY COMMON STRUCTURES** across the proven queries
        3. **ADAPT THE PATTERN** to answer the current question specifically
        4. **USE SIMILAR TABLES** and join patterns that have been proven to work
        5. **FOLLOW THE ESTABLISHED CONVENTIONS** from the examples
        
        **PATTERN ANALYSIS GUIDELINES:**
        - Look for common table combinations (e.g., CustomerInvoice + CustomerInvoiceDetail)
        - Follow similar WHERE clause patterns for time filtering
        - Use similar GROUP BY and ORDER BY structures
        - Adopt proven column naming conventions
        - Maintain the same level of complexity/simplicity
        
        **ADAPTATION RULES:**
        - Modify only what's necessary to answer the current question
        - Keep the proven table relationships and join patterns
        - Use similar aggregation functions and date handling
        - Follow the same Cube.js syntax patterns
        
        **CRITICAL:** These examples show PROVEN approaches to similar questions.
        Adapt their successful patterns rather than creating entirely new structures.
        """
        
        return sql_examples_text, strategy_instructions
    
    def _build_hybrid_context(self, question: str, similar_sqls: List[Dict]) -> tuple:
        """Build context for hybrid strategy (medium similarity)."""
        sql_examples_text = f"""
        💡 REFERENCE EXAMPLES AVAILABLE 💡
        These queries provide some relevant patterns:
        
        """
        
        # Add top 3 examples
        for i, sql_data in enumerate(similar_sqls[:3], 1):
            sql_examples_text += f"""
        Example {i} (Similarity: {sql_data['similarity']:.3f}):
        Question: {sql_data['question']}
        SQL: {sql_data['sql']}
        
        """
        
        strategy_instructions = f"""
        === HYBRID STRATEGY ===
        🎯 PRIORITY: Use SELECTIVE GUIDANCE from examples + Schema-based reasoning!
        
        **INSTRUCTIONS:**
        1. **EXTRACT USEFUL PATTERNS** from the reference examples (table choices, join patterns)
        2. **APPLY SCHEMA KNOWLEDGE** to determine the best approach for this specific question
        3. **COMBINE PROVEN TECHNIQUES** with fresh analysis based on the question requirements
        4. **VALIDATE TABLE RELEVANCE** - only use tables that are actually needed
        5. **PRIORITIZE SIMPLICITY** - don't over-engineer based on complex examples
        
        **SELECTIVE GUIDANCE RULES:**
        - Adopt proven table combinations IF they're relevant to your question
        - Use similar date filtering patterns IF time-based queries are involved
        - Follow Cube.js syntax patterns from examples
        - Ignore irrelevant complexity from examples
        
        **SCHEMA-FIRST APPROACH:**
        - Start with the most direct tables for your question
        - Add joins only when necessary for the specific question
        - Use the simplest query structure that answers the question
        - Refer to examples for syntax patterns, not query structure
        
        **CRITICAL:** Examples provide technique guidance, but your primary goal is to 
        answer THIS specific question in the most direct and simple way possible.
        """
        
        return sql_examples_text, strategy_instructions
    
    def _build_schema_based_context(self, question: str) -> tuple:
        """Build context for schema-based strategy (no good matches)."""
        sql_examples_text = """
        📚 SCHEMA-BASED GENERATION 📚
        No highly relevant examples found. Generating based on schema and Cube.js best practices.
        
        FUNDAMENTAL PATTERNS TO FOLLOW:
        - Sales queries: Use CustomerInvoice + CustomerInvoiceDetail
        - Order queries: Use Order + OrderDetail  
        - Customer info: Use ViewCustomer
        - SKU info: Use Sku + Category/Brand (if needed)
        - Time filtering: Use DATE_TRUNC with appropriate intervals
        - Aggregations: Use MEASURE() inside WITH clauses, SUM/COUNT/AVG outside
        """
        
        strategy_instructions = f"""
        === SCHEMA-BASED STRATEGY ===
        🎯 PRIORITY: Generate OPTIMAL SQL based on schema analysis and question requirements!
        
        **INSTRUCTIONS:**
        1. **ANALYZE THE QUESTION** to identify required data elements
        2. **MAP TO SCHEMA TABLES** to find the most direct data sources
        3. **USE MINIMAL JOINS** - only include tables that are absolutely necessary
        4. **FOLLOW CUBE.JS PATTERNS** for syntax and functions
        5. **PRIORITIZE PERFORMANCE** - simpler queries are better
        
        **QUESTION ANALYSIS PROCESS:**
        1. Identify the main subject (sales, orders, customers, etc.)
        2. Determine required aggregations (SUM, COUNT, AVG, etc.)
        3. Identify time constraints (this month, last quarter, etc.)
        4. Determine grouping requirements (by month, by customer, etc.)
        5. Map each requirement to appropriate tables and columns
        
        **TABLE SELECTION GUIDELINES:**
        - Sales/Revenue questions → CustomerInvoice (primary)
        - Order questions → Order (primary)
        - Customer details → ViewCustomer
        - Product details → Sku, Category, Brand
        - Only join additional tables when their data is explicitly needed
        
        **CUBE.JS COMPLIANCE:**
        - Use CROSS JOIN (never INNER/LEFT/RIGHT JOIN)
        - Use MEASURE() inside WITH clauses for aggregations
        - Use DATE_TRUNC for date grouping and filtering
        - Follow the provided syntax patterns exactly
        
        **CRITICAL:** Create the SIMPLEST query that correctly answers the question.
        Avoid unnecessary complexity and focus on direct, efficient data retrieval.
        """
        
        return sql_examples_text, strategy_instructions
    
    def _build_conversation_context(self, message_log: List[Dict], question: str, 
                                   similar_sqls: List[Dict], strategy: str) -> List[Dict]:
        """
        Build intelligent conversation context based on the strategy and available examples.
        """
        if not similar_sqls or strategy == 'schema_based':
            # No examples to use, return as-is
            return message_log
            
        if strategy == 'exact_match':
            # For exact matches, add the proven Q&A pair as conversation history
            best_match = similar_sqls[0]
            message_log.append(self._create_user_message(best_match['question']))
            message_log.append(self._create_assistant_message(json.dumps({"sql": best_match['sql']})))
            
        elif strategy == 'high_similarity':
            # For high similarity, add 2-3 best examples as conversation history
            for sql_data in similar_sqls[:2]:  # Top 2 for better context
                if isinstance(sql_data, dict) and "question" in sql_data and "sql" in sql_data:
                    message_log.append(self._create_user_message(sql_data["question"]))
                    message_log.append(self._create_assistant_message(json.dumps({"sql": sql_data["sql"]})))
                    
        elif strategy == 'hybrid':
            # For hybrid, add only the best example to avoid confusion
            if len(similar_sqls) > 0:
                best_example = similar_sqls[0]
                if isinstance(best_example, dict) and "question" in best_example and "sql" in best_example:
                    message_log.append(self._create_user_message(best_example["question"]))
                    message_log.append(self._create_assistant_message(json.dumps({"sql": best_example["sql"]})))
        
        return message_log
    
    def _generate_simplified_sql(self, question: str, original_sql: str) -> str:
        """
        Generate a simplified version of complex SQL for basic aggregation queries.
        """
        question_lower = question.lower()
        
        # Pattern 1: Simple average order value for time period
        if "average order value" in question_lower:
            if "september" in question_lower:
                return '{"sql": "SELECT AVG(Order.value) AS AverageOrderValue FROM Order WHERE DATE_TRUNC(\'month\', Order.datetime) = \'2024-09-01\'"}'
            elif "this month" in question_lower:
                return '{"sql": "SELECT AVG(Order.value) AS AverageOrderValue FROM Order WHERE DATE_TRUNC(\'month\', Order.datetime) = DATE_TRUNC(\'month\', CURRENT_DATE)"}'
            elif "last month" in question_lower:
                return '{"sql": "SELECT AVG(Order.value) AS AverageOrderValue FROM Order WHERE DATE_TRUNC(\'month\', Order.datetime) = DATE_TRUNC(\'month\', CURRENT_DATE - INTERVAL \'1 month\')"}'
            elif "last quarter" in question_lower:
                return '{"sql": "SELECT AVG(Order.value) AS AverageOrderValue FROM Order WHERE Order.datetime >= DATE_TRUNC(\'quarter\', CURRENT_DATE - INTERVAL \'3 months\') AND Order.datetime < DATE_TRUNC(\'quarter\', CURRENT_DATE)"}'
        
        # Pattern 2: Simple sales totals for time period  
        if ("total sales" in question_lower or "sales for" in question_lower) and not any(breakdown in question_lower for breakdown in ["by customer", "by user", "customer wise"]):
            if "this month" in question_lower:
                return '{"sql": "SELECT SUM(CustomerInvoice.dispatchedvalue) AS TotalSales FROM CustomerInvoice WHERE DATE_TRUNC(\'month\', CustomerInvoice.dispatchedDate) = DATE_TRUNC(\'month\', CURRENT_DATE)"}'
            elif "last month" in question_lower:
                return '{"sql": "SELECT SUM(CustomerInvoice.dispatchedvalue) AS TotalSales FROM CustomerInvoice WHERE DATE_TRUNC(\'month\', CustomerInvoice.dispatchedDate) = DATE_TRUNC(\'month\', CURRENT_DATE - INTERVAL \'1 month\')"}'
        
        # Pattern 3: Simple order counts
        if "total orders" in question_lower or "number of orders" in question_lower:
            if "this month" in question_lower:
                return '{"sql": "SELECT COUNT(*) AS TotalOrders FROM Order WHERE DATE_TRUNC(\'month\', Order.datetime) = DATE_TRUNC(\'month\', CURRENT_DATE)"}'
        
        return None
    
    def _parse_sql_response(self, content: str) -> Dict[str, Any]:
        """Parse the SQL response and return structured result."""
        try:
            # Clean up the response content
            content = content.strip()
            
            # Try to parse as JSON first
            if content.startswith('{') and content.endswith('}'):
                import json
                data = json.loads(content)
                
                if "sql" in data:
                    return {
                        "success": True,
                        "sql": data["sql"],
                        "type": "query",
                        "metadata": {"response_format": "json"}
                    }
                elif "error" in data:
                    return {
                        "success": False,
                        "error": data["error"],
                        "type": "error_response"
                    }
                elif "follow_up" in data:
                    return {
                        "success": False,
                        "error": data["follow_up"],
                        "type": "follow_up_needed"
                    }
            
            # If not JSON, try to extract SQL directly
            if content.upper().strip().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
                return {
                    "success": True,
                    "sql": content,
                    "type": "query",
                    "metadata": {"response_format": "raw_sql"}
                }
            
            # If we can't parse it, return error
            return {
                "success": False,
                "error": "Could not parse SQL query from response",
                "type": "parsing_error",
                "raw_response": content
            }
            
        except Exception as e:
            logger.error(f"Error parsing SQL response: {e}")
            return {
                "success": False,
                "error": f"Response parsing error: {str(e)}",
                "type": "parsing_error",
                "raw_response": content
            }
                
        except Exception as e:
            logger.error(f"Error parsing SQL response: {e}")
            return {
                "success": False,
                "error": f"Response parsing error: {str(e)}",
                "type": "parsing_error",
                "raw_response": content
            }
    
    def _get_adaptive_temperature_llm(self, similar_sqls: List[Dict], generation_strategy: str):
        """
        Create LLM instance with adaptive temperature based on similarity scores and strategy.
        
        Temperature Strategy:
        - High similarity (>0.85): Low temperature (0.05) - stick close to proven patterns
        - Medium similarity (0.7-0.85): Medium temperature (0.15) - balanced creativity  
        - Low similarity (<0.7): High temperature (0.3-0.4) - more creative and exploratory
        - Schema-based: Highest temperature (0.4) - maximum creativity for novel queries
        """
        try:
            from langchain_openai import ChatOpenAI
            import os
            
            # Determine optimal temperature based on similarity and strategy
            if generation_strategy == 'exact_match':
                temperature = 0.02  # Very low - almost deterministic for exact matches
                temp_reason = "exact match - deterministic"
            elif generation_strategy == 'high_similarity':
                temperature = 0.05  # Low - stay close to proven patterns
                temp_reason = "high similarity - follow patterns"
            elif generation_strategy == 'hybrid':
                # Check actual similarity score for fine-tuning
                if similar_sqls and len(similar_sqls) > 0:
                    similarity = similar_sqls[0].get('similarity', 0.7)
                    if similarity > 0.75:
                        temperature = 0.15  # Medium-low
                        temp_reason = f"hybrid with good similarity ({similarity:.3f}) - moderate creativity"
                    else:
                        temperature = 0.25  # Medium-high  
                        temp_reason = f"hybrid with lower similarity ({similarity:.3f}) - increased creativity"
                else:
                    temperature = 0.2  # Default medium
                    temp_reason = "hybrid - balanced approach"
            else:  # schema_based
                # Low similarity - need maximum creativity
                if similar_sqls and len(similar_sqls) > 0:
                    similarity = similar_sqls[0].get('similarity', 0.5)
                    if similarity < 0.5:
                        temperature = 0.4  # High creativity for very low similarity
                        temp_reason = f"very low similarity ({similarity:.3f}) - maximum creativity"
                    elif similarity < 0.65:
                        temperature = 0.3  # High creativity for low similarity
                        temp_reason = f"low similarity ({similarity:.3f}) - high creativity"
                    else:
                        temperature = 0.25  # Medium-high for borderline cases
                        temp_reason = f"borderline similarity ({similarity:.3f}) - enhanced creativity"
                else:
                    temperature = 0.35  # High creativity when no context
                    temp_reason = "no similar context - high creativity"
            
            logger.info(f"🌡️  ADAPTIVE TEMPERATURE: {temperature} ({temp_reason})")
            
            # Create new LLM instance with adaptive temperature
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
    
    def process(self, state: BaseAgentState) -> DBAgentState:
        """
        Process method required by BaseAgent interface.
        Generates a single SQL query for the given question.
        """
        db_state = DBAgentState(**state)
        
        try:
            similar_sqls = state.get("retrieved_sql_context", [])
            previous_results = state.get("previous_step_results", None)
            
            result = self.generate_sql(
                question=state["query"],
                similar_sqls=similar_sqls,
                previous_results=previous_results
            )
            
            if result["success"]:
                db_state["query_type"] = result["query_type"]
                db_state["sql_query"] = result["sql"]
                db_state["status"] = "completed"
                db_state["success_message"] = "Generated SQL query successfully"
                db_state["result"] = result
                
                # Use clean logging
                try:
                    from clean_logging import SQLLogger
                    SQLLogger.generation_complete(result['sql'], result['query_type'])
                except ImportError:
                    logger.info(f"📝 SQL Generated ({result['query_type']}): {len(result['sql'])} chars")
                
            else:
                db_state["error_message"] = result["error"]
                db_state["status"] = "failed"
                db_state["result"] = result
                
                logger.error(f"🔧 SQLGeneratorAgent - Generation Failed:")
                logger.error(f"Question: {state['query']}")
                logger.error(f"Error: {result['error']}")
                logger.error(f"Error Type: {result.get('type', 'unknown')}")
                
        except Exception as e:
            db_state["error_message"] = f"SQL generator error: {str(e)}"
            db_state["status"] = "failed"
            logger.error(f"SQLGeneratorAgent process failed: {e}")
        
        return db_state