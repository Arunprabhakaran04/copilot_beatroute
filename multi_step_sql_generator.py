"""
Multi-Step SQL Generator - Optimized for Cube.js Compatibility
================================================================

This generator is specifically designed for multi-step queries where each step
must be a SIMPLE, FLAT query that Cube.js can understand.

Key Differences from improved_sql_generator.py:
1. ENFORCES flat queries (no CTEs, no subqueries, no nested SELECTs)
2. Uses previous step results as filters instead of complex logic
3. Breaks down complex operations into multiple simple steps
4. Stricter validation for Cube.js compatibility

When to Use:
- Multi-step queries (step 2+) where previous results are available
- Queries that need to reference previous step data
- Complex aggregations that should be split across steps

When NOT to Use:
- Single-step queries (use improved_sql_generator.py)
- Step 1 of multi-step queries (use improved_sql_generator.py)
"""

import re
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from base_agent import BaseAgent
from token_tracker import track_llm_call
from loguru import logger


class MultiStepSQLGenerator(BaseAgent):
    """
    Specialized SQL Generator for multi-step queries with strict Cube.js compatibility.
    Enforces flat query structure and proper use of previous step results.
    """
    
    def __init__(self, llm, schema_file_path: str = None, schema_manager=None):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.schema_manager = schema_manager
        
        # Cube.js compatibility rules (STRICT for multi-step)
        self.cube_rules = {
            "forbidden_patterns": [
                r"WITH\s+\w+\s+AS",  # CTEs
                r"SELECT.*FROM\s*\(",  # Subqueries in FROM
                r"WHERE.*IN\s*\(SELECT",  # Subqueries in WHERE
                r"ROW_NUMBER\(\)",  # Window functions (unless proven working)
                r"OVER\s*\(",  # Window functions
                r"CASE\s+WHEN.*THEN.*ELSE.*END",  # Complex CASE (unless simple)
            ],
            "required_patterns": [
                r"FROM\s+\w+",  # Must have FROM clause
                r"CROSS\s+JOIN",  # Must use CROSS JOIN (not regular JOIN)
            ],
            "allowed_aggregations": [
                "SUM", "COUNT", "AVG", "MIN", "MAX", "MEASURE"
            ]
        }
    
    def get_agent_type(self) -> str:
        return "multi_step_sql_generator"
    
    def process(self, state: Any) -> Any:
        """
        Process method required by BaseAgent abstract class.
        This generator is called directly via generate_sql(), not through process().
        """
        logger.warning("MultiStepSQLGenerator.process() called - this should not happen in normal flow")
        logger.warning("Use generate_sql() method instead")
        
        # Return state unchanged
        from base_agent import BaseAgentState
        if isinstance(state, dict):
            return BaseAgentState(**state)
        return state
    
    def _validate_cube_compatibility(self, sql: str) -> Dict[str, Any]:
        """
        Validate SQL against strict Cube.js rules for multi-step queries.
        Returns: {"valid": bool, "violations": List[str], "warnings": List[str]}
        """
        violations = []
        warnings = []
        
        # Check forbidden patterns
        for pattern in self.cube_rules["forbidden_patterns"]:
            if re.search(pattern, sql, re.IGNORECASE):
                violations.append(f"Found forbidden pattern: {pattern}")
        
        # Check for CTEs specifically (most common issue)
        if "WITH " in sql.upper() and " AS (" in sql.upper():
            violations.append("CTEs (WITH clause) are not supported by Cube.js - use flat queries with previous step results")
        
        # Check for subqueries
        if sql.count("(SELECT") > 0:
            violations.append("Subqueries are not supported - use previous step results instead")
        
        # Check for JOIN instead of CROSS JOIN
        if re.search(r'\sJOIN\s+(?!.*CROSS)', sql, re.IGNORECASE):
            warnings.append("Use CROSS JOIN instead of regular JOIN")
        
        # Check if using CustomerInvoiceDetail without proper table reference
        if "CustomerInvoiceDetail" in sql and "FROM CustomerInvoiceDetail" not in sql:
            warnings.append("CustomerInvoiceDetail should be joined via CustomerInvoice CROSS JOIN CustomerInvoiceDetail")
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings
        }
    
    def _extract_previous_step_values(self, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract usable values from previous step results for filtering.
        
        Returns: {
            "sku_names": ["SKU1", "SKU2", ...],
            "customer_names": ["Customer1", ...],
            "dates": ["2024-01-01", ...],
            "ids": [123, 456, ...],
            ...
        }
        """
        extracted = {
            "sku_names": [],
            "customer_names": [],
            "dates": [],
            "ids": [],
            "values": []
        }
        
        if not previous_results:
            logger.warning("No previous_results provided to multi-step generator")
            return extracted
        
        logger.info(f"📥 Extracting values from {len(previous_results)} previous step(s)")
        
        # Iterate through all previous steps
        for step_key, step_data in previous_results.items():
            if not isinstance(step_data, dict):
                logger.warning(f"Step {step_key} data is not a dict: {type(step_data)}")
                continue
            
            logger.debug(f"Processing {step_key}, keys: {list(step_data.keys())}")
            
            # Look for query_results -> data
            query_results = step_data.get("query_results", {})
            
            # Handle case where query_results might be a string (JSON)
            if isinstance(query_results, str):
                try:
                    query_results = json.loads(query_results)
                    logger.debug(f"Parsed query_results from JSON string for {step_key}")
                except:
                    logger.warning(f"Failed to parse query_results JSON for {step_key}")
                    query_results = {}
            
            data_list = query_results.get("data", [])
            
            # Handle case where data might be a JSON string
            if isinstance(data_list, str):
                try:
                    data_list = json.loads(data_list)
                    logger.debug(f"Parsed data from JSON string for {step_key}: {len(data_list)} rows")
                except:
                    logger.warning(f"Failed to parse data JSON for {step_key}")
                    data_list = []
            
            if not data_list:
                # Try json_results as backup
                json_results = step_data.get("json_results", "[]")
                if isinstance(json_results, str):
                    try:
                        data_list = json.loads(json_results)
                        logger.info(f"Extracted data from json_results for {step_key}: {len(data_list)} rows")
                    except:
                        logger.warning(f"No data available in {step_key}")
                        continue
            
            if not data_list:
                logger.warning(f"No data found in {step_key} after trying all sources")
                continue
            
            logger.info(f"✅ Extracting values from {step_key}: {len(data_list)} rows")
            
            # Extract values from each row
            for row in data_list:
                if not isinstance(row, dict):
                    logger.debug(f"Row is not a dict: {type(row)}")
                    continue
                
                for key, value in row.items():
                    key_lower = key.lower()
                    
                    # SKU identification
                    if "sku" in key_lower and "name" in key_lower:
                        if value and str(value).strip():
                            extracted["sku_names"].append(str(value))
                    
                    # Customer identification
                    elif "customer" in key_lower and "name" in key_lower:
                        if value and str(value).strip():
                            extracted["customer_names"].append(str(value))
                    
                    # Date identification
                    elif "date" in key_lower or "month" in key_lower or "year" in key_lower:
                        if value:
                            extracted["dates"].append(str(value))
                    
                    # ID identification
                    elif "id" in key_lower:
                        if value:
                            extracted["ids"].append(value)
                    
                    # Generic numeric values
                    elif isinstance(value, (int, float)):
                        extracted["values"].append(value)
        
        # Deduplicate and log
        for key in extracted:
            extracted[key] = list(set(extracted[key]))
            if extracted[key]:
                logger.info(f"  Extracted {len(extracted[key])} unique {key}: {extracted[key][:5]}{'...' if len(extracted[key]) > 5 else ''}")
        
        return extracted
    
    def _build_filter_clause(self, extracted_values: Dict[str, Any], question: str) -> str:
        """
        Build WHERE clause filter based on extracted values and question context.
        
        Returns: SQL WHERE clause fragment (without WHERE keyword)
        """
        filters = []
        
        # SKU filtering
        if extracted_values.get("sku_names"):
            sku_names = extracted_values["sku_names"]
            if len(sku_names) == 1:
                filters.append(f"Sku.name = '{sku_names[0]}'")
            else:
                sku_list = "', '".join(sku_names[:50])  # Limit to 50 to avoid huge IN clauses
                filters.append(f"Sku.name IN ('{sku_list}')")
        
        # Customer filtering
        if extracted_values.get("customer_names"):
            customer_names = extracted_values["customer_names"]
            if len(customer_names) == 1:
                filters.append(f"ViewCustomer.name = '{customer_names[0]}'")
            else:
                customer_list = "', '".join(customer_names[:50])
                filters.append(f"ViewCustomer.name IN ('{customer_list}')")
        
        # Date filtering (extract from question)
        time_filter = self._extract_time_filter(question)
        if time_filter:
            filters.append(time_filter)
        
        return " AND ".join(filters) if filters else "1=1"
    
    def _extract_time_filter(self, question: str) -> Optional[str]:
        """Extract time-based filter from question."""
        question_lower = question.lower()
        
        # Last X months pattern
        if "last 12 months" in question_lower or "12 months" in question_lower:
            return "CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '12 months'"
        elif "last 6 months" in question_lower or "6 months" in question_lower:
            return "CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '6 months'"
        elif "last 3 months" in question_lower or "3 months" in question_lower:
            return "CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '3 months'"
        elif "last month" in question_lower:
            return "CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'"
        elif "this month" in question_lower:
            return "CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE)"
        
        return None
    
    def _create_multi_step_system_prompt(self, question: str, previous_results: Dict[str, Any],
                                         extracted_values: Dict[str, Any], focused_schema: str = None) -> str:
        """
        Create system prompt specifically for multi-step queries.
        """
        
        # Build filter guidance
        filter_guidance = "No previous step values available - generate standalone query."
        if any(extracted_values.values()):
            filter_guidance = "Use these values from previous steps in your WHERE clause:\n"
            if extracted_values.get("sku_names"):
                filter_guidance += f"   - SKU Names: {extracted_values['sku_names'][:5]}\n"
            if extracted_values.get("customer_names"):
                filter_guidance += f"   - Customer Names: {extracted_values['customer_names'][:5]}\n"
        
        prompt = f"""You are a SQL generator for Cube.js database with STRICT multi-step query rules.

===========================================
CRITICAL: MULTI-STEP QUERY RULES
===========================================

This is Step N of a multi-step query. Previous steps have already been executed.

🚫 ABSOLUTELY FORBIDDEN:
1. WITH clauses (CTEs) - Cube.js does NOT support CTEs
2. Subqueries - Use previous step results instead
3. Nested SELECT statements
4. Complex window functions (ROW_NUMBER, RANK, etc.)
5. Self-joins or complex join logic

✅ REQUIRED PATTERNS:
1. FLAT queries: SELECT ... FROM TableA CROSS JOIN TableB WHERE ...
2. Use CROSS JOIN (never regular JOIN)
3. Use previous step results in WHERE clause
4. Simple aggregations: SUM(), COUNT(), AVG(), MEASURE()
5. Direct table references only

===========================================
PREVIOUS STEP VALUES (Use in WHERE clause)
===========================================

{filter_guidance}

===========================================
QUERY TEMPLATES
===========================================

For "monthly trend of specific SKUs" (SECONDARY SALES):
```sql
SELECT
    Sku.name AS SkuName,
    DATE_TRUNC('month', CustomerInvoice.dispatchedDate) AS MonthYear,
    MEASURE(CustomerInvoiceDetail.skuAmount) AS MonthlySalesValue
FROM CustomerInvoice
CROSS JOIN CustomerInvoiceDetail
CROSS JOIN Sku
WHERE
    CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '12 months'
    AND Sku.name IN ('SKU1', 'SKU2', 'SKU3')
GROUP BY
    Sku.name,
    DATE_TRUNC('month', CustomerInvoice.dispatchedDate)
ORDER BY
    Sku.name,
    MonthYear
```

⚠️ CRITICAL TABLE SELECTION:
- SECONDARY SALES (customer dispatch) = CustomerInvoice + CustomerInvoiceDetail
  * Date field: CustomerInvoice.dispatchedDate
  * Amount field: CustomerInvoiceDetail.skuAmount
  * SKU name: Sku.name (join via CROSS JOIN Sku)
  
- PRIMARY SALES (distributor) = DistributorSales + DistributorSalesDetail
  * Date field: DistributorSales.erp_invoice_date
  * Amount field: DistributorSalesDetail.total_value
  * SKU name: Sku.name (join via CROSS JOIN Sku)

DEFAULT: Use CustomerInvoice for SKU sales trends unless explicitly asked for distributor/primary sales.

For "customer-specific aggregation":
```sql
SELECT
    ViewCustomer.name AS CustomerName,
    MEASURE(CustomerInvoice.dispatchedvalue) AS TotalSales
FROM CustomerInvoice
CROSS JOIN ViewCustomer
WHERE
    ViewCustomer.external_id = CustomerInvoice.externalCode
    AND ViewCustomer.name IN ('Customer1', 'Customer2')
    AND CustomerInvoice.dispatchedDate >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '3 months'
GROUP BY
    ViewCustomer.name
ORDER BY
    TotalSales DESC
```

===========================================
TABLE RELATIONSHIPS (CRITICAL)
===========================================

CustomerInvoice ↔ ViewCustomer:
   WHERE ViewCustomer.external_id = CustomerInvoice.externalCode

CustomerInvoice ↔ CustomerInvoiceDetail:
   FROM CustomerInvoice CROSS JOIN CustomerInvoiceDetail
   (No explicit filter needed - implicit relationship)

CustomerInvoiceDetail ↔ Sku:
   FROM CustomerInvoiceDetail CROSS JOIN Sku
   WHERE Sku.name = CustomerInvoiceDetail.skuName

===========================================
SCHEMA (Focused)
===========================================

{focused_schema if focused_schema else "Full schema not provided - use core tables: CustomerInvoice, CustomerInvoiceDetail, ViewCustomer, Sku"}

===========================================
RESPONSE FORMAT
===========================================

Return ONLY valid JSON:
{{
    "sql": "SELECT ... FROM ... WHERE ...",
    "explanation": "Brief explanation of the query logic"
}}

Question: {question}

Generate a FLAT SQL query using previous step values. No CTEs, no subqueries, no window functions.
"""
        
        return prompt
    
    def generate_sql(self, question: str, similar_sqls: List[Dict] = None,
                     previous_results: Dict[str, Any] = None,
                     original_query: str = None,
                     entity_info: Dict[str, Any] = None,
                     conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SQL for multi-step queries with strict Cube.js compatibility.
        
        This method:
        1. Extracts values from previous step results
        2. Creates a focused prompt with previous values
        3. Generates FLAT SQL (no CTEs, no subqueries)
        4. Validates against Cube.js rules
        5. Returns corrected SQL if needed
        """
        try:
            # Validate this is actually a multi-step scenario
            if not previous_results:
                logger.warning("⚠️  MultiStepSQLGenerator called without previous_results - consider using ImprovedSQLGenerator instead")
            
            # Extract values from previous steps
            extracted_values = self._extract_previous_step_values(previous_results)
            
            # Generate SQL using LLM
            result = self._generate_with_llm_multistep(
                question=question,
                similar_sqls=similar_sqls,
                previous_results=previous_results,
                extracted_values=extracted_values,
                entity_info=entity_info,
                conversation_history=conversation_history
            )
            
            # Validate Cube.js compatibility
            if result.get("success") and result.get("sql"):
                validation = self._validate_cube_compatibility(result["sql"])
                
                if not validation["valid"]:
                    logger.error(f"❌ Generated SQL violates Cube.js rules:")
                    for violation in validation["violations"]:
                        logger.error(f"   - {violation}")
                    
                    # Attempt auto-correction
                    corrected_sql = self._attempt_correction(result["sql"], validation)
                    if corrected_sql:
                        logger.info("✅ Auto-corrected SQL to Cube.js compatible format")
                        result["sql"] = corrected_sql
                        result["auto_corrected"] = True
                    else:
                        result["success"] = False
                        result["error"] = f"Generated SQL violates Cube.js rules: {'; '.join(validation['violations'])}"
                
                if validation["warnings"]:
                    logger.warning("⚠️  SQL warnings:")
                    for warning in validation["warnings"]:
                        logger.warning(f"   - {warning}")
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-step SQL generation error: {e}")
            return {
                "success": False,
                "error": f"Multi-step SQL generation error: {str(e)}",
                "type": "generation_error"
            }
    
    def _attempt_correction(self, sql: str, validation: Dict[str, Any]) -> Optional[str]:
        """
        Attempt to auto-correct SQL to be Cube.js compatible.
        Returns corrected SQL or None if correction not possible.
        """
        corrected = sql
        
        # Remove CTEs by flattening
        if "WITH " in corrected.upper():
            logger.info("Attempting to remove CTEs...")
            # This is complex - for now, return None to trigger regeneration
            return None
        
        # Replace regular JOIN with CROSS JOIN
        if re.search(r'\sJOIN\s+(?!.*CROSS)', corrected, re.IGNORECASE):
            corrected = re.sub(r'\sJOIN\s+', ' CROSS JOIN ', corrected, flags=re.IGNORECASE)
            logger.info("Replaced JOIN with CROSS JOIN")
        
        # Remove subqueries (basic attempt)
        if "(SELECT" in corrected.upper():
            logger.info("Cannot auto-correct subqueries - regeneration needed")
            return None
        
        return corrected if corrected != sql else None
    
    def _generate_with_llm_multistep(self, question: str, similar_sqls: List[Dict],
                                      previous_results: Dict[str, Any],
                                      extracted_values: Dict[str, Any],
                                      entity_info: Dict[str, Any] = None,
                                      conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate SQL using LLM with multi-step context.
        """
        import time
        start_time = time.time()
        
        # Get focused schema
        cached_schema = entity_info.get("cached_focused_schema") if entity_info else None
        focused_schema = None
        
        if cached_schema:
            focused_schema = cached_schema
            logger.info("💾 Using cached focused schema for multi-step query")
        elif self.schema_manager:
            try:
                focused_schema = self.schema_manager.get_schema_to_use_in_prompt(
                    current_question=question,
                    list_similar_question_sql_pair=similar_sqls or [],
                    k=10
                )
            except Exception as e:
                logger.warning(f"Failed to get focused schema: {e}")
        
        # Create multi-step prompt
        system_prompt = self._create_multi_step_system_prompt(
            question=question,
            previous_results=previous_results,
            extracted_values=extracted_values,
            focused_schema=focused_schema
        )
        
        # Build message log (simplified - fewer examples for multi-step)
        message_log = [{"role": "system", "content": system_prompt}]
        
        # Add only top 5 most similar examples (multi-step needs less noise)
        if similar_sqls:
            top_examples = sorted(similar_sqls, key=lambda x: x.get('similarity', 0), reverse=True)[:5]
            for example in top_examples:
                if example.get('question') and example.get('sql'):
                    message_log.append({
                        "role": "user",
                        "content": example['question']
                    })
                    message_log.append({
                        "role": "assistant",
                        "content": json.dumps({"sql": example['sql'], "explanation": "Example query"})
                    })
        
        # Add current question
        message_log.append({"role": "user", "content": f"Generate SQL for: {question}"})
        
        # Call LLM with deterministic settings
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.05,  # Very low temperature for deterministic output
            seed=42,
            max_tokens=1500
        )
        
        response = llm.invoke(message_log)
        content = response.content.strip()
        
        # Parse JSON response
        try:
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            sql = parsed.get("sql", "").strip()
            explanation = parsed.get("explanation", "")
            
            if not sql:
                raise ValueError("No SQL in response")
            
            generation_time = time.time() - start_time
            logger.info(f"✅ Multi-step SQL generated in {generation_time:.2f}s")
            
            return {
                "success": True,
                "sql": sql,
                "explanation": explanation,
                "method": "multi_step_llm",
                "generation_time": generation_time,
                "extracted_values_count": sum(len(v) for v in extracted_values.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response content: {content[:500]}")
            return {
                "success": False,
                "error": f"Failed to parse SQL response: {str(e)}",
                "type": "parse_error"
            }
