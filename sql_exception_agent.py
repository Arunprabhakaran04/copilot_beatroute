import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from base_agent import BaseAgent, BaseAgentState, DBAgentState

logger = logging.getLogger(__name__)

class SQLExceptionAgent(BaseAgent):
    """
    SQL Exception Agent - A thinking agent that analyzes SQL execution errors,
    understands the root cause, and iteratively fixes SQL queries.
    
    This agent acts like a coding agent that learns from failures and improves its solutions.
    """
    
    def __init__(self, llm, schema_file_path: str = "schema", max_iterations: int = 3):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.max_iterations = max_iterations
        self.schema_content = self._load_schema_file()
        
        # Error patterns and their solutions
        self.error_patterns = {
            "syntax_error": {
                "patterns": [
                    "syntax error", "invalid syntax", "unexpected token",
                    "malformed query", "parse error"
                ],
                "category": "syntax"
            },
            "column_error": {
                "patterns": [
                    "column does not exist", "unknown column", "column not found",
                    "ambiguous column", "column reference"
                ],
                "category": "schema"
            },
            "table_error": {
                "patterns": [
                    "table does not exist", "unknown table", "table not found",
                    "relation does not exist"
                ],
                "category": "schema"
            },
            "join_error": {
                "patterns": [
                    "join condition", "cross join", "join clause",
                    "missing join", "invalid join"
                ],
                "category": "relationship"
            },
            "function_error": {
                "patterns": [
                    "function does not exist", "unknown function", "function not found",
                    "aggregate function", "MEASURE", "TO_CHAR", "EXTRACT"
                ],
                "category": "function"
            },
            "date_error": {
                "patterns": [
                    "date format", "datetime", "timestamp", "DATE_TRUNC",
                    "date function", "time zone"
                ],
                "category": "date"
            },
            "cube_js_error": {
                "patterns": [
                    "cube.js", "not supported in cube", "cube sql api",
                    "measure outside", "select *"
                ],
                "category": "cube_js"
            }
        }
        
        logger.info("SQLExceptionAgent initialized with advanced error analysis capabilities")
        logger.info(f"Max iterations: {max_iterations}")
        logger.info(f"Error patterns loaded: {len(self.error_patterns)} categories")
    
    def get_agent_type(self) -> str:
        return "sql_exception"
    
    def _load_schema_file(self) -> str:
        """Load the database schema for error analysis."""
        try:
            with open(self.schema_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                content = content.replace('\\n', '\n')
                return content
        except FileNotFoundError:
            logger.error(f"Schema file not found: {self.schema_file_path}")
            return "Schema file not found."
        except Exception as e:
            logger.error(f"Error loading schema file: {e}")
            return f"Error loading schema: {str(e)}"
    
    def analyze_and_fix_sql_error(self, original_question: str, failed_sql: str, 
                                  error_message: str, similar_sqls: List[str] = None,
                                  previous_attempts: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main method to analyze SQL error and generate a corrected query.
        
        Args:
            original_question: The user's original question
            failed_sql: The SQL query that failed
            error_message: The error message from SQL execution
            similar_sqls: Context examples for reference
            previous_attempts: List of previous fix attempts
            
        Returns:
            Dict with analysis results and corrected SQL
        """
        try:
            logger.info(f"🔧 ANALYZING SQL ERROR:")
            logger.info(f"Question: {original_question}")
            logger.info(f"Failed SQL: {failed_sql}")
            logger.info(f"Error: {error_message}")
            
            # Step 1: Categorize the error
            error_analysis = self._categorize_error(error_message, failed_sql)
            
            # Step 2: Perform deep error analysis
            root_cause = self._analyze_root_cause(
                original_question, failed_sql, error_message, error_analysis
            )
            
            # Step 3: Generate corrected SQL
            correction_result = self._generate_corrected_sql(
                original_question=original_question,
                failed_sql=failed_sql,
                error_message=error_message,
                error_analysis=error_analysis,
                root_cause=root_cause,
                similar_sqls=similar_sqls or [],
                previous_attempts=previous_attempts or []
            )
            
            return {
                "success": correction_result.get("success", False),
                "corrected_sql": correction_result.get("sql", ""),
                "error_analysis": error_analysis,
                "root_cause": root_cause,
                "correction_explanation": correction_result.get("explanation", ""),
                "confidence_score": correction_result.get("confidence", 0.0),
                "fix_type": correction_result.get("fix_type", "unknown"),
                "learning_points": correction_result.get("learning_points", [])
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_and_fix_sql_error: {e}")
            return {
                "success": False,
                "error": f"Exception analysis failed: {str(e)}",
                "error_analysis": {"category": "analysis_error", "severity": "high"}
            }
    
    def _categorize_error(self, error_message: str, failed_sql: str) -> Dict[str, Any]:
        """Categorize the error based on patterns and content analysis."""
        error_lower = error_message.lower()
        sql_lower = failed_sql.lower()
        
        matched_categories = []
        severity = "low"
        
        # Check each error pattern
        for error_type, config in self.error_patterns.items():
            for pattern in config["patterns"]:
                if pattern.lower() in error_lower or pattern.lower() in sql_lower:
                    matched_categories.append({
                        "type": error_type,
                        "category": config["category"],
                        "pattern": pattern,
                        "confidence": self._calculate_pattern_confidence(pattern, error_message, failed_sql)
                    })
        
        # Determine severity
        if any("syntax" in cat["category"] for cat in matched_categories):
            severity = "high"
        elif any("schema" in cat["category"] for cat in matched_categories):
            severity = "medium"
        elif any("function" in cat["category"] for cat in matched_categories):
            severity = "medium"
        
        # Sort by confidence
        matched_categories.sort(key=lambda x: x["confidence"], reverse=True)
        
        primary_category = matched_categories[0]["category"] if matched_categories else "unknown"
        
        return {
            "primary_category": primary_category,
            "severity": severity,
            "matched_patterns": matched_categories,
            "error_complexity": len(matched_categories),
        }
    
    def _calculate_pattern_confidence(self, pattern: str, error_message: str, failed_sql: str) -> float:
        """Calculate confidence score for pattern match."""
        error_matches = error_message.lower().count(pattern.lower())
        sql_matches = failed_sql.lower().count(pattern.lower())
        
        # Base confidence from exact matches
        base_confidence = min((error_matches + sql_matches) * 0.3, 1.0)
        
        # Boost for specific patterns
        if pattern in ["syntax error", "column does not exist", "table does not exist"]:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _analyze_root_cause(self, question: str, failed_sql: str, error_message: str, 
                           error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep root cause analysis of the SQL error."""
        
        root_cause_prompt = f"""You are an expert SQL error analyst. Analyze the SQL error and identify the root cause.

ERROR CONTEXT:
- Original Question: {question}
- Failed SQL Query: {failed_sql}
- Error Message: {error_message}
- Error Category: {error_analysis['primary_category']}
- Severity: {error_analysis['severity']}

ANALYSIS REQUIREMENTS:
1. Identify the specific root cause of the error
2. Explain why this error occurred in the context of Cube.js SQL API
3. Suggest the type of fix needed
4. Rate the complexity of the fix (1-10)

Respond in JSON format:
{{
    "root_cause": "specific reason for the error",
    "why_it_happened": "explanation of the underlying issue",
    "fix_complexity": 1-10,
    "fix_approach": "high-level approach to fix the error",
    "key_insights": ["insight1", "insight2", "insight3"]
}}"""

        try:
            message_log = [
                {"role": "system", "content": "You are an expert SQL error analyst specialized in Cube.js SQL API."},
                {"role": "user", "content": root_cause_prompt}
            ]
            
            response = self.llm.invoke(message_log)
            content = response.content.strip()
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
        except Exception as e:
            logger.error(f"Root cause analysis failed: {e}")
        
        # Fallback analysis
        return {
            "root_cause": f"Error in {error_analysis['primary_category']} category",
            "why_it_happened": "Analysis could not determine specific cause",
            "fix_complexity": 5,
            "fix_approach": "Apply standard fixes for this error category",
            "key_insights": ["Manual analysis required"]
        }
    
    def _generate_corrected_sql(self, original_question: str, failed_sql: str, 
                               error_message: str, error_analysis: Dict[str, Any],
                               root_cause: Dict[str, Any], similar_sqls: List[str],
                               previous_attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate corrected SQL query based on error analysis."""
        
        # Format similar SQLs for context
        sql_examples_text = ""
        if similar_sqls:
            sql_examples_text = "\n".join([
                f"Example {i+1}:\n{sql}" for i, sql in enumerate(similar_sqls[:5])
            ])
        
        # Format previous attempts
        previous_attempts_text = ""
        if previous_attempts:
            previous_attempts_text = "\n".join([
                f"Attempt {i+1}: {attempt['sql']} -> Error: {attempt.get('error', 'Unknown')}"
                for i, attempt in enumerate(previous_attempts)
            ])
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        correction_prompt = f"""You are an expert SQL error correction specialist for Cube.js SQL API.
Your task is to fix the SQL query that failed and generate a corrected version.

CONTEXT:
- Original Question: {original_question}
- Failed SQL: {failed_sql}
- Error Message: {error_message}
- Error Category: {error_analysis['primary_category']}
- Root Cause: {root_cause.get('root_cause', 'Unknown')}
- Fix Complexity: {root_cause.get('fix_complexity', 5)}/10
- Today's Date: {current_date}

PREVIOUS ATTEMPTS:
{previous_attempts_text if previous_attempts_text else "This is the first attempt"}

REFERENCE SQL EXAMPLES:
{sql_examples_text if sql_examples_text else "No examples available"}

DATABASE SCHEMA:
{self.schema_content}

CUBE.JS SQL API RULES (CRITICAL):
1. ONLY use CROSS JOIN - NO other join types
2. NO JOIN conditions with ON clause
3. Use MEASURE() inside WITH clauses only
4. DO NOT use SELECT *
5. NO unsupported functions: TO_CHAR(), EXTRACT(), COALESCE()
6. Use DATE_TRUNC() for date operations
7. Always use proper aliases in WITH clauses
8. NO subqueries in WHERE clauses

CORRECTION STRATEGY:
Based on the error analysis, apply these fixes:

For SYNTAX errors:
- Fix SQL syntax issues
- Correct parentheses, commas, quotes
- Fix keyword placement

For SCHEMA errors:
- Use correct table/column names from schema
- Add missing CROSS JOINs for related tables
- Use proper table aliases

For FUNCTION errors:
- Replace unsupported functions with Cube.js compatible ones
- Use MEASURE() only in WITH clauses
- Replace TO_CHAR with proper formatting

For CUBE.JS specific errors:
- Convert to CROSS JOIN only
- Remove SELECT *
- Move MEASURE to WITH clause
- Fix date functions

LEARNING FROM ERRORS:
- Analyze what went wrong in the failed SQL
- Identify the specific issue that needs fixing
- Apply targeted corrections without changing working parts

Respond ONLY in this JSON format:
{{
    "success": true/false,
    "sql": "corrected SQL query here",
    "explanation": "explanation of what was fixed",
    "confidence": 0.0-1.0,
    "fix_type": "type of fix applied",
    "learning_points": ["what was learned from this fix"]
}}

If the error cannot be fixed, set success to false and explain why in the explanation field."""

        try:
            # Clear system message to focus on correction task
            system_msg = "You are an expert SQL correction specialist. Your task is to fix the provided SQL query and return a corrected version in JSON format."
            
            message_log = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"TASK: SQL CORRECTION\n\n{correction_prompt}"}
            ]
            
            response = self.llm.invoke(message_log)
            content = response.content.strip()
            
            logger.debug(f"LLM response for SQL correction: {content[:200]}...")
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                logger.debug(f"Extracted JSON content: {json_content}")
                
                try:
                    parsed_response = json.loads(json_content)
                    logger.debug(f"Parsed JSON response: {parsed_response}")
                    
                    # Validate the response
                    if "sql" in parsed_response and parsed_response.get("success", False):
                        # Additional validation of the corrected SQL
                        corrected_sql = parsed_response["sql"]
                        validation_issues = self._validate_corrected_sql(corrected_sql)
                        
                        if validation_issues:
                            parsed_response["success"] = False
                            parsed_response["explanation"] = f"Validation failed: {'; '.join(validation_issues)}"
                            logger.warning(f"SQL validation failed: {validation_issues}")
                        else:
                            logger.info(f" SQL correction successful: {corrected_sql}")
                        
                    return parsed_response
                    
                except json.JSONDecodeError as je:
                    logger.error(f"JSON parsing failed: {je}")
                    logger.error(f"JSON content that failed: {json_content}")
            else:
                logger.error("No JSON found in LLM response")
                logger.error(f"Full response: {content}")
            
        except Exception as e:
            logger.error(f"SQL correction generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Fallback response
        logger.warning("Using fallback response for SQL correction")
        return {
            "success": False,
            "sql": "",
            "explanation": "Could not generate corrected SQL - LLM response parsing failed",
            "confidence": 0.0,
            "fix_type": "parsing_failed",
            "learning_points": ["LLM response could not be parsed", "Check response format"]
        }
    
    def _validate_corrected_sql(self, sql: str) -> List[str]:
        """Validate the corrected SQL against Cube.js rules."""
        issues = []
        sql_lower = sql.lower()
        
        # Check for forbidden patterns
        if "select *" in sql_lower:
            issues.append("SELECT * is not allowed")
        
        if " join " in sql_lower and "cross join" not in sql_lower:
            issues.append("Only CROSS JOIN is allowed")
        
        if " on " in sql_lower and "cross join" not in sql_lower:
            issues.append("JOIN conditions with ON clause are not allowed")
        
        if "to_char(" in sql_lower:
            issues.append("TO_CHAR() function is not supported")
        
        if "extract(" in sql_lower:
            issues.append("EXTRACT() function is not supported")
        
        # Check for MEASURE usage
        if "measure(" in sql_lower:
            # MEASURE should only be in WITH clauses
            with_clauses = re.findall(r'with\s+\w+\s+as\s*\([^)]*\)', sql_lower, re.DOTALL)
            if not with_clauses:
                issues.append("MEASURE() used outside WITH clause")
        
        return issues
    
    def iterative_fix_sql(self, original_question: str, failed_sql: str, 
                         error_message: str, similar_sqls: List[str] = None) -> Dict[str, Any]:
        """
        Perform iterative SQL fixing with multiple attempts.
        This is the main entry point for the exception agent.
        """
        logger.info(f" STARTING ITERATIVE SQL FIX PROCESS")
        logger.info(f"Max iterations: {self.max_iterations}")
        
        attempts = []
        current_sql = failed_sql
        current_error = error_message
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f" ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"Analyzing error: {current_error[:100]}...")
            
            # Analyze and fix the current error
            fix_result = self.analyze_and_fix_sql_error(
                original_question=original_question,
                failed_sql=current_sql,
                error_message=current_error,
                similar_sqls=similar_sqls,
                previous_attempts=attempts
            )
            
            # Record this attempt
            attempt_record = {
                "iteration": iteration,
                "input_sql": current_sql,
                "input_error": current_error,
                "fix_result": fix_result,
                "timestamp": datetime.now().isoformat()
            }
            attempts.append(attempt_record)
            
            if fix_result["success"]:
                corrected_sql = fix_result["corrected_sql"]
                
                logger.info(f" ITERATION {iteration} - SQL CORRECTION SUCCESSFUL")
                logger.info(f"Corrected SQL: {corrected_sql}")
                logger.info(f"Fix Type: {fix_result.get('fix_type', 'unknown')}")
                logger.info(f"Confidence: {fix_result.get('confidence_score', 0):.2f}")
                
                return {
                    "success": True,
                    "final_sql": corrected_sql,
                    "total_iterations": iteration,
                    "fix_summary": {
                        "primary_error_category": fix_result["error_analysis"]["primary_category"],
                        "root_cause": fix_result["root_cause"]["root_cause"],
                        "fix_type": fix_result.get("fix_type", "unknown"),
                        "confidence": fix_result.get("confidence_score", 0),
                        "learning_points": fix_result.get("learning_points", [])
                    },
                    "all_attempts": attempts,
                    "final_attempt": fix_result
                }
            
            else:
                logger.warning(f"ITERATION {iteration} - FIX FAILED")
                logger.warning(f"Reason: {fix_result.get('correction_explanation', 'Unknown')}")
                
                if iteration < self.max_iterations:
                    logger.info(f"Preparing for next iteration...")
                    # For next iteration, we would need to simulate running the corrected SQL
                    # and getting a new error. For now, we'll break if we can't fix it.
                    break
        
        # All iterations failed
        logger.error(f" ALL {self.max_iterations} ITERATIONS FAILED")
        
        return {
            "success": False,
            "error": "Could not fix SQL after maximum iterations",
            "total_iterations": len(attempts),
            "failure_analysis": {
                "primary_issues": [attempt["fix_result"]["error_analysis"]["primary_category"] 
                                 for attempt in attempts],
                "attempted_fixes": [attempt["fix_result"].get("fix_type", "unknown") 
                                  for attempt in attempts],
                "final_error": attempts[-1]["fix_result"] if attempts else {}
            },
            "all_attempts": attempts,
            "recommendation": "Manual review required - automated fixing failed"
        }
    
    def process(self, state: BaseAgentState) -> DBAgentState:
        """
        Process method for BaseAgent interface.
        Expects state to contain error information for fixing.
        """
        db_state = DBAgentState(**state)
        
        try:
            # Extract error context from state
            original_question = state.get("original_query", state["query"])
            failed_sql = state.get("failed_sql", "")
            error_message = state.get("error_message", "Unknown error")
            similar_sqls = state.get("retrieved_sql_context", [])
            
            if not failed_sql:
                db_state["error_message"] = "No failed SQL provided for exception analysis"
                db_state["status"] = "failed"
                return db_state
            
            # Perform iterative fixing
            fix_result = self.iterative_fix_sql(
                original_question=original_question,
                failed_sql=failed_sql,
                error_message=error_message,
                similar_sqls=similar_sqls
            )
            
            if fix_result["success"]:
                db_state["sql_query"] = fix_result["final_sql"]
                db_state["query_type"] = "SELECT"  
                db_state["status"] = "completed"
                db_state["success_message"] = f"SQL fixed successfully in {fix_result['total_iterations']} iterations"
                db_state["result"] = fix_result
                
            else:
                db_state["error_message"] = fix_result["error"]
                db_state["status"] = "failed"
                db_state["result"] = fix_result
            
        except Exception as e:
            logger.error(f"SQLExceptionAgent process failed: {e}")
            db_state["error_message"] = f"Exception agent error: {str(e)}"
            db_state["status"] = "failed"
        
        return db_state
    
    def get_exception_capabilities(self) -> Dict[str, Any]:
        """Return information about exception handling capabilities."""
        return {
            "max_iterations": self.max_iterations,
            "supported_error_categories": list(self.error_patterns.keys()),
            "fix_success_rate_estimate": 0.75,  # Estimated success rate
            "supported_platforms": ["cube.js"],
            "learning_enabled": True,
            "error_pattern_count": len(self.error_patterns)
        }