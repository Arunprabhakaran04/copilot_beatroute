import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from base_agent import BaseAgent, BaseAgentState, DBAgentState

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
            similar_sqls: List of similar SQL queries for context
            previous_results: Results from previous steps (for multi-step queries)
            
        Returns:
            Dict containing the generated SQL and metadata
        """
        try:
            schema_info = self.get_schema_info()
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Process similar SQLs
            sql_examples_text = ""
            if similar_sqls:
                sql_examples_text = "\n".join([f"Example {i+1}:\n{sql}" for i, sql in enumerate(similar_sqls)])
            else:
                sql_examples_text = "No SQL examples available."
            
            # Process previous results if available
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
                            Your task is to generate a SINGLE, well-formed SQL query to answer the specific question provided.
                            
                            IMPORTANT: Generate SQL for THIS SPECIFIC QUESTION ONLY. Do not try to answer multi-step queries.
                            If the question seems to require multiple steps, focus on the immediate question asked.
                            
                            STRICTLY follow the rules below and generate SQL in the format:
                            {{ "sql": "SQL_QUERY_HERE" }}
                            
                            Failure to follow these rules will result in an invalid response.
                            
                            === Additional System Assumptions ===
                            1. Today's date is {current_date}
                            2. Assume that the tables provided are **always updated up to and including the current date**.
                            3. You can safely generate queries for time frames such as **this month**, **this quarter**, **this year**, or **future-looking filters**, assuming the data is available up to the current date.
                            4. Do NOT ask the user to confirm whether data for a specific timeframe exists; **assume data availability unless explicitly stated otherwise**.
                            
                            ===Response Guidelines 
                            1. If the provided context is sufficient, generate a valid SQL query without explanations. Generate STRICTLY in format {{\"sql\" : \"SQL_QUERY_HERE\" }}.
                            2. If the provided context is insufficient, explain why it can't be generated in format {{\"error\" : \"ERROR_EXPLANATION_HERE\" }}.
                            3. If you need clarification, ask a follow-up question in format {{\"follow_up\" : \"FOLLOW_UP_QUESTION_HERE\" }}.
                            4. Always assume data is up-to-date unless explicitly stated otherwise.
                            5. Use the most relevant table(s) based on the schema.
                            6. Keep queries simple and focused on the specific question asked.
                            
                            ===SQL Generation Guidelines (Cube.js API Compatible)===
                            - STRICTLY ONLY USE CROSS JOIN for joining two tables 
                            - STRICTLY DO NOT USE ANY JOIN CONDITION EITHER WITH ON CLAUSE OR IN WHERE CLAUSE
                            - ONLY Use **CROSS JOIN** , DO NOT USE INNER JOIN or LEFT JOIN or RIGHT JOIN when joining tables.
                            - DO NOT CREATE TWO TABLES USING WITH CLAUSE AND JOIN THEM. 
                            - IF THERE IS NO DIRECT CROSS JOIN IN THE EXAMPLE BETWEEN TWO TABLES USE UNION ALL INSTEAD WITH PROPER CAST OF TYPE. 
                            - STRICTLY DO NOT USE FUNCTIONS NOT SUPPORTED IN CUBE JS SQL API:
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
                            
                            Generate a single SQL query to answer this specific question only."""
            
            message_log = [self._create_system_message(sql_generation_prompt)]
            message_log.append(self._create_user_message(f"Generate SQL for: {question}"))
            
            # Add examples from similar SQLs as conversation context
            if similar_sqls:
                for i, sql in enumerate(similar_sqls[:3]):  # Limit to top 3 for context
                    if isinstance(sql, dict) and "question" in sql and "sql" in sql:
                        message_log.append(self._create_user_message(sql["question"]))
                        message_log.append(self._create_assistant_message(json.dumps({"sql": sql["sql"]})))
            
            # Generate response
            response = self.llm.invoke(message_log)
            content = response.content.strip()
            
            # Parse JSON response
            try:
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
    
    def process(self, state: BaseAgentState) -> DBAgentState:
        """
        Process method required by BaseAgent interface.
        Generates a single SQL query for the given question.
        """
        db_state = DBAgentState(**state)
        
        try:
            # Extract context from state
            similar_sqls = state.get("retrieved_sql_context", [])
            previous_results = state.get("previous_step_results", None)
            
            # Generate SQL
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
                
                logger.info(f"🔧 SQLGeneratorAgent - SQL Generated Successfully:")
                logger.info(f"Question: {state['query']}")
                logger.info(f"Generated SQL: {result['sql']}")
                logger.info(f"Query Type: {result['query_type']}")
                logger.info(f"Used Examples: {result.get('used_examples', False)}")
                logger.info(f"Used Previous Results: {result.get('used_previous_results', False)}")
                
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