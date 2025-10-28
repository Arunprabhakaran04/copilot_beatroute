"""
Summary Agent
============

This agent generates structured HTML summaries of SQL query results.
It takes a pandas DataFrame from the DB Agent and creates user-friendly summaries.
"""

import logging
import pandas as pd
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from base_agent import BaseAgent, BaseAgentState
from token_tracker import track_llm_call, get_token_tracker

logger = logging.getLogger(__name__)

class SummaryAgentState(BaseAgentState):
    """Extended state for Summary Agent"""
    question: str
    dataframe: pd.DataFrame
    summary_html: str
    summary_type: str

class SummaryAgent(BaseAgent):
    """
    Summary Agent that generates structured HTML summaries of query results.
    Designed to work with output from DB Agent to provide user-friendly data summaries.
    """
    
    def __init__(self, llm, model_name: str = "gpt-4o"):
        super().__init__(llm)
        self.model_name = model_name
        logger.info(f"SummaryAgent initialized with model: {model_name}")
    
    def get_agent_type(self) -> str:
        return "summary"
    
    def process(self, state: BaseAgentState) -> BaseAgentState:
        """
        Process the state and generate a summary.
        
        Expected state to contain:
        - query: The original question asked
        - result: Dictionary containing 'query_data' with pandas DataFrame or list of dicts
        - intermediate_results: Results from previous workflow steps (checked first)
        
        If query references previous data (e.g., "summarize that"), will attempt to 
        retrieve from intermediate_results first, then fall back to cached database results.
        """
        try:
            logger.info(f"SummaryAgent processing query: {state['query']}")
            
            question = state.get('query', '')
            result_data = state.get('result', {})
            
            # Check if query references previous data
            if self.should_use_cached_data(question):
                logger.info("Query references previous data - checking sources...")
                
                # PRIORITY 1: Check intermediate_results from current workflow
                intermediate_results = state.get('intermediate_results', {})
                if intermediate_results:
                    logger.info(f"📊 Found {len(intermediate_results)} step(s) in intermediate_results")
                    intermediate_data = self._extract_data_from_intermediate_results(intermediate_results)
                    
                    if intermediate_data:
                        logger.info("✅ Using data from intermediate_results (current workflow)")
                        result_data = {**result_data, **intermediate_data}
                        state['result'] = result_data
                    else:
                        logger.info("No query data found in intermediate_results, checking cache...")
                        # PRIORITY 2: Fall back to Redis cache
                        cached_result = self.get_cached_db_result(state)
                        if cached_result:
                            logger.info("Using cached database result for summary")
                            result_data = {**result_data, **cached_result}
                            state['result'] = result_data
                        else:
                            logger.warning("No data found in intermediate_results or cache")
                else:
                    logger.info("No intermediate_results, checking cache...")
                    # PRIORITY 2: Fall back to Redis cache
                    cached_result = self.get_cached_db_result(state)
                    if cached_result:
                        logger.info("Using cached database result for summary")
                        result_data = {**result_data, **cached_result}
                        state['result'] = result_data
                    else:
                        logger.warning("Query references previous data but no cached result found")
            
            df = self._extract_dataframe_from_result(result_data, question)
            
            if df is None or df.empty:
                return self._handle_no_data_case(state)
            
            summary_html = self.generate_summary(question, df)
            
            state['status'] = 'completed'
            state['success_message'] = 'Summary generated successfully'
            
            if 'result' not in state:
                state['result'] = {}
            
            state['result']['summary'] = summary_html
            
            state['result']['summary_metadata'] = {
                'question': question,
                'row_count': len(df),
                'columns': list(df.columns)
            }
            
            logger.info(f"Summary generated successfully for {len(df)} rows")
            return state
            
        except Exception as e:
            logger.error(f"SummaryAgent error: {e}")
            state['status'] = 'failed'
            state['error_message'] = f"Summary generation error: {str(e)}"
            return state
    
    def _extract_data_from_intermediate_results(self, intermediate_results: dict) -> dict:
        """
        Extract query data from intermediate_results of previous workflow steps.
        
        Args:
            intermediate_results: Dict mapping step keys to their results
            
        Returns:
            Dict containing query_data if found, empty dict otherwise
        """
        try:
            for step_key, step_data in intermediate_results.items():
                if not isinstance(step_data, dict):
                    continue
                
                # Look for database query results
                if step_data.get("agent_type") in ["db_query", "sql"] or "query" in step_key.lower():
                    # Check for query_data field
                    if "query_data" in step_data:
                        logger.info(f"📊 Found query_data in intermediate step: {step_key}")
                        return {"query_data": step_data["query_data"]}
                    
                    # Check for query_results field
                    if "query_results" in step_data:
                        logger.info(f"📊 Found query_results in intermediate step: {step_key}")
                        return {"query_data": step_data["query_results"]}
                    
                    # Check in nested result dict
                    if "result" in step_data and isinstance(step_data["result"], dict):
                        result_dict = step_data["result"]
                        if "query_data" in result_dict:
                            logger.info(f"📊 Found query_data in intermediate step result: {step_key}")
                            return {"query_data": result_dict["query_data"]}
                        if "query_results" in result_dict:
                            logger.info(f"📊 Found query_results in intermediate step result: {step_key}")
                            return {"query_data": result_dict["query_results"]}
            
            logger.info("No query data found in intermediate_results")
            return {}
            
        except Exception as e:
            logger.error(f"Error extracting data from intermediate_results: {str(e)}")
            return {}
    
    def _extract_dataframe_from_result(self, result_data: Dict[str, Any], question: str) -> pd.DataFrame:
        """
        Extract pandas DataFrame from various result formats.
        
        Handles multiple data formats from DB Agent:
        - Direct DataFrame in 'query_data'
        - Serialized DataFrame from Redis cache (with _type: "dataframe")
        - List of dictionaries in 'data'
        - Nested result structures
        """
        try:
            logger.info(f"Extracting DataFrame from result_data. Keys: {list(result_data.keys())}")
            
            # Case 1: Check all common keys for DataFrame or serialized DataFrame
            for key in ['query_data', 'query_results', 'data', 'rows_returned']:
                if key in result_data:
                    value = result_data[key]
                    
                    # Direct DataFrame
                    if isinstance(value, pd.DataFrame):
                        logger.info(f"Found DataFrame in '{key}'")
                        return value
                    
                    # Serialized DataFrame from cache
                    if isinstance(value, dict) and value.get("_type") == "dataframe":
                        logger.info(f"Found serialized DataFrame in '{key}' - deserializing...")
                        return self._deserialize_dataframe(value)
                    
                    # List of dicts
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        logger.info(f"Converting list in '{key}' to DataFrame")
                        return pd.DataFrame(value)
            
            # Case 2: Query results structure from DB connection
            if 'query_results' in result_data:
                query_results = result_data['query_results']
                if isinstance(query_results, dict):
                    if 'data' in query_results and isinstance(query_results['data'], list):
                        logger.info("Converting query_results data to DataFrame")
                        return pd.DataFrame(query_results['data'])
                    if 'query_data' in query_results and isinstance(query_results['query_data'], list):
                        logger.info("Converting query_results query_data to DataFrame")
                        return pd.DataFrame(query_results['query_data'])
            
            # Case 6: Direct list of dictionaries
            if isinstance(result_data, list) and len(result_data) > 0:
                logger.info("Converting direct list to DataFrame")
                return pd.DataFrame(result_data)
            
            # Case 7: Check for any list-like data in nested structures
            for key, value in result_data.items():
                if isinstance(value, list) and len(value) > 0:
                    # Check if it's a list of dictionaries (tabular data)
                    if isinstance(value[0], dict):
                        logger.info(f"Found data list in key '{key}', converting to DataFrame")
                        return pd.DataFrame(value)
                    # Check if it's a list of strings that might be formatted data
                    elif isinstance(value[0], str) and '|' in value[0]:
                        logger.info(f"Found formatted string data in key '{key}', attempting to parse")
                        # Try to parse pipe-separated data
                        lines = [line.strip() for line in value if line.strip() and '|' in line]
                        if len(lines) > 1:
                            # First line might be headers
                            headers = [col.strip() for col in lines[0].split('|')]
                            data_rows = []
                            for line in lines[1:]:
                                if '---' not in line:  # Skip separator lines
                                    row_data = [col.strip() for col in line.split('|')]
                                    if len(row_data) == len(headers):
                                        data_rows.append(dict(zip(headers, row_data)))
                            if data_rows:
                                logger.info(f"Parsed {len(data_rows)} rows from formatted string")
                                return pd.DataFrame(data_rows)
            
            logger.warning(f"Could not extract DataFrame from result_data. Keys: {list(result_data.keys())}")
            logger.warning(f"Data types: {[(k, type(v)) for k, v in result_data.items()]}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting DataFrame: {e}")
            return None
    
    def _deserialize_dataframe(self, serialized_df: Dict[str, Any]) -> pd.DataFrame:
        """
        Deserialize a DataFrame that was serialized by redis_memory_manager._serialize_result()
        
        Args:
            serialized_df: Dict with {"_type": "dataframe", "data": [...], "columns": [...]}
            
        Returns:
            pandas DataFrame
        """
        try:
            columns = serialized_df.get("columns", [])
            data = serialized_df.get("data", [])
            
            if not data:
                logger.warning("Serialized DataFrame has no data")
                return pd.DataFrame()
            
            # Convert list of dicts back to DataFrame
            df = pd.DataFrame(data, columns=columns)
            logger.info(f"Deserialized DataFrame: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error deserializing DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def _handle_no_data_case(self, state: BaseAgentState) -> BaseAgentState:
        """Handle case when no data is available for summarization"""
        logger.warning("No data available for summarization")
        
        no_data_html = """
        <ul>
        <li> <strong>Query Status:</strong> No data returned</li>
        <li> <strong>Reason:</strong> The query executed successfully but returned no results</li>
        <li> <strong>Suggestion:</strong> Check if the search criteria match existing data</li>
        </ul>
        """
        
        state['status'] = 'completed'
        state['success_message'] = 'Summary generated for empty result set'
        
        if 'result' not in state:
            state['result'] = {}
        
        state['result']['summary'] = no_data_html.strip()
        
        state['result']['summary_metadata'] = {
            'question': state.get('query', ''),
            'row_count': 0,
            'columns': []
        }
        
        return state
    
    def generate_summary(self, question: str, df: pd.DataFrame) -> str:
        """
        Generate a summary of the results of a SQL query.
        
        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.
            
        Returns:
            str: The HTML summary of the results.
        """
        try:
            logger.info(f"Generating summary for question: {question}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            logger.info(f"DataFrame head:\n{df.head()}")
            
            # Get response language configuration (defaulting to English)
            response_language = self._get_response_language()
            
            # Convert DataFrame to markdown safely
            try:
                df_markdown = df.to_markdown(index=False)
                logger.info(f"DataFrame markdown conversion successful, length: {len(df_markdown)}")
            except Exception as md_error:
                logger.error(f"Error converting DataFrame to markdown: {md_error}")
                # Fallback to string representation
                df_markdown = df.to_string(index=False)
                logger.info(f"Using string representation instead, length: {len(df_markdown)}")
            
            # Escape curly braces to prevent template interpretation issues
            df_markdown_escaped = df_markdown.replace('{', '{{').replace('}', '}}')
            question_escaped = question.replace('{', '{{').replace('}', '}}')
            
            system_message_content = (
                f"You are a helpful data assistant. The user asked the question: '{question_escaped}'\n\n"
                f"The following is a pandas DataFrame with the results of the query:\n"
                f"{df_markdown_escaped}\n\n"
            )
            
            # Create user message with detailed instructions
            user_message_content = f"""
            📊 Briefly summarize the data accurately based on the question that was asked, summarize the data clearly and accurately in a structured, point-wise format.
            ✅ Include the question on the top.
            ✅ Include the timeframe if mentioned in the question.
            📌 Use bold formatting for key metrics or highlights.
            🆔 Do not round off or alter ID fields — treat them as integer or string and retain their exact values.
            💡 Avoid associating numbers with currency unless explicitly specified.
            🔢 Display all numeric values without scientific notation.
            📈 If the question involves analysis, provide a multi-faceted analysis with noteworthy insights.
            🧠 Highlight any trends, anomalies, or significant observations in a concise and readable format.
            🚫 Do not mention or refer to table names.
            🔇 Do not include any extra explanation outside the summary itself.

            - Use <ul> and <li> tags to format the summary as bullet points.
            - Use <strong> to bold important values like totals, IDs, or key metrics.

            ⚠️ Output must be a valid HTML string. Example:
            
            <ul>
            <li>📅 <strong>Timeframe:</strong> Jan to June 2024</li>
            <li>✅ <strong>Total Orders:</strong> 3,245</li>
            <li>📈 <strong>Top Month:</strong> March (812 orders)</li>
            <li>🧾 <strong>Top Customer ID:</strong> 10231</li>
            </ul>
            {response_language}
            """
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message_content),
                ("human", user_message_content)
            ])
            
            # Format and invoke the LLM
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            summary_html = response.content.strip()
            
            # Track token usage
            track_llm_call(
                input_prompt=messages,
                output=summary_html,
                agent_type="summary",
                operation="generate_summary",
                model_name=self.model_name
            )
            
            # Clean up code fences if LLM added them
            # Remove ```html and ``` markers
            if summary_html.startswith("```html"):
                summary_html = summary_html.replace("```html", "", 1).strip()
            if summary_html.startswith("```"):
                summary_html = summary_html.replace("```", "", 1).strip()
            if summary_html.endswith("```"):
                summary_html = summary_html.rsplit("```", 1)[0].strip()
            
            logger.info("Summary generated successfully")
            logger.info(f"Summary length: {len(summary_html)} characters")
            
            return summary_html
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            logger.error(f"Full exception details:", exc_info=True)
            # Return a basic error summary
            return f"""
            <ul>
            <li>❌ <strong>Error:</strong> Failed to generate summary - {str(e)}</li>
            <li>🔍 <strong>Question:</strong> {question}</li>
            <li>📊 <strong>Data Points:</strong> {len(df)} rows available</li>
            <li>💡 <strong>Status:</strong> Please try again or check the data format</li>
            </ul>
            """
    
    def _get_response_language(self) -> str:
        """
        Get response language configuration.
        Can be extended to support multiple languages.
        """
        return ""  # Default to English, no additional language instructions
    
    def generate_summary_from_db_result(self, question: str, db_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method to generate summary directly from DB agent result.
        
        Args:
            question: The original question
            db_result: Result dictionary from DB agent
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Extract DataFrame from DB result
            df = self._extract_dataframe_from_result(db_result, question)
            
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': 'No data available for summarization',
                    'summary': self._handle_no_data_case({'query': question, 'result': {}})['result']['summary']
                }
            
            # Generate summary
            summary_html = self.generate_summary(question, df)
            
            return {
                'success': True,
                'summary': {
                    'html': summary_html,
                    'type': 'data_summary',
                    'question': question,
                    'row_count': len(df),
                    'columns': list(df.columns),
                    'agent_type': 'summary'
                },
                'metadata': {
                    'dataframe_shape': df.shape,
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict() if hasattr(df.dtypes, 'to_dict') else {}
                }
            }
            
        except Exception as e:
            logger.error(f"Error in generate_summary_from_db_result: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': {
                    'html': f"<ul><li> <strong>Error:</strong> {str(e)}</li></ul>",
                    'type': 'error_summary',
                    'question': question,
                    'row_count': 0,
                    'columns': [],
                    'agent_type': 'summary'
                }
            }
    
    def get_summary_capabilities(self) -> Dict[str, Any]:
        """Return information about summary capabilities"""
        return {
            "supports_html_output": True,
            "supported_data_formats": [
                "pandas_dataframe",
                "list_of_dictionaries", 
                "db_agent_result",
                "query_results_structure"
            ],
            "output_format": "structured_html_with_emojis",
            "features": [
                "automatic_data_type_handling",
                "timeframe_extraction",
                "key_metrics_highlighting",
                "trend_analysis",
                "no_table_name_references",
                "emoji_formatting",
                "token_usage_tracking"
            ],
            "model": self.model_name,
            "max_data_points": 10000  
        }