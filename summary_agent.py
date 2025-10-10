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
        """
        try:
            logger.info(f"SummaryAgent processing query: {state['query']}")
            
            question = state.get('query', '')
            result_data = state.get('result', {})
            
            df = self._extract_dataframe_from_result(result_data, question)
            
            if df is None or df.empty:
                return self._handle_no_data_case(state)
            
            summary_html = self.generate_summary(question, df)
            
            state['status'] = 'completed'
            state['success_message'] = 'Summary generated successfully'
            
            if 'result' not in state:
                state['result'] = {}
            
            state['result']['summary'] = {
                'html': summary_html,
                'type': 'data_summary',
                'question': question,
                'row_count': len(df),
                'columns': list(df.columns),
                'agent_type': 'summary'
            }
            
            logger.info(f"Summary generated successfully for {len(df)} rows")
            return state
            
        except Exception as e:
            logger.error(f"SummaryAgent error: {e}")
            state['status'] = 'failed'
            state['error_message'] = f"Summary generation error: {str(e)}"
            return state
    
    def _extract_dataframe_from_result(self, result_data: Dict[str, Any], question: str) -> pd.DataFrame:
        """
        Extract pandas DataFrame from various result formats.
        
        Handles multiple data formats from DB Agent:
        - Direct DataFrame in 'query_data'
        - List of dictionaries in 'data'
        - Nested result structures
        """
        try:
            # Case 1: Direct DataFrame
            if 'query_data' in result_data and isinstance(result_data['query_data'], pd.DataFrame):
                logger.info("Found DataFrame in 'query_data'")
                return result_data['query_data']
            
            # Case 2: DataFrame in nested structure
            if isinstance(result_data.get('query_data'), pd.DataFrame):
                logger.info("Found DataFrame in nested 'query_data'")
                return result_data['query_data']
            
            # Case 3: List of dictionaries in 'data'
            if 'data' in result_data and isinstance(result_data['data'], list):
                logger.info("Converting list of dicts to DataFrame")
                return pd.DataFrame(result_data['data'])
            
            # Case 4: Query results structure from DB connection
            if 'query_results' in result_data:
                query_results = result_data['query_results']
                if 'data' in query_results and isinstance(query_results['data'], list):
                    logger.info("Converting query_results data to DataFrame")
                    return pd.DataFrame(query_results['data'])
            
            # Case 5: Direct list of dictionaries
            if isinstance(result_data, list) and len(result_data) > 0:
                logger.info("Converting direct list to DataFrame")
                return pd.DataFrame(result_data)
            
            # Case 6: Check for any list-like data in nested structures
            for key, value in result_data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    logger.info(f"Found data list in key '{key}', converting to DataFrame")
                    return pd.DataFrame(value)
            
            logger.warning(f"Could not extract DataFrame from result_data. Keys: {list(result_data.keys())}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting DataFrame: {e}")
            return None
    
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
        
        state['result']['summary'] = {
            'html': no_data_html.strip(),
            'type': 'no_data_summary',
            'question': state.get('query', ''),
            'row_count': 0,
            'columns': [],
            'agent_type': 'summary'
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
            
            # Get response language configuration (defaulting to English)
            response_language = self._get_response_language()
            
            system_message_content = (
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\n"
                f"The following is a pandas DataFrame with the results of the query:\n"
                f"{df.to_markdown()}\n\n"
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
            
            logger.info("Summary generated successfully")
            logger.info(f"Summary length: {len(summary_html)} characters")
            
            return summary_html
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Return a basic error summary
            return f"""
            <ul>
            <li>❌ <strong>Error:</strong> Failed to generate summary</li>
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