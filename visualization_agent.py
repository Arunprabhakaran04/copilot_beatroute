"""
Visualization Agent
==================

This agent generates visual representations (charts, graphs, plots) of SQL query results.
It takes a pandas DataFrame from the DB Agent and creates HTML-based visualizations using Plotly.
"""

import logging
import pandas as pd
import json
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from base_agent import BaseAgent, BaseAgentState
from token_tracker import track_llm_call, get_token_tracker

logger = logging.getLogger(__name__)

class VisualizationAgentState(BaseAgentState):
    """Extended state for Visualization Agent"""
    question: str
    dataframe: pd.DataFrame
    visualization_html: str
    visualization_type: str
    chart_config: Dict[str, Any]

class VisualizationAgent(BaseAgent):
    """
    Visualization Agent that generates interactive charts and plots from query results.
    Creates HTML-based visualizations using Plotly for web display.
    """
    
    def __init__(self, llm, model_name: str = "gpt-4o"):
        super().__init__(llm)
        self.model_name = model_name
        logger.info(f"VisualizationAgent initialized with model: {model_name}")
    
    def get_agent_type(self) -> str:
        return "visualization"
    
    def process(self, state: BaseAgentState) -> BaseAgentState:
        """
        Process the state and generate a visualization.
        
        Expected state to contain:
        - query: The original question asked
        - result: Dictionary containing 'query_data' with pandas DataFrame or list of dicts
        """
        try:
            logger.info(f"VisualizationAgent processing query: {state['query']}")
            
            # Extract data from state
            question = state.get('query', '')
            result_data = state.get('result', {})
            
            # Get DataFrame from various possible sources
            df = self._extract_dataframe_from_result(result_data, question)
            
            if df is None or df.empty:
                return self._handle_no_data_case(state)
            
            # Generate visualization
            visualization_result = self.generate_visualization(question, df)
            
            # Update state with visualization results
            state['status'] = 'completed'
            state['success_message'] = 'Visualization generated successfully'
            
            # Store visualization in result
            if 'result' not in state:
                state['result'] = {}
            
            state['result']['visualization'] = {
                'html': visualization_result['html'],
                'type': visualization_result['chart_type'],
                'config': visualization_result.get('config', {}),
                'question': question,
                'row_count': len(df),
                'columns': list(df.columns),
                'agent_type': 'visualization'
            }
            
            logger.info(f"Visualization generated successfully for {len(df)} rows - Chart type: {visualization_result['chart_type']}")
            return state
            
        except Exception as e:
            logger.error(f"VisualizationAgent error: {e}")
            state['status'] = 'failed'
            state['error_message'] = f"Visualization generation error: {str(e)}"
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
            # Import pandas here to avoid global import issues
            import pandas as pd
            
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
        """Handle case when no data is available for visualization"""
        logger.warning("No data available for visualization")
        
        no_data_html = """
        <div style="text-align: center; padding: 20px; border: 2px dashed #ccc; margin: 10px;">
            <h3>📊 No Data Available for Visualization</h3>
            <p>The query executed successfully but returned no results to visualize.</p>
            <p>💡 <strong>Suggestion:</strong> Check if the search criteria match existing data.</p>
        </div>
        """
        
        state['status'] = 'completed'
        state['success_message'] = 'Visualization handled for empty result set'
        
        if 'result' not in state:
            state['result'] = {}
        
        state['result']['visualization'] = {
            'html': no_data_html.strip(),
            'type': 'no_data_chart',
            'config': {},
            'question': state.get('query', ''),
            'row_count': 0,
            'columns': [],
            'agent_type': 'visualization'
        }
        
        return state
    
    def generate_visualization(self, question: str, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate a visualization of the results of a SQL query.
        
        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.
            **kwargs: Additional parameters including _cost for token tracking.
            
        Returns:
            Dict: Contains HTML visualization, chart type, and configuration.
        """
        try:
            logger.info(f"Generating visualization for question: {question}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            
            # Analyze data to determine best chart type
            chart_analysis = self._analyze_data_for_chart_type(df, question)
            
            # Create visualization prompt
            visualization_prompt = ChatPromptTemplate.from_template("""
            You are an expert data visualization assistant. Create an interactive HTML visualization using Plotly.js for the given data and question.
            
            Question: {question}
            Data Analysis: {analysis}
            DataFrame Info:
            - Shape: {shape}
            - Columns: {columns}
            - Data Types: {dtypes}
            - Sample Data: {sample}
            
            Requirements:
            1. Generate COMPLETE HTML with embedded Plotly.js CDN
            2. Choose the most appropriate chart type: {suggested_chart}
            3. Make it interactive and responsive
            4. Include proper titles, labels, and formatting
            5. Use professional color schemes
            6. Handle data type conversions properly
            7. Add hover information and tooltips
            
            Chart Type Guidelines:
            - Bar Chart: For categorical comparisons, top N analysis
            - Line Chart: For time series, trends over time
            - Pie Chart: For parts of whole (max 10 categories)
            - Scatter Plot: For correlation analysis
            - Histogram: For distribution analysis
            - Box Plot: For statistical summaries
            
            Data Processing Notes:
            - Convert date strings to proper date format if needed
            - Handle numeric vs categorical data appropriately
            - Limit pie charts to top 10 categories for readability
            - Sort data meaningfully (e.g., by value for top N queries)
            
            Output a complete HTML page with:
            - Plotly.js CDN link
            - Proper div container
            - Complete JavaScript plot configuration
            - Responsive design
            - Professional styling
            
            Start with: <!DOCTYPE html>
            """)
            
            # Format data information
            sample_data = df.head(3).to_dict('records') if len(df) > 0 else []
            dtypes_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            messages = visualization_prompt.format_messages(
                question=question,
                analysis=chart_analysis['analysis'],
                shape=df.shape,
                columns=list(df.columns),
                dtypes=dtypes_info,
                sample=sample_data,
                suggested_chart=chart_analysis['suggested_chart']
            )
            
            # Generate visualization
            response = self.llm.invoke(messages)
            html_content = response.content.strip()
            
            # Track token usage
            track_llm_call(
                input_prompt=messages,
                output=html_content,
                agent_type="visualization",
                operation="generate_visualization",
                model_name=self.model_name
            )
            
            # Extract chart type from analysis
            chart_type = chart_analysis['suggested_chart']
            
            # Create chart configuration for metadata
            chart_config = {
                'data_shape': df.shape,
                'columns': list(df.columns),
                'chart_type': chart_type,
                'data_types': dtypes_info,
                'analysis': chart_analysis['analysis']
            }
            
            logger.info("Visualization generated successfully")
            logger.info(f"Chart type: {chart_type}")
            logger.info(f"HTML length: {len(html_content)} characters")
            
            return {
                'html': html_content,
                'chart_type': chart_type,
                'config': chart_config,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            # Return a basic error visualization
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Visualization Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; padding: 20px; }}
                    .error-container {{ 
                        border: 2px solid #ff6b6b; 
                        border-radius: 8px; 
                        padding: 20px; 
                        background-color: #ffe0e0; 
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <div class="error-container">
                    <h2>📊 Visualization Generation Failed</h2>
                    <p><strong>Question:</strong> {question}</p>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p><strong>Data Points:</strong> {len(df)} rows available</p>
                    <p>💡 Please try again or check the data format</p>
                </div>
            </body>
            </html>
            """
            
            return {
                'html': error_html,
                'chart_type': 'error_chart',
                'config': {'error': str(e)},
                'success': False
            }
    
    def _analyze_data_for_chart_type(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """
        Analyze the DataFrame and question to suggest the best chart type.
        
        Returns:
            Dict with 'suggested_chart' and 'analysis' keys
        """
        try:
            question_lower = question.lower()
            num_rows, num_cols = df.shape
            
            # Detect numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Analyze question for visualization hints
            chart_keywords = {
                'bar': ['top', 'bottom', 'compare', 'comparison', 'versus', 'vs', 'ranking', 'rank'],
                'line': ['trend', 'over time', 'time series', 'growth', 'change', 'progression'],
                'pie': ['distribution', 'breakdown', 'share', 'proportion', 'percentage'],
                'scatter': ['correlation', 'relationship', 'vs', 'against', 'compared to'],
                'histogram': ['distribution', 'frequency', 'range', 'spread']
            }
            
            suggested_charts = []
            analysis_points = []
            
            # Question-based analysis
            for chart_type, keywords in chart_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    suggested_charts.append(chart_type)
                    analysis_points.append(f"Question suggests {chart_type} chart (keywords: {keywords})")
            
            # Data structure analysis
            if len(datetime_cols) > 0:
                suggested_charts.append('line')
                analysis_points.append(f"Time-based data detected: {datetime_cols}")
            
            if len(numeric_cols) >= 2:
                suggested_charts.append('scatter')
                analysis_points.append(f"Multiple numeric columns for correlation: {numeric_cols}")
            
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                suggested_charts.append('bar')
                analysis_points.append(f"Categorical vs numeric data good for bar charts")
            
            if len(categorical_cols) == 1 and len(numeric_cols) == 1:
                unique_categories = df[categorical_cols[0]].nunique()
                if unique_categories <= 10:
                    suggested_charts.append('pie')
                    analysis_points.append(f"Few categories ({unique_categories}) suitable for pie chart")
                else:
                    suggested_charts.append('bar')
                    analysis_points.append(f"Many categories ({unique_categories}) better as bar chart")
            
            # Top N queries
            if any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst']):
                suggested_charts.append('bar')
                analysis_points.append("Top/bottom ranking query suggests bar chart")
            
            # Default fallback logic
            if not suggested_charts:
                if len(numeric_cols) >= 1:
                    suggested_charts.append('bar')
                    analysis_points.append("Default: Bar chart for numeric data visualization")
                else:
                    suggested_charts.append('bar')
                    analysis_points.append("Default: Bar chart as general purpose visualization")
            
            # Select the most appropriate chart
            chart_priority = ['line', 'bar', 'pie', 'scatter', 'histogram']
            final_chart = next((chart for chart in chart_priority if chart in suggested_charts), suggested_charts[0])
            
            analysis = {
                'suggested_chart': final_chart,
                'analysis': '; '.join(analysis_points),
                'data_summary': f"{num_rows} rows, {num_cols} columns",
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'datetime_columns': datetime_cols,
                'all_suggestions': suggested_charts
            }
            
            logger.info(f"Chart analysis completed: {final_chart} - {analysis['analysis']}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in chart type analysis: {e}")
            return {
                'suggested_chart': 'bar',
                'analysis': f'Analysis failed ({str(e)}), defaulting to bar chart',
                'data_summary': f"{df.shape[0]} rows, {df.shape[1]} columns",
                'numeric_columns': [],
                'categorical_columns': [],
                'datetime_columns': [],
                'all_suggestions': ['bar']
            }
    
    def generate_visualization_from_db_result(self, question: str, db_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method to generate visualization directly from DB agent result.
        
        Args:
            question: The original question
            db_result: Result dictionary from DB agent
            
        Returns:
            Dictionary containing visualization and metadata
        """
        try:
            # Extract DataFrame from DB result
            df = self._extract_dataframe_from_result(db_result, question)
            
            if df is None or df.empty:
                return {
                    'success': False,
                    'error': 'No data available for visualization',
                    'visualization': self._handle_no_data_case({'query': question, 'result': {}})['result']['visualization']
                }
            
            # Generate visualization
            viz_result = self.generate_visualization(question, df)
            
            if viz_result['success']:
                return {
                    'success': True,
                    'visualization': {
                        'html': viz_result['html'],
                        'type': viz_result['chart_type'],
                        'config': viz_result['config'],
                        'question': question,
                        'row_count': len(df),
                        'columns': list(df.columns),
                        'agent_type': 'visualization'
                    },
                    'metadata': {
                        'dataframe_shape': df.shape,
                        'columns': list(df.columns),
                        'chart_type': viz_result['chart_type'],
                        'data_types': df.dtypes.to_dict() if hasattr(df.dtypes, 'to_dict') else {}
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'Visualization generation failed',
                    'visualization': viz_result
                }
            
        except Exception as e:
            logger.error(f"Error in generate_visualization_from_db_result: {e}")
            return {
                'success': False,
                'error': str(e),
                'visualization': {
                    'html': f"<div>Error generating visualization: {str(e)}</div>",
                    'type': 'error_chart',
                    'config': {'error': str(e)},
                    'question': question,
                    'row_count': 0,
                    'columns': [],
                    'agent_type': 'visualization'
                }
            }
    
    def get_visualization_capabilities(self) -> Dict[str, Any]:
        """Return information about visualization capabilities"""
        return {
            "supports_html_output": True,
            "visualization_library": "plotly.js",
            "supported_chart_types": [
                "bar_chart", "line_chart", "pie_chart", "scatter_plot", 
                "histogram", "box_plot", "area_chart", "bubble_chart"
            ],
            "supported_data_formats": [
                "pandas_dataframe", "list_of_dictionaries", 
                "db_agent_result", "query_results_structure"
            ],
            "output_format": "interactive_html_with_plotly",
            "features": [
                "automatic_chart_type_selection",
                "responsive_design", "interactive_tooltips",
                "professional_styling", "data_type_handling",
                "top_n_analysis", "time_series_support",
                "token_usage_tracking", "error_handling"
            ],
            "model": self.model_name,
            "max_data_points": 50000  # Reasonable limit for browser performance
        }