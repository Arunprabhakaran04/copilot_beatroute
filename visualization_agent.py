import logging
import pandas as pd
import json
import base64
import os
from datetime import datetime
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from base_agent import BaseAgent, BaseAgentState
from token_tracker import track_llm_call, get_token_tracker
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

logger = logging.getLogger(__name__)

class VisualizationAgentState(BaseAgentState):
    """Extended state for Visualization Agent"""
    question: str
    dataframe: pd.DataFrame
    image_path: str
    image_base64: str
    visualization_type: str
    chart_config: Dict[str, Any]

class VisualizationAgent(BaseAgent):
    """
    Visualization Agent that generates interactive charts and plots from query results.
    Creates static image files stored in the visualizations directory for frontend integration.
    """
    
    def __init__(self, llm, model_name: str = "gpt-4o"):
        super().__init__(llm)
        self.model_name = model_name
        self.visualizations_dir = os.path.join(os.path.dirname(__file__), "visualizations")
        
        # Ensure visualizations directory exists
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        logger.info(f"VisualizationAgent initialized with model: {model_name}")
        logger.info(f"Visualizations directory: {self.visualizations_dir}")
    
    def get_agent_type(self) -> str:
        return "visualization"
    
    def process(self, state: BaseAgentState) -> BaseAgentState:
        """
        Process the state and generate a visualization image.
        
        Expected state to contain:
        - query: The original question asked
        - result: Dictionary containing 'query_data' with pandas DataFrame or list of dicts
        - intermediate_results: Results from previous workflow steps (checked first)
        
        If query references previous data (e.g., "visualize that"), will attempt to 
        retrieve from intermediate_results first, then fall back to cached database results.
        """
        try:
            logger.info(f"VisualizationAgent processing query: {state['query']}")
            
            # Extract data from state
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
                            logger.info("Using cached database result for visualization")
                            result_data = {**result_data, **cached_result}
                            state['result'] = result_data
                        else:
                            logger.warning("No data found in intermediate_results or cache")
                else:
                    logger.info("No intermediate_results, checking cache...")
                    # PRIORITY 2: Fall back to Redis cache
                    cached_result = self.get_cached_db_result(state)
                    if cached_result:
                        logger.info("Using cached database result for visualization")
                        result_data = {**result_data, **cached_result}
                        state['result'] = result_data
                    else:
                        logger.warning("Query references previous data but no cached result found")
            
            # Get DataFrame from various possible sources
            df = self._extract_dataframe_from_result(result_data, question)
            
            if df is None or df.empty:
                return self._handle_no_data_case(state)
            
            # Generate visualization image
            visualization_result = self.generate_visualization_image(question, df)
            
            # Update state with visualization results
            state['status'] = 'completed'
            state['success_message'] = 'Visualization image generated successfully'
            
            # Store visualization in result
            if 'result' not in state:
                state['result'] = {}
            
            base64_with_prefix = f"data:image/png;base64,{visualization_result['image_base64']}"
            
            state['result']['visualization'] = base64_with_prefix
            
            state['result']['visualization_metadata'] = {
                'chart_type': visualization_result['chart_type'],
                'question': question,
                'row_count': len(df),
                'columns': list(df.columns)
            }
            
            logger.info(f"Visualization image generated successfully for {len(df)} rows - Chart type: {visualization_result['chart_type']}")
            logger.info(f"Image saved at: {visualization_result['image_path']}")
            return state
            
        except Exception as e:
            logger.error(f"VisualizationAgent error: {e}")
            state['status'] = 'failed'
            state['error_message'] = f"Visualization generation error: {str(e)}"
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
            import pandas as pd
            
            # Case 1: Direct DataFrame in various keys
            for key in ['query_data', 'query_results', 'data', 'rows_returned']:
                if key in result_data:
                    value = result_data[key]
                    
                    # Check if it's a direct DataFrame
                    if isinstance(value, pd.DataFrame):
                        logger.info(f"Found DataFrame in '{key}'")
                        return value
                    
                    # Check if it's a serialized DataFrame from cache
                    if isinstance(value, dict) and value.get("_type") == "dataframe":
                        logger.info(f"Found serialized DataFrame in '{key}' - deserializing...")
                        return self._deserialize_dataframe(value)
            
            # Case 2: List of dictionaries in 'data'
            if 'data' in result_data and isinstance(result_data['data'], list):
                logger.info("Converting list of dicts to DataFrame")
                return pd.DataFrame(result_data['data'])
            
            # Case 3: Query results structure from DB connection
            if 'query_results' in result_data:
                query_results = result_data['query_results']
                if isinstance(query_results, dict):
                    if 'data' in query_results and isinstance(query_results['data'], list):
                        logger.info("Converting query_results data to DataFrame")
                        return pd.DataFrame(query_results['data'])
            
            # Case 4: Direct list of dictionaries
            if isinstance(result_data, list) and len(result_data) > 0:
                logger.info("Converting direct list to DataFrame")
                return pd.DataFrame(result_data)
            
            # Case 5: Check for any list-like data in nested structures
            for key, value in result_data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    logger.info(f"Found data list in key '{key}', converting to DataFrame")
                    return pd.DataFrame(value)
            
            logger.warning(f"Could not extract DataFrame from result_data. Keys: {list(result_data.keys())}")
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
        """Handle case when no data is available for visualization"""
        logger.warning("No data available for visualization")
        
        # Create a simple "no data" image
        no_data_image_path, no_data_base64 = self._create_no_data_image()
        
        state['status'] = 'completed'
        state['success_message'] = 'Visualization handled for empty result set'
        
        if 'result' not in state:
            state['result'] = {}
        
        base64_with_prefix = f"data:image/png;base64,{no_data_base64}"
        
        state['result']['visualization'] = base64_with_prefix
        
        state['result']['visualization_metadata'] = {
            'chart_type': 'no_data_chart',
            'question': state.get('query', ''),
            'row_count': 0,
            'columns': []
        }
        
        return state
    
    def _create_no_data_image(self) -> tuple[str, str]:
        """Create a simple 'no data' visualization image"""
        try:
            # Create a simple figure with text
            fig = go.Figure()
            
            fig.add_annotation(
                text="📊 No Data Available<br>for Visualization",
                x=0.5, y=0.7,
                xref="paper", yref="paper",
                font=dict(size=24, color="#666666"),
                showarrow=False,
                align="center"
            )
            
            fig.add_annotation(
                text="The query executed successfully but<br>returned no results to visualize.",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                font=dict(size=14, color="#888888"),
                showarrow=False,
                align="center"
            )
            
            fig.add_annotation(
                text="💡 Suggestion: Check if the search criteria match existing data.",
                x=0.5, y=0.3,
                xref="paper", yref="paper",
                font=dict(size=12, color="#aaaaaa"),
                showarrow=False,
                align="center"
            )
            
            fig.update_layout(
                width=800,
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"no_data_{timestamp}.png"
            image_path = os.path.join(self.visualizations_dir, filename)
            
            # Save image
            fig.write_image(image_path, format="png")
            
            # Convert to base64
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            logger.info(f"No data image created: {image_path}")
            return image_path, image_base64
            
        except Exception as e:
            logger.error(f"Error creating no data image: {e}")
            # Return empty values if image creation fails
            return "", ""
    
    def generate_visualization_image(self, question: str, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate a visualization image from the results of a SQL query.
        
        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.
            **kwargs: Additional parameters including _cost for token tracking.
            
        Returns:
            Dict: Contains image path, base64 data, chart type, and configuration.
        """
        try:
            logger.info(f"Generating visualization image for question: {question}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {list(df.columns)}")
            
            # Preprocess data for better visualization
            processed_df = self._preprocess_data_for_visualization(df, question)
            logger.info(f"Processed DataFrame shape: {processed_df.shape}")
            
            # Analyze data to determine best chart type
            chart_analysis = self._analyze_data_for_chart_type(processed_df, question)
            chart_type = chart_analysis['suggested_chart']
            
            # Generate the plotly figure based on chart type
            fig = self._create_plotly_figure(processed_df, question, chart_type, chart_analysis)
            
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{chart_type}_{timestamp}.png"
            image_path = os.path.join(self.visualizations_dir, filename)
            
            # Save image with high quality
            fig.write_image(
                image_path, 
                format="png",
                width=1200,
                height=800,
                scale=2  # Higher resolution
            )
            
            # Convert to base64
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create chart configuration for metadata
            chart_config = {
                'data_shape': df.shape,
                'columns': list(df.columns),
                'chart_type': chart_type,
                'data_types': {col: str(dtype) for col, dtype in processed_df.dtypes.items()},
                'analysis': chart_analysis['analysis'],
                'image_path': image_path,
                'timestamp': timestamp
            }
            
            # Track token usage (for consistency with original implementation)
            track_llm_call(
                input_prompt=f"Visualization request: {question}",
                output=f"Generated {chart_type} chart with {len(processed_df)} data points",
                agent_type="visualization",
                operation="generate_visualization_image",
                model_name=self.model_name
            )
            
            logger.info("Visualization image generated successfully")
            logger.info(f"Chart type: {chart_type}")
            logger.info(f"Image saved at: {image_path}")
            
            return {
                'image_path': image_path,
                'image_base64': image_base64,
                'chart_type': chart_type,
                'config': chart_config,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization image: {e}")
            # Return error information
            return {
                'image_path': '',
                'image_base64': '',
                'chart_type': 'error_chart',
                'config': {'error': str(e)},
                'success': False
            }
    
    def _create_plotly_figure(self, df: pd.DataFrame, question: str, chart_type: str, chart_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create a plotly figure based on the chart type and data.
        
        Args:
            df: Preprocessed DataFrame
            question: Original question
            chart_type: Determined chart type
            chart_analysis: Analysis results from _analyze_data_for_chart_type
            
        Returns:
            Plotly Figure object
        """
        try:
            logger.info(f"Creating {chart_type} chart for {len(df)} data points")
            
            # Get column information
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Create figure based on chart type
            if chart_type == 'line':
                fig = self._create_line_chart(df, question, numeric_cols, categorical_cols, datetime_cols)
            elif chart_type == 'bar':
                fig = self._create_bar_chart(df, question, numeric_cols, categorical_cols)
            elif chart_type == 'pie':
                fig = self._create_pie_chart(df, question, numeric_cols, categorical_cols)
            elif chart_type == 'scatter':
                fig = self._create_scatter_chart(df, question, numeric_cols, categorical_cols)
            elif chart_type == 'histogram':
                fig = self._create_histogram_chart(df, question, numeric_cols)
            else:
                # Default to bar chart
                fig = self._create_bar_chart(df, question, numeric_cols, categorical_cols)
            
            # Apply common styling
            fig = self._apply_professional_styling(fig, question)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating plotly figure: {e}")
            # Return a simple error chart
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization:<br>{str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                font=dict(size=16, color="red"),
                showarrow=False,
                align="center"
            )
            fig.update_layout(
                title="Visualization Error",
                width=800,
                height=600
            )
            return fig
    
    def _create_line_chart(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list, datetime_cols: list) -> go.Figure:
        """Create a line chart"""
        fig = go.Figure()
        
        # Determine x and y axes
        if datetime_cols:
            x_col = datetime_cols[0]
            y_col = numeric_cols[0] if numeric_cols else categorical_cols[0]
        elif len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
        else:
            # Single column - create index-based line chart
            x_col = 'index'
            y_col = df.columns[0]
            df = df.reset_index()
        
        # Create line trace
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines+markers',
            name=y_col,
            line=dict(shape='spline', width=3, color='#1f77b4'),
            marker=dict(size=8, color='#1f77b4'),
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Line Chart: {question}",
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> go.Figure:
        """Create a bar chart"""
        fig = go.Figure()
        
        # Determine x and y axes
        if categorical_cols and numeric_cols:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
        elif len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
        else:
            # Single column - create value counts
            x_col = df.columns[0]
            if df[x_col].dtype in ['object', 'category']:
                value_counts = df[x_col].value_counts()
                df = pd.DataFrame({'category': value_counts.index, 'count': value_counts.values})
                x_col, y_col = 'category', 'count'
            else:
                y_col = x_col
                x_col = 'index'
                df = df.reset_index()
        
        # Create bar trace
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y_col],
            name=y_col,
            marker=dict(
                color=df[y_col] if y_col in df.columns else '#1f77b4',
                colorscale='viridis',
                showscale=False
            ),
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Bar Chart: {question}",
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def _create_pie_chart(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> go.Figure:
        """Create a pie chart"""
        fig = go.Figure()
        
        # Determine labels and values
        if categorical_cols and numeric_cols:
            labels_col = categorical_cols[0]
            values_col = numeric_cols[0]
        elif len(df.columns) >= 2:
            labels_col = df.columns[0]
            values_col = df.columns[1]
        else:
            # Single column - create value counts
            col = df.columns[0]
            value_counts = df[col].value_counts()
            labels_col, values_col = 'category', 'count'
            df = pd.DataFrame({'category': value_counts.index, 'count': value_counts.values})
        
        # Create pie trace
        fig.add_trace(go.Pie(
            labels=df[labels_col],
            values=df[values_col],
            hole=0.3,  # Donut chart
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Pie Chart: {question}"
        )
        
        return fig
    
    def _create_scatter_chart(self, df: pd.DataFrame, question: str, numeric_cols: list, categorical_cols: list) -> go.Figure:
        """Create a scatter plot"""
        fig = go.Figure()
        
        # Need at least 2 numeric columns for scatter
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
        elif len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
        else:
            # Fallback to index vs single column
            x_col = 'index'
            y_col = df.columns[0]
            df = df.reset_index()
        
        # Create scatter trace
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(
                size=10,
                color=df[y_col] if y_col in numeric_cols else '#1f77b4',
                colorscale='viridis',
                showscale=True if y_col in numeric_cols else False
            ),
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br><b>{y_col}</b>: %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Scatter Plot: {question}",
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def _create_histogram_chart(self, df: pd.DataFrame, question: str, numeric_cols: list) -> go.Figure:
        """Create a histogram"""
        fig = go.Figure()
        
        # Use first numeric column or first column
        col = numeric_cols[0] if numeric_cols else df.columns[0]
        
        fig.add_trace(go.Histogram(
            x=df[col],
            nbinsx=20,
            marker=dict(color='#1f77b4', opacity=0.7),
            hovertemplate=f'<b>{col}</b>: %{{x}}<br><b>Count</b>: %{{y}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Histogram: {question}",
            xaxis_title=col,
            yaxis_title="Frequency"
        )
        
        return fig
    
    def _apply_professional_styling(self, fig: go.Figure, question: str) -> go.Figure:
        """Apply professional styling to the figure"""
        fig.update_layout(
            # Size and spacing
            width=1200,
            height=800,
            margin=dict(l=80, r=80, t=100, b=80),
            
            # Colors and background
            plot_bgcolor='white',
            paper_bgcolor='white',
            
            # Font styling
            font=dict(
                family="Inter, Roboto, Arial, sans-serif",
                size=12,
                color="#333333"
            ),
            
            # Title styling
            title=dict(
                font=dict(size=18, color="#1a1a1a"),
                x=0.5,
                xanchor='center'
            ),
            
            # Grid styling
            xaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.3)',
                showgrid=True,
                zeroline=False,
                automargin=True
            ),
            yaxis=dict(
                gridcolor='rgba(200, 200, 200, 0.3)',
                showgrid=True,
                zeroline=False,
                automargin=True
            ),
            
            # Legend styling
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            
            # Hover styling
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Inter, Roboto, Arial, sans-serif"
            )
        )
        
        return fig
    
    def _preprocess_data_for_visualization(self, df: pd.DataFrame, question: str) -> pd.DataFrame:
        """
        Preprocess data to optimize for visualization.
        Handles large datasets, sorting, and data type conversion.
        """
        try:
            processed_df = df.copy()
            question_lower = question.lower()
            
            # Handle large datasets (> 20 rows for categorical data, > 50 for time series)
            max_rows_categorical = 20
            max_rows_timeseries = 50
            
            is_time_series = any(word in question_lower for word in ['trend', 'time', 'series', 'over'])
            max_rows = max_rows_timeseries if is_time_series else max_rows_categorical
            
            if len(processed_df) > max_rows:
                logger.info(f"Large dataset detected ({len(processed_df)} rows). Applying smart filtering (max: {max_rows}).")
                
                # For sales/numerical data, show top performers
                numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    # Find the most relevant numeric column (usually the largest values)
                    main_numeric_col = None
                    for col in numeric_cols:
                        if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'value', 'amount', 'total', 'quantity']):
                            main_numeric_col = col
                            break
                    
                    if not main_numeric_col:
                        main_numeric_col = numeric_cols[0]
                    
                    # Sort by the main numeric column and take top N
                    processed_df = processed_df.sort_values(by=main_numeric_col, ascending=False).head(max_rows)
                    logger.info(f"Filtered to top {max_rows} by {main_numeric_col}")
                
                else:
                    # If no numeric columns, just take first N rows
                    processed_df = processed_df.head(max_rows)
                    logger.info(f"No numeric columns found, taking first {max_rows} rows")
            
            # Handle time series data - ensure proper sorting
            datetime_cols = processed_df.select_dtypes(include=['datetime64']).columns.tolist()
            if datetime_cols and ('trend' in question_lower or 'time' in question_lower or 'over' in question_lower):
                sort_col = datetime_cols[0]
                processed_df = processed_df.sort_values(by=sort_col)
                logger.info(f"Sorted by datetime column: {sort_col}")
            
            # Clean data - remove null/None values where possible
            numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols:
                # Replace None with 0 for visualization
                processed_df[col] = processed_df[col].fillna(0)
            
            # Convert string numbers to numeric if possible
            for col in processed_df.columns:
                if processed_df[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        converted = pd.to_numeric(processed_df[col], errors='coerce')
                        if not converted.isna().all():
                            processed_df[col] = converted.fillna(0)
                    except:
                        pass
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return df
    
    def _analyze_data_for_chart_type(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """
        Analyze the DataFrame and question to suggest the best chart type.
        
        Returns:
            Dict with 'suggested_chart' and 'analysis' keys
        """
        try:
            question_lower = question.lower()
            num_rows, num_cols = df.shape
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            chart_keywords = {
                'bar': ['top', 'bottom', 'compare', 'comparison', 'versus', 'vs', 'ranking', 'rank'],
                'line': ['trend', 'over time', 'time series', 'growth', 'change', 'progression'],
                'pie': ['distribution', 'breakdown', 'share', 'proportion', 'percentage'],
                'scatter': ['correlation', 'relationship', 'vs', 'against', 'compared to'],
                'histogram': ['distribution', 'frequency', 'range', 'spread']
            }
            
            suggested_charts = []
            analysis_points = []
            
            for chart_type, keywords in chart_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    suggested_charts.append(chart_type)
                    analysis_points.append(f"Question suggests {chart_type} chart (keywords: {keywords})")
            
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
            
            if any(word in question_lower for word in ['top', 'bottom', 'highest', 'lowest', 'best', 'worst']):
                suggested_charts.append('bar')
                analysis_points.append("Top/bottom ranking query suggests bar chart")
            
            if not suggested_charts:
                if len(numeric_cols) >= 1:
                    suggested_charts.append('bar')
                    analysis_points.append("Default: Bar chart for numeric data visualization")
                else:
                    suggested_charts.append('bar')
                    analysis_points.append("Default: Bar chart as general purpose visualization")
            
            chart_priority = ['line', 'bar', 'pie', 'scatter', 'histogram']
            final_chart = next((chart for chart in chart_priority if chart in suggested_charts), suggested_charts[0])
            
            analysis = {
                'suggested_chart': final_chart,
                'analysis': '; '.join(analysis_points),
                'data_summary': f"{num_rows} rows, {num_cols} columns",
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'datetime_columns': datetime_cols,
                'all_suggestions': suggested_charts,
                'styling_hints': self._get_styling_hints(final_chart, question_lower, num_rows)
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
                'all_suggestions': ['bar'],
                'styling_hints': 'Use modern styling with clean layout'
            }
    
    def _get_styling_hints(self, chart_type: str, question_lower: str, num_rows: int) -> str:
        """Generate specific styling hints based on chart type and data characteristics"""
        hints = []
        
        if chart_type == 'line':
            hints.append("Use smooth spline curves for elegant line rendering")
            hints.append("Add gradient fill under the line for visual appeal")
            hints.append("Use a modern blue-to-teal gradient color scheme")
            if 'trend' in question_lower or 'time' in question_lower:
                hints.append("Add subtle animation for line drawing effect")
                hints.append("Use proper time axis formatting with readable date labels")
        
        elif chart_type == 'bar':
            hints.append("Use rounded corners on bars for modern appearance") 
            hints.append("Apply subtle shadow effects for depth")
            hints.append("Use a professional color gradient from light to dark")
        
        elif chart_type == 'pie':
            hints.append("Use distinct, harmonious colors for pie segments")
            hints.append("Add subtle 3D effect or shadows")
            hints.append("Show percentages and values in hover tooltips")
        
        # General styling hints
        if num_rows > 20:
            hints.append("Consider data aggregation for cleaner visualization")
        
        if 'sales' in question_lower or 'revenue' in question_lower:
            hints.append("Format currency values with appropriate units (K, M)")
            hints.append("Use business-appropriate color scheme (blues, grays)")
        
        hints.append("Ensure mobile responsiveness with flexible layout")
        hints.append("Add professional hover effects and smooth transitions")
        
        return "; ".join(hints)
    
    def generate_visualization_from_db_result(self, question: str, db_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method to generate visualization image directly from DB agent result.
        
        Args:
            question: The original question
            db_result: Result dictionary from DB agent
            
        Returns:
            Dictionary containing visualization image and metadata
        """
        try:
            # Extract DataFrame from DB result
            df = self._extract_dataframe_from_result(db_result, question)
            
            if df is None or df.empty:
                image_path, image_base64 = self._create_no_data_image()
                return {
                    'success': False,
                    'error': 'No data available for visualization',
                    'visualization': {
                        'image_path': image_path,
                        'image_base64': image_base64,
                        'type': 'no_data_chart',
                        'config': {},
                        'question': question,
                        'row_count': 0,
                        'columns': [],
                        'agent_type': 'visualization'
                    }
                }
            
            # Generate visualization image
            viz_result = self.generate_visualization_image(question, df)
            
            if viz_result['success']:
                return {
                    'success': True,
                    'visualization': {
                        'image_path': viz_result['image_path'],
                        'image_base64': viz_result['image_base64'],
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
                    'visualization': {
                        'image_path': '',
                        'image_base64': '',
                        'type': 'error_chart',
                        'config': {'error': 'Generation failed'},
                        'question': question,
                        'row_count': len(df),
                        'columns': list(df.columns),
                        'agent_type': 'visualization'
                    }
                }
            
        except Exception as e:
            logger.error(f"Error in generate_visualization_from_db_result: {e}")
            return {
                'success': False,
                'error': str(e),
                'visualization': {
                    'image_path': '',
                    'image_base64': '',
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
            "supports_image_output": True,
            "supports_base64_encoding": True,
            "visualization_library": "plotly",
            "supported_chart_types": [
                "bar_chart", "line_chart", "pie_chart", "scatter_plot", 
                "histogram", "no_data_chart", "error_chart"
            ],
            "supported_data_formats": [
                "pandas_dataframe", "list_of_dictionaries", 
                "db_agent_result", "query_results_structure"
            ],
            "output_format": "static_image_png_with_base64",
            "image_specifications": {
                "format": "PNG",
                "width": 1200,
                "height": 800,
                "scale": 2,
                "dpi": 300
            },
            "features": [
                "automatic_chart_type_selection",
                "professional_styling", "data_type_handling",
                "top_n_analysis", "time_series_support",
                "token_usage_tracking", "error_handling",
                "base64_encoding", "file_storage"
            ],
            "storage_directory": self.visualizations_dir,
            "model": self.model_name,
            "max_data_points": 50000  
        }