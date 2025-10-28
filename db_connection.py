"""
Database Configuration:
- Host: Loaded from .env (DB_HOST)
- Database: Loaded from .env (DB_DATABASE)
- User: Loaded from .env (DB_USER)
- Port: Loaded from .env (DB_PORT)
- Password: Uses session_id from WebSocket connection
"""

import logging
import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, date, time
import decimal
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Manages database connections and query execution for Cube.js PostgreSQL backend"""
    
    def __init__(self, session_id: str = None):
        """
        Initialize database connection manager
        
        Args:
            session_id: Session ID from WebSocket connection, used as database password
                       If not provided, uses a default password from environment (for backward compatibility)
        """
        self.db_config = {
            'host': os.getenv('DB_HOST', ''),
            'database': os.getenv('DB_DATABASE', ''),
            'user': os.getenv('DB_USER', ''),
            'password': session_id if session_id else os.getenv('DB_PASSWORD', ''),
            'port': int(os.getenv('DB_PORT', ''))
        }
        self.session_id = session_id
        self.connection_pool = None
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        """Initialize connection pool for efficient database connections"""
        try:
            logger.info("Initializing database connection pool...")
            logger.info(f"Connecting to: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
            
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                **self.db_config
            )
            
            # Test connection
            test_result = self.test_connection()
            if test_result['success']:
                logger.info(" Database connection pool initialized successfully")
                logger.info(f"Database version: {test_result['db_version']}")
            else:
                logger.error(f" Database connection test failed: {test_result['error']}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database connection pool: {e}")
            self.connection_pool = None
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection and return connection info"""
        if not self.connection_pool:
            return {
                'success': False,
                'error': 'Connection pool not initialized'
            }
        
        connection = None
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor()
            
            # Test basic query
            cursor.execute("SELECT version()")
            db_version = cursor.fetchone()[0]
            
            # Test current database info
            cursor.execute("SELECT current_database(), current_user")
            db_info = cursor.fetchone()
            
            cursor.close()
            
            return {
                'success': True,
                'db_version': db_version,
                'database': db_info[0],
                'user': db_info[1],
                'host': self.db_config['host'],
                'port': self.db_config['port']
            }
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    def execute_query(self, sql_query: str, params: tuple = None) -> Dict[str, Any]:
        """
        Execute a SQL query and return results with metadata
        
        Args:
            sql_query: The SQL query to execute
            params: Optional parameters for parameterized queries
            
        Returns:
            Dict containing success status, data, metadata, and any errors
        """
        if not self.connection_pool:
            return {
                'success': False,
                'error': 'Database connection pool not available',
                'data': [],
                'metadata': {}
            }
        
        connection = None
        start_time = datetime.now()
        
        try:
            # Use clean logging
            try:
                from clean_logging import SQLLogger
                SQLLogger.execution_start(sql_query)
            except ImportError:
                logger.info("Executing SQL query...")
                logger.debug(f"SQL: {sql_query}")
            
            # Get connection from pool
            connection = self.connection_pool.getconn()
            cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Execute query
            cursor.execute(sql_query, params)
            
            # Handle different query types
            if sql_query.strip().upper().startswith(('SELECT', 'WITH')):
                # SELECT query - fetch results
                rows = cursor.fetchall()
                
                # Get column information from cursor
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Create pandas DataFrame from results with column names
                if rows:
                    # Convert rows to list of dicts for DataFrame
                    data_for_df = []
                    for row in rows:
                        row_dict = {}
                        for key, value in dict(row).items():
                            # Handle special data types for JSON serialization
                            if isinstance(value, (datetime, date, time)):
                                row_dict[key] = value.isoformat()
                            elif isinstance(value, decimal.Decimal):
                                row_dict[key] = float(value)
                            else:
                                row_dict[key] = value
                        data_for_df.append(row_dict)
                    
                    # Create DataFrame with explicit column names
                    df = pd.DataFrame(data_for_df, columns=columns)
                    
                    # Convert DataFrame to JSON string with orient="records"
                    # This ensures frontend gets proper array of objects with column names
                    data = df.to_json(orient="records")
                    
                    logger.info(f"📊 DATAFRAME CREATED: {df.shape[0]} rows x {df.shape[1]} columns")
                    logger.info(f"   Column names: {list(df.columns)}")
                    logger.info(f"   First row: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
                else:
                    data = "[]"  # Empty JSON array string
                
                row_count = len(rows)
                
            else:
                # INSERT, UPDATE, DELETE queries
                row_count = cursor.rowcount
                data = "[]"  # Empty JSON array string
                columns = []
            
            # Commit transaction
            connection.commit()
            cursor.close()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f" SQL QUERY EXECUTED SUCCESSFULLY")
            logger.info(f"Rows affected/returned: {row_count}")
            logger.info(f"Execution time: {execution_time:.3f}s")
            logger.info(f"Columns: {columns}")
            
            return {
                'success': True,
                'data': data,
                'metadata': {
                    'row_count': row_count,
                    'columns': columns,
                    'execution_time': execution_time,
                    'query_type': sql_query.strip().split()[0].upper(),
                    'executed_at': start_time.isoformat(),
                    'database': self.db_config['database'],
                    'host': self.db_config['host']
                },
                'error': None
            }
            
        except psycopg2.Error as e:
            # PostgreSQL specific errors
            if connection:
                connection.rollback()
            
            error_msg = f"Database error: {e}"
            logger.error(f" SQL EXECUTION FAILED: {error_msg}")
            logger.error(f"Failed SQL: {sql_query}")
            
            return {
                'success': False,
                'error': error_msg,
                'data': "[]",  # Empty JSON array string
                'metadata': {
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'error_code': e.pgcode if hasattr(e, 'pgcode') else None,
                    'error_type': 'database_error',
                    'failed_sql': sql_query
                }
            }
            
        except Exception as e:
            if connection:
                connection.rollback()
                
            error_msg = f"Query execution error: {e}"
            logger.error(f" SQL EXECUTION FAILED: {error_msg}")
            logger.error(f"Failed SQL: {sql_query}")
            
            return {
                'success': False,
                'error': error_msg,
                'data': "[]",  # Empty JSON array string
                'metadata': {
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'error_type': 'execution_error',
                    'failed_sql': sql_query
                }
            }
            
        finally:
            # Return connection to pool
            if connection:
                self.connection_pool.putconn(connection)
    
    def execute_query_with_results_formatting(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute query and format results for agent consumption
        
        Returns formatted results with data as JSON string from df.to_json(orient="records")
        """
        result = self.execute_query(sql_query)
        
        if result['success']:
            # data is already a JSON string from df.to_json(orient="records")
            data_json_string = result['data']
            
            # Format for agent consumption
            formatted_result = {
                'success': True,
                'query_executed': sql_query,
                'results': {
                    'data': data_json_string,  # JSON string that frontend can parse
                    'summary': {
                        'total_rows': result['metadata']['row_count'],
                        'columns': result['metadata']['columns'],
                        'execution_time': f"{result['metadata']['execution_time']:.3f}s",
                        'query_type': result['metadata']['query_type']
                    }
                },
                'formatted_output': self._format_results_for_display(data_json_string, result['metadata']['columns']),
                'json_results': data_json_string,  # Same as results.data
                'execution_metadata': result['metadata']
            }
            
            logger.info(f" QUERY RESULTS FORMATTED FOR AGENT CONSUMPTION")
            logger.info(f"Total rows: {formatted_result['results']['summary']['total_rows']}")
            logger.info(f"Data format: JSON string from df.to_json(orient='records')")
            
            return formatted_result
        else:
            return {
                'success': False,
                'error': result['error'],
                'query_executed': sql_query,
                'results': None,
                'execution_metadata': result['metadata']
            }
    
    def _format_results_for_display(self, data: str, columns: List[str]) -> str:
        """
        Format query results as a readable table string
        
        Args:
            data: JSON string from df.to_json(orient="records")
            columns: List of column names
        """
        # Parse JSON string to list of dicts
        try:
            data_list = json.loads(data) if isinstance(data, str) else data
        except:
            data_list = []
        
        if not data_list:
            return "No results returned."
        
        # Create formatted table
        formatted_lines = []
        
        # Header
        header = " | ".join(columns)
        formatted_lines.append(header)
        formatted_lines.append("-" * len(header))
        
        # Data rows (limit to first 10 for readability)
        display_data = data_list[:10]
        for row in display_data:
            row_values = []
            for col in columns:
                value = row.get(col, 'NULL')
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 50:
                    str_value = str_value[:47] + "..."
                row_values.append(str_value)
            formatted_lines.append(" | ".join(row_values))
        
        # Add summary if more rows exist
        if len(data_list) > 10:
            formatted_lines.append(f"... and {len(data_list) - 10} more rows")
        
        return "\n".join(formatted_lines)
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a specific table"""
        schema_query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = %s
        ORDER BY ordinal_position
        """
        
        result = self.execute_query(schema_query, (table_name,))
        
        if result['success']:
            return {
                'success': True,
                'table_name': table_name,
                'schema': result['data']
            }
        else:
            return {
                'success': False,
                'error': result['error']
            }
    
    def get_available_tables(self) -> Dict[str, Any]:
        """Get list of available tables in the database"""
        tables_query = """
        SELECT 
            table_name,
            table_type
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        
        result = self.execute_query(tables_query)
        
        if result['success']:
            return {
                'success': True,
                'tables': result['data'],
                'table_count': result['metadata']['row_count']
            }
        else:
            return {
                'success': False,
                'error': result['error']
            }
    
    def close_connections(self):
        """Close all database connections"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Database connection pool closed")

# Global database instance per session
_db_instances = {}

def get_database_connection(session_id: str = None) -> DatabaseConnection:
    """
    Get database connection instance for a specific session
    
    Args:
        session_id: Session ID from WebSocket connection, used as database password
        
    Returns:
        DatabaseConnection instance
    """
    # If no session_id provided, use default instance for backward compatibility
    if session_id is None:
        session_id = "default"
    
    # Create new instance if not exists for this session
    if session_id not in _db_instances:
        _db_instances[session_id] = DatabaseConnection(session_id=session_id)
    
    return _db_instances[session_id]

# Convenience functions for direct use
def execute_sql(sql_query: str, session_id: str = None) -> Dict[str, Any]:
    """
    Execute SQL query using the database connection
    
    Args:
        sql_query: SQL query to execute
        session_id: Session ID for database password (optional for backward compatibility)
    """
    db = get_database_connection(session_id)
    return db.execute_query_with_results_formatting(sql_query)

def test_database_connection(session_id: str = None) -> Dict[str, Any]:
    """Test the database connection"""
    db = get_database_connection(session_id)
    return db.test_connection()