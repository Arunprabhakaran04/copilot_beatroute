"""
Schema Fetcher - Fetches complete database schema from CubeJS metadata and INFORMATION_SCHEMA

This module provides functionality to:
1. Decode Base64 encoded authentication tokens
2. Fetch CubeJS metadata for column descriptions
3. Query INFORMATION_SCHEMA for table/column information
4. Combine both sources into a formatted schema with descriptions
"""

import os
import base64
import requests
import numpy as np
import sqlglot
from typing import Dict, List, Tuple, Optional, Set
from dotenv import load_dotenv
from db_connection import execute_sql
from loguru import logger
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


def decode_base64_token(base64_token_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Decode a Base64 encoded token to extract auth token and user ID
    
    Token format expected: "auth_token-user_id"
    
    Args:
        base64_token_str: Base64 encoded token string
        
    Returns:
        Tuple of (auth_token, user_id) or (None, None) if decoding fails
    """
    try:
        if not base64_token_str:
            raise ValueError("The provided Base64 token is empty.")

        # Fix padding if needed
        padding = len(base64_token_str) % 4
        if padding != 0:
            base64_token_str += "=" * (4 - padding)

        # Decode base64
        decoded_token = base64.b64decode(base64_token_str).decode('utf-8')
        logger.info(f"Decoded token: {decoded_token}")

        # Extract the last hyphen-separated integer as user_id
        parts = decoded_token.rsplit('-', 1)  # Split only on the last '-'
        if len(parts) != 2:
            raise ValueError("Token format invalid (no hyphen separating user_id).")

        auth_token, user_id = parts[0], parts[1]

        # Validate user_id
        if not user_id.isdigit():
            raise ValueError(f"User ID '{user_id}' is not numeric.")

        logger.info(f"✅ Auth Token: {auth_token}")
        logger.info(f"✅ User ID: {user_id}")

        return auth_token, user_id

    except Exception as e:
        logger.error(f"❌ Error decoding Base64 token: {e}")
        return None, None


def get_final_schema_from_token(
    cubejs_api_url: str,
    auth_token: str,
    excluded_keywords: Optional[List[str]] = None,
    session_id: Optional[str] = None
) -> Tuple[List[str], Dict[str, str], Dict]:
    """
    Fetch complete database schema from CubeJS and INFORMATION_SCHEMA
    
    This function:
    1. Fetches CubeJS metadata for column descriptions (shortTitle)
    2. Queries INFORMATION_SCHEMA for table/column information
    3. Filters out excluded columns
    4. Combines both sources into formatted schema strings
    
    Args:
        cubejs_api_url: CubeJS API URL (e.g., "https://analytics.vwbeatroute.com/api/v1/meta")
        auth_token: Authentication token for CubeJS API
        excluded_keywords: List of keywords to exclude from column names (default: ["__", "monthYear", "year", "quarter"])
        
    Returns:
        Tuple containing:
            - table_schema_list: List of formatted schema strings (one per table)
            - table_to_schema_map: Dict mapping table name to its formatted schema string
            - cubejs_data: Raw CubeJS metadata response
    """
    if excluded_keywords is None:
        excluded_keywords = ["__", "monthYear", "year", "quarter"]
    
    logger.info(f"🔍 Fetching schema from CubeJS: {cubejs_api_url}")
    
    # Step 1: Fetch CubeJS metadata
    try:
        headers = {"Authorization": f"{auth_token}"}
        response = requests.get(cubejs_api_url, headers=headers, timeout=10)
        response.raise_for_status()
        cubejs_data = response.json()
        
        logger.info(f"✅ Fetched CubeJS metadata: {len(cubejs_data.get('cubes', []))} cubes")
        logger.debug(f"CubeJS Data: {cubejs_data.get('cubes', [])}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch CubeJS metadata: {e}")
        cubejs_data = {"cubes": []}
    
    # Step 2: Build short title map using (table, column) tuple as key
    short_title_map = {}
    for cube in cubejs_data.get("cubes", []):
        table = cube["name"]
        
        # Process dimensions
        for dim in cube.get("dimensions", []):
            col = dim["name"].split(".")[1] if "." in dim["name"] else dim["name"]
            short_title_map[(table, col)] = dim.get("shortTitle", "")
        
        # Process measures
        for measure in cube.get("measures", []):
            col = measure["name"].split(".")[1] if "." in measure["name"] else measure["name"]
            short_title_map[(table, col)] = measure.get("shortTitle", "")
    
    logger.info(f"📋 Built short title map: {len(short_title_map)} entries")
    
    # Step 3: Query INFORMATION_SCHEMA for all tables
    sql_query = """
    SELECT 
        table_name,
        column_name,
        data_type,
        ordinal_position
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE table_name IN 
        ('Order', 'OrderDetail','CustomerInvoice','CustomerInvoiceDetail',
        'DistributorOrderHeader', 'DistributorOrderDetail', 
        'DistributorSales', 'DistributorSalesDetail',
        'TeamPerformance','UserAttendance','UserDistanceLog',
        'UserSchedule',
        'Sku','Category','Brand',
        'SkuCustomField',
        'ViewCustomer','ViewUser','ViewDistributor','ViewCustomerActivity',
        'CustomerCustomField','UserCustomField','DistributorCustomField',
        'CustomerInventory', 'DistributorInventory',
        'Offtake', 'OfftakeDetail', 
        'OrderReturn', 'OrderReturnDetail',
        'CustomerDebitNote', 'CustomerCreditNote',
        'CustomerPayment', 'DistributorPayment',
        'CustomerOutstanding',
        'Scheme',
        'VmCampaign')
        OR table_name LIKE 'CampaignResponse%'
    ORDER BY table_name, ordinal_position
    """
    
    logger.info("📊 Querying INFORMATION_SCHEMA for table columns...")
    
    try:
        result = execute_sql(sql_query, session_id=session_id)
        
        if not result.get('success'):
            logger.error(f"❌ Failed to query INFORMATION_SCHEMA: {result.get('error')}")
            return [], {}, cubejs_data
        
        rows = result.get('results', {}).get('data', [])
        logger.info(f"✅ Retrieved {len(rows)} column definitions")
        
    except Exception as e:
        logger.error(f"❌ Error querying INFORMATION_SCHEMA: {e}")
        return [], {}, cubejs_data
    
    # Step 4: Filter excluded columns
    filtered_rows = []
    excluded_count = 0
    
    # Specific columns to exclude
    excluded_columns = [
        ('Order', 'available'),
        ('Sku', 'AvailableStatus'),
        ('ViewUser', 'availableStatus'),
        ('CustomerInvoiceDetail', 'skuName'),
        ('UserAttendance', 'duration'),
        ('DistributorInventory', 'amount')
    ]
    
    for row in rows:
        col = row['column_name']
        tbl = row['table_name']
        
        # Check if column should be excluded
        if any(kw in col for kw in excluded_keywords):
            excluded_count += 1
            continue
        
        if (tbl, col) in excluded_columns:
            excluded_count += 1
            continue
        
        filtered_rows.append(row)
    
    logger.info(f"🔧 Filtered out {excluded_count} excluded columns")
    logger.info(f"📝 Remaining columns: {len(filtered_rows)}")
    
    # Step 5: Format schema by table
    table_to_schema_map = {}
    table_schema_list = []
    
    # Group by table
    current_table = None
    prompt_lines = []
    
    for row in filtered_rows:
        table = row['table_name']
        column = row['column_name']
        dtype = row['data_type']
        
        # Start new table if needed
        if current_table != table:
            if current_table is not None:
                # Save previous table
                schema_str = "\n".join(prompt_lines)
                table_to_schema_map[current_table] = schema_str
                table_schema_list.append(schema_str)
            
            # Start new table
            current_table = table
            prompt_lines = [f"Table: {table}"]
        
        # Add column with short title if available
        short_title = short_title_map.get((table, column), "")
        if short_title:
            prompt_lines.append(f"- {column}:")
            prompt_lines.append(f"    Type: {dtype}")
            prompt_lines.append(f"    Description: {short_title}")
        else:
            prompt_lines.append(f"- {column}: {dtype}")
    
    # Save last table
    if current_table is not None:
        schema_str = "\n".join(prompt_lines)
        table_to_schema_map[current_table] = schema_str
        table_schema_list.append(schema_str)
    
    logger.info(f"✅ Successfully formatted schema for {len(table_to_schema_map)} tables")
    
    return table_schema_list, table_to_schema_map, cubejs_data


class SchemaManager:
    """
    Manages database schema with embeddings for semantic search
    
    This class:
    1. Fetches schema from CubeJS and INFORMATION_SCHEMA
    2. Generates embeddings for schema texts
    3. Stores schema data for easy access
    """
    
    def __init__(self):
        """Initialize the schema manager"""
        self.schema_map: Dict[str, str] = {}
        self.list_of_table_schemas: List[str] = []
        self.embeddings_schema: Optional[np.ndarray] = None
        self.cubejs_data: Dict = {}
        
        logger.info("SchemaManager initialized")
    
    def get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
        """
        Generate embeddings for a list of texts using OpenAI API
        
        Args:
            texts: List of text strings to embed
            model: OpenAI embedding model to use
            
        Returns:
            Numpy array of embeddings
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        
        try:
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(model=model, input=texts)
            embeddings = np.array([r.embedding for r in response.data])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def set_embeddings_schema(self, embeddings_schema: np.ndarray):
        """Set the schema embeddings"""
        self.embeddings_schema = embeddings_schema
        logger.info(f"✅ Set embeddings_schema: shape {embeddings_schema.shape}")
    
    def set_schema_map(self, schema_map: Dict[str, str]):
        """Set the schema map (table -> schema string)"""
        self.schema_map = schema_map
        logger.info(f"✅ Set schema_map: {len(schema_map)} tables")
    
    def set_list_of_table_schemas(self, list_of_table_schemas: List[str]):
        """Set the list of table schemas"""
        self.list_of_table_schemas = list_of_table_schemas
        logger.info(f"✅ Set list_of_table_schemas: {len(list_of_table_schemas)} schemas")
    
    def load_schema_from_token(
        self, 
        base64_token: str, 
        cubejs_api_url: Optional[str] = None,
        generate_embeddings: bool = True
    ) -> Dict[str, any]:
        """
        Load complete schema from base64 token and optionally generate embeddings
        
        Args:
            base64_token: Base64 encoded token containing auth_token-user_id
            cubejs_api_url: Optional CubeJS API URL
            generate_embeddings: Whether to generate embeddings for schemas
            
        Returns:
            Dict with success status and loaded data
        """
        logger.info("=" * 80)
        logger.info("LOADING SCHEMA WITH SCHEMA MANAGER")
        logger.info("=" * 80)
        
        # Step 1: Decode token
        logger.info("🔐 Decoding Base64 token...")
        auth_token, user_id = decode_base64_token(base64_token)
        
        if not auth_token:
            return {
                "success": False,
                "error": "Failed to decode base64 token"
            }
        
        logger.info(f"✅ Token decoded - User ID: {user_id}")
        
        # Step 2: Get CubeJS URL
        if not cubejs_api_url:
            cubejs_api_url = os.getenv("CUBEJS_API_URL", "analytics.vwbeatroute.com/api/v1/meta")
        
        if not cubejs_api_url.startswith(('http://', 'https://')):
            cubejs_api_url = f"https://{cubejs_api_url}"
        
        # Step 3: Fetch schema
        logger.info("📊 Fetching schema from CubeJS and INFORMATION_SCHEMA...")
        
        try:
            schema_list, schema_map, cubejs_data = get_final_schema_from_token(
                cubejs_api_url,
                auth_token,
                session_id=base64_token  # Pass the full base64 token as session_id for DB auth
            )
            
            # Store the data
            self.set_schema_map(schema_map)
            self.set_list_of_table_schemas(schema_list)
            self.cubejs_data = cubejs_data
            
            logger.info(f"✅ Schema loaded: {len(schema_map)} tables")
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch schema: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
        # Step 4: Generate embeddings if requested
        if generate_embeddings and schema_list:
            try:
                embeddings_schema = self.get_embeddings(schema_list)
                self.set_embeddings_schema(embeddings_schema)
                logger.info(f"Schema embeddings generated: {embeddings_schema.shape[0]} tables")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                return {
                    "success": False,
                    "error": f"Schema loaded but embeddings failed: {e}",
                    "schema_loaded": True
                }
        
        logger.info("🎉 Schema Manager loaded successfully!")
        
        return {
            "success": True,
            "table_count": len(schema_map),
            "schema_map": self.schema_map,
            "list_of_table_schemas": self.list_of_table_schemas,
            "embeddings_schema": self.embeddings_schema,
            "embeddings_generated": generate_embeddings,
            "user_id": user_id,
            "auth_token": auth_token
        }
    
    def get_schema_for_table(self, table_name: str) -> Optional[str]:
        """Get schema string for a specific table"""
        return self.schema_map.get(table_name)
    
    def get_all_table_names(self) -> List[str]:
        """Get list of all table names"""
        return list(self.schema_map.keys())
    
    def get_top_k_similar_tables(self, current_question: str, k: int = 10) -> List[str]:
        """
        Get top K most similar table schemas to the current question using embeddings
        
        Args:
            current_question: User's question
            k: Number of top similar tables to return
            
        Returns:
            List of table schema strings
        """
        if self.embeddings_schema is None:
            logger.error("Embeddings not generated. Cannot perform similarity search.")
            return []
        
        # Get embedding for current question
        current_embedding = self.get_embeddings([current_question])[0]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([current_embedding], self.embeddings_schema)[0]
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Get corresponding schemas
        similar_schemas = [self.list_of_table_schemas[i] for i in top_indices]
        
        return similar_schemas
    
    def extract_table_names_from_sql(self, sql_queries: List[str]) -> Set[str]:
        """
        Extract unique table names from a list of SQL queries using sqlglot
        
        Args:
            sql_queries: List of SQL query strings
            
        Returns:
            Set of unique table names used in the queries
        """
        tables = set()
        
        for i, query in enumerate(sql_queries, 1):
            try:
                parsed = sqlglot.parse_one(query)
                found_tables = [t.name for t in parsed.find_all(sqlglot.exp.Table)]
                tables.update(found_tables)
                    
            except Exception as e:
                logger.debug(f"Error parsing query {i}: {e}")
                continue
        
        return tables
    
    def get_schema_to_use_in_prompt(
        self, 
        current_question: str, 
        list_similar_question_sql_pair: List[Dict[str, str]],
        k: int = 10
    ) -> str:
        """
        Get focused schema context for SQL generation prompt
        
        This method combines two approaches:
        1. Find top K similar tables based on question embedding similarity
        2. Extract tables from retrieved similar SQL queries
        
        Args:
            current_question: User's natural language question
            list_similar_question_sql_pair: List of dicts with 'sql' key containing SQL queries
            k: Number of top similar tables to retrieve (default: 10)
            
        Returns:
            Formatted schema string containing only relevant tables
        """
        import time
        start_time = time.time()
        
        # Step 1: Get top K similar tables based on question embedding
        list_schema_tables = self.get_top_k_similar_tables(current_question, k=k)
        
        # Extract table names from similar schemas
        set_tables = set()
        for table_schema in list_schema_tables:
            # Table name is in format "Table: TableName"
            table_name = table_schema.split("\n")[0][7:]  # Remove "Table: " prefix
            set_tables.add(table_name)
        
        # Step 2: Extract tables from retrieved SQL queries
        sql_queries = [item['sql'] for item in list_similar_question_sql_pair if 'sql' in item]
        tables_from_sql = self.extract_table_names_from_sql(sql_queries)
        
        # Step 3: Combine both sets
        set_tables.update(tables_from_sql)
        
        # Step 4: Build schema for prompt
        schema_for_prompt = []
        tables_present = []
        tables_missing = []
        
        for table in sorted(set_tables):
            schema = self.schema_map.get(table, None)
            if schema is not None:
                schema_for_prompt.append(schema)
                tables_present.append(table)
            else:
                tables_missing.append(table)
        
        # Log final result with timing
        schema_time = time.time() - start_time
        logger.info(f"SCHEMA | Focused schema: {len(tables_present)} tables ({schema_time:.2f}s)")
        if tables_missing:
            logger.warning(f"SCHEMA | Missing {len(tables_missing)} tables: {tables_missing}")
        
        # Return combined schema
        return "\n\n".join(schema_for_prompt)
    
    def __repr__(self):
        """String representation of the schema manager"""
        embeddings_info = f"shape {self.embeddings_schema.shape}" if self.embeddings_schema is not None else "not generated"
        return (f"SchemaManager(tables={len(self.schema_map)}, "
                f"embeddings={embeddings_info})")


def fetch_complete_schema(
    cubejs_api_url: Optional[str] = None,
    auth_token: Optional[str] = None,
    base64_token: Optional[str] = None
) -> Dict[str, any]:
    """
    Convenience function to fetch complete schema with flexible authentication
    
    Args:
        cubejs_api_url: CubeJS API URL (defaults to env var CUBEJS_API_URL)
        auth_token: Plain auth token (optional if base64_token provided)
        base64_token: Base64 encoded token (will be decoded to get auth_token)
        
    Returns:
        Dict containing:
            - success: bool
            - table_schemas: List of schema strings
            - table_map: Dict of table -> schema
            - cubejs_metadata: Raw CubeJS data
            - table_count: Number of tables
    """
    # Get CubeJS URL from params or env
    if not cubejs_api_url:
        cubejs_api_url = os.getenv("CUBEJS_API_URL", "analytics.vwbeatroute.com/api/v1/meta")
    
    # Add protocol if missing
    if not cubejs_api_url.startswith(('http://', 'https://')):
        cubejs_api_url = f"https://{cubejs_api_url}"
    
    # Get auth token
    if base64_token:
        auth_token, user_id = decode_base64_token(base64_token)
        if not auth_token:
            return {
                "success": False,
                "error": "Failed to decode base64 token",
                "table_schemas": [],
                "table_map": {},
                "cubejs_metadata": {}
            }
    elif not auth_token:
        auth_token = os.getenv("CUBEJS_API_TOKEN", "")
    
    if not auth_token:
        logger.error("❌ No authentication token provided")
        return {
            "success": False,
            "error": "No authentication token provided",
            "table_schemas": [],
            "table_map": {},
            "cubejs_metadata": {}
        }
    
    # Fetch schema
    try:
        table_schemas, table_map, cubejs_data = get_final_schema_from_token(
            cubejs_api_url, 
            auth_token
        )
        
        return {
            "success": True,
            "table_schemas": table_schemas,
            "table_map": table_map,
            "cubejs_metadata": cubejs_data,
            "table_count": len(table_map)
        }
    except Exception as e:
        logger.error(f"❌ Error fetching schema: {e}")
        return {
            "success": False,
            "error": str(e),
            "table_schemas": [],
            "table_map": {},
            "cubejs_metadata": {}
        }


def main(base64_token: str, cubejs_api_url: Optional[str] = None) -> Dict[str, any]:
    """
    Main function to fetch complete schema from a base64 encoded token
    
    Usage:
        from schema_fetcher import main
        
        # Pass your base64 token directly
        result = main("your_base64_token_here")
        
        if result["success"]:
            print(f"Got schema for {result['table_count']} tables")
            # Access schemas
            for table_name, schema in result['table_map'].items():
                print(f"\\n{schema}")
    
    Args:
        base64_token: Base64 encoded token containing auth_token-user_id
        cubejs_api_url: Optional CubeJS API URL (defaults to env var or analytics.vwbeatroute.com/api/v1/meta)
        
    Returns:
        Dict containing:
            - success: bool
            - table_schemas: List of formatted schema strings
            - table_map: Dict mapping table name to schema string
            - cubejs_metadata: Raw CubeJS metadata
            - table_count: Number of tables
            - auth_token: Decoded auth token (for reference)
            - user_id: Decoded user ID (for reference)
    """
    print("=" * 80)
    print("FETCHING SCHEMA FROM BASE64 TOKEN")
    print("=" * 80)
    
    # Step 1: Decode the token
    print("\n🔐 Decoding Base64 token...")
    auth_token, user_id = decode_base64_token(base64_token)
    
    if not auth_token:
        return {
            "success": False,
            "error": "Failed to decode base64 token",
            "table_schemas": [],
            "table_map": {},
            "cubejs_metadata": {},
            "table_count": 0,
            "auth_token": None,
            "user_id": None
        }
    
    print(f"✅ Token decoded successfully")
    print(f"   User ID: {user_id}")
    
    # Step 2: Fetch schema
    print(f"\n📊 Fetching complete schema...")
    result = fetch_complete_schema(
        cubejs_api_url=cubejs_api_url,
        auth_token=auth_token
    )
    
    # Add decoded credentials to result
    result["auth_token"] = auth_token
    result["user_id"] = user_id
    
    if result["success"]:
        print(f"\n✅ Schema fetched successfully!")
        print(f"📋 Total tables: {result['table_count']}")
        print(f"📄 Tables: {', '.join(list(result['table_map'].keys())[:5])}{'...' if result['table_count'] > 5 else ''}")
        
        # Show sample
        if result['table_schemas']:
            print(f"\n📝 Sample schema (first table):")
            first_schema = result['table_schemas'][0]
            print(first_schema[:300] + "..." if len(first_schema) > 300 else first_schema)
    else:
        print(f"\n❌ Failed to fetch schema: {result.get('error')}")
    
    return result


if __name__ == "__main__":
    import sys
    
    # Check if token provided as command line argument
    if len(sys.argv) > 1:
        base64_token = sys.argv[1]
        cubejs_url = sys.argv[2] if len(sys.argv) > 2 else "analytics.vwbeatroute.com/api/v1/meta"
        
        print(f"\n💡 Running with provided token...")
        
        # Use SchemaManager for complete workflow
        schema_manager = SchemaManager()
        result = schema_manager.load_schema_from_token(base64_token, cubejs_url)
        
        if result["success"]:
            print(f"\n🎉 SUCCESS! Schema loaded with SchemaManager")
            print(f"📋 Total tables: {result['table_count']}")
            print(f"🔮 Embeddings: {'Generated' if result['embeddings_generated'] else 'Not generated'}")
            print(f"\n📊 Access schema data:")
            print(f"   - schema_manager.schema_map: {len(schema_manager.schema_map)} tables")
            print(f"   - schema_manager.list_of_table_schemas: {len(schema_manager.list_of_table_schemas)} schemas")
            print(f"   - schema_manager.embeddings_schema: {schema_manager.embeddings_schema.shape if schema_manager.embeddings_schema is not None else 'None'}")
        else:
            print(f"\n❌ FAILED: {result.get('error')}")
    else:
        # Run with hardcoded base64 token for testing
        base64_token = "R05zeTl5dDgxVkJBUkpfcU11QXhoYlFESk9UcWhRb2wtNg"
        cubejs_url = "analytics.vwbeatroute.com/api/v1/meta"
        
        print(f"\n💡 Running SchemaManager with hardcoded base64 token...")
        print("=" * 80)
        
        # Create SchemaManager and load schema
        schema_manager = SchemaManager()
        result = schema_manager.load_schema_from_token(
            base64_token, 
            cubejs_url,
            generate_embeddings=True  # Set to False to skip embeddings
        )
        
        if result["success"]:
            print(f"\n🎉 SUCCESS! Schema loaded with embeddings")
            print(f"\n📊 Schema Manager State:")
            print(f"   - Tables loaded: {len(schema_manager.schema_map)}")
            print(f"   - Schema list length: {len(schema_manager.list_of_table_schemas)}")
            print(f"   - Embeddings shape: {schema_manager.embeddings_schema.shape if schema_manager.embeddings_schema is not None else 'None'}")
            
            print(f"\n� Available tables:")
            for i, table_name in enumerate(schema_manager.get_all_table_names()[:10], 1):
                print(f"   {i}. {table_name}")
            
            if len(schema_manager.schema_map) > 10:
                print(f"   ... and {len(schema_manager.schema_map) - 10} more tables")
            
            print(f"\n💡 Usage Example:")
            print(f"   schema_manager.schema_map['CustomerInvoice']  # Get specific table schema")
            print(f"   schema_manager.list_of_table_schemas[0]  # Get first schema")
            print(f"   schema_manager.embeddings_schema  # Access embeddings array")
            
        else:
            print(f"\n❌ FAILED: {result.get('error')}")
        
        # Exit to prevent running old test code
        sys.exit(0)
    
    if False:  # Disabled old test code
        # Test with environment variables
        print("=" * 80)
        print("TESTING SCHEMA FETCHER")
        print("=" * 80)
        print("\n💡 Usage: python schema_fetcher.py <base64_token> [cubejs_url]")
        print("   Or set environment variables and run without arguments\n")
        
        # Test 1: Using environment variables
        print("📝 Test 1: Fetching schema using environment variables")
        result = fetch_complete_schema()
        
        if result["success"]:
            print(f"\n✅ Success!")
            print(f"📊 Total tables: {result['table_count']}")
            print(f"📋 Tables: {list(result['table_map'].keys())}")
            
            # Show first table schema as example
            if result['table_schemas']:
                first_schema = result['table_schemas'][0]
                print(f"\n📄 Example schema (first 500 chars):")
                print(first_schema[:500] + "..." if len(first_schema) > 500 else first_schema)
        else:
            print(f"\n❌ Failed: {result['error']}")
        
        # Test 2: Decode a base64 token (example)
        print("\n" + "=" * 80)
        print("📝 Test 2: Decoding Base64 token (if you have one)")
        print("=" * 80)
        
        example_token = os.getenv("TEST_BASE64_TOKEN", "")
        if example_token:
            test_result = main(example_token)
            if test_result["success"]:
                print(f"\n✅ Test successful: {test_result['table_count']} tables fetched")
            else:
                print(f"\n❌ Test failed: {test_result.get('error')}")
        else:
            print("ℹ️  No TEST_BASE64_TOKEN in environment, skipping test")
