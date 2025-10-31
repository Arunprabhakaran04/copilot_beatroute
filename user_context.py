import json
import pickle
import base64
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import requests
from loguru import logger

from schema_fetcher import SchemaManager, decode_base64_token
from constants import SESSION_TO_USER_ID


class UserContext:
    """
    Manages all user-specific data and schema for SQL generation.
    
    This class is initialized once per user connection and stores:
    - User identity (user_id, user_name, email)
    - Schema data (schema_map, list_of_table_schemas, embeddings_schema)
    - Campaign mappings and question lists
    - User role/designation lists
    
    Can be serialized to Redis for caching across sessions.
    """
    
    def __init__(
        self,
        user_id: str,
        user_name: str = "",
        email: str = "",
        auth_token: str = "",
    ):
        """
        Initialize UserContext with user credentials.
        
        Args:
            user_id: Unique user identifier
            user_name: User's display name
            email: User's email address
            auth_token: Authentication token (will be used to load schema)
        """
        # User identity
        self.user_id = user_id
        self.user_name = user_name
        self.email = email
        self.auth_token = auth_token
        
        # Campaign data (populated from database)
        self.campaign_table: str = ""
        self.campaign_info_table: str = ""
        self.list_questions: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Schema data (loaded from SchemaManager)
        self.schema_map: Dict[str, str] = {}
        self.list_of_table_schemas: List[str] = []
        self.embeddings_schema: Optional[np.ndarray] = None
        
        # User metadata (populated from database)
        self.customer_subtype_list: List[str] = []
        self.user_role_list: List[str] = []
        self.user_designation_list: List[str] = []
        self.campaign_custom_map: Dict[str, Any] = {}
        
        # Internal state
        self._schema_loaded = False
        self._schema_manager: Optional[SchemaManager] = None
    
    async def load_schema_from_token(
        self,
        base64_token: str,
        cubejs_api_url: str = "analytics.vwbeatroute.com/api/v1/meta",
        generate_embeddings: bool = True,
        session_id: Optional[str] = None,
        db_connection = None
    ) -> bool:
        """
        Load database schema using the base64 token.
        
        This should be called once when the user first connects.
        
        Args:
            base64_token: Base64 encoded token containing auth credentials
            cubejs_api_url: CubeJS API endpoint
            generate_embeddings: Whether to generate embeddings for schema
            session_id: Optional session ID to map to user_id
            db_connection: Database connection object for querying user metadata
            
        Returns:
            bool: True if schema loaded successfully
        """
        try:
            logger.info(f"🔐 Loading schema for user {self.user_id}...")
            
            # Decode token to get user_id from token
            auth_token, token_user_id = decode_base64_token(base64_token)
            
            if not auth_token or not token_user_id:
                logger.error("❌ Failed to decode base64 token")
                return False
            
            # Store auth token
            self.auth_token = auth_token
            
            # Store the mapping in constants if session_id is provided
            if session_id:
                SESSION_TO_USER_ID[session_id] = token_user_id
                logger.info(f"✅ Stored session mapping: {session_id} -> user_{token_user_id}")
            
            # Update user_id if it was auto-generated
            if self.user_id != token_user_id:
                logger.info(f"📝 Updating user_id from {self.user_id} to {token_user_id}")
                self.user_id = token_user_id
            
            # Create schema manager
            self._schema_manager = SchemaManager()
            
            # Load schema (this generates embeddings internally)
            result = self._schema_manager.load_schema_from_token(
                base64_token=base64_token,
                cubejs_api_url=cubejs_api_url,
                generate_embeddings=generate_embeddings
            )
            
            if not result["success"]:
                logger.error(f"❌ Schema load failed: {result.get('error')}")
                return False
            
            # Store schema data
            self.schema_map = self._schema_manager.schema_map
            self.list_of_table_schemas = self._schema_manager.list_of_table_schemas
            self.embeddings_schema = self._schema_manager.embeddings_schema
            
            self._schema_loaded = True
            
            logger.info(f"✅ Schema loaded for user {self.user_id}")
            logger.info(f"   📋 Tables: {len(self.schema_map)}")
            logger.info(f"   🔮 Embeddings: {self.embeddings_schema.shape if self.embeddings_schema is not None else 'None'}")
            
            # ============================================================================
            # POPULATE USER METADATA FROM DATABASE
            # ============================================================================
            if db_connection:
                logger.info(f"👤 Populating user metadata for user {self.user_id}...")
                
                # 1. Load user details (name, email, role, designation)
                user_details = self._get_user_details(db_connection, self.user_id)
                if user_details:
                    self.user_name = user_details.get("user_name", "")
                    self.email = user_details.get("email", "")
                    logger.info(f"   ✅ User: {self.user_name} ({self.email})")
                
                # 2. Load all user roles
                self.user_role_list = self._get_distinct_user_roles(db_connection)
                logger.info(f"   ✅ User roles: {len(self.user_role_list)} distinct roles")
                
                # 3. Load all user designations
                self.user_designation_list = self._get_distinct_user_designations(db_connection)
                logger.info(f"   ✅ User designations: {len(self.user_designation_list)} distinct designations")
                
                # 4. Load all customer subtypes
                self.customer_subtype_list = self._get_distinct_customer_subtypes(db_connection)
                logger.info(f"   ✅ Customer subtypes: {len(self.customer_subtype_list)} distinct subtypes")
                
                # 5. Load campaign information
                campaign_result = self._load_campaign_information(db_connection, cubejs_api_url, auth_token)
                if campaign_result["success"]:
                    logger.info(f"   ✅ Campaign data: {campaign_result['campaign_count']} campaigns loaded")
                else:
                    logger.warning(f"   ⚠️ Campaign data load failed: {campaign_result.get('error', 'Unknown error')}")
            else:
                logger.warning("⚠️ No database connection provided, skipping user metadata population")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading schema for user {self.user_id}: {e}")
            return False
    
    def get_focused_schema(
        self,
        question: str,
        retrieved_sqls: List[Dict[str, Any]],
        k: int = 10
    ) -> str:
        """
        Get focused schema based on question and retrieved SQL queries.
        
        Uses embeddings + SQL parsing to find relevant tables.
        
        Args:
            question: User's natural language question
            retrieved_sqls: List of similar SQL queries from vector DB
            k: Number of top similar tables to retrieve
            
        Returns:
            str: Focused schema containing only relevant tables
        """
        if not self._schema_loaded or self._schema_manager is None:
            logger.warning("⚠️ Schema not loaded, returning empty schema")
            return ""
        
        return self._schema_manager.get_schema_to_use_in_prompt(
            current_question=question,
            list_similar_question_sql_pair=retrieved_sqls,
            k=k
        )
    
    def get_schema_manager(self) -> Optional[SchemaManager]:
        """
        Get the underlying SchemaManager instance.
        
        Returns:
            SchemaManager or None if schema not loaded
        """
        return self._schema_manager
    
    def is_schema_loaded(self) -> bool:
        """Check if schema has been loaded."""
        return self._schema_loaded
    
    # ============================================================================
    # USER METADATA POPULATION METHODS
    # ============================================================================
    
    def _get_user_details(self, db_connection, user_id: str) -> Optional[Dict[str, str]]:
        """
        Get user name, email, role, and designation from database.
        
        Args:
            db_connection: Database connection object
            user_id: User ID to query
            
        Returns:
            Dict with user_name, email, role, designation or None if query fails
        """
        try:
            sql = f"""
                SELECT 
                    ViewUser.name AS UserName, 
                    ViewUser.email AS Email, 
                    ViewUser.role AS Role,
                    ViewUser.designation AS Designation 
                FROM ViewUser 
                WHERE ViewUser.id = {user_id}
            """
            
            result = db_connection.execute_query(sql)
            
            if result['success'] and result['metadata']['row_count'] > 0:
                # Parse JSON string to get data
                data = json.loads(result['data'])
                logger.debug(f"User details raw data: {data}")
                
                if data and len(data) > 0:
                    row = data[0]
                    # Column names from PostgreSQL are lowercase
                    user_details = {
                        "user_name": row.get("username") or row.get("UserName", ""),
                        "email": row.get("email") or row.get("Email", ""),
                        "role": row.get("role") or row.get("Role", ""),
                        "designation": row.get("designation") or row.get("Designation", "")
                    }
                    logger.debug(f"Extracted user details: {user_details}")
                    return user_details
            else:
                logger.warning(f"No user found with ID {user_id}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting user details: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _get_distinct_user_roles(self, db_connection) -> List[str]:
        """
        Get all distinct user roles from database.
        
        Args:
            db_connection: Database connection object
            
        Returns:
            List of distinct user roles
        """
        try:
            sql = "SELECT DISTINCT ViewUser.role AS UserRoles FROM ViewUser WHERE ViewUser.role IS NOT NULL"
            result = db_connection.execute_query(sql)
            
            if result['success']:
                # Parse JSON string to get data
                data = json.loads(result['data'])
                logger.debug(f"User roles raw data (first 3): {data[:3] if data else []}")
                
                if data:
                    # Extract roles and filter out None/empty values (check both lowercase and original case)
                    roles = [row.get("userroles") or row.get("UserRoles") for row in data]
                    roles = [r for r in roles if r]  # Filter out None/empty
                    unique_roles = list(set(roles))  # Remove duplicates
                    logger.debug(f"Extracted {len(unique_roles)} unique roles")
                    return unique_roles
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting user roles: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _get_distinct_user_designations(self, db_connection) -> List[str]:
        """
        Get all distinct user designations from database.
        
        Args:
            db_connection: Database connection object
            
        Returns:
            List of distinct user designations
        """
        try:
            sql = "SELECT DISTINCT ViewUser.designation AS UserDesignations FROM ViewUser WHERE ViewUser.designation IS NOT NULL"
            result = db_connection.execute_query(sql)
            
            if result['success']:
                # Parse JSON string to get data
                data = json.loads(result['data'])
                logger.debug(f"User designations raw data (first 3): {data[:3] if data else []}")
                
                if data:
                    # Extract designations and filter out None/empty values (check both lowercase and original case)
                    designations = [row.get("userdesignations") or row.get("UserDesignations") for row in data]
                    designations = [d for d in designations if d]  # Filter out None/empty
                    unique_designations = list(set(designations))  # Remove duplicates
                    logger.debug(f"Extracted {len(unique_designations)} unique designations")
                    return unique_designations
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting user designations: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _get_distinct_customer_subtypes(self, db_connection) -> List[str]:
        """
        Get all distinct customer subtypes from database.
        
        Args:
            db_connection: Database connection object
            
        Returns:
            List of distinct customer subtypes
        """
        try:
            sql = "SELECT DISTINCT ViewCustomer.subType AS CustomerSubtype FROM ViewCustomer WHERE ViewCustomer.subType IS NOT NULL"
            result = db_connection.execute_query(sql)
            
            if result['success']:
                # Parse JSON string to get data
                data = json.loads(result['data'])
                logger.debug(f"Customer subtypes raw data (first 3): {data[:3] if data else []}")
                
                if data:
                    # Extract subtypes and filter out None/empty values (check both lowercase and original case)
                    subtypes = [row.get("customersubtype") or row.get("CustomerSubtype") for row in data]
                    subtypes = [s for s in subtypes if s]  # Filter out None/empty
                    unique_subtypes = list(set(subtypes))  # Remove duplicates
                    logger.debug(f"Extracted {len(unique_subtypes)} unique customer subtypes")
                    return unique_subtypes
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting customer subtypes: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _load_campaign_information(
        self, 
        db_connection, 
        cubejs_api_url: str,
        auth_token: str
    ) -> Dict[str, Any]:
        """
        Load campaign information including campaign tables and custom column mappings.
        
        This method:
        1. Gets all campaigns (id, name, dates)
        2. Gets available CampaignResponse tables
        3. Builds campaign info dataframe linking campaigns to their response tables
        4. Builds custom column shortTitle mapping from CubeJS metadata
        
        Args:
            db_connection: Database connection object
            cubejs_api_url: CubeJS API endpoint
            auth_token: Authentication token for CubeJS API
            
        Returns:
            Dict with success status and campaign count
        """
        try:
            # 1. Get campaign information (id, name, dates)
            campaign_map, campaign_list = self._get_campaign_info(db_connection)
            
            if not campaign_map:
                return {"success": False, "error": "No campaigns found"}
            
            # 2. Get available CampaignResponse tables
            campaign_tables = self._get_campaign_response_tables(db_connection)
            
            # 3. Build campaign info DataFrame
            campaign_info_df = self._build_campaign_info_dataframe(campaign_map, campaign_tables)
            
            # Store as formatted string for use in prompts
            self.campaign_table = campaign_info_df.to_string(index=False)
            self.campaign_info_table = self.campaign_table  # Alias for backward compatibility
            
            # 4. Build custom column shortTitle mapping
            self.campaign_custom_map = self._build_custom_column_map(cubejs_api_url, auth_token)
            
            return {
                "success": True,
                "campaign_count": len(campaign_map),
                "campaign_tables": len(campaign_tables),
                "custom_columns": len(self.campaign_custom_map)
            }
            
        except Exception as e:
            logger.error(f"❌ Error loading campaign information: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_campaign_info(self, db_connection) -> tuple[Dict[int, str], List[Dict[str, Any]]]:
        """
        Get campaign information from VmCampaign table.
        
        Returns:
            Tuple of (campaign_map, campaign_list)
            - campaign_map: {campaign_id: campaign_name}
            - campaign_list: List of campaign records
        """
        try:
            sql = "SELECT name, id, startDate, endDate, publishedOn FROM VmCampaign"
            result = db_connection.execute_query(sql)
            
            if result['success']:
                # Parse JSON string to get data
                data = json.loads(result['data'])
                if data:
                    campaign_map = {int(row["id"]): row["name"] for row in data}
                    return campaign_map, data
            
            return {}, []
            
        except Exception as e:
            logger.error(f"❌ Error getting campaign info: {e}")
            return {}, []
    
    def _get_campaign_response_tables(self, db_connection) -> List[str]:
        """
        Get list of available CampaignResponse tables.
        
        Returns:
            List of table names matching pattern 'CampaignResponse%'
        """
        try:
            sql = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' 
                  AND table_name LIKE 'CampaignResponse%'
            """
            
            result = db_connection.execute_query(sql)
            
            if result['success']:
                # Parse JSON string to get data
                data = json.loads(result['data'])
                if data:
                    return [row["table_name"] for row in data]
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting campaign response tables: {e}")
            return []
    
    def _build_campaign_info_dataframe(
        self, 
        campaign_map: Dict[int, str], 
        available_tables: List[str]
    ) -> pd.DataFrame:
        """
        Build DataFrame linking campaigns to their response tables.
        
        Args:
            campaign_map: Dict mapping campaign_id to campaign_name
            available_tables: List of available CampaignResponse table names
            
        Returns:
            DataFrame with columns: campaign_name, campaign_id, response_table_name, is_table_available
        """
        data = []
        
        for campaign_id, campaign_name in campaign_map.items():
            table_name = f"CampaignResponse{campaign_id}"
            is_available = table_name in available_tables
            
            data.append({
                "campaign_name": campaign_name,
                "campaign_id": int(campaign_id),
                "response_table_name": table_name if is_available else None,
                "is_table_available": "yes" if is_available else "no"
            })
        
        return pd.DataFrame(data)
    
    def _build_custom_column_map(self, cubejs_api_url: str, auth_token: str) -> Dict[str, Dict[str, str]]:
        """
        Build mapping of CampaignResponse tables to their custom column shortTitles.
        
        This fetches CubeJS metadata and extracts shortTitle for custom_* columns.
        
        Args:
            cubejs_api_url: CubeJS API endpoint
            auth_token: Authentication token
            
        Returns:
            Dict: {table_name: {custom_column_name: shortTitle}}
        """
        try:
            # Ensure URL has scheme
            if not cubejs_api_url.startswith(('http://', 'https://')):
                cubejs_api_url = f"https://{cubejs_api_url}"
            
            # Fetch CubeJS metadata
            headers = {"Authorization": f"{auth_token}"}
            response = requests.get(cubejs_api_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            result_map = {}
            
            for cube in data.get("cubes", []):
                table = cube["name"]
                
                # Only process CampaignResponse tables
                if "CampaignResponse" not in table:
                    continue
                
                table_map = {}
                
                # Process dimensions
                for dim in cube.get("dimensions", []):
                    col = dim["name"].split(".")[1] if "." in dim["name"] else dim["name"]
                    if col.startswith("custom_"):
                        table_map[col] = dim.get("shortTitle", "")
                
                # Process measures
                for measure in cube.get("measures", []):
                    col = measure["name"].split(".")[1] if "." in measure["name"] else measure["name"]
                    if col.startswith("custom_"):
                        table_map[col] = measure.get("shortTitle", "")
                
                if table_map:
                    result_map[table] = table_map
            
            return result_map
            
        except Exception as e:
            logger.error(f"❌ Error building custom column map: {e}")
            return {}
    
    # ============================================================================
    # SERIALIZATION METHODS (for Redis storage)
    # ============================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert UserContext to dictionary for JSON serialization.
        
        Note: NumPy arrays are converted to lists for JSON compatibility.
        
        Returns:
            Dict containing all user context data
        """
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "email": self.email,
            "auth_token": self.auth_token,
            "campaign_table": self.campaign_table,
            "campaign_info_table": self.campaign_info_table,
            "list_questions": self.list_questions,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "schema_map": self.schema_map,
            "list_of_table_schemas": self.list_of_table_schemas,
            "embeddings_schema": self.embeddings_schema.tolist() if self.embeddings_schema is not None else None,
            "customer_subtype_list": self.customer_subtype_list,
            "user_role_list": self.user_role_list,
            "user_designation_list": self.user_designation_list,
            "campaign_custom_map": self.campaign_custom_map,
            "_schema_loaded": self._schema_loaded,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserContext":
        """
        Create UserContext from dictionary.
        
        Args:
            data: Dictionary containing user context data
            
        Returns:
            UserContext instance
        """
        # Create instance
        context = cls(
            user_id=data["user_id"],
            user_name=data.get("user_name", ""),
            email=data.get("email", ""),
            auth_token=data.get("auth_token", "")
        )
        
        # Restore all fields
        context.campaign_table = data.get("campaign_table", "")
        context.campaign_info_table = data.get("campaign_info_table", "")
        context.list_questions = data.get("list_questions", [])
        context.embeddings = np.array(data["embeddings"]) if data.get("embeddings") else None
        context.schema_map = data.get("schema_map", {})
        context.list_of_table_schemas = data.get("list_of_table_schemas", [])
        context.embeddings_schema = np.array(data["embeddings_schema"]) if data.get("embeddings_schema") else None
        context.customer_subtype_list = data.get("customer_subtype_list", [])
        context.user_role_list = data.get("user_role_list", [])
        context.user_designation_list = data.get("user_designation_list", [])
        context.campaign_custom_map = data.get("campaign_custom_map", {})
        context._schema_loaded = data.get("_schema_loaded", False)
        
        # Recreate schema manager if schema was loaded
        if context._schema_loaded:
            context._schema_manager = SchemaManager()
            context._schema_manager.schema_map = context.schema_map
            context._schema_manager.list_of_table_schemas = context.list_of_table_schemas
            context._schema_manager.embeddings_schema = context.embeddings_schema
        
        return context
    
    def to_json(self) -> str:
        """
        Serialize UserContext to JSON string.
        
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "UserContext":
        """
        Deserialize UserContext from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            UserContext instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def to_redis_value(self) -> bytes:
        """
        Serialize UserContext for Redis storage using pickle.
        
        Pickle preserves NumPy arrays better than JSON.
        
        Returns:
            Pickled bytes
        """
        return pickle.dumps(self.to_dict())
    
    @classmethod
    def from_redis_value(cls, redis_bytes: bytes) -> "UserContext":
        """
        Deserialize UserContext from Redis bytes.
        
        Args:
            redis_bytes: Pickled bytes from Redis
            
        Returns:
            UserContext instance
        """
        data = pickle.loads(redis_bytes)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """String representation of UserContext."""
        return (
            f"UserContext(user_id={self.user_id}, "
            f"user_name={self.user_name}, "
            f"schema_loaded={self._schema_loaded}, "
            f"tables={len(self.schema_map)})"
        )
