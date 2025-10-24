import json
import pickle
import base64
from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger

from schema_fetcher import SchemaManager


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
        generate_embeddings: bool = True
    ) -> bool:
        """
        Load database schema using the base64 token.
        
        This should be called once when the user first connects.
        
        Args:
            base64_token: Base64 encoded token containing auth credentials
            cubejs_api_url: CubeJS API endpoint
            generate_embeddings: Whether to generate embeddings for schema
            
        Returns:
            bool: True if schema loaded successfully
        """
        try:
            logger.info(f"🔐 Loading schema for user {self.user_id}...")
            
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
