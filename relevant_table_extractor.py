"""
Extracts relevant tables for a given query by:
1. Fetching table schemas from CubeJS metadata
2. Using LLM to identify relevant tables based on the query
3. Extracting tables mentioned in retrieved SQL queries
4. Combining both sets to create a focused schema context
"""

import os
import re
import requests
from typing import Dict, List, Set, Any, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from db_connection import execute_sql
from loguru import logger

load_dotenv()


class RelevantTableExtractor:
    """Extracts relevant tables for SQL generation based on query context"""
    
    # All available tables in the database
    ALL_TABLES = [
        'Order', 'OrderDetail', 'CustomerInvoice', 'CustomerInvoiceDetail',
        'DistributorOrderHeader', 'DistributorOrderDetail',
        'DistributorSales', 'DistributorSalesDetail',
        'TeamPerformance', 'UserAttendance', 'UserDistanceLog',
        'UserSchedule',
        'Sku', 'Category', 'Brand',
        'SkuCustomField',
        'ViewCustomer', 'ViewUser', 'ViewDistributor', 'ViewCustomerActivity',
        'CustomerCustomField', 'UserCustomField', 'DistributorCustomField',
        'CustomerInventory', 'DistributorInventory',
        'Offtake', 'OfftakeDetail',
        'OrderReturn', 'OrderReturnDetail',
        'CustomerDebitNote', 'CustomerCreditNote',
        'CustomerPayment', 'DistributorPayment',
        'CustomerOutstanding',
        'Scheme',
        'VmCampaign'
    ]
    
    # Columns to exclude from schema
    EXCLUDED_KEYWORDS = ["__", "monthYear", "year", "quarter"]
    
    # Specific columns to exclude
    EXCLUDED_COLUMNS = [
        ('Order', 'available'),
        ('Sku', 'AvailableStatus'),
        ('ViewUser', 'availableStatus'),
        ('CustomerInvoiceDetail', 'skuName'),
        ('UserAttendance', 'duration'),
        ('DistributorInventory', 'amount')
    ]
    
    def __init__(self):
        """Initialize the relevant table extractor"""
        self.cubejs_api_url = os.getenv("CUBEJS_API_URL", "analytics.vwbeatroute.com/api/v1/meta")
        self.auth_token = os.getenv("CUBEJS_API_TOKEN", "")
        
        # Add protocol if missing
        if not self.cubejs_api_url.startswith(('http://', 'https://')):
            self.cubejs_api_url = f"https://{self.cubejs_api_url}"
        
        # Initialize LLM for table identification
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Using mini for cost efficiency
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        logger.info(f"RelevantTableExtractor initialized with CubeJS URL: {self.cubejs_api_url}")
    
    def fetch_cubejs_metadata(self) -> Dict[str, Any]:
        """Fetch schema metadata from CubeJS API"""
        try:
            headers = {"Authorization": f"{self.auth_token}"}
            response = requests.get(self.cubejs_api_url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"✅ Fetched CubeJS metadata: {len(data.get('cubes', []))} cubes")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to fetch CubeJS metadata: {e}")
            return {"cubes": []}
    
    def build_short_title_map(self, cubejs_data: Dict) -> Dict[Tuple[str, str], str]:
        """Build a map of (table, column) -> short_title from CubeJS metadata"""
        short_title_map = {}
        
        for cube in cubejs_data.get("cubes", []):
            table = cube["name"]
            
            # Process dimensions
            for dim in cube.get("dimensions", []):
                col = dim["name"].split(".")[1]
                short_title = dim.get("shortTitle", "")
                if short_title:
                    short_title_map[(table, col)] = short_title
            
            # Process measures
            for measure in cube.get("measures", []):
                col = measure["name"].split(".")[1]
                short_title = measure.get("shortTitle", "")
                if short_title:
                    short_title_map[(table, col)] = short_title
        
        return short_title_map
    
    def fetch_table_schemas(self, table_names: List[str]) -> Dict[str, str]:
        """
        Fetch column information for specified tables from INFORMATION_SCHEMA
        
        Args:
            table_names: List of table names to fetch schemas for
            
        Returns:
            Dict mapping table name to formatted schema string
        """
        if not table_names:
            return {}
        
        # Build SQL query to fetch schema
        table_list = "', '".join(table_names)
        sql_query = f"""
        SELECT 
            table_name,
            column_name,
            data_type
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE table_name IN ('{table_list}')
        ORDER BY table_name, ordinal_position
        """
        
        try:
            result = execute_sql(sql_query)
            
            if not result.get('success'):
                logger.error(f"❌ Failed to fetch table schemas: {result.get('error')}")
                return {}
            
            data = result.get('results', {}).get('data', [])
            
            # Fetch CubeJS metadata for short titles
            cubejs_data = self.fetch_cubejs_metadata()
            short_title_map = self.build_short_title_map(cubejs_data)
            
            # Group by table and format
            table_schemas = {}
            current_table = None
            schema_lines = []
            
            for row in data:
                table = row['table_name']
                column = row['column_name']
                dtype = row['data_type']
                
                # Check if column should be excluded
                if any(kw in column for kw in self.EXCLUDED_KEYWORDS):
                    continue
                if (table, column) in self.EXCLUDED_COLUMNS:
                    continue
                
                # Start new table if needed
                if current_table != table:
                    if current_table is not None:
                        table_schemas[current_table] = "\n".join(schema_lines)
                    current_table = table
                    schema_lines = [f"Table: {table}"]
                
                # Add column with short title if available
                short_title = short_title_map.get((table, column), "")
                if short_title:
                    schema_lines.append(f"- {column}:")
                    schema_lines.append(f"    Type: {dtype}")
                    schema_lines.append(f"    Description: {short_title}")
                else:
                    schema_lines.append(f"- {column}: {dtype}")
            
            # Add last table
            if current_table is not None:
                table_schemas[current_table] = "\n".join(schema_lines)
            
            logger.info(f"✅ Fetched schemas for {len(table_schemas)} tables")
            return table_schemas
            
        except Exception as e:
            logger.error(f"❌ Error fetching table schemas: {e}")
            return {}
    
    def extract_tables_from_sql(self, sql_query: str) -> Set[str]:
        """
        Extract table names mentioned in a SQL query
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Set of table names found in the query
        """
        if not sql_query:
            return set()
        
        tables = set()
        sql_upper = sql_query.upper()
        
        # Pattern to match FROM and JOIN clauses
        # Matches: FROM TableName, JOIN TableName, CROSS JOIN TableName
        patterns = [
            r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'CROSS\s+JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, sql_upper)
            for match in matches:
                table_name = match.group(1)
                # Find the original case-sensitive version
                for available_table in self.ALL_TABLES:
                    if available_table.upper() == table_name:
                        tables.add(available_table)
                        break
        
        return tables
    
    def identify_relevant_tables_with_llm(self, question: str) -> List[str]:
        """
        Use LLM to identify which tables are likely relevant for the question
        
        Args:
            question: User's natural language question
            
        Returns:
            List of relevant table names
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a database expert. Given a user question and a list of available tables, identify which tables are likely needed to answer the question.

Available Tables and Their Purpose:
- CustomerInvoice, CustomerInvoiceDetail: Customer sales/invoice data
- DistributorSales, DistributorSalesDetail: Distributor sales data
- DistributorOrderHeader, DistributorOrderDetail: Distributor orders
- Order, OrderDetail: General order data
- ViewCustomer: Customer information (name, location, contact, etc.)
- ViewDistributor: Distributor information
- ViewUser: User/employee information
- Sku, Category, Brand: Product information
- CustomerInventory, DistributorInventory: Inventory data
- TeamPerformance: Team performance metrics
- UserAttendance, UserDistanceLog, UserSchedule: User activity tracking
- CustomerPayment, DistributorPayment: Payment records
- CustomerOutstanding: Outstanding payments
- OrderReturn, OrderReturnDetail: Product returns
- CustomerDebitNote, CustomerCreditNote: Credit/debit notes
- Offtake, OfftakeDetail: Offtake data
- Scheme: Scheme/promotion data
- VmCampaign, CampaignResponse: Campaign data
- CustomerCustomField, UserCustomField, DistributorCustomField: Custom fields
- SkuCustomField: Product custom fields
- ViewCustomerActivity: Customer activity tracking

Return ONLY a JSON array of table names, nothing else. Be conservative - include tables that might be needed.

Examples:
Question: "What are the sales of customer ABC in September?"
Response: ["CustomerInvoice", "ViewCustomer"]

Question: "Show me top selling products last month"
Response: ["CustomerInvoiceDetail", "Sku"]

Question: "List distributors in Delhi with their sales"
Response: ["ViewDistributor", "DistributorSales"]
"""),
            ("user", "Question: {question}\n\nResponse:")
        ])
        
        try:
            response = self.llm.invoke(prompt.format(question=question))
            content = response.content.strip()
            
            # Extract JSON array
            import json
            # Remove markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            tables = json.loads(content)
            
            # Validate and filter
            valid_tables = [t for t in tables if t in self.ALL_TABLES]
            
            logger.info(f"✅ LLM identified {len(valid_tables)} relevant tables: {valid_tables}")
            return valid_tables
            
        except Exception as e:
            logger.error(f"❌ Error identifying tables with LLM: {e}")
            # Fallback: return common tables
            return ["CustomerInvoice", "ViewCustomer", "Sku"]
    
    def get_relevant_tables_and_schemas(
        self, 
        question: str, 
        retrieved_sql_queries: List[str] = None
    ) -> Dict[str, Any]:
        """
        Main method to get relevant tables and their schemas
        
        Args:
            question: User's natural language question
            retrieved_sql_queries: List of SQL queries retrieved from embeddings
            
        Returns:
            Dict containing:
                - relevant_tables: Set of table names
                - table_schemas: Dict mapping table name to schema string
                - schema_context: Combined schema string for LLM
        """
        logger.info(f"🔍 Extracting relevant tables for question: {question[:100]}...")
        
        # Step 1: Get tables from LLM analysis
        llm_tables = self.identify_relevant_tables_with_llm(question)
        relevant_tables = set(llm_tables)
        
        logger.info(f"📋 Tables from LLM analysis: {relevant_tables}")
        
        # Step 2: Extract tables from retrieved SQL queries
        if retrieved_sql_queries:
            for sql in retrieved_sql_queries:
                sql_tables = self.extract_tables_from_sql(sql)
                relevant_tables.update(sql_tables)
            
            logger.info(f"📋 Additional tables from retrieved queries: {sql_tables if 'sql_tables' in locals() else 'None'}")
        
        logger.info(f"✅ Total relevant tables: {len(relevant_tables)} - {sorted(relevant_tables)}")
        
        # Step 3: Fetch schemas for relevant tables
        table_schemas = self.fetch_table_schemas(list(relevant_tables))
        
        # Step 4: Create combined schema context
        schema_context = "\n\n".join(table_schemas.values())
        
        return {
            "relevant_tables": relevant_tables,
            "table_schemas": table_schemas,
            "schema_context": schema_context,
            "table_count": len(relevant_tables)
        }


# Convenience function for direct use
def get_relevant_schema_context(question: str, retrieved_sql_queries: List[str] = None) -> str:
    """
    Get focused schema context for a question
    
    Args:
        question: User's natural language question
        retrieved_sql_queries: Optional list of retrieved SQL queries
        
    Returns:
        Formatted schema string containing only relevant tables
    """
    extractor = RelevantTableExtractor()
    result = extractor.get_relevant_tables_and_schemas(question, retrieved_sql_queries)
    return result["schema_context"]


if __name__ == "__main__":
    # Test the extractor
    print("=" * 80)
    print("TESTING RELEVANT TABLE EXTRACTOR")
    print("=" * 80)
    
    extractor = RelevantTableExtractor()
    
    # Test 1: Simple customer query
    question1 = "What is the sales of customer arman08 in September 2025?"
    retrieved_queries1 = [
        "SELECT SUM(CustomerInvoice.dispatchedvalue) FROM CustomerInvoice CROSS JOIN ViewCustomer WHERE ViewCustomer.name = 'test'"
    ]
    
    print(f"\n📝 Question: {question1}")
    print(f"📄 Retrieved queries: {len(retrieved_queries1)}")
    
    result1 = extractor.get_relevant_tables_and_schemas(question1, retrieved_queries1)
    
    print(f"\n✅ Relevant tables: {sorted(result1['relevant_tables'])}")
    print(f"📊 Total tables: {result1['table_count']}")
    print(f"\n📋 Schema Context (first 500 chars):")
    print(result1['schema_context'][:500] + "..." if len(result1['schema_context']) > 500 else result1['schema_context'])
    
    # Test 2: Product query
    print("\n" + "=" * 80)
    question2 = "Show me top 10 selling products last month"
    
    print(f"\n📝 Question: {question2}")
    
    result2 = extractor.get_relevant_tables_and_schemas(question2)
    
    print(f"\n✅ Relevant tables: {sorted(result2['relevant_tables'])}")
    print(f"📊 Total tables: {result2['table_count']}")
