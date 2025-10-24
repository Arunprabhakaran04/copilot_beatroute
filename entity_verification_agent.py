import logging
import json
import time
from typing import Dict, Any, List, Optional
from base_agent import BaseAgent, BaseAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

class EntityVerificationAgent(BaseAgent):
    def __init__(self, llm, db_executor, model_name: str = "gpt-4o"):
        super().__init__(llm)
        self.model_name = model_name
        self.db_executor = db_executor
        logger.info(f"EntityVerificationAgent initialized with model: {model_name}")
    
    def get_agent_type(self) -> str:
        return "entity_verification"
    
    def process(self, state: BaseAgentState) -> BaseAgentState:
        try:
            start_time = time.time()
            question = state["query"]
            user_id = state.get("user_id", "default_user")
            
            logger.info(f"Starting entity verification for query: {question}")
            
            llm_response = self._identify_entities_and_generate_sql(question)
            
            entity_data = self._parse_llm_response(llm_response)
            
            if not entity_data or not entity_data.get("entities"):
                logger.info("No entities detected in query")
                state["status"] = "entity_verified"
                state["result"] = {
                    "original_query": question,
                    "needs_clarification": False,
                    "entity_info": {
                        "entities": [],
                        "entity_types": [],
                        "entity_mapping": {},
                        "verified_entities": {}
                    }
                }
                state["success_message"] = "No entities to verify"
                state["execution_time"] = time.time() - start_time
                return state
            
            verification_results = []
            verified_entities = {}
            
            entities = entity_data.get("entities", [])
            entity_types = entity_data.get("entity_types", [])
            verification_sqls = entity_data.get("verification_sqls", [])
            
            for idx, (entity_name, entity_type, verify_sql) in enumerate(zip(entities, entity_types, verification_sqls)):
                result = self._verify_single_entity(entity_name, entity_type, verify_sql)
                
                if result["status"] == "present":
                    verified_entities[entity_name] = {
                        "type": entity_type,
                        "status": "present",
                        "db_name": result["db_name"]
                    }
                elif result["status"] == "partial":
                    verification_results.append({
                        "entity": entity_name,
                        "type": entity_type,
                        "status": "partial",
                        "matches": result["matches"]
                    })
                else:
                    verification_results.append({
                        "entity": entity_name,
                        "type": entity_type,
                        "status": "not_present",
                        "matches": []
                    })
            
            if verification_results:
                clarification_messages = []
                for result in verification_results:
                    entity_name = result["entity"]
                    entity_type = result["type"].replace("View", "").strip()
                    
                    if result["status"] == "partial":
                        matches = result["matches"][:10]
                        match_list = "\n".join([f"- {m}" for m in matches])
                        clarification_messages.append(
                            f"There is no exact {entity_type} named '{entity_name}' in the database, "
                            f"but there are {len(result['matches'])} similar ones. Here are a few of them:\n\n"
                            f"{match_list}\n\nPlease clarify which one you are referring to."
                        )
                    else:
                        clarification_messages.append(
                            f"There is no {entity_type} named '{entity_name}' in the database."
                        )
                
                state["status"] = "entity_verification_needed"
                state["result"] = {
                    "verification_message": "\n\n".join(clarification_messages),
                    "original_query": question,
                    "needs_clarification": True,
                    "entity_info": {
                        "entities": entities,
                        "entity_types": entity_types,
                        "entity_mapping": {e: t for e, t in zip(entities, entity_types)},
                        "verified_entities": verified_entities,
                        "verification_results": verification_results
                    }
                }
                state["success_message"] = "Entity verification requires user clarification"
            else:
                state["status"] = "entity_verified"
                state["result"] = {
                    "original_query": question,
                    "needs_clarification": False,
                    "entity_info": {
                        "entities": entities,
                        "entity_types": entity_types,
                        "entity_mapping": {e: t for e, t in zip(entities, entity_types)},
                        "verified_entities": verified_entities
                    }
                }
                state["success_message"] = "All entities verified successfully"
            
            state["execution_time"] = time.time() - start_time
            return state
            
        except Exception as e:
            logger.error(f"Entity verification error: {e}", exc_info=True)
            state["status"] = "entity_verified"
            state["result"] = {
                "original_query": state["query"],
                "needs_clarification": False,
                "entity_info": {
                    "entities": [],
                    "entity_types": [],
                    "entity_mapping": {},
                    "verified_entities": {}
                }
            }
            state["success_message"] = "Entity verification skipped due to error"
            return state
    
    def _identify_entities_and_generate_sql(self, question: str) -> str:
        start_time = time.perf_counter()
        
        system_prompt = """You are an expert AI assistant responsible for identifying entity names in user questions and generating SQL queries to verify their existence.

DO NOT process campaign-related questions.

WHEN TO IDENTIFY ENTITIES:
- Specific names mentioned (e.g., "arman08", "Brown Tree", "Widget 2000")
- ALWAYS identify entities even if they look partial

WHEN NOT TO IDENTIFY:
- Campaign names or campaign-related queries
- Entity is an ID number
- Locations (cities, states, countries)
- Wildcard keywords: 'like', 'similar to', 'starting with'
- Generic types: "sku", "customers", "users"

CUBE.JS SQL SYNTAX (MANDATORY):
- Use fully qualified column names: TableName.columnName
- Use LIKE with % for case-insensitive matching
- Use lowercase in LIKE pattern for case-insensitive search
- Alias result as 'entityName'
- LIMIT 20 for search results

VERIFICATION SQL TEMPLATE:
SELECT TableName.name AS entityName 
FROM TableName 
WHERE TableName.name LIKE '%entity_name_lowercase%' 
LIMIT 20

TABLE MAPPING:
- SKU names: Sku.name
- Category names: Category.name
- Brand names: Brand.name
- Customer names: ViewCustomer.name
- User names: ViewUser.name
- Distributor names: ViewDistributor.name

RESPONSE FORMAT (JSON):
No entities:
{"question": "query"}

Single entity:
{
  "question": "query",
  "entities": ["arman08"],
  "entity_types": ["ViewCustomer"],
  "verification_sqls": ["SELECT ViewCustomer.name AS entityName FROM ViewCustomer WHERE ViewCustomer.name LIKE '%arman08%' LIMIT 20"]
}

Multiple entities: use arrays for all fields

EXAMPLES:

1. Customer "arman08":
{
  "question": "what is the sales for the customer arman08 for september month?",
  "entities": ["arman08"],
  "entity_types": ["ViewCustomer"],
  "verification_sqls": ["SELECT ViewCustomer.name AS entityName FROM ViewCustomer WHERE ViewCustomer.name LIKE '%arman08%' LIMIT 20"]
}

2. SKU "Widget":
{
  "question": "show sales for Widget product",
  "entities": ["Widget"],
  "entity_types": ["Sku"],
  "verification_sqls": ["SELECT Sku.name AS entityName FROM Sku WHERE Sku.name LIKE '%widget%' LIMIT 20"]
}

3. No entity:
{
  "question": "show top 5 customers by sales"
}

Database Schema:
Table: Sku (columns: id, name, categoryId, brandId)
Table: Category (columns: id, name)
Table: Brand (columns: id, name)
Table: ViewCustomer (columns: id, name, subType)
Table: ViewUser (columns: id, name, role, designation)
Table: ViewDistributor (columns: id, name)"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        response = self.llm.invoke(messages)
        answer = response.content.strip()
        
        track_llm_call(
            input_prompt=messages,
            output=answer,
            agent_type="entity_verification",
            operation="identify_entities",
            model_name=self.model_name
        )
        
        total_time = time.perf_counter() - start_time
        logger.info(f"Entity identification completed in {total_time:.2f} sec")
        
        return answer
    
    def _parse_llm_response(self, llm_response: str) -> Optional[Dict[str, Any]]:
        try:
            cleaned_response = llm_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            data = json.loads(cleaned_response)
            
            if "entities" not in data or not data["entities"]:
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response was: {llm_response}")
            return None
    
    def _verify_single_entity(self, entity_name: str, entity_type: str, verify_sql: str) -> Dict[str, Any]:
        try:
            logger.info(f"Verifying entity '{entity_name}' in {entity_type}")
            logger.info(f"SQL: {verify_sql}")
            
            result = self.db_executor(verify_sql)
            
            data = []
            if isinstance(result, dict):
                if 'data' in result:
                    data = result['data']
                elif 'results' in result and isinstance(result['results'], dict):
                    data = result['results'].get('data', [])
            
            logger.info(f"Query returned {len(data)} results")
            
            if not data:
                return {
                    "status": "not_present",
                    "matches": []
                }
            
            entity_lower = entity_name.strip().lower()
            exact_match = None
            all_matches = []
            
            for row in data:
                db_name = str(row.get('entityName', '')).strip()
                all_matches.append(db_name)
                
                if db_name.lower() == entity_lower:
                    exact_match = db_name
                    break
            
            if exact_match:
                logger.info(f"Exact match found: '{exact_match}'")
                return {
                    "status": "present",
                    "matches": [exact_match],
                    "db_name": exact_match
                }
            else:
                logger.info(f"No exact match, found {len(all_matches)} partial matches")
                return {
                    "status": "partial",
                    "matches": all_matches
                }
                
        except Exception as e:
            logger.error(f"Error verifying entity '{entity_name}': {e}", exc_info=True)
            return {
                "status": "not_present",
                "matches": []
            }
