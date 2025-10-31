import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)


class EnrichAgent:
    """
    AI-powered query enrichment agent that replaces heuristic-based enrichment.
    
    This agent maintains its own message log to track conversation flow and can return:
    - complete_question: Enriched query ready for downstream agents
    - follow_up: Question to ask user for clarification
    - answer: Direct response to user (e.g., friendly greeting, answer about previous result)
    """
    
    def __init__(
        self, 
        openai_client: OpenAI,
        schema_manager=None,
        sql_retriever_agent=None,
        campaign_table: Optional[str] = None,
        campaign_custom_map: Optional[Dict] = None,
        user_role_list: Optional[List] = None,
        user_designation_list: Optional[List] = None,
        customer_subtype_list: Optional[List] = None,
        user_context=None,  # NEW: Accept UserContext object
        model: str = "gpt-4o",
        max_tokens: int = 2000,
        temperature: float = 0.3
    ):
        """
        Initialize EnrichAgent with necessary components.
        
        Args:
            openai_client: OpenAI client instance
            schema_manager: SchemaManager instance for focused schema generation
            sql_retriever_agent: SQLRetrieverAgent for similar questions
            campaign_table: Campaign table data for campaign context (deprecated, use user_context)
            campaign_custom_map: Custom field mapping for campaigns (deprecated, use user_context)
            user_role_list: List of user roles (deprecated, use user_context)
            user_designation_list: List of user designations (deprecated, use user_context)
            customer_subtype_list: List of customer subtypes (deprecated, use user_context)
            user_context: UserContext object containing all user metadata (recommended)
            model: OpenAI model to use
            max_tokens: Maximum tokens for response
            temperature: Temperature for LLM
        """
        self.client = openai_client
        self.schema_manager = schema_manager
        self.sql_retriever = sql_retriever_agent
        
        # NEW: Store user_context reference
        self.user_context = user_context
        
        # Backward compatibility: Use parameters if provided, otherwise use user_context
        if user_context is not None:
            self.campaign_table = user_context.campaign_table
            self.campaign_custom_map = user_context.campaign_custom_map
            self.user_role_list = user_context.user_role_list
            self.user_designation_list = user_context.user_designation_list
            self.customer_subtype_list = user_context.customer_subtype_list
            logger.info("EnrichAgent initialized with UserContext metadata")
        else:
            # Fallback to individual parameters (backward compatibility)
            self.campaign_table = campaign_table or ""
            self.campaign_custom_map = campaign_custom_map or {}
            self.user_role_list = user_role_list or []
            self.user_designation_list = user_designation_list or []
            self.customer_subtype_list = customer_subtype_list or []
            logger.warning("EnrichAgent initialized without UserContext, using individual parameters")
        
        self.config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Message log per session to maintain conversation state
        # Format: {session_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        self.session_message_logs: Dict[str, List[Dict[str, str]]] = {}
        
        logger.info(f"EnrichAgent initialized with model: {model}")
    
    def update_user_context(self, user_context) -> None:
        """
        Update EnrichAgent with new UserContext metadata.
        
        This method allows updating the agent with user-specific metadata
        after initialization, which is useful when UserContext is loaded
        after the agent is created.
        
        Args:
            user_context: UserContext object containing all user metadata
        """
        if user_context is None:
            logger.warning("Attempted to update EnrichAgent with None UserContext")
            return
        
        self.user_context = user_context
        
        # Update all metadata from user_context
        self.campaign_table = user_context.campaign_table
        self.campaign_custom_map = user_context.campaign_custom_map
        self.user_role_list = user_context.user_role_list
        self.user_designation_list = user_context.user_designation_list
        self.customer_subtype_list = user_context.customer_subtype_list
        
        # Also update schema_manager if available
        if user_context.is_schema_loaded():
            self.schema_manager = user_context.get_schema_manager()
        
        logger.info(f"✅ EnrichAgent updated with UserContext metadata:")
        logger.info(f"   - User roles: {len(self.user_role_list)}")
        logger.info(f"   - User designations: {len(self.user_designation_list)}")
        logger.info(f"   - Customer subtypes: {len(self.customer_subtype_list)}")
        logger.info(f"   - Campaign tables: {len(self.campaign_custom_map)}")
    
    
    def enrich_query(
        self,
        session_id: str,
        user_id: str,
        original_query: str,
        conversation_history: List[Dict[str, Any]],
        user_name: Optional[str] = None,
        email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main enrichment method that replaces redis_memory_manager.enrich_query().
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            original_query: The raw user query
            conversation_history: Recent conversation history from RedisMemoryManager
            user_name: User's name for personalization
            email: User's email
            
        Returns:
            Dict with one of three keys:
            - {"complete_question": str}: Enriched query ready for execution
            - {"follow_up": str}: Question to ask user
            - {"answer": str}: Direct answer to user
        """
        start_time = time.perf_counter()
        logger.info(f"ENRICH AGENT STARTED for query: '{original_query}'")
        
        # Build system prompt
        system_prompt = self._build_system_prompt(user_name, email, user_id)
        
        # Get conversation history context
        history_context = self._format_conversation_history(conversation_history)
        
        # Get similar questions for context
        similar_questions = []
        if self.sql_retriever:
            try:
                similar_questions = self.sql_retriever.get_top_k_similar_questions(
                    original_query, k=20
                )
                logger.info(f"Retrieved {len(similar_questions)} similar questions")
            except Exception as e:
                logger.warning(f"Failed to get similar questions: {e}")
        
        # Get focused schema if available
        schema_context = ""
        if self.schema_manager and similar_questions:
            try:
                schema_context = self.schema_manager.get_schema_to_use_in_prompt(
                    current_question=original_query,
                    list_similar_question_sql_pair=similar_questions,
                    k=10
                )
                logger.info(f"Retrieved focused schema ({len(schema_context)} chars)")
            except Exception as e:
                logger.warning(f"Failed to get focused schema: {e}")
        
        # Add schema and history to system prompt
        if schema_context:
            system_prompt += (
                "\n\nThe below additional context is the list of schemas of various tables of the database. "
                "Remember that the user is trying to ask questions to get data from these tables.\n"
                f"{schema_context}"
            )
        
        if self.user_role_list:
            system_prompt += (
                "\n\nThe below list contains distinct roles that different users can have. "
                "Remember that questions can mention the role of a user (ViewUser.role).\n"
                f"{self.user_role_list}"
            )
        
        if self.user_designation_list:
            system_prompt += (
                "\n\nThe below list contains distinct designations that different users can have. "
                "Remember that questions can mention the designation of a user (ViewUser.designation).\n"
                f"{self.user_designation_list}"
            )
        
        if self.customer_subtype_list:
            system_prompt += (
                "\n\nThe below list contains distinct subtypes that a customer can have. "
                "Remember that questions can mention the subtype of a customer (ViewCustomer.subType).\n"
                f"{self.customer_subtype_list}"
            )
        
        if history_context:
            system_prompt += (
                "\n\nThe following is a list of previously asked questions along with their SQL queries and results. "
                "Use this context to better understand the current question and maintain consistency.\n"
                f"{history_context}"
            )
        
        if similar_questions:
            similar_q_str = json.dumps(similar_questions, indent=2)
            system_prompt += (
                "\n\nThe following is a list of potentially similar questions with their SQL queries. "
                "Use this context to understand the current question better. "
                "If there is an exact word-by-word match of the asked question in the similar questions, "
                "do not ask any follow-up question.\n"
                f"Similar Questions: {similar_q_str}"
            )
        
        system_prompt += "\n\nUse all this information to better interpret the current question."
        
        # Initialize or retrieve message log for this session
        if session_id not in self.session_message_logs:
            self.session_message_logs[session_id] = []
            logger.info(f"Created new message log for session: {session_id}")
        
        # Build messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history from this session
        messages.extend(self.session_message_logs[session_id])
        
        # Add current user query
        messages.append({"role": "user", "content": original_query})
        
        # Check for exact match in similar questions (skip LLM call)
        if self._has_exact_match(original_query, similar_questions):
            logger.info("Exact match found in similar questions. Skipping enrichment.")
            complete_response = {"complete_question": original_query}
            
            # Update message log
            self.session_message_logs[session_id].append({"role": "user", "content": original_query})
            self.session_message_logs[session_id].append({
                "role": "assistant", 
                "content": json.dumps(complete_response)
            })
            
            # Trim message log if too long (keep last 10 messages)
            if len(self.session_message_logs[session_id]) > 10:
                self.session_message_logs[session_id] = self.session_message_logs[session_id][-10:]
            
            return complete_response
        
        # Call LLM for enrichment
        try:
            t_api_start = time.perf_counter()
            
            params = {
                "model": self.config["model"],
                "messages": messages,
                "n": 1,
                "stop": None
            }
            
            if "gpt-5" in self.config["model"]:
                params["max_completion_tokens"] = self.config["max_tokens"]
            else:
                params["max_tokens"] = self.config["max_tokens"]
                params["temperature"] = self.config["temperature"]
            
            response = self.client.chat.completions.create(**params)
            
            t_api_end = time.perf_counter()
            logger.info(f"LLM API call completed in {t_api_end - t_api_start:.2f}s")
            
            # Extract response
            answer = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                parsed_response = json.loads(answer)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {answer}")
                # Fallback: return original query
                parsed_response = {"complete_question": original_query}
            
            # Validate response format
            if not isinstance(parsed_response, dict):
                logger.error(f"LLM response is not a dict: {parsed_response}")
                parsed_response = {"complete_question": original_query}
            
            # Ensure only valid keys are present
            valid_keys = {"complete_question", "follow_up", "answer"}
            if not any(key in parsed_response for key in valid_keys):
                logger.error(f"LLM response missing valid keys: {parsed_response}")
                parsed_response = {"complete_question": original_query}
            
            # Update message log with user query and assistant response
            self.session_message_logs[session_id].append({"role": "user", "content": original_query})
            self.session_message_logs[session_id].append({
                "role": "assistant",
                "content": json.dumps(parsed_response)
            })
            
            # Trim message log if too long (keep last 10 messages = 5 turns)
            if len(self.session_message_logs[session_id]) > 10:
                self.session_message_logs[session_id] = self.session_message_logs[session_id][-10:]
            
            total_time = time.perf_counter() - start_time
            logger.info(f"ENRICH AGENT COMPLETED in {total_time:.2f}s")
            logger.info(f"Response type: {list(parsed_response.keys())[0]}")
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"LLM enrichment failed: {e}", exc_info=True)
            # Fallback: return original query
            fallback_response = {"complete_question": original_query}
            
            # Still update message log
            self.session_message_logs[session_id].append({"role": "user", "content": original_query})
            self.session_message_logs[session_id].append({
                "role": "assistant",
                "content": json.dumps(fallback_response)
            })
            
            return fallback_response
    
    
    def _build_system_prompt(self, user_name: Optional[str], email: Optional[str], user_id: str) -> str:
        """Build comprehensive system prompt for enrichment agent."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        prompt = f"""System Instructions:
You are an expert data analyst AI assistant responsible for understanding user questions completely step by step.
Think and ask relevant questions.
STRICTLY respond with DIRECTLY PARSABLE JSON string only.
DO NOT RESPOND IN FORMAT json or as a JSON code block.
Today's date is {current_date}. Talk politely like a person and greet back as a follow up when there is a greeting.
Never ask to provide data or table name.

IMPORTANT CAPABILITIES:
- You CAN execute SQL queries to retrieve data from the database.
- You CAN create visualizations (charts, graphs, plots) from query results.
- You CAN generate summaries and insights from data.
- When user asks for visualization of previous results, acknowledge that you can do it and rephrase as a complete_question like: "Create a bar chart showing the top 3 SKUs based on sales value from the previous query results."
"""
        
        if user_name:
            prompt += f"\n\nYou are talking to {user_name} with user id {user_id}. "
            prompt += "The final understood question should have user name or id if the user is referring to himself using pronoun. "
            prompt += "Also add 'user' before the user's name. Please have a friendly conversation and can use user's first name."
        
        if email:
            prompt += f"\n\nThe email id of the user {user_name} is {email}. "
            prompt += f"So if the user has asked to send an email just reply that you are sending an email to {email}"
        
        if self.campaign_table:
            prompt += (
                "\n\nThe table below contains campaign 'id', 'name', 'startDate', 'endDate', 'publishedOn' columns of all campaigns.\n"
                f"{self.campaign_table}\n"
            )
            
            prompt += """
Use the following campaign-related context to understand user questions more accurately:

- Question may refer to a campaign by its name, with or without the word "campaign", and possibly with spelling or casing variations.
- There might be multiple campaigns with the same name but they will have different ids.
- Your task is to:
    - Think carefully and ask a follow up if the question is mentioning a campaign name. ASK only if relevant to campaigns.
    - If it is a campaign:
        - See if there is an exact match (ignoring case) of campaign name in the campaign map.
        - If there are multiple campaigns with the same exact name but different ids and other information (ids, startDate, endDate, and publishedOn), 
          follow up with the ids, startDate, endDate, and publishedOn (information present in the campaign table provided) of the duplicate campaigns 
          and ask for which among these should be considered.
          Example:
          'There are multiple campaigns named "XYZ" with the following details:
          1. (id: 123, startDate: 2024-01-01, endDate: 2024-03-31, publishedOn: 2023-12-15)
          2. (id: 456, startDate: 2024-04-01, endDate: 2024-06-30, publishedOn: 2024-03-20)
          Please confirm which one you would like to use.'
        - If match then:
            - In the interpreted version of the question (i.e., "complete_question"), add a line in the format: 
              "<campaign_name> is campaign name" to make it clear that the name refers to a campaign.
            - Example: "How many farm trials were completed?" becomes 
              "How many Farm Trial were completed?. Farm Trial is campaign name with id <ID>."
        - If partial match then:
            - Don't assume campaign name from partial match, ask follow up question and try to get the exact campaign name by showing names like the name mentioned.
    - Do not ask for table context. Each campaign has its data in a table named CampaignResponse<ID> where <ID> is the campaign ID.
"""
        
        if self.campaign_custom_map:
            map_str = "\nCustom Field Short Titles by Table for Campaigns:\n"
            for table, cols in self.campaign_custom_map.items():
                map_str += f"\n{table}:\n"
                for col, short_title in cols.items():
                    map_str += f"  - {col} to {short_title}\n"
            
            prompt += f"""
The following mapping provides context for custom fields in your database.
If a question refers to a "short title" (human-friendly description), use this mapping to resolve which field/column it relates to.

{map_str}
"""
        
        prompt += """
Logical Steps for AI Assistant to Follow in Understanding the User's Question:

Step 0: Prefer Default Assumptions When Safe
    - If the user's question is slightly ambiguous but a logical default can be assumed (e.g., "overall comparison" if not specified), proceed with the default.
    - Only ask a follow-up if the ambiguity significantly affects the outcome or could lead to a materially different answer.
    - Maintain consistency in time frames and filters based on prior conversation.
    - If time frame cannot be inferred then ask for timeframe.
    - CRITICAL DEFAULT: When user says 'sales' without specifying primary/secondary, ALWAYS default to 'secondary sales' (customer dispatch from CustomerInvoice table). Add this clarification in the complete_question.
    - Examples:
      - "Compare the orders of last 3 months" means Assume "overall" comparison unless context or prior question indicates otherwise.
      - "Show me sales decline this month" means Assume comparison against "last month" unless another baseline is specified.
      - "What is the monthly sales for last 3 months?" means "What is the monthly secondary sales (customer dispatch) for last 3 months?"
    - Avoid follow-ups if a clear industry standard or previously used default exists.

Step 1: Handle Requests About Previous Results
    - If the question is asking for visualization, chart, graph, or plot of previous results:
      - Check if there are previous results available in chat history
      - If yes, return complete_question in format: "Create a [chart_type] visualization showing [what to visualize] from the previous query results"
      - Example: User asks "can you provide a visualization?" → complete_question: "Create a bar chart showing the top 3 SKUs based on sales value from the previous query results"
    - If the question is about the previous asked questions and can be answered using the results of the previous question:
      - Answer in the format { "answer" : "answer to the question about previous question" }
    
Step 2: Infer Information from Chat History
    - Retrieve the chat history to check if the current question is a follow-up or depends on previous responses.
    - Infer the time frame if the user is referring to prior queries (e.g., "What about last quarter?" should inherit the last known time frame).
    - Check for references to previous data in system context and provide a meaningful continuation.
    
Step 3: Handling Context from Previous Questions
    - If the current question references or extends a previous question, update it by incorporating relevant data from previously asked question before generating the final question.
    - Steps to Follow:
        - Identify if the current question depends on a previous question's result.
        - Extract the relevant data from the data of previously asked question.
        - Replace placeholders in the new question (e.g., "bottom 2 customers") with actual values from the previous answer.
        - Ensure consistency in the time frame and filters.
    - Example Transformation:
        Previous Question: "Show me the bottom 2 customers in the last 3 months."
        Previous Answer (Data): "Customer1" and "Customer2"
        Current Question: "Show me the SKUs purchased by these customers."
        Complete Question (Final Output): "Show me the SKUs purchased by customers Customer1 and Customer2 in the last 3 months."
    - Key Rules:
        - If the question refers to previously retrieved data, replace general terms (e.g., "bottom 2 customers") with their specific values from past responses 
          while generating the complete question.
        - If the previous question explicitly requested ranked or filtered data, extract the exact values.
        - Include information about the extracted value in the complete question. E.g. customer 'Customer1' and 'Customer2'
        - Maintain consistency in time frames and filters based on prior queries.
        - If the current question does not reference past results, process it independently.
        - If there is no relevant previous data, process the question independently.
        - If the reference is unclear, ask a follow-up question before finalizing.
        
Step 4: Detect Ambiguities and Ask Follow-up Questions
    - If ambiguity exists but a reasonable and low-risk default can be assumed, prefer to answer directly using that default.
    - DO NOT ASK for clarification if the question implies a standard answer (e.g., "overall" when no dimension is specified).
    - Examples:
      - User: "Compare orders of last 3 months" means Default to "overall" comparison unless chat history implies category/brand-level view.
    - DO NOT ASK QUESTIONS about:
      - 'lines sold' or 'lines ordered' (see existing rules)
      - Obvious or auto-resolvable references (e.g., "this city", "Delhi", "last month")
    - Only ask for clarification when:
      - The ambiguity affects the logic or structure of the query
      - Multiple equally valid interpretations exist and the wrong one might mislead
    - Some information:
        - 'lines' refers to unique count of skus.
        - 'lines sold' refers to unique count of skus sold.
        - 'sales' metric refer to total sales value unless some categorization is mentioned E.g. monthly sales, sku wise sales etc.
        - IMPORTANT: When user says 'sales' without specifying, it means 'Secondary sales' (customer sales/dispatch).
        - 'Primary sales' is distributor sales (DistributorSales table).
        - 'Secondary sales' is customer sales or dispatch or sales (CustomerInvoice table with dispatchedvalue).
        - If user asks for 'sales', clarify in the complete_question: 'secondary sales (customer dispatch)' to avoid confusion.
        - SR or Sales Rep is a user whose 'role' is 'Sales Representative'
        - A week is defined from Monday to Sunday unless otherwise mentioned
        - There are two kinds of id of a user:
            - 'user id' or 'br id' or 'beatroute id' maps to the 'id' column of ViewUser table
            - 'external id' or 'sap' maps to 'external_id' column of ViewUser table
        - Productive call/visit means a sales visit that resulted in an order.
        - Range sold is total number of unique sku sold.
        - Offtake / Sellout means total quantity recorded in log sales module on customers.
        - 'Total calls' means total customer visits.
        - Unique outlets billed is number of unique customers with sales.
    - Refer to the schema of tables provided in the additional context.
    - FOR Analysis or Analyse related questions do not ask questions about metrics.
    - Ask relevant questions.
        
Step 5: Ask for timeframe if cannot infer from the running conversation and is needed
    - DO NOT ASK question for confirmation of 'this month' or 'last month' or any other such examples
        - Mention of 'this month' should be interpreted as the current calendar month and similarly for year
        - Mention of 'last month' in the question should be considered as the previous calendar month.
    - Conditions to ask for a time frame:
        - No time frame is explicitly mentioned in the user's query.
        - It cannot be inferred from chat history.
        - The question logically requires a time frame, e.g.:
            - "Show me the list of all SKUs" (No time frame needed)
            - "Show me the SKUs sold" (Needs a time frame)
            - "Total orders placed" (Needs a time frame)
            - "Total sales by category" (Needs a time frame)
            - "Total active users" (No time frame needed)
    - If Needed, Include the time frame in the final understood question.
    - For degrowth, decline of performance wrt a metric related questions there should be two timeframes. Ask for the time frame to compare against if not provided explicitly.
        - If the question mentions only one time frame, Ask for the timeframe to compare against with some suggestions.
          Example - if the question was 'Which customer's sales declined this month.' ask should i compare it against previous month
        
Step 6: Identify Word and Map to Entity Type
    - DO NOT Map generic words to any Entity Type.
        - Example: 'any Sku', 'customers', 'this Category', 'this campaign'
    - Include the information of the mapped entity in the understood question.
        - Add sku before sku_name
        - Add customer before customer_name
        - Add campaign before campaign_name, campaign names are provided in the campaign map along with id.
    - Map the unknown word in the question to entity type:
        - Location names (Automatically recognize without asking for confirmation).
            - Delhi is a City
            - Haryana is a State
            - India is a Country
        - Other potential entity types:
            - SKU name
            - Category name
            - User name
            - Customer name
            - Distributor name
            - Brand name
            - Campaign name
    - If a word is unknown or ambiguous and is not clear what its entity type is, ask for clarification:
        - If word is not associated with an entity type, ask the user what it represents (e.g., SKU, category, campaign or distributor).
        - Do NOT ask obvious questions like "Is Delhi a city?" cities should be auto-recognized.
        - If entity has wildcard characters (%, _), assume the user is asking for similar entity and add this information to the complete question.
        
Step 7: Understand the complete Question:
    - Check If a Follow-Up Response Exists:
       - Look at the chat history.
       - If the user has provided a follow-up response, integrate it into the original question.
       - Preserve the original casing of entity names (e.g., "HERBICIDE", "Ready To Eat", "SK123") exactly as mentioned in the asked question when constructing the complete_question.
    - Combine Original Question and Follow-Up Response:
       - Ensure that the follow-up response clarifies missing information from the original question.
       - Modify the original question accordingly.
       - Example:
         Original Question: "What is the total sales of Ready to Eat in the last 3 months?"
         Follow-up Question: "Please clarify what 'ready to eat' refers to. Is it a specific category or SKU? If it's a category, please mention the category name."
         Follow-Up Response: "Category name is 'ready to eat'."
         complete_question: What is the total sales of the category 'ready to eat' in the last 3 months?
    
Step 8: Ensure JSON Output is Well-Structured
    - Use double quotes for all keys and string values.
    - The result must be valid for json.loads(response).
    - Example of correct output: {"complete_question": "What is the sales trend?"}
    - Do not use single quotes anywhere in the response.
    - Do not return a Python dict like {'follow_up': 'value'}.
    - If answering a response to the previous question or giving friendly reply, return:
        { "answer" : "answer to the question about previous question or a friendly reply" }
    - If clarification is needed, return:
        { "follow_up": "Follow-up question for missing details" }
    - If all required details are present of the question asked, return a complete question:
        {"complete_question": "Final understood question"}
    - DO NOT USE ANY OTHER KEY IN THE JSON RESPONSE OTHER THAN WHAT IS MENTIONED HERE.
"""
        
        return prompt
    
    
    def _format_conversation_history(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Format conversation history from RedisMemoryManager into readable string."""
        if not conversation_history:
            return ""
        
        context_lines = []
        for i, entry in enumerate(reversed(conversation_history)):
            priority_marker = "[MOST RECENT]" if i == 0 else f"[{i} ago]"
            
            original = entry.get("original", "")
            enriched = entry.get("enriched", "")
            result = entry.get("result", {})
            
            # Extract SQL and data if available
            sql = ""
            data_str = ""
            
            if isinstance(result, dict):
                sql = result.get("sql", "")
                data = result.get("data", [])
                
                # Truncate data for context (max 3 rows)
                if isinstance(data, list) and len(data) > 0:
                    data_sample = data[:3]
                    data_str = json.dumps(data_sample, indent=2)
                    if len(data) > 3:
                        data_str += f"\n... ({len(data) - 3} more rows)"
            
            context_lines.append(
                f"{priority_marker} Question: {original}\n"
                f"SQL: {sql}\n"
                f"Data: {data_str}\n"
            )
        
        return "\n".join(context_lines)
    
    
    def _has_exact_match(self, query: str, similar_questions: List[Dict]) -> bool:
        """Check if query has exact match in similar questions."""
        if not similar_questions:
            return False
        
        query_normalized = query.strip().lower()
        
        for sq in similar_questions:
            sq_text = sq.get("question", "").strip().lower()
            if query_normalized == sq_text:
                return True
        
        return False
    
    
    def clear_session(self, session_id: str):
        """Clear message log for a session."""
        if session_id in self.session_message_logs:
            del self.session_message_logs[session_id]
            logger.info(f"Cleared message log for session: {session_id}")
