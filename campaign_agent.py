import json
import re
import logging
from typing import Optional, List, Dict
from langchain.prompts import ChatPromptTemplate
from base_agent import BaseAgent, BaseAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

class CampaignAgent(BaseAgent):
    def __init__(self, llm, campaign_info_table: Optional[List[Dict]] = None):
        super().__init__(llm)
        self.campaign_info_table = campaign_info_table or [
            {
                "campaign_id": 1,
                "campaign_name": "Holiday Sale",
                "response_table_name": "CampaignResponse1163",
                "is_table_available": "yes"
            },
            {
                "campaign_id": 2,
                "campaign_name": "Summer Promo",
                "response_table_name": None,
                "is_table_available": "no"
            },
            {
                "campaign_id": 3,
                "campaign_name": "Black Friday",
                "response_table_name": "CampaignResponseBF2025",
                "is_table_available": "yes"
            }
        ]
    
    def get_agent_type(self) -> str:
        return "campaign"
    
    def process(self, state: BaseAgentState) -> BaseAgentState:
        try:
            logger.info(f"CampaignAgent processing: '{state['query']}'")
            
            prompt = """
You are a smart assistant designed to determine whether the given question is related to a specific campaign.

Question may refer to a campaign name **explicitly or implicitly**, including:
- Partial or misspelled campaign names
- Case-insensitive mentions

You are provided with:
1. A **Campaign Info Table** — a dataframe with the following columns:
    - `campaign_id`
    - `campaign_name`
    - `response_table_name`
    - `is_table_available` (either "yes" or "no")
2. Additional knowledge that a table named `VmCampaign` exists in the database with the following columns:
    - campaign id
    - campaign name
    - start date 
    - end date
    - modification date, 
    - published date.

---

### 🔍 Your task:

For a given question:
- Step 1: Detect if it references a campaign (directly or indirectly).
- Step 2: Take the appropriate action based on the type of question and available data.

---

### 🎯 Logic:

- **If no campaign is referenced**:
    {{
        "complete_question": "<original question>."
    }}

- **If a campaign is referenced**:

    - If the question can be answered using the provided **Campaign Info Table** (campaign_name and campaign_id):
        {{
            "answer": "<your generated answer>"
        }}

    - If the question can be answered using the 'VmCampaign' table's data :
        - Example :  question - "Show me all the active campaigns"
        {{
            "complete_question": "<original question>.",
            "is_campaign": "yes"
        }}

    - If the question requires campaign-specific data:
    
        - Example : "How many 'Dealer Issues' were reported last month?"
        
        - Lets say 'XYZ' is the campaign name mentioned in the question. You must perform the following steps:
            1. Look up the campaign_name in the provided Campaign Info Table where the value matches 'XYZ'. DO NOT USE ANY OTHER ROW
            2. From that row, read the values of:
                - is_table_available
                - response_table_name

        - If `is_table_available` is "yes":
            - Use the exact value of response_table_name from that row in your output (NOTE : if 'is_table_available' is "yes" then 'response_table_name' will not be 'None').
            - Do NOT output placeholders like ACTUAL_TABLE_NAME or CAMPAIGN_NAME — replace them with the real values from the row of the table.
            - Format the response as:
            {{
                "complete_question": "<original question>. Refer to table ACTUAL_TABLE_NAME which has data exclusively for campaign CAMPAIGN_NAME.",
                "is_campaign": "yes"
            }}
        - If `is_table_available` is "no":
            {{
                "answer": "There is no valid campaign data."
            }}

---

### 📌 Campaign Info Table:
{campaign_table}

### ⚠️ Output Policy:
- Respond with ONLY valid JSON in one of the exact formats shown above.
- Do **not** include any explanatory text before or after the JSON.
- Do **not** hallucinate or fabricate campaign IDs or table names.
- For questions asking for campaign ID, name, or basic info, use the "answer" format.

Question: {question}
"""
            
            campaign_prompt = ChatPromptTemplate.from_template(prompt)
            messages = campaign_prompt.format_messages(
                campaign_table=str(self.campaign_info_table),
                question=state["query"]
            )
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Track token usage
            track_llm_call(
                input_prompt=messages,
                output=content,
                agent_type="campaign",
                operation="process_campaign_query",
                model_name="gpt-4.1-mini"
            )
            
            logger.info(f"Campaign LLM response: {content}")
            
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
                    result_data = json.loads(json_content)
                else:
                    logger.warning(f"No JSON found in campaign response: {content}")
                    state["status"] = "campaign_processed"
                    state["result"] = {"source": "campaign_agent_no_json"}
                    return state
                
                logger.info(f"Campaign JSON parsed: {result_data}")
                
                if "answer" in result_data:
                    state["status"] = "completed"
                    state["agent_type"] = "campaign"
                    state["success_message"] = "Campaign query answered directly"
                    state["result"] = {
                        "answer": result_data["answer"],
                        "source": "campaign_agent_direct"
                    }
                    logger.info(f"Campaign direct answer: {result_data['answer']}")
                    return state
                elif "complete_question" in result_data:
                    enriched_query = result_data["complete_question"]
                    state["query"] = enriched_query
                    state["status"] = "campaign_processed"
                    state["result"] = {
                        "original_query": state.get("original_query", state["query"]),
                        "enriched_query": enriched_query,
                        "is_campaign": result_data.get("is_campaign", "no"),
                        "source": "campaign_agent_enrichment"
                    }
                    logger.info(f"Campaign enriched query: {enriched_query}")
                    return state
                else:
                    state["status"] = "campaign_processed"
                    state["result"] = {"source": "campaign_agent_passthrough"}
                    logger.warning("Campaign agent: unexpected JSON format, passing through")
                    return state
                    
            except json.JSONDecodeError as e:
                logger.error(f"Campaign JSON decode error: {e}, content: {content}")
                state["status"] = "campaign_processed"
                state["result"] = {"source": "campaign_agent_json_error"}
                return state
                
        except Exception as e:
            logger.error(f"Campaign analysis error: {str(e)}")
            state["error_message"] = f"Campaign analysis error: {str(e)}"
            state["status"] = "campaign_processed"  
            return state