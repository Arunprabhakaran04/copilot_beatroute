import os
import re
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from langchain.prompts import ChatPromptTemplate
from base_agent import BaseAgent, BaseAgentState, MeetingAgentState
from token_tracker import track_llm_call
from loguru import logger
from db_connection import DatabaseConnection

def date_str_to_epoch_ddmmyyyy(date_str: str) -> int:
    """Convert DD/MM/YYYY string to epoch time in milliseconds."""
    dt = datetime.strptime(date_str, "%d/%m/%Y")
    return int(time.mktime(dt.timetuple()) * 1000)  # Convert to milliseconds

class MeetingSchedulerAgent(BaseAgent):
    def __init__(self, llm, auth_token: str = None, user_id: str = None, session_id: str = None):
        super().__init__(llm)
        self.auth_token = auth_token or os.getenv("BEATROUTE_AUTH_TOKEN")
        self.user_id = user_id
        self.session_id = session_id
        self.db_connection = None
        
        # Initialize DB connection if session_id provided
        if session_id:
            try:
                self.db_connection = DatabaseConnection(session_id=session_id)
                logger.info(f"✅ Meeting agent initialized with DB connection for session: {session_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize DB connection: {e}")
    
    def get_agent_type(self) -> str:
        return "meeting"
    
    def process(self, state: BaseAgentState) -> MeetingAgentState:
        meeting_state = MeetingAgentState(**state)
        
        try:
            # Get conversation history from state for context
            conversation_history = state.get('conversation_history', [])
            chat_history_formatted = self._format_conversation_history(conversation_history)
            
            # Get intermediate results for additional context
            intermediate_results = state.get('intermediate_results', {})
            
            logger.info(f"📅 MEETING AGENT | Processing query: {state['query'][:100]}")
            
            # Step 1: Detect schedule intent
            intent_response = self.detect_schedule_intent(
                question=state["query"],
                chat_history=chat_history_formatted,
                _history={"get_history": lambda: self._build_history_for_intent(intermediate_results)}
            )
            
            logger.info(f"📋 MEETING AGENT | Intent detection response received")
            
            # Step 2: Process schedule intent
            result = self.schedule_visit(intent_response)
            
            # Step 3: Build response based on result
            if result.get("is_scheduling") == "no":
                # Not a scheduling request
                meeting_state["status"] = "failed"
                meeting_state["error_message"] = result.get("reason", "Not a scheduling request")
                logger.info(f"❌ MEETING AGENT | Not a scheduling request: {result.get('reason')}")
                return meeting_state
            
            # Check for errors
            if result.get("errors"):
                meeting_state["status"] = "failed"
                meeting_state["error_message"] = "; ".join(result["errors"])
                logger.error(f"❌ MEETING AGENT | Errors: {result['errors']}")
                return meeting_state
            
            # Check API result
            api_result = result.get("result", {})
            if "error" in api_result:
                meeting_state["status"] = "failed"
                meeting_state["error_message"] = f"API Error: {api_result['error']}"
                logger.error(f"❌ MEETING AGENT | API error: {api_result['error']}")
                return meeting_state
            
            # Success!
            request_data = result.get("request", {})
            meeting_state["status"] = "completed"
            meeting_state["parsed_date"] = request_data.get("assign_date", "")
            meeting_state["meeting_date"] = request_data.get("assign_date", "")
            
            # Build success message
            schedule_type = request_data.get("type", "visit")
            retailer_ids = request_data.get("retailer_id", [])
            assign_to = request_data.get("assign_to", self.user_id)
            assign_end_date = request_data.get("assign_end_date")
            
            success_msg = f"✅ {schedule_type.capitalize()} scheduled successfully!\n"
            success_msg += f"📅 Date: {meeting_state['parsed_date']}"
            if assign_end_date:
                success_msg += f" to {assign_end_date}"
            success_msg += f"\n👤 Assigned to: User {assign_to}"
            success_msg += f"\n🏪 Retailers: {len(retailer_ids)} customer(s)"
            
            if result.get("warnings"):
                success_msg += f"\n⚠️ Warnings: {'; '.join(result['warnings'])}"
            
            meeting_state["success_message"] = success_msg
            meeting_state["result"] = {
                "schedule_type": schedule_type,
                "assign_date": meeting_state['parsed_date'],
                "assign_end_date": assign_end_date,
                "assign_to": assign_to,
                "retailer_count": len(retailer_ids),
                "retailer_ids": retailer_ids,
                "api_response": api_result,
                "warnings": result.get("warnings", []),
                "agent_type": "meeting"
            }
            
            logger.success(f"✅ MEETING AGENT | {schedule_type.capitalize()} scheduled for {len(retailer_ids)} retailer(s)")
            
        except Exception as e:
            logger.exception(f"❌ MEETING AGENT | Error: {e}")
            meeting_state["status"] = "failed"
            meeting_state["error_message"] = f"Processing error: {str(e)}"
        
        return meeting_state
    
    def _format_conversation_history(self, conversation_history: List[Dict]) -> list:
        """Format conversation history for the LLM"""
        if not conversation_history:
            return []
        
        formatted = []
        for entry in conversation_history[-5:]:  # Last 5 conversations
            if isinstance(entry, dict):
                q = entry.get('question', '')
                r = entry.get('response', '')
                if q:
                    formatted.append(f"Q: {q}")
                if r:
                    formatted.append(f"A: {r}")
        return formatted
    
    def _build_history_for_intent(self, intermediate_results: dict):
        """Build history from intermediate results for intent detection"""
        history = []
        for step_key, step_data in intermediate_results.items():
            if isinstance(step_data, dict) and step_data.get('query'):
                query = step_data['query']
                result_data = step_data.get('query_data', [])
                if result_data:
                    # Convert to JSON string for LLM context
                    df_str = json.dumps(result_data[:5], indent=2)  # First 5 rows
                    history.append((query, "SELECT ...", df_str))
        return history
    
    def detect_schedule_intent(self, question: str, chat_history: list, **kwargs):
        """
        Determine whether the query is about scheduling a visit/call.
        Returns JSON with scheduling details or indication it's not a scheduling request.
        """
        now = datetime.now()
        current_date_str = now.strftime("%d/%m/%Y")
        tomorrow_date = (now + timedelta(days=1)).strftime("%d/%m/%Y")
        day_after_tomorrow = (now + timedelta(days=2)).strftime("%d/%m/%Y")
        next_week_same_day = (now + timedelta(days=7)).strftime("%d/%m/%Y")
        next_monday = now + timedelta(days=(7 - now.weekday()) % 7)
        if next_monday.date() == now.date():
            next_monday = now + timedelta(days=7)
        next_monday_str = next_monday.strftime("%d/%m/%Y")

        history_compact = "\n".join(chat_history) if chat_history else "No previous conversation"

        prompt = f"""
You are a STRICT parser for a scheduling system.

Goal:
1) Decide if the latest user query is a request to **schedule** a **visit** or **call**.
2) If YES, return the exact 5 DB fields needed.
3) If NO, clearly say it's not a scheduling request.

Use both the "Recent Chat History" and the "Latest Query" to resolve references like "these customers".

CURRENT DATE CONTEXT:
- Today: {current_date_str} ({now.strftime("%A")})
- Tomorrow: {tomorrow_date}
- Day after tomorrow: {day_after_tomorrow}
- Next week (same weekday): {next_week_same_day}
- Next Monday: {next_monday_str}

DATE RULES:
- "today" → {current_date_str}
- "tomorrow" → {tomorrow_date}
- "day after tomorrow" → {day_after_tomorrow}
- "next week"/"over the next week" → a range: start = tomorrow, end = {next_week_same_day}
- "next monday" → {next_monday_str}
- Written dates (e.g., "September 20, 2025") → convert to DD/MM/YYYY
- If single day → use assign_date, set assign_end_date = null
- If a range → set assign_date=start_date, assign_end_date=end_date
- If unclear date → leave assign_date=null and explain in ERROR

TYPE RULES:
- "call", "phone", "dial", "Zoom", "Meet", "Teams" → type="call"
- "visit", "in-person", "on-site", "meet at" → type="visit"
- "meeting" alone is ambiguous; infer from context; otherwise leave type=null

CUSTOMERS:
- Include customers explicitly named in the latest query.
- If "these customers" or similar, resolve from chat history.
- Do NOT invent names. If none can be resolved, return empty list.

USER ID:
- Detect from query or history. If not found, assign_to={self.user_id or "null"}.

STRICT OUTPUT POLICY:
Respond with ONLY ONE of the following JSON shapes (no extra text).

If this IS a scheduling request:
{{
  "is_scheduling": "yes",
  "payload": {{
    "assign_date": "<DD/MM/YYYY | null>",
    "assign_end_date": "<DD/MM/YYYY | null>",
    "assign_to": "<string or number | null>",
    "retailer_id": ["<id-or-name>", "..."],
    "type": "<visit|call|null>"
  }},
  "ERROR": "<empty string if OK, else short reason>"
}}

If this is NOT a scheduling request:
{{
  "is_scheduling": "no",
  "reason": "<short reason why this is not about scheduling>"
}}

Recent Chat History:
{history_compact}

Latest Query:
{question}
""".strip()

        # Add previous query results if available
        history_obj = kwargs.get("_history")
        if history_obj and callable(history_obj.get("get_history")):
            chat_data = history_obj["get_history"]()
            if chat_data:
                prompt += "\n\nThe following is previous query data for context:\n"
                for prev_q, sql, df_str in chat_data:
                    prompt += f"\n**Previous Question:** {prev_q}\n"
                    prompt += f"**Data Retrieved:** ```json\n{df_str}\n```\n"

        message_log = [{"role": "system", "content": prompt}, {"role": "user", "content": question}]

        try:
            response = self.llm.invoke(message_log)
            answer = response.content.strip()
            
            # Track token usage
            track_llm_call(
                input_prompt=message_log,
                output=answer,
                agent_type="meeting",
                operation="detect_schedule_intent",
                model_name="gpt-4.1-mini"
            )
            
            logger.info(f"🔍 MEETING AGENT | Intent detection completed")
            return answer
            
        except Exception as e:
            logger.exception(f"❌ MEETING AGENT | Intent detection error: {e}")
            return json.dumps({"is_scheduling": "no", "reason": f"Error: {str(e)}"})
    
    def get_customer_ids_list_with_customer_name(self, list_of_customer: List[str]) -> Optional[List[str]]:
        """
        Resolve customer names to IDs using database query.
        """
        if not list_of_customer:
            return None
        
        if not self.db_connection:
            logger.warning("⚠️ MEETING AGENT | No DB connection available for customer name resolution")
            return None
        
        try:
            # Build quoted list for SQL IN clause
            quoted_list = []
            for n in list_of_customer:
                s = str(n).strip()
                s = s.replace("'", "''")  # Escape single quotes
                quoted_list.append(f"'{s}'")
            
            if not quoted_list:
                return None
            
            quoted_names = ", ".join(quoted_list)
            
            query = f"""
                SELECT DISTINCT id, name
                FROM ViewCustomer
                WHERE ViewCustomer.name IN ({quoted_names})
            """
            
            logger.info(f"🔍 MEETING AGENT | Resolving {len(list_of_customer)} customer name(s) to IDs")
            
            result = self.db_connection.execute_query(query)
            
            if result.get('success') and result.get('data'):
                ids_list = [str(row['id']) for row in result['data']]
                logger.success(f"✅ MEETING AGENT | Resolved {len(ids_list)} customer ID(s)")
                return ids_list
            else:
                logger.warning(f"⚠️ MEETING AGENT | No customers found for names: {list_of_customer}")
                return []
                
        except Exception as e:
            logger.exception(f"❌ MEETING AGENT | Error resolving customer names: {e}")
            return None
    
    def create_schedule(self, token: str, assign_date: str, assign_end_date: str,
                        assign_to: str, retailer_ids: list, schedule_type: str = "visit"):
        """
        Create a schedule (visit/call) in BeatRoute via API.
        
        Args:
            token: Bearer token for authorization
            assign_date: Start date (DD/MM/YYYY)
            assign_end_date: End date (DD/MM/YYYY) or None
            assign_to: User ID (assigned to)
            retailer_ids: List of retailer IDs
            schedule_type: 'visit' or 'call'
            
        Returns:
            dict: API response JSON or error info
        """
        url = "https://env2.api.vwbeatroute.com/v1/schedule/create"
        
        # Convert date strings to epoch milliseconds
        start_ms = date_str_to_epoch_ddmmyyyy(assign_date)
        end_ms = date_str_to_epoch_ddmmyyyy(assign_end_date) if assign_end_date else start_ms
        
        # Convert IDs and types
        retailer_ids_str = [str(int(float(r))) for r in retailer_ids]
        assign_to_str = str(assign_to) if assign_to is not None else ""
        
        payload = {
            "assign_date": str(start_ms),
            "assign_end_date": str(end_ms),
            "assign_to": assign_to_str,
            "retailer_id": retailer_ids_str,
            "type": schedule_type,
            "agenda": ""  # Mandatory field
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"� MEETING AGENT | Called API for: {payload}")
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            logger.info(f"🔗 MEETING AGENT | Response status: {response.status_code}")
            
            try:
                resp_json = response.json()
                logger.info(f"📋 MEETING AGENT | Response JSON: {resp_json}")
                
                if response.status_code == 200:
                    logger.success(f"✅ MEETING AGENT | Schedule created successfully via API")
                    return resp_json
                else:
                    logger.error(f"❌ MEETING AGENT | API failed (status {response.status_code})")
                    return {"error": resp_json, "status_code": response.status_code}
                    
            except json.JSONDecodeError:
                logger.error(f"❌ MEETING AGENT | Invalid JSON response: {response.text}")
                return {"error": response.text, "status_code": response.status_code}
                
        except requests.exceptions.RequestException as e:
            logger.exception(f"⚠️ MEETING AGENT | API request error")
            return {"error": str(e)}
    
    def schedule_visit(self, answer: str):
        """
        Process the JSON response from detect_schedule_intent and create schedule via API.
        
        Returns:
            dict: Structured response with scheduling details or errors
        """
        def _is_ddmmyyyy(s: str) -> bool:
            try:
                datetime.strptime(s, "%d/%m/%Y")
                return True
            except Exception:
                return False
        
        warnings = []
        errors = []
        
        # Parse LLM output
        try:
            data = json.loads(answer)
        except Exception as e:
            logger.error(f"❌ MEETING AGENT | Invalid JSON from LLM: {e}")
            return {
                "error": "invalid_json_from_llm",
                "detail": str(e),
                "raw": answer
            }
        
        # Not a scheduling request
        if str(data.get("is_scheduling", "")).lower() == "no":
            return {
                "is_scheduling": "no",
                "reason": data.get("reason", "Not a scheduling request.")
            }
        
        # Extract payload
        payload = data.get("payload", {}) or {}
        
        assign_date = payload.get("assign_date")
        assign_end_date = payload.get("assign_end_date")
        assign_to = payload.get("assign_to")
        retailer_list_in = payload.get("retailer_id") or []
        schedule_type = payload.get("type")
        
        logger.info(f"📋 MEETING AGENT | Extracted: date={assign_date}, end={assign_end_date}, "
                   f"to={assign_to}, retailers={len(retailer_list_in)}, type={schedule_type}")
        
        # Validate dates
        if assign_date is not None and not _is_ddmmyyyy(assign_date):
            errors.append(f"assign_date is not DD/MM/YYYY: {assign_date}")
        if assign_end_date is not None and not _is_ddmmyyyy(assign_end_date):
            errors.append(f"assign_end_date is not DD/MM/YYYY: {assign_end_date}")
        
        # Type validation
        if schedule_type not in ("visit", "call", None):
            warnings.append(f"Unknown schedule type '{schedule_type}', defaulting to 'visit'.")
            schedule_type = "visit"
        if schedule_type is None:
            schedule_type = "visit"
        
        # assign_to fallback
        if not assign_to:
            assign_to = self.user_id
            if not assign_to:
                warnings.append("assign_to not provided and no user_id available.")
        
        # Resolve retailer IDs from names
        retailer_ids_final = []
        retailer_names = []
        
        for item in retailer_list_in:
            s = str(item).strip()
            if re.fullmatch(r"\d+", s):  # Pure numeric = ID
                retailer_ids_final.append(s)
            else:  # Treat as name
                retailer_names.append(s)
        
        logger.info(f"🏪 MEETING AGENT | IDs: {retailer_ids_final}, Names to resolve: {retailer_names}")
        
        # Resolve names to IDs
        if retailer_names:
            try:
                resolved_ids = self.get_customer_ids_list_with_customer_name(retailer_names) or []
                if not resolved_ids:
                    warnings.append(f"Could not resolve any IDs for names: {retailer_names}")
                retailer_ids_final.extend(map(str, resolved_ids))
                if len(resolved_ids) < len(retailer_names):
                    warnings.append(f"Some retailer names may be unresolved. Input: {retailer_names}, Resolved: {len(resolved_ids)}")
            except Exception as e:
                logger.exception(f"❌ MEETING AGENT | Name resolution error")
                errors.append(f"Name → ID resolution failed: {e}")
        
        # Deduplicate IDs
        seen = set()
        deduped_ids = []
        for rid in retailer_ids_final:
            if rid not in seen:
                seen.add(rid)
                deduped_ids.append(rid)
        retailer_ids_final = deduped_ids
        
        # If errors, return without API call
        if errors:
            return {
                "is_scheduling": "yes",
                "request": {
                    "assign_date": assign_date,
                    "assign_end_date": assign_end_date,
                    "assign_to": assign_to,
                    "retailer_id": retailer_ids_final,
                    "type": schedule_type
                },
                "warnings": warnings,
                "errors": errors
            }
        
        # Call BeatRoute API
        token = self.auth_token
        if not token:
            errors.append("No auth token available for API call")
            return {
                "is_scheduling": "yes",
                "request": {
                    "assign_date": assign_date,
                    "assign_end_date": assign_end_date,
                    "assign_to": assign_to,
                    "retailer_id": retailer_ids_final,
                    "type": schedule_type
                },
                "warnings": warnings,
                "errors": errors
            }
        
        logger.info(f"🆔 MEETING AGENT | Final retailer IDs: {retailer_ids_final}")
        
        result = self.create_schedule(
            token=token,
            assign_date=assign_date,
            assign_end_date=assign_end_date,
            assign_to=str(assign_to) if assign_to is not None else None,
            retailer_ids=retailer_ids_final,
            schedule_type=schedule_type
        )
        
        return {
            "is_scheduling": "yes",
            "request": {
                "assign_date": assign_date,
                "assign_end_date": assign_end_date,
                "assign_to": assign_to,
                "retailer_id": retailer_ids_final,
                "type": schedule_type
            },
            "warnings": warnings,
            "errors": errors,
            "result": result
        }
    
    def _parse_meeting_query(self, query: str, additional_context: str = "") -> Dict[str, Any]:
        """Legacy method - kept for backward compatibility but not used in new flow"""
        try:
            current_date = datetime.now()
            current_date_str = current_date.strftime("%d/%m/%Y")
            tomorrow_date = (current_date + timedelta(days=1)).strftime("%d/%m/%Y")
            day_after_tomorrow = (current_date + timedelta(days=2)).strftime("%d/%m/%Y")
            next_week_date = (current_date + timedelta(days=7)).strftime("%d/%m/%Y")
            next_monday = current_date + timedelta(days=(7 - current_date.weekday()) % 7)
            if next_monday == current_date:
                next_monday = current_date + timedelta(days=7)
            next_monday_str = next_monday.strftime("%d/%m/%Y")
            
            context_section = f"\n\nADDITIONAL CONTEXT FROM PREVIOUS STEPS:\n{additional_context}\n" if additional_context else ""
            
            parse_prompt = ChatPromptTemplate.from_template("""
            You are a query parser for a meeting scheduling system. Extract the user ID, meeting date, and topic from the following query.
            
            CURRENT DATE CONTEXT:
            - Today is: {current_date} ({current_day})
            - Tomorrow is: {tomorrow_date}
            - Day after tomorrow is: {day_after_tomorrow}
            - Next week (same day): {next_week_date}
            - Next Monday: {next_monday}
            
            Query: {query}
            {context_section}
            
            IMPORTANT: Respond ONLY in this EXACT format (no extra text):
            USER_ID: [number]
            USER_NAME: [name if mentioned, or "Unknown"]
            DATE: [DD/MM/YYYY format]
            TOPIC: [meeting topic/agenda if mentioned, or "General Discussion"]
            
            Rules for date conversion:
            1. "tomorrow" → {tomorrow_date}
            2. "day after tomorrow" → {day_after_tomorrow}
            3. "next week" → {next_week_date}
            4. "next monday" → {next_monday}
            5. "today" → {current_date}
            6. Convert written dates (like "September 20, 2025") to DD/MM/YYYY format
            7. Keep DD/MM/YYYY dates as they are
            8. If date is unclear, respond with: ERROR: [reason]
            
            Rules for user identification:
            - Extract user number from phrases like "user 3", "user3", "with user 5"
            - If additional context provides user information, use it
            - Extract name if mentioned (e.g., "with John", "schedule with Sarah")
            
            Rules for topic extraction:
            - Look for phrases like "regarding", "about", "to discuss", "for"
            - Extract the subject matter (e.g., "cost per click ad campaign", "Q1 review")
            - If no topic mentioned, use "General Discussion"
            """)
            
            messages = parse_prompt.format_messages(
                query=query,
                context_section=context_section,
                current_date=current_date_str,
                current_day=current_date.strftime("%A"),
                tomorrow_date=tomorrow_date,
                day_after_tomorrow=day_after_tomorrow,
                next_week_date=next_week_date,
                next_monday=next_monday_str
            )
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Track token usage
            track_llm_call(
                input_prompt=messages,
                output=content,
                agent_type="meeting",
                operation="schedule_meeting",
                model_name="gpt-4.1-mini"
            )
            
            if content.startswith("ERROR:"):
                return {"success": False, "error": content.replace("ERROR: ", "")}
            
            user_match = re.search(r'USER_ID:\s*(\d+)', content)
            user_name_match = re.search(r'USER_NAME:\s*(.+)', content)
            date_match = re.search(r'DATE:\s*(\d{2}/\d{2}/\d{4})', content)
            topic_match = re.search(r'TOPIC:\s*(.+)', content)
            
            if user_match and date_match:
                result = {
                    "success": True,
                    "user_id": user_match.group(1),
                    "meeting_date": date_match.group(1),
                    "parsed_date": date_match.group(1)
                }
                
                if user_name_match:
                    user_name = user_name_match.group(1).strip()
                    if user_name and user_name.lower() != "unknown":
                        result["user_name"] = user_name
                
                if topic_match:
                    topic = topic_match.group(1).strip()
                    if topic and topic.lower() != "general discussion":
                        result["meeting_topic"] = topic
                
                return result
            else:
                return {"success": False, "error": "Could not extract user ID or date"}
                
        except Exception as e:
            return {"success": False, "error": f"Parsing error: {str(e)}"}

