import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from base_agent import BaseAgent, BaseAgentState, MeetingAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

class MeetingSchedulerAgent(BaseAgent):
    def __init__(self, llm, files_directory: str = "./user_files"):
        super().__init__(llm)
        self.files_directory = files_directory
        os.makedirs(self.files_directory, exist_ok=True)
    
    def get_agent_type(self) -> str:
        return "meeting"
    
    def process(self, state: BaseAgentState) -> MeetingAgentState:
        meeting_state = MeetingAgentState(**state)
        
        # Check for intermediate_results that might contain user/participant info
        additional_context = ""
        intermediate_results = state.get('intermediate_results', {})
        if intermediate_results:
            logger.info(f"📅 Meeting agent found {len(intermediate_results)} previous step(s)")
            additional_context = self._extract_context_from_intermediate_results(intermediate_results)
        
        parsed_data = self._parse_meeting_query(state["query"], additional_context)
        if not parsed_data["success"]:
            meeting_state["error_message"] = parsed_data["error"]
            meeting_state["status"] = "failed"
            return meeting_state
        
        meeting_state["user_id"] = parsed_data["user_id"]
        meeting_state["user_name"] = parsed_data.get("user_name", "")
        meeting_state["meeting_date"] = parsed_data["meeting_date"]
        meeting_state["parsed_date"] = parsed_data["parsed_date"]
        meeting_state["meeting_topic"] = parsed_data.get("meeting_topic", "")
        
        if self._validate_meeting_data(meeting_state):
            self._write_meeting_to_file(meeting_state)
        
        return meeting_state
    
    def _parse_meeting_query(self, query: str, additional_context: str = "") -> Dict[str, Any]:
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
                model_name="gpt-4o"
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
    
    def _validate_meeting_data(self, state: MeetingAgentState) -> bool:
        try:
            if not state["user_id"] or int(state["user_id"]) <= 0:
                state["error_message"] = "Invalid user ID"
                state["status"] = "failed"
                return False
            
            parsed_date = datetime.strptime(state["meeting_date"], "%d/%m/%Y")
            if parsed_date.date() < datetime.now().date():
                state["error_message"] = "Cannot schedule meeting in the past"
                state["status"] = "failed"
                return False
            
            return True
        except Exception as e:
            state["error_message"] = f"Validation error: {str(e)}"
            state["status"] = "failed"
            return False
    
    def _write_meeting_to_file(self, state: MeetingAgentState):
        try:
            filename = f"user{state['user_id']}.txt"
            file_path = os.path.join(self.files_directory, filename)
            state["file_path"] = file_path
            
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(f"=== User {state['user_id']} Meeting Schedule ===\n\n")
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build meeting entry with all available details
            meeting_entry = f"Meeting on date {state['parsed_date']}"
            if state.get('meeting_topic'):
                meeting_entry += f" - Topic: {state['meeting_topic']}"
            if state.get('user_name'):
                meeting_entry += f" - With: {state['user_name']}"
            meeting_entry += f" (Scheduled on: {timestamp})\n"
            
            with open(file_path, 'a') as f:
                f.write(meeting_entry)
            
            state["status"] = "completed"
            
            # Build success message
            success_msg = f"Meeting scheduled with User {state['user_id']}"
            if state.get('user_name'):
                success_msg = f"Meeting scheduled with {state['user_name']} (User {state['user_id']})"
            success_msg += f" on {state['parsed_date']}"
            if state.get('meeting_topic'):
                success_msg += f" regarding {state['meeting_topic']}"
            
            state["success_message"] = success_msg
            
            # Store detailed result
            state["result"] = {
                "user_id": state["user_id"],
                "meeting_date": state["parsed_date"],
                "file_path": file_path,
                "agent_type": "meeting"
            }
            
            if state.get('user_name'):
                state["result"]["user_name"] = state['user_name']
            if state.get('meeting_topic'):
                state["result"]["meeting_topic"] = state['meeting_topic']
            
        except Exception as e:
            state["error_message"] = f"File writing error: {str(e)}"
            state["status"] = "failed"    
    def _extract_context_from_intermediate_results(self, intermediate_results: dict) -> str:
        '''
        Extract relevant context from previous workflow steps for meeting scheduling.
        
        Args:
            intermediate_results: Dict mapping step keys to their results
            
        Returns:
            Formatted string with context that might help identify meeting participants
        '''
        try:
            context_parts = []
            
            for step_key, step_data in intermediate_results.items():
                if not isinstance(step_data, dict):
                    continue
                
                # Look for query results that might contain user/participant info
                if step_data.get('agent_type') in ['db_query', 'sql'] or 'query' in step_key.lower():
                    if 'query_data' in step_data:
                        # Check if query_data contains user information
                        query_data = step_data['query_data']
                        if isinstance(query_data, list) and query_data:
                            # Get first row as example
                            first_row = query_data[0] if isinstance(query_data, list) else None
                            if first_row and isinstance(first_row, dict):
                                # Look for user-related fields
                                user_fields = [k for k in first_row.keys() if 'user' in k.lower() or 'name' in k.lower() or 'id' in k.lower()]
                                if user_fields:
                                    context_parts.append(f'Query returned data with fields: {", ".join(user_fields)}')
                                    # Add sample value
                                    for field in user_fields[:2]:  # Just first 2 fields
                                        context_parts.append(f'  - {field}: {first_row.get(field)}')
                
                # Look for any result that might have participant info
                elif 'result' in step_data and isinstance(step_data['result'], dict):
                    result_dict = step_data['result']
                    for key in ['user', 'participant', 'attendee', 'name']:
                        if key in result_dict:
                            context_parts.append(f'Previous step mentioned {key}: {result_dict[key]}')
            
            if context_parts:
                return '\n'.join(context_parts)
            else:
                return ''
                
        except Exception as e:
            logger.error(f'Error extracting context from intermediate_results: {str(e)}')
            return ''
