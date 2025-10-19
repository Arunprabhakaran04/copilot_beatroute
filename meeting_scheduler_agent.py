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
        
        parsed_data = self._parse_meeting_query(state["query"])
        if not parsed_data["success"]:
            meeting_state["error_message"] = parsed_data["error"]
            meeting_state["status"] = "failed"
            return meeting_state
        
        meeting_state["user_id"] = parsed_data["user_id"]
        meeting_state["meeting_date"] = parsed_data["meeting_date"]
        meeting_state["parsed_date"] = parsed_data["parsed_date"]
        
        if self._validate_meeting_data(meeting_state):
            self._write_meeting_to_file(meeting_state)
        
        return meeting_state
    
    def _parse_meeting_query(self, query: str) -> Dict[str, Any]:
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
            
            parse_prompt = ChatPromptTemplate.from_template("""
            You are a query parser for a meeting scheduling system. Extract the user ID and meeting date from the following query.
            
            CURRENT DATE CONTEXT:
            - Today is: {current_date} ({current_day})
            - Tomorrow is: {tomorrow_date}
            - Day after tomorrow is: {day_after_tomorrow}
            - Next week (same day): {next_week_date}
            - Next Monday: {next_monday}
            
            Query: {query}
            
            IMPORTANT: Respond ONLY in this EXACT format (no extra text):
            USER_ID: [number]
            DATE: [DD/MM/YYYY format]
            
            Rules for date conversion:
            1. "tomorrow" → {tomorrow_date}
            2. "day after tomorrow" → {day_after_tomorrow}
            3. "next week" → {next_week_date}
            4. "next monday" → {next_monday}
            5. "today" → {current_date}
            6. Convert written dates (like "September 20, 2025") to DD/MM/YYYY format
            7. Keep DD/MM/YYYY dates as they are
            8. If date is unclear, respond with: ERROR: [reason]
            """)
            
            messages = parse_prompt.format_messages(
                query=query,
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
            date_match = re.search(r'DATE:\s*(\d{2}/\d{2}/\d{4})', content)
            
            if user_match and date_match:
                return {
                    "success": True,
                    "user_id": user_match.group(1),
                    "meeting_date": date_match.group(1),
                    "parsed_date": date_match.group(1)
                }
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
            meeting_entry = f"Meeting on date {state['parsed_date']} (Scheduled on: {timestamp})\n"
            
            with open(file_path, 'a') as f:
                f.write(meeting_entry)
            
            state["status"] = "completed"
            state["success_message"] = f"Meeting scheduled with User {state['user_id']} on {state['parsed_date']}"
            state["result"] = {
                "user_id": state["user_id"],
                "meeting_date": state["parsed_date"],
                "file_path": file_path
            }
            
        except Exception as e:
            state["error_message"] = f"File writing error: {str(e)}"
            state["status"] = "failed"