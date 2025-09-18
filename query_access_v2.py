import os
import re
from datetime import datetime, timedelta
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

class AgentState(TypedDict):
    query: str
    user_id: str
    meeting_date: str
    parsed_date: str
    file_path: str
    status: str
    error_message: str
    success_message: str

class MeetingSchedulerAgent:
    
    def __init__(self, files_directory: str = "./user_files"):
        load_dotenv()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=150
        )
        
        self.files_directory = files_directory
        os.makedirs(self.files_directory, exist_ok=True)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("parse_query", self._parse_query)
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("handle_file", self._handle_file)
        workflow.add_node("write_meeting", self._write_meeting)
        workflow.add_node("confirm_success", self._confirm_success)
        workflow.add_node("handle_error", self._handle_error)
        
        workflow.set_entry_point("parse_query")
        
        workflow.add_edge("parse_query", "validate_input")
        workflow.add_conditional_edges(
            "validate_input",
            self._should_continue_after_validation,
            {
                "continue": "handle_file",
                "error": "handle_error"
            }
        )
        workflow.add_edge("handle_file", "write_meeting")
        workflow.add_conditional_edges(
            "write_meeting",
            self._should_continue_after_writing,
            {
                "success": "confirm_success",
                "error": "handle_error"
            }
        )
        workflow.add_edge("confirm_success", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _parse_query(self, state: AgentState) -> AgentState:
        query = state["query"]
        
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
        
        try:
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
            
            if content.startswith("ERROR:"):
                state["error_message"] = content.replace("ERROR: ", "")
                state["status"] = "parse_error"
                return state
            
            user_match = re.search(r'USER_ID:\s*(\d+)', content)
            date_match = re.search(r'DATE:\s*(\d{2}/\d{2}/\d{4})', content)
            
            if user_match and date_match:
                state["user_id"] = user_match.group(1)
                state["meeting_date"] = date_match.group(1)
                state["status"] = "parsed"
            else:
                state["error_message"] = "Could not extract user ID or date from the query"
                state["status"] = "parse_error"
                
        except Exception as e:
            state["error_message"] = f"Error parsing query: {str(e)}"
            state["status"] = "parse_error"
        
        return state
    
    def _validate_input(self, state: AgentState) -> AgentState:
        if state["status"] == "parse_error":
            return state
        
        try:
            user_id = int(state["user_id"])
            if user_id <= 0:
                state["error_message"] = "User ID must be a positive number"
                state["status"] = "validation_error"
                return state
            
            date_str = state["meeting_date"]
            try:
                parsed_date = datetime.strptime(date_str, "%d/%m/%Y")
                state["parsed_date"] = parsed_date.strftime("%d/%m/%Y")
                
                if parsed_date.date() < datetime.now().date():
                    state["error_message"] = f"Cannot schedule meeting in the past: {date_str}"
                    state["status"] = "validation_error"
                    return state
                    
            except ValueError:
                state["error_message"] = f"Invalid date format: {date_str}. Expected DD/MM/YYYY"
                state["status"] = "validation_error"
                return state
            
            state["status"] = "validated"
            
        except ValueError:
            state["error_message"] = f"Invalid user ID: {state['user_id']}"
            state["status"] = "validation_error"
        except Exception as e:
            state["error_message"] = f"Validation error: {str(e)}"
            state["status"] = "validation_error"
        
        return state
    
    def _handle_file(self, state: AgentState) -> AgentState:
        if state["status"] != "validated":
            return state
        
        try:
            filename = f"user{state['user_id']}.txt"
            file_path = os.path.join(self.files_directory, filename)
            state["file_path"] = file_path
            
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    f.write(f"=== User {state['user_id']} Meeting Schedule ===\n\n")
            
            state["status"] = "file_ready"
            
        except Exception as e:
            state["error_message"] = f"File handling error: {str(e)}"
            state["status"] = "file_error"
        
        return state
    
    def _write_meeting(self, state: AgentState) -> AgentState:
        if state["status"] != "file_ready":
            return state
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meeting_entry = f"Meeting on date {state['parsed_date']} (Scheduled on: {timestamp})\n"
            
            with open(state["file_path"], 'a') as f:
                f.write(meeting_entry)
            
            state["status"] = "meeting_written"
            state["success_message"] = f"Meeting successfully scheduled with User {state['user_id']} on {state['parsed_date']}"
            
        except Exception as e:
            state["error_message"] = f"Error writing to file: {str(e)}"
            state["status"] = "write_error"
        
        return state
    
    def _confirm_success(self, state: AgentState) -> AgentState:
        state["status"] = "completed"
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        state["status"] = "failed"
        return state
    
    def _should_continue_after_validation(self, state: AgentState) -> Literal["continue", "error"]:
        return "continue" if state["status"] == "validated" else "error"
    
    def _should_continue_after_writing(self, state: AgentState) -> Literal["success", "error"]:
        return "success" if state["status"] == "meeting_written" else "error"
    
    def visualize_graph(self, save_path: str = "agent_graph.png"):
        """Visualize the LangGraph structure and save as PNG"""
        png_data = self.graph.get_graph().draw_mermaid_png()
        with open(save_path, 'wb') as f:
            f.write(png_data)
        print(f"Graph visualization saved to: {save_path}")
        print("Open the PNG file to view the agent structure")
    
    def schedule_meeting(self, query: str) -> dict:
        initial_state = AgentState(
            query=query,
            user_id="",
            meeting_date="",
            parsed_date="",
            file_path="",
            status="",
            error_message="",
            success_message=""
        )
        
        result = self.graph.invoke(initial_state)
        
        if result["status"] == "completed":
            return {
                "success": True,
                "message": result["success_message"],
                "user_id": result["user_id"],
                "meeting_date": result["parsed_date"],
                "file_path": result["file_path"]
            }
        else:
            return {
                "success": False,
                "error": result.get("error_message", "Unknown error occurred"),
                "status": result["status"]
            }

def main():
    try:
        agent = MeetingSchedulerAgent(files_directory="./user_files")
        
        # Display graph structure
        print("Visualizing agent graph structure...")
        agent.visualize_graph()
        
    except ValueError as e:
        print(f"Error: {e}")
        print("groq api key failed to initialize.")
        return
    
    print("\nMeeting Scheduler Agent")
    print("=" * 40)
    print("Enter scheduling requests (type 'quit' to exit)")
    print("Examples:")
    print("  - Schedule meeting with user 3 tomorrow")
    print("  - Book meeting with user 15 next week")
    print("  - Meet user 7 on 25/12/2025")
    print("=" * 40)
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q', '']:
            break
        
        print(f"\nProcessing: {query}")
        result = agent.schedule_meeting(query)
        
        if result["success"]:
            print(f"Success: {result['message']}")
            print(f"File: {result['file_path']}")
        else:
            print(f"Error: {result['error']}")
            print(f"Status: {result['status']}")

if __name__ == "__main__":
    main()

