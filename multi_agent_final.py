import os
import re
import smtplib
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict, Literal, Dict, Any, List, Optional
from collections import deque
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgentState(TypedDict):
    query: str
    agent_type: str
    user_id: str
    status: str
    error_message: str
    success_message: str
    result: Dict[str, Any]
    start_time: float
    end_time: float
    execution_time: float

class MeetingAgentState(BaseAgentState):
    meeting_date: str
    parsed_date: str
    file_path: str

class DBAgentState(BaseAgentState):
    sql_query: str
    query_type: str

class EmailAgentState(BaseAgentState):
    email_to: str
    email_subject: str
    email_content: str

class BaseAgent(ABC):
    def __init__(self, llm):
        self.llm = llm
    
    @abstractmethod
    def process(self, state: BaseAgentState) -> BaseAgentState:
        pass
    
    @abstractmethod
    def get_agent_type(self) -> str:
        pass

class DBQueryAgent(BaseAgent):
    def __init__(self, llm, schema_file_path: str = "schema"):
        super().__init__(llm)
        self.schema_file_path = schema_file_path
        self.schema_content = self._load_schema_file()
        
        self.query_templates = {
            "insert": "INSERT INTO {table} ({columns}) VALUES ({values})",
            "select": "SELECT {columns} FROM {table} WHERE {condition}",
            "update": "UPDATE {table} SET {updates} WHERE {condition}",
            "delete": "DELETE FROM {table} WHERE {condition}"
        }
    
    def get_agent_type(self) -> str:
        return "db_query"
    
    def _load_schema_file(self) -> str:
        """Load the database schema from the schema file."""
        try:
            with open(self.schema_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                content = content.replace('\\n', '\n')
                
                logger.info(f"Successfully loaded schema file: {self.schema_file_path}")
                logger.info(f"Schema content length: {len(content)} characters")
                
                return content
        except FileNotFoundError:
            error_msg = f"Schema file not found: {self.schema_file_path}"
            logger.error(error_msg)
            return "Schema file not found. Unable to load database schema."
        except Exception as e:
            error_msg = f"Error loading schema file: {e}"
            logger.error(error_msg)
            return f"Error loading schema: {str(e)}"
    
    def get_schema_info(self) -> str:
        """Return the loaded schema information."""
        return self.schema_content
    
    def process(self, state: BaseAgentState) -> DBAgentState:
        db_state = DBAgentState(**state)
        
        try:
            schema_info = self.get_schema_info()
            
            query_prompt = ChatPromptTemplate.from_template("""
            You are an expert SQL query generator with access to a comprehensive database schema.
            Convert the user's natural language request into an accurate PostgreSQL query.
            
            DATABASE SCHEMA:
            {schema_info}
            
            USER REQUEST: {query}
            
            INSTRUCTIONS:
            1. Analyze the user request carefully to understand the intent
            2. Identify the most relevant table(s) from the schema above
            3. Use exact table and column names as specified in the schema
            4. Pay attention to data types (text, numeric, timestamp, bigint, boolean, etc.)
            5. Use proper PostgreSQL syntax and functions
            6. For date/time operations, use appropriate PostgreSQL date functions
            7. Consider relationships between tables when necessary
            8. Ensure the query is efficient and returns meaningful results
            
            RESPONSE FORMAT (respond with EXACTLY this format):
            QUERY_TYPE: [INSERT/SELECT/UPDATE/DELETE]
            SQL: [Your complete SQL query here - ensure it's valid PostgreSQL syntax]
            EXPLANATION: [Brief explanation of what the query does, which tables/columns are used, and the logic behind your choice]
            
            EXAMPLES OF GOOD RESPONSES:
            - For "show all brands": 
              QUERY_TYPE: SELECT
              SQL: SELECT * FROM Brand ORDER BY name;
              EXPLANATION: Retrieves all brand records from the Brand table, ordered by name for better readability.
            
            - For "find customers in Mumbai":
              QUERY_TYPE: SELECT  
              SQL: SELECT * FROM ViewCustomer WHERE city = 'Mumbai';
              EXPLANATION: Uses the ViewCustomer table which contains customer information including city field to filter for Mumbai customers.
            """)
            
            messages = query_prompt.format_messages(
                schema_info=schema_info,
                query=state["query"]
            )
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            query_type_match = re.search(r'QUERY_TYPE:\s*(\w+)', content)
            sql_match = re.search(r'SQL:\s*(.+?)(?=\nEXPLANATION:|$)', content, re.DOTALL)
            explanation_match = re.search(r'EXPLANATION:\s*(.+)', content, re.DOTALL)
            
            if query_type_match and sql_match:
                db_state["query_type"] = query_type_match.group(1).upper()
                db_state["sql_query"] = sql_match.group(1).strip()
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
                
                db_state["status"] = "completed"
                db_state["success_message"] = f"Generated {db_state['query_type']} query successfully"
                db_state["result"] = {
                    "sql_query": db_state["sql_query"],
                    "query_type": db_state["query_type"],
                    "explanation": explanation,
                    "schema_source": "loaded_from_file",
                    "schema_file": self.schema_file_path
                }
                
                print(f"\nGenerated SQL Query:")
                print(f"SQL: {db_state['sql_query']}")
                
            else:
                db_state["error_message"] = "Could not parse SQL query from response"
                db_state["status"] = "failed"
                
        except Exception as e:
            db_state["error_message"] = f"DB query generation error: {str(e)}"
            db_state["status"] = "failed"
        
        return db_state

class EmailAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(llm)
        load_dotenv()
        
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_email = os.getenv("SMTP_EMAIL")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        
        if not self.smtp_email or not self.smtp_password:
            logger.warning("SMTP credentials not found in environment variables. Email sending will be simulated.")
            self.smtp_enabled = False
        else:
            self.smtp_enabled = True
    
    def get_agent_type(self) -> str:
        return "email"
    
    def process(self, state: BaseAgentState) -> EmailAgentState:
        email_state = EmailAgentState(**state)
        
        try:
            all_emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', state["query"])
            
            content_prompt = ChatPromptTemplate.from_template("""
            You are an email content and recipient parser. Analyze the user's request and extract email details.
            
            User Request: {query}
            
            Respond in this EXACT format:
            TO: [primary recipient email addresses, comma-separated]
            CC: [cc recipient email addresses, comma-separated, or "none" if no CC]
            SUBJECT: [Email subject line - keep it concise and relevant]
            CONTENT: [Email body content - professional and well-formatted]
            
            IMPORTANT PARSING RULES:
            1. If query says "send to john@x.com and cc mary@y.com" → TO: john@x.com, CC: mary@y.com
            2. If query says "cc mary@y.com" without explicit TO → assume CC is additional to a previous recipient
            3. If query has multiple emails without CC indication → put first email in TO, others in CC
            4. If query says "send to bob@x.com" → TO: bob@x.com, CC: none
            5. Extract subject from "subject:" or infer from context
            
            Guidelines for content:
            - Keep the email professional, relevant, and concise
            - Make the subject line clear and specific
            - Structure the content with proper paragraphs
            - End with appropriate closing
            """)
            
            messages = content_prompt.format_messages(query=state["query"])
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            to_match = re.search(r'TO:\s*(.+)', content)
            cc_match = re.search(r'CC:\s*(.+)', content)
            subject_match = re.search(r'SUBJECT:\s*(.+)', content)
            content_match = re.search(r'CONTENT:\s*(.+)', content, re.DOTALL)
            
            if to_match and subject_match and content_match:
                to_emails = to_match.group(1).strip()
                cc_emails = cc_match.group(1).strip() if cc_match else "none"
                
                # If CC is specified but TO seems incomplete, try to be smart about it
                if cc_emails != "none" and len(all_emails) >= 2:
                    all_recipients = []
                    if to_emails and to_emails != "none":
                        all_recipients.extend([email.strip() for email in to_emails.split(',')])
                    if cc_emails != "none":
                        all_recipients.extend([email.strip() for email in cc_emails.split(',')])
                    
                    final_recipients = list(dict.fromkeys(all_recipients)) 
                    recipients_str = ", ".join(final_recipients)
                else:
                    recipients_str = to_emails
                
                email_state["email_to"] = recipients_str
                email_state["email_subject"] = subject_match.group(1).strip()
                email_state["email_content"] = content_match.group(1).strip()
                
                send_success = self._send_email(
                    email_state["email_to"],
                    email_state["email_subject"],
                    email_state["email_content"]
                )
                
                if send_success:
                    email_state["status"] = "completed"
                    email_state["success_message"] = f"Email sent successfully to {email_state['email_to']}"
                    email_state["result"] = {
                        "email_to": email_state["email_to"],
                        "cc_emails": cc_emails if cc_emails != "none" else None,
                        "subject": email_state["email_subject"],
                        "content": email_state["email_content"],
                        "sent_via": "SMTP" if self.smtp_enabled else "Simulated"
                    }
                else:
                    email_state["error_message"] = "Failed to send email via SMTP"
                    email_state["status"] = "failed"
                    
            else:
                email_state["error_message"] = "Could not parse email content from response"
                email_state["status"] = "failed"
                
        except Exception as e:
            email_state["error_message"] = f"Email generation error: {str(e)}"
            email_state["status"] = "failed"
        
        return email_state
    
    def _send_email(self, to_email: str, subject: str, content: str) -> bool:
        if not self.smtp_enabled:
            return self._simulate_send_email(to_email, subject, content)
        
        try:
            recipients = [email.strip() for email in to_email.split(',')]
            
            msg = MIMEMultipart()
            msg['From'] = self.smtp_email
            msg['To'] = to_email  
            msg['Subject'] = subject
            
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                            {subject}
                        </h2>
                        <div style="background-color: #f8f9fa; padding: 20px; border-left: 4px solid #3498db; margin: 20px 0;">
                            <div style="white-space: pre-wrap; font-size: 16px;">
                                {content}
                            </div>
                        </div>
                        <hr style="border: 1px solid #eee; margin: 30px 0;">
                        <p style="font-size: 12px; color: #666; text-align: center;">
                            This email was sent by the Multi-Agent System
                        </p>
                    </div>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_email, self.smtp_password)
            text = msg.as_string()
            
            server.sendmail(self.smtp_email, recipients, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {to_email}")
            print(f"\nEmail Sent Successfully:")
            print(f"To: {to_email}")
            print(f"Subject: {subject}")
            print(f"Content: {content[:100]}{'...' if len(content) > 100 else ''}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            print(f"\nEmail sending failed: {e}")
            return self._simulate_send_email(to_email, subject, content)
    
    def _simulate_send_email(self, to_email: str, subject: str, content: str) -> bool:
        print(f"\nEmail Sent (Simulated):")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Content: {content}")
        print("Note: SMTP credentials not configured. Email was simulated.")
        return True

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
            
            logger.info(f"Campaign LLM response: {content}")
            
            import json
            try:
                # Try to extract JSON from the response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
                    result_data = json.loads(json_content)
                else:
                    # If no JSON found, treat as passthrough
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


class MemoryManager:
    """In memory conversattion manager - uses llm to enrich the query with previous memory

    Stored entry shape: {"original": str, "enriched": str, "agent_type": str, "result": dict}
    """
    def __init__(self, llm, max_history: int = 3):
        self.llm = llm
        self.max_history = max_history
        self.history = deque(maxlen=max_history)

    def add_entry(self, original_query: str, enriched_query: str, agent_type: str, result: Dict[str, Any]):
        entry = {
            "original": original_query,
            "enriched": enriched_query,
            "agent_type": agent_type,
            "result": result
        }
        self.history.append(entry)

    def get_recent(self, n: int = 3) -> List[Dict[str, Any]]:
        return list(self.history)[-n:]

    def _heuristic_expand(self, query: str) -> Optional[str]:
        # Basic heuristic
        lowered = query.lower()
        if any(token in lowered for token in ["that date", "that day", "that time", "that slot"]):
            for entry in reversed(self.history):
                if entry.get("agent_type") == "meeting":
                    meeting_date = entry.get("result", {}).get("meeting_date") or entry.get("result", {}).get("parsed_date")
                    if meeting_date:
                        return re.sub(r"(?i)that date|that day|that time|that slot", meeting_date, query)
        
        # Check for ambiguous follow-up patterns that need LLM enrichment
        ambiguous_patterns = [
            "also invite", "also send", "also include", "also add",
            "confirm the", "update the", "change the", "modify the",
            "for the meeting", "for that meeting", "to the meeting",
            "the same", "same meeting", "same time", "same date"
        ]
        
        if any(pattern in lowered for pattern in ambiguous_patterns):
            # Return None to force LLM enrichment for these cases
            return None
        
        return None

    def enrich_query(self, original_query: str) -> str:
        """Return an enriched query. Try simple heuristics first; if not
        sufficient, build a short context from recent history and ask the LLM
        (Groq) to rewrite the query filling ambiguous references.
        """
        if not self.history:
            return original_query

        heur = self._heuristic_expand(original_query)
        if heur:
            return heur

        lowered = original_query.lower()
        needs_enrichment = any([
            "also" in lowered,
            "the meeting" in lowered,
            "that" in lowered and ("date" in lowered or "time" in lowered or "meeting" in lowered),
            "same" in lowered,
            "confirm" in lowered and "location" in lowered,
            "invite" in lowered and "@" in original_query,  
            "cc" in lowered and "@" in original_query, 
            "same subject" in lowered,  
            "same email" in lowered,
            "add to" in lowered and ("email" in lowered or "@" in original_query),
            "send to" in lowered and "also" in lowered,
        ])

        if not needs_enrichment:
            return original_query

        recent = self.get_recent(self.max_history)
        context_lines = []
        for e in recent:
            small_result = {k: v for k, v in (e.get("result") or {}).items() if isinstance(v, (str, int, float))}
            context_lines.append(f"User: {e['original']} --> Agent: {e.get('agent_type')} --> Result: {small_result}")

        context_text = "\n".join(context_lines)

        enrich_prompt = ChatPromptTemplate.from_template("""
        You are a context-aware query rewriter for a multi-agent system. Your job is to combine RECENT CONTEXT with the CURRENT USER QUERY to create a complete, unambiguous instruction.

        RECENT INTERACTIONS (context for reference):
        {recent_context}

        CURRENT USER QUERY (the new instruction): {query}

        CRITICAL RULES FOR COMBINING CONTEXT + CURRENT QUERY:
        1. The CURRENT QUERY contains NEW INSTRUCTIONS that must be followed
        2. Use RECENT CONTEXT only to fill in missing details (subjects, dates, original recipients)
        3. When current query says "cc someone@email.com" - ADD this as CC, don't replace original recipient
        4. When current query says "use same subject" - get the subject from recent context
        5. When current query says "also invite" - ADD the new person, don't replace the original
        6. PRESERVE all new information from current query (new emails, new dates, new instructions)

        SPECIFIC EXAMPLES:
        Context: "User: send email to john@x.com subject: Meeting Reminder → Agent: email → Result: {{'email_to': 'john@x.com', 'subject': 'Meeting Reminder'}}"
        Current: "cc mary@y.com and use same subject"
        Correct Output: "Send email to john@x.com and cc mary@y.com with subject: Meeting Reminder"

        Context: "User: schedule meeting with user 3 on 2025-09-30 → Agent: meeting → Result: {{'user_id': '3', 'meeting_date': '2025-09-30'}}"
        Current: "also invite bob@x.com"
        Correct Output: "Invite bob@x.com to the meeting with user 3 scheduled on 2025-09-30"

        Context: "User: send email to alice@x.com about Q3 report → Agent: email → Result: {{'email_to': 'alice@x.com', 'subject': 'Q3 Report'}}"
        Current: "send to bob@y.com with same subject"
        Correct Output: "Send email to bob@y.com with subject: Q3 Report"

        IMPORTANT: 
        - If current query introduces NEW recipients, use those NEW recipients
        - If current query says "cc" or "add", combine with original recipients
        - Always preserve the intent and new information from the current query
        - Use context only to fill gaps, not to override new instructions

        Respond in this EXACT format (no extra commentary):
        ENRICHED_QUERY: [the complete instruction combining context + current query]
        """)

        try:
            messages = enrich_prompt.format_messages(recent_context=context_text, query=original_query)
            response = self.llm.invoke(messages)
            content = response.content.strip()
            match = re.search(r'ENRICHED_QUERY:\s*(.+)', content, re.DOTALL)
            if match:
                enriched = match.group(1).strip()
                if enriched:
                    logger.info(f"Query enriched: '{original_query}' → '{enriched}'")
                    return enriched
        except Exception as e:
            logger.warning(f"LLM enrichment failed: {e}")

        # fallback: return original
        return original_query

class CentralOrchestrator:
    def __init__(self, files_directory: str = "./user_files", schema_file_path: str = "schema"):
        load_dotenv()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=groq_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        
        self.db_llm = ChatOpenAI(
            model="gpt-4o", 
            api_key=openai_api_key,
            temperature=0.1,
            max_tokens=2000
        )
        
        self.agents = {
            "db_query": DBQueryAgent(self.db_llm, schema_file_path),
            "email": EmailAgent(self.llm),
            "meeting": MeetingSchedulerAgent(self.llm, files_directory),
            "campaign": CampaignAgent(self.llm)
        }
        
        logger.info("Initialized CentralOrchestrator with:")
        logger.info(f"  - DB Query Agent: OpenAI GPT-4o (large context for schema)")
        logger.info(f"  - Email Agent: Groq Llama-3.1-8b-instant")
        logger.info(f"  - Meeting Agent: Groq Llama-3.1-8b-instant")
        
        self.classification_keywords = {
            "db_query": [
                "database", "query", "sql", "insert", "select", "update", "delete",
                "add to cart", "product", "order", "table", "record", "data",
                "create", "remove", "modify", "store", "retrieve", "cart", "user profile",
                "show", "find", "get", "list", "search", "display"
            ],
            "email": [
                "email", "mail", "send", "notify", "message", "contact",
                "inform", "alert", "notification", "compose", "write to", "@",
                "cc", "bcc", "subject", "recipient", "sender", "reply"
            ],
            "meeting": [
                "meeting", "schedule", "appointment", "book", "calendar",
                "tomorrow", "today", "next week", "date", "time", "meet",
                "demo", "call", "session", "conference", "invite", "attendee"
            ],
            "campaign": [
                "campaign", "holiday sale", "summer promo", "black friday",
                "campaign id", "campaign name", "campaign data", "campaign info",
                "active campaigns", "campaign metrics", "campaign response"
            ]
        }
        #email classification patterns.
        self.classification_patterns = {
            "email": [
                r"\b\w+@\w+\.\w+\b",  
                r"\bcc\s+\w+@\w+\.\w+",  
                r"\bsend\s+.*?@", 
                r"\bsubject\s*:", 
                r"\bemail.*?to\b", 
            ],
            "meeting": [
                r"\bschedule.*?(meeting|demo|call|appointment)",
                r"\b(book|arrange).*?(meeting|demo|call)",
                r"\bmeet.*?(with|at|on)",
                r"\b(tomorrow|today|next\s+\w+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                r"\b\d{1,2}[:/]\d{2}(\s*(am|pm))?",
            ],
            "db_query": [
                r"\b(show|find|get|list|display|retrieve).*?(all|from)",
                r"\b(add|insert|create|store).*?(to|in|into)",
                r"\b(update|modify|change|edit)",
                r"\b(delete|remove|drop)",
                r"\btable\b.*?\b(records?|data|entries)",
            ]
        }
        
        self.graph = self._build_orchestrator_graph()
        self.memory = MemoryManager(self.llm, max_history=3)
    
    def _build_orchestrator_graph(self) -> StateGraph:
        workflow = StateGraph(BaseAgentState)
        
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("route_to_agent", self._route_to_agent)
        workflow.add_node("handle_error", self._handle_error)
        
        workflow.set_entry_point("classify_query")
        
        workflow.add_conditional_edges(
            "classify_query",
            self._should_classify_or_error,
            {
                "route": "route_to_agent",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "route_to_agent",
            self._should_route_or_error,
            {
                "complete": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: BaseAgentState) -> BaseAgentState:
        """Pure LLM classification for maximum accuracy"""
        try:
            classify_prompt = ChatPromptTemplate.from_template("""
            You are a production query classifier. Classify this query into exactly ONE category.
            
            Query: {query}
            
            Categories:
            1. campaign - Questions about campaigns, campaign IDs, campaign names, campaign data, Holiday Sale, Summer Promo, Black Friday, active campaigns, campaign metrics, etc.
            2. db_query - Database operations (show, find, insert, select, update, delete, list, display, retrieve data, etc.)
            3. email - Email operations (send, compose, cc, bcc, notify, @ symbols, email addresses, etc.)  
            4. meeting - Meeting/scheduling (schedule, book, appointment, meet, demo, call, dates, times, etc.)
            
            EXAMPLES:
            "what is the holiday sale campaign id" → campaign
            "show me all campaigns" → campaign
            "send email to john@x.com" → email
            "schedule meeting with user 2" → meeting
            "show all customers" → db_query
            "find customers in Mumbai" → db_query
            "schedule a meet with user 3 on 30th sep 2025" → meeting
            
            Respond with ONLY one word: campaign, db_query, email, or meeting
            """)
            
            messages = classify_prompt.format_messages(query=state["query"])
            response = self.llm.invoke(messages)
            agent_type = response.content.strip().lower()
            
            # Validate response
            valid_agents = ["campaign", "db_query", "email", "meeting"]
            if agent_type in valid_agents:
                state["agent_type"] = agent_type
                state["status"] = "classified"
                logger.info(f"LLM Classification: {agent_type}")
                return state
            else:
                # Fallback classification based on simple heuristics
                query_lower = state["query"].lower()
                if any(word in query_lower for word in ["campaign", "holiday sale", "summer promo", "black friday"]):
                    fallback_type = "campaign"
                elif "@" in state["query"]:
                    fallback_type = "email"
                elif any(word in query_lower for word in ["schedule", "meeting", "meet", "book"]):
                    fallback_type = "meeting"
                else:
                    fallback_type = "db_query"
                
                state["agent_type"] = fallback_type
                state["status"] = "classified"
                logger.warning(f"LLM gave invalid response '{agent_type}', using fallback: {fallback_type}")
                return state
                
        except Exception as e:
            # Ultimate fallback
            query_lower = state["query"].lower()
            if any(word in query_lower for word in ["campaign", "holiday sale", "summer promo", "black friday"]):
                fallback_type = "campaign"
            elif "@" in state["query"]:
                fallback_type = "email"
            elif any(word in query_lower for word in ["schedule", "meeting", "meet"]):
                fallback_type = "meeting"
            else:
                fallback_type = "db_query"
            
            state["agent_type"] = fallback_type
            state["status"] = "classified"
            logger.error(f"Classification completely failed: {e}, using fallback: {fallback_type}")
            return state
    
    def _route_to_agent(self, state: BaseAgentState) -> BaseAgentState:
        try:
            agent_type = state["agent_type"]
            if agent_type not in self.agents:
                state["error_message"] = f"No agent found for type: {agent_type}"
                state["status"] = "routing_error"
                return state
            
            logger.info(f"Routing to {agent_type} agent")
            agent = self.agents[agent_type]
            
            # Special handling for campaign agent - it may enrich the query or provide direct answer
            if agent_type == "campaign":
                campaign_result = agent.process(state)
                
                # If campaign agent provided a direct answer, we're done
                if campaign_result["status"] == "completed":
                    logger.info("Campaign agent provided direct answer")
                    return campaign_result
                
                # If campaign agent enriched the query, reclassify the enriched query
                if campaign_result["status"] == "campaign_processed" and "enriched_query" in campaign_result.get("result", {}):
                    enriched_query = campaign_result["result"]["enriched_query"]
                    logger.info(f"Campaign agent enriched query: '{state['query']}' → '{enriched_query}'")
                    
                    # Reclassify the enriched query (excluding campaign this time)
                    reclassify_prompt = ChatPromptTemplate.from_template("""
                    Classify this enriched query into exactly ONE category.
                    
                    Query: {query}
                    
                    Categories:
                    1. db_query - Database operations (show, find, insert, select, update, delete, list, display, retrieve data, etc.)
                    2. email - Email operations (send, compose, cc, bcc, notify, @ symbols, email addresses, etc.)  
                    3. meeting - Meeting/scheduling (schedule, book, appointment, meet, demo, call, dates, times, etc.)
                    
                    Respond with ONLY one word: db_query, email, or meeting
                    """)
                    
                    messages = reclassify_prompt.format_messages(query=enriched_query)
                    response = self.llm.invoke(messages)
                    new_agent_type = response.content.strip().lower()
                    
                    if new_agent_type in ["db_query", "email", "meeting"]:
                        # Update state with new classification and enriched query
                        state["agent_type"] = new_agent_type
                        state["query"] = enriched_query
                        
                        # Route to the correct agent
                        logger.info(f"Reclassified enriched query as: {new_agent_type}")
                        final_agent = self.agents[new_agent_type]
                        return final_agent.process(state)
                    else:
                        # Fallback to db_query if reclassification fails
                        state["agent_type"] = "db_query"
                        state["query"] = enriched_query
                        return self.agents["db_query"].process(state)
                else:
                    # Campaign agent processed but didn't enrich, treat as passthrough
                    logger.info("Campaign agent processed without enrichment")
                    return campaign_result
            else:
                # For non-campaign agents, process directly
                result_state = agent.process(state)
                return result_state
            
        except Exception as e:
            state["error_message"] = f"Routing error: {str(e)}"
            state["status"] = "routing_error"
            logger.error(f"Routing error: {e}")
            return state
    
    def _handle_error(self, state: BaseAgentState) -> BaseAgentState:
        state["status"] = "failed"
        logger.error(f"Error state: {state.get('error_message', 'Unknown error')}")
        return state
    
    def _should_classify_or_error(self, state: BaseAgentState) -> Literal["route", "error"]:
        if state["status"] == "classified":
            return "route"
        elif state["status"] == "completed":  # Campaign agent answered directly
            return "route"  # This will go to route_to_agent but it should return immediately
        else:
            return "error"
    
    def _should_route_or_error(self, state: BaseAgentState) -> Literal["complete", "error"]:
        if state["status"] in ["completed", "classification_complete"]:
            return "complete"
        else:
            return "error"
    
    def add_agent(self, agent_type: str, agent: BaseAgent, keywords: List[str]):
        self.agents[agent_type] = agent
        self.classification_keywords[agent_type] = keywords
        print(f"Added new agent: {agent_type}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        enriched_query = self.memory.enrich_query(query)

        initial_state = BaseAgentState(
            query=enriched_query,
            agent_type="",
            user_id="",
            status="",
            error_message="",
            success_message="",
            result={},
            start_time=time.time(),
            end_time=0.0,
            execution_time=0.0
        )
        initial_state["original_query"] = query
        
        print(f"\nProcessing Query: '{query}'")
        
        start_time = time.time()
        result = self.graph.invoke(initial_state)
        end_time = time.time()
        execution_time = end_time - start_time
        
        result["start_time"] = start_time
        result["end_time"] = end_time
        result["execution_time"] = execution_time
        
        print(f"Execution Time: {execution_time:.4f} seconds")
        
        if result["status"] == "completed":
            print(f"Success: {result['success_message']}")
            try:
                self.memory.add_entry(query, enriched_query, result.get("agent_type", ""), result.get("result", {}))
            except Exception:
                pass
            return {
                "success": True,
                "message": result["success_message"],
                "agent_type": result.get("agent_type", "campaign"),
                "result": result["result"],
                "execution_time": execution_time
            }
        else:
            print(f"Error: {result['error_message']}")
            return {
                "success": False,
                "error": result.get("error_message", "Unknown error occurred"),
                "status": result["status"],
                "agent_type": result.get("agent_type", "unknown"),
                "execution_time": execution_time
            }



def main():
    try:
        orchestrator = CentralOrchestrator(files_directory="./user_files", schema_file_path="schema")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check your GROQ API key configuration.")
        return
    
    print("Multi-Agent Orchestrator System")
    print("=" * 70)
    print("Commands:")
    print("  - 'quit' - Exit the system")
    print("=" * 70)
    print("Example queries:")
    print("  - 'Show all brands'")
    print("  - 'Find customers in Mumbai'")
    print("  - 'Get campaign responses from last week'")
    print("  - 'Show user performance data'")
    print("  - 'Send email to john@example.com about campaign results'")
    print("  - 'Schedule meeting with user 3 tomorrow'")
    print("=" * 70)
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q', '']:
            print("Goodbye!")
            break
        
        result = orchestrator.process_query(query)
        
        if not result["success"]:
            print(f"Status: {result.get('status', 'unknown')}")

if __name__ == "__main__":
    main()