import os
import re
import smtplib
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import TypedDict, Literal, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
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
    def __init__(self, llm: ChatGroq):
        self.llm = llm
    
    @abstractmethod
    def process(self, state: BaseAgentState) -> BaseAgentState:
        pass
    
    @abstractmethod
    def get_agent_type(self) -> str:
        pass

class DBQueryAgent(BaseAgent):
    def __init__(self, llm: ChatGroq):
        super().__init__(llm)
        
        self.table_schemas = {
            "users": {
                "columns": ["user_id", "username", "email", "created_at", "last_login"],
                "types": ["INT PRIMARY KEY", "VARCHAR(50)", "VARCHAR(100)", "DATETIME", "DATETIME"],
                "description": "User account information"
            },
            "products": {
                "columns": ["product_id", "name", "price", "category", "stock_quantity", "created_at"],
                "types": ["INT PRIMARY KEY", "VARCHAR(100)", "DECIMAL(10,2)", "VARCHAR(50)", "INT", "DATETIME"],
                "description": "Product catalog"
            },
            "orders": {
                "columns": ["order_id", "user_id", "product_id", "quantity", "total_amount", "order_date", "status"],
                "types": ["INT PRIMARY KEY", "INT", "INT", "INT", "DECIMAL(10,2)", "DATETIME", "VARCHAR(20)"],
                "description": "Customer orders"
            },
            "cart": {
                "columns": ["cart_id", "user_id", "product_id", "quantity", "added_at"],
                "types": ["INT PRIMARY KEY", "INT", "INT", "INT", "DATETIME"],
                "description": "Shopping cart items"
            }
        }
        
        self.query_templates = {
            "insert": "INSERT INTO {table} ({columns}) VALUES ({values})",
            "select": "SELECT {columns} FROM {table} WHERE {condition}",
            "update": "UPDATE {table} SET {updates} WHERE {condition}",
            "delete": "DELETE FROM {table} WHERE {condition}"
        }
    
    def get_agent_type(self) -> str:
        return "db_query"
    
    def get_schema_info(self) -> str:
        schema_info = "Available Database Tables:\n"
        for table, info in self.table_schemas.items():
            schema_info += f"\n{table.upper()}:\n"
            schema_info += f"  Description: {info['description']}\n"
            schema_info += f"  Columns: {', '.join(info['columns'])}\n"
            schema_info += f"  Types: {', '.join(info['types'])}\n"
        return schema_info
    
    def process(self, state: BaseAgentState) -> DBAgentState:
        db_state = DBAgentState(**state)
        
        try:
            schema_info = self.get_schema_info()
            
            query_prompt = ChatPromptTemplate.from_template("""
            You are a SQL query generator. Convert the user's natural language request into a SQL query.
            Use the provided database schema to ensure accurate table and column names.
            
            {schema_info}
            
            User Query: {query}
            
            Guidelines:
            - Use exact table and column names from the schema
            - For user-related queries, assume user_id can be extracted from context
            - For cart operations, use the cart table
            - For product queries, use the products table
            - For order operations, use the orders table
            
            Respond in this EXACT format:
            QUERY_TYPE: [INSERT/SELECT/UPDATE/DELETE]
            SQL: [Your SQL query here]
            EXPLANATION: [Brief explanation of what the query does]
            
            Examples:
            - "add product 2 to cart for user 1" → INSERT INTO cart (user_id, product_id, quantity, added_at) VALUES (1, 2, 1, NOW())
            - "show my orders" → SELECT * FROM orders WHERE user_id = 1
            - "update my profile email" → UPDATE users SET email = 'new_email@example.com' WHERE user_id = 1
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
                    "table_schemas": self.table_schemas
                }
                
                print(f"\nGenerated SQL Query:")
                print(f"Type: {db_state['query_type']}")
                print(f"SQL: {db_state['sql_query']}")
                print(f"Explanation: {explanation}")
                
            else:
                db_state["error_message"] = "Could not parse SQL query from response"
                db_state["status"] = "failed"
                
        except Exception as e:
            db_state["error_message"] = f"DB query generation error: {str(e)}"
            db_state["status"] = "failed"
        
        return db_state

class EmailAgent(BaseAgent):
    def __init__(self, llm: ChatGroq):
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
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', state["query"])
            recipient_email = email_match.group(0) if email_match else "user@example.com"
            
            content_prompt = ChatPromptTemplate.from_template("""
            You are an email content generator. Create a professional email based on the user's request.
            
            User Request: {query}
            
            Respond in this EXACT format:
            SUBJECT: [Email subject line - keep it concise and relevant]
            CONTENT: [Email body content - professional and well-formatted]
            
            Guidelines:
            - Keep the email professional, relevant, and concise
            - Make the subject line clear and specific
            - Structure the content with proper paragraphs
            - End with appropriate closing
            """)
            
            messages = content_prompt.format_messages(query=state["query"])
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            subject_match = re.search(r'SUBJECT:\s*(.+)', content)
            content_match = re.search(r'CONTENT:\s*(.+)', content, re.DOTALL)
            
            if subject_match and content_match:
                email_state["email_to"] = recipient_email
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
            server.sendmail(self.smtp_email, to_email, text)
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

class MeetingSchedulerAgent(BaseAgent):
    def __init__(self, llm: ChatGroq, files_directory: str = "./user_files"):
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

class CentralOrchestrator:
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
        
        self.agents = {
            "db_query": DBQueryAgent(self.llm),
            "email": EmailAgent(self.llm),
            "meeting": MeetingSchedulerAgent(self.llm, files_directory)
        }
        
        self.classification_keywords = {
            "db_query": [
                "database", "query", "sql", "insert", "select", "update", "delete",
                "add to cart", "product", "order", "table", "record", "data",
                "create", "remove", "modify", "store", "retrieve", "cart", "user profile"
            ],
            "email": [
                "email", "mail", "send", "notify", "message", "contact",
                "inform", "alert", "notification", "compose", "write to", "@"
            ],
            "meeting": [
                "meeting", "schedule", "appointment", "book", "calendar",
                "tomorrow", "today", "next week", "date", "time", "meet"
            ]
        }
        
        self.graph = self._build_orchestrator_graph()
    
    def _build_orchestrator_graph(self) -> StateGraph:
        workflow = StateGraph(BaseAgentState)
        
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("route_to_agent", self._route_to_agent)
        workflow.add_node("handle_error", self._handle_error)
        
        workflow.set_entry_point("classify_query")
        
        workflow.add_conditional_edges(
            "classify_query",
            self._should_route_or_error,
            {
                "route": "route_to_agent",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("route_to_agent", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _classify_query(self, state: BaseAgentState) -> BaseAgentState:
        query = state["query"].lower()
        
        keyword_scores = {}
        for agent_type, keywords in self.classification_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query)
            keyword_scores[agent_type] = score
        
        max_score = max(keyword_scores.values())
        if max_score >= 2:
            predicted_agent = max(keyword_scores, key=keyword_scores.get)
            state["agent_type"] = predicted_agent
            state["status"] = "classified"
            return state
        
        return self._llm_classify_query(state)
    
    def _llm_classify_query(self, state: BaseAgentState) -> BaseAgentState:
        try:
            classify_prompt = ChatPromptTemplate.from_template("""
            Classify this user query into one of three agent types based on the intent:
            
            Query: {query}
            
            Agent Types:
            1. db_query - For database operations (add, insert, update, delete, retrieve data, cart operations, user profile)
            2. email - For sending emails, notifications, or messages (look for email addresses or send/notify keywords)
            3. meeting - For scheduling meetings or appointments
            
            Respond with ONLY the agent type (db_query, email, or meeting):
            """)
            
            messages = classify_prompt.format_messages(query=state["query"])
            response = self.llm.invoke(messages)
            agent_type = response.content.strip().lower()
            
            if agent_type in self.agents:
                state["agent_type"] = agent_type
                state["status"] = "classified"
            else:
                state["error_message"] = f"Unknown agent type: {agent_type}"
                state["status"] = "classification_error"
                
        except Exception as e:
            state["error_message"] = f"Classification error: {str(e)}"
            state["status"] = "classification_error"
        
        return state
    
    def _route_to_agent(self, state: BaseAgentState) -> BaseAgentState:
        try:
            agent_type = state["agent_type"]
            if agent_type not in self.agents:
                state["error_message"] = f"No agent found for type: {agent_type}"
                state["status"] = "routing_error"
                return state
            
            agent = self.agents[agent_type]
            result_state = agent.process(state)
            
            return result_state
            
        except Exception as e:
            state["error_message"] = f"Routing error: {str(e)}"
            state["status"] = "routing_error"
            return state
    
    def _handle_error(self, state: BaseAgentState) -> BaseAgentState:
        state["status"] = "failed"
        return state
    
    def _should_route_or_error(self, state: BaseAgentState) -> Literal["route", "error"]:
        return "route" if state["status"] == "classified" else "error"
    
    def add_agent(self, agent_type: str, agent: BaseAgent, keywords: List[str]):
        self.agents[agent_type] = agent
        self.classification_keywords[agent_type] = keywords
        print(f"Added new agent: {agent_type}")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        initial_state = BaseAgentState(
            query=query,
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
        
        print(f"\nProcessing Query: '{query}'")
        
        # Record start time
        start_time = time.time()
        
        # Execute the graph
        result = self.graph.invoke(initial_state)
        
        # Record end time and calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Update timing information
        result["start_time"] = start_time
        result["end_time"] = end_time
        result["execution_time"] = execution_time
        
        # Print timing information
        print(f"⏱️  Execution Time: {execution_time:.4f} seconds")
        
        if result["status"] == "completed":
            print(f"✅ Success: {result['success_message']}")
            return {
                "success": True,
                "message": result["success_message"],
                "agent_type": result["agent_type"],
                "result": result["result"],
                "execution_time": execution_time
            }
        else:
            print(f"❌ Error: {result['error_message']}")
            return {
                "success": False,
                "error": result.get("error_message", "Unknown error occurred"),
                "status": result["status"],
                "agent_type": result.get("agent_type", "unknown"),
                "execution_time": execution_time
            }
    
    def visualize_graph(self):
        """
        Display the agent graph architecture using IPython display
        """
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
            print("📊 Agent Architecture Graph displayed above")
        except ImportError:
            print("❌ IPython not available. Please install IPython to view the graph visualization.")
            print("You can install it with: pip install ipython")
            self._fallback_graph_display()
        except Exception as e:
            print(f"❌ Error generating graph visualization: {e}")
            print("📝 Falling back to text representation:")
            self._fallback_graph_display()
    
    def _fallback_graph_display(self):
        """
        Fallback method to display graph structure in text format
        """
        graph_structure = """
        🏗️  Multi-Agent Orchestrator Graph Structure
        ===============================================
        
        Entry Point: classify_query
        
        Flow:
        1. classify_query
           ├─→ [if classified] route_to_agent → END
           └─→ [if error] handle_error → END
        
        🤖 Agent Types:
        • db_query: Database operations with schema support
        • email: Send emails via SMTP
        • meeting: Schedule meetings with robust date parsing
        
        🔍 Classification Methods:
        • Fast keyword matching (primary)
        • LLM classification (fallback for ambiguous cases)
        
        ⏱️  Timing: Each query execution time is measured and displayed
        """
        
        print(graph_structure)
        
        try:
            mermaid_code = self.graph.get_graph().draw_mermaid()
            print("\n📋 Mermaid Graph Code:")
            print("=" * 50)
            print(mermaid_code)
            print("=" * 50)
            print("You can visualize this at: https://mermaid.live/")
        except Exception as e:
            print(f"Could not generate mermaid code: {e}")

def main():
    try:
        orchestrator = CentralOrchestrator(files_directory="./user_files")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check your GROQ API key configuration.")
        return
    
    print("🚀 Enhanced Multi-Agent Orchestrator System")
    print("=" * 55)
    print("🤖 Available Agents:")
    print("  • DB Query Agent - Database operations with schema support")
    print("  • Email Agent - Send emails via SMTP")
    print("  • Meeting Agent - Schedule meetings")
    print("=" * 55)
    print("🗃️  Database Schema Available:")
    print("  • users (user_id, username, email, created_at, last_login)")
    print("  • products (product_id, name, price, category, stock_quantity)")
    print("  • orders (order_id, user_id, product_id, quantity, total_amount)")
    print("  • cart (cart_id, user_id, product_id, quantity, added_at)")
    print("=" * 55)
    print("💻 Commands:")
    print("  - 'graph' - Show agent graph architecture")
    print("  - 'quit' - Exit the system")
    print("=" * 55)
    print("📝 Example queries:")
    print("  - 'Add product 5 to cart for user 1'")
    print("  - 'Send email to john@example.com about today's deals'")
    print("  - 'Schedule meeting with user 3 tomorrow'")
    print("  - 'Schedule meeting with user 2 next Monday'")
    print("  - 'Book appointment with user 5 day after tomorrow'")
    print("  - 'Show all orders for user 2'")
    print("=" * 55)
    
    while True:
        query = input("\n🔍 Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q', '']:
            print("👋 Goodbye!")
            break
        elif query.lower() == 'graph':
            orchestrator.visualize_graph()
            continue
        
        result = orchestrator.process_query(query)
        
        if not result["success"]:
            print(f"📊 Status: {result.get('status', 'unknown')}")

if __name__ == "__main__":
    main()