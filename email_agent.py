import os
import re
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from base_agent import BaseAgent, BaseAgentState, EmailAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

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
            
            # Check for intermediate results from previous steps (e.g., meeting scheduling)
            previous_context = ""
            intermediate_results = state.get("intermediate_results", {})
            if intermediate_results:
                logger.info(f"📧 Email agent found {len(intermediate_results)} previous step(s)")
                previous_context = self._format_intermediate_results(intermediate_results)
            
            # Check if query references previous data/query results
            cached_data_context = ""
            query_lower = state["query"].lower()
            data_reference_phrases = [
                "that data", "this data", "the data", "those results", 
                "previous query", "database results", "query results",
                "send the results", "send data", "send that", "send this",
                "regarding the meet", "about the meeting", "for the meeting"
            ]
            
            references_data = any(phrase in query_lower for phrase in data_reference_phrases)
            
            if references_data and state.get("session_id") and state.get("user_id"):
                try:
                    # Import the standalone function for getting cached results
                    import sys
                    import os
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    from redis_memory_manager import get_last_db_query_result
                    
                    # Get the last database query result
                    cached_result = get_last_db_query_result(
                        state["session_id"], 
                        state.get("user_id")
                    )
                    
                    if cached_result:
                        # Format the cached data for email
                        cached_data_context = self._format_cached_data_for_email(cached_result)
                        logger.info(f"Retrieved cached data for email: {len(cached_data_context)} chars")
                    else:
                        logger.warning("No cached database results found for this session")
                        
                except Exception as cache_error:
                    logger.error(f"Error retrieving cached data: {str(cache_error)}")
            
            content_prompt = ChatPromptTemplate.from_template("""
            You are an email content and recipient parser. Analyze the user's request and extract email details.
            
            User Request: {query}
            
            {previous_context}
            
            {cached_data}
            
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
            - If previous context is provided (meeting details, etc.), INCLUDE the relevant information in the email body
            - If cached data is provided, INCLUDE IT in the email body in a clear, formatted way
            - End with appropriate closing
            """)
            
            messages = content_prompt.format_messages(
                query=state["query"],
                previous_context=previous_context if previous_context else "No previous context available.",
                cached_data=cached_data_context if cached_data_context else "No cached data available."
            )
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Track token usage
            track_llm_call(
                input_prompt=messages,
                output=content,
                agent_type="email",
                operation="compose_email",
                model_name="gpt-4.1-mini"
            )
            
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
    
    def _format_cached_data_for_email(self, cached_result: dict) -> str:
        """
        Format cached database query results for inclusion in email content.
        
        Args:
            cached_result: Dict containing query results, may include DataFrames
            
        Returns:
            Formatted string representation of the data
        """
        try:
            # Look for query results in various field names
            result_fields = ["query_results", "query_data", "data", "rows_returned"]
            query_data = None
            
            for field in result_fields:
                if field in cached_result:
                    query_data = cached_result[field]
                    break
            
            if query_data is None:
                return "Cached Data:\nNo query results found in cached data."
            
            # Check if it's a serialized DataFrame (from our _serialize_result method)
            if isinstance(query_data, dict) and query_data.get("_type") == "dataframe":
                return self._format_dataframe_dict(query_data)
            
            # Handle pandas DataFrame directly (in case it wasn't serialized)
            try:
                import pandas as pd
                if isinstance(query_data, pd.DataFrame):
                    return self._format_dataframe_direct(query_data)
            except ImportError:
                pass
            
            # Handle list of dicts
            if isinstance(query_data, list) and query_data and isinstance(query_data[0], dict):
                return self._format_list_of_dicts(query_data)
            
            # Fallback to string representation
            return f"Cached Data:\n{str(query_data)}"
            
        except Exception as e:
            logger.error(f"Error formatting cached data: {str(e)}")
            return f"Cached Data: (Error formatting data: {str(e)})"
    
    def _format_dataframe_dict(self, df_dict: dict) -> str:
        """Format a serialized DataFrame dictionary into a readable table."""
        try:
            columns = df_dict.get("columns", [])
            data = df_dict.get("data", [])
            
            if not columns or not data:
                return "Cached Data:\nEmpty dataset"
            
            # Create table header
            header = " | ".join(columns)
            separator = "-+-".join(["-" * len(col) for col in columns])
            
            # Create table rows
            rows = []
            for row in data:
                formatted_row = " | ".join([str(row.get(col, "")) for col in columns])
                rows.append(formatted_row)
            
            table = f"\nCached Database Query Results ({len(data)} rows):\n\n{header}\n{separator}\n" + "\n".join(rows)
            return table
            
        except Exception as e:
            logger.error(f"Error formatting DataFrame dict: {str(e)}")
            return f"Cached Data: (Error formatting: {str(e)})"
    
    def _format_dataframe_direct(self, df) -> str:
        """Format a pandas DataFrame directly into a readable table."""
        try:
            # Convert to string with nice formatting
            table_str = df.to_string(index=False)
            return f"\nCached Database Query Results ({len(df)} rows):\n\n{table_str}"
        except Exception as e:
            logger.error(f"Error formatting DataFrame: {str(e)}")
            return f"Cached Data: (Error formatting: {str(e)})"
    
    def _format_list_of_dicts(self, data_list: list) -> str:
        """Format a list of dictionaries into a readable table."""
        try:
            if not data_list:
                return "Cached Data:\nEmpty dataset"
            
            # Get column names from first dict
            columns = list(data_list[0].keys())
            
            # Create table header
            header = " | ".join(columns)
            separator = "-+-".join(["-" * len(col) for col in columns])
            
            # Create table rows
            rows = []
            for item in data_list:
                formatted_row = " | ".join([str(item.get(col, "")) for col in columns])
                rows.append(formatted_row)
            
            table = f"\nCached Database Query Results ({len(data_list)} rows):\n\n{header}\n{separator}\n" + "\n".join(rows)
            return table
            
        except Exception as e:
            logger.error(f"Error formatting list of dicts: {str(e)}")
            return f"Cached Data: (Error formatting: {str(e)})"
    
    def _format_intermediate_results(self, intermediate_results: dict) -> str:
        """
        Format intermediate results from previous workflow steps for email context.
        Extracts meeting details, query results, or other relevant context.
        
        Args:
            intermediate_results: Dict mapping step keys to their results
            
        Returns:
            Formatted string with context from previous steps
        """
        try:
            context_parts = []
            
            for step_key, step_data in intermediate_results.items():
                if not isinstance(step_data, dict):
                    continue
                
                # Check for meeting-related data
                if step_data.get("agent_type") == "meeting" or "meeting" in step_key.lower():
                    meeting_info = []
                    meeting_info.append("\n=== Meeting Details ===")
                    
                    # Extract meeting participant
                    if step_data.get("user_name"):
                        meeting_info.append(f"Participant: {step_data['user_name']}")
                    elif step_data.get("user_id"):
                        meeting_info.append(f"Participant: User {step_data['user_id']}")
                    
                    # Extract meeting date/time
                    if step_data.get("meeting_date"):
                        meeting_info.append(f"Date: {step_data['meeting_date']}")
                    elif step_data.get("scheduled_date"):
                        meeting_info.append(f"Date: {step_data['scheduled_date']}")
                    
                    # Extract meeting topic/agenda
                    if step_data.get("meeting_topic"):
                        meeting_info.append(f"Topic: {step_data['meeting_topic']}")
                    elif step_data.get("agenda"):
                        meeting_info.append(f"Agenda: {step_data['agenda']}")
                    elif step_data.get("description"):
                        meeting_info.append(f"Description: {step_data['description']}")
                    
                    # Extract meeting location if available
                    if step_data.get("location"):
                        meeting_info.append(f"Location: {step_data['location']}")
                    
                    # Extract any success message or result details
                    if step_data.get("success_message"):
                        meeting_info.append(f"Status: {step_data['success_message']}")
                    
                    if len(meeting_info) > 1:  # More than just the header
                        context_parts.append("\n".join(meeting_info))
                
                # Check for database query results
                elif step_data.get("agent_type") == "db_query" or "query" in step_key.lower():
                    if step_data.get("query_data") or step_data.get("query_results"):
                        context_parts.append("\n=== Previous Query Results Available ===")
                        context_parts.append("(Database results from previous step)")
                
                # Check for campaign or other agent results
                elif step_data.get("result"):
                    result = step_data["result"]
                    if isinstance(result, dict):
                        result_info = [f"\n=== {step_data.get('agent_type', 'Previous Step').title()} Results ==="]
                        for key, value in result.items():
                            if key not in ["query_data", "raw_data"]:  # Skip large data fields
                                result_info.append(f"{key}: {value}")
                        if len(result_info) > 1:
                            context_parts.append("\n".join(result_info))
            
            if context_parts:
                return "\n\n".join(context_parts)
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error formatting intermediate results: {str(e)}")
            return ""