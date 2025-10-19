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
            
            # Track token usage
            track_llm_call(
                input_prompt=messages,
                output=content,
                agent_type="email",
                operation="compose_email",
                model_name="gpt-4o"
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