import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from base_agent import BaseAgent, BaseAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

class SQLQueryDecomposer(BaseAgent):
    """
    Agent responsible for analyzing complex SQL queries and breaking them down
    into sequential steps that can be executed individually.
    """
    
    def __init__(self, llm):
        super().__init__(llm)
        self.decomposition_examples = self._load_decomposition_examples()
    
    def get_agent_type(self) -> str:
        return "sql_decomposer"
    
    def _load_decomposition_examples(self) -> List[Dict[str, Any]]:
        """Load examples of query decomposition for training the LLM"""
        return [
            {
                "original_question": "Show me the sales trend of the top 3 SKUs in the last month.",
                "decomposed_questions": [
                    "Get the top 3 SKUs by sales in the last month",
                    "Show me the sales trend for these SKUs over the last year"
                ],
                "explanation": "Need to first identify the top SKUs, then analyze their trends"
            },
            {
                "original_question": "Find customers who placed orders both last month and this month.",
                "decomposed_questions": [
                    "List customers who placed orders last month",
                    "Out of these customers, who placed orders this month"
                ],
                "explanation": "Need to find last month's customers first, then filter for current month"
            },
            {
                "original_question": "Show me customers who had degrowth in the last 3 months, and their top 3 purchased SKUs during this period.",
                "decomposed_questions": [
                    "Get the list of customers who had degrowth in the last 3 months",
                    "Show me the top 3 SKUs purchased by these customers in the last 3 months"
                ],
                "explanation": "First identify customers with degrowth, then find their top SKUs"
            },
            {
                "original_question": "Find the sales value in the last 6 months for products that had degrowth last month compared to the previous month.",
                "decomposed_questions": [
                    "Find SKUs with degrowth last month compared to the previous month",
                    "Show me the sales value for these SKUs in the last 6 months"
                ],
                "explanation": "First identify products with degrowth, then get their 6-month sales data"
            }
        ]
    
    def _create_system_message(self, content: str) -> dict:
        """Create a system message for chat completions"""
        return {"role": "system", "content": content}
    
    def _create_user_message(self, content: str) -> dict:
        """Create a user message for chat completions"""
        return {"role": "user", "content": content}
    
    def _create_assistant_message(self, content: str) -> dict:
        """Create an assistant message for chat completions"""
        return {"role": "assistant", "content": content}
    
    def analyze_query_complexity(self, question: str) -> Dict[str, Any]:
        """
        Analyze if a question requires multiple SQL queries to answer.
        
        Args:
            question: The user's question to analyze
            
        Returns:
            Dict containing analysis results and decomposed questions if needed
        """
        try:
            decomposition_prompt = self._create_decomposition_prompt()
            
            message_log = [self._create_system_message(decomposition_prompt)]
            
            for example in self.decomposition_examples:
                message_log.append(self._create_user_message(example["original_question"]))
                response = json.dumps(example["decomposed_questions"])
                message_log.append(self._create_assistant_message(response))
            
            single_step_examples = [
                {
                    "question": "Show me customer-wise order value for the last 3 months.",
                    "answer": ["Show me customer-wise order value for the last 3 months"]
                },
                {
                    "question": "Get the top 5 SKUs by order value in the last 6 months.",
                    "answer": ["Get the top 5 SKUs by order value in the last 6 months"]
                },
                {
                    "question": "List all distributors and their total sales in the last quarter.",
                    "answer": ["List all distributors and their total sales in the last quarter"]
                },
                {
                    "question": "Show me the percentage degrowth of each category in Q1-2024 compared to Q4-2023.",
                    "answer": ["Show me the percentage degrowth of each category in Q1-2024 compared to Q4-2023"]
                }
            ]
            
            for example in single_step_examples:
                message_log.append(self._create_user_message(example["question"]))
                response = json.dumps(example["answer"])
                message_log.append(self._create_assistant_message(response))
            
            message_log.append(self._create_user_message(f"User question: '{question}'"))
            
            # Get LLM response
            response = self.llm.invoke(message_log)
            content = response.content.strip()
            
            # Track token usage
            track_llm_call(
                input_prompt=message_log,
                output=content,
                agent_type="sql_decomposer",
                operation="analyze_query",
                model_name="gpt-4o"
            )
            
            try:
                if content.startswith('[') and content.endswith(']'):
                    decomposed_questions = json.loads(content)
                else:
                    json_match = re.search(r'\\[.*\\]', content, re.DOTALL)
                    if json_match:
                        decomposed_questions = json.loads(json_match.group(0))
                    else:
                        # Fallback: assume it's a single question
                        decomposed_questions = [question]
                
                if not isinstance(decomposed_questions, list):
                    decomposed_questions = [question]
                
                is_multi_step = len(decomposed_questions) > 1
                
                analysis_result = {
                    "is_multi_step": is_multi_step,
                    "question_count": len(decomposed_questions),
                    "decomposed_questions": decomposed_questions,
                    "original_question": question,
                    "analysis_successful": True
                }
                
                logger.info(f"Query analysis complete: {question}")
                logger.info(f"Multi-step: {is_multi_step}, Steps: {len(decomposed_questions)}")
                
                return analysis_result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse decomposition response: {e}")
                return {
                    "is_multi_step": False,
                    "question_count": 1,
                    "decomposed_questions": [question],
                    "original_question": question,
                    "analysis_successful": False,
                    "error": "Failed to parse LLM response"
                }
                
        except Exception as e:
            logger.error(f"Query decomposition error: {e}")
            return {
                "is_multi_step": False,
                "question_count": 1,
                "decomposed_questions": [question],
                "original_question": question,
                "analysis_successful": False,
                "error": f"Decomposition analysis failed: {str(e)}"
            }
    
    def _create_decomposition_prompt(self) -> str:
        """Create the system prompt for query decomposition"""
        return """You are an expert AI assistant responsible for analyzing user questions to determine whether they can be translated into a single SQL query, or if they must be broken into two sequential questions, where the result of the first question is required to construct the WHERE clause of the second SQL query.

1. Refer to the context provided for questions that can be answered using just a single question.

2. When to Split into Two Sequential Questions
   - Split the question ONLY if:
     - Question is related to degrowth of something and there is additional information asked about the degrown entity.
     - You need data from one time frame and then check something about them in another time frame.
     - The WHERE clause depends on dynamically generated values (e.g., top N customers, best-selling SKUs).
     - The final query's WHERE clause depends on a dynamic list of values extracted by a prior query.
     - You need to rank or filter entities (top N customers, bottom SKUs, etc.), and those entities need to be used in a second query.
       - If the information about the top/bottom N is possible in single query do not split.
     - Using a subquery in the WHERE clause would be complex or inefficient, and separating queries makes it clearer.
     - ONLY split if the values weren't given explicitly, and you had to first get the list dynamically (e.g., "top 5 SKUs" or "SKUs with sales drop").
   
   - Examples Where Two SQL Queries Are Needed:
     - Example 1: Sales Trend for Top 3 SKUs
       Original Question: "Show me the sales trend of the top 3 SKUs in the last month."
       Reason for Split: The final SQL WHERE clause requires the names of the top 3 SKUs, which are determined dynamically from the first query.
       Output: ["Get the top 3 SKUs by sales in the last month", "Show me the sales trend for these SKUs over the last year"]
     - Example 2: Customers with Orders in Both Last Month and This Month
       Original Question: "Find customers who placed orders both last month and this month."
       Reason for Split: You need to first find customers who ordered last month. Then you filter those who also ordered this month using the output of the first query.
       Output: ["List customers who placed orders last month", "Out of these customers, who placed orders this month"]

3. When NOT to Split (Single Query Cases)
   - Do NOT split if:
     - If the answer to the question can be generated via a single query using GROUP BY
     - Question is related to degrowth of something and there is no additional information asked about the degrown entity.
     - If the question asks to compare two entities of the same type in the same timeframe.
     - If values for the WHERE clause are already known, do not split.
     - The query doesn't need to reuse dynamic data in the WHERE clause.
     - You can group by, aggregate, or filter directly without a subquery.
     - The user does not explicitly request additional details, comparisons, or next steps.
       - Example: "Get the top 2 SKUs by order value in the last 3 months." (Single step)
       - Example: "Get the bottom 10 customers by order value in the last 9 months" (Single step)
     - If the information (ranking + additional attribute) can be obtained with grouping, ranking, and simple aggregation in a single query, DO NOT split.
     - The final answer is directly retrievable without sequential processing.

4. Rules for Generating Questions
   - Think like you're generating SQL queries.
   - Split the question only if the WHERE clause in the second SQL query requires dynamic values that come from the result of a previous query.
   - If a single SQL query can efficiently handle the logic (e.g., using JOINs or subqueries that are not complex), do not split.
   - Ensure the first question retrieves the data necessary for the WHERE clause of the second question.
   - Maintain consistency in timeframes, filters, and groupings between the two questions.
   - Assign timeframes to both the questions appropriately.

5. Output Format
   - Output should be STRICTLY a list of strings in JSON format.
   - If splitting is required: ["QUESTION_1_HERE", "QUESTION_2_HERE"]
   - If no split is needed: ["original input question"]
   - No additional explanation or formatting.

Examples of correct outputs:
- ["Show me customer-wise order value for the last 3 months"]
- ["Get the top 3 SKUs by sales in the last month", "Show me the sales trend for these SKUs over the last year"]
- ["Find SKUs with degrowth last month compared to the previous month", "Show me the sales value for these SKUs in the last 6 months"]"""
    
    def process(self, state: BaseAgentState) -> BaseAgentState:
        """
        Process method required by BaseAgent interface.
        Analyzes query complexity and returns decomposition results.
        """
        try:
            question = state["query"]
            analysis_result = self.analyze_query_complexity(question)
            
            state["status"] = "completed"
            state["success_message"] = "Query analysis completed successfully"
            state["result"] = analysis_result
            
            state["is_multi_step"] = analysis_result["is_multi_step"]
            state["remaining_tasks"] = analysis_result["decomposed_questions"]
            state["current_step"] = 0
            
            logger.info(f"SQLQueryDecomposer completed analysis for: {question}")
            
        except Exception as e:
            state["error_message"] = f"Query decomposition error: {str(e)}"
            state["status"] = "failed"
            logger.error(f"SQLQueryDecomposer process failed: {e}")
        
        return state