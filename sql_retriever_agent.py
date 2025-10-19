import os
import pickle
import logging
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Any
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from base_agent import BaseAgent, BaseAgentState
from token_tracker import track_llm_call

logger = logging.getLogger(__name__)

class SQLRetrieverAgent(BaseAgent):
    def __init__(self, llm, embedding_file_path: str = "embeddings.pkl"):
        super().__init__(llm)
        self.embedding_file_path = embedding_file_path
        self.embeddings = None
        self.indexed_questions = None
        self.openai_client = None
        self._initialize_retriever()
    
    def get_agent_type(self) -> str:
        return "sql_retriever"
    
    def _initialize_retriever(self):
        """Initialize the retriever by loading embeddings and setting up OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.openai_client = OpenAI(api_key=api_key)
            
            self._load_embeddings()
            
            logger.info(f"SQLRetrieverAgent initialized successfully")
            logger.info(f"Loaded {len(self.indexed_questions)} indexed questions")
            logger.info(f"Embeddings shape: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLRetrieverAgent: {e}")
            raise
    
    def _load_embeddings(self):
        """Load embeddings and indexed questions from pickle file"""
        try:
            with open(self.embedding_file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract embeddings and indexed questions
            self.embeddings = data["embeddings"]  
            self.indexed_questions = data["indexed_questions"]  
            
            # Validate data
            if not isinstance(self.embeddings, np.ndarray):
                raise ValueError("Embeddings should be a NumPy array")
            
            if not isinstance(self.indexed_questions, list):
                raise ValueError("Indexed questions should be a list")
            
            if len(self.embeddings) != len(self.indexed_questions):
                raise ValueError("Number of embeddings and indexed questions must match")
            
            logger.info(f"Successfully loaded embeddings from {self.embedding_file_path}")
            
        except FileNotFoundError:
            error_msg = f"Embedding file not found: {self.embedding_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Error loading embeddings: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
        """Generate embeddings for given texts using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(model=model, input=texts)
            
            # Track token usage for embeddings
            total_tokens = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else len(' '.join(texts)) // 4
            track_llm_call(
                input_prompt=texts,
                output="",  # Embeddings don't have text output
                agent_type="sql_retriever",
                operation="generate_embeddings",
                model_name=model
            )
            
            return np.array([r.embedding for r in response.data])
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _enhance_similarity_scores(self, user_query: str, similarities: np.ndarray) -> np.ndarray:
        """
        Enhance similarity scores by boosting relevance for time-based and domain-specific queries
        
        Args:
            user_query (str): The user's query
            similarities (np.ndarray): Original cosine similarities
            
        Returns:
            np.ndarray: Enhanced similarity scores
        """
        enhanced_similarities = similarities.copy()
        user_query_lower = user_query.lower()
        
        # Time-based query keywords that should get priority
        time_keywords = [
            'past year', 'last year', 'last 12 months', '12 months',
            'past 6 months', 'last 6 months', '6 months', 
            'past 3 months', 'last 3 months', '3 months',
            'monthly trend', 'monthly sales', 'month wise', 'monthly data',
            'this month', 'last month', 'this year', 'trend', 'over time',
            'time series', 'quarterly', 'yearly', 'annual'
        ]
        
        # Sales/revenue keywords
        sales_keywords = [
            'sales', 'revenue', 'dispatch', 'invoice', 'value', 'amount',
            'total sales', 'sales data', 'sales trend', 'revenue trend'
        ]
        
        # Visualization keywords 
        viz_keywords = [
            'visualize', 'chart', 'graph', 'plot', 'trend', 'visualization'
        ]
        
        # Check if user query contains time-based terms
        is_time_query = any(keyword in user_query_lower for keyword in time_keywords)
        is_sales_query = any(keyword in user_query_lower for keyword in sales_keywords)  
        is_viz_query = any(keyword in user_query_lower for keyword in viz_keywords)
        
        # Boost scores for relevant indexed questions
        for idx, question_data in enumerate(self.indexed_questions):
            question = question_data["question"].lower()
            sql = question_data["sql"].lower()
            
            boost_factor = 1.0
            
            # Boost time-based queries
            if is_time_query:
                if any(keyword in question for keyword in time_keywords):
                    boost_factor *= 1.3
                if any(keyword in sql for keyword in ['date_trunc', 'interval', 'month', 'year']):
                    boost_factor *= 1.2
            
            # Boost sales queries
            if is_sales_query:
                if any(keyword in question for keyword in sales_keywords):
                    boost_factor *= 1.2
                if any(keyword in sql for keyword in ['customerinvoice', 'dispatchedvalue', 'sales']):
                    boost_factor *= 1.2
            
            # Boost visualization-related queries
            if is_viz_query:
                if any(keyword in question for keyword in viz_keywords + ['trend', 'analysis']):
                    boost_factor *= 1.15
            
            # Specific boosts for monthly/time series patterns
            if 'monthly' in user_query_lower and 'monthly' in question:
                boost_factor *= 1.4
            
            if ('past year' in user_query_lower or 'last 12 months' in user_query_lower):
                if any(term in question for term in ['12 months', 'past year', 'last year', 'annual']):
                    boost_factor *= 1.5
                if '12 months' in sql or 'interval' in sql:
                    boost_factor *= 1.3
            
            # Apply the boost
            enhanced_similarities[idx] = min(similarities[idx] * boost_factor, 1.0)  # Cap at 1.0
        
        return enhanced_similarities
    
    def retrieve_similar_sql(self, user_query: str, k: int = 6) -> List[str]:
        """
        Retrieve top-k most similar SQL queries for the given user query with enhanced relevance scoring
        
        Args:
            user_query (str): The user's natural language query
            k (int): Number of similar SQL queries to return
            
        Returns:
            List[str]: List of top-k most similar SQL queries
        """
        import time
        start_time = time.time()
        
        try:
            # Generate embedding for the user query
            current_embedding = self.get_embeddings([user_query])[0]
            
            # Compute cosine similarity with all stored embeddings
            similarities = cosine_similarity([current_embedding], self.embeddings)[0]
            
            # Enhanced relevance scoring for time-based and visualization queries
            enhanced_similarities = self._enhance_similarity_scores(user_query, similarities)
            
            # Get top-k indices (sorted by descending similarity)
            top_indices = np.argsort(enhanced_similarities)[-k:][::-1]  # Reverse to get descending order
            
            # Clean logging using SQLLogger
            try:
                from clean_logging import SQLLogger
                SQLLogger.retrieval_complete(len(top_indices), enhanced_similarities[top_indices[0]])
            except ImportError:
                logger.info(f"Retrieved {len(top_indices)} similar queries (max similarity: {enhanced_similarities[top_indices[0]]:.3f})")
            
            similar_sqls = []
            similar_queries_detailed = []
            
            for rank, idx in enumerate(top_indices, 1):
                sql_query = self.indexed_questions[idx]["sql"]
                similarity_score = enhanced_similarities[idx]  # Use enhanced score for logging
                original_similarity = similarities[idx]  # Keep original for reference
                question = self.indexed_questions[idx]["question"]
                
                # Return structured data instead of just SQL strings
                similar_sqls.append({
                    'question': question,
                    'sql': sql_query,
                    'similarity': similarity_score,
                    'rank': rank
                })
                similar_queries_detailed.append({
                    'question': question,
                    'sql': sql_query,
                    'similarity': similarity_score,
                    'original_similarity': original_similarity
                })
            
            # Log the retrieved queries using the new method
            try:
                SQLLogger.retrieved_queries(similar_queries_detailed, max_display=6)
            except ImportError:
                for rank, query_info in enumerate(similar_queries_detailed[:6], 1):
                    logger.info(f"  {rank}. ({query_info['similarity']:.3f}) {query_info['question'][:80]}...")
                    logger.info(f"     SQL: {query_info['sql'][:120]}...")
                
                # Log SQL type and complexity
                # sql_lower = sql_query.lower().strip()
                # if sql_lower.startswith('select'):
                #     sql_type = 'SELECT'
                # elif sql_lower.startswith('insert'):
                #     sql_type = 'INSERT'
                # elif sql_lower.startswith('update'):
                #     sql_type = 'UPDATE'
                # elif sql_lower.startswith('delete'):
                #     sql_type = 'DELETE'
                # elif sql_lower.startswith('with'):
                #     sql_type = 'CTE (Common Table Expression)'
                # else:
                #     sql_type = 'OTHER'
                
                # # Count joins and complexity indicators
                # join_count = sql_query.lower().count(' join ')
                # has_group_by = 'group by' in sql_query.lower()
                # has_order_by = 'order by' in sql_query.lower()
                # has_where = 'where' in sql_query.lower()
                
                # logger.info(f"SQL Type: {sql_type}")
                # logger.info(f"JOIN Count: {join_count}")
                # logger.info(f"Has GROUP BY: {has_group_by} | ORDER BY: {has_order_by} | WHERE: {has_where}")
                # logger.info(f"-" * 60)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            similarity_scores = enhanced_similarities[top_indices]
            # self.log_performance_stats(user_query, similarity_scores.tolist(), execution_time, len(similar_sqls))
            
            return similar_sqls
            
        except Exception as e:
            logger.error(f"Error retrieving similar SQL queries: {e}")
            return []  # Return empty list on error
    
    def get_top_k_similar_questions(self, current_question: str, k: int = 6) -> List[dict]:
        """
        Get top-k similar questions with their SQL queries and similarity scores
        
        Args:
            current_question (str): The current question to find similarities for
            k (int): Number of similar questions to return
            
        Returns:
            List[dict]: List of similar question-SQL pairs with similarity scores
        """
        try:
            current_embedding = self.get_embeddings([current_question])[0]
            
            similarities = cosine_similarity([current_embedding], self.embeddings)[0]
            
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                result = {
                    "question": self.indexed_questions[idx]["question"],
                    "sql": self.indexed_questions[idx]["sql"],
                    "similarity": similarities[idx]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting similar questions: {e}")
            return []
    
    def process(self, state: BaseAgentState) -> BaseAgentState:
        """
        Process method required by BaseAgent interface
        This retrieves similar SQL queries and stores them in the state
        """
        try:
            user_query = state["query"]
            
            similar_sqls = self.retrieve_similar_sql(user_query, k=20)
            
            state["status"] = "completed"
            state["success_message"] = f"Retrieved {len(similar_sqls)} similar SQL queries"
            state["result"] = {
                "similar_sqls": similar_sqls,
                "query_count": len(similar_sqls),
                "source": "sql_retriever_agent"
            }
            
            logger.info(f"SQLRetrieverAgent processed query successfully: {len(similar_sqls)} results")
            
        except Exception as e:
            state["error_message"] = f"SQL retrieval error: {str(e)}"
            state["status"] = "failed"
            logger.error(f"SQLRetrieverAgent process failed: {e}")
        
        return state
    
    def log_performance_stats(self, query: str, similarity_scores: List[float], 
                            execution_time: float, retrieved_count: int) -> None:
        """Log performance statistics to a file for analysis"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "execution_time_ms": execution_time * 1000,
            "retrieved_count": retrieved_count,
            "avg_similarity": float(np.mean(similarity_scores)) if similarity_scores else 0.0,
            "max_similarity": float(np.max(similarity_scores)) if similarity_scores else 0.0,
            "min_similarity": float(np.min(similarity_scores)) if similarity_scores else 0.0,
            "similarity_distribution": {
                "above_0.8": sum(1 for s in similarity_scores if s > 0.8),
                "above_0.6": sum(1 for s in similarity_scores if s > 0.6),
                "above_0.4": sum(1 for s in similarity_scores if s > 0.4),
                "below_0.4": sum(1 for s in similarity_scores if s <= 0.4)
            }
        }
        
        # Append to performance log file
        log_file = "sql_retriever_performance.json"
        try:
            # Try to load existing data
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {"performance_logs": []}
            
            # Add new stats
            data["performance_logs"].append(stats)
            
            # Keep only last 1000 entries to prevent file from growing too large
            if len(data["performance_logs"]) > 1000:
                data["performance_logs"] = data["performance_logs"][-1000:]
            
            # Save updated data
            with open(log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Performance stats logged to {log_file}")
        except Exception as e:
            logger.error(f"Error logging performance stats: {e}")
    
    # def get_performance_summary(self) -> Dict[str, Any]:
    #     """Get a summary of recent retrieval performance"""
    #     log_file = "sql_retriever_performance.json"
    #     if not os.path.exists(log_file):
    #         return {"message": "No performance data available"}
        
    #     try:
    #         with open(log_file, 'r') as f:
    #             data = json.load(f)
            
    #         logs = data.get("performance_logs", [])
    #         if not logs:
    #             return {"message": "No performance data available"}
            
    #         # Calculate summary statistics
    #         recent_logs = logs[-50:]  # Last 50 queries
    #         avg_times = [log["execution_time_ms"] for log in recent_logs]
    #         avg_similarities = [log["avg_similarity"] for log in recent_logs]
            
    #         summary = {
    #             "total_queries": len(logs),
    #             "recent_queries": len(recent_logs),
    #             "avg_execution_time_ms": np.mean(avg_times),
    #             "avg_similarity_score": np.mean(avg_similarities),
    #             "performance_trend": {
    #                 "fast_queries": sum(1 for t in avg_times if t < 100),
    #                 "medium_queries": sum(1 for t in avg_times if 100 <= t < 500),
    #                 "slow_queries": sum(1 for t in avg_times if t >= 500)
    #             },
    #             "quality_distribution": {
    #                 "high_quality": sum(1 for s in avg_similarities if s > 0.7),
    #                 "medium_quality": sum(1 for s in avg_similarities if 0.4 <= s <= 0.7),
    #                 "low_quality": sum(1 for s in avg_similarities if s < 0.4)
    #             }
    #         }
            
    #         return summary
    #     except Exception as e:
    #         return {"error": f"Error reading performance data: {e}"}