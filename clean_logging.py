from loguru import logger
import sys

def setup_logging():
    """Setup clean logging configuration"""
    logger.remove()
    
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file handler for detailed logs
    logger.add(
        "logs/system.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days"
    )

class AgentLogger:
    @staticmethod
    def query_start(agent_type: str, query: str):
        logger.info(f"{agent_type.upper()} | Processing: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    @staticmethod
    def query_complete(agent_type: str, execution_time: float):
        logger.success(f"{agent_type.upper()} | Completed in {execution_time:.2f}s")
    
    @staticmethod
    def query_error(agent_type: str, error: str):
        logger.error(f"{agent_type.upper()} | Error: {error}")

class SQLLogger:
    @staticmethod
    def retrieval_start(query: str, k: int):
        logger.info(f"SQL RETRIEVAL | Finding {k} similar queries for: {query[:50]}...")
    
    @staticmethod
    def retrieval_complete(count: int, max_similarity: float):
        logger.info(f"SQL RETRIEVAL | Found {count} examples (max similarity: {max_similarity:.3f})")
    
    @staticmethod
    def retrieved_queries(similar_queries: list, max_display: int = 6):
        """Log the retrieved SQL queries"""
        logger.info(f"SQL RETRIEVAL | Retrieved queries (top {min(len(similar_queries), max_display)}):")
        for i, query_info in enumerate(similar_queries[:max_display], 1):
            if isinstance(query_info, dict):
                question = query_info.get('question', 'N/A')[:80]
                sql = query_info.get('sql', 'N/A')[:120]
                similarity = query_info.get('similarity', 0)
                logger.info(f"  {i}. ({similarity:.3f}) {question}{'...' if len(query_info.get('question', '')) > 80 else ''}")
                logger.info(f"     SQL: {sql}{'...' if len(query_info.get('sql', '')) > 120 else ''}")
            else:
                logger.info(f"  {i}. {str(query_info)[:120]}{'...' if len(str(query_info)) > 120 else ''}")
    
    @staticmethod
    def generation_start(question: str):
        logger.info(f"SQL GENERATION | Generating SQL for: {question[:50]}...")
    
    @staticmethod
    def generation_complete(sql: str, query_type: str):
        logger.info(f"SQL GENERATION | Generated {query_type} query ({len(sql)} chars)")
    
    @staticmethod
    def execution_start(sql: str):
        logger.info(f"SQL EXECUTION | Executing query...")
        logger.debug(f"SQL: {sql}")
    
    @staticmethod
    def execution_complete(rows: int, execution_time: float):
        logger.success(f"SQL EXECUTION | Retrieved {rows} rows in {execution_time:.3f}s")

class TokenLogger:
    @staticmethod
    def usage(agent_type: str, operation: str, tokens: int, cost: float):
        logger.info(f"TOKEN | {agent_type}:{operation} - {tokens} tokens (${cost:.4f})")
    
    @staticmethod
    def session_summary(total_tokens: int, total_cost: float, total_calls: int):
        logger.info(f"TOTAL USAGE | {total_tokens} tokens, ${total_cost:.4f}, {total_calls} calls")

class OrchestrationLogger:
    @staticmethod
    def multi_step_detected(tasks: list):
        logger.info(f"MULTI-STEP | Detected {len(tasks)} tasks:")
        for i, task in enumerate(tasks, 1):
            logger.info(f"   {i}. {task}")
    
    @staticmethod
    def step_routing(step: int, task: str, agent_type: str):
        logger.info(f"STEP {step} | Routing to {agent_type.upper()}: {task[:60]}...")
    
    @staticmethod
    def step_complete(step: int, agent_type: str):
        logger.success(f"STEP {step} | {agent_type.upper()} completed")
    
    @staticmethod
    def workflow_complete(steps: int, total_time: float):
        logger.success(f"WORKFLOW COMPLETE | {steps} steps in {total_time:.2f}s")