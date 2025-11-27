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
    def retrieved_queries(similar_queries: list, max_display: int = 20):
        """Log the retrieved SQL queries - simplified to question + similarity only"""
        logger.info(f"SQL RETRIEVAL | Retrieved queries (top {min(len(similar_queries), max_display)}):")
        for i, query_info in enumerate(similar_queries[:max_display], 1):
            if isinstance(query_info, dict):
                question = query_info.get('question', 'N/A')
                similarity = query_info.get('similarity', 0)
                # Only log question + similarity, no SQL text
                logger.info(f"  [{i}] sim={similarity:.3f} | {question}")
    
    @staticmethod
    def generation_start(question: str):
        logger.info(f"SQL GENERATION | Generating SQL for: {question[:50]}...")
    
    @staticmethod
    def generation_complete(sql: str, query_type: str):
        logger.info(f"SQL GENERATION | Generated {query_type} query ({len(sql)} chars)")
    
    @staticmethod
    def generated_sql(sql: str):
        """Log the final generated SQL query that will be executed"""
        logger.info(f"SQL GENERATION | Final SQL to execute:")
        logger.info(f"{sql}")
    
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


class TimingTracker:
    """Track and display execution time statistics for all agents"""
    
    def __init__(self):
        self.timings = {}
        self.total_start_time = None
        self.total_end_time = None
    
    def start_tracking(self):
        """Start tracking total query time"""
        self.total_start_time = __import__('time').time()
        self.timings = {}
    
    def record(self, agent_name: str, execution_time: float):
        """Record execution time for an agent"""
        if agent_name not in self.timings:
            self.timings[agent_name] = []
        self.timings[agent_name].append(execution_time)
    
    def end_tracking(self):
        """End tracking total query time"""
        self.total_end_time = __import__('time').time()
    
    def get_total_time(self) -> float:
        """Get total query execution time"""
        if self.total_start_time and self.total_end_time:
            return self.total_end_time - self.total_start_time
        return 0.0
    
    def display_stats(self):
        """Display timing statistics in a formatted table"""
        if not self.timings:
            logger.info("No timing data available")
            return
        
        total_query_time = self.get_total_time()
        
        # Calculate statistics
        stats = []
        total_agent_time = 0.0
        
        for agent_name, times in self.timings.items():
            total_time = sum(times)
            avg_time = total_time / len(times)
            max_time = max(times)
            min_time = min(times)
            count = len(times)
            
            percentage = (total_time / total_query_time * 100) if total_query_time > 0 else 0
            
            stats.append({
                'agent': agent_name,
                'total': total_time,
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
                'count': count,
                'percentage': percentage
            })
            
            total_agent_time += total_time
        
        # Sort by total time (descending)
        stats.sort(key=lambda x: x['total'], reverse=True)
        
        # Display header
        logger.info("\n" + "="*100)
        logger.info("⏱️  AGENT EXECUTION TIME STATISTICS")
        logger.info("="*100)
        
        # Display table header
        header = f"{'Agent':<25} {'Calls':<8} {'Total(s)':<12} {'Avg(s)':<12} {'Min(s)':<12} {'Max(s)':<12} {'% of Total':<12}"
        logger.info(header)
        logger.info("-"*100)
        
        # Display each agent's stats
        for stat in stats:
            row = (
                f"{stat['agent']:<25} "
                f"{stat['count']:<8} "
                f"{stat['total']:<12.3f} "
                f"{stat['avg']:<12.3f} "
                f"{stat['min']:<12.3f} "
                f"{stat['max']:<12.3f} "
                f"{stat['percentage']:<12.1f}"
            )
            logger.info(row)
        
        # Display footer
        logger.info("-"*100)
        
        # Calculate overhead (time not accounted for by agents)
        overhead = total_query_time - total_agent_time
        overhead_percentage = (overhead / total_query_time * 100) if total_query_time > 0 else 0
        
        logger.info(f"{'Total Agent Time':<25} {'':<8} {total_agent_time:<12.3f} {'':<12} {'':<12} {'':<12} {(total_agent_time/total_query_time*100):<12.1f}")
        logger.info(f"{'Overhead (non-agent)':<25} {'':<8} {overhead:<12.3f} {'':<12} {'':<12} {'':<12} {overhead_percentage:<12.1f}")
        logger.info(f"{'TOTAL QUERY TIME':<25} {'':<8} {total_query_time:<12.3f}")
        logger.info("="*100 + "\n")
        
        # Highlight slowest agent
        if stats:
            slowest = stats[0]
            logger.warning(f"🐌 SLOWEST AGENT: {slowest['agent']} ({slowest['total']:.3f}s total, {slowest['percentage']:.1f}% of query time)")


# Global timing tracker instance
_timing_tracker = TimingTracker()

def get_timing_tracker() -> TimingTracker:
    """Get the global timing tracker instance"""
    return _timing_tracker