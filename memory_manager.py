import time
import re
import logging
from collections import deque
from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class ClassificationValidator:
    """Validates classification accuracy and provides feedback loop for improvement"""
    def __init__(self):
        self.classification_history = deque(maxlen=100)
        self.error_patterns = {}
    
    def validate_classification(self, query: str, predicted_agent: str, actual_result: str) -> bool:
        """Validate if classification was correct based on execution result"""
        is_correct = actual_result == "completed"
        
        self.classification_history.append({
            "query": query,
            "predicted": predicted_agent,
            "correct": is_correct,
            "timestamp": time.time()
        })
        
        if not is_correct:
            self._record_error_pattern(query, predicted_agent)
        
        return is_correct
    
    def _record_error_pattern(self, query: str, wrong_agent: str):
        """Record patterns that lead to misclassification"""
        key_phrases = re.findall(r'\b\w+\b', query.lower())
        for phrase in key_phrases:
            if phrase not in self.error_patterns:
                self.error_patterns[phrase] = {}
            if wrong_agent not in self.error_patterns[phrase]:
                self.error_patterns[phrase][wrong_agent] = 0
            self.error_patterns[phrase][wrong_agent] += 1
    
    def get_classification_accuracy(self) -> float:
        """Calculate recent classification accuracy"""
        if not self.classification_history:
            return 0.0
        
        correct = sum(1 for entry in self.classification_history if entry["correct"])
        return correct / len(self.classification_history)
    
    def get_problematic_patterns(self) -> List[str]:
        """Get patterns that frequently lead to misclassification"""
        problematic = []
        for phrase, agents in self.error_patterns.items():
            total_errors = sum(agents.values())
            if total_errors >= 3:
                problematic.append(phrase)
        return problematic


class MemoryManager:
    """In memory conversattion manager - uses llm to enrich the query with previous memory

    Stored entry shape: {"original": str, "enriched": str, "agent_type": str, "result": dict, "timestamp": float}
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
            "result": result,
            "timestamp": time.time()
        }
        self.history.append(entry)
        logger.info(f"Added memory entry: {agent_type} - {original_query[:50]}...")

    def get_recent(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get recent entries, prioritizing the most recent ones"""
        recent_entries = list(self.history)[-n:]
        return sorted(recent_entries, key=lambda x: x.get('timestamp', 0), reverse=True)
    
    def clear_memory(self):
        """Clear all memory entries - useful for session cleanup"""
        self.history.clear()
        logger.info("Memory cleared")
    
    def get_memory_summary(self) -> str:
        """Get a summary of current memory state"""
        if not self.history:
            return "No memory entries"
        
        summary = f"Memory: {len(self.history)} entries\n"
        for i, entry in enumerate(self.history):
            summary += f"  {i+1}. {entry['agent_type']}: {entry['original'][:30]}...\n"
        return summary

    def _heuristic_expand(self, query: str) -> Optional[str]:
        lowered = query.lower()
        if any(token in lowered for token in ["that date", "that day", "that time", "that slot"]):
            for entry in reversed(self.history):
                if entry.get("agent_type") == "meeting":
                    meeting_date = entry.get("result", {}).get("meeting_date") or entry.get("result", {}).get("parsed_date")
                    if meeting_date:
                        return re.sub(r"(?i)that date|that day|that time|that slot", meeting_date, query)
        
        ambiguous_patterns = [
            "also invite", "also send", "also include", "also add",
            "confirm the", "update the", "change the", "modify the",
            "for the meeting", "for that meeting", "to the meeting",
            "the same", "same meeting", "same time", "same date"
        ]
        
        if any(pattern in lowered for pattern in ambiguous_patterns):
            return None
        
        return None

    def enrich_query(self, original_query: str) -> str:
        """Return an enriched query. Try simple heuristics first; if not
        sufficient, build a short context from recent history and ask the LLM
        (OpenAI) to rewrite the query filling ambiguous references.
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
        
        # Prioritize the most recent context (first item in sorted list)
        context_lines = []
        for i, e in enumerate(recent):
            small_result = {k: v for k, v in (e.get("result") or {}).items() if isinstance(v, (str, int, float))}
            priority_marker = "[MOST RECENT]" if i == 0 else f"[{len(recent)-i} ago]"
            context_lines.append(f"{priority_marker} User: {e['original']} --> Agent: {e.get('agent_type')} --> Result: {small_result}")

        context_text = "\n".join(context_lines)

        enrich_prompt = ChatPromptTemplate.from_template("""
        You are a context-aware query rewriter for a multi-agent system. Your job is to combine RECENT CONTEXT with the CURRENT USER QUERY to create a complete, unambiguous instruction.

        RECENT INTERACTIONS (ordered by recency, MOST RECENT first):
        {recent_context}

        CURRENT USER QUERY (the new instruction): {query}

        CRITICAL RULES FOR COMBINING CONTEXT + CURRENT QUERY:
        1. **PRIORITIZE THE [MOST RECENT] ENTRY** - This is the most relevant context
        2. The CURRENT QUERY contains NEW INSTRUCTIONS that must be followed
        3. Use context ONLY to fill in missing details (subjects, dates, recipients, meeting info)
        4. When current query says "also send details about the meeting" - get meeting details from MOST RECENT meeting entry
        5. When current query says "cc someone@email.com" - ADD this as CC to existing context
        6. PRESERVE all new information from current query (new emails, new dates, new instructions)

        SPECIFIC EXAMPLES:
        Context: "[MOST RECENT] User: schedule meeting with user 3 on 08/10/2025 → Agent: meeting → Result: {{'user_id': '3', 'meeting_date': '08/10/2025'}}"
        Current: "also send the details about the meeting to arun@gmail.com"
        Correct Output: "Send email to arun@gmail.com about the meeting with user 3 scheduled on 08/10/2025"

        Context: "[MOST RECENT] User: send email to john@x.com subject: Meeting Reminder → Agent: email → Result: {{'email_to': 'john@x.com', 'subject': 'Meeting Reminder'}}"
        Current: "cc mary@y.com and use same subject"
        Correct Output: "Send email to john@x.com and cc mary@y.com with subject: Meeting Reminder"

        IMPORTANT: 
        - Focus on the [MOST RECENT] context entry first
        - Only use older context if [MOST RECENT] doesn't have the needed information
        - If current query mentions "the meeting", "that meeting" - look for meeting details in [MOST RECENT] meeting entry
        - Always preserve the intent and new information from the current query

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

        return original_query


class EnhancedMemoryManager(MemoryManager):
    """Enhanced memory manager with robust context enrichment and failure handling"""
    def __init__(self, llm, max_history: int = 3):
        super().__init__(llm, max_history)
        self.enrichment_cache = {}  
        self.failed_patterns = set() 
        self.enrichment_timeout = 10  
    
    def enrich_query(self, original_query: str) -> str:
        """Enhanced query enrichment with failure handling"""
        cache_key = self._get_cache_key(original_query)
        if cache_key in self.enrichment_cache:
            logger.info(f"Using cached enrichment for: {original_query}")
            return self.enrichment_cache[cache_key]
        
        if self._is_problematic_pattern(original_query):
            logger.warning(f"Skipping enrichment for problematic pattern: {original_query}")
            return original_query
        
        try:
            enriched = self._safe_enrich_query(original_query)
            
            if self._validate_enriched_query(original_query, enriched):
                self.enrichment_cache[cache_key] = enriched
                return enriched
            else:
                logger.warning(f"Enriched query validation failed, using original")
                return original_query
                
        except Exception as e:
            logger.error(f"Query enrichment failed: {e}")
            self.failed_patterns.add(self._extract_pattern(original_query))
            return original_query
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return f"{query.lower().strip()}_{len(self.history)}"
    
    def _is_problematic_pattern(self, query: str) -> bool:
        """Check if query contains known problematic patterns"""
        pattern = self._extract_pattern(query)
        return pattern in self.failed_patterns
    
    def _extract_pattern(self, query: str) -> str:
        """Extract key pattern from query for failure tracking"""
        words = query.lower().split()[:3]
        return " ".join(words)
    
    def _safe_enrich_query(self, original_query: str) -> str:
        """Enrichment with timeout and retry logic"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                enriched = super().enrich_query(original_query)
                if enriched and len(enriched.strip()) > 0:
                    return enriched
            except Exception as e:
                logger.warning(f"Enrichment attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        
        return original_query
    
    def _validate_enriched_query(self, original: str, enriched: str) -> bool:
        """Validate that enriched query makes sense and doesn't mix incorrect context"""
        if not enriched or len(enriched.strip()) == 0:
            logger.warning("Enriched query is empty")
            return False
        
        if len(enriched) > len(original) * 4:  # Allow slightly more expansion
            logger.warning(f"Enriched query too long: {len(enriched)} vs {len(original)}")
            return False
        
        # Preserve original emails
        original_emails = re.findall(r'\b\w+@[\w.-]+\.\w+\b', original)
        enriched_emails = re.findall(r'\b\w+@[\w.-]+\.\w+\b', enriched)
        
        for email in original_emails:
            if email not in enriched:
                logger.warning(f"Original email {email} lost in enrichment")
                return False
        
        enriched_lower = enriched.lower()
        suspicious_patterns = [
            # Multiple different email subjects mixed together
            (r'subject:.*subject:', "Multiple subjects detected"),
            # Multiple different recipients mixed
            (r'send.*send.*send', "Multiple send commands detected"), 
            # Mixed dates that don't make sense
            (r'\d{4}-\d{2}-\d{2}.*\d{4}-\d{2}-\d{2}', "Multiple dates detected"),
        ]
        
        for pattern, reason in suspicious_patterns:
            if re.search(pattern, enriched_lower):
                logger.warning(f"Suspicious enrichment pattern: {reason}")
                return False
        
        return True