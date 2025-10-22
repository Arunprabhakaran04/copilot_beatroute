from enum import Enum


class MessageType(Enum):
    STATUS = "status"
    TEXT = "text"
    TABLE = "table"
    GRAPH = "graph"
    SUMMARY = "summary"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    SUGGESTED_QUESTIONS = "suggested_questions"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_STEP = "workflow_step"
    WORKFLOW_COMPLETE = "workflow_complete"
