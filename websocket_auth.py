import os
from dotenv import load_dotenv

load_dotenv()


def validate_session_token(session_id: str) -> bool:
    """
    Validates the session token.
    For production, implement proper token validation logic.
    This is a placeholder implementation.
    """
    if not session_id or len(session_id) < 8:
        return False
    
    return True


def get_user_id_from_session(session_id: str) -> str:
    """
    Extracts user_id from session_id.
    For production, implement proper user identification logic.
    This is a placeholder implementation.
    """
    return f"user_{session_id[:8]}"
