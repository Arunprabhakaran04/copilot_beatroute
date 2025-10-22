from typing import Dict, Any
from fastapi import WebSocket


class WSManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        self.active_connections[session_id] = ws

    async def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        ws = self.active_connections.get(session_id)
        if ws:
            await ws.send_json(message)

    def is_session_active(self, session_id: str) -> bool:
        return session_id in self.active_connections


ws_manager = WSManager()
