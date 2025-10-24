from fastapi import FastAPI, WebSocket, status, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
import json
import asyncio
from typing import Dict

from websocket_manager import ws_manager
from message_type import MessageType
from websocket_auth import validate_session_token, get_user_id_from_session
from constants import SUGGESTED_QUESTIONS_LIST, ERROR_RESPONSE
from redis_memory_manager import initialize_session
from user_context import UserContext

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = None

# Store UserContext instances per user_id (in production, use Redis)
user_contexts: Dict[str, UserContext] = {}

# Hardcoded base64 token for testing (contains auth token + user_id)
# This is the token: "GNsy9yt81VBARJPqMuAxhbQDJOTqhQol-6"
HARDCODED_BASE64_TOKEN = "R05zeTl5dDgxVkJBUkpfcU11QXhoYlFESk9UcWhRb2wtNg"


@app.on_event("startup")
async def startup_event():
    global orchestrator
    from main import CentralOrchestrator
    try:
        orchestrator = CentralOrchestrator(files_directory="./user_files", schema_file_path="schema")
        logger.info("✅ Central Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        raise


async def ensure_user_context(user_id: str) -> UserContext:
    """
    Ensure UserContext is loaded for the user.
    Loads schema ONCE per user and caches it.
    
    Args:
        user_id: User identifier
        
    Returns:
        UserContext instance
    """
    if user_id in user_contexts:
        logger.info(f"♻️ Using cached UserContext for user {user_id}")
        return user_contexts[user_id]
    
    logger.info(f"🆕 Creating new UserContext for user {user_id}")
    
    # Create UserContext
    context = UserContext(
        user_id=user_id,
        user_name=f"User_{user_id}",
        email=f"user{user_id}@example.com"
    )
    
    # Load schema using hardcoded token (ONE-TIME LOAD)
    logger.info(f"📊 Loading schema for user {user_id}... This takes 5-10 seconds")
    success = await context.load_schema_from_token(
        base64_token=HARDCODED_BASE64_TOKEN,
        cubejs_api_url="analytics.vwbeatroute.com/api/v1/meta",
        generate_embeddings=True
    )
    
    if not success:
        raise Exception("Failed to load schema")
    
    # Cache the context
    user_contexts[user_id] = context
    
    logger.info(f"✅ UserContext cached for user {user_id}")
    logger.info(f"   📋 Tables: {len(context.schema_map)}")
    logger.info(f"   🔮 Embeddings: {context.embeddings_schema.shape}")
    
    return context


async def ping_scheduler(ws: WebSocket, session_id: str, interval: int = 20):
    """Background task for sending periodic pings to keep connection alive"""
    while True:
        try:
            await ws.send_json({'type': MessageType.PING.value, 'content': 'PING'})
            await asyncio.sleep(interval)
        except Exception as e:
            logger.warning(f"Ping failed for session {session_id}: {e}")
            break


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    if not validate_session_token(session_id):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning(f"Rejected WebSocket connection for invalid session: {session_id}")
        return

    await websocket.accept()
    await ws_manager.connect(session_id, websocket)
    logger.info(f"WebSocket connected for session: {session_id}")
    
    user_id = get_user_id_from_session(session_id)
    initialize_session(user_id, session_id)
    logger.info(f"Session initialized for user: {user_id}, session: {session_id}")
    
    # Load UserContext with schema (happens ONCE per user)
    try:
        user_context = await ensure_user_context(user_id)
        logger.info(f"✅ UserContext ready for user: {user_id}")
    except Exception as e:
        logger.error(f"❌ Failed to load UserContext for user {user_id}: {e}")
        await websocket.send_json({
            "type": MessageType.ERROR.value,
            "content": "Failed to initialize user context. Please try again."
        })
        await websocket.close()
        return
    
    ping_task = asyncio.create_task(ping_scheduler(websocket, session_id))
    
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received from {session_id}: {data}")
            await ws_manager.send_message(session_id, {"type": MessageType.STATUS.value, "content": "Received"})
            
            try:
                json_data = json.loads(data)
                question = json_data.get("question", None)
                get_suggested = json_data.get("get_suggested_questions", False)
                
                pong_response = json_data.get("type", None)
                if pong_response is not None and pong_response == MessageType.PONG.value:
                    continue
                
                if question:
                    # Pass UserContext to process_question
                    asyncio.create_task(
                        process_question(websocket, question, session_id, user_id, user_context)
                    )
                    
                elif get_suggested:
                    await ws_manager.send_message(session_id, {
                        "type": MessageType.SUGGESTED_QUESTIONS.value,
                        "content": SUGGESTED_QUESTIONS_LIST
                    })
                    
                else:
                    await ws_manager.send_message(session_id, {
                        "type": MessageType.ERROR.value,
                        "content": f"Invalid Data: {data}"
                    })
                    await ws_manager.send_message(session_id, {
                        "type": MessageType.STATUS.value,
                        "content": "END"
                    })
                    
            except json.JSONDecodeError:
                await ws_manager.send_message(session_id, {
                    "type": MessageType.ERROR.value,
                    "content": f"Invalid JSON: {data}"
                })
                await ws_manager.send_message(session_id, {
                    "type": MessageType.STATUS.value,
                    "content": "END"
                })
                
    except WebSocketDisconnect as e:
        if e.code == 1001:
            logger.warning(f"Client closed WebSocket normally (1001) - session: {session_id}")
        elif e.code == 1006:
            logger.error(f"Abnormal WebSocket closure (1006) - session: {session_id}")
        else:
            logger.exception(f"WebSocket disconnect (code={e.code}) - session: {session_id}")
            
    except Exception as e:
        logger.exception(f"WebSocket error for session {session_id}: {e}")
        
    finally:
        ping_task.cancel()
        await ws_manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected for session: {session_id}")


async def process_question(websocket: WebSocket, question: str, session_id: str, user_id: str, user_context: UserContext):
    """Process user question through the orchestrator and stream results"""
    try:
        # Log that we're using focused schema approach
        logger.info(f"Processing question with UserContext (focused schema enabled)")
        logger.info(f"   Available tables: {len(user_context.schema_map)}")
        
        result = await asyncio.to_thread(
            orchestrator.process_query,
            query=question,
            session_id=session_id,
            user_id=user_id,
            user_context=user_context  # Pass UserContext to orchestrator
        )
        
        if not result.get("success", False):
            await websocket.send_json({
                "type": MessageType.ERROR.value,
                "content": result.get("error", ERROR_RESPONSE)
            })
            await websocket.send_json({
                "type": MessageType.STATUS.value,
                "content": "END"
            })
            return
        
        # Check if this is a direct answer or follow-up from EnrichAgent
        agent_type = result.get("agent_type", "unknown")
        result_data = result.get("result", {})
        
        if agent_type == "enrich_agent":
            message_type = result_data.get("type", "")
            message_content = result_data.get("message", "")
            
            if message_type == "direct_answer":
                # Send direct answer without START/END wrapper
                logger.info("EnrichAgent provided direct answer")
                await websocket.send_json({
                    "type": MessageType.STATUS.value,
                    "content": "START"
                })
                await websocket.send_json({
                    "type": MessageType.TEXT.value,
                    "content": message_content
                })
                await websocket.send_json({
                    "type": MessageType.STATUS.value,
                    "content": "END"
                })
                return
            
            elif message_type == "follow_up":
                # Send follow-up question for user clarification
                logger.info("EnrichAgent needs clarification")
                await websocket.send_json({
                    "type": MessageType.STATUS.value,
                    "content": "START"
                })
                await websocket.send_json({
                    "type": MessageType.TEXT.value,
                    "content": message_content
                })
                await websocket.send_json({
                    "type": MessageType.STATUS.value,
                    "content": "END"
                })
                return
        
        # Normal processing for other agents
        await websocket.send_json({
            "type": MessageType.STATUS.value,
            "content": "START"
        })
        
        if result.get("is_multi_step", False):
            for idx, step in enumerate(result.get("completed_steps", []), 1):
                step_agent_type = step.get("agent_type")
                step_result = step.get("result", {})
                
                await send_agent_result(websocket, step_agent_type, step_result)
        else:
            await send_agent_result(websocket, agent_type, result_data)
        
        await websocket.send_json({
            "type": MessageType.STATUS.value,
            "content": "END"
        })
        
    except Exception as e:
        logger.exception(f"Error processing question for session {session_id}: {e}")
        await websocket.send_json({
            "type": MessageType.ERROR.value,
            "content": ERROR_RESPONSE
        })
        await websocket.send_json({
            "type": MessageType.STATUS.value,
            "content": "END"
        })


async def send_agent_result(websocket: WebSocket, agent_type: str, result_data: dict):
    """Send agent-specific results to the client"""
    if agent_type == "entity_verification":
        if "type" in result_data and result_data["type"] == "entity_verification_error":
            await websocket.send_json({
                "type": MessageType.TEXT.value,
                "content": result_data.get("message", "Entity verification required")
            })
    
    elif agent_type == "db_query":
        if "query_results" in result_data:
            query_results = result_data["query_results"]
            
            if "data" in query_results and query_results["data"]:
                table_content = query_results["data"]
                if isinstance(table_content, str):
                    import json
                    try:
                        table_content = json.loads(table_content)
                    except:
                        pass
                
                await websocket.send_json({
                    "type": MessageType.TABLE.value,
                    "content": table_content
                })
            
            if "summary" in query_results and query_results["summary"]:
                await websocket.send_json({
                    "type": MessageType.SUMMARY.value,
                    "content": query_results["summary"]
                })
    
    elif agent_type == "summary":
        if "summary" in result_data and result_data["summary"]:
            await websocket.send_json({
                "type": MessageType.SUMMARY.value,
                "content": result_data["summary"]
            })
    
    elif agent_type == "visualization":
        if "visualization" in result_data and result_data["visualization"]:
            await websocket.send_json({
                "type": MessageType.GRAPH.value,
                "content": result_data["visualization"]
            })
    
    elif agent_type in ["email", "meeting", "campaign"]:
        message_content = result_data.get("message", str(result_data))
        await websocket.send_json({
            "type": MessageType.TEXT.value,
            "content": message_content
        })



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
