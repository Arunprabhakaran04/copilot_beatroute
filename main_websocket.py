from fastapi import FastAPI, WebSocket, status, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
import json
import asyncio

from websocket_manager import ws_manager
from message_type import MessageType
from websocket_auth import validate_session_token, get_user_id_from_session
from constants import SUGGESTED_QUESTIONS_LIST, ERROR_RESPONSE
from redis_memory_manager import initialize_session

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


@app.on_event("startup")
async def startup_event():
    global orchestrator
    from main import CentralOrchestrator
    try:
        orchestrator = CentralOrchestrator(files_directory="./user_files", schema_file_path="schema")
        logger.info("Central Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise


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
                    asyncio.create_task(process_question(websocket, question, session_id, user_id))
                    
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


async def process_question(websocket: WebSocket, question: str, session_id: str, user_id: str):
    """Process user question through the orchestrator and stream results"""
    try:
        result = await asyncio.to_thread(
            orchestrator.process_query,
            query=question,
            session_id=session_id,
            user_id=user_id
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
            agent_type = result.get("agent_type", "unknown")
            result_data = result.get("result", {})
            
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
    if agent_type == "db_query":
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
