import asyncio
from typing import AsyncIterable
import asyncio
from functools import partial
from langchain.schema import AIMessage
from fastapi import  HTTPException,APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from BACKEND.chatbot.orchestrator.graph import call_ai_agentic
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chatbot",
    tags=["AI chatbot"],
)

router = APIRouter(
    prefix="/chatbot",
    tags=["AI chatbot"],
    responses={404: {"description": "Not found"}},
)
class Query(BaseModel):
    user_input: str
    user_id: str = None
    delay: float = 0.005 



async def ai_agent(user_input: str, user_id: str = None, delay: float = 0.01):
    logger.info("Streaming started")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        partial(call_ai_agentic, user_input=user_input, user_id=user_id)
    )
    try:
        if not response:
            raise HTTPException(status_code=404, detail="No response")

        # Handle AIMessage or string content
        if isinstance(response, AIMessage):
            content = response.content
        elif isinstance(response, dict) and "content" in response:
            content = response["content"]
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)

        # Stream character by character
        for i in range(0, len(content), 3):
            chunk = content[i:i+3]
            yield chunk
            await asyncio.sleep(delay * 3)

    except Exception as e:
        logger.error(f"Error in ai_agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# FastAPI POST endpoint to receive a query and stream response
@router.post("/", response_class=StreamingResponse)
async def stream_chat(message: Query):
    logger.info(f"Received request: {message}")
    generator = ai_agent(message.user_input, message.user_id, message.delay)
    return StreamingResponse(generator, media_type="text/event-stream")


@router.post("/test")
async def test_call(message: Query):
    response = call_ai_agentic(message.user_input, message.user_id)
    return {"response": response}