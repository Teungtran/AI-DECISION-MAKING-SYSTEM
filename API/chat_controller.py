import asyncio
from typing import AsyncIterable
from langchain.schema import AIMessage
from fastapi import  HTTPException,APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.orchestrator.graph import call_ai_agentic
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chatbot",
    tags=["AI chatbot"],
    responses={404: {"description": "Not found"}},
)
class Query(BaseModel):
    user_input: str
    user_id: str = None
    delay: float = 0.005 

# Function to call the AI agent
async def ai_agent(user_input: str, user_id: str = None, delay: float = 0.01) -> AsyncIterable[str]:
    logger.info(f"Processing query: {user_input}, user_id: {user_id}, char delay: {delay}")
    try:
        response = call_ai_agentic(user_input=user_input, user_id=user_id)

        # Handle AIMessage
        if isinstance(response, AIMessage):
            content = response.content

        # If response is a dict
        elif isinstance(response, dict):
            content = response.get("content", "No content found")

        # If response is a string
        elif isinstance(response, str):
            content = response

        # If it's async iterable
        elif hasattr(response, "__aiter__"):
            async for chunk in response:
                # Handle chunk as AIMessage or string
                if isinstance(chunk, AIMessage):
                    text = chunk.content
                elif isinstance(chunk, str):
                    text = chunk
                else:
                    raise ValueError(f"Unexpected stream chunk type: {type(chunk)}")

                for char in text:
                    yield char
                    if char in [".", "!", "?"]:
                        await asyncio.sleep(delay * 6)
                    elif char in [",", ";", ":"]:
                        await asyncio.sleep(delay * 3)
                    else:
                        await asyncio.sleep(delay)
            return
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

        # If it's not async iterable, stream in 3-char chunks
        for i in range(0, len(content), 3):
            yield content[i:i+3]
            await asyncio.sleep(delay * 3)

    except Exception as e:
        logger.error(f"Error in ai_agent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# FastAPI POST endpoint to receive a query and stream response
@router.post("/", response_class=StreamingResponse)
async def stream_chat(message: Query):
    logger.info(f"Received request: {message}")
    generator = ai_agent(message.user_input, message.user_id, message.delay)
    return StreamingResponse(generator, media_type="text/event-stream")

