from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from ML_controller import SentimentController

router = APIRouter(
    prefix="/sentiment",
    tags=["Sentiment Analysis"],
    responses={404: {"description": "Not found"}},
)

class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")

class SentimentResponse(BaseModel):
    sentiment: str
    rating: float

class SentimentBatchResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    return await SentimentController.analyze_text(request.text)

@router.post("/batch/", response_model=SentimentBatchResponse)
async def analyze_sentiment_batch(
    file: UploadFile = File(...)
):
    return await SentimentController.analyze_batch(file)