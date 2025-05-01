from fastapi import APIRouter, File, UploadFile,Form
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from ML_controller import ForecastController

router = APIRouter(
    prefix="/forecast",
    tags=["Profit Forecasting"],
    responses={404: {"description": "Not found"}},
)

class ForecastResponse(BaseModel):
    results: List[Dict[str, Any]]
    start_date: str
    end_date: str

@router.post("/", response_model=ForecastResponse)
async def create_forecast(
    future_periods: int = Form(12),
    file: UploadFile = File(...)
):
    return await ForecastController.generate_forecast(file=file, future_periods=future_periods)
