from fastapi import APIRouter, File, UploadFile, Form, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from API.ML_controller import SegmentationController

router = APIRouter(
    prefix="/segment",
    tags=["Customer Segmentation"],
    responses={404: {"description": "Not found"}},
)

class SegmentationResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/", response_model=SegmentationResponse)
async def create_segmentation(
    file: UploadFile = File(...)
):
    return await SegmentationController.segment_customers(file)