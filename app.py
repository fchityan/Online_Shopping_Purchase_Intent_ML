from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import build_prediction_response

app = FastAPI(title='Purchase Intent Prediction API', version='0.1.0')


class PredictionRequest(BaseModel):
    CustomerType: str
    SpecialDayProximity: float
    ExitRate: float
    PageValue: float
    TrafficSource: float
    GeographicRegion: int
    BounceRate: float
    ProductPageTime: float


class PredictionPayload(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    prediction: int


@app.get('/health')
def health_check() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict', response_model=List[PredictionPayload])
def predict(requests: List[PredictionRequest]):
    if not requests:
        raise HTTPException(status_code=400, detail='At least one request is required.')

    records = [request.model_dump() for request in requests]
    return build_prediction_response(records, model_path='model.joblib')
