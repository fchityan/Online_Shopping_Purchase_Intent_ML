from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import RedirectResponse

from src.inference import build_prediction_response, load_model_bundle

MODEL_PATH = Path(os.getenv("MODEL_PATH", "outputs/model_bundle.joblib"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))


class PredictionRequest(BaseModel):
    CustomerType: str = Field(min_length=1)
    SpecialDayProximity: float = Field(ge=0.0, le=1.0)
    ExitRate: float = Field(ge=0.0, le=1.0)
    PageValue: float = Field(ge=0.0)
    TrafficSource: float
    GeographicRegion: int
    BounceRate: float = Field(ge=0.0, le=1.0)
    ProductPageTime: float = Field(ge=0.0)


class PredictionPayload(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    prediction: int
    threshold: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_version: str


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.bundle = None
    application.state.load_error = None
    try:
        application.state.bundle = load_model_bundle(MODEL_PATH)
    except Exception as exc:
        application.state.load_error = str(exc)
    yield


app = FastAPI(title="Purchase Intent Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    """Backward-compatible liveness endpoint."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error or "Model not loaded")
    return {"status": "ready", "model_version": bundle["model_version"]}


@app.get("/metadata")
def metadata(request: Request) -> dict[str, Any]:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error or "Model not loaded")
    return {
        "model_name": bundle.get("model_display_name", bundle["model_name"]),
        "model_version": bundle["model_version"],
        "threshold": bundle["threshold"],
        "feature_columns": bundle["feature_columns"],
        "created_at_utc": bundle.get("created_at_utc"),
    }


@app.post("/predict", response_model=list[PredictionPayload])
def predict(requests: list[PredictionRequest], request: Request):
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error or "Model not loaded")
    if not requests:
        raise HTTPException(status_code=400, detail="At least one request is required.")
    if len(requests) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Batch exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}.")

    try:
        return build_prediction_response(
            [item.model_dump() for item in requests],
            bundle=bundle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
