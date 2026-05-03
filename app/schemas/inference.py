from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float


class ImagePredictionResponse(BaseModel):
    filename: str
    model_name: str
    detection_count: int = Field(..., ge=0)
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_path: str
    model_available: bool
