from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.schemas.inference import HealthResponse, ImagePredictionResponse
from app.services.model import FindMyBallModel

settings = get_settings()
model_service = FindMyBallModel(settings.model_path, settings.model_name)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=settings.model_name,
        model_path=str(settings.model_path),
        model_available=model_service.model_available,
    )


@router.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.25, ge=0.01, le=1.0),
) -> ImagePredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be read as an image.",
        ) from exc

    detections = model_service.predict(image, confidence)
    return ImagePredictionResponse(
        filename=file.filename or "uploaded_image",
        model_name=settings.model_name,
        detection_count=len(detections),
        detections=detections,
    )
