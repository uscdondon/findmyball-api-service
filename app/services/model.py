from pathlib import Path

from fastapi import HTTPException, status
from PIL import Image
from ultralytics import YOLO

from app.schemas.inference import Detection


class FindMyBallModel:
    """Thin wrapper around the trained YOLOv8 golf-ball detector."""

    def __init__(self, model_path: Path, model_name: str) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self._model: YOLO | None = None

    @property
    def model_available(self) -> bool:
        return self.model_path.exists()

    def _ensure_model_file(self) -> None:
        if not self.model_available:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model file is missing. Expected model at: {self.model_path}",
            )

    def load(self) -> YOLO:
        self._ensure_model_file()
        if self._model is None:
            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, image: Image.Image, confidence: float) -> list[Detection]:
        model = self.load()
        results = model.predict(source=image, conf=confidence, verbose=False)

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            label = str(names.get(class_id, class_id))
            score = float(box.conf[0].item())

            detections.append(
                Detection(
                    label=label,
                    confidence=score,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    center_x=float((x1 + x2) / 2),
                    center_y=float((y1 + y2) / 2),
                )
            )

        return detections
