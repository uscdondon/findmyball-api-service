from fastapi import FastAPI

from app.api.endpoints import router as api_router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AIM230 FastAPI deployment prototype for FindMyBall YOLOv8 inference.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "FindMyBall API Service",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }


app.include_router(api_router, prefix=settings.api_prefix)
