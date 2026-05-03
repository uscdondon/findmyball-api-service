from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the FindMyBall inference API."""

    app_name: str = "FindMyBall API Service"
    api_prefix: str = "/api/v1"
    model_name: str = "findmyball_yolov8n"
    model_path: Path = Path("models/findmyball_yolov8n.pt")

    model_config = SettingsConfigDict(env_prefix="FINDMYBALL_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
