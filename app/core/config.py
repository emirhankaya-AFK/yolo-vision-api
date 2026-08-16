from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "YOLO Vision API"
    app_env: str = "development"
    model_path: str = "yolo26n.pt"
    confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.7, ge=0, le=1)
    image_size: int = Field(default=640, ge=160, le=2048)
    device: str = "cpu"
    max_upload_mb: int = 50
    video_frame_stride: int = Field(default=10, ge=1, le=300)
    max_video_frames: int = Field(default=120, ge=1, le=1_000)


@lru_cache
def get_settings() -> Settings:
    return Settings()
