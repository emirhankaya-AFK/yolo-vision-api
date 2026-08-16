from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import Settings, get_settings
from app.schemas import HealthResponse, ImageDetectionResponse, VideoDetectionResponse
from app.services.detector import Detector, get_detector
from app.services.media import MediaValidationError, analyze_image, analyze_video

router = APIRouter()
DetectorDep = Annotated[Detector, Depends(get_detector)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
UploadDep = Annotated[UploadFile, File()]


async def read_limited_upload(file: UploadFile, max_upload_mb: int) -> bytes:
    limit = max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    await file.close()
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds the {max_upload_mb} MB limit")
    if not data:
        raise HTTPException(status_code=422, detail="The uploaded file is empty")
    return data


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(status="ok", model=settings.model_path, environment=settings.app_env)


@router.post(
    "/api/v1/detect/image",
    response_model=ImageDetectionResponse,
    tags=["detection"],
)
async def detect_image(
    file: UploadDep,
    detector: DetectorDep,
    settings: SettingsDep,
) -> ImageDetectionResponse:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="An image file is required")
    try:
        return await analyze_image(
            await read_limited_upload(file, settings.max_upload_mb), detector
        )
    except MediaValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/api/v1/detect/video",
    response_model=VideoDetectionResponse,
    tags=["detection"],
)
async def detect_video(
    file: UploadDep,
    detector: DetectorDep,
    settings: SettingsDep,
) -> VideoDetectionResponse:
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(status_code=415, detail="A video file is required")
    try:
        data = await read_limited_upload(file, settings.max_upload_mb)
        return await analyze_video(data, file.filename or "video.mp4", detector, settings)
    except MediaValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
