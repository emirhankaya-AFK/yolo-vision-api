from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox


class SpeedMetrics(BaseModel):
    preprocess_ms: float = 0
    inference_ms: float = 0
    postprocess_ms: float = 0
    total_ms: float = 0


class ConfidenceMetrics(BaseModel):
    mean: float = 0
    minimum: float = 0
    maximum: float = 0


class ImageDetectionResponse(BaseModel):
    model: str
    image_width: int
    image_height: int
    detection_count: int
    detections: list[Detection]
    speed: SpeedMetrics
    confidence: ConfidenceMetrics


class VideoFrameResult(BaseModel):
    frame_number: int
    timestamp_seconds: float
    detections: list[Detection]


class VideoDetectionResponse(BaseModel):
    model: str
    source_fps: float
    source_frame_count: int
    processed_frame_count: int
    duration_seconds: float
    elapsed_seconds: float
    effective_fps: float
    average_inference_ms: float
    total_detections: int
    confidence: ConfidenceMetrics
    frames: list[VideoFrameResult]


class HealthResponse(BaseModel):
    status: str
    model: str
    environment: str


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message")
