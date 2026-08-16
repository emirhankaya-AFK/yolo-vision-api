import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import Settings
from app.schemas import (
    ConfidenceMetrics,
    ImageDetectionResponse,
    VideoDetectionResponse,
    VideoFrameResult,
)
from app.services.detector import Detector


class MediaValidationError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    from io import BytesIO

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise MediaValidationError("The uploaded file is not a valid image") from exc


async def analyze_image(data: bytes, detector: Detector) -> ImageDetectionResponse:
    prediction = await detector.predict(decode_image(data))
    return ImageDetectionResponse(
        model=detector.model_name,
        image_width=prediction.width,
        image_height=prediction.height,
        detection_count=len(prediction.detections),
        detections=prediction.detections,
        speed=prediction.speed,
        confidence=prediction.confidence_metrics,
    )


async def analyze_video(
    data: bytes,
    filename: str,
    detector: Detector,
    settings: Settings,
) -> VideoDetectionResponse:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video processing") from exc

    suffix = Path(filename).suffix.lower() or ".mp4"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise MediaValidationError("The uploaded file is not a readable video")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = source_frame_count / source_fps if source_fps > 0 else 0
        frames: list[VideoFrameResult] = []
        confidence_values: list[float] = []
        inference_times: list[float] = []
        frame_number = 0
        started = time.perf_counter()

        while len(frames) < settings.max_video_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % settings.video_frame_stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prediction = await detector.predict(rgb_frame)
                frames.append(
                    VideoFrameResult(
                        frame_number=frame_number,
                        timestamp_seconds=round(frame_number / source_fps, 3)
                        if source_fps > 0
                        else 0,
                        detections=prediction.detections,
                    )
                )
                confidence_values.extend(item.confidence for item in prediction.detections)
                inference_times.append(prediction.speed.inference_ms)
            frame_number += 1

        capture.release()
        elapsed = time.perf_counter() - started
        processed = len(frames)
        confidence = (
            ConfidenceMetrics(
                mean=round(sum(confidence_values) / len(confidence_values), 4),
                minimum=round(min(confidence_values), 4),
                maximum=round(max(confidence_values), 4),
            )
            if confidence_values
            else ConfidenceMetrics()
        )
        return VideoDetectionResponse(
            model=detector.model_name,
            source_fps=round(source_fps, 3),
            source_frame_count=source_frame_count,
            processed_frame_count=processed,
            duration_seconds=round(duration, 3),
            elapsed_seconds=round(elapsed, 3),
            effective_fps=round(processed / elapsed, 3) if elapsed > 0 else 0,
            average_inference_ms=round(sum(inference_times) / processed, 3) if processed else 0,
            total_detections=sum(len(frame.detections) for frame in frames),
            confidence=confidence,
            frames=frames,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
