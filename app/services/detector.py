import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.core.config import Settings, get_settings
from app.schemas import BoundingBox, ConfidenceMetrics, Detection, SpeedMetrics


@dataclass(slots=True)
class Prediction:
    width: int
    height: int
    detections: list[Detection]
    speed: SpeedMetrics

    @property
    def confidence_metrics(self) -> ConfidenceMetrics:
        values = [item.confidence for item in self.detections]
        if not values:
            return ConfidenceMetrics()
        return ConfidenceMetrics(
            mean=round(sum(values) / len(values), 4),
            minimum=round(min(values), 4),
            maximum=round(max(values), 4),
        )


class Detector(ABC):
    model_name: str

    @abstractmethod
    async def predict(self, image: np.ndarray) -> Prediction:
        raise NotImplementedError


class UltralyticsDetector(Detector):
    def __init__(self, settings: Settings) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install the inference extra."
            ) from exc

        self.settings = settings
        self.model_name = settings.model_path
        self.model = YOLO(settings.model_path)
        self._lock = asyncio.Lock()

    async def predict(self, image: np.ndarray) -> Prediction:
        async with self._lock:
            result = await asyncio.to_thread(self._predict_sync, image)
        return result

    def _predict_sync(self, image: np.ndarray) -> Prediction:
        result = self.model.predict(
            source=image,
            conf=self.settings.confidence_threshold,
            iou=self.settings.iou_threshold,
            imgsz=self.settings.image_size,
            device=self.settings.device,
            verbose=False,
        )[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is not None:
            coordinates = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            for box, confidence, class_id_raw in zip(
                coordinates, confidences, classes, strict=True
            ):
                class_id = int(class_id_raw)
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(names[class_id]),
                        confidence=round(float(confidence), 4),
                        box=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                    )
                )

        speed = result.speed or {}
        metrics = SpeedMetrics(
            preprocess_ms=round(float(speed.get("preprocess", 0)), 3),
            inference_ms=round(float(speed.get("inference", 0)), 3),
            postprocess_ms=round(float(speed.get("postprocess", 0)), 3),
        )
        metrics.total_ms = round(
            metrics.preprocess_ms + metrics.inference_ms + metrics.postprocess_ms, 3
        )
        height, width = image.shape[:2]
        return Prediction(width=width, height=height, detections=detections, speed=metrics)


@lru_cache
def get_detector() -> Detector:
    return UltralyticsDetector(get_settings())
