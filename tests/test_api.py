from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.schemas import BoundingBox, Detection, SpeedMetrics
from app.services.detector import Detector, Prediction, get_detector


class FakeDetector(Detector):
    model_name = "fake-yolo.pt"

    async def predict(self, image: np.ndarray) -> Prediction:
        height, width = image.shape[:2]
        return Prediction(
            width=width,
            height=height,
            detections=[
                Detection(
                    class_id=0,
                    class_name="person",
                    confidence=0.91,
                    box=BoundingBox(x1=1, y1=2, x2=20, y2=30),
                )
            ],
            speed=SpeedMetrics(inference_ms=12.5, total_ms=14.0),
        )


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 48), color=(20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_does_not_load_model() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_image_detection_returns_boxes_and_metrics() -> None:
    app = create_app()
    app.dependency_overrides[get_detector] = lambda: FakeDetector()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["detection_count"] == 1
    assert body["detections"][0]["class_name"] == "person"
    assert body["speed"]["inference_ms"] == 12.5


def test_image_endpoint_rejects_wrong_media_type() -> None:
    app = create_app()
    app.dependency_overrides[get_detector] = lambda: FakeDetector()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("sample.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 415
