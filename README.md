# YOLO Vision API

[English](README.md) | [Türkçe](README_TR.md) | [Deutsch](README_DE.md)

![Demo](docs/demo.svg)

## Portfolio demo

Upload an image or video, call the detection endpoint, and inspect JSON detections, latency and confidence metrics. The UI and sample request are designed for a short screen recording or GIF.

A production-style FastAPI service that exposes Ultralytics YOLO image and video inference through typed REST endpoints. It returns bounding boxes, class labels, confidence statistics, and latency/throughput metrics, with a responsive browser workbench for visual testing.

## Features

- Image upload with annotated browser preview
- Video upload with configurable frame sampling
- Bounding boxes, class IDs, labels, and confidence scores
- Preprocess, inference, postprocess, and total latency metrics
- Video duration, processed frames, effective FPS, and average inference time
- Lazy model loading and serialized inference for model safety
- Configurable model path for official or custom `.pt` weights
- Upload size limits, media validation, temporary-file cleanup, and typed errors
- Docker image, Compose setup, tests, Ruff, and GitHub Actions

> Confidence is not the same as model accuracy. Endpoint responses report prediction confidence and runtime speed. Dataset-level accuracy such as mAP50-95 must be measured with labeled validation data using YOLO validation or benchmark mode.

## API

![Architecture](docs/demo.svg)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service and configured model status |
| `POST` | `/api/v1/detect/image` | Detect objects in an image |
| `POST` | `/api/v1/detect/video` | Sample a video and return per-frame detections |

Swagger UI is available at `/docs`.

## Docker quick start

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:8001`. The first inference may take longer while official model weights are downloaded. Set `MODEL_PATH=/path/to/best.pt` to use custom weights.

## Local development

```bash
python -m venv .venv
pip install -e ".[inference,dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8001
```

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_PATH` | `yolo26n.pt` | Official model name or custom weights path |
| `CONFIDENCE_THRESHOLD` | `0.25` | Minimum prediction confidence |
| `IOU_THRESHOLD` | `0.70` | NMS intersection-over-union threshold |
| `IMAGE_SIZE` | `640` | Inference image size |
| `DEVICE` | `cpu` | `cpu`, GPU index, or supported accelerator |
| `VIDEO_FRAME_STRIDE` | `10` | Process every nth frame |
| `MAX_VIDEO_FRAMES` | `120` | Maximum sampled frames per request |

## Quality checks

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

Tests use a fake detector, so CI validates the API contract without downloading model weights or PyTorch.

For hiring review, report dataset-level mAP50-95 and class-wise precision/recall from a labeled validation split. Runtime confidence and FPS in this service are operational metrics, not accuracy claims.

## Production roadmap

- Queue long videos as background jobs and return job IDs
- Store annotated output in S3-compatible object storage
- Add API keys, rate limiting, Prometheus metrics, and OpenTelemetry traces
- Add model registry/version metadata and labeled-dataset mAP reports
- Export to ONNX/TensorRT for deployment-specific benchmarking
