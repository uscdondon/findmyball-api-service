# FindMyBall API Service

AIM230 Case Study - API Deployment

This project is a local FastAPI deployment prototype for FindMyBall, a golf ball computer vision MVP. It exposes a trained YOLOv8 detector through an image-upload REST API so a client can submit an image and receive detected golf-ball bounding boxes.

This is a local deployment prototype for coursework, not a production deployment.

## Live Project Links

- [FindMyBall.io](https://findmyball.io/) - public landing page for the golf ball computer vision MVP
- [Don M. Inouye](https://donminouye.com/) - professional AI / ML portfolio site

## Project Structure

```text
findmyball-api-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── inference.py
│   └── services/
│       ├── __init__.py
│       └── model.py
├── models/
│   └── findmyball_yolov8n.pt
├── sample_images/
│   └── white_red_1280.jpg
├── screenshots/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | API welcome response with docs and health links |
| GET | `/api/v1/health` | Health check and model availability status |
| POST | `/api/v1/predict/image` | Upload an image and receive YOLOv8 detections |

The prediction endpoint accepts an optional `confidence` query parameter. The default is `0.25`, with a minimum of `0.01` and maximum of `1.0`.

## Expected Model Path

The trained YOLOv8 model must be available at:

```text
models/findmyball_yolov8n.pt
```

If the model file is missing, prediction requests return a clear HTTP 500 error.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn app.main:app --reload
```

Swagger docs:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Image prediction:

```bash
curl -X POST "http://localhost:8000/api/v1/predict/image?confidence=0.25" \
  -F "file=@sample_images/white_red_1280.jpg"
```

## Docker Compose

Build and run:

```bash
docker compose up --build
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Image prediction:

```bash
curl -X POST "http://localhost:8000/api/v1/predict/image?confidence=0.25" \
  -F "file=@sample_images/white_red_1280.jpg"
```

Stop the container:

```bash
docker compose down
```

## Screenshot Checklist

1. Swagger docs page at `http://localhost:8000/docs`
2. Successful prediction response from `/api/v1/predict/image`
