FROM node:22-alpine AS frontend-builder

WORKDIR /workspace/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run lint && npm run build

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MOTION_LAB_WEB=1 \
    MOTION_LAB_CAMERA=/dev/video0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        espeak-ng \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py README.md ./
COPY backend/ ./backend/
COPY models/ ./models/
COPY dance.mp4 ./dance.mp4
COPY --from=frontend-builder /workspace/frontend/dist ./frontend/dist

RUN mkdir -p motion_cache uploaded_sources

EXPOSE 5000

CMD ["python", "app.py", "dance.mp4"]
