# AI Dance Coach

AI Dance Coach is a local movement-analysis application for dance practice. It combines an OpenCV and MediaPipe pose engine with a React frontend served by Flask.

## Project Structure

```text
app.py                 Main AI runtime and Flask launcher
backend/README.md      Backend and API documentation
frontend/README.md     React frontend documentation
backend/               Flask bridge and launcher
frontend/              React/Vite application
models/                MediaPipe pose model
motion_cache/          Generated dance-pose cache
dance.mp4              Demo dance reference video
requirements.txt       Python dependencies
```

## Requirements

- Linux with a V4L2-compatible camera
- Python 3.11
- Node.js and npm
- Optional: `espeak-ng` for voice coaching
- `yt-dlp` for YouTube reference links

## Install

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
```

## Run

```bash
source venv/bin/activate
MOTION_LAB_CAMERA=/dev/video0 python app.py dance.mp4
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Docker

Build and run the complete app with Docker Compose:

```bash
docker compose up --build
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Compose maps `/dev/video0` into the container by default. Select another host device with `MOTION_LAB_DEVICE=/dev/video2 docker compose up --build`.

The CI workflow validates the Compose file and image build. The container workflow publishes `main` and version tags to GitHub Container Registry.

The root `app.py` is the main entry point. `backend/app.py` is a compatible launcher that runs the same application.

## Application Flow

```text
Browser
  -> React frontend in frontend/dist
  -> Flask routes in backend/web_bridge.py
  -> existing app.py camera and MediaPipe runtime
  -> JSON session state
  -> React UI
```

The Camera Lab lets you upload a local reference video or paste a YouTube link before practice. The backend prepares the source and restarts the runtime so MediaPipe analyzes the selected video. The camera opens through `/api/control`, then Practice displays the live session returned by `/api/session`.

## API

- `GET /api/health`
- `GET /api/session`
- `GET /api/source-video`
- `POST /api/source` with a `video` multipart file or `youtube_url` form field
- `POST /api/control` with `open_camera`, `start_practice`, `stop_practice`, `previous`, `next`, `restart`, `pause`, `voice`, or `language`

## Validation

```bash
python -m py_compile app.py backend/app.py backend/web_bridge.py
cd frontend && npm run lint && npm run build
```

The pose model and `dance.mp4` are required runtime assets. Generated caches, virtual environments, build output, and temporary media should not be committed.
