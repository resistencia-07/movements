# Backend

The backend uses Flask to expose the existing AI Dance Coach runtime to the React frontend. The pose extraction, scoring, voice coaching, video cache, and OpenCV camera loop remain in the root `app.py`.

## Files

- `../app.py`: actual AI runtime and application entry point
- `app.py`: compatible launcher for the root application
- `web_bridge.py`: Flask app factory and HTTP server wrapper
- `../models/pose_landmarker_lite.task`: MediaPipe model
- `../motion_cache/`: generated pose cache

There is no duplicate AI implementation in `backend/`. `backend/app.py` only launches the root runtime.

## Flask Routes

- `GET /`: compiled React frontend
- `GET /api/health`: service health
- `GET /api/session`: current runtime state
- `GET /api/source-video`: demo reference video
- `GET /api/camera-stream`: backend-owned MJPEG camera feed for React practice
- `POST /api/source`: accepts a local video upload or YouTube URL, then restarts the runtime with the new reference
- `POST /api/control`: forwards commands to the existing runtime

Supported control actions:

```text
open_camera
start_practice
stop_practice
previous
next
restart
pause
voice
language
```

Invalid JSON or actions return JSON errors with an appropriate HTTP status.

## Run

Build the React frontend first:

```bash
cd frontend
npm install
npm run build
cd ..
```

Start Flask and the AI runtime:

```bash
source venv/bin/activate
MOTION_LAB_CAMERA=/dev/video0 python app.py dance.mp4
```

The server listens at `http://127.0.0.1:5000`.

Use `MOTION_LAB_CAMERA` to select another V4L2 path or numeric camera index. In web mode, the browser opens the camera through the `open_camera` API action.

The source chooser accepts `.mp4`, `.mov`, `.avi`, `.mkv`, and `.webm` files. YouTube links use `yt-dlp`; changing the reference restarts `app.py` so MediaPipe can rebuild the motion cache for the new video.
