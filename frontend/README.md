# Frontend

The frontend is a React 19 and Vite application. It preserves the Motion Lab visual system while rendering the workspace from React components.

## Files

- `src/App.jsx`: workspace views, session polling, camera controls, and practice flow
- `src/main.jsx`: React entry point
- `src/motion-lab.css`: existing workspace layout and component styles
- `src/light-theme.css`: existing light theme overrides
- `src/index.css`: React reset and integration styles
- `vite.config.js`: Vite configuration and Flask API proxy

## Development

From the project root:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server proxies `/api` requests to Flask at `http://127.0.0.1:5000`.

## Production Build

```bash
npm run lint
npm run build
```

The build is written to `frontend/dist/`. Flask serves that directory when the root application is started.

## Backend Connection

The frontend uses same-origin relative requests:

- `GET /api/session` reads camera, practice, pose, score, and voice state.
- `POST /api/control` sends camera and practice commands.
- `GET /api/health` checks the Flask process.
- `GET /api/source-video` serves the demo reference video.
- `POST /api/source` accepts a local video file or YouTube URL before practice.
- `GET /api/camera-stream` supplies the backend-owned live camera feed shown in Practice.

The browser does not request the webcam directly, avoiding permission and
device-busy conflicts with OpenCV. No separate CORS configuration is needed
when Flask serves the production build.

Changing the reference restarts the backend so the pose engine can analyze the
new video and rebuild its motion cache.
