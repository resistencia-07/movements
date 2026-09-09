"""Flask web bridge for the existing Motion Lab runtime."""

from pathlib import Path
from threading import Thread
from typing import Callable

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.serving import make_server


Snapshot = Callable[[], dict]
Command = Callable[[str], dict]
CameraStream = Callable[[], object]
SourceUpload = Callable[[object, str], dict]


def create_app(
    snapshot: Snapshot,
    command: Command,
    frontend_dir: str | Path,
    source_video: str | Path | None = None,
    camera_stream: CameraStream | None = None,
    source_upload: SourceUpload | None = None,
) -> Flask:
    """Create the Flask app around the existing runtime callbacks."""
    frontend_root = Path(frontend_dir).resolve()
    source_video_path = Path(source_video).resolve() if source_video else None
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_from_directory(frontend_root, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "service": "motion-lab-core"})

    @app.get("/api/session")
    def session():
        return jsonify(snapshot())

    @app.get("/api/source-video")
    def source_video_route():
        if source_video_path is None or not source_video_path.is_file():
            return jsonify({"ok": False, "error": "Source video not found"}), 404
        return send_file(source_video_path, mimetype="video/mp4")

    @app.get("/api/camera-stream")
    def camera_stream_route():
        if camera_stream is None:
            return jsonify({"ok": False, "error": "Camera stream unavailable"}), 503
        return Response(
            camera_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/source")
    def source_upload_route():
        if source_upload is None:
            return jsonify({"ok": False, "error": "Source upload unavailable"}), 503

        upload = request.files.get("video")
        youtube_url = request.form.get("youtube_url", "").strip()
        if upload is not None and youtube_url:
            return jsonify({"ok": False, "error": "Choose a file or a YouTube URL, not both"}), 400
        if upload is None and not youtube_url:
            return jsonify({"ok": False, "error": "A video file or YouTube URL is required"}), 400

        try:
            return jsonify(source_upload(upload, youtube_url))
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400
        except Exception:
            app.logger.exception("Source upload failed")
            return jsonify({"ok": False, "error": "Could not prepare the reference video"}), 500

    @app.post("/api/control")
    def control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
            return jsonify({"ok": False, "error": "JSON action is required"}), 400

        action = payload["action"].strip()
        if not action:
            return jsonify({"ok": False, "error": "Action cannot be empty"}), 400

        camera = payload.get("camera")
        if camera is not None and not isinstance(camera, str):
            return jsonify({"ok": False, "error": "Camera must be a string"}), 400

        try:
            command_action = action
            if action == "open_camera" and camera:
                command_action = f"open_camera:{camera.strip()}"
            return jsonify({"ok": True, "session": command(command_action)})
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400
        except Exception:
            app.logger.exception("Control action failed")
            return jsonify({"ok": False, "error": "Control action failed"}), 500

    @app.get("/api/<path:unknown>")
    def unknown_api(unknown):
        return jsonify({"ok": False, "error": "Route not found"}), 404

    @app.get("/<path:filename>")
    def frontend_asset(filename):
        return send_from_directory(frontend_root, filename)

    @app.errorhandler(404)
    def not_found(error):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "Route not found"}), 404
        return error

    @app.errorhandler(Exception)
    def server_error(error):
        app.logger.exception("Unhandled server error", exc_info=error)
        return jsonify({"ok": False, "error": "Internal server error"}), 500

    return app


def start_web_bridge(
    snapshot: Snapshot,
    command: Command,
    host: str = "127.0.0.1",
    port: int = 5000,
    frontend_dir: str | Path | None = None,
    source_video: str | Path | None = None,
    camera_stream: CameraStream | None = None,
    source_upload: SourceUpload | None = None,
):
    """Start Flask in a daemon thread and return its WSGI server."""
    if frontend_dir is None:
        raise ValueError("frontend_dir is required")

    app = create_app(snapshot, command, frontend_dir, source_video, camera_stream, source_upload)
    server = make_server(host, port, app, threaded=True)
    Thread(target=server.serve_forever, daemon=True).start()
    return server
