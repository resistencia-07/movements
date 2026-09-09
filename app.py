import cv2
import numpy as np
import mediapipe as mp

import gzip
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from bisect import bisect_right
from pathlib import Path
from urllib.parse import urlparse

from backend.web_bridge import start_web_bridge

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Camera selection. Set MOTION_LAB_CAMERA to a device path or index when needed.
# Prefer the standard V4L2 device node first because the Linux by-id path is
# often stale when the camera is re-enumerated and doesn't match the live device.
USB_CAMERA = "/dev/video0"
CAMERA_SOURCE = os.getenv("MOTION_LAB_CAMERA", USB_CAMERA)
camera_options_cache = None


def resolve_camera_candidates():
    """Return the ordered list of possible camera sources for this machine."""
    env_value = os.getenv("MOTION_LAB_CAMERA")
    if env_value:
        candidates = [item.strip() for item in env_value.split(",") if item.strip()]
    else:
        candidates = [CAMERA_SOURCE]

    candidates.extend(
        str(device)
        for device in sorted(Path("/dev").glob("video*"))
        if device.is_char_device()
    )
    candidates.extend(["0", "1"])

    ordered = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return ordered


def detect_camera_options():
    """Return only camera sources that OpenCV can actually open."""
    global camera_options_cache

    if camera_options_cache is not None:
        return camera_options_cache

    configured = [
        item.strip()
        for item in os.getenv("MOTION_LAB_CAMERA", "").split(",")
        if item.strip()
    ]
    device_paths = [
        str(device)
        for device in sorted(Path("/dev").glob("video*"))
        if device.is_char_device()
    ]
    candidates = configured + device_paths
    if not candidates:
        candidates = ["0", "1"]

    detected = []
    physical_devices = set()
    for candidate in candidates:
        if candidate in detected:
            continue
        if candidate.startswith("/dev/video"):
            device_name = Path(candidate).name
            physical_path = os.path.realpath(
                f"/sys/class/video4linux/{device_name}/device"
            )
            if physical_path in physical_devices:
                continue
        source = int(candidate) if candidate.isdigit() else candidate
        camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
        opened = camera.isOpened()
        readable = False
        if opened:
            for _ in range(3):
                readable, _ = camera.read()
                if readable:
                    break
        camera.release()
        if opened and readable:
            detected.append(candidate)
            if candidate.startswith("/dev/video"):
                physical_devices.add(physical_path)

    camera_options_cache = detected
    return detected

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
WINDOW_NAME = "AR Dance Teacher V11"

DEFAULT_DANCE_FILE = "dance.mp4"

FULLSCREEN_DEFAULT = True

# Teacher
TEACHER_TARGET_BODY_SIZE = 520.0
TEACHER_SIZE_MIN = 330.0
TEACHER_SIZE_MAX = 720.0
TEACHER_SIZE_STEP = 30.0

TEACHER_ANCHOR_X = int(CAMERA_WIDTH * 0.79)
TEACHER_ANCHOR_Y = int(CAMERA_HEIGHT * 0.56)
TEACHER_MIRROR = True

# MediaPipe
POSE_VISIBILITY_THRESHOLD = 0.45
TEACHER_VISIBILITY_THRESHOLD = 0.30
EXTRACTION_SMOOTH_ALPHA = 0.68

# Key-pose teaching
TARGET_KEY_POSES = 15
MIN_KEY_POSES = 8
MAX_KEY_POSES = 20

PASS_THRESHOLD = 80.0

# User must remain above the threshold for this long.
MATCH_HOLD_SECONDS = 0.45

# After a correct pose, show CORRECT for this long.
SUCCESS_DISPLAY_SECONDS = 1.0

# Avoid repeating correction voice continuously.
CORRECTION_VOICE_COOLDOWN = 2.7

# Static pose scoring.
ANGLE_WEIGHT = 0.70
POSITION_WEIGHT = 0.30

# Angle difference at or above this is treated as 0 accuracy.
ANGLE_ZERO_SCORE_ERROR = 55.0

# Normalized point distance at or above this is treated as 0.
POSITION_ZERO_SCORE_ERROR = 0.95

# Cache
CACHE_FOLDER = "motion_cache"
CACHE_VERSION = 11
SOURCE_UPLOAD_FOLDER = Path("uploaded_sources")
ALLOWED_SOURCE_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Source preview
SOURCE_PREVIEW_ENABLED_DEFAULT = True

# Portrait max box
PORTRAIT_PREVIEW_MAX_W = 250
PORTRAIT_PREVIEW_MAX_H = 390

# Landscape max box
LANDSCAPE_PREVIEW_MAX_W = 370
LANDSCAPE_PREVIEW_MAX_H = 220

PREVIEW_TOP = 18
PREVIEW_CENTER_X = int(CAMERA_WIDTH * 0.50)

# Voice
VOICE_ENABLED_DEFAULT = True

LANGUAGES = [
    "ENGLISH",
    "TAMIL",
    "TANGLISH",
]

VOICE_SPEED = 155

WEB_MODE = os.getenv("MOTION_LAB_WEB", "1") == "1"

# INPUT DANCE VIDEO

DANCE_VIDEO_PATH = (
    sys.argv[1]
    if len(sys.argv) >= 2
    else DEFAULT_DANCE_FILE
)

DANCE_VIDEO_PATH = os.path.abspath(
    DANCE_VIDEO_PATH
)

if not os.path.exists(DANCE_VIDEO_PATH):
    print("\nERROR: Dance video not found:")
    print(DANCE_VIDEO_PATH)
    print("\nRun:")
    print("python -u app_v11_final.py dance.mp4")
    sys.exit(1)

# SOURCE VIDEO METADATA
metadata_cap = cv2.VideoCapture(
    DANCE_VIDEO_PATH
)

if not metadata_cap.isOpened():
    print("\nERROR: Could not open dance video.")
    sys.exit(1)

SOURCE_VIDEO_WIDTH = int(
    metadata_cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

SOURCE_VIDEO_HEIGHT = int(
    metadata_cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

SOURCE_VIDEO_FPS = float(
    metadata_cap.get(
        cv2.CAP_PROP_FPS
    )
)

SOURCE_VIDEO_FRAME_COUNT = int(
    metadata_cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

if not SOURCE_VIDEO_FPS or SOURCE_VIDEO_FPS <= 0:
    SOURCE_VIDEO_FPS = 30.0

SOURCE_VIDEO_DURATION_MS = int(
    max(
        0,
        SOURCE_VIDEO_FRAME_COUNT
        / SOURCE_VIDEO_FPS
        * 1000.0
    )
)

metadata_cap.release()

SOURCE_ORIENTATION = (
    "PORTRAIT"
    if SOURCE_VIDEO_HEIGHT > SOURCE_VIDEO_WIDTH
    else "LANDSCAPE"
)

print()
print("Source video:")
print(DANCE_VIDEO_PATH)
print(
    f"Video size: "
    f"{SOURCE_VIDEO_WIDTH}x{SOURCE_VIDEO_HEIGHT}"
)
print(
    f"Orientation: {SOURCE_ORIENTATION}"
)

# VIDEO FINGERPRINT

def video_fingerprint(path):
    """
    Detect replaced dance.mp4 automatically.
    """
    chunk_size = 1024 * 1024
    size = os.path.getsize(path)

    hasher = hashlib.sha256()
    hasher.update(
        str(size).encode("utf-8")
    )

    with open(path, "rb") as file:
        hasher.update(
            file.read(chunk_size)
        )

        if size > chunk_size:
            file.seek(
                max(
                    0,
                    size - chunk_size
                )
            )

            hasher.update(
                file.read(chunk_size)
            )

    return hasher.hexdigest()


CURRENT_VIDEO_FINGERPRINT = (
    video_fingerprint(
        DANCE_VIDEO_PATH
    )
)

os.makedirs(
    CACHE_FOLDER,
    exist_ok=True
)

dance_stem = Path(
    DANCE_VIDEO_PATH
).stem

MOTION_CACHE_PATH = os.path.join(
    CACHE_FOLDER,
    f"{dance_stem}_motion_v11.json.gz"
)

LEGACY_MOTION_CACHE_PATH = os.path.join(
    CACHE_FOLDER,
    f"{dance_stem}_motion_v11.json"
)

# MEDIAPIPE MODEL

MODEL_FOLDER = "models"

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "pose_landmarker_lite.task"
)

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/"
    "pose_landmarker/"
    "pose_landmarker_lite/"
    "float16/1/"
    "pose_landmarker_lite.task"
)


def ensure_pose_model():
    if os.path.exists(MODEL_PATH):
        print(
            f"Pose model found: "
            f"{MODEL_PATH}"
        )
        return

    print(
        "Pose model missing. "
        "Downloading..."
    )

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print(
            "Pose model downloaded."
        )

    except Exception as error:
        print(
            "\nERROR: Could not "
            "download MediaPipe model."
        )
        print(error)
        sys.exit(1)


ensure_pose_model()

# MEDIAPIPE TASKS API

BaseOptions = (
    mp.tasks.BaseOptions
)

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


def create_pose_landmarker():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return (
        PoseLandmarker.create_from_options(
            options
        )
    )
# LANDMARK CONSTANTS

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),

    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),

    (9, 10),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),
    (23, 24),

    (23, 25),
    (25, 27),
    (27, 31),

    (24, 26),
    (26, 28),
    (28, 32),
]

MAJOR_JOINTS = [
    11, 12,
    13, 14,
    15, 16,
    23, 24,
    25, 26,
    27, 28,
    31, 32,
]

ANGLE_JOINTS = {
    "LEFT_ELBOW": (
        11, 13, 15
    ),
    "RIGHT_ELBOW": (
        12, 14, 16
    ),
    "LEFT_SHOULDER": (
        13, 11, 23
    ),
    "RIGHT_SHOULDER": (
        14, 12, 24
    ),
    "LEFT_KNEE": (
        23, 25, 27
    ),
    "RIGHT_KNEE": (
        24, 26, 28
    ),
}

# MATH HELPERS

def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def midpoint(a, b):
    return (
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
    )


def calculate_angle(a, b, c):
    """
    Angle ABC in degrees.
    """
    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    c = np.asarray(
        c,
        dtype=np.float32
    )

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine = float(
        np.dot(ba, bc)
        / denominator
    )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0
        )
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def pose_body_measure(
    landmarks
):
    points = [
        (
            float(lm["x"]),
            float(lm["y"])
        )
        for lm in landmarks
    ]

    visibility = [
        float(
            lm.get(
                "visibility",
                1.0
            )
        )
        for lm in landmarks
    ]

    segments = [
        (0, 11),
        (0, 12),
        (11, 12),
        (23, 24),
        (11, 23),
        (12, 24),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28),
    ]

    lengths = []

    for a, b in segments:
        if (
            visibility[a] >= 0.35
            and visibility[b] >= 0.35
        ):
            lengths.append(
                distance(
                    points[a],
                    points[b]
                )
            )

    if len(lengths) < 5:
        return None

    return float(
        np.mean(lengths) * 6.0
    )

# REFERENCE MOTION SMOOTHING

def smooth_landmarks(
    current,
    previous
):
    if previous is None:
        return current

    output = []

    for current_lm, previous_lm in zip(
        current,
        previous
    ):
        visibility = float(
            current_lm.get(
                "visibility",
                1.0
            )
        )

        alpha = (
            EXTRACTION_SMOOTH_ALPHA
            if visibility >= 0.45
            else 0.35
        )

        output.append(
            {
                "x": (
                    alpha
                    * current_lm["x"]
                    +
                    (1.0 - alpha)
                    * previous_lm["x"]
                ),
                "y": (
                    alpha
                    * current_lm["y"]
                    +
                    (1.0 - alpha)
                    * previous_lm["y"]
                ),
                "z": (
                    alpha
                    * current_lm["z"]
                    +
                    (1.0 - alpha)
                    * previous_lm["z"]
                ),
                "visibility": visibility,
            }
        )

    return output

# ANALYZE DANCE VIDEO

def extract_dance_motion(
    video_path,
    cache_path
):
    print()
    print(
        "========================================"
    )
    print(
        "ANALYZING DANCE VIDEO"
    )
    print(
        "========================================"
    )
    print(video_path)
    print()
    print(
        "This runs only when "
        "the source video changed."
    )
    print()

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():
        print(
            "ERROR: Could not open "
            "dance video."
        )
        sys.exit(1)

    fps = float(
        video.get(
            cv2.CAP_PROP_FPS
        )
    )

    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    detector = (
        create_pose_landmarker()
    )

    frames = []

    hip_x_values = []
    hip_y_values = []
    body_measures = []

    previous_smoothed = None

    frame_index = 0
    detected_count = 0

    while True:
        ok, frame = video.read()

        if not ok:
            break

        timestamp_ms = int(
            frame_index
            / fps
            * 1000.0
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb,
        )

        try:
            result = (
                detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

        except Exception as error:
            print(
                f"\nPose error at frame "
                f"{frame_index}: {error}"
            )

            result = None

        if (
            result is not None
            and result.pose_landmarks
        ):
            raw = []

            for lm in (
                result.pose_landmarks[0]
            ):
                raw.append(
                    {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(
                            lm.visibility
                            if lm.visibility
                            is not None
                            else 1.0
                        ),
                    }
                )

            landmarks = (
                smooth_landmarks(
                    raw,
                    previous_smoothed
                )
            )

            previous_smoothed = (
                landmarks
            )

            frames.append(
                {
                    "time_ms": (
                        timestamp_ms
                    ),
                    "landmarks": (
                        landmarks
                    ),
                }
            )

            detected_count += 1

            hip = midpoint(
                (
                    landmarks[23]["x"],
                    landmarks[23]["y"],
                ),
                (
                    landmarks[24]["x"],
                    landmarks[24]["y"],
                ),
            )

            hip_x_values.append(
                hip[0]
            )

            hip_y_values.append(
                hip[1]
            )

            measure = (
                pose_body_measure(
                    landmarks
                )
            )

            if measure is not None:
                body_measures.append(
                    measure
                )

        frame_index += 1

        if (
            total_frames > 0
            and (
                frame_index % 30 == 0
                or
                frame_index
                == total_frames
            )
        ):
            percent = (
                frame_index
                / total_frames
                * 100.0
            )

            print(
                f"\rAnalyzing: "
                f"{percent:5.1f}%   "
                f"Poses: "
                f"{detected_count}",
                end="",
                flush=True,
            )

    print()

    detector.close()
    video.release()

    if len(frames) < 5:
        print()
        print(
            "ERROR: Not enough "
            "full-body poses detected."
        )
        print(
            "Use a dance video with "
            "one clear full-body dancer."
        )
        sys.exit(1)

    reference_body = (
        float(
            np.median(
                body_measures
            )
        )
        if body_measures
        else 1.0
    )

    reference_hip_x = float(
        np.median(
            hip_x_values
        )
    )

    reference_hip_y = float(
        np.median(
            hip_y_values
        )
    )

    duration_ms = max(
        1,
        int(
            frames[-1][
                "time_ms"
            ]
        )
    )

    data = {
        "cache_version": (
            CACHE_VERSION
        ),
        "video_fingerprint": (
            CURRENT_VIDEO_FINGERPRINT
        ),
        "source_video": (
            os.path.basename(
                video_path
            )
        ),
        "fps": float(fps),
        "duration_ms": (
            duration_ms
        ),
        "detected_frames": (
            len(frames)
        ),
        "reference": {
            "hip_x": (
                reference_hip_x
            ),
            "hip_y": (
                reference_hip_y
            ),
            "body_measure": (
                reference_body
            ),
        },
        "frames": frames,
    }

    with gzip.open(
        cache_path,
        "wt",
        encoding="utf-8",
        compresslevel=6,
    ) as file:
        json.dump(
            data,
            file,
            separators=(",", ":"),
        )

    print()
    print(
        "Motion extraction complete."
    )
    print(
        f"Detected pose frames: "
        f"{len(frames)}"
    )
    print(
        f"Saved cache: "
        f"{cache_path}"
    )

    return data


def load_or_extract_motion():
    cache_path = None

    if os.path.exists(MOTION_CACHE_PATH):
        cache_path = MOTION_CACHE_PATH
    elif os.path.exists(LEGACY_MOTION_CACHE_PATH):
        cache_path = LEGACY_MOTION_CACHE_PATH

    if cache_path:
        print()
        print(
            "Checking V11 motion cache..."
        )

        try:
            opener = (
                gzip.open
                if cache_path.endswith(".gz")
                else open
            )

            with opener(
                cache_path,
                "rt",
                encoding="utf-8"
            ) as file:
                data = json.load(
                    file
                )

            valid = (
                data.get(
                    "cache_version"
                )
                == CACHE_VERSION
                and
                data.get(
                    "video_fingerprint"
                )
                == CURRENT_VIDEO_FINGERPRINT
                and
                data.get("frames")
                and
                data.get("reference")
            )

            if valid:
                print(
                    "Dance video unchanged."
                )
                print(
                    "Loading cached motion:"
                )
                print(
                    cache_path
                )

                if cache_path == LEGACY_MOTION_CACHE_PATH:
                    with gzip.open(
                        MOTION_CACHE_PATH,
                        "wt",
                        encoding="utf-8",
                        compresslevel=6,
                    ) as compressed_file:
                        json.dump(
                            data,
                            compressed_file,
                            separators=(",", ":"),
                        )

                    print(
                        "Migrated legacy cache to gzip."
                    )

                return data

            print(
                "Video changed or "
                "cache is old."
            )
            print(
                "Re-analyzing current "
                "dance video..."
            )

        except Exception as error:
            print(
                "Cache invalid:",
                error
            )

    return extract_dance_motion(
        DANCE_VIDEO_PATH,
        MOTION_CACHE_PATH
    )


motion_data = (
    load_or_extract_motion()
)

MOTION_FRAMES = (
    motion_data["frames"]
)

MOTION_TIMES = [
    int(
        frame["time_ms"]
    )
    for frame in MOTION_FRAMES
]

MOTION_DURATION_MS = max(
    1,
    int(
        motion_data[
            "duration_ms"
        ]
    )
)

REFERENCE_HIP_X = float(
    motion_data[
        "reference"
    ]["hip_x"]
)

REFERENCE_HIP_Y = float(
    motion_data[
        "reference"
    ]["hip_y"]
)

REFERENCE_BODY_MEASURE = max(
    0.001,
    float(
        motion_data[
            "reference"
        ]["body_measure"]
    )
)
# KEY POSE EXTRACTION

def raw_pose_normalized_vector(
    landmarks
):
    """
    Build a body-size-independent feature vector.
    """
    left_hip = np.array(
        [
            landmarks[23]["x"],
            landmarks[23]["y"],
        ],
        dtype=np.float32,
    )

    right_hip = np.array(
        [
            landmarks[24]["x"],
            landmarks[24]["y"],
        ],
        dtype=np.float32,
    )

    left_shoulder = np.array(
        [
            landmarks[11]["x"],
            landmarks[11]["y"],
        ],
        dtype=np.float32,
    )

    right_shoulder = np.array(
        [
            landmarks[12]["x"],
            landmarks[12]["y"],
        ],
        dtype=np.float32,
    )

    hip_center = (
        left_hip
        + right_hip
    ) / 2.0

    shoulder_center = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    torso = float(
        np.linalg.norm(
            shoulder_center
            - hip_center
        )
    )

    shoulder_width = float(
        np.linalg.norm(
            right_shoulder
            - left_shoulder
        )
    )

    scale = max(
        torso,
        shoulder_width,
        0.03,
    )

    values = []

    for joint in MAJOR_JOINTS:
        lm = landmarks[joint]

        point = np.array(
            [
                lm["x"],
                lm["y"],
            ],
            dtype=np.float32,
        )

        normalized = (
            point - hip_center
        ) / scale

        visibility = float(
            lm.get(
                "visibility",
                1.0
            )
        )

        if visibility < 0.25:
            normalized *= 0.25

        values.extend(
            [
                float(
                    normalized[0]
                ),
                float(
                    normalized[1]
                ),
            ]
        )

    raw_points = [
        (
            float(lm["x"]),
            float(lm["y"]),
        )
        for lm in landmarks
    ]

    for indexes in (
        ANGLE_JOINTS.values()
    ):
        angle = calculate_angle(
            raw_points[indexes[0]],
            raw_points[indexes[1]],
            raw_points[indexes[2]],
        )

        values.append(
            angle / 180.0
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def feature_distance(a, b):
    return float(
        np.linalg.norm(
            a - b
        )
        / math.sqrt(
            max(
                1,
                len(a)
            )
        )
    )


def choose_key_pose_count():
    duration_seconds = (
        MOTION_DURATION_MS
        / 1000.0
    )

    estimate = int(
        round(
            duration_seconds
            * 1.15
        )
    )

    estimate = int(
        np.clip(
            estimate,
            MIN_KEY_POSES,
            MAX_KEY_POSES,
        )
    )

    # Keep the requested ~15 as the center target.
    if (
        MIN_KEY_POSES
        <= TARGET_KEY_POSES
        <= MAX_KEY_POSES
    ):
        estimate = int(
            round(
                (
                    estimate
                    + TARGET_KEY_POSES
                )
                / 2
            )
        )

    return max(
        2,
        min(
            estimate,
            len(MOTION_FRAMES)
        )
    )


def select_key_pose_indices(
    frames
):
    """
    Greedy pose-diversity selection.

    It chooses poses that are visually different from
    already selected poses, while keeping a minimum
    time separation.
    """
    count = choose_key_pose_count()

    if len(frames) <= count:
        return list(
            range(
                len(frames)
            )
        )

    features = [
        raw_pose_normalized_vector(
            frame["landmarks"]
        )
        for frame in frames
    ]

    times = [
        int(
            frame["time_ms"]
        )
        for frame in frames
    ]

    selected = {
        0,
        len(frames) - 1,
    }

    min_gap_ms = max(
        300,
        int(
            MOTION_DURATION_MS
            / max(
                1,
                count * 3
            )
        )
    )

    candidate_indexes = list(
        range(
            1,
            len(frames) - 1
        )
    )

    while (
        len(selected) < count
    ):
        best_index = None
        best_score = -1.0

        for index in candidate_indexes:
            if index in selected:
                continue

            if any(
                abs(
                    times[index]
                    - times[s]
                )
                < min_gap_ms
                for s in selected
            ):
                continue

            min_pose_distance = min(
                feature_distance(
                    features[index],
                    features[s]
                )
                for s in selected
            )

            # Slight preference for frames away from
            # already-selected times.
            nearest_time_gap = min(
                abs(
                    times[index]
                    - times[s]
                )
                for s in selected
            )

            temporal_bonus = (
                nearest_time_gap
                / max(
                    1,
                    MOTION_DURATION_MS
                )
            )

            score = (
                min_pose_distance
                + 0.18
                * temporal_bonus
            )

            if score > best_score:
                best_score = score
                best_index = index

        if best_index is None:
            # Relax time separation if needed.
            remaining = [
                index
                for index
                in candidate_indexes
                if index not in selected
            ]

            if not remaining:
                break

            best_index = max(
                remaining,
                key=lambda index: min(
                    feature_distance(
                        features[index],
                        features[s]
                    )
                    for s in selected
                ),
            )

        selected.add(
            best_index
        )

    result = sorted(
        selected,
        key=lambda index: (
            times[index]
        ),
    )

    return result


KEY_POSE_INDICES = (
    select_key_pose_indices(
        MOTION_FRAMES
    )
)

KEY_POSES = [
    MOTION_FRAMES[index]
    for index in KEY_POSE_INDICES
]

TOTAL_KEY_POSES = len(
    KEY_POSES
)

print()
print(
    "========================================"
)
print(
    "V11 KEY POSES"
)
print(
    "========================================"
)
print(
    f"Selected key poses: "
    f"{TOTAL_KEY_POSES}"
)

for index, pose in enumerate(
    KEY_POSES
):
    print(
        f"Pose {index + 1:02d}: "
        f"{pose['time_ms'] / 1000.0:.2f}s"
    )

# TEACHER TRANSFORM

teacher_size = float(
    TEACHER_TARGET_BODY_SIZE
)


def transform_teacher_landmarks(
    landmarks
):
    scale = (
        teacher_size
        / REFERENCE_BODY_MEASURE
    )

    points = []
    visibility = []
    depths = []

    for lm in landmarks:
        dx = (
            float(lm["x"])
            - REFERENCE_HIP_X
        )

        dy = (
            float(lm["y"])
            - REFERENCE_HIP_Y
        )

        z = float(
            lm["z"]
        )

        if TEACHER_MIRROR:
            dx = -dx

        x = int(
            TEACHER_ANCHOR_X
            + dx * scale
        )

        y = int(
            TEACHER_ANCHOR_Y
            + dy * scale
        )

        points.append(
            (x, y)
        )

        visibility.append(
            float(
                lm.get(
                    "visibility",
                    1.0
                )
            )
        )

        depths.append(z)

    return (
        points,
        visibility,
        depths,
    )

# LIVE USER POSE

def get_mirrored_user_pose(
    landmarks,
    width,
    height
):
    points = []
    visibility = []

    for lm in landmarks:
        x = int(
            (
                1.0
                - float(lm.x)
            )
            * width
        )

        y = int(
            float(lm.y)
            * height
        )

        points.append(
            (x, y)
        )

        visibility.append(
            float(
                lm.visibility
                if lm.visibility
                is not None
                else 1.0
            )
        )

    return (
        points,
        visibility,
    )


def draw_user_pose(
    frame,
    points,
    visibility
):
    for start, end in (
        POSE_CONNECTIONS
    ):
        if (
            visibility[start]
            >= POSE_VISIBILITY_THRESHOLD
            and
            visibility[end]
            >= POSE_VISIBILITY_THRESHOLD
        ):
            cv2.line(
                frame,
                points[start],
                points[end],
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

    for index in MAJOR_JOINTS:
        if (
            visibility[index]
            >= POSE_VISIBILITY_THRESHOLD
        ):
            cv2.circle(
                frame,
                points[index],
                7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.circle(
                frame,
                points[index],
                4,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

# HUMAN-LIKE 2D TRAINER AVATAR

def tapered_polygon(
    a,
    b,
    width_a,
    width_b
):
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    length = max(
        1.0,
        math.hypot(
            dx,
            dy
        )
    )

    nx = -dy / length
    ny = dx / length

    return np.array(
        [
            (
                int(
                    ax
                    + nx
                    * width_a
                    / 2
                ),
                int(
                    ay
                    + ny
                    * width_a
                    / 2
                ),
            ),
            (
                int(
                    ax
                    - nx
                    * width_a
                    / 2
                ),
                int(
                    ay
                    - ny
                    * width_a
                    / 2
                ),
            ),
            (
                int(
                    bx
                    - nx
                    * width_b
                    / 2
                ),
                int(
                    by
                    - ny
                    * width_b
                    / 2
                ),
            ),
            (
                int(
                    bx
                    + nx
                    * width_b
                    / 2
                ),
                int(
                    by
                    + ny
                    * width_b
                    / 2
                ),
            ),
        ],
        dtype=np.int32,
    )


def draw_tapered_limb(
    canvas,
    a,
    b,
    width_a,
    width_b,
    fill_color,
    outline_color,
):
    outer = tapered_polygon(
        a,
        b,
        width_a + 7,
        width_b + 7,
    )

    inner = tapered_polygon(
        a,
        b,
        width_a,
        width_b,
    )

    cv2.fillConvexPoly(
        canvas,
        outer,
        outline_color,
        cv2.LINE_AA,
    )

    cv2.fillConvexPoly(
        canvas,
        inner,
        fill_color,
        cv2.LINE_AA,
    )


def draw_human_trainer(
    frame,
    points,
    visibility,
    depths,
    gender
):
    """
    Clean 2D human trainer:
      skin
      face + hair
      sports top
      pants
      hands
      shoes
      tapered human-like limbs
    """
    overlay = frame.copy()

    # BGR colors
    skin = (
        168,
        198,
        226,
    )

    skin_shadow = (
        120,
        148,
        176,
    )

    outline = (
        52,
        54,
        62,
    )

    hair = (
        32,
        34,
        40,
    )

    if gender == "male":
        shirt = (
            215,
            125,
            45,
        )

        shirt_light = (
            240,
            165,
            85,
        )

        pants = (
            68,
            64,
            62,
        )

        shoulder_expand = 1.07

    else:
        shirt = (
            165,
            80,
            210,
        )

        shirt_light = (
            195,
            115,
            235,
        )

        pants = (
            72,
            62,
            82,
        )

        shoulder_expand = 1.00

    shoes = (
        225,
        225,
        230,
    )

    LS = 11
    RS = 12
    LE = 13
    RE = 14
    LW = 15
    RW = 16
    LH = 23
    RH = 24
    LK = 25
    RK = 26
    LA = 27
    RA = 28

    # Shadow

    visible_ankles = [
        points[index]
        for index in [LA, RA]
        if (
            visibility[index]
            >= TEACHER_VISIBILITY_THRESHOLD
        )
    ]

    if visible_ankles:
        shadow_center = (
            int(
                np.mean(
                    [
                        p[0]
                        for p
                        in visible_ankles
                    ]
                )
            ),
            int(
                max(
                    p[1]
                    for p
                    in visible_ankles
                )
                + 12
            ),
        )

        cv2.ellipse(
            overlay,
            shadow_center,
            (72, 16),
            0,
            0,
            360,
            (45, 45, 48),
            -1,
            cv2.LINE_AA,
        )

    # Legs - pants

    for a, b, wa, wb in [
        (LH, LK, 34, 27),
        (LK, LA, 28, 20),
        (RH, RK, 34, 27),
        (RK, RA, 28, 20),
    ]:
        if (
            visibility[a]
            >= TEACHER_VISIBILITY_THRESHOLD
            and
            visibility[b]
            >= TEACHER_VISIBILITY_THRESHOLD
        ):
            draw_tapered_limb(
                overlay,
                points[a],
                points[b],
                wa,
                wb,
                pants,
                outline,
            )

    # Torso - sports shirt

    if all(
        visibility[index]
        >= TEACHER_VISIBILITY_THRESHOLD
        for index in [
            LS,
            RS,
            LH,
            RH,
        ]
    ):
        shoulder_mid = midpoint(
            points[LS],
            points[RS]
        )

        hip_mid = midpoint(
            points[LH],
            points[RH]
        )

        torso_center = (
            int(
                (
                    shoulder_mid[0]
                    + hip_mid[0]
                )
                / 2
            ),
            int(
                (
                    shoulder_mid[1]
                    + hip_mid[1]
                )
                / 2
            ),
        )

        def expand(
            point,
            x_factor,
            y_factor=1.0,
        ):
            return (
                int(
                    torso_center[0]
                    +
                    (
                        point[0]
                        - torso_center[0]
                    )
                    * x_factor
                ),
                int(
                    torso_center[1]
                    +
                    (
                        point[1]
                        - torso_center[1]
                    )
                    * y_factor
                ),
            )

        torso_outer = np.array(
            [
                expand(
                    points[LS],
                    shoulder_expand * 1.10,
                ),
                expand(
                    points[RS],
                    shoulder_expand * 1.10,
                ),
                expand(
                    points[RH],
                    1.02,
                ),
                expand(
                    points[LH],
                    1.02,
                ),
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(
            overlay,
            torso_outer,
            outline,
            cv2.LINE_AA,
        )

        torso_inner = np.array(
            [
                expand(
                    points[LS],
                    shoulder_expand,
                ),
                expand(
                    points[RS],
                    shoulder_expand,
                ),
                expand(
                    points[RH],
                    0.94,
                ),
                expand(
                    points[LH],
                    0.94,
                ),
            ],
            dtype=np.int32,
        )

        cv2.fillConvexPoly(
            overlay,
            torso_inner,
            shirt,
            cv2.LINE_AA,
        )

        # Shirt highlight
        cv2.line(
            overlay,
            (
                torso_center[0] - 8,
                int(shoulder_mid[1] + 16),
            ),
            (
                torso_center[0] - 8,
                int(hip_mid[1] - 16),
            ),
            shirt_light,
            6,
            cv2.LINE_AA,
        )
    # Arms - skin

    for a, b, wa, wb in [
        (LS, LE, 24, 19),
        (LE, LW, 20, 14),
        (RS, RE, 24, 19),
        (RE, RW, 20, 14),
    ]:
        if (
            visibility[a]
            >= TEACHER_VISIBILITY_THRESHOLD
            and
            visibility[b]
            >= TEACHER_VISIBILITY_THRESHOLD
        ):
            draw_tapered_limb(
                overlay,
                points[a],
                points[b],
                wa,
                wb,
                skin,
                outline,
            )

    # Sleeve caps
    for shoulder in [LS, RS]:
        if (
            visibility[shoulder]
            >= TEACHER_VISIBILITY_THRESHOLD
        ):
            cv2.circle(
                overlay,
                points[shoulder],
                15,
                outline,
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                overlay,
                points[shoulder],
                11,
                shirt,
                -1,
                cv2.LINE_AA,
            )

    # Hands

    for hand in [LW, RW]:
        if (
            visibility[hand]
            >= TEACHER_VISIBILITY_THRESHOLD
        ):
            cv2.circle(
                overlay,
                points[hand],
                13,
                outline,
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                overlay,
                points[hand],
                10,
                skin,
                -1,
                cv2.LINE_AA,
            )

            cv2.circle(
                overlay,
                (
                    points[hand][0] - 3,
                    points[hand][1] - 3,
                ),
                3,
                (
                    195,
                    220,
                    242,
                ),
                -1,
                cv2.LINE_AA,
            )

    # Shoes

    for ankle in [LA, RA]:
        if (
            visibility[ankle]
            >= TEACHER_VISIBILITY_THRESHOLD
        ):
            foot_center = (
                points[ankle][0] + 10,
                points[ankle][1] + 2,
            )

            cv2.ellipse(
                overlay,
                foot_center,
                (29, 12),
                0,
                0,
                360,
                outline,
                -1,
                cv2.LINE_AA,
            )

            cv2.ellipse(
                overlay,
                (
                    foot_center[0],
                    foot_center[1] - 2,
                ),
                (25, 9),
                0,
                0,
                360,
                shoes,
                -1,
                cv2.LINE_AA,
            )

    # Neck and head

    if (
        visibility[LS]
        >= TEACHER_VISIBILITY_THRESHOLD
        and
        visibility[RS]
        >= TEACHER_VISIBILITY_THRESHOLD
    ):
        shoulder_mid = midpoint(
            points[LS],
            points[RS]
        )

        nose = points[0]

        neck = (
            int(
                shoulder_mid[0]
                +
                (
                    nose[0]
                    - shoulder_mid[0]
                )
                * 0.35
            ),
            int(
                shoulder_mid[1]
                +
                (
                    nose[1]
                    - shoulder_mid[1]
                )
                * 0.35
            ),
        )

        cv2.line(
            overlay,
            (
                int(
                    shoulder_mid[0]
                ),
                int(
                    shoulder_mid[1]
                ),
            ),
            neck,
            outline,
            23,
            cv2.LINE_AA,
        )

        cv2.line(
            overlay,
            (
                int(
                    shoulder_mid[0]
                ),
                int(
                    shoulder_mid[1]
                ),
            ),
            neck,
            skin_shadow,
            16,
            cv2.LINE_AA,
        )

        shoulder_width = distance(
            points[LS],
            points[RS]
        )

        head_w = int(
            np.clip(
                shoulder_width * 0.42,
                39,
                74,
            )
        )

        head_h = int(
            head_w * 1.18
        )

        if (
            visibility[0]
            >= 0.20
        ):
            head_center = (
                points[0][0],
                points[0][1] - 2,
            )

            cv2.ellipse(
                overlay,
                head_center,
                (
                    head_w // 2 + 4,
                    head_h // 2 + 4,
                ),
                0,
                0,
                360,
                outline,
                -1,
                cv2.LINE_AA,
            )

            cv2.ellipse(
                overlay,
                head_center,
                (
                    head_w // 2,
                    head_h // 2,
                ),
                0,
                0,
                360,
                skin,
                -1,
                cv2.LINE_AA,
            )

            # Hair
            hair_center = (
                head_center[0],
                int(
                    head_center[1]
                    - head_h * 0.24
                ),
            )

            cv2.ellipse(
                overlay,
                hair_center,
                (
                    head_w // 2,
                    max(
                        8,
                        int(
                            head_h * 0.23
                        )
                    ),
                ),
                0,
                180,
                360,
                hair,
                -1,
                cv2.LINE_AA,
            )

            # Eyes
            eye_y = (
                head_center[1] - 2
            )

            eye_gap = max(
                5,
                head_w // 7
            )

            for eye_x in [
                head_center[0]
                - eye_gap,
                head_center[0]
                + eye_gap,
            ]:
                cv2.circle(
                    overlay,
                    (
                        eye_x,
                        eye_y,
                    ),
                    2,
                    outline,
                    -1,
                    cv2.LINE_AA,
                )

            # Friendly mouth
            cv2.ellipse(
                overlay,
                (
                    head_center[0],
                    head_center[1]
                    + max(
                        5,
                        head_h // 8
                    ),
                ),
                (
                    max(
                        5,
                        head_w // 9
                    ),
                    4,
                ),
                0,
                0,
                180,
                outline,
                2,
                cv2.LINE_AA,
            )

    # -----------------------------------------------------
    # Joint emphasis (subtle)
    # -----------------------------------------------------

    for joint in [
        LE,
        RE,
        LK,
        RK,
    ]:
        if (
            visibility[joint]
            >= TEACHER_VISIBILITY_THRESHOLD
        ):
            cv2.circle(
                overlay,
                points[joint],
                7,
                skin_shadow
                if joint in [LE, RE]
                else pants,
                -1,
                cv2.LINE_AA,
            )

    cv2.addWeighted(
        overlay,
        0.96,
        frame,
        0.04,
        0,
        frame,
    )
# NORMALIZED SKELETON FOR POSITION SCORE

def normalize_screen_pose(
    points
):
    left_hip = np.asarray(
        points[23],
        dtype=np.float32
    )

    right_hip = np.asarray(
        points[24],
        dtype=np.float32
    )

    left_shoulder = np.asarray(
        points[11],
        dtype=np.float32
    )

    right_shoulder = np.asarray(
        points[12],
        dtype=np.float32
    )

    hip_center = (
        left_hip
        + right_hip
    ) / 2.0

    shoulder_center = (
        left_shoulder
        + right_shoulder
    ) / 2.0

    torso = float(
        np.linalg.norm(
            shoulder_center
            - hip_center
        )
    )

    shoulder_width = float(
        np.linalg.norm(
            right_shoulder
            - left_shoulder
        )
    )

    scale = max(
        torso,
        shoulder_width,
        20.0,
    )

    normalized = {}

    for joint in MAJOR_JOINTS:
        normalized[joint] = (
            np.asarray(
                points[joint],
                dtype=np.float32
            )
            - hip_center
        ) / scale

    return normalized

# STATIC KEY-POSE SCORE

def score_static_pose(
    user_points,
    user_visibility,
    teacher_points,
    teacher_visibility,
):
    angle_scores = []
    angle_details = {}

    for (
        joint_name,
        indexes
    ) in ANGLE_JOINTS.items():

        if not all(
            user_visibility[index]
            >= POSE_VISIBILITY_THRESHOLD
            for index in indexes
        ):
            continue

        if not all(
            teacher_visibility[index]
            >= TEACHER_VISIBILITY_THRESHOLD
            for index in indexes
        ):
            continue

        user_angle = calculate_angle(
            user_points[indexes[0]],
            user_points[indexes[1]],
            user_points[indexes[2]],
        )

        teacher_angle = calculate_angle(
            teacher_points[indexes[0]],
            teacher_points[indexes[1]],
            teacher_points[indexes[2]],
        )

        error = abs(
            user_angle
            - teacher_angle
        )

        score = (
            100.0
            - error
            / ANGLE_ZERO_SCORE_ERROR
            * 100.0
        )

        score = float(
            np.clip(
                score,
                0.0,
                100.0
            )
        )

        angle_scores.append(
            score
        )

        angle_details[
            joint_name
        ] = {
            "user": user_angle,
            "teacher": teacher_angle,
            "error": error,
            "score": score,
        }

    if len(angle_scores) < 4:
        return None

    angle_score = float(
        np.mean(
            angle_scores
        )
    )

    user_norm = (
        normalize_screen_pose(
            user_points
        )
    )

    teacher_norm = (
        normalize_screen_pose(
            teacher_points
        )
    )

    position_errors = []

    for joint in MAJOR_JOINTS:
        if (
            user_visibility[joint]
            < POSE_VISIBILITY_THRESHOLD
            or
            teacher_visibility[joint]
            < TEACHER_VISIBILITY_THRESHOLD
        ):
            continue

        position_errors.append(
            float(
                np.linalg.norm(
                    user_norm[joint]
                    - teacher_norm[joint]
                )
            )
        )

    if len(position_errors) < 8:
        return None

    mean_position_error = float(
        np.mean(
            position_errors
        )
    )

    position_score = (
        100.0
        - mean_position_error
        / POSITION_ZERO_SCORE_ERROR
        * 100.0
    )

    position_score = float(
        np.clip(
            position_score,
            0.0,
            100.0
        )
    )

    overall = (
        ANGLE_WEIGHT
        * angle_score
        +
        POSITION_WEIGHT
        * position_score
    )

    worst_joint = None

    if angle_details:
        worst_joint = max(
            angle_details,
            key=lambda name: (
                angle_details[name][
                    "error"
                ]
            ),
        )

    return {
        "overall": float(
            np.clip(
                overall,
                0.0,
                100.0
            )
        ),
        "angle": angle_score,
        "position": (
            position_score
        ),
        "worst_joint": (
            worst_joint
        ),
        "angle_details": (
            angle_details
        ),
        "user_normalized": (
            user_norm
        ),
        "teacher_normalized": (
            teacher_norm
        ),
    }

# COACHING INSTRUCTION KEY

def get_pose_action_key(
    previous_pose,
    current_pose
):
    """
    Describe the largest visible movement from one key pose
    to the next. This is used for the initial voice hint.
    """
    if previous_pose is None:
        return "COPY_START_POSE"

    previous = (
        previous_pose["landmarks"]
    )

    current = (
        current_pose["landmarks"]
    )

    actions = []

    # Wrist vertical movement
    for side, wrist in [
        ("LEFT", 15),
        ("RIGHT", 16),
    ]:
        dy = (
            current[wrist]["y"]
            - previous[wrist]["y"]
        )

        if abs(dy) > 0.035:
            if dy < 0:
                key = (
                    f"RAISE_{side}_ARM"
                )
            else:
                key = (
                    f"LOWER_{side}_ARM"
                )

            actions.append(
                (
                    abs(dy) * 2.2,
                    key,
                )
            )

    # Knee bending
    raw_previous = [
        (
            lm["x"],
            lm["y"],
        )
        for lm in previous
    ]

    raw_current = [
        (
            lm["x"],
            lm["y"],
        )
        for lm in current
    ]

    for side, indexes in [
        (
            "LEFT",
            (23, 25, 27),
        ),
        (
            "RIGHT",
            (24, 26, 28),
        ),
    ]:
        previous_angle = (
            calculate_angle(
                raw_previous[
                    indexes[0]
                ],
                raw_previous[
                    indexes[1]
                ],
                raw_previous[
                    indexes[2]
                ],
            )
        )

        current_angle = (
            calculate_angle(
                raw_current[
                    indexes[0]
                ],
                raw_current[
                    indexes[1]
                ],
                raw_current[
                    indexes[2]
                ],
            )
        )

        difference = (
            current_angle
            - previous_angle
        )

        if abs(difference) > 12:
            if difference < 0:
                key = (
                    f"BEND_{side}_KNEE"
                )
            else:
                key = (
                    f"STRAIGHTEN_{side}_KNEE"
                )

            actions.append(
                (
                    abs(difference)
                    / 60.0,
                    key,
                )
            )

    # Foot side movement
    for side, ankle, hip in [
        (
            "LEFT",
            27,
            23,
        ),
        (
            "RIGHT",
            28,
            24,
        ),
    ]:
        previous_offset = (
            previous[ankle]["x"]
            - previous[hip]["x"]
        )

        current_offset = (
            current[ankle]["x"]
            - current[hip]["x"]
        )

        difference = (
            current_offset
            - previous_offset
        )

        if abs(difference) > 0.035:
            outward = (
                difference < 0
                if side == "LEFT"
                else difference > 0
            )

            key = (
                f"MOVE_{side}_FOOT_OUT"
                if outward
                else
                f"MOVE_{side}_FOOT_IN"
            )

            actions.append(
                (
                    abs(difference)
                    * 2.0,
                    key,
                )
            )

    if not actions:
        return "COPY_THIS_POSE"

    actions.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    return actions[0][1]


def correction_key_from_score(
    score_data
):
    worst_joint = (
        score_data[
            "worst_joint"
        ]
    )

    if worst_joint is None:
        return "COPY_THIS_POSE"

    detail = (
        score_data[
            "angle_details"
        ][worst_joint]
    )

    user_angle = (
        detail["user"]
    )

    teacher_angle = (
        detail["teacher"]
    )

    user_norm = (
        score_data[
            "user_normalized"
        ]
    )

    teacher_norm = (
        score_data[
            "teacher_normalized"
        ]
    )

    if worst_joint == "LEFT_ELBOW":
        return (
            "BEND_LEFT_ELBOW"
            if user_angle
            > teacher_angle
            else
            "OPEN_LEFT_ELBOW"
        )

    if worst_joint == "RIGHT_ELBOW":
        return (
            "BEND_RIGHT_ELBOW"
            if user_angle
            > teacher_angle
            else
            "OPEN_RIGHT_ELBOW"
        )

    if worst_joint == "LEFT_KNEE":
        return (
            "BEND_LEFT_KNEE"
            if user_angle
            > teacher_angle
            else
            "STRAIGHTEN_LEFT_KNEE"
        )

    if worst_joint == "RIGHT_KNEE":
        return (
            "BEND_RIGHT_KNEE"
            if user_angle
            > teacher_angle
            else
            "STRAIGHTEN_RIGHT_KNEE"
        )

    if worst_joint == "LEFT_SHOULDER":
        if (
            user_norm[15][1]
            >
            teacher_norm[15][1]
            + 0.10
        ):
            return "RAISE_LEFT_ARM"

        if (
            user_norm[15][1]
            <
            teacher_norm[15][1]
            - 0.10
        ):
            return "LOWER_LEFT_ARM"

        return "ADJUST_LEFT_SHOULDER"

    if worst_joint == "RIGHT_SHOULDER":
        if (
            user_norm[16][1]
            >
            teacher_norm[16][1]
            + 0.10
        ):
            return "RAISE_RIGHT_ARM"

        if (
            user_norm[16][1]
            <
            teacher_norm[16][1]
            - 0.10
        ):
            return "LOWER_RIGHT_ARM"

        return "ADJUST_RIGHT_SHOULDER"

    return "COPY_THIS_POSE"
# MULTILINGUAL VOICE TEXT

VOICE_TEXT = {
    "ENGLISH": {
        "COPY_START_POSE":
            "Copy this starting pose.",
        "COPY_THIS_POSE":
            "Copy the teacher pose.",
        "RAISE_LEFT_ARM":
            "Raise your left arm.",
        "RAISE_RIGHT_ARM":
            "Raise your right arm.",
        "LOWER_LEFT_ARM":
            "Lower your left arm.",
        "LOWER_RIGHT_ARM":
            "Lower your right arm.",
        "BEND_LEFT_ELBOW":
            "Bend your left elbow more.",
        "BEND_RIGHT_ELBOW":
            "Bend your right elbow more.",
        "OPEN_LEFT_ELBOW":
            "Open your left elbow more.",
        "OPEN_RIGHT_ELBOW":
            "Open your right elbow more.",
        "BEND_LEFT_KNEE":
            "Bend your left knee more.",
        "BEND_RIGHT_KNEE":
            "Bend your right knee more.",
        "STRAIGHTEN_LEFT_KNEE":
            "Straighten your left knee.",
        "STRAIGHTEN_RIGHT_KNEE":
            "Straighten your right knee.",
        "MOVE_LEFT_FOOT_OUT":
            "Move your left foot outward.",
        "MOVE_RIGHT_FOOT_OUT":
            "Move your right foot outward.",
        "MOVE_LEFT_FOOT_IN":
            "Bring your left foot inward.",
        "MOVE_RIGHT_FOOT_IN":
            "Bring your right foot inward.",
        "ADJUST_LEFT_SHOULDER":
            "Adjust your left shoulder.",
        "ADJUST_RIGHT_SHOULDER":
            "Adjust your right shoulder.",
        "SHOW_FULL_BODY":
            "Move back and show your full body.",
        "CORRECT":
            "Correct. Good.",
        "NEXT":
            "Next pose.",
        "COMPLETED":
            "Great. You completed the dance.",
    },

    "TAMIL": {
        "COPY_START_POSE":
            "இந்த ஆரம்ப போஸை பார்த்து செய்யுங்கள்.",
        "COPY_THIS_POSE":
            "ஆசிரியர் காட்டும் போஸை பார்த்து செய்யுங்கள்.",
        "RAISE_LEFT_ARM":
            "இடது கையை மேலே தூக்குங்கள்.",
        "RAISE_RIGHT_ARM":
            "வலது கையை மேலே தூக்குங்கள்.",
        "LOWER_LEFT_ARM":
            "இடது கையை கீழே இறக்குங்கள்.",
        "LOWER_RIGHT_ARM":
            "வலது கையை கீழே இறக்குங்கள்.",
        "BEND_LEFT_ELBOW":
            "இடது முழங்கையை இன்னும் மடக்குங்கள்.",
        "BEND_RIGHT_ELBOW":
            "வலது முழங்கையை இன்னும் மடக்குங்கள்.",
        "OPEN_LEFT_ELBOW":
            "இடது முழங்கையை இன்னும் திறக்குங்கள்.",
        "OPEN_RIGHT_ELBOW":
            "வலது முழங்கையை இன்னும் திறக்குங்கள்.",
        "BEND_LEFT_KNEE":
            "இடது முழங்காலை இன்னும் மடக்குங்கள்.",
        "BEND_RIGHT_KNEE":
            "வலது முழங்காலை இன்னும் மடக்குங்கள்.",
        "STRAIGHTEN_LEFT_KNEE":
            "இடது காலை நேராக வையுங்கள்.",
        "STRAIGHTEN_RIGHT_KNEE":
            "வலது காலை நேராக வையுங்கள்.",
        "MOVE_LEFT_FOOT_OUT":
            "இடது காலை வெளியே நகர்த்துங்கள்.",
        "MOVE_RIGHT_FOOT_OUT":
            "வலது காலை வெளியே நகர்த்துங்கள்.",
        "MOVE_LEFT_FOOT_IN":
            "இடது காலை உள்ளே கொண்டு வாருங்கள்.",
        "MOVE_RIGHT_FOOT_IN":
            "வலது காலை உள்ளே கொண்டு வாருங்கள்.",
        "ADJUST_LEFT_SHOULDER":
            "இடது தோளை சரி செய்யுங்கள்.",
        "ADJUST_RIGHT_SHOULDER":
            "வலது தோளை சரி செய்யுங்கள்.",
        "SHOW_FULL_BODY":
            "சிறிது பின்னால் சென்று முழு உடலையும் காட்டுங்கள்.",
        "CORRECT":
            "சரி. நன்றாக செய்தீர்கள்.",
        "NEXT":
            "அடுத்த போஸ்.",
        "COMPLETED":
            "அருமை. நீங்கள் நடனத்தை முடித்துவிட்டீர்கள்.",
    },

    "TANGLISH": {
        "COPY_START_POSE":
            "Indha starting pose ah copy pannunga.",
        "COPY_THIS_POSE":
            "Teacher pose ah same ah copy pannunga.",
        "RAISE_LEFT_ARM":
            "Left hand ah mela thookunga.",
        "RAISE_RIGHT_ARM":
            "Right hand ah mela thookunga.",
        "LOWER_LEFT_ARM":
            "Left hand ah konjam keela kondu vaanga.",
        "LOWER_RIGHT_ARM":
            "Right hand ah konjam keela kondu vaanga.",
        "BEND_LEFT_ELBOW":
            "Left elbow ah innum bend pannunga.",
        "BEND_RIGHT_ELBOW":
            "Right elbow ah innum bend pannunga.",
        "OPEN_LEFT_ELBOW":
            "Left elbow ah konjam open pannunga.",
        "OPEN_RIGHT_ELBOW":
            "Right elbow ah konjam open pannunga.",
        "BEND_LEFT_KNEE":
            "Left knee ah innum bend pannunga.",
        "BEND_RIGHT_KNEE":
            "Right knee ah innum bend pannunga.",
        "STRAIGHTEN_LEFT_KNEE":
            "Left leg ah konjam straight pannunga.",
        "STRAIGHTEN_RIGHT_KNEE":
            "Right leg ah konjam straight pannunga.",
        "MOVE_LEFT_FOOT_OUT":
            "Left foot ah veliya move pannunga.",
        "MOVE_RIGHT_FOOT_OUT":
            "Right foot ah veliya move pannunga.",
        "MOVE_LEFT_FOOT_IN":
            "Left foot ah ulla kondu vaanga.",
        "MOVE_RIGHT_FOOT_IN":
            "Right foot ah ulla kondu vaanga.",
        "ADJUST_LEFT_SHOULDER":
            "Left shoulder ah correct pannunga.",
        "ADJUST_RIGHT_SHOULDER":
            "Right shoulder ah correct pannunga.",
        "SHOW_FULL_BODY":
            "Konjam pinnadi poi full body camera la kaattunga.",
        "CORRECT":
            "Correct. Super.",
        "NEXT":
            "Next pose.",
        "COMPLETED":
            "Super. Dance complete aayiduchu.",
    },
}


def voice_phrase(
    language,
    key
):
    return (
        VOICE_TEXT
        .get(
            language,
            VOICE_TEXT["ENGLISH"]
        )
        .get(
            key,
            VOICE_TEXT["ENGLISH"]
            .get(
                key,
                "Copy the teacher pose."
            )
        )
    )


def display_phrase(
    language,
    key
):
    """
    OpenCV Hershey fonts do not support Tamil Unicode.
    For Tamil voice mode the HUD uses a readable
    English/Tanglish fallback while speech is Tamil.
    """
    if language == "TAMIL":
        return (
            VOICE_TEXT[
                "TANGLISH"
            ].get(
                key,
                key.replace(
                    "_",
                    " "
                )
            )
        )

    return (
        VOICE_TEXT[
            language
        ].get(
            key,
            key.replace(
                "_",
                " "
            )
        )
    )
# LOCAL VOICE COACH

class VoiceCoach:
    def __init__(self):
        self.engine = (
            shutil.which(
                "espeak-ng"
            )
            or
            shutil.which(
                "espeak"
            )
        )

        self.enabled = bool(
            VOICE_ENABLED_DEFAULT
        )

        self.queue = queue.Queue(
            maxsize=8
        )

        self.stop_event = (
            threading.Event()
        )

        self.worker = threading.Thread(
            target=self._worker,
            daemon=True,
        )

        self.worker.start()

        if self.engine:
            print(
                f"Voice engine: "
                f"{self.engine}"
            )
        else:
            print()
            print(
                "VOICE WARNING:"
            )
            print(
                "espeak-ng not found."
            )
            print(
                "Install with:"
            )
            print(
                "sudo apt install espeak-ng"
            )

    def _voice_code(
        self,
        language
    ):
        if language == "TAMIL":
            return "ta"

        return "en"

    def speak(
        self,
        text,
        language,
        replace_pending=False,
    ):
        if (
            not self.enabled
            or
            not self.engine
            or
            not text
        ):
            return

        if replace_pending:
            try:
                while True:
                    self.queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.queue.put_nowait(
                (
                    text,
                    language,
                )
            )

        except queue.Full:
            pass

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                (
                    text,
                    language,
                ) = self.queue.get(
                    timeout=0.2
                )

            except queue.Empty:
                continue

            if self.stop_event.is_set():
                break

            try:
                subprocess.run(
                    [
                        self.engine,
                        "-v",
                        self._voice_code(
                            language
                        ),
                        "-s",
                        str(
                            VOICE_SPEED
                        ),
                        text,
                    ],
                    stdout=(
                        subprocess.DEVNULL
                    ),
                    stderr=(
                        subprocess.DEVNULL
                    ),
                    check=False,
                )

            except Exception:
                pass

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def close(self):
        self.stop_event.set()


voice_coach = VoiceCoach()

# SOURCE KEY-FRAME PREVIEW

class SourcePreview:
    def __init__(
        self,
        path
    ):
        self.cap = cv2.VideoCapture(
            path
        )

        self.last_time_ms = None
        self.cached_frame = None

    def get_frame(
        self,
        time_ms
    ):
        time_ms = int(
            max(
                0,
                time_ms
            )
        )

        if (
            self.cached_frame
            is not None
            and
            self.last_time_ms
            == time_ms
        ):
            return self.cached_frame

        self.cap.set(
            cv2.CAP_PROP_POS_MSEC,
            float(time_ms),
        )

        ok, frame = self.cap.read()

        if not ok:
            return self.cached_frame

        self.last_time_ms = (
            time_ms
        )

        self.cached_frame = frame

        return frame

    def close(self):
        self.cap.release()


source_preview = SourcePreview(
    DANCE_VIDEO_PATH
)


def resize_keep_aspect(
    image,
    max_width,
    max_height
):
    height, width = (
        image.shape[:2]
    )

    scale = min(
        max_width / max(
            1,
            width
        ),
        max_height / max(
            1,
            height
        ),
    )

    new_width = max(
        1,
        int(
            width * scale
        )
    )

    new_height = max(
        1,
        int(
            height * scale
        )
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else
            cv2.INTER_LINEAR
        ),
    )


def draw_source_preview(
    frame,
    source_image
):
    if source_image is None:
        return

    if SOURCE_ORIENTATION == "PORTRAIT":
        preview = resize_keep_aspect(
            source_image,
            PORTRAIT_PREVIEW_MAX_W,
            PORTRAIT_PREVIEW_MAX_H,
        )
    else:
        preview = resize_keep_aspect(
            source_image,
            LANDSCAPE_PREVIEW_MAX_W,
            LANDSCAPE_PREVIEW_MAX_H,
        )

    ph, pw = preview.shape[:2]

    x1 = int(
        PREVIEW_CENTER_X
        - pw / 2
    )

    y1 = PREVIEW_TOP

    x2 = x1 + pw
    y2 = y1 + ph

    # Dark panel behind preview
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            x1 - 10,
            y1 - 10,
        ),
        (
            x2 + 10,
            y2 + 34,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.46,
        frame,
        0.54,
        0,
        frame,
    )

    frame[
        y1:y2,
        x1:x2
    ] = preview

    cv2.rectangle(
        frame,
        (
            x1 - 2,
            y1 - 2,
        ),
        (
            x2 + 2,
            y2 + 2,
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            f"SOURCE - "
            f"{SOURCE_ORIENTATION}"
        ),
        (
            x1,
            y2 + 24
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

# HUD HELPERS

def draw_transparent_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.38,
):
    overlay = frame.copy()

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame.shape[1] - 1, int(x2))
    y2 = min(frame.shape[0] - 1, int(y2))

    glass = (32, 43, 58)
    highlight = (150, 202, 224)
    radius = min(18, max(4, (x2 - x1) // 12))

    cv2.rectangle(
        overlay,
        (x1 + radius, y1),
        (x2 - radius, y2),
        glass,
        -1,
    )

    cv2.rectangle(
        overlay,
        (x1, y1 + radius),
        (x2, y2 - radius),
        glass,
        -1,
    )

    for center in [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ]:
        cv2.circle(
            overlay,
            center,
            radius,
            glass,
            -1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        highlight,
        1,
        cv2.LINE_AA,
    )


def draw_progress_dots(
    frame,
    current_index,
    total
):
    if total <= 0:
        return

    max_width = 520

    start_x = 30
    y = CAMERA_HEIGHT - 55

    spacing = min(
        30,
        max(
            13,
            int(
                max_width
                / max(
                    1,
                    total
                )
            )
        )
    )

    radius = max(
        4,
        min(
            7,
            spacing // 3
        )
    )

    for index in range(total):
        x = (
            start_x
            + index * spacing
        )

        if index < current_index:
            fill = (
                0,
                220,
                0
            )

        elif index == current_index:
            fill = (
                0,
                255,
                255
            )

        else:
            fill = (
                120,
                120,
                120
            )

        cv2.circle(
            frame,
            (
                x,
                y
            ),
            radius,
            fill,
            -1,
            cv2.LINE_AA,
        )

# OPEN CAMERA

def open_camera(candidates=None):
    candidates = candidates or resolve_camera_candidates()

    for candidate in candidates:
        source = int(candidate) if candidate.isdigit() else candidate
        print(f"Opening camera: {candidate}")
        camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if camera.isOpened():
            return camera, candidate
        camera.release()

    return None, None

print()
cap, active_camera = (
    (None, None)
    if WEB_MODE
    else open_camera()
)

camera_available = cap is not None
practice_active = not WEB_MODE

if not camera_available:
    print()
    print("ERROR: No V4L2 camera could be opened.")
    print("Set MOTION_LAB_CAMERA to a device path or index if needed.")
    if not WEB_MODE:
        sys.exit(1)
else:
    print(f"Camera ready: {active_camera}")

if camera_available:
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            "M",
            "J",
            "P",
            "G",
        )
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

# LIVE POSE DETECTOR

live_pose_detector = (
    create_pose_landmarker()
    if camera_available
    else None
)

live_pose_start_time = (
    time.monotonic()
)

# WINDOW

fullscreen_enabled = bool(
    FULLSCREEN_DEFAULT
)

if camera_available:
    if not WEB_MODE:
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )
window_ready = camera_available and not WEB_MODE


def apply_window_mode():
    if fullscreen_enabled:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
    else:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
        )


if camera_available:
    apply_window_mode()
# APP STATE

avatar_gender = "female"

language_index = 0
current_language = (
    LANGUAGES[
        language_index
    ]
)

source_preview_enabled = bool(
    SOURCE_PREVIEW_ENABLED_DEFAULT
)

current_pose_index = 0

paused = False
completed = False

match_hold_started = None

success_active = False
success_started = None

last_correction_voice_at = 0.0
last_correction_key = None

last_full_body_voice_at = 0.0

score_data = None

display_score = 0.0
latest_camera_jpeg = None

current_instruction_key = (
    "COPY_START_POSE"
)

pose_instruction_pending = True

# POSE CONTROL

def set_pose(
    index,
    speak_instruction=True
):
    global current_pose_index
    global match_hold_started
    global success_active
    global success_started
    global score_data
    global display_score
    global current_instruction_key
    global pose_instruction_pending
    global last_correction_key
    global last_correction_voice_at
    global completed

    current_pose_index = int(
        np.clip(
            index,
            0,
            TOTAL_KEY_POSES - 1,
        )
    )

    match_hold_started = None

    success_active = False
    success_started = None

    score_data = None
    display_score = 0.0

    last_correction_key = None
    last_correction_voice_at = 0.0

    completed = False

    previous_pose = (
        KEY_POSES[
            current_pose_index - 1
        ]
        if current_pose_index > 0
        else None
    )

    current_pose = (
        KEY_POSES[
            current_pose_index
        ]
    )

    current_instruction_key = (
        get_pose_action_key(
            previous_pose,
            current_pose
        )
    )

    pose_instruction_pending = bool(
        speak_instruction
    )


def speak_current_pose():
    global pose_instruction_pending

    if not pose_instruction_pending:
        return

    pose_number = (
        current_pose_index + 1
    )

    phrase = voice_phrase(
        current_language,
        current_instruction_key,
    )

    if current_language == "TAMIL":
        text = (
            f"போஸ் {pose_number}. "
            f"{phrase}"
        )
    else:
        text = (
            f"Pose {pose_number}. "
            f"{phrase}"
        )

    voice_coach.speak(
        text,
        current_language,
        replace_pending=True,
    )

    pose_instruction_pending = False


def next_pose():
    global completed
    global success_active
    global success_started

    if (
        current_pose_index
        + 1
        >= TOTAL_KEY_POSES
    ):
        completed = True
        success_active = False
        success_started = None

        voice_coach.speak(
            voice_phrase(
                current_language,
                "COMPLETED",
            ),
            current_language,
            replace_pending=True,
        )

        return

    voice_coach.speak(
        voice_phrase(
            current_language,
            "NEXT",
        ),
        current_language,
        replace_pending=True,
    )

    set_pose(
        current_pose_index + 1,
        speak_instruction=True,
    )


def previous_pose():
    set_pose(
        max(
            0,
            current_pose_index - 1
        ),
        speak_instruction=True,
    )


def restart_dance():
    set_pose(
        0,
        speak_instruction=True,
    )


set_pose(
    0,
    speak_instruction=True,
)

# START INFO

print()
print(
    "========================================"
)
print(
    "AR DANCE TEACHER V11"
)
print(
    "========================================"
)
print(
    f"Key poses: "
    f"{TOTAL_KEY_POSES}"
)
print(
    f"Pass threshold: "
    f"{PASS_THRESHOLD:.0f}%"
)
print(
    f"Voice language: "
    f"{current_language}"
)
print()
print(
    "L     = English / Tamil / Tanglish"
)
print(
    "A     = Voice ON / OFF"
)
print(
    "N/P   = Next / Previous pose"
)
print(
    "R     = Restart dance"
)
print(
    "SPACE = Pause / Resume"
)
print(
    "V     = Source preview ON / OFF"
)
print(
    "M/F   = Male / Female trainer"
)
print(
    "+/-   = Trainer bigger / smaller"
)
print(
    "X     = Fullscreen ON / OFF"
)
print(
    "ESC   = Exit"
)
print(
    "========================================"
)
print()

def web_session_snapshot():
    available_cameras = detect_camera_options()
    if active_camera and active_camera not in available_cameras:
        available_cameras = [active_camera] + available_cameras

    return {
        "ready": True,
        "camera_available": camera_available,
        "camera_source": active_camera,
        "camera_options": available_cameras,
        "practice_active": practice_active,
        "source_video": os.path.basename(DANCE_VIDEO_PATH),
        "duration_ms": MOTION_DURATION_MS,
        "detected_frames": len(MOTION_FRAMES),
        "current_step": current_pose_index + 1,
        "total_steps": TOTAL_KEY_POSES,
        "paused": paused,
        "completed": completed,
        "voice_enabled": voice_coach.enabled,
        "language": current_language,
        "score": round(display_score, 1),
        "body_detected": body_detected if "body_detected" in globals() else False,
    }


def camera_stream():
    """Yield backend-owned camera frames as an MJPEG browser stream."""
    while True:
        if latest_camera_jpeg is None:
            time.sleep(0.1)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + latest_camera_jpeg
            + b"\r\n"
        )


def handle_web_command(command):
    global cap
    global active_camera
    global camera_available
    global live_pose_detector
    global practice_active
    global paused

    if command == "open_camera" or command.startswith("open_camera:"):
        selected_camera = command.partition(":")[2] or None
        if camera_available and selected_camera and active_camera != selected_camera:
            cap.release()
            cap = None
            camera_available = False

        if not camera_available:
            camera_candidates = resolve_camera_candidates()
            if selected_camera:
                camera_candidates = [selected_camera]
            cap, active_camera = open_camera(camera_candidates)
            camera_available = cap is not None
            if camera_available:
                live_pose_detector = create_pose_landmarker()
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return web_session_snapshot()
    if command == "start_practice":
        if not camera_available:
            raise ValueError("No camera is available")
        practice_active = True
        paused = False
        return web_session_snapshot()
    if command == "stop_practice":
        practice_active = False
        paused = True
        return web_session_snapshot()

    if command == "previous":
        previous_pose()
    elif command == "next":
        next_pose()
    elif command == "restart":
        restart_dance()
    elif command == "pause":
        paused = not paused
    elif command == "voice":
        voice_coach.toggle()
    elif command == "language":
        global language_index, current_language
        language_index = (language_index + 1) % len(LANGUAGES)
        current_language = LANGUAGES[language_index]
    else:
        raise ValueError("Unknown control")

    return web_session_snapshot()


def restart_application_with_source(source_path):
    time.sleep(0.25)
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(source_path),
        ],
    )


def handle_source_upload(upload, youtube_url):
    SOURCE_UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    if upload is not None:
        original_name = Path(upload.filename or "").name
        extension = Path(original_name).suffix.lower()
        if extension not in ALLOWED_SOURCE_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_SOURCE_EXTENSIONS))
            raise ValueError(f"Unsupported video type. Use: {allowed}")
        source_path = SOURCE_UPLOAD_FOLDER / f"reference_{int(time.time())}{extension}"
        upload.save(source_path)
    else:
        parsed_url = urlparse(youtube_url)
        host = parsed_url.netloc.lower().split(":", 1)[0]
        if host not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
            raise ValueError("Enter a valid YouTube URL")

        source_path = SOURCE_UPLOAD_FOLDER / f"youtube_reference_{int(time.time())}.mp4"
        downloader = shutil.which("yt-dlp")
        command = (
            [downloader]
            if downloader
            else [sys.executable, "-m", "yt_dlp"]
        )
        command.extend(
            [
                "--no-playlist",
                "--merge-output-format",
                "mp4",
                "-o",
                str(source_path),
                youtube_url,
            ]
        )
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise ValueError("YouTube support needs yt-dlp. Install it in the project venv.") from error
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or "").strip().splitlines()[-1:] or ["Download failed"]
            raise ValueError(detail[0]) from error

    if not source_path.is_file() or source_path.stat().st_size == 0:
        raise ValueError("The reference video could not be saved")

    threading.Thread(
        target=restart_application_with_source,
        args=(source_path.resolve(),),
        daemon=True,
    ).start()
    return {
        "ok": True,
        "restarting": True,
        "source_video": source_path.name,
        "message": "Reference ready. Restarting the dance engine.",
    }


start_web_bridge(
    web_session_snapshot,
    handle_web_command,
    frontend_dir=PROJECT_ROOT / "frontend" / "dist",
    source_video=DANCE_VIDEO_PATH,
    camera_stream=camera_stream,
    source_upload=handle_source_upload,
)

print(
    "Flask web app ready: http://127.0.0.1:5000"
)

if WEB_MODE and not camera_available:
    print("Web-only mode active: waiting for a camera connection.")

# MAIN LOOP


try:
    while True:
        now = time.monotonic()

        if not camera_available or cap is None or (
            WEB_MODE and not practice_active
        ):
            time.sleep(0.1)
            continue

        if not window_ready and not WEB_MODE:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            apply_window_mode()
            window_ready = True

        # Camera
    

        ok, original_frame = cap.read()

        if not ok:
            print(
                "Failed to read "
                "Logitech BRIO frame."
            )
            break

        original_frame = cv2.resize(
            original_frame,
            (
                CAMERA_WIDTH,
                CAMERA_HEIGHT
            ),
        )

        
        # Live MediaPipe
        

        rgb = cv2.cvtColor(
            original_frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb,
        )

        timestamp_ms = int(
            (
                time.monotonic()
                - live_pose_start_time
            )
            * 1000
        )

        try:
            pose_result = (
                live_pose_detector
                .detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

        except Exception as error:
            print(
                "Live MediaPipe error:",
                error
            )
            pose_result = None

        # Selfie display
        frame = cv2.flip(
            original_frame,
            1
        )

        body_detected = False
        user_points = None
        user_visibility = None

        if (
            pose_result is not None
            and
            pose_result.pose_landmarks
        ):
            body_detected = True

            (
                user_points,
                user_visibility
            ) = get_mirrored_user_pose(
                pose_result
                .pose_landmarks[0],
                CAMERA_WIDTH,
                CAMERA_HEIGHT,
            )

            draw_user_pose(
                frame,
                user_points,
                user_visibility,
            )

    
        # Current key-pose teacher
        

        current_pose = (
            KEY_POSES[
                current_pose_index
            ]
        )

        (
            teacher_points,
            teacher_visibility,
            teacher_depths
        ) = transform_teacher_landmarks(
            current_pose[
                "landmarks"
            ]
        )

        draw_human_trainer(
            frame,
            teacher_points,
            teacher_visibility,
            teacher_depths,
            avatar_gender,
        )
        # Source preview - exact current key pose
        

        if source_preview_enabled:
            source_image = (
                source_preview.get_frame(
                    current_pose[
                        "time_ms"
                    ]
                )
            )

            draw_source_preview(
                frame,
                source_image,
            )

        
        # Speak pose instruction once
    

        if (
            not paused
            and
            not completed
            and
            not success_active
        ):
            speak_current_pose()

        
        # Pose-lock scoring
        

        correction_key = None

        if (
            not paused
            and
            not completed
            and
            not success_active
            and
            body_detected
            and
            user_points is not None
            and
            user_visibility is not None
        ):
            score_data = (
                score_static_pose(
                    user_points,
                    user_visibility,
                    teacher_points,
                    teacher_visibility,
                )
            )

            if score_data is not None:
                raw_score = (
                    score_data[
                        "overall"
                    ]
                )

                display_score = (
                    display_score
                    * 0.72
                    +
                    raw_score
                    * 0.28
                )

                if (
                    raw_score
                    >= PASS_THRESHOLD
                ):
                    if (
                        match_hold_started
                        is None
                    ):
                        match_hold_started = (
                            now
                        )

                    stable_time = (
                        now
                        - match_hold_started
                    )

                    if (
                        stable_time
                        >= MATCH_HOLD_SECONDS
                    ):
                        success_active = True
                        success_started = now
                        match_hold_started = None

                        voice_coach.speak(
                            voice_phrase(
                                current_language,
                                "CORRECT",
                            ),
                            current_language,
                            replace_pending=True,
                        )

                else:
                    match_hold_started = None

                    correction_key = (
                        correction_key_from_score(
                            score_data
                        )
                    )

                    if (
                        now
                        - last_correction_voice_at
                        >=
                        CORRECTION_VOICE_COOLDOWN
                    ):
                        if (
                            correction_key
                            != last_correction_key
                            or
                            now
                            - last_correction_voice_at
                            >=
                            CORRECTION_VOICE_COOLDOWN
                            * 1.6
                        ):
                            voice_coach.speak(
                                voice_phrase(
                                    current_language,
                                    correction_key,
                                ),
                                current_language,
                                replace_pending=False,
                            )

                            last_correction_key = (
                                correction_key
                            )

                            last_correction_voice_at = (
                                now
                            )

            else:
                match_hold_started = None

        else:
            if (
                not body_detected
                and
                not paused
                and
                not completed
                and
                not success_active
            ):
                if (
                    now
                    - last_full_body_voice_at
                    >= 5.0
                ):
                    voice_coach.speak(
                        voice_phrase(
                            current_language,
                            "SHOW_FULL_BODY",
                        ),
                        current_language,
                        replace_pending=False,
                    )

                    last_full_body_voice_at = (
                        now
                    )

        
        # Correct -> wait 1 second -> next pose
    

        if (
            success_active
            and
            success_started is not None
        ):
            if (
                now
                - success_started
                >=
                SUCCESS_DISPLAY_SECONDS
            ):
                next_pose()

        
        # HUD
        

        draw_transparent_panel(
            frame,
            16,
            16,
            430,
            330,
            alpha=0.40,
        )

        draw_transparent_panel(
            frame,
            CAMERA_WIDTH - 355,
            16,
            CAMERA_WIDTH - 16,
            180,
            alpha=0.36,
        )

        cv2.putText(
            frame,
            "AR DANCE TEACHER",
            (
                34,
                54
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.92,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"KEY POSE "
                f"{current_pose_index + 1}"
                f"/{TOTAL_KEY_POSES}"
            ),
            (
                34,
                94
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        body_status = (
            "BODY DETECTED"
            if body_detected
            else
            "SHOW FULL BODY"
        )

        body_color = (
            (0, 255, 0)
            if body_detected
            else
            (0, 190, 255)
        )

        cv2.putText(
            frame,
            body_status,
            (
                34,
                132
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            body_color,
            2,
            cv2.LINE_AA,
        )

        if score_data is not None:
            score_text = (
                f"MATCH: "
                f"{int(display_score)}%"
            )

            detail_text = (
                f"ANGLE "
                f"{int(score_data['angle'])}%"
                f"   POSITION "
                f"{int(score_data['position'])}%"
            )
        else:
            score_text = "MATCH: --"
            detail_text = (
                "ANGLE --   POSITION --"
            )

        cv2.putText(
            frame,
            score_text,
            (
                34,
                177
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            detail_text,
            (
                34,
                211
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if success_active:
            instruction_display = (
                "CORRECT - NEXT POSE..."
            )

            instruction_color = (
                0,
                255,
                0
            )

        elif completed:
            instruction_display = (
                "DANCE COMPLETE"
            )

            instruction_color = (
                0,
                255,
                0
            )

        elif paused:
            instruction_display = (
                "PAUSED"
            )

            instruction_color = (
                0,
                255,
                255
            )

        elif not body_detected:
            instruction_display = (
                display_phrase(
                    current_language,
                    "SHOW_FULL_BODY",
                )
            )

            instruction_color = (
                0,
                255,
                255
            )

        elif correction_key:
            instruction_display = (
                display_phrase(
                    current_language,
                    correction_key,
                )
            )

            instruction_color = (
                0,
                255,
                255
            )

        else:
            instruction_display = (
                display_phrase(
                    current_language,
                    current_instruction_key,
                )
            )

            instruction_color = (
                255,
                255,
                255
            )

        cv2.putText(
            frame,
            instruction_display,
            (
                34,
                255
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            instruction_color,
            2,
            cv2.LINE_AA,
        )

        # Match hold progress
        if (
            match_hold_started
            is not None
            and
            not success_active
        ):
            hold_progress = float(
                np.clip(
                    (
                        now
                        - match_hold_started
                    )
                    / MATCH_HOLD_SECONDS,
                    0.0,
                    1.0,
                )
            )

            cv2.putText(
                frame,
                "HOLD THE POSE...",
                (
                    34,
                    294
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.rectangle(
                frame,
                (
                    34,
                    307
                ),
                (
                    330,
                    316
                ),
                (
                    100,
                    100,
                    100
                ),
                1,
            )

            cv2.rectangle(
                frame,
                (
                    34,
                    307
                ),
                (
                    34
                    + int(
                        296
                        * hold_progress
                    ),
                    316
                ),
                (
                    0,
                    255,
                    0
                ),
                -1,
            )

        # Right status
        cv2.putText(
            frame,
            (
                f"Trainer: "
                f"{avatar_gender.upper()}"
            ),
            (
                CAMERA_WIDTH - 330,
                54
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"Voice: "
                f"{current_language}"
            ),
            (
                CAMERA_WIDTH - 330,
                88
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                255,
                0
                if voice_coach.enabled
                else 180
            ),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"Threshold: "
                f"{int(PASS_THRESHOLD)}%"
            ),
            (
                CAMERA_WIDTH - 330,
                122
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            (
                f"Trainer size: "
                f"{int(teacher_size)}"
            ),
            (
                CAMERA_WIDTH - 330,
                154
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Large success banner
        if success_active:
            text = "CORRECT"

            size, _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.4,
                6,
            )

            x = int(
                CAMERA_WIDTH / 2
                - size[0] / 2
            )

            y = int(
                CAMERA_HEIGHT / 2
            )

            cv2.putText(
                frame,
                text,
                (
                    x,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.4,
                (0, 255, 0),
                6,
                cv2.LINE_AA,
            )

        if completed:
            text = "DANCE COMPLETE"

            size, _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.7,
                5,
            )

            x = int(
                CAMERA_WIDTH / 2
                - size[0] / 2
            )

            y = int(
                CAMERA_HEIGHT / 2
            )

            cv2.putText(
                frame,
                text,
                (
                    x,
                    y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.7,
                (0, 255, 0),
                5,
                cv2.LINE_AA,
            )

        # Bottom panel
        draw_transparent_panel(
            frame,
            16,
            CAMERA_HEIGHT - 88,
            CAMERA_WIDTH - 16,
            CAMERA_HEIGHT - 10,
            alpha=0.34,
        )

        draw_progress_dots(
            frame,
            current_pose_index,
            TOTAL_KEY_POSES,
        )

        cv2.putText(
            frame,
            (
                "L Language  A Voice  N/P Pose  "
                "R Restart  SPACE Pause  V Preview  "
                "M/F Trainer  +/- Size  X Fullscreen  ESC Exit"
            ),
            (
                30,
                CAMERA_HEIGHT - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    
        # Display

        encoded_frame = cv2.imencode(".jpg", frame)[1]
        if encoded_frame is not None:
            latest_camera_jpeg = encoded_frame.tobytes()


        key = -1
        if not WEB_MODE:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

    
        # Keyboard
    

        if key == 27:
            break

        elif key == ord("m"):
            avatar_gender = "male"

        elif key == ord("f"):
            avatar_gender = "female"

        elif key == ord("+") or key == ord("="):
            teacher_size = min(
                TEACHER_SIZE_MAX,
                teacher_size
                + TEACHER_SIZE_STEP,
            )

        elif key == ord("-") or key == ord("_"):
            teacher_size = max(
                TEACHER_SIZE_MIN,
                teacher_size
                - TEACHER_SIZE_STEP,
            )

        elif key == ord("x"):
            fullscreen_enabled = (
                not fullscreen_enabled
            )

            apply_window_mode()

        elif key == ord("v"):
            source_preview_enabled = (
                not source_preview_enabled
            )

        elif key == ord("a"):
            enabled = (
                voice_coach.toggle()
            )

            print(
                "Voice ON"
                if enabled
                else
                "Voice OFF"
            )

            if enabled:
                pose_instruction_pending = (
                    True
                )

        elif key == ord("l"):
            language_index = (
                language_index + 1
            ) % len(LANGUAGES)

            current_language = (
                LANGUAGES[
                    language_index
                ]
            )

            print(
                "Voice language:",
                current_language
            )

            pose_instruction_pending = (
                True
            )

        elif key == ord("n"):
            if (
                current_pose_index
                + 1
                < TOTAL_KEY_POSES
            ):
                set_pose(
                    current_pose_index + 1,
                    speak_instruction=True,
                )

        elif key == ord("p"):
            previous_pose()

        elif key == ord("r"):
            restart_dance()

        elif key == 32:
            paused = not paused

            if not paused:
                pose_instruction_pending = (
                    True
                )


finally:
    print()
    print(
        "Closing AR Dance Teacher V11..."
    )

    voice_coach.close()

    source_preview.close()

    if live_pose_detector is not None:
        live_pose_detector.close()

    if cap is not None:
        cap.release()

    if not WEB_MODE:
        cv2.destroyAllWindows()

    print("Done.")

