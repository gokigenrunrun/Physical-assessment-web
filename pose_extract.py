# pose_extract.py
import csv
from pathlib import Path
from typing import Callable, Generator, Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import time

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


LANDMARK_HEADER = ["frame", "landmark_index", "x", "y", "z", "visibility"]
DEFAULT_MODEL_CANDIDATES = (
    "pose_landmarker_heavy.task",
    "pose_landmarker_full.task",
    "pose_landmarker_lite.task",
)


def _resolve_model_path(model_asset_path: Optional[str] = None) -> str:
    """Looks for a bundled PoseLandmarker model when no path is provided."""
    if model_asset_path:
        return model_asset_path

    mp_root = Path(mp.__file__).resolve().parent
    for candidate in DEFAULT_MODEL_CANDIDATES:
        bundled = mp_root / "modules" / "pose_landmark" / candidate
        if bundled.exists():
            return str(bundled)

    raise FileNotFoundError(
        "PoseLandmarker model (.task) not found. Specify model_asset_path explicitly."
    )


def _create_pose_landmarker(model_asset_path: Optional[str] = None) -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=_resolve_model_path(model_asset_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_pose_landmarks(
    frame_bgr: np.ndarray,
    pose_landmarks: Optional[list],
) -> np.ndarray:
    """
    Tasks API の pose landmarks を OpenCV で描画する
    """
    if pose_landmarks is None:
        return frame_bgr

    annotated = frame_bgr.copy()
    h, w, _ = annotated.shape

    # 各関節を描画
    for lm in pose_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

    # （任意）骨格の接続線も描きたい場合はここで定義できる
    # 例：肩〜肘〜手首など

    return annotated


def pose_capture_generator(
    cap: cv2.VideoCapture,
    resize_scale: float = 1.0,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    model_asset_path: Optional[str] = None,
    fallback_fps: float = 30.0,
) -> Generator[Tuple[int, np.ndarray, np.ndarray, Optional[list]], None, None]:
    """
    MediaPipe Tasks Pose推論を行いながらランドマークを逐次返すジェネレーター。
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(fallback_fps) if fallback_fps > 0 else 30.0
    frame_interval_ms = 1000.0 / fps

    with _create_pose_landmarker(model_asset_path) as landmarker:
        frame_idx = 0
        processed = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_stride > 1 and frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            frame_for_processing = frame_bgr
            if resize_scale != 1.0:
                w = int(frame_bgr.shape[1] * resize_scale)
                h = int(frame_bgr.shape[0] * resize_scale)
                frame_for_processing = cv2.resize(frame_bgr, (w, h))

            frame_rgb = cv2.cvtColor(frame_for_processing, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * frame_interval_ms)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

            yield frame_idx, frame_bgr, frame_for_processing, pose_landmarks

            processed += 1
            frame_idx += 1

            if max_frames is not None and processed >= max_frames:
                break


def write_landmarks_to_csv(rows: Iterable[Tuple[int, int, float, float, float, float]], out_csv_path: str) -> None:
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LANDMARK_HEADER)
        writer.writerows(rows)


def video_to_pose_csv(
    video_path: str,
    out_csv_path: str,
    resize_scale: float = 1.0,
    frame_stride: int = 1,
    model_asset_path: Optional[str] = None,
) -> str:
    """
    動画 -> 単一人物の33ランドマークをフレームごとにCSV保存
    出力カラム: frame, landmark_index, x, y, z, visibility
    """
    video_path = str(video_path)
    out_csv_path = str(out_csv_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LANDMARK_HEADER)
        try:
            for frame_idx, _, _, landmarks in pose_capture_generator(
                cap=cap,
                resize_scale=resize_scale,
                frame_stride=frame_stride,
                model_asset_path=model_asset_path,
            ):
                if landmarks:
                    for idx, lm in enumerate(landmarks):
                        writer.writerow((frame_idx, idx, lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)))
        finally:
            cap.release()

    return out_csv_path


def capture_pose_from_camera(
    camera_index: int = 0,
    warmup_camera: Optional[cv2.VideoCapture] = None,
    resize_scale: float = 1.0,
    frame_stride: int = 1,
    capture_seconds: Optional[int] = 10,
    target_fps: int = 15,
    max_frames: Optional[int] = None,
    out_csv_path: Optional[str] = None,
    frame_callback: Optional[Callable[[int, np.ndarray], None]] = None,
    return_start_timestamp: bool = False,
    model_asset_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Webカメラから一定時間ポーズ推論を行い、ランドマークをDataFrameで返す。
    warmup_cameraが指定された場合は、そのVideoCaptureを再利用する。
    """
    reuse_capture = warmup_camera is not None
    cap = warmup_camera if reuse_capture else cv2.VideoCapture(camera_index)

    if cap is None or not cap.isOpened():
        if reuse_capture and cap is not None:
            cap.release()
        cap = cv2.VideoCapture(camera_index)
        reuse_capture = False

    if not cap or not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: index={camera_index}")

    if not reuse_capture:
        for _ in range(10):
            ok, _ = cap.read()
            if not ok:
                break

    if max_frames is None and capture_seconds is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = float(target_fps)
        max_frames = int(fps * capture_seconds)

    rows = []
    start_timestamp: Optional[float] = None
    try:
        for frame_idx, frame_original_bgr, frame_processed_bgr, landmarks in pose_capture_generator(
            cap=cap,
            resize_scale=resize_scale,
            frame_stride=frame_stride,
            max_frames=max_frames,
            fallback_fps=target_fps,
            model_asset_path=model_asset_path,
        ):
            if frame_callback is not None:
                frame_rgb = cv2.cvtColor(frame_original_bgr, cv2.COLOR_BGR2RGB)
                frame_callback(frame_idx, frame_rgb)
            if landmarks:
                if start_timestamp is None:
                    start_timestamp = time.time()
                for idx, lm in enumerate(landmarks):
                    rows.append(
                        (
                            frame_idx,
                            idx,
                            lm.x,
                            lm.y,
                            lm.z,
                            getattr(lm, "visibility", 0.0),
                        )
                    )
    finally:
        if cap is not None:
            cap.release()

    df = pd.DataFrame(rows, columns=LANDMARK_HEADER)
    if out_csv_path:
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False)
    if return_start_timestamp:
        return df, start_timestamp
    return df
