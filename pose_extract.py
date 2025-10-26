# pose_extract.py
import csv
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.framework.formats import landmark_pb2

# 軽量のSolutions API
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LANDMARK_HEADER = ["frame", "landmark_index", "x", "y", "z", "visibility"]


def draw_pose_landmarks(
    frame_bgr: np.ndarray,
    pose_landmarks: Optional[landmark_pb2.NormalizedLandmarkList],
) -> np.ndarray:
    """
    MediaPipeのランドマークをフレームに重畳して返す。
    """
    if pose_landmarks is None:
        return frame_bgr

    annotated = frame_bgr.copy()
    mp_drawing.draw_landmarks(
        annotated,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
    )
    return annotated


def pose_capture_generator(
    cap: cv2.VideoCapture,
    resize_scale: float = 1.0,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray, Optional[landmark_pb2.NormalizedLandmarkList]], None, None]:
    """
    MediaPipe Pose推論を行いながらランドマークを逐次返すジェネレーター。
    """
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
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
            result = pose.process(frame_rgb)

            yield frame_idx, frame_for_processing, result.pose_landmarks

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
            for frame_idx, _, landmarks in pose_capture_generator(
                cap=cap,
                resize_scale=resize_scale,
                frame_stride=frame_stride,
            ):
                if landmarks:
                    for idx, lm in enumerate(landmarks.landmark):
                        writer.writerow((frame_idx, idx, lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)))
        finally:
            cap.release()

    return out_csv_path


def capture_pose_from_camera(
    camera_index: int = 0,
    resize_scale: float = 1.0,
    frame_stride: int = 1,
    capture_seconds: Optional[int] = 10,
    target_fps: int = 15,
    max_frames: Optional[int] = None,
    out_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Webカメラから一定時間ポーズ推論を行い、ランドマークをDataFrameで返す。
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"カメラを開けませんでした: index={camera_index}")

    if max_frames is None and capture_seconds is not None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = float(target_fps)
        max_frames = int(fps * capture_seconds)

    rows = []
    try:
        for frame_idx, _, landmarks in pose_capture_generator(
            cap=cap,
            resize_scale=resize_scale,
            frame_stride=frame_stride,
            max_frames=max_frames,
        ):
            if landmarks:
                for idx, lm in enumerate(landmarks.landmark):
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
        cap.release()

    df = pd.DataFrame(rows, columns=LANDMARK_HEADER)
    if out_csv_path:
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False)
    return df
