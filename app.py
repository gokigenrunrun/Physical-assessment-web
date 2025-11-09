import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculate_metrics import (
    SCORE_COLUMNS,
    calculate_metrics_by_frame,
    calculate_metrics_from_df,
    classify_action,
    evaluate_banzai_pose_auto,
    get_score_range,
    preprocess_landmarks,
)
from calculate_metrics import batch_evaluate_banzai
from pose_extract import capture_pose_from_camera, video_to_pose_csv

st.set_page_config(page_title="運動スコア自動採点アプリ", layout="centered")

REFERENCE_VIDEO_PATH = Path("otehon.mp4")
METRIC_LABELS = {
    "head_movement": "頭のブレ",
    "shoulder_tilt": "肩の傾き",
    "torso_tilt": "体幹の傾き",
    "leg_lift": "足上げ高さ",
    "foot_sway": "接地足の横ブレ",
    "arm_sag": "腕の垂れ下がり",
    "banzai_score": "バンザイ姿勢",
    "average_score": "平均スコア",
}

ACTION_LABELS = {
    "right_leg_1": "右足上げ (1回目)",
    "right_leg_2": "右足上げ (2回目)",
    "left_leg_1": "左足上げ (1回目)",
    "left_leg_2": "左足上げ (2回目)",
}

LEG_PHASE_ORDER = ["right_leg_1", "left_leg_1", "right_leg_2", "left_leg_2"]
LEG_PHASE_GROUPS = {
    "right_leg": ["right_leg_1", "right_leg_2"],
    "left_leg": ["left_leg_1", "left_leg_2"],
}
LEG_GROUP_LABELS = {
    "right_leg": "右足平均",
    "left_leg": "左足平均",
}
LEG_PHASE_SHADING = [
    ("right_leg_1", 15, 29, "skyblue"),
    ("left_leg_1", 51, 65, "lightpink"),
    ("right_leg_2", 86, 100, "skyblue"),
    ("left_leg_2", 120, 134, "lightpink"),
]
ATTEMPT_COLOR_PINK = "#FF69B4"
ATTEMPT_COLOR_BLUE = "#007BFF"
ATTEMPT_FILL_PINK = "rgba(255,105,180,0.4)"
ATTEMPT_FILL_BLUE = "rgba(0,123,255,0.6)"
AVERAGE_SCORE_COLOR = "#FF8C00"
LEG_RADAR_STYLES = {
    "right_leg": [
        ("right_leg_1", "1回目", ATTEMPT_COLOR_PINK, ATTEMPT_FILL_PINK),
        ("right_leg_2", "2回目", ATTEMPT_COLOR_BLUE, ATTEMPT_FILL_BLUE),
    ],
    "left_leg": [
        ("left_leg_1", "1回目", ATTEMPT_COLOR_PINK, ATTEMPT_FILL_PINK),
        ("left_leg_2", "2回目", ATTEMPT_COLOR_BLUE, ATTEMPT_FILL_BLUE),
    ],
}
LEG_RADAR_TITLES = {
    "right_leg": "右足上げ（1回目・2回目）",
    "left_leg": "左足上げ（1回目・2回目）",
}
SCORE_TIER_RULES = [
    (85, "#2ECC71", "Excellent", "動きが非常に安定しています"),
    (70, "#2979FF", "Good", "バランス良く動けています"),
    (55, "#FFB300", "Fair", "動作の安定性をさらに高めましょう"),
    (0, "#FF4081", "Needs Improvement", "改善の余地があります"),
]
DEFAULT_SCORE_COLOR = "#FF4081"
DEFAULT_SCORE_LABEL = "No Score"
DEFAULT_SCORE_MESSAGE = "計測データが不足しています"
METRIC_FEEDBACK_TEMPLATES = {
    "head_movement": {
        "high": "頭部が安定しており視線がぶれません。",
        "mid": "頭部のブレは小さいですが維持を意識しましょう。",
        "low": "頭部の揺れを抑える意識を高めてください。",
    },
    "shoulder_tilt": {
        "high": "肩のラインが水平に保たれています。",
        "mid": "肩の傾きは許容範囲ですがさらに安定させましょう。",
        "low": "左右の肩の高さを揃える意識を持ちましょう。",
    },
    "torso_tilt": {
        "high": "体幹がまっすぐ維持できています。",
        "mid": "体幹は概ね安定。呼吸を合わせてブレを抑えましょう。",
        "low": "上体が揺れているので姿勢のキープを意識してください。",
    },
    "leg_lift": {
        "high": "足上げ高さは十分です。",
        "mid": "もう少し高く上げるとさらに評価が上がります。",
        "low": "足を大きく引き上げる動きを意識しましょう。",
    },
    "foot_sway": {
        "high": "接地足が安定し軸がぶれていません。",
        "mid": "接地足は概ね安定。体重の乗せ方を一定にしましょう。",
        "low": "接地足が揺れているため軸を意識して立ちましょう。",
    },
    "arm_sag": {
        "high": "腕の高さをしっかり保てています。",
        "mid": "腕は保てていますが肩からの引き上げを意識しましょう。",
        "low": "腕が下がりやすいので肘を高く維持しましょう。",
    },
    "banzai_score": {
        "high": "バンザイ姿勢が美しくキープできています。",
        "mid": "バンザイ姿勢は概ね良好。フィニッシュを丁寧に。",
        "low": "肩と腕を大きく伸ばし、姿勢を保ちましょう。",
    },
}
DEFAULT_FEEDBACK_TEMPLATE = {
    "high": "動きが安定しています。",
    "mid": "引き続き安定性を意識しましょう。",
    "low": "改善ポイントを意識して動作を整えましょう。",
}


def describe_total_score(score: float) -> Tuple[str, str, str]:
    if not np.isfinite(score):
        return DEFAULT_SCORE_COLOR, DEFAULT_SCORE_LABEL, DEFAULT_SCORE_MESSAGE
    for threshold, color, label, message in SCORE_TIER_RULES:
        if score >= threshold:
            return color, label, message
    return DEFAULT_SCORE_COLOR, DEFAULT_SCORE_LABEL, DEFAULT_SCORE_MESSAGE


def score_to_color(score: float) -> str:
    if not np.isfinite(score):
        return "#9E9E9E"
    for threshold, color, *_ in SCORE_TIER_RULES:
        if score >= threshold:
            return color
    return DEFAULT_SCORE_COLOR


def select_metric_feedback(metric_key: str, score: float) -> str:
    template = METRIC_FEEDBACK_TEMPLATES.get(metric_key, DEFAULT_FEEDBACK_TEMPLATE)
    if not np.isfinite(score):
        return "データ不足のため評価できませんでした。"
    if score >= 75:
        return template.get("high") or DEFAULT_FEEDBACK_TEMPLATE["high"]
    if score >= 50:
        return template.get("mid") or template.get("high") or DEFAULT_FEEDBACK_TEMPLATE["mid"]
    return template.get("low") or template.get("mid") or DEFAULT_FEEDBACK_TEMPLATE["low"]

DEFAULT_DISPLAY_ASPECT_RATIO = 3 / 4  # width / height
DEFAULT_DISPLAY_HEIGHT = 720
DEFAULT_CAPTURE_SECONDS = 12
COUNTDOWN_SECONDS = 5


def _get_reference_dimensions(path: Path) -> Optional[tuple[int, int]]:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        return None
    return width, height


def _get_reference_duration(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return None
    return frame_count / fps


_REF_DIMS = _get_reference_dimensions(REFERENCE_VIDEO_PATH)
if _REF_DIMS:
    DISPLAY_WIDTH, DISPLAY_HEIGHT = _REF_DIMS
    DISPLAY_ASPECT_RATIO = DISPLAY_WIDTH / DISPLAY_HEIGHT if DISPLAY_HEIGHT else DEFAULT_DISPLAY_ASPECT_RATIO
else:
    DISPLAY_ASPECT_RATIO = DEFAULT_DISPLAY_ASPECT_RATIO
    DISPLAY_HEIGHT = DEFAULT_DISPLAY_HEIGHT
    DISPLAY_WIDTH = int(DISPLAY_HEIGHT * DISPLAY_ASPECT_RATIO)

REFERENCE_DURATION_SECONDS = _get_reference_duration(REFERENCE_VIDEO_PATH) or DEFAULT_CAPTURE_SECONDS


def render_reference_video_element(
    placeholder,
    *,
    autoplay: bool = False,
    loop: bool = False,
    muted: bool = True,
) -> None:
    """
    Render the reference video with consistent options or show a fallback message.
    """
    if not REFERENCE_VIDEO_PATH.exists():
        placeholder.info("お手本動画が見つかりません。")
        return
    placeholder.video(
        str(REFERENCE_VIDEO_PATH),
        start_time=0,
        autoplay=autoplay,
        loop=loop,
        muted=muted,
    )


def warm_up_camera(camera_index: int = 0, frames: int = 10, delay: float = 0.3) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return
    try:
        for _ in range(frames):
            cap.read()
        if delay > 0:
            time.sleep(delay)
    finally:
        cap.release()


def crop_to_aspect_ratio(frame: np.ndarray, target_ratio: float = DISPLAY_ASPECT_RATIO) -> np.ndarray:
    """
    Center-crop the frame to match the desired width/height ratio.
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame

    current_ratio = w / h
    if np.isclose(current_ratio, target_ratio, atol=0.01):
        return frame

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        start_x = max(0, (w - new_w) // 2)
        end_x = start_x + new_w
        return frame[:, start_x:end_x]
    else:
        new_h = int(w / target_ratio)
        start_y = max(0, (h - new_h) // 2)
        end_y = start_y + new_h
        return frame[start_y:end_y, :]

def init_session_state() -> None:
    defaults = {
        "page": "start",
        "source_type": "動画アップロード",
        "measurement_config": None,
        "measurement_ready": False,
        "result_df": None,
        "frame_metrics_df": None,
        "frame_scores_df": None,
        "pose_dataframe": None,
        "pose_csv_bytes": None,
        "frame_scores_csv": None,
        "source_label": None,
        "wait_until": None,
        "temp_paths": [],
        "countdown_active": False,
        "countdown_duration": COUNTDOWN_SECONDS,
        "camera_warmed": False,
        "warmup_camera": None,
        "warmup_camera_initialized": False,
        "measurement_start_timestamp": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def cleanup_temp_paths() -> None:
    temp_paths = st.session_state.get("temp_paths", [])
    for path_str in temp_paths:
        try:
            Path(path_str).unlink(missing_ok=True)
        except Exception:
            pass
    st.session_state["temp_paths"] = []


def release_warmup_camera() -> None:
    cap = st.session_state.get("warmup_camera")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state["warmup_camera"] = None
    st.session_state["warmup_camera_initialized"] = False
    st.session_state["camera_warmed"] = False


def reset_measurement_state() -> None:
    cleanup_temp_paths()
    release_warmup_camera()
    keys_to_reset = [
        "measurement_config",
        "measurement_ready",
        "result_df",
        "frame_metrics_df",
        "frame_scores_df",
        "pose_dataframe",
        "pose_csv_bytes",
        "frame_scores_csv",
        "source_label",
        "wait_until",
    ]
    for key in keys_to_reset:
        st.session_state[key] = None if key != "measurement_ready" else False
    st.session_state["page"] = "start"
    st.session_state["countdown_active"] = False
    st.session_state["countdown_duration"] = COUNTDOWN_SECONDS
    st.session_state["camera_warmed"] = False
    st.session_state["measurement_start_timestamp"] = None


def scale_score(value: float, min_val: float, max_val: float) -> float:
    if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val):
        return np.nan
    low = min(min_val, max_val)
    high = max(min_val, max_val)
    if np.isclose(low, high):
        return 100.0 if value <= low else 0.0
    if value <= low:
        return 100.0
    if value >= high:
        return 0.0
    ratio = (value - low) / (high - low)
    score = 100.0 * (1.0 - ratio)
    return float(np.clip(score, 0.0, 100.0))


def score_data(
    pose_df: pd.DataFrame,
    label: str,
    frame_metrics: Optional[pd.DataFrame] = None,
    action: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if action == "banzai":
        banzai_result = evaluate_banzai_pose_auto(pose_df)
        banzai_result["file"] = label
        base_metrics = {metric: np.nan for metric in SCORE_COLUMNS}
        base_scores = {f"{metric}_score": np.nan for metric in SCORE_COLUMNS}
        result = {
            "file_name": label,
            **base_metrics,
            **base_scores,
            "total_score": np.nan,
            **banzai_result,
        }
        return pd.DataFrame([result]), None

    metrics = calculate_metrics_from_df(pose_df)
    if frame_metrics is None:
        frame_metrics = calculate_metrics_by_frame(pose_df)

    frame_scores_df = build_frame_score_table(frame_metrics) if frame_metrics is not None else None

    metric_scores: Dict[str, float] = {}
    if frame_scores_df is not None and not frame_scores_df.empty:
        for key in SCORE_COLUMNS:
            score_col = f"{key}_score"
            if score_col in frame_scores_df.columns:
                metric_scores[score_col] = float(frame_scores_df[score_col].mean(skipna=True))
            else:
                metric_scores[score_col] = np.nan
    else:
        for key in SCORE_COLUMNS:
            low, high = get_score_range(key, None)
            metric_scores[f"{key}_score"] = scale_score(metrics.get(key, np.nan), low, high)

    total = float(
        np.nanmean([metric_scores[f"{k}_score"] for k in SCORE_COLUMNS])
    ) if SCORE_COLUMNS else np.nan

    result = {"file_name": label, **metrics, **metric_scores, "total_score": total}
    return pd.DataFrame([result]), frame_scores_df


def build_frame_score_table(frame_metrics: pd.DataFrame) -> pd.DataFrame:
    if frame_metrics is None or frame_metrics.empty:
        return pd.DataFrame()
    score_df = frame_metrics.copy()
    for key in SCORE_COLUMNS:
        if key in score_df.columns:
            def _score_row(row: pd.Series) -> float:
                low, high = get_score_range(key, row.get("action"))
                return scale_score(row[key], low, high)

            score_df[f"{key}_score"] = score_df.apply(_score_row, axis=1)
    score_cols = [col for col in score_df.columns if col.endswith("_score")]
    if score_cols:
        score_df["average_score"] = score_df[score_cols].mean(axis=1, skipna=True)
    return score_df


def build_summary_display_df(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df is None or result_df.empty:
        return result_df
    display_df = result_df.copy()
    duplicate_pair = {"banzai_score", "banzai_score_score"}
    if duplicate_pair.issubset(display_df.columns):
        display_df = display_df.drop(columns=["banzai_score_score"])
    return display_df


def build_frame_chart(frame_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame_scores.empty or "frame" not in frame_scores.columns:
        return fig
    x_values = frame_scores["frame"]
    for col in frame_scores.columns:
        if col in {"frame", "action"}:
            continue
        if col == "banzai_score":
            continue
        if col.endswith("_score"):
            base = col.replace("_score", "")
            label = f"{METRIC_LABELS.get(base, base)} (score)"
            fig.add_trace(go.Scatter(x=x_values, y=frame_scores[col], mode="lines", name=label))
        elif col == "average_score":
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=frame_scores[col],
                    mode="lines+markers",
                    name="平均スコア（重要指標）",
                    line=dict(width=4, color=AVERAGE_SCORE_COLOR),
                    marker=dict(size=6, color=AVERAGE_SCORE_COLOR),
                    legendrank=1,
                )
            )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Frame",
        yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
    )
    for _, start, end, color in LEG_PHASE_SHADING:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=0.15,
            line_width=0,
            layer="below",
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="skyblue", width=14),
            name="背景色: 右足上げフェーズ",
            legendrank=1000,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="lightpink", width=14),
            name="背景色: 左足上げフェーズ",
            legendrank=1001,
        )
    )
    return fig


def render_metric_feedback_cards(result_row: pd.Series) -> None:
    st.markdown("### 🧩 指標別フィードバック")
    metric_keys = [key for key in SCORE_COLUMNS if f"{key}_score" in result_row.index]
    if not metric_keys:
        st.info("指標スコアがまだ計算されていません。")
        return
    columns = st.columns(2)
    for idx, metric_key in enumerate(metric_keys):
        col = columns[idx % 2]
        score_val = float(result_row.get(f"{metric_key}_score", np.nan))
        color = score_to_color(score_val)
        label = METRIC_LABELS.get(metric_key, metric_key)
        feedback = select_metric_feedback(metric_key, score_val)
        score_text = "--" if not np.isfinite(score_val) else f"{score_val:.1f}"
        card_html = f"""
        <div style="
            background-color:#FFFFFF;
            border:1px solid #F0F0F0;
            border-radius:16px;
            padding:20px;
            margin-bottom:16px;
            box-shadow:0 8px 20px rgba(0,0,0,0.04);
        ">
            <div style="font-size:15px;color:#8A8A8A;margin-bottom:6px;">{label}</div>
            <div style="font-size:34px;font-weight:700;color:{color};line-height:1;">{score_text}</div>
            <div style="font-size:14px;color:#4F4F4F;margin-top:6px;">{feedback}</div>
        </div>
        """
        col.markdown(card_html, unsafe_allow_html=True)


def extract_pose_from_video(video_path: str, resize_scale: float, frame_stride: int) -> pd.DataFrame:
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp_csv_path = tmp_csv.name
    tmp_csv.close()
    try:
        video_to_pose_csv(
            video_path=video_path,
            out_csv_path=tmp_csv_path,
            resize_scale=resize_scale,
            frame_stride=frame_stride,
        )
        pose_df = pd.read_csv(tmp_csv_path)
        return preprocess_landmarks(pose_df)
    finally:
        Path(tmp_csv_path).unlink(missing_ok=True)


def run_measurement(config: Dict) -> Dict:
    mode = config["mode"]
    label = config.get("label", "measurement")
    frame_callback = config.get("frame_callback")

    if mode == "video":
        pose_df = extract_pose_from_video(
            video_path=config["video_path"],
            resize_scale=config["resize_scale"],
            frame_stride=config["frame_stride"],
        )
    elif mode == "webcam":
        raw_df = capture_pose_from_camera(
            camera_index=config.get("camera_index", 0),
            warmup_camera=config.get("warmup_camera"),
            resize_scale=config["resize_scale"],
            frame_stride=config["frame_stride"],
            capture_seconds=config["capture_seconds"],
            target_fps=config["target_fps"],
            frame_callback=frame_callback,
            return_start_timestamp=True,
        )
        raw_df, start_ts = raw_df
        pose_df = preprocess_landmarks(raw_df)
    elif mode == "csv":
        pose_df = preprocess_landmarks(config["dataframe"])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    frame_metrics_df = calculate_metrics_by_frame(pose_df)
    result_df, frame_scores_df = score_data(
        pose_df,
        label,
        frame_metrics_df,
        config.get("action"),
    )

    pose_csv_bytes = pose_df.to_csv(index=False).encode("utf-8")
    frame_scores_csv = None
    if frame_scores_df is not None and not frame_scores_df.empty:
        frame_scores_csv = frame_scores_df.to_csv(index=False).encode("utf-8")

    return {
        "result_df": result_df,
        "frame_metrics_df": frame_metrics_df,
        "frame_scores_df": frame_scores_df,
        "pose_dataframe": pose_df,
        "pose_csv_bytes": pose_csv_bytes,
        "frame_scores_csv": frame_scores_csv,
        "label": label,
        "start_timestamp": locals().get("start_ts"),
    }


def render_start_view() -> None:
    st.title("💪 運動スコア自動採点アプリ")
    st.write("スタートボタンを押して計測を開始しましょう。")

    if not st.session_state.get("warmup_camera_initialized", False):
        camera_index = 0
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError("カメラを初期化できませんでした。")
            for _ in range(10):
                ok, _ = cap.read()
                if not ok:
                    break
            st.session_state["warmup_camera"] = cap
            st.session_state["warmup_camera_initialized"] = True
            st.session_state["camera_warmed"] = True
            st.info("📸 カメラのウォームアップが完了しました。")
        except Exception as exc:
            if cap is not None:
                cap.release()
            release_warmup_camera()
            st.warning(f"カメラのウォームアップに失敗しました: {exc}")
    elif st.session_state.get("camera_warmed"):
        st.caption("📸 カメラの準備が整っています。")

    st.session_state["source_type"] = st.radio(
        "入力ソースを選択",
        ["動画アップロード", "Webカメラ"],
        index=0 if st.session_state["source_type"] == "動画アップロード" else 1,
        horizontal=True,
    )

    video_file = None
    resize_scale = 0.7
    frame_stride = 1
    capture_seconds = 8
    target_fps = 15

    if st.session_state["source_type"] == "動画アップロード":
        video_file = st.file_uploader("動画ファイルを選択 (mp4 / mov / avi / mkv)", type=["mp4", "mov", "avi", "mkv"])
        col1, col2 = st.columns(2)
        resize_scale = col1.slider("縮小倍率（軽量化）", 0.3, 1.0, 0.7, 0.1)
        frame_stride = col2.slider("フレーム間引き", 1, 5, 1, 1)
    else:
        col1, col2, col3 = st.columns(3)
        default_capture = max(3, int(round(REFERENCE_DURATION_SECONDS)))
        slider_max = max(default_capture, 20)
        capture_seconds = col1.slider("計測時間（秒）", 3, slider_max, default_capture)
        frame_stride = col2.slider("フレーム間引き", 1, 5, 1, 1)
        resize_scale = col3.slider("縮小倍率（軽量化）", 0.4, 1.0, 0.7, 0.1)

    csv_debug_df = None
    csv_debug_file = None
    with st.expander("🔧 Expert Mode (CSV デバッグ)"):
        csv_debug_file = st.file_uploader("骨格CSVを直接アップロード", type=["csv"], key="csv_debug_uploader")
        if csv_debug_file is not None:
            try:
                csv_debug_file.seek(0)
                csv_debug_df = pd.read_csv(csv_debug_file)
                st.success("CSVを読み込みました。")
            except Exception as exc:
                st.error(f"CSVの読み込みに失敗しました: {exc}")
                csv_debug_df = None

    start_disabled = bool(st.session_state.get("measurement_ready")) or st.session_state.get("countdown_active", False)
    if st.button("🟢 計測スタート", type="primary", use_container_width=True, disabled=start_disabled):
        if csv_debug_df is not None:
            config = {
                "mode": "csv",
                "dataframe": csv_debug_df,
                "label": csv_debug_file.name if csv_debug_file else "csv_input",
            }
        elif st.session_state["source_type"] == "動画アップロード":
            if video_file is None:
                st.warning("動画ファイルを選択してください。")
                return
            tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix)
            tmp_video.write(video_file.getbuffer())
            tmp_video_path = tmp_video.name
            tmp_video.close()
            st.session_state["temp_paths"].append(tmp_video_path)
            config = {
                "mode": "video",
                "video_path": tmp_video_path,
                "label": video_file.name,
                "resize_scale": resize_scale,
                "frame_stride": frame_stride,
            }
        else:
            config = {
                "mode": "webcam",
                "label": "webcam_capture",
                "resize_scale": resize_scale,
                "frame_stride": frame_stride,
                "capture_seconds": capture_seconds,
                "target_fps": target_fps,
                "camera_index": 0,
            }

        st.session_state["measurement_config"] = config
        st.session_state["measurement_ready"] = False
        st.session_state["countdown_active"] = True
        st.session_state["countdown_duration"] = COUNTDOWN_SECONDS
        st.session_state["camera_warmed"] = False
        st.session_state["page"] = "measuring"
        st.rerun()


def render_measuring_view() -> None:
    config = st.session_state.get("measurement_config")
    if not config:
        reset_measurement_state()
        st.rerun()
        return

    if st.session_state.get("countdown_active"):
        st.header("🎬 計測開始準備中…")
        message_placeholder = st.empty()
        countdown_placeholder = st.empty()
        message_placeholder.info("🎬 計測開始までお待ちください…")
        duration = int(st.session_state.get("countdown_duration", COUNTDOWN_SECONDS) or COUNTDOWN_SECONDS)
        for value in range(duration, 0, -1):
            countdown_placeholder.markdown(
                f"""
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:70vh;">
                    <div style="font-size:9rem; font-weight:700; color:#F3722C; line-height:1;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(1)
        countdown_placeholder.markdown(
            """
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:70vh;">
                <div style="font-size:6rem; font-weight:700; color:#43AA8B; line-height:1;">スタート!</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.3)
        st.session_state["countdown_active"] = False
        st.session_state["measurement_ready"] = True
        st.rerun()
        return

    st.header("🏃‍♀️ 計測中…")
    col1, col2 = st.columns([1, 1])
    reference_video_placeholder = None
    with col1:
        st.subheader("お手本")
        reference_video_placeholder = st.empty()
        if not st.session_state.get("measurement_ready"):
            reference_video_placeholder.info("カウントダウン完了後にお手本動画が再生されます。")
    live_placeholder = None
    with col2:
        st.subheader("あなたの動き")
        if config["mode"] == "video":
            st.video(config["video_path"])
        elif config["mode"] == "webcam":
            live_placeholder = st.empty()
            live_placeholder.info("Webカメラ映像を初期化しています…")
        else:
            st.info("CSVデータを解析しています…")

    phase_placeholder = st.empty()
    if config["mode"] != "webcam":
        phase_placeholder.markdown("**🏃‍♀️ 計測中：解析中…**")

    st.markdown("### 🏃‍♀️ 計測中です…")
    st.caption("分析が完了すると自動的に結果画面へ移動します。")

    config_for_run = dict(config)
    if config["mode"] == "webcam":
        config_for_run["warmup_camera"] = st.session_state.get("warmup_camera")
    if config["mode"] == "webcam" and live_placeholder is not None:
        def frame_callback(frame_idx: int, frame_rgb: np.ndarray) -> None:
            flipped = np.ascontiguousarray(frame_rgb[:, ::-1, :])
            cropped = crop_to_aspect_ratio(flipped)
            if cropped is None:
                return
            resized = cv2.resize(
                cropped,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            live_placeholder.image(
                resized,
                channels="RGB",
                caption=f"Frame {frame_idx}",
            )
            action_key = classify_action(frame_idx)
            phase_label = ACTION_LABELS.get(action_key, "動作中")
            phase_placeholder.markdown(f"**🏃‍♀️ 計測中：{phase_label}**")

        config_for_run["frame_callback"] = frame_callback

    measurement_ready = st.session_state.get("measurement_ready")
    if measurement_ready and reference_video_placeholder is not None:
        render_reference_video_element(
            reference_video_placeholder,
            autoplay=True,
            loop=False,
        )

    if measurement_ready:
        measurement_result: Dict = {}
        try:
            with st.spinner("分析中…"):
                measurement_result = run_measurement(config_for_run)
        finally:
            st.session_state["measurement_ready"] = False
            if config["mode"] == "webcam":
                release_warmup_camera()
        st.session_state["result_df"] = measurement_result["result_df"]
        st.session_state["frame_metrics_df"] = measurement_result["frame_metrics_df"]
        st.session_state["frame_scores_df"] = measurement_result["frame_scores_df"]
        st.session_state["pose_dataframe"] = measurement_result["pose_dataframe"]
        st.session_state["pose_csv_bytes"] = measurement_result["pose_csv_bytes"]
        st.session_state["frame_scores_csv"] = measurement_result["frame_scores_csv"]
        st.session_state["source_label"] = measurement_result["label"]
        st.session_state["measurement_start_timestamp"] = measurement_result.get("start_timestamp")
        st.session_state["wait_until"] = time.time() + 2.0
        st.session_state["page"] = "waiting"
        st.rerun()


def render_waiting_view() -> None:
    st.header("🧠 分析しています…")
    st.info("まもなく結果を表示します。")
    wait_until = st.session_state.get("wait_until")
    if wait_until is None or time.time() >= wait_until:
        st.session_state["page"] = "result"
        st.rerun()
    else:
        time.sleep(min(1.0, max(0.0, wait_until - time.time())))
        st.rerun()


# ============================================================
# BANZAI EVALUATION TEST VIEW
# ------------------------------------------------------------
# Allows the user to run batch_evaluate_banzai() from Streamlit
# to confirm that Banzai scoring logic is working properly.
# ============================================================
def render_banzai_test_view():
    import streamlit as st
    from pathlib import Path

    st.header("🕺 Banzai Evaluation Test")
    st.write("Run Banzai scoring for all CSVs in a selected folder.")

    folder = st.text_input("Enter folder path", "data_banzai_landmarks")
    if st.button("Run Banzai Evaluation"):
        folder_path = Path(folder)
        if not folder_path.exists():
            st.error(f"❌ Folder not found: {folder}")
        else:
            with st.spinner("Evaluating all CSVs..."):
                df = batch_evaluate_banzai(folder_path)
                if df.empty:
                    st.warning("No valid CSV files found.")
                else:
                    st.success("✅ Evaluation complete!")
                    st.dataframe(df)


def render_result_view() -> None:
    result_df = st.session_state.get("result_df")
    frame_scores_df = st.session_state.get("frame_scores_df")
    if result_df is None:
        reset_measurement_state()
        st.rerun()
        return

    summary_table = build_summary_display_df(result_df)
    if summary_table is None or summary_table.empty:
        st.info("スコアデータが見つかりませんでした。もう一度計測してください。")
        return

    st.header("📊 採点結果")
    summary_row = summary_table.iloc[0]
    total_score = float(summary_row.get("total_score", np.nan))
    tier_color, tier_label, tier_message = describe_total_score(total_score)
    total_score_text = "--" if not np.isfinite(total_score) else f"{total_score:.1f}"

    score_card_html = f"""
    <div style="text-align:center;padding:32px 0;">
        <div style="font-size:20px;color:#7B7B7B;">総合スコア（0〜100）</div>
        <div style="font-size:92px;font-weight:800;color:{tier_color};line-height:1;">
            {total_score_text}
        </div>
        <div style="font-size:30px;font-weight:600;color:{tier_color};margin-top:8px;">
            {tier_label}
        </div>
        <div style="font-size:16px;color:#555555;margin-top:4px;">
            {tier_message}
        </div>
    </div>
    """
    st.markdown(score_card_html, unsafe_allow_html=True)

    english_keys = SCORE_COLUMNS
    values = [
        float(np.nan_to_num(summary_row.get(f"{k}_score", np.nan), nan=0.0))
        for k in english_keys
    ]
    labels_closed = english_keys + [english_keys[0]]
    radar_values = values + values[:1]

    st.markdown("### 📊 モーションプロファイル")
    fig = go.Figure(
        data=go.Scatterpolar(
            r=radar_values,
            theta=labels_closed,
            fill="toself",
            line_color="#4A90E2",
            fillcolor="rgba(74,144,226,0.4)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        width=640,
        height=520,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    if np.isfinite(total_score):
        fig.add_annotation(
            dict(
                text=f"{total_score:.1f}",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#FF4081", size=44, family="Helvetica",),
            )
        )
    st.plotly_chart(fig, use_container_width=True)

    render_metric_feedback_cards(summary_row)

    if frame_scores_df is not None and not frame_scores_df.empty:
        st.markdown("### ⏱ フレームごとの推移")
        avg_frame_score = float(frame_scores_df["average_score"].mean(skipna=True)) if "average_score" in frame_scores_df else np.nan
        if np.isfinite(avg_frame_score):
            st.metric("平均フレームスコア（重要指標）", f"{avg_frame_score:.1f} 点")
        st.plotly_chart(build_frame_chart(frame_scores_df), use_container_width=True)
        with st.expander("フレーム別スコアを表示"):
            st.dataframe(frame_scores_df, use_container_width=True)

        if "action" in frame_scores_df.columns:
            score_cols = [col for col in frame_scores_df.columns if col.endswith("_score")]
            include_avg = "average_score" in frame_scores_df.columns
            if score_cols:
                group_cols = score_cols + (["average_score"] if include_avg else [])
                action_means = frame_scores_df.groupby("action")[group_cols].mean().round(1)
                action_means = action_means.loc[action_means.index.isin(ACTION_LABELS.keys())]
                if not action_means.empty:
                    ordered = [phase for phase in LEG_PHASE_ORDER if phase in action_means.index]
                    remainder = [idx for idx in action_means.index if idx not in ordered]
                    action_means = action_means.reindex(ordered + remainder)

                    column_map: Dict[str, str] = {}
                    display_df = action_means.rename(index=lambda k: ACTION_LABELS.get(k, k))
                    for col_name in display_df.columns:
                        if col_name.endswith("_score"):
                            metric_key = col_name.replace("_score", "")
                            column_map[col_name] = f"{METRIC_LABELS.get(metric_key, metric_key)}(score)"
                        elif col_name == "average_score":
                            column_map[col_name] = "平均スコア"
                    if column_map:
                        display_df = display_df.rename(columns=column_map)
                    display_df = display_df.loc[:, ~display_df.columns.duplicated()]
                    st.subheader("🧭 動作フェーズ別平均スコア")
                    st.dataframe(display_df, use_container_width=True)

                    combined_rows = {}
                    for group_key, members in LEG_PHASE_GROUPS.items():
                        existing_members = [m for m in members if m in action_means.index]
                        if not existing_members:
                            continue
                        combined_rows[group_key] = action_means.loc[existing_members].mean().round(1)
                    if combined_rows:
                        combined_df = pd.DataFrame(combined_rows).T
                        combined_df = combined_df.rename(index=lambda k: LEG_GROUP_LABELS.get(k, k))
                        if column_map:
                            combined_df = combined_df.rename(columns=column_map)
                        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                        st.subheader("🦵 左右レッグ平均スコア")
                        st.dataframe(combined_df, use_container_width=True)

                    radar_groups = [
                        group_key
                        for group_key in LEG_PHASE_GROUPS
                        if any(member in action_means.index for member in LEG_PHASE_GROUPS[group_key])
                    ]
                    radar_cols = st.columns(len(radar_groups)) if radar_groups else []
                    for col_slot, group_key in zip(radar_cols, radar_groups):
                        with col_slot:
                            styles = LEG_RADAR_STYLES.get(group_key, [])
                            traces = []
                            metric_labels = [
                                METRIC_LABELS.get(metric, metric)
                                for metric in SCORE_COLUMNS
                                if f"{metric}_score" in action_means.columns
                            ]
                            if not metric_labels:
                                continue
                            labels_closed = metric_labels + [metric_labels[0]]
                            fig_action = go.Figure()
                            for phase_key, suffix, line_color, fill_color in styles:
                                if phase_key not in action_means.index:
                                    continue
                                per_action_values = [
                                    float(action_means.loc[phase_key, f"{metric}_score"])
                                    for metric in SCORE_COLUMNS
                                    if f"{metric}_score" in action_means.columns
                                ]
                                if not per_action_values:
                                    continue
                                values_closed = per_action_values + per_action_values[:1]
                                fig_action.add_trace(
                                    go.Scatterpolar(
                                        r=values_closed,
                                        theta=labels_closed,
                                        fill="toself",
                                        name=f"{ACTION_LABELS.get(phase_key, phase_key)} {suffix}",
                                        line_color=line_color,
                                        fillcolor=fill_color,
                                        opacity=1.0,
                                    )
                                )
                            if not fig_action.data:
                                continue
                            fig_action.update_layout(
                                title=dict(text=LEG_RADAR_TITLES.get(group_key, group_key), x=0.5, font=dict(size=16)),
                                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                                margin=dict(l=20, r=20, t=60, b=60),
                                height=360,
                            )
                            st.plotly_chart(fig_action, use_container_width=True)

    with st.expander("スコア詳細テーブルを表示"):
        st.dataframe(summary_table, use_container_width=True)

    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.get("frame_scores_csv") is not None:
            st.download_button(
                "💾 フレームスコアをCSVで保存",
                data=st.session_state["frame_scores_csv"],
                file_name="frame_scores.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if st.session_state.get("pose_csv_bytes") is not None:
            st.download_button(
                "💾 骨格データをCSVで保存",
                data=st.session_state["pose_csv_bytes"],
                file_name="pose_landmarks.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with col2:
        st.button("🔁 再計測", on_click=reset_measurement_state, use_container_width=True)


def main() -> None:
    init_session_state()
    page = st.session_state["page"]

    if page == "start":
        render_start_view()
    elif page == "measuring":
        render_measuring_view()
    elif page == "waiting":
        render_waiting_view()
    elif page == "result":
        render_result_view()
    else:
        reset_measurement_state()
        render_start_view()


if __name__ == "__main__":
    main()
