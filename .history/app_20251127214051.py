import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import base64
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from html import escape

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

st.set_page_config(page_title="Motion Score Auto Evaluation App", layout="centered")

REFERENCE_VIDEO_PATH = Path("otehon.mp4")
METRIC_LABELS = {
    "head_movement": "Head Stability",
    "shoulder_tilt": "Shoulder Tilt",
    "torso_tilt": "Torso Lean",
    "leg_lift": "Leg Lift Height",
    "foot_sway": "Foot Sway",
    "arm_sag": "Arm Drop",
    "banzai_score": "Banzai Posture",
    "average_score": "Average Score",
}
SCORE_COLORS = {
    "Banzai Posture": "#1E3A8A",
    "Head Stability": "#EC4899",
    "Shoulder Tilt": "#3B82F6",
    "Torso Lean": "#0EA5E9",
    "Arm Drop": "#F59E0B",
    "Foot Sway": "#10B981",
    "Leg Lift Height": "#8B5CF6",
}
NEUTRAL_COLOR = "#9CA3AF"

ACTION_LABELS = {
    "right_leg_1": "Right Leg Lift (Attempt 1)",
    "right_leg_2": "Right Leg Lift (Attempt 2)",
    "left_leg_1": "Left Leg Lift (Attempt 1)",
    "left_leg_2": "Left Leg Lift (Attempt 2)",
}

LEG_PHASE_ORDER = ["right_leg_1", "left_leg_1", "right_leg_2", "left_leg_2"]
LEG_PHASE_GROUPS = {
    "right_leg": ["right_leg_1", "right_leg_2"],
    "left_leg": ["left_leg_1", "left_leg_2"],
}
LEG_GROUP_LABELS = {
    "right_leg": "Right Leg Average",
    "left_leg": "Left Leg Average",
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
AVERAGE_SCORE_COLOR = NEUTRAL_COLOR
LEG_RADAR_STYLES = {
    "right_leg": [
        ("right_leg_1", "Attempt 1", ATTEMPT_COLOR_PINK, ATTEMPT_FILL_PINK),
        ("right_leg_2", "Attempt 2", ATTEMPT_COLOR_BLUE, ATTEMPT_FILL_BLUE),
    ],
    "left_leg": [
        ("left_leg_1", "Attempt 1", ATTEMPT_COLOR_PINK, ATTEMPT_FILL_PINK),
        ("left_leg_2", "Attempt 2", ATTEMPT_COLOR_BLUE, ATTEMPT_FILL_BLUE),
    ],
}
LEG_RADAR_TITLES = {
    "right_leg": "Right Leg Lifts (Attempts 1 & 2)",
    "left_leg": "Left Leg Lifts (Attempts 1 & 2)",
}
SCORE_TIER_RULES = [
    (85, "#2ECC71", "Excellent", "Movement is exceptionally stable."),
    (70, "#2979FF", "Good", "Great balance throughout the motion."),
    (55, "#FFB300", "Fair", "Keep refining overall stability."),
    (0, "#FF4081", "Needs Improvement", "There is still room to improve the form."),
]
DEFAULT_SCORE_COLOR = "#FF4081"
DEFAULT_SCORE_LABEL = "No Score"
DEFAULT_SCORE_MESSAGE = "Not enough measurement data is available."


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_metric_title(metric_key: str) -> str:
    return METRIC_LABELS.get(metric_key, metric_key)


def get_metric_color_by_title(title: str) -> str:
    return SCORE_COLORS.get(title, "#3B82F6")


def get_metric_color_by_key(metric_key: str) -> str:
    return get_metric_color_by_title(get_metric_title(metric_key))
METRIC_FEEDBACK_TEMPLATES = {
    "head_movement": {
        "high": "Head stays steady with no noticeable wobble.",
        "mid": "Head movement is minor - keep focusing on stability.",
        "low": "Focus on reducing head sway during the motion.",
    },
    "shoulder_tilt": {
        "high": "Shoulder line remains level.",
        "mid": "Shoulders are acceptable but can be steadier.",
        "low": "Work on keeping both shoulders at the same height.",
    },
    "torso_tilt": {
        "high": "Torso stays upright throughout the motion.",
        "mid": "Torso is mostly stable - coordinate breathing to reduce sway.",
        "low": "Upper body is wobbling; keep the core engaged.",
    },
    "leg_lift": {
        "high": "Leg lift height is sufficient.",
        "mid": "Lifting a bit higher will boost the score.",
        "low": "Drive the knee higher to emphasize the lift.",
    },
    "foot_sway": {
        "high": "Supporting foot is steady with a solid axis.",
        "mid": "Footing is mostly stable - keep the weight placement consistent.",
        "low": "Grounded foot is swaying; focus on building a stable axis.",
    },
    "arm_sag": {
        "high": "Arms stay lifted throughout the motion.",
        "mid": "Arms hold up but engage the shoulders to lift more.",
        "low": "Arms drop easily - keep elbows lifted and active.",
    },
    "banzai_score": {
        "high": "Banzai posture is crisp and well held.",
        "mid": "Banzai posture is mostly good - finish with intent.",
        "low": "Extend the shoulders and arms fully to hold the pose.",
    },
}
DEFAULT_FEEDBACK_TEMPLATE = {
    "high": "Motion is stable and balanced.",
    "mid": "Stay mindful of stability as you move.",
    "low": "Focus on the improvement cues to refine your form.",
}
DETAIL_CARD_METRICS = [
    "head_movement",
    "shoulder_tilt",
    "torso_tilt",
    "leg_lift",
    "foot_sway",
    "arm_sag",
    "banzai_score",
]
METRIC_TITLE_OVERRIDES = {
    "banzai_score": "Banzai Posture",
    "head_movement": "Head Stability",
    "shoulder_tilt": "Shoulder Tilt",
    "torso_tilt": "Torso Lean",
    "leg_lift": "Leg Lift Height",
    "foot_sway": "Foot Sway",
    "arm_sag": "Arm Drop",
}
RESEARCH_UI_CSS = """
<style>
section[data-testid="stSidebar"] {display:none !important;}
div[data-testid="collapsedControl"] {display:none !important;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
body {background-color:#FFFFFF;}
.block-container {max-width:1000px;padding:2rem 1.5rem;margin:0 auto;}
.hero-score-section {
    background:#F8FAFF;
    border:1px solid #E0E7FF;
    border-radius:24px;
    padding:2.5rem 1rem;
    text-align:center;
    margin-bottom:2rem;
}
.hero-badge {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    padding:0.25rem 0.75rem;
    border-radius:999px;
    background:#DBEAFE;
    color:#1E3A8A;
    font-weight:600;
    font-size:0.9rem;
    margin-bottom:0.75rem;
}
.hero-score {font-size:64px;font-weight:700;color:#2563EB;line-height:1;}
.hero-comment {font-size:18px;color:#374151;margin-top:0.5rem;}
.section-title {
    font-size:20px;
    font-weight:600;
    color:#1E3A8A;
    margin:1rem 0 0.5rem;
}
.subsection-title {
    font-size:16px;
    font-weight:600;
    color:#1E40AF;
    margin:1.5rem 0 0.5rem;
}
.radar-wrap {display:flex;justify-content:center;margin-bottom:2rem;}
.metric-grid {
    display:flex;
    flex-direction:column;
    align-items:center;
    gap:1.5rem;
    width:100%;
}
.metric-row {
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    gap:1rem;
    width:100%;
    max-width:900px;
}
.metric-row.single {
    display:grid;
    grid-template-columns:repeat(2, minmax(0, 1fr));
    justify-content:center;
    width:100%;
    max-width:900px;
}
.metric-row.single .metric-card {
    grid-column:span 2;
    width:100%;
}
.metric-card {
    width:100%;
    background:#F9FAFB;
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:1rem 1.2rem;
}
.metric-card.long {
    height:auto;
    padding:2rem;
}
.metric-title {
    font-size:16px;
    font-weight:600;
    color:#1E3A8A;
    border-bottom:1px solid #E5E7EB;
    padding-bottom:0.3rem;
    margin-bottom:0.8rem;
}
.metric-content {
    display:flex;
    align-items:center;
}
.metric-left {
    flex:0 0 120px;
    display:flex;
    justify-content:center;
    align-items:center;
}
.metric-chart {
    width:120px;
    height:120px;
    object-fit:contain;
}
.metric-right {
    flex:1;
    padding-left:1rem;
}
.metric-score {
    font-size:22px;
    font-weight:700;
    color:#2563EB;
    margin-bottom:0.2rem;
}
.metric-comment {
    font-size:14px;
    color:#6B7280;
    line-height:1.4;
}
.metric-comment.long-comment {
    white-space:pre-line;
    line-height:1.6;
    font-size:14px;
    color:#4B5563;
    margin-top:0.5rem;
}
</style>
"""


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
        return "Unable to evaluate due to insufficient data."
    if score >= 75:
        return template.get("high") or DEFAULT_FEEDBACK_TEMPLATE["high"]
    if score >= 50:
        return template.get("mid") or template.get("high") or DEFAULT_FEEDBACK_TEMPLATE["mid"]
    return template.get("low") or template.get("mid") or DEFAULT_FEEDBACK_TEMPLATE["low"]


def render_score_block(score: float, label: str, comment_text: str) -> None:
    if np.isfinite(score):
        if score < 50:
            label_bg, label_border, label_color = "#FEE2E2", "#EF4444", "#991B1B"
        elif score < 70:
            label_bg, label_border, label_color = "#FEF3C7", "#F59E0B", "#92400E"
        elif score < 85:
            label_bg, label_border, label_color = "#DBEAFE", "#3B82F6", "#1E40AF"
        else:
            label_bg, label_border, label_color = "#DCFCE7", "#22C55E", "#14532D"
        score_text = f"{score:.1f} pts"
    else:
        label_bg, label_border, label_color = "#E5E7EB", "#9CA3AF", "#374151"
        score_text = "--"
        label = label or "No Score"
        comment_text = comment_text or DEFAULT_SCORE_MESSAGE

    st.markdown(
        f"""
        <div style="
            background-color:#EFF6FF;
            border-radius:16px;
            text-align:center;
            padding:1.8rem 1rem;
            margin-bottom:2rem;
        ">
            <div style="
                display:inline-block;
                background-color:{label_bg};
                border:2px solid {label_border};
                color:{label_color};
                font-weight:600;
                border-radius:999px;
                padding:6px 18px;
                font-size:16px;
            ">
                {label}
            </div>
            <div style="font-size:64px; font-weight:700; color:#2563EB; margin-top:8px; line-height:1;">
                {score_text}
            </div>
            <div style="font-size:18px; color:#1E3A8A; margin-top:6px;">
                {comment_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

DEFAULT_DISPLAY_ASPECT_RATIO = 3 / 4  # width / height
DEFAULT_DISPLAY_HEIGHT = 720
DEFAULT_CAPTURE_SECONDS = 12
COUNTDOWN_SECONDS = 5


def make_donut_chart(score: float, color: str = "#3B82F6") -> go.Figure:
    safe_score = float(np.clip(score if np.isfinite(score) else 0.0, 0.0, 100.0))
    remainder = max(0.0, 100.0 - safe_score)
    fig = go.Figure(
        go.Pie(
            values=[safe_score, remainder],
            marker=dict(colors=[color, "#E5E7EB"]),
            hole=0.75,
            sort=False,
            direction="clockwise",
            textinfo="none",
        )
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_metric_card_html(title: str, score: float, comment: str) -> str:
    display_value = float(score) if np.isfinite(score) else np.nan
    score_for_chart = float(np.clip(display_value, 0.0, 100.0)) if np.isfinite(display_value) else 0.0
    card_color = "#3B82F6"
    fig = make_donut_chart(score_for_chart, color=card_color)
    image_bytes = fig.to_image(format="png", width=240, height=240, scale=2)
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    title_html = escape(title)
    score_text = "-- pts" if not np.isfinite(display_value) else f"{display_value:.1f} pts"
    raw_comment = (comment or "").strip()
    is_banzai = title == "Banzai Posture"
    if is_banzai:
        detail_lines = [raw_comment] if raw_comment else []
        detail_lines.extend(
            [
                "Match the angle of both arms and draw the shoulders back for stability.",
                "Keep the head and torso aligned in a straight line.",
            ]
        )
        comment_text = "\n".join(line for line in detail_lines if line)
        if not comment_text:
            comment_text = "Banzai posture score data is missing."
        comment_class = "metric-comment long-comment"
        card_modifier = " long"
        comment_html = escape(comment_text)
    else:
        comment_class = "metric-comment"
        card_modifier = ""
        comment_text = raw_comment or "Unable to evaluate due to insufficient data."
        comment_html = escape(comment_text).replace("\n", "<br />")
    return (
        f'<div class="metric-card{card_modifier}">'
        f'\n    <div class="metric-title">{title_html}</div>'
        f'\n    <div class="metric-content">'
        f'\n        <div class="metric-left">'
        f'\n            <img class="metric-chart" src="data:image/png;base64,{encoded_image}" alt="{title_html} score chart" />'
        f"\n        </div>"
        f'\n        <div class="metric-right">'
        f'\n            <div class="metric-score">{score_text}</div>'
        f'\n            <div class="{comment_class}">{comment_html}</div>'
        f"\n        </div>"
        f"\n    </div>"
        f"\n</div>"
    )


def render_metric_card(title: str, score: float, comment: str) -> None:
    card_html = render_metric_card_html(title, score, comment)
    st.markdown(card_html, unsafe_allow_html=True)


def inject_research_ui_styles() -> None:
    if st.session_state.get("research_styles_injected"):
        return
    st.markdown(RESEARCH_UI_CSS, unsafe_allow_html=True)
    st.session_state["research_styles_injected"] = True


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
        placeholder.info("Reference video not found.")
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
        "source_type": "Video Upload",
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
        "research_styles_injected": False,
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
        if col in {"frame", "action", "average_score"}:
            continue
        if col == "banzai_score":
            continue
        if col.endswith("_score"):
            base = col.replace("_score", "")
            label = f"{METRIC_LABELS.get(base, base)} (score)"
            color = get_metric_color_by_key(base)
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=frame_scores[col],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2.5),
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
            name="Background: Right Leg Lift Phase",
            legendrank=1000,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="lightpink", width=14),
            name="Background: Left Leg Lift Phase",
            legendrank=1001,
        )
    )
    return fig


def render_metric_feedback_cards(result_row: pd.Series) -> None:
    st.markdown('<div class="section-title">🧩 Detailed Metrics</div>', unsafe_allow_html=True)
    card_order = [
        ("banzai_score", "Banzai Posture"),
        ("head_movement", "Head Stability"),
        ("shoulder_tilt", "Shoulder Tilt"),
        ("torso_tilt", "Torso Lean"),
        ("arm_sag", "Arm Drop"),
        ("foot_sway", "Foot Sway"),
        ("leg_lift", "Leg Lift Height"),
    ]
    if not any(f"{key}_score" in result_row.index for key, _ in card_order):
        st.info("Metric scores have not been calculated yet.")
        return
    def card_html(metric_key: str, title: str) -> str:
        score_val = float(result_row.get(f"{metric_key}_score", np.nan))
        feedback = select_metric_feedback(metric_key, score_val)
        return render_metric_card_html(title, score_val, feedback)

    cards_html = """
<div class="metric-grid">
    <div class="metric-row single">
        {banzai}
    </div>
    <div class="metric-row">
        {head}
        {shoulder}
    </div>
    <div class="metric-row">
        {torso}
        {arm}
    </div>
    <div class="metric-row">
        {foot}
        {leg}
    </div>
</div>
""".format(
        banzai=card_html("banzai_score", "Banzai Posture"),
        head=card_html("head_movement", "Head Stability"),
        shoulder=card_html("shoulder_tilt", "Shoulder Tilt"),
        torso=card_html("torso_tilt", "Torso Lean"),
        arm=card_html("arm_sag", "Arm Drop"),
        foot=card_html("foot_sway", "Foot Sway"),
        leg=card_html("leg_lift", "Leg Lift Height"),
    )
    st.markdown(cards_html, unsafe_allow_html=True)


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
    st.title("💪 Motion Score Auto Evaluation App")
    st.markdown("Press the start button to begin a new measurement.")

    if not st.session_state.get("warmup_camera_initialized", False):
        camera_index = 0
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError("Failed to initialize the camera.")
            for _ in range(10):
                ok, _ = cap.read()
                if not ok:
                    break
            st.session_state["warmup_camera"] = cap
            st.session_state["warmup_camera_initialized"] = True
            st.session_state["camera_warmed"] = True
            st.info("📸 Camera warm-up completed.")
        except Exception as exc:
            if cap is not None:
                cap.release()
            release_warmup_camera()
            st.warning(f"Camera warm-up failed: {exc}")
    elif st.session_state.get("camera_warmed"):
        st.caption("📸 Camera is ready.")

    st.session_state["source_type"] = st.radio(
        "Select input source",
        ["Video Upload", "Webcam"],
        index=0 if st.session_state["source_type"] == "Video Upload" else 1,
        horizontal=True,
    )

    video_file = None
    resize_scale = 0.7
    frame_stride = 1
    capture_seconds = 8
    target_fps = 15

    if st.session_state["source_type"] == "Video Upload":
        video_file = st.file_uploader("Select a video file (mp4 / mov / avi / mkv)", type=["mp4", "mov", "avi", "mkv"])
        col1, col2 = st.columns(2)
        resize_scale = col1.slider("Resize scale (lighter processing)", 0.3, 1.0, 0.7, 0.1)
        frame_stride = col2.slider("Frame stride", 1, 5, 1, 1)
    else:
        col1, col2, col3 = st.columns(3)
        default_capture = max(3, int(round(REFERENCE_DURATION_SECONDS)))
        slider_max = max(default_capture, 20)
        capture_seconds = col1.slider("Capture duration (seconds)", 3, slider_max, default_capture)
        frame_stride = col2.slider("Frame stride", 1, 5, 1, 1)
        resize_scale = col3.slider("Resize scale (lighter processing)", 0.4, 1.0, 0.7, 0.1)

    csv_debug_df = None
    csv_debug_file = None
    with st.expander("🔧 Expert Mode (CSV Debug)"):
        csv_debug_file = st.file_uploader("Upload skeleton CSV directly", type=["csv"], key="csv_debug_uploader")
        if csv_debug_file is not None:
            try:
                csv_debug_file.seek(0)
                csv_debug_df = pd.read_csv(csv_debug_file)
                st.success("CSV loaded successfully.")
            except Exception as exc:
                st.error(f"Failed to load CSV: {exc}")
                csv_debug_df = None

    start_disabled = bool(st.session_state.get("measurement_ready")) or st.session_state.get("countdown_active", False)
    if st.button("🟢 Start Measurement", type="primary", disabled=start_disabled):
        if csv_debug_df is not None:
            config = {
                "mode": "csv",
                "dataframe": csv_debug_df,
                "label": csv_debug_file.name if csv_debug_file else "csv_input",
            }
        elif st.session_state["source_type"] == "Video Upload":
            if video_file is None:
                st.warning("Please select a video file.")
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
        st.header("🎬 Preparing Measurement...")
        message_placeholder = st.empty()
        countdown_placeholder = st.empty()
        message_placeholder.info("🎬 Please wait until the measurement begins...")
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
                <div style="font-size:6rem; font-weight:700; color:#43AA8B; line-height:1;">Start!</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.3)
        st.session_state["countdown_active"] = False
        st.session_state["measurement_ready"] = True
        st.rerun()
        return

    st.header("🏃‍♀️ Measuring...")
    col1, col2 = st.columns([1, 1])
    reference_video_placeholder = None
    with col1:
        st.subheader("Reference")
        reference_video_placeholder = st.empty()
        if not st.session_state.get("measurement_ready"):
            reference_video_placeholder.info("The reference video will play after the countdown finishes.")
    live_placeholder = None
    with col2:
        st.subheader("Your Movement")
        if config["mode"] == "video":
            st.video(config["video_path"])
        elif config["mode"] == "webcam":
            live_placeholder = st.empty()
            live_placeholder.info("Initializing webcam feed...")
        else:
            st.info("Analyzing CSV data...")

    phase_placeholder = st.empty()
    if config["mode"] != "webcam":
        phase_placeholder.markdown("**🏃‍♀️ Measuring: Processing...**")

    st.markdown("### 🏃‍♀️ Measurement in progress...")
    st.caption("You will be redirected to the results screen once analysis is complete.")

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
            phase_label = ACTION_LABELS.get(action_key, "In motion")
            phase_placeholder.markdown(f"**🏃‍♀️ Measuring: {phase_label}**")

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
            with st.spinner("Analyzing..."):
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
    st.header("🧠 Analyzing...")
    st.info("Results will appear shortly.")
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
    st.markdown("Run Banzai scoring for all CSVs in a selected folder.")

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
    inject_research_ui_styles()
    result_df = st.session_state.get("result_df")
    frame_scores_df = st.session_state.get("frame_scores_df")
    if result_df is None:
        reset_measurement_state()
        st.rerun()
        return

    summary_table = build_summary_display_df(result_df)
    if summary_table is None or summary_table.empty:
        st.info("No score data was found. Please run another measurement.")
        return

    summary_row = summary_table.iloc[0]
    total_score = float(summary_row.get("total_score", np.nan))
    tier_color, tier_label, tier_message = describe_total_score(total_score)
    # === HEADER: TOTAL SCORE ===
    render_score_block(total_score, tier_label, tier_message)

    # === RADAR CHART ===
    st.markdown('<div class="section-title">📊 Motion Profile</div>', unsafe_allow_html=True)
    english_keys = SCORE_COLUMNS
    metric_labels = [METRIC_LABELS.get(k, k) for k in english_keys]
    values = [
        float(np.nan_to_num(summary_row.get(f"{k}_score", np.nan), nan=0.0))
        for k in english_keys
    ]
    labels_closed = metric_labels + [metric_labels[0]]
    radar_values = values + values[:1]
    angular_axis = dict(
        categoryorder="array",
        categoryarray=labels_closed,
        rotation=0,
        direction="clockwise",
        tickfont=dict(color="#6B7280", size=11),
        linecolor="#E5E7EB",
        gridcolor="#E5E7EB",
    )
    radial_axis = dict(
        visible=True,
        range=[0, 100],
        tickfont=dict(color="#6B7280", size=11),
        gridcolor="#E5E7EB",
        linecolor="#E5E7EB",
    )
    radar_primary_color = "#3B82F6"
    fig = go.Figure(
        data=go.Scatterpolar(
            r=radar_values,
            theta=labels_closed,
            fill="toself",
            line_color=radar_primary_color,
            fillcolor="rgba(59,130,246,0.3)",
        )
    )
    fig.update_layout(
        polar=dict(
            domain=dict(x=[0.15, 0.85], y=[0.0, 1.0]),
            radialaxis=dict(range=[0, 100], showline=False, gridcolor="rgba(0,0,0,0.1)"),
            angularaxis=dict(showline=False, gridcolor="rgba(0,0,0,0.05)"),
        ),
        showlegend=False,
        autosize=False,
        width=600,
        height=500,
        margin=dict(l=0, r=0, t=40, b=40),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    st.markdown('<div style="display:flex; justify-content:center;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=False, config={"displayModeBar": False, "staticPlot": True})
    st.markdown('</div>', unsafe_allow_html=True)

    # === DETAIL CARDS ===
    render_metric_feedback_cards(summary_row)

    # === ADDITIONAL GRAPHS ===
    st.markdown('<div class="section-title">📈 Additional Charts</div>', unsafe_allow_html=True)
    if frame_scores_df is not None and not frame_scores_df.empty:
        st.markdown('<div class="subsection-title">⏱ Frame-by-Frame Trends</div>', unsafe_allow_html=True)
        avg_frame_score = (
            float(frame_scores_df["average_score"].mean(skipna=True))
            if "average_score" in frame_scores_df
            else np.nan
        )
        if np.isfinite(avg_frame_score):
            st.metric("Average Frame Score (Key Metric)", f"{avg_frame_score:.1f} pts")
        st.plotly_chart(build_frame_chart(frame_scores_df), width="stretch")
        with st.expander("Show frame-by-frame scores"):
            st.dataframe(frame_scores_df)

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
                            column_map[col_name] = "Average Score"
                    if column_map:
                        display_df = display_df.rename(columns=column_map)
                    display_df = display_df.loc[:, ~display_df.columns.duplicated()]
                    # Removed bar chart for action phase average scores

                    def build_leg_radar(group_key: str) -> Optional[go.Figure]:
                        styles = LEG_RADAR_STYLES.get(group_key, [])
                        metric_keys = [
                            metric
                            for metric in SCORE_COLUMNS
                            if metric != "banzai_score" and f"{metric}_score" in action_means.columns
                        ]
                        if not metric_keys:
                            return None
                        metric_labels_group = [METRIC_LABELS.get(metric, metric) for metric in metric_keys]
                        if not metric_labels_group:
                            return None
                        labels_closed_group = metric_labels_group + [metric_labels_group[0]]
                        fig_action = go.Figure()
                        for phase_key, suffix, line_color, fill_color in styles:
                            if phase_key not in action_means.index:
                                continue
                            per_action_values = []
                            for metric in metric_keys:
                                column_name = f"{metric}_score"
                                if column_name in action_means.columns:
                                    per_action_values.append(float(action_means.loc[phase_key, column_name]))
                            if not per_action_values:
                                per_action_values = [0.0] * len(metric_keys)
                            if not per_action_values:
                                continue
                            values_closed = per_action_values + per_action_values[:1]
                            fig_action.add_trace(
                                go.Scatterpolar(
                                    r=values_closed,
                                    theta=labels_closed_group,
                                    fill="toself",
                                    name=f"{ACTION_LABELS.get(phase_key, phase_key)} {suffix}",
                                    line_color=line_color,
                                    fillcolor=fill_color,
                                    opacity=1.0,
                                )
                            )
                        if not fig_action.data:
                            return None
                        fig_action.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                            margin=dict(l=20, r=20, t=60, b=60),
                            height=360,
                        )
                        return fig_action

                    leg_radars = []
                    for group_key in ["right_leg", "left_leg"]:
                        radar_fig = build_leg_radar(group_key)
                        if radar_fig is not None:
                            leg_radars.append((group_key, radar_fig))
                    if leg_radars:
                        st.markdown('<div class="subsection-title">🦵 Left vs Right Leg Average Scores</div>', unsafe_allow_html=True)
                        cols = st.columns(len(leg_radars))
                        for col_slot, (group_key, radar_fig) in zip(cols, leg_radars):
                            with col_slot:
                                st.subheader(LEG_RADAR_TITLES.get(group_key, group_key))
                                st.plotly_chart(radar_fig, width="stretch")

    with st.expander("Show detailed score table"):
        st.dataframe(summary_table)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.get("frame_scores_csv") is not None:
            st.download_button(
                "💾 Save frame scores as CSV",
                data=st.session_state["frame_scores_csv"],
                file_name="frame_scores.csv",
                mime="text/csv",
            )
        if st.session_state.get("pose_csv_bytes") is not None:
            st.download_button(
                "💾 Save pose data as CSV",
                data=st.session_state["pose_csv_bytes"],
                file_name="pose_landmarks.csv",
                mime="text/csv",
            )
    with col2:
        st.button("🔁 Measure Again", on_click=reset_measurement_state)


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
