import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import base64
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
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
from violin_data import VIOLIN_DATA

plt.rcParams["font.family"] = [
    "Hiragino Sans",
    "IPAexGothic",
    "Noto Sans CJK JP",
    "Yu Gothic",
    "sans-serif",
]

st.set_page_config(page_title="運動機能評価支援システム", layout="centered")

REFERENCE_VIDEO_PATH = Path("otehon.mp4")
METRIC_LABELS = {
    "head_movement": "頭部の安定性",
    "shoulder_tilt": "肩の傾き",
    "torso_tilt": "体幹の傾き",
    "leg_lift": "脚上げの高さ",
    "foot_sway": "軸足のブレ",
    "arm_sag": "腕の保持",
    "banzai_score": "バンザイ姿勢",
    "average_score": "平均スコア",
}
SCORE_COLORS = {
    "バンザイ姿勢": "#1E3A8A",
    "頭部の安定性": "#EC4899",
    "肩の傾き": "#3B82F6",
    "体幹の傾き": "#0EA5E9",
    "腕の保持": "#F59E0B",
    "軸足のブレ": "#10B981",
    "脚上げの高さ": "#8B5CF6",
}
NEUTRAL_COLOR = "#9CA3AF"

ACTION_LABELS = {
    "right_leg_1": "右脚上げ（1回目）",
    "right_leg_2": "右脚上げ（2回目）",
    "left_leg_1": "左脚上げ（1回目）",
    "left_leg_2": "左脚上げ（2回目）",
}

LEG_PHASE_ORDER = ["right_leg_1", "left_leg_1", "right_leg_2", "left_leg_2"]
LEG_PHASE_GROUPS = {
    "right_leg": ["right_leg_1", "right_leg_2"],
    "left_leg": ["left_leg_1", "left_leg_2"],
}
LEG_GROUP_LABELS = {
    "right_leg": "右脚平均",
    "left_leg": "左脚平均",
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
        ("right_leg_1", "1回目", ATTEMPT_COLOR_PINK, ATTEMPT_FILL_PINK),
        ("right_leg_2", "2回目", ATTEMPT_COLOR_BLUE, ATTEMPT_FILL_BLUE),
    ],
    "left_leg": [
        ("left_leg_1", "1回目", ATTEMPT_COLOR_PINK, ATTEMPT_FILL_PINK),
        ("left_leg_2", "2回目", ATTEMPT_COLOR_BLUE, ATTEMPT_FILL_BLUE),
    ],
}
LEG_RADAR_TITLES = {
    "right_leg": "右脚上げ（1回目・2回目）",
    "left_leg": "左脚上げ（1回目・2回目）",
}
SCORE_TIER_RULES = [
    (85, "#2ECC71", "とても良い", "非常に安定した動きです。"),
    (70, "#2979FF", "良い", "動作全体を通してバランスがとれています。"),
    (55, "#FFB300", "まずまず", "全体の安定性をさらに磨きましょう。"),
    (0, "#FF4081", "要改善", "フォーム改善の余地があります。"),
]
DEFAULT_SCORE_COLOR = "#FF4081"
DEFAULT_SCORE_LABEL = "スコアなし"
DEFAULT_SCORE_MESSAGE = "測定に必要なデータが不足しています。"




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
        "high": "頭がほとんど揺れず安定しています。",
        "mid": "頭の揺れは小さいですが、引き続き安定性を意識しましょう。",
        "low": "動作中の頭の揺れを抑えることに集中しましょう。",
    },
    "shoulder_tilt": {
        "high": "肩のラインが水平に保たれています。",
        "mid": "肩の傾きは許容範囲ですが、さらに安定させましょう。",
        "low": "両肩を同じ高さに保つよう意識しましょう。",
    },
    "torso_tilt": {
        "high": "動作全体で体幹がまっすぐ保たれています。",
        "mid": "体幹は概ね安定しています。呼吸を合わせて揺れを減らしましょう。",
        "low": "上半身が揺れているので体幹を意識して引き締めましょう。",
    },
    "leg_lift": {
        "high": "脚上げの高さは十分です。",
        "mid": "もう少し高く上げるとスコアが伸びます。",
        "low": "ひざをさらに高く持ち上げて動きを強調しましょう。",
    },
    "foot_sway": {
        "high": "軸足がしっかり安定しています。",
        "mid": "軸足は概ね安定しています。体重の位置を一定に保ちましょう。",
        "low": "接地している足が揺れています。軸を安定させましょう。",
    },
    "arm_sag": {
        "high": "動作全体で腕がしっかり持ち上がっています。",
        "mid": "腕は保てていますが、肩を意識してもう少し高く上げましょう。",
        "low": "腕がすぐに下がってしまいます。ひじを持ち上げたまま意識しましょう。",
    },
    "banzai_score": {
        "high": "バンザイ姿勢が明確に保たれています。",
        "mid": "概ね良いバンザイ姿勢です。最後まで意識してキメましょう。",
        "low": "肩と腕をしっかり伸ばして姿勢を保ちましょう。",
    },
}
DEFAULT_FEEDBACK_TEMPLATE = {
    "high": "動きは安定しバランスが取れています。",
    "mid": "動作中も常に安定性を意識しましょう。",
    "low": "改善ポイントを意識してフォームを整えましょう。",
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
    "banzai_score": "バンザイ姿勢",
    "head_movement": "頭部の安定性",
    "shoulder_tilt": "肩の傾き",
    "torso_tilt": "体幹の傾き",
    "leg_lift": "脚上げの高さ",
    "foot_sway": "軸足のブレ",
    "arm_sag": "腕の保持",
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
        return "データが不足しているため評価できません。"
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
        score_text = f"{score:.1f} 点"
    else:
        label_bg, label_border, label_color = "#E5E7EB", "#9CA3AF", "#374151"
        score_text = "--"
        label = label or "スコアなし"
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
    score_text = "-- 点" if not np.isfinite(display_value) else f"{display_value:.1f} 点"
    raw_comment = (comment or "").strip()
    is_banzai = title == "バンザイ姿勢"
    if is_banzai:
        detail_lines = [raw_comment] if raw_comment else []
        detail_lines.extend(
            [
                "両腕の角度をそろえ、肩を引いて安定させましょう。",
                "頭と体幹を一直線に保ちましょう。",
            ]
        )
        comment_text = "\n".join(line for line in detail_lines if line)
        if not comment_text:
            comment_text = "バンザイ姿勢のスコアデータがありません。"
        comment_class = "metric-comment long-comment"
        card_modifier = " long"
        comment_html = escape(comment_text)
    else:
        comment_class = "metric-comment"
        card_modifier = ""
        comment_text = raw_comment or "データが不足しているため評価できません。"
        comment_html = escape(comment_text).replace("\n", "<br />")
    return (
        f'<div class="metric-card{card_modifier}">'
        f'\n    <div class="metric-title">{title_html}</div>'
        f'\n    <div class="metric-content">'
        f'\n        <div class="metric-left">'
        f'\n            <img class="metric-chart" src="data:image/png;base64,{encoded_image}" alt="{title_html} のスコアチャート" />'
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
            label = f"{METRIC_LABELS.get(base, base)}（スコア）"
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
        xaxis_title="フレーム",
        yaxis_title="スコア（0-100）",
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
            name="背景：右脚上げフェーズ",
            legendrank=1000,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="lightpink", width=14),
            name="背景：左脚上げフェーズ",
            legendrank=1001,
        )
    )
    return fig


def render_head_violin_plot(violin_data: Optional[Dict[str, Any]]) -> None:
    data = violin_data or VIOLIN_DATA.get("head_stability", {})
    st.markdown(
        '<div class="subsection-title">🎻 頭部安定性の分布（右＝上・左＝下）</div>',
        unsafe_allow_html=True,
    )
    if not data:
        st.info("頭部安定性の分布データがありません。")
        return

    right_samples = np.asarray(data.get("right_population", []), dtype=float)
    left_samples = np.asarray(data.get("left_population", []), dtype=float)
    right_samples = right_samples[np.isfinite(right_samples)]
    left_samples = left_samples[np.isfinite(left_samples)]

    if right_samples.size < 2 or left_samples.size < 2:
        st.info("データ点が不足しているためバイオリンプロットを描画できません。")
        return

    right_p95 = data.get("right_p95")
    left_p95 = data.get("left_p95")
    xmin = 0.0
    xmax_candidates = [v for v in [right_p95, left_p95] if isinstance(v, (int, float))]
    xmax = max(xmax_candidates) if xmax_candidates else 0.05
    xs = np.linspace(xmin, xmax, 400)

    def _normalized_density(samples: np.ndarray) -> Optional[np.ndarray]:
        if np.allclose(samples, samples[0]):
            return None
        kde = gaussian_kde(samples)
        density = kde(xs)
        max_val = float(np.max(density))
        if max_val <= 0:
            return None
        return density / max_val

    density_right = _normalized_density(right_samples)
    density_left = _normalized_density(left_samples)

    if density_right is None or density_left is None:
        st.info("現在の頭部安定性データではKDEを計算できません。")
        return

    import matplotlib.font_manager as fm
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "IPAexGothic",
        "Noto Sans CJK JP",
        "Hiragino Sans",
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(xs, 0, density_right, color="#FF69B4", alpha=0.6)
    ax.fill_between(xs, 0, -density_left, color="#007BFF", alpha=0.6)

    def _safe_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
            if np.isfinite(parsed):
                return parsed
            return None
        except (TypeError, ValueError):
            return None

    user_right = _safe_float(data.get("user_right"))
    user_left = _safe_float(data.get("user_left"))
    if user_right is not None:
        ax.axvline(user_right, color="#FF69B4", linestyle="--", linewidth=2)
    if user_left is not None:
        ax.axvline(user_left, color="#007BFF", linestyle="--", linewidth=2)

    max_density = float(max(np.max(density_right), np.max(density_left)))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.05 * max_density, 1.05 * max_density)
    ax.set_yticks([0.3, -0.3])
    ax.set_yticklabels(["右足", "左足"])
    ax.set_xlabel("頭部の動き")
    ax.set_ylabel("")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("頭部安定性の分布（右脚＝上・左脚＝下）")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _prepare_population(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _normalized_density(samples: np.ndarray, xs: np.ndarray) -> Optional[np.ndarray]:
    if samples.size < 2:
        return None
    if np.allclose(samples, samples[0]):
        return None
    kde = gaussian_kde(samples)
    density = kde(xs)
    max_val = float(np.max(density))
    if max_val <= 0:
        return None
    return density / max_val


RIGHT_LEG_PHASES = ["right_leg_1", "right_leg_2"]
LEFT_LEG_PHASES = ["left_leg_1", "left_leg_2"]


def compute_side_metric_series(
    frame_scores_df: Optional[pd.DataFrame],
    metric_names: Iterable[str],
) -> pd.Series:
    if frame_scores_df is None or frame_scores_df.empty or "action" not in frame_scores_df.columns:
        return pd.Series(dtype=float)

    side_map = {
        "right": RIGHT_LEG_PHASES,
        "left": LEFT_LEG_PHASES,
    }
    data: Dict[str, float] = {}
    for metric in metric_names:
        if metric not in frame_scores_df.columns:
            continue
        for side_key, actions in side_map.items():
            subset = frame_scores_df[frame_scores_df["action"].isin(actions)]
            if subset.empty:
                value = np.nan
            else:
                metric_series = subset[metric]
                value = float(metric_series.mean(skipna=True)) if metric_series.notna().any() else np.nan
            data[f"{metric}_{side_key}"] = value
    return pd.Series(data, dtype=float)


def draw_violin_mirror(
    metric_name: str,
    data_dict: Dict[str, Any],
    user_right: float,
    user_left: float,
) -> None:
    if not data_dict:
        return

    right_samples = _prepare_population(data_dict.get("right_population", []))
    left_samples = _prepare_population(data_dict.get("left_population", []))
    if right_samples.size == 0 and left_samples.size == 0:
        st.info(f"{metric_name} の母集団データがありません。")
        return

    combined_values: List[float] = []
    for arr in (right_samples, left_samples):
        if arr.size:
            combined_values.extend([float(np.min(arr)), float(np.max(arr))])
    if not combined_values:
        combined_values = [0.0, 1.0]

    xmin = float(np.min(combined_values))
    xmax = float(np.max(combined_values))
    if np.isclose(xmin, xmax):
        xmax = xmin + 1e-6
    xs = np.linspace(xmin, xmax, 400)

    density_right = _normalized_density(right_samples, xs)
    density_left = _normalized_density(left_samples, xs)
    if density_right is None and density_left is None:
        st.info(f"{metric_name} のバイオリンプロットを計算できません。")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    if density_right is not None:
        ax.fill_between(xs, 0, density_right, color=ATTEMPT_COLOR_PINK, alpha=0.65)
    if density_left is not None:
        ax.fill_between(xs, 0, -density_left, color=ATTEMPT_COLOR_BLUE, alpha=0.65)

    def _draw_user_line(value: Any, color: str) -> None:
        if value is None:
            return
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(val):
            ax.axvline(val, color=color, linestyle="-", linewidth=2.2)

    _draw_user_line(data_dict.get("user_right"), ATTEMPT_COLOR_PINK)
    _draw_user_line(data_dict.get("user_left"), ATTEMPT_COLOR_BLUE)

    max_density = max(
        float(np.max(density_right)) if density_right is not None else 0.0,
        float(np.max(density_left)) if density_left is not None else 0.0,
    )
    if max_density <= 0:
        max_density = 1.0
    if np.isfinite(user_right):
        ax.vlines(user_right, ymin=0, ymax=max_density, color=ATTEMPT_COLOR_PINK, linewidth=2.0, zorder=5)
    if np.isfinite(user_left):
        ax.vlines(user_left, ymin=0, ymax=-max_density, color=ATTEMPT_COLOR_BLUE, linewidth=2.0, zorder=5)
    ax.axhline(0, color="#111827", linewidth=0.6)
    ax.set_xlim(xmin, xmax)
    if metric_name == "shoulder_tilt":
        left_label, right_label = "傾きが小さい", "傾きが大きい"
    elif metric_name == "torso_tilt":
        left_label, right_label = "傾きが小さい", "傾きが大きい"
    elif metric_name == "leg_lift":
        left_label, right_label = "高い", "低い"
    elif metric_name == "foot_sway":
        left_label, right_label = "安定", "不安定"
    elif metric_name == "arm_sag":
        left_label, right_label = "傾きが小さい", "傾きが大きい"
    elif metric_name == "head_movement":
        left_label, right_label = "安定", "不安定"
    else:
        left_label, right_label = f"{xmin:.3f}", f"{xmax:.3f}"
    ax.set_xticks([xmin, xmax])
    ax.set_xticklabels([left_label, right_label])
    ax.set_ylim(-1.05 * max_density, 1.05 * max_density)
    ax.set_yticks([])
    ax.set_xlabel("値")
    ax.set_ylabel("")
    ax.set_title(METRIC_LABELS.get(metric_name, metric_name))
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def draw_all_violin_plots(
    violin_dataset: Dict[str, Dict[str, Any]],
    summary_row: pd.Series,
) -> None:
    if not violin_dataset:
        st.info("バイオリンプロット用の母集団データがありません。")
        return
    st.markdown('<div class="subsection-title">🎯 指標の分布（バイオリンプロット）</div>', unsafe_allow_html=True)
    for metric_name, metric_data in violin_dataset.items():
        if metric_name == "head_stability":
            continue
        if summary_row is None:
            user_right = np.nan
            user_left = np.nan
        else:
            user_right = float(summary_row.get(f"{metric_name}_right", np.nan))
            user_left = float(summary_row.get(f"{metric_name}_left", np.nan))
        draw_violin_mirror(metric_name, metric_data, user_right, user_left)


def plot_violin(metric_name: str, data: Dict[str, Any]) -> None:
    """Render a mirrored violin identical to the head-stability style."""
    right_samples = np.asarray(data.get("right_population", []), dtype=float)
    left_samples = np.asarray(data.get("left_population", []), dtype=float)
    right_samples = right_samples[np.isfinite(right_samples)]
    left_samples = left_samples[np.isfinite(left_samples)]

    if right_samples.size == 0 and left_samples.size == 0:
        print(f"{metric_name}: 有効なデータがありません。")
        return

    xmax_candidates = [
        val
        for val in (data.get("right_p95"), data.get("left_p95"))
        if isinstance(val, (int, float)) and np.isfinite(val)
    ]
    if xmax_candidates:
        xmax = float(max(xmax_candidates))
    else:
        combined = np.concatenate([arr for arr in (right_samples, left_samples) if arr.size])
        xmax = float(np.max(combined)) if combined.size else 1.0
    xmin = 0.0
    if xmax <= xmin:
        xmax = xmin + 1e-6
    xs = np.linspace(xmin, xmax, 400)

    def _kde(samples: np.ndarray) -> Optional[np.ndarray]:
        if samples.size < 2 or np.allclose(samples, samples[0]):
            return None
        kde = gaussian_kde(samples)
        density = kde(xs)
        max_val = float(np.max(density))
        return density / max_val if max_val > 0 else None

    density_right = _kde(right_samples)
    density_left = _kde(left_samples)
    if density_right is None and density_left is None:
        print(f"{metric_name}: KDE を計算できません。")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    if density_right is not None:
        ax.fill_between(xs, 0, density_right, color=ATTEMPT_COLOR_PINK, alpha=0.65)
    if density_left is not None:
        ax.fill_between(xs, 0, -density_left, color=ATTEMPT_COLOR_BLUE, alpha=0.65)

    def _draw_user_line(value: Any, color: str) -> None:
        if value is None:
            return
        try:
            val = float(value)
        except (TypeError, ValueError):
            return
        if np.isfinite(val):
            ax.axvline(val, color=color, linestyle="-", linewidth=2.2)

    _draw_user_line(data.get("user_right"), ATTEMPT_COLOR_PINK)
    _draw_user_line(data.get("user_left"), ATTEMPT_COLOR_BLUE)

    max_density = max(
        float(np.max(density_right)) if density_right is not None else 0.0,
        float(np.max(density_left)) if density_left is not None else 0.0,
    )
    if max_density <= 0:
        max_density = 1.0

    ax.axhline(0, color="#111111", linewidth=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.05 * max_density, 1.05 * max_density)
    ax.set_yticks([])
    ax.set_xlabel("値")
    ax.set_ylabel("")
    ax.set_title(metric_name)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def render_metric_feedback_cards(result_row: pd.Series, violin_data: Optional[Dict[str, Any]] = None) -> None:
    st.markdown('<div class="section-title">🧩 詳細指標</div>', unsafe_allow_html=True)
    card_order = [
        ("banzai_score", "バンザイ姿勢"),
        ("head_movement", "頭部の安定性"),
        ("shoulder_tilt", "肩の傾き"),
        ("torso_tilt", "体幹の傾き"),
        ("arm_sag", "腕の保持"),
        ("foot_sway", "軸足のブレ"),
        ("leg_lift", "脚上げの高さ"),
    ]
    if not any(f"{key}_score" in result_row.index for key, _ in card_order):
        st.info("指標スコアがまだ計算されていません。")
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
        banzai=card_html("banzai_score", "バンザイ姿勢"),
        head=card_html("head_movement", "頭部の安定性"),
        shoulder=card_html("shoulder_tilt", "肩の傾き"),
        torso=card_html("torso_tilt", "体幹の傾き"),
        arm=card_html("arm_sag", "腕の保持"),
        foot=card_html("foot_sway", "軸足のブレ"),
        leg=card_html("leg_lift", "脚上げの高さ"),
    )
    st.markdown(cards_html, unsafe_allow_html=True)
    draw_all_violin_plots(VIOLIN_DATA, result_row)


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
            model_asset_path="pose_landmarker_lite.task",
        )
        raw_df, start_ts = raw_df
        pose_df = preprocess_landmarks(raw_df)
    elif mode == "csv":
        pose_df = preprocess_landmarks(config["dataframe"])
    else:
        raise ValueError(f"未対応のモード: {mode}")

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
    st.title("💪 運動機能評価支援システム")
    st.markdown("新しい測定を始めるにはスタートボタンを押してください。")

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
            st.warning(f"カメラのウォームアップに失敗しました：{exc}")
    elif st.session_state.get("camera_warmed"):
        st.caption("📸 カメラの準備ができています。")

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
        video_file = st.file_uploader("動画ファイルを選択（mp4 / mov / avi / mkv）", type=["mp4", "mov", "avi", "mkv"])
        col1, col2 = st.columns(2)
        resize_scale = col1.slider("リサイズ倍率（処理を軽くする）", 0.3, 1.0, 0.7, 0.1)
        frame_stride = col2.slider("フレーム間引き", 1, 5, 1, 1)
    else:
        col1, col2, col3 = st.columns(3)
        default_capture = max(3, int(round(REFERENCE_DURATION_SECONDS)))
        slider_max = max(default_capture, 20)
        capture_seconds = col1.slider("撮影時間（秒）", 3, slider_max, default_capture)
        frame_stride = col2.slider("フレーム間引き", 1, 5, 1, 1)
        resize_scale = col3.slider("リサイズ倍率（処理を軽くする）", 0.4, 1.0, 0.7, 0.1)

    csv_debug_df = None
    csv_debug_file = None
    with st.expander("🔧 エキスパートモード（CSVデバッグ）"):
        csv_debug_file = st.file_uploader("骨格CSVを直接アップロード", type=["csv"], key="csv_debug_uploader")
        if csv_debug_file is not None:
            try:
                csv_debug_file.seek(0)
                csv_debug_df = pd.read_csv(csv_debug_file)
                st.success("CSVを読み込みました。")
            except Exception as exc:
                st.error(f"CSVを読み込めませんでした：{exc}")
                csv_debug_df = None

    start_disabled = bool(st.session_state.get("measurement_ready")) or st.session_state.get("countdown_active", False)
    if st.button("🟢 測定を開始", type="primary", disabled=start_disabled):
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
        st.header("🎬 測定の準備中...")
        message_placeholder = st.empty()
        countdown_placeholder = st.empty()
        message_placeholder.info("🎬 測定が始まるまでお待ちください...")
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
                <div style="font-size:6rem; font-weight:700; color:#43AA8B; line-height:1;">スタート！</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.3)
        st.session_state["countdown_active"] = False
        st.session_state["measurement_ready"] = True
        st.rerun()
        return

    st.header("🏃‍♀️ 測定中...")
    col1, col2 = st.columns([1, 1])
    reference_video_placeholder = None
    with col1:
        st.subheader("参考映像")
        reference_video_placeholder = st.empty()
        if not st.session_state.get("measurement_ready"):
            reference_video_placeholder.info("カウントダウンが終わるとお手本動画が再生されます。")
    live_placeholder = None
    with col2:
        st.subheader("あなたの動き")
        if config["mode"] == "video":
            st.video(config["video_path"])
        elif config["mode"] == "webcam":
            live_placeholder = st.empty()
            live_placeholder.info("Webカメラを初期化しています...")
        else:
            st.info("CSVデータを解析中です...")

    phase_placeholder = st.empty()
    if config["mode"] != "webcam":
        phase_placeholder.markdown("**🏃‍♀️ 測定中：処理しています...**")

    st.markdown("### 🏃‍♀️ 測定を実行しています...")
    st.caption("解析が完了すると結果画面へ移動します。")

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
                caption=f"フレーム {frame_idx}",
            )
            action_key = classify_action(frame_idx)
            phase_label = ACTION_LABELS.get(action_key, "動作中")
            phase_placeholder.markdown(f"**🏃‍♀️ 測定中：{phase_label}**")

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
            with st.spinner("解析中です..."):
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
    st.header("🧠 解析中...")
    st.info("まもなく結果が表示されます。")
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

    st.header("🕺 バンザイ評価テスト")
    st.markdown("選択したフォルダー内の全CSVでバンザイスコアを計算します。")

    folder = st.text_input("フォルダーパスを入力", "data_banzai_landmarks")
    if st.button("バンザイ評価を実行"):
        folder_path = Path(folder)
        if not folder_path.exists():
            st.error(f"❌ フォルダーが見つかりません：{folder}")
        else:
            with st.spinner("すべてのCSVを評価中..."):
                df = batch_evaluate_banzai(folder_path)
                if df.empty:
                    st.warning("有効なCSVファイルが見つかりません。")
                else:
                    st.success("✅ 評価が完了しました！")
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
        st.info("スコアデータが見つかりません。再度測定してください。")
        return

    summary_row = summary_table.iloc[0].copy()
    side_metric_series = compute_side_metric_series(
        st.session_state.get("frame_scores_df"),
        VIOLIN_DATA.keys(),
    )
    if not side_metric_series.empty:
        for key, value in side_metric_series.items():
            summary_row[key] = value
    total_score = float(summary_row.get("total_score", np.nan))
    tier_color, tier_label, tier_message = describe_total_score(total_score)
    # === HEADER: TOTAL SCORE ===
    render_score_block(total_score, tier_label, tier_message)

    # === RADAR CHART ===
    st.markdown('<div class="section-title">📊 モーションプロフィール</div>', unsafe_allow_html=True)
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
    render_metric_feedback_cards(summary_row, st.session_state.get("violin_data"))

    # === ADDITIONAL GRAPHS ===
    SHOW_ADDITIONAL_CHARTS = False
    if SHOW_ADDITIONAL_CHARTS:
        st.markdown('<div class="section-title">📈 追加のチャート</div>', unsafe_allow_html=True)
        if frame_scores_df is not None and not frame_scores_df.empty:
            st.markdown('<div class="subsection-title">⏱ フレームごとの推移</div>', unsafe_allow_html=True)
            avg_frame_score = (
                float(frame_scores_df["average_score"].mean(skipna=True))
                if "average_score" in frame_scores_df
                else np.nan
            )
            if np.isfinite(avg_frame_score):
                st.metric("フレーム平均スコア（主要指標）", f"{avg_frame_score:.1f} 点")
            st.plotly_chart(build_frame_chart(frame_scores_df), width="stretch")
            with st.expander("フレームごとのスコアを表示"):
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
                                column_map[col_name] = f"{METRIC_LABELS.get(metric_key, metric_key)}（スコア）"
                            elif col_name == "average_score":
                                column_map[col_name] = "平均スコア"
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
                                legend_label = ACTION_LABELS.get(phase_key, phase_key)
                                if suffix and suffix not in legend_label:
                                    legend_label = f"{legend_label} {suffix}"
                                fig_action.add_trace(
                                    go.Scatterpolar(
                                        r=values_closed,
                                        theta=labels_closed_group,
                                        fill="toself",
                                        name=legend_label,
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
                            st.markdown('<div class="subsection-title">🦵 左右脚の平均スコア比較</div>', unsafe_allow_html=True)
                            cols = st.columns(len(leg_radars))
                            for col_slot, (group_key, radar_fig) in zip(cols, leg_radars):
                                with col_slot:
                                    st.subheader(LEG_RADAR_TITLES.get(group_key, group_key))
                                    st.plotly_chart(radar_fig, width="stretch")

    with st.expander("詳細スコア表を表示"):
        st.dataframe(summary_table)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.get("frame_scores_csv") is not None:
            st.download_button(
                "💾 フレームスコアをCSVで保存",
                data=st.session_state["frame_scores_csv"],
                file_name="frame_scores.csv",
                mime="text/csv",
            )
        if st.session_state.get("pose_csv_bytes") is not None:
            st.download_button(
                "💾 ポーズデータをCSVで保存",
                data=st.session_state["pose_csv_bytes"],
                file_name="pose_landmarks.csv",
                mime="text/csv",
            )
    with col2:
        st.button("🔁 もう一度測定する", on_click=reset_measurement_state)


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
