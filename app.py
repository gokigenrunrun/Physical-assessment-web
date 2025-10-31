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

from calculate_metrics import (
    SCORE_COLUMNS,
    calculate_metrics_by_frame,
    calculate_metrics_from_df,
    classify_action,
    get_score_range,
    preprocess_landmarks,
)
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
    "average_score": "平均スコア",
}

ACTION_LABELS = {
    "right_leg": "右足上げ",
    "left_leg": "左足上げ",
    "raise": "両腕上げ",
}

DEFAULT_DISPLAY_ASPECT_RATIO = 3 / 4  # width / height
DEFAULT_DISPLAY_HEIGHT = 720
DEFAULT_CAPTURE_SECONDS = 12


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


@st.cache_data(show_spinner=False)
def load_reference_video_payload(path: Path) -> Optional[tuple[str, str]]:
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        suffix = path.suffix.lower()
        mime = "video/mp4"
        if suffix == ".mov":
            mime = "video/quicktime"
        elif suffix == ".webm":
            mime = "video/webm"
        return b64, mime
    except Exception:
        return None


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
        "countdown_start": None,
        "countdown_duration": 3,
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
    st.session_state["countdown_start"] = None
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
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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


def build_frame_chart(frame_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame_scores.empty or "frame" not in frame_scores.columns:
        return fig
    x_values = frame_scores["frame"]
    for col in frame_scores.columns:
        if col in {"frame", "action"}:
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
                    mode="lines",
                    name="平均スコア",
                    line=dict(width=3),
                )
            )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Frame",
        yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
    )
    return fig


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
    result_df, frame_scores_df = score_data(pose_df, label, frame_metrics_df)

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

    if st.session_state.get("countdown_active", False):
        measurement_config = st.session_state.get("measurement_config")
        if measurement_config is None:
            st.session_state["countdown_active"] = False
            st.session_state["countdown_start"] = None
            return

        duration = max(1, int(st.session_state.get("countdown_duration", 3)))

        placeholder = st.empty()
        subtitle = st.empty()
        subtitle.markdown("**姿勢を整えてください…**")
        for value in range(duration, 0, -1):
            placeholder.markdown(
                f"""
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:70vh;">
                    <div style=\"font-size:9rem; font-weight:700; color:#1C6DD0; line-height:1;\">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(1)
        placeholder.markdown(
            """
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:70vh;">
                <div style="font-size:7rem; font-weight:700; color:#1C6DD0; line-height:1;">スタート!</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.3)
        st.session_state["countdown_active"] = False
        st.session_state["countdown_start"] = None
        st.session_state["measurement_ready"] = True
        st.session_state["page"] = "measuring"
        st.rerun()
        return

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
        st.session_state["countdown_start"] = time.time()
        st.session_state["camera_warmed"] = False
        st.session_state["page"] = "start"
        st.rerun()


def render_measuring_view() -> None:
    config = st.session_state.get("measurement_config")
    if not config:
        reset_measurement_state()
        st.rerun()
        return

    st.header("🏃‍♀️ 計測中…")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("お手本")
        if REFERENCE_VIDEO_PATH.exists():
            payload = load_reference_video_payload(REFERENCE_VIDEO_PATH)
            if payload:
                b64, mime = payload
                st.markdown(
                    f"""
                    <div style="width:100%; max-width:{DISPLAY_WIDTH}px; margin:auto;">
                        <video autoplay loop muted playsinline style="width:100%; height:auto; aspect-ratio:{DISPLAY_ASPECT_RATIO}; border-radius:12px; background:#000;">
                            <source src="data:{mime};base64,{b64}" type="{mime}">
                        </video>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.video(str(REFERENCE_VIDEO_PATH))
        else:
            st.info("お手本動画が見つかりません。")
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

    if st.session_state.get("measurement_ready"):
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


def render_result_view() -> None:
    result_df = st.session_state.get("result_df")
    frame_scores_df = st.session_state.get("frame_scores_df")
    if result_df is None:
        reset_measurement_state()
        st.rerun()
        return

    st.header("📊 採点結果")
    st.dataframe(result_df, use_container_width=True)

    total_score = result_df["total_score"].iloc[0]
    st.metric("総合スコア（0〜100）", f"{total_score:.1f} 点")

    english_keys = SCORE_COLUMNS
    values = [
        float(np.nan_to_num(result_df.at[0, f"{k}_score"], nan=0.0))
        for k in english_keys
    ]
    labels_closed = english_keys + [english_keys[0]]
    radar_values = values + values[:1]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=radar_values,
            theta=labels_closed,
            fill="toself",
            line_color="#4A90E2",
            fillcolor="rgba(74,144,226,0.3)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        width=640,
        height=520,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    if frame_scores_df is not None and not frame_scores_df.empty:
        st.subheader("🕒 フレームごとのスコア推移")
        st.plotly_chart(build_frame_chart(frame_scores_df), use_container_width=True)
        with st.expander("フレーム別スコアを表示"):
            st.dataframe(frame_scores_df, use_container_width=True)

        if "action" in frame_scores_df.columns:
            score_cols = [col for col in frame_scores_df.columns if col.endswith("_score")]
            include_avg = "average_score" in frame_scores_df.columns
            if score_cols:
                group_cols = score_cols + (["average_score"] if include_avg else [])
                action_means = frame_scores_df.groupby("action")[group_cols].mean().round(1)
                if not action_means.empty:
                    display_df = action_means.rename(index=lambda k: ACTION_LABELS.get(k, k))
                    column_map = {}
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

                    action_order = list(action_means.index)
                    radar_cols = st.columns(len(action_order)) if action_order else []
                    for col_slot, action_key in zip(radar_cols, action_order):
                        with col_slot:
                            per_action_values = [
                                float(action_means.loc[action_key, f"{metric}_score"])
                                for metric in SCORE_COLUMNS
                                if f"{metric}_score" in action_means.columns
                            ]
                            if not per_action_values:
                                continue
                            per_action_labels = [
                                METRIC_LABELS.get(metric, metric)
                                for metric in SCORE_COLUMNS
                                if f"{metric}_score" in action_means.columns
                            ]
                            labels_closed = per_action_labels + [per_action_labels[0]]
                            values_closed = per_action_values + per_action_values[:1]
                            fig_action = go.Figure(
                                data=go.Scatterpolar(
                                    r=values_closed,
                                    theta=labels_closed,
                                    fill="toself",
                                    name=ACTION_LABELS.get(action_key, action_key),
                                )
                            )
                            fig_action.update_layout(
                                title=dict(text=ACTION_LABELS.get(action_key, action_key), x=0.5, font=dict(size=16)),
                                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                showlegend=False,
                                margin=dict(l=20, r=20, t=60, b=20),
                                height=340,
                            )
                            st.plotly_chart(fig_action, use_container_width=True)

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
