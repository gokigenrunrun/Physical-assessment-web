import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from calculate_metrics import (
    SCORE_COLUMNS,
    calculate_metrics,
    calculate_metrics_by_frame,
    calculate_metrics_from_df,
    compare_motion_profiles,
    preprocess_landmarks,
)
from pose_extract import (
    LANDMARK_HEADER,
    draw_pose_landmarks,
    pose_capture_generator,
    video_to_pose_csv,
)

st.set_page_config(page_title="運動スコア自動採点アプリ", layout="centered")
st.title("💪 運動スコア自動採点アプリ")
st.write("CSV / 動画 / Webカメラのいずれかを入力すると、自動で骨格抽出とスコアリングを行います。")

SCORE_RANGES = {
    "head_movement": (0.0001, 0.0088),
    "shoulder_tilt": (0.01, 0.1),
    "torso_tilt": (0.01, 0.1),
    "leg_lift": (0.1, 0.6),
    "foot_sway": (0.005, 0.1),
    "arm_sag": (0.01, 0.4),
}

METRIC_LABELS = {
    "head_movement": "頭のブレ",
    "shoulder_tilt": "肩の傾き",
    "torso_tilt": "体幹の傾き",
    "leg_lift": "足上げ高さ",
    "foot_sway": "接地足の横ブレ",
    "arm_sag": "腕の垂れ下がり",
    "average_score": "平均スコア",
    "overall_similarity": "総合類似度",
}


def scale_score(value: float, min_val: float, max_val: float) -> float:
    if pd.isna(value):
        return np.nan
    if value >= max_val:
        return 0.0
    if value <= min_val:
        return 100.0
    return float((1 - (value - min_val) / (max_val - min_val)) * 100)


def score_data(source, label: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        metrics = calculate_metrics_from_df(source)
    else:
        metrics = calculate_metrics(source)
    scores = {}
    for key, (mn, mx) in SCORE_RANGES.items():
        value = metrics.get(key, np.nan)
        scores[f"{key}_score"] = scale_score(value, mn, mx)
    total = float(np.nanmean(list(scores.values()))) if scores else np.nan
    result = {"file_name": label, **metrics, **scores, "total_score": total}
    return pd.DataFrame([result])


def build_frame_score_table(frame_metrics: pd.DataFrame) -> pd.DataFrame:
    if frame_metrics is None or frame_metrics.empty:
        return pd.DataFrame()

    score_df = frame_metrics.copy()
    for key, (mn, mx) in SCORE_RANGES.items():
        if key in score_df.columns:
            score_df[f"{key}_score"] = score_df[key].apply(lambda v: scale_score(v, mn, mx))

    score_cols = [col for col in score_df.columns if col.endswith("_score")]
    if score_cols:
        score_df["average_score"] = score_df[score_cols].mean(axis=1, skipna=True)
    return score_df


def build_frame_chart(frame_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame_scores.empty:
        return fig

    x_values = frame_scores["frame"]
    for col in frame_scores.columns:
        if col == "frame":
            continue

        if col.endswith("_score"):
            base = col.replace("_score", "")
            label = f"{METRIC_LABELS.get(base, base)} (score)"
            fig.add_trace(go.Scatter(x=x_values, y=frame_scores[col], mode="lines", name=label))
        elif col == "average_score":
            fig.add_trace(go.Scatter(x=x_values, y=frame_scores[col], mode="lines", name=METRIC_LABELS[col], line=dict(width=3)))

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Frame",
        yaxis_title="Score (0-100)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
    )
    return fig


def build_similarity_dataframe(similarity_scores: dict) -> pd.DataFrame:
    if not similarity_scores:
        return pd.DataFrame()
    display_dict = {}
    for key, value in similarity_scores.items():
        label = METRIC_LABELS.get(key, key)
        display_dict[label] = value
    return pd.DataFrame([display_dict])


def load_pose_from_upload(uploaded_file, resize_scale: float = 1.0, frame_stride: int = 1) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if suffix == ".csv":
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file)
            return preprocess_landmarks(raw_df)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_video:
            tmp_video.write(uploaded_file.getbuffer())
            video_path = tmp_video.name

        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        try:
            tmp_csv_path = tmp_csv.name
        finally:
            tmp_csv.close()

        video_to_pose_csv(
            video_path=video_path,
            out_csv_path=tmp_csv_path,
            resize_scale=resize_scale,
            frame_stride=frame_stride,
        )

        df = pd.read_csv(tmp_csv_path)
        df = preprocess_landmarks(df)
        return df
    finally:
        # 後片付け（存在しない場合も静かに処理）
        if "video_path" in locals():
            Path(video_path).unlink(missing_ok=True)
        if "tmp_csv_path" in locals():
            Path(tmp_csv_path).unlink(missing_ok=True)


mode = st.radio(
    "入力方法を選択",
    ["CSV をアップロード", "動画から採点（自動変換）", "Webカメラでリアルタイム採点"],
)

df_result: Optional[pd.DataFrame] = None
pose_dataframe: Optional[pd.DataFrame] = None
frame_metrics_df: Optional[pd.DataFrame] = None
frame_scores_df: Optional[pd.DataFrame] = None
pose_csv_bytes: Optional[bytes] = None

if mode == "CSV をアップロード":
    uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("採点中です…"):
            try:
                pose_dataframe = load_pose_from_upload(uploaded_file)
                if pose_dataframe is None or pose_dataframe.empty:
                    st.warning("CSVから骨格データを読み取れませんでした。")
                else:
                    df_result = score_data(pose_dataframe, uploaded_file.name)
                    frame_metrics_df = calculate_metrics_by_frame(pose_dataframe)
                    frame_scores_df = build_frame_score_table(frame_metrics_df)
                    pose_csv_bytes = pose_dataframe.to_csv(index=False).encode("utf-8")
                    st.success("✅ 採点が完了しました！")
            except Exception as exc:
                st.error(f"❌ CSV処理に失敗しました: {exc}")

elif mode == "動画から採点（自動変換）":
    video_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi", "mkv"])
    col1, col2 = st.columns(2)
    resize_scale = col1.slider("縮小倍率（軽量化）", 0.3, 1.0, 0.7, 0.1)
    frame_stride = col2.slider("フレーム間引き", 1, 5, 1, 1, help="2以上にすると計算が速くなります")

    if video_file is not None:
        with st.spinner("🧠 動画から骨格を抽出しています…"):
            try:
                pose_dataframe = load_pose_from_upload(video_file, resize_scale=resize_scale, frame_stride=frame_stride)
                if pose_dataframe is None or pose_dataframe.empty:
                    st.warning("動画から骨格データを抽出できませんでした。")
                else:
                    df_result = score_data(pose_dataframe, video_file.name)
                    frame_metrics_df = calculate_metrics_by_frame(pose_dataframe)
                    frame_scores_df = build_frame_score_table(frame_metrics_df)
                    pose_csv_bytes = pose_dataframe.to_csv(index=False).encode("utf-8")
                    st.success("✅ 骨格抽出と採点が完了しました！")
            except Exception as exc:
                st.error(f"❌ 骨格抽出に失敗しました: {exc}")

elif mode == "Webカメラでリアルタイム採点":
    st.info("Webカメラから一定時間キャプチャして、リアルタイムで採点します。")
    col1, col2, col3 = st.columns(3)
    capture_seconds = col1.slider("計測時間（秒）", 3, 20, 8)
    frame_stride = col2.slider("フレーム間引き", 1, 5, 1, 1)
    resize_scale = col3.slider("縮小倍率（軽量化）", 0.4, 1.0, 0.7, 0.1)

    start_capture = st.button("計測スタート", type="primary")
    if start_capture:
        video_placeholder = st.empty()
        chart_placeholder = st.empty()
        status_placeholder = st.empty()

        rows = []
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Webカメラを初期化できませんでした。")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 15.0
            max_frames = int(fps * capture_seconds)
            update_every = max(1, int(fps // 3))
            processed = 0

            status_placeholder.info("計測を開始します…")
            start_time = time.time()

            for frame_idx, frame_bgr, landmarks in pose_capture_generator(
                cap=cap,
                resize_scale=resize_scale,
                frame_stride=frame_stride,
                max_frames=max_frames,
            ):
                annotated = draw_pose_landmarks(frame_bgr, landmarks)
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", caption=f"Frame {frame_idx}", use_column_width=True)

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

                processed += 1
                if processed % update_every == 0 and rows:
                    pose_dataframe = pd.DataFrame(rows, columns=LANDMARK_HEADER)
                    frame_metrics_df = calculate_metrics_by_frame(pose_dataframe)
                    frame_scores_df = build_frame_score_table(frame_metrics_df)
                    if not frame_scores_df.empty:
                        chart_placeholder.plotly_chart(build_frame_chart(frame_scores_df), use_container_width=True)

                remaining = max_frames - processed
                status_placeholder.info(f"計測中… {processed}/{max_frames} frame　残り {max(0, remaining)}")

            elapsed = time.time() - start_time
            status_placeholder.success(f"計測完了！{processed} frame（{elapsed:.1f} 秒）を解析しました。採点結果を集計中…")

        except Exception as exc:
            st.error(f"❌ リアルタイム計測に失敗しました: {exc}")
        finally:
            if cap is not None:
                cap.release()

        if rows:
            pose_dataframe = pd.DataFrame(rows, columns=LANDMARK_HEADER)
            df_result = score_data(pose_dataframe, "webcam_capture")
            frame_metrics_df = calculate_metrics_by_frame(pose_dataframe)
            frame_scores_df = build_frame_score_table(frame_metrics_df)
            pose_csv_bytes = pose_dataframe.to_csv(index=False).encode("utf-8")
        else:
            st.warning("骨格が検出できませんでした。照明や背景を調整して再試行してください。")


# ===== 結果表示 =====
if df_result is not None:
    st.subheader("📊 採点結果")
    st.dataframe(df_result, use_container_width=True)

    total_score = df_result["total_score"].iloc[0]
    st.metric("総合スコア（0〜100）", f"{total_score:.1f} 点")

    english_keys = SCORE_COLUMNS
    values = [
        float(np.nan_to_num(df_result.at[0, f"{k}_score"], nan=0.0))
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
        st.dataframe(frame_scores_df, use_container_width=True)

    if pose_csv_bytes:
        st.download_button(
            "骨格データをCSVでダウンロード",
            data=pose_csv_bytes,
            file_name="pose_landmarks.csv",
            mime="text/csv",
        )

    # ===== お手本比較 =====
    if frame_metrics_df is not None and not frame_metrics_df.empty:
        st.subheader("🥋 お手本データとの比較（試作）")
        with st.expander("お手本CSV / 動画をアップロード", expanded=False):
            reference_file = st.file_uploader(
                "お手本データを選択（任意）",
                type=["csv", "mp4", "mov", "avi", "mkv"],
                key="reference_uploader",
            )
            if reference_file is not None:
                with st.spinner("お手本データを処理しています…"):
                    try:
                        reference_pose_df = load_pose_from_upload(reference_file)
                        if reference_pose_df is None or reference_pose_df.empty:
                            st.warning("お手本データから骨格を抽出できませんでした。")
                        else:
                            reference_frame_metrics = calculate_metrics_by_frame(reference_pose_df)
                            if reference_frame_metrics.empty:
                                st.warning("お手本データにフレーム情報が含まれていません。")
                            else:
                                similarity_scores = compare_motion_profiles(frame_metrics_df, reference_frame_metrics)
                                similarity_df = build_similarity_dataframe(similarity_scores)
                                st.success("お手本との比較結果を算出しました。")
                                st.dataframe(similarity_df, use_container_width=True)
                    except Exception as exc:
                        st.error(f"お手本データの処理に失敗しました: {exc}")
else:
    st.info("👆 入力データをアップロードするか、Webカメラ計測を開始してください。")
