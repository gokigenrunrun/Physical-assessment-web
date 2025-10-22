# app.py

import streamlit as st
import pandas as pd
from calculate_metrics import calculate_metrics
import numpy as np

st.set_page_config(page_title="運動スコア自動採点アプリ", layout="centered")

st.title("💪 運動スコア自動採点アプリ")
st.write("CSVファイルをアップロードすると、自動でスコアリングを行います。")

# ====== スコア関数（さっきの使い回し） ======
def scale_score(value, min_val, max_val):
    if pd.isna(value):
        return np.nan
    if value >= max_val:
        return 0
    if value <= min_val:
        return 100
    return (1 - (value - min_val) / (max_val - min_val)) * 100


def score_csv(file_path):
    metrics = calculate_metrics(file_path)
    score_ranges = {
        "head_movement": (0.0001, 0.0088),
        "shoulder_tilt": (0.01, 0.1),
        "torso_tilt": (0.01, 0.1),
        "leg_lift": (0.1, 0.6),
        "foot_sway": (0.005, 0.1),
        "arm_sag": (0.01, 0.4)
    }

    scores = {}
    for key, value in metrics.items():
        if key in score_ranges:
            min_val, max_val = score_ranges[key]
            scores[f"{key}_score"] = scale_score(value, min_val, max_val)

    total = np.nanmean(list(scores.values()))
    # ファイル名を安全に取得
    file_name = getattr(file_path, "name", str(file_path))
    result = {"file_name": file_name, **metrics, **scores, "total_score": total}
    return pd.DataFrame([result])

# ====== ファイルアップロード ======
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    st.success("✅ ファイルを読み込みました！採点中です…")

    # CSVを一時保存
    temp_path = "uploaded_temp.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df_result = score_csv(temp_path)

    st.subheader("📊 採点結果")
    st.dataframe(df_result)

    st.metric(label="総合スコア（0〜100）", value=f"{df_result['total_score'].iloc[0]:.1f} 点")
else:
    st.info("👆 上のボックスにCSVファイルをアップロードしてください")
import plotly.graph_objects as go

# ===== レーダーチャート表示 =====
st.subheader("📊 各スコアのバランス（レーダーチャート）")

# 可視化するスコア項目
score_labels = ["head_movement", "shoulder_tilt", "torso_tilt", "leg_lift", "foot_sway", "arm_sag"]

# スコア値をリストに変換（0〜100）
values = [df_result[f"{label}_score"].values[0] for label in score_labels]

# レーダーチャートの形を閉じるために最初の値を最後にも追加
values += values[:1]
labels_closed = score_labels + [score_labels[0]]

# Plotlyで作図
fig = go.Figure(
    data=go.Scatterpolar(
        r=values,
        theta=labels_closed,
        fill="toself",
        line_color="#4A90E2",
        fillcolor="rgba(74, 144, 226, 0.3)",
        name="スコア"
    )
)

# 軸・スタイル設定
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            showline=True,
            linewidth=1,
            gridcolor="lightgray"
        ),
    ),
    showlegend=False,
    width=600,
    height=500,
)

# Streamlitで表示
st.plotly_chart(fig)    