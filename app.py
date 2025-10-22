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
    result = {"file_name": file_path.name, **metrics, **scores, "total_score": total}
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