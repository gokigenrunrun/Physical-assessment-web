import streamlit as st
import pandas as pd
from calculate_metrics import calculate_metrics
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# 🔧 Streamlit CloudでPlotlyが表示されない問題の対策
pio.renderers.default = "iframe"

# ===== ページ設定 =====
st.set_page_config(page_title="運動スコア自動採点アプリ", layout="centered")

# ===== タイトル・説明 =====
st.title("💪 運動スコア自動採点アプリ")
st.write("CSVファイルをアップロードすると、自動でスコアリングを行います。")

# ====== スコア関数 ======
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
    file_name = getattr(file_path, "name", str(file_path))
    result = {"file_name": file_name, **metrics, **scores, "total_score": total}
    return pd.DataFrame([result])


# ====== ファイルアップロード ======
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    st.success("✅ ファイルを読み込みました！採点中です…")

    # 一時保存して読み込み
    temp_path = "uploaded_temp.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df_result = score_csv(temp_path)

    # ===== 採点結果表示 =====
    st.subheader("📊 採点結果")
    st.dataframe(df_result)
    st.metric(label="総合スコア（0〜100）", value=f"{df_result['total_score'].iloc[0]:.1f} 点")

    # ===== レーダーチャート（英語ラベル） =====
    if df_result is not None and len(df_result) > 0:
        st.subheader("📈 各スコアのバランス（Radar Chart）")

        try:
            # レーダーチャート用ラベル（英語表記）
            score_labels = ["Head", "Shoulder", "Torso", "Leg Lift", "Foot Sway", "Arm Sag"]
            english_keys = ["head_movement", "shoulder_tilt", "torso_tilt", "leg_lift", "foot_sway", "arm_sag"]

            # 値を取得
            values = [float(df_result[f"{key}_score"].values[0]) for key in english_keys]
            values += values[:1]  # 円を閉じる
            labels_closed = score_labels + [score_labels[0]]

            # Plotlyで描画
            fig = go.Figure(
                data=go.Scatterpolar(
                    r=values,
                    theta=labels_closed,
                    fill="toself",
                    line_color="#4A90E2",
                    fillcolor="rgba(74, 144, 226, 0.3)",
                    name="Score"
                )
            )

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
                margin=dict(l=40, r=40, t=40, b=40),
            )

            # Streamlit上で描画
            st.plotly_chart(fig, use_container_width=True)
            st.success("✅ レーダーチャートを表示しました！")

        except Exception as e:
            st.error(f"⚠️ チャート描画中にエラーが発生しました: {e}")

else:
    st.info("👆 上のボックスにCSVファイルをアップロードしてください")