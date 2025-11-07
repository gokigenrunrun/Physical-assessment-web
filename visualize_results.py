import pandas as pd
import matplotlib.pyplot as plt
from calculate_metrics import calculate_metrics_by_frame, plot_frame_metrics

def calculate_metrics_by_frame(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    results = []

    # 🔧 frame列がない場合 → 全データを1フレーム扱いにする
    if "frame" not in df.columns:
        df["frame"] = 0

    frames = sorted(df["frame"].unique())

    for frame in frames:
        frame_df = df[df["frame"] == frame]

        # ---- 頭のブレ ----
        head = frame_df[frame_df["landmark_index"] == 0]
        head_movement = np.sqrt(
            (head["x"].diff()**2 + head["y"].diff()**2 + head["z"].diff()**2)
        ).mean()

        # ---- 肩の傾き ----
        left_shoulder = frame_df[frame_df["landmark_index"] == 11]
        right_shoulder = frame_df[frame_df["landmark_index"] == 12]
        shoulder_tilt = abs(left_shoulder["y"].values - right_shoulder["y"].values).mean() if len(left_shoulder) and len(right_shoulder) else np.nan

        # ---- 体幹の傾き ----
        left_hip = frame_df[frame_df["landmark_index"] == 23]
        right_hip = frame_df[frame_df["landmark_index"] == 24]
        torso_tilt = abs(left_hip["y"].values - right_hip["y"].values).mean() if len(left_hip) and len(right_hip) else np.nan

        # ---- 足上げ高さ ----
        hip = frame_df[frame_df["landmark_index"] == 24]
        ankle = frame_df[frame_df["landmark_index"] == 28]
        leg_lift = (ankle["y"].values - hip["y"].values).mean() if len(hip) and len(ankle) else np.nan

        # ---- 足の横ブレ ----
        foot = frame_df[frame_df["landmark_index"] == 28]
        foot_sway = foot["x"].std(skipna=True)

        # ---- 腕の垂れ下がり ----
        shoulder = frame_df[frame_df["landmark_index"] == 12]
        wrist = frame_df[frame_df["landmark_index"] == 16]
        arm_sag = (wrist["y"].values - shoulder["y"].values).mean() if len(shoulder) and len(wrist) else np.nan

        results.append({
            "frame": frame,
            "head_movement": head_movement,
            "shoulder_tilt": shoulder_tilt,
            "torso_tilt": torso_tilt,
            "leg_lift": leg_lift,
            "foot_sway": foot_sway,
            "arm_sag": arm_sag
        })

    return pd.DataFrame(results)

# =============================
# 🆕 追加部分：フレームごとのスコア計算とグラフ表示
# =============================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_metrics_by_frame(file_path: str) -> pd.DataFrame:
    """
    各フレームごとに動きの大きさを数値化してDataFrameで返す関数
    """
    df = pd.read_csv(file_path)
    results = []

    # 🔧 frame列がない場合 → 全データを1フレーム扱いにする
    if "frame" not in df.columns:
        df["frame"] = 0

    frames = sorted(df["frame"].unique())

    for frame in frames:
        frame_df = df[df["frame"] == frame]

        # ---- 頭のブレ ----
        head = frame_df[frame_df["landmark_index"] == 0]
        head_movement = np.sqrt(
            (head["x"].diff()**2 + head["y"].diff()**2 + head["z"].diff()**2)
        ).mean()

        # ---- 肩の傾き ----
        left_shoulder = frame_df[frame_df["landmark_index"] == 11]
        right_shoulder = frame_df[frame_df["landmark_index"] == 12]
        shoulder_tilt = abs(left_shoulder["y"].values - right_shoulder["y"].values).mean() if len(left_shoulder) and len(right_shoulder) else np.nan

        # ---- 体幹の傾き ----
        left_hip = frame_df[frame_df["landmark_index"] == 23]
        right_hip = frame_df[frame_df["landmark_index"] == 24]
        torso_tilt = abs(left_hip["y"].values - right_hip["y"].values).mean() if len(left_hip) and len(right_hip) else np.nan

        # ---- 足上げ高さ ----
        hip = frame_df[frame_df["landmark_index"] == 24]
        ankle = frame_df[frame_df["landmark_index"] == 28]
        leg_lift = (ankle["y"].values - hip["y"].values).mean() if len(hip) and len(ankle) else np.nan

        # ---- 足の横ブレ ----
        foot = frame_df[frame_df["landmark_index"] == 28]
        foot_sway = foot["x"].std(skipna=True)

        # ---- 腕の垂れ下がり ----
        shoulder = frame_df[frame_df["landmark_index"] == 12]
        wrist = frame_df[frame_df["landmark_index"] == 16]
        arm_sag = (wrist["y"].values - shoulder["y"].values).mean() if len(shoulder) and len(wrist) else np.nan

        results.append({
            "frame": frame,
            "head_movement": head_movement,
            "shoulder_tilt": shoulder_tilt,
            "torso_tilt": torso_tilt,
            "leg_lift": leg_lift,
            "foot_sway": foot_sway,
            "arm_sag": arm_sag
        })

    return pd.DataFrame(results)


def plot_frame_metrics(df: pd.DataFrame, title="Frame-wise Motion Dynamics"):
    """
    各フレームのスコア変化を折れ線グラフで表示する関数
    """
    plt.figure(figsize=(10, 5))
    for col in df.columns[1:]:
        plt.plot(df["frame"], df[col], label=col)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import matplotlib.font_manager as fm

# # ==== 日本語フォント設定（Colab・Streamlit共通）====
# try:
#     font_path = "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
#     plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
# except Exception:
#     plt.rcParams['font.family'] = "Arial"

# plt.rcParams['axes.unicode_minus'] = False

# # ==== グラフのスタイル共通設定 ====
# sns.set(style="whitegrid", palette="pastel")

# def show_distribution(df, column, bins=30, title_ja=None):
#     """
#     指定したカラムの分布をヒストグラムで表示
#     """
#     if column not in df.columns:
#         print(f"⚠️ カラム {column} が存在しません。スキップします。")
#         return

#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[column].dropna(), bins=bins, kde=True, color="#85C1E9")
#     plt.title(title_ja if title_ja else f"{column} の分布")
#     plt.xlabel(title_ja if title_ja else column)
#     plt.ylabel("人数")
#     plt.tight_layout()
#     plt.show()


# def show_boxplot(df, columns, title_ja="選択した特徴量の箱ひげ図"):
#     """
#     複数カラムをまとめて箱ひげ図で表示
#     """
#     valid_columns = [c for c in columns if c in df.columns]
#     if not valid_columns:
#         print("⚠️ 有効なカラムがありません。スキップします。")
#         return

#     plt.figure(figsize=(8, 5))
#     sns.boxplot(data=df[valid_columns], palette="coolwarm")
#     plt.title(title_ja)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()


# def visualize_overall(result_dict, title_ja="各特徴量のスコア比較"):
#     """
#     calculate_metrics の結果辞書を棒グラフで表示
#     """
#     if not result_dict:
#         print("⚠️ 結果が空です。スキップします。")
#         return

#     plt.figure(figsize=(6, 4))
#     names = list(result_dict.keys())
#     values = list(result_dict.values())
#     sns.barplot(x=names, y=values, palette="cool")
#     plt.title(title_ja)
#     plt.ylabel("スコア")
#     plt.ylim(0, max(values) * 1.2 if len(values) else 1)
#     plt.xticks(rotation=30)
#     plt.tight_layout()
#     plt.show()
