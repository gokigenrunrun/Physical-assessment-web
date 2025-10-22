import pandas as pd
import numpy as np

def calculate_metrics(file_path: str) -> dict:
    try:
        # 読み込み時のずれ対策：空白行を無視して強制読み込み
        df = pd.read_csv(file_path, skip_blank_lines=True)
        df.columns = df.columns.str.strip()  # 余分な空白を削除
    except Exception as e:
        print(f"⚠️ 読み込みエラー: {file_path} → {e}")
        return {k: np.nan for k in ["head_movement", "shoulder_tilt", "torso_tilt", "leg_lift", "foot_sway", "arm_sag"]}

    # 想定列がない場合スキップ
    required_cols = {"landmark_index", "x", "y", "z"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ 列不足: {file_path} → {df.columns.tolist()}")
        return {k: np.nan for k in ["head_movement", "shoulder_tilt", "torso_tilt", "leg_lift", "foot_sway", "arm_sag"]}

    result = {}

    # ===== 頭のブレ（head_movement）=====
    try:
        head = df[df["landmark_index"] == 0].copy()
        head["diff"] = np.sqrt(
            head["x"].diff()**2 + head["y"].diff()**2 + head["z"].diff()**2
        )
        result["head_movement"] = head["diff"].mean(skipna=True)
    except Exception as e:
        print(f"⚠️ head_movement error ({file_path}): {e}")
        result["head_movement"] = np.nan

    # ===== 肩の傾き（shoulder_tilt）=====
    try:
        left_shoulder = df[df["landmark_index"] == 11]
        right_shoulder = df[df["landmark_index"] == 12]
        if len(left_shoulder) and len(right_shoulder):
            result["shoulder_tilt"] = abs(left_shoulder["y"].values - right_shoulder["y"].values).mean()
        else:
            result["shoulder_tilt"] = np.nan
    except Exception as e:
        print(f"⚠️ shoulder_tilt error ({file_path}): {e}")
        result["shoulder_tilt"] = np.nan

    # ===== 体幹の傾き（torso_tilt）=====
    try:
        left_hip = df[df["landmark_index"] == 23]
        right_hip = df[df["landmark_index"] == 24]
        if len(left_hip) and len(right_hip):
            result["torso_tilt"] = abs(left_hip["y"].values - right_hip["y"].values).mean()
        else:
            result["torso_tilt"] = np.nan
    except Exception as e:
        print(f"⚠️ torso_tilt error ({file_path}): {e}")
        result["torso_tilt"] = np.nan

    # ===== 足上げ高さ（leg_lift）=====
    try:
        hip = df[df["landmark_index"] == 24]
        ankle = df[df["landmark_index"] == 28]
        if len(hip) and len(ankle):
            result["leg_lift"] = (ankle["y"].values - hip["y"].values).min()
        else:
            result["leg_lift"] = np.nan
    except Exception as e:
        print(f"⚠️ leg_lift error ({file_path}): {e}")
        result["leg_lift"] = np.nan

    # ===== 接地足の横ブレ（foot_sway）=====
    try:
        foot = df[df["landmark_index"] == 28]
        result["foot_sway"] = foot["x"].std(skipna=True)
    except Exception as e:
        print(f"⚠️ foot_sway error ({file_path}): {e}")
        result["foot_sway"] = np.nan

    # ===== 腕の垂れ下がり（arm_sag）=====
    try:
        shoulder = df[df["landmark_index"] == 12]
        wrist = df[df["landmark_index"] == 16]
        if len(shoulder) and len(wrist):
            result["arm_sag"] = (wrist["y"].values - shoulder["y"].values).mean()
        else:
            result["arm_sag"] = np.nan
    except Exception as e:
        print(f"⚠️ arm_sag error ({file_path}): {e}")
        result["arm_sag"] = np.nan

    return result
