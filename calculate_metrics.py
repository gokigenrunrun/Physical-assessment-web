import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

SCORE_COLUMNS: List[str] = [
    "head_movement",
    "shoulder_tilt",
    "torso_tilt",
    "leg_lift",
    "foot_sway",
    "arm_sag",
]

DEFAULT_SCORE_RANGES_RIGHT: Dict[str, Tuple[float, float]] = {
    "head_movement": (-0.039802, 0.096293),
    "shoulder_tilt": (-0.004085, 0.100632),
    "torso_tilt": (-0.009802, 0.056854),
    "leg_lift": (0.066994, 0.493938),
    "foot_sway": (-0.025375, 0.113312),
    "arm_sag": (-0.022492, 0.132132),
}

DEFAULT_SCORE_RANGES_LEFT: Dict[str, Tuple[float, float]] = {
    "head_movement": (-0.045299, 0.096693),
    "shoulder_tilt": (-0.006431, 0.083754),
    "torso_tilt": (-0.005356, 0.044649),
    "leg_lift": (0.359228, 0.615523),
    "foot_sway": (-0.073904, 0.214358),
    "arm_sag": (-0.031104, 0.168304),
}

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_SCORE_RANGE_FILES: Dict[str, Path] = {
    "right_leg": _MODULE_DIR / "score_ranges_right_leg.json",
    "left_leg": _MODULE_DIR / "score_ranges_left_leg.json",
}
_DEFAULT_ACTION_KEY = "right_leg"


def _coerce_range_pair(value: Union[Dict[str, float], Iterable[float]]) -> Optional[Tuple[float, float]]:
    """
    Convert JSON values into a (low, high) tuple when possible.
    """
    low: Optional[float]
    high: Optional[float]

    if isinstance(value, dict):
        low = value.get("low")
        high = value.get("high")
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            return None
        low, high = value
    else:
        return None

    try:
        return float(low), float(high)
    except (TypeError, ValueError):
        return None


def _load_ranges_from_json(path: Path) -> Dict[str, Tuple[float, float]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    loaded: Dict[str, Tuple[float, float]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            pair = _coerce_range_pair(value)
            if pair is not None:
                loaded[key] = pair
    return loaded


def load_score_ranges(base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Load score ranges for each action phase, falling back to defaults when JSON
    overrides are unavailable or invalid.
    """
    base_path = Path(base_dir) if base_dir is not None else _MODULE_DIR
    ranges: Dict[str, Dict[str, Tuple[float, float]]] = {
        "right_leg": DEFAULT_SCORE_RANGES_RIGHT.copy(),
        "left_leg": DEFAULT_SCORE_RANGES_LEFT.copy(),
    }

    for action_key, file_path in _DEFAULT_SCORE_RANGE_FILES.items():
        target_path = base_path / file_path.name if base_dir is not None else file_path
        if target_path.is_file():
            try:
                overrides = _load_ranges_from_json(target_path)
                ranges[action_key].update(overrides)
            except Exception as exc:
                print(f"⚠️ score range load error ({target_path}): {exc}")

    # Provide sensible fallbacks for phases without dedicated ranges
    ranges["raise"] = ranges["right_leg"].copy()
    ranges["unknown"] = ranges["right_leg"].copy()
    ranges["default"] = ranges["right_leg"].copy()
    return ranges


SCORE_RANGES = load_score_ranges()


def get_score_range(metric: str, action: Optional[str] = None) -> Tuple[float, float]:
    """
    Return the (low, high) range for the given metric and action.
    """
    action_key = action if action in SCORE_RANGES else _DEFAULT_ACTION_KEY
    ranges_for_action = SCORE_RANGES.get(action_key) or SCORE_RANGES[_DEFAULT_ACTION_KEY]
    if metric in ranges_for_action:
        return ranges_for_action[metric]

    fallback_ranges = SCORE_RANGES[_DEFAULT_ACTION_KEY]
    if metric in fallback_ranges:
        return fallback_ranges[metric]
    raise KeyError(f"Unknown score range for metric '{metric}'")

COLUMN_ALIASES = {
    "landmark_idx": "landmark_index",
    "landmarkId": "landmark_index",
    "landmarkID": "landmark_index",
    "landmark": "landmark_index",
    "frame_index": "frame",
    "frame_idx": "frame",
    "Frame": "frame",
}

NUMERIC_COLUMNS = ["x", "y", "z", "visibility"]

REFERENCE_ACTION_PHASES = [
    (0.0, 0.867, "raise"),
    (0.867, 2.0, "right_leg"),
    (2.0, 3.033, "raise"),
    (3.033, 4.267, "left_leg"),
    (4.267, 5.4, "raise"),
    (5.4, 6.467, "right_leg"),
    (6.467, 7.533, "raise"),
    (7.533, 8.767, "left_leg"),
    (8.767, 9.9, "raise"),
    (9.9, 11.033, "right_leg"),
    (11.033, 11.667, "raise"),
]

REFERENCE_MAX_FRAME = 350


def _empty_metric_dict() -> Dict[str, float]:
    return {k: np.nan for k in SCORE_COLUMNS}


def _source_label(source: Union[str, Path, pd.DataFrame]) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return "dataframe"


def classify_action(frame_idx: int, fps: float = 30.0) -> str:
    if frame_idx < 0 or frame_idx > REFERENCE_MAX_FRAME:
        return "unknown"
    if fps <= 0:
        fps = 30.0

    time_position = frame_idx / fps

    for start, end, label in REFERENCE_ACTION_PHASES:
        if start <= time_position < end:
            return label

    last_start, last_end, last_label = REFERENCE_ACTION_PHASES[-1]
    if last_start <= time_position <= last_end:
        return last_label

    return "unknown"


def _extract_frame_indices(series: pd.Series) -> pd.Series:
    """
    Try to parse numeric frame indices from mixed string/integer columns.
    """
    numeric_frames = pd.to_numeric(series, errors="coerce")
    if numeric_frames.notna().any():
        return numeric_frames

    as_str = series.astype(str)
    extracted = as_str.str.extract(r"(\d+)")[0]
    numeric = pd.to_numeric(extracted, errors="coerce")
    return numeric


def _normalize_by_frame(df: pd.DataFrame, column: str) -> pd.Series:
    def _normalize(series: pd.Series) -> pd.Series:
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            return series
        min_val = finite.min()
        max_val = finite.max()
        if np.isclose(max_val, min_val):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - min_val) / (max_val - min_val)

    return df.groupby("frame")[column].transform(_normalize)


def preprocess_landmarks(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    前処理：列統一 → 正規化 → visibilityフィルタ → フレーム補間
    """
    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    required_cols = {"landmark_index", "x", "y", "z"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(f"missing columns: {sorted(missing)}")

    df["landmark_index"] = pd.to_numeric(df["landmark_index"], errors="coerce")
    df = df.dropna(subset=["landmark_index"])
    df["landmark_index"] = df["landmark_index"].astype(int)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "frame" in df.columns:
        frame_series = df["frame"]
    else:
        frame_series = pd.Series(np.nan, index=df.index)

    numeric_frame = _extract_frame_indices(frame_series)
    if numeric_frame.notna().any():
        df["frame"] = numeric_frame
    else:
        df["frame"] = np.nan

    df["frame"] = df.groupby("landmark_index")["frame"].transform(
        lambda s: s.ffill().bfill()
    )
    df["frame"] = df["frame"].fillna(0)
    df["frame"] = df["frame"].round().astype(int)
    if not df["frame"].empty:
        min_frame = df["frame"].min()
        df["frame"] = df["frame"] - int(min_frame)

    if "visibility" in df.columns:
        low_vis_mask = df["visibility"] < 0.5
        df.loc[low_vis_mask, ["x", "y", "z"]] = np.nan

    df = df.sort_values(["landmark_index", "frame"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["landmark_index", "frame"], keep="last").reset_index(drop=True)
    df = df[df["landmark_index"].between(0, 32)]

    for col in ["x", "y", "z"]:
        if col in df.columns:
            df[col] = _normalize_by_frame(df, col)

    if "y" in df.columns:
        direction = _determine_vertical_direction(df)
        if direction == "bottom_up":
            df["y"] = 1.0 - df["y"]

    if df.empty:
        return df

    landmarks = sorted(df["landmark_index"].unique())
    max_frame = int(df["frame"].max())

    multi_index = pd.MultiIndex.from_product(
        [landmarks, range(max_frame + 1)],
        names=["landmark_index", "frame"],
    )

    df = df.set_index(["landmark_index", "frame"]).sort_index()
    df = df.reindex(multi_index)

    for col in ["x", "y", "z"]:
        if col in df.columns:
            df[col] = df.groupby(level=0)[col].transform(
                lambda s: s.interpolate(limit_direction="both")
            )
            df[col] = df[col].ffill().bfill()

    if "visibility" in df.columns:
        df["visibility"] = df.groupby(level=0)["visibility"].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        df["visibility"] = df["visibility"].ffill().bfill().fillna(0.0)

    df = df.reset_index().sort_values(["frame", "landmark_index"]).reset_index(drop=True)
    return df


def load_pose_dataframe(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        raw_df = source
    else:
        path = Path(source)
        raw_df = pd.read_csv(path, skip_blank_lines=True)
    return preprocess_landmarks(raw_df)


def _get_landmark_series(
    df: pd.DataFrame,
    landmark_index: int,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    subset = df[df["landmark_index"] == landmark_index].copy()
    subset = subset.set_index("frame").sort_index()
    if columns:
        missing = set(columns).difference(subset.columns)
        if missing:
            raise ValueError(f"columns {missing} missing for landmark {landmark_index}")
        subset = subset[list(columns)]
    return subset


def _compute_metrics(df: pd.DataFrame, source_label: str = "dataframe") -> Dict[str, float]:
    result: Dict[str, float] = {}
    frames = sorted(df["frame"].unique())
    action_by_frame = {frame: classify_action(frame) for frame in frames}

    try:
        head = _get_landmark_series(df, 0, ["x", "y", "z"])
        if head.empty:
            result["head_movement"] = np.nan
        else:
            deltas = head.diff().pow(2).sum(axis=1).pow(0.5).fillna(0.0)
            result["head_movement"] = float(deltas.std(ddof=0)) if not deltas.empty else np.nan
    except Exception as exc:
        print(f"⚠️ head_movement error ({source_label}): {exc}")
        result["head_movement"] = np.nan

    try:
        left_shoulder = _get_landmark_series(df, 11, ["y"])
        right_shoulder = _get_landmark_series(df, 12, ["y"])
        if left_shoulder.empty or right_shoulder.empty:
            result["shoulder_tilt"] = np.nan
        else:
            diff = (left_shoulder["y"] - right_shoulder["y"]).abs()
            result["shoulder_tilt"] = float(diff.mean())
    except Exception as exc:
        print(f"⚠️ shoulder_tilt error ({source_label}): {exc}")
        result["shoulder_tilt"] = np.nan

    try:
        left_hip = _get_landmark_series(df, 23, ["y"])
        right_hip = _get_landmark_series(df, 24, ["y"])
        if left_hip.empty or right_hip.empty:
            result["torso_tilt"] = np.nan
        else:
            diff = (left_hip["y"] - right_hip["y"]).abs()
            result["torso_tilt"] = float(diff.mean())
    except Exception as exc:
        print(f"⚠️ torso_tilt error ({source_label}): {exc}")
        result["torso_tilt"] = np.nan

    try:
        right_hip = _get_landmark_series(df, 24, ["y"])
        right_ankle = _get_landmark_series(df, 28, ["y"])
        left_hip = _get_landmark_series(df, 23, ["y"])
        left_ankle = _get_landmark_series(df, 27, ["y"])

        leg_samples: List[float] = []
        for frame in frames:
            action = action_by_frame.get(frame)
            if action == "left_leg":
                if frame in left_hip.index and frame in left_ankle.index:
                    value = float(left_ankle.loc[frame, "y"] - left_hip.loc[frame, "y"])
                    leg_samples.append(value)
            else:
                if frame in right_hip.index and frame in right_ankle.index:
                    value = float(right_ankle.loc[frame, "y"] - right_hip.loc[frame, "y"])
                    leg_samples.append(value)

        leg_array = np.array(leg_samples, dtype=float) if leg_samples else np.array([], dtype=float)
        if leg_array.size == 0 or np.all(np.isnan(leg_array)):
            result["leg_lift"] = np.nan
        else:
            result["leg_lift"] = float(np.nanpercentile(leg_array, 5))
    except Exception as exc:
        print(f"⚠️ leg_lift error ({source_label}): {exc}")
        result["leg_lift"] = np.nan

    try:
        right_stance = _get_landmark_series(df, 27, ["x"])
        left_stance = _get_landmark_series(df, 28, ["x"])

        foot_samples: List[float] = []
        for frame in frames:
            action = action_by_frame.get(frame)
            if action == "left_leg":
                if frame in left_stance.index:
                    foot_samples.append(float(left_stance.loc[frame, "x"]))
            else:
                if frame in right_stance.index:
                    foot_samples.append(float(right_stance.loc[frame, "x"]))

        foot_array = np.array(foot_samples, dtype=float) if foot_samples else np.array([], dtype=float)
        if foot_array.size == 0 or np.all(np.isnan(foot_array)):
            result["foot_sway"] = np.nan
        else:
            result["foot_sway"] = float(np.nanstd(foot_array, ddof=0))
    except Exception as exc:
        print(f"⚠️ foot_sway error ({source_label}): {exc}")
        result["foot_sway"] = np.nan

    try:
        left_shoulder = _get_landmark_series(df, 11, ["y"])
        left_elbow = _get_landmark_series(df, 13, ["y"])
        right_shoulder = _get_landmark_series(df, 12, ["y"])
        right_elbow = _get_landmark_series(df, 14, ["y"])
        if left_shoulder.empty or left_elbow.empty or right_shoulder.empty or right_elbow.empty:
            result["arm_sag"] = np.nan
        else:
            left_sag = (left_shoulder["y"] - left_elbow["y"]).abs().mean()
            right_sag = (right_shoulder["y"] - right_elbow["y"]).abs().mean()
            result["arm_sag"] = float(np.nanmean([left_sag, right_sag]))
    except Exception as exc:
        print(f"⚠️ arm_sag error ({source_label}): {exc}")
        result["arm_sag"] = np.nan

    return result


def calculate_metrics(data: Union[str, Path, pd.DataFrame]) -> Dict[str, float]:
    source = _source_label(data)
    try:
        df = load_pose_dataframe(data)
    except Exception as exc:
        print(f"⚠️ 読み込みエラー: {source} → {exc}")
        return _empty_metric_dict()
    return _compute_metrics(df, source)


def calculate_metrics_from_df(dataframe: pd.DataFrame) -> Dict[str, float]:
    processed = preprocess_landmarks(dataframe)
    return _compute_metrics(processed, "dataframe")


def calculate_metrics_by_frame(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    df = load_pose_dataframe(data)

    frames = sorted(df["frame"].unique())

    head = _get_landmark_series(df, 0, ["x", "y", "z"])
    head_shift = head.shift(1)
    head_movement_series = (head - head_shift).pow(2).sum(axis=1).pow(0.5).fillna(0.0)

    left_shoulder = _get_landmark_series(df, 11, ["y"])
    right_shoulder = _get_landmark_series(df, 12, ["y"])
    left_hip = _get_landmark_series(df, 23, ["y"])
    right_hip = _get_landmark_series(df, 24, ["y"])
    right_ankle = _get_landmark_series(df, 28, ["y"])
    left_ankle = _get_landmark_series(df, 27, ["y"])
    right_foot = _get_landmark_series(df, 27, ["x"])
    left_foot = _get_landmark_series(df, 28, ["x"])

    right_baseline_series = right_foot["x"].dropna()
    right_baseline = float(right_baseline_series.iloc[0]) if not right_baseline_series.empty else np.nan
    left_baseline_series = left_foot["x"].dropna()
    left_baseline = float(left_baseline_series.iloc[0]) if not left_baseline_series.empty else np.nan
    left_elbow = _get_landmark_series(df, 13, ["y"])
    right_elbow = _get_landmark_series(df, 14, ["y"])

    records: List[Dict[str, float]] = []
    for frame in frames:
        action = classify_action(frame)
        metrics: Dict[str, float] = {"frame": frame, "action": action}

        metrics["head_movement"] = float(head_movement_series.get(frame, np.nan))

        if frame in left_shoulder.index and frame in right_shoulder.index:
            metrics["shoulder_tilt"] = float(
                abs(left_shoulder.loc[frame, "y"] - right_shoulder.loc[frame, "y"])
            )
        else:
            metrics["shoulder_tilt"] = np.nan

        if frame in left_hip.index and frame in right_hip.index:
            metrics["torso_tilt"] = float(
                abs(left_hip.loc[frame, "y"] - right_hip.loc[frame, "y"])
            )
        else:
            metrics["torso_tilt"] = np.nan

        if action == "left_leg":
            if frame in left_hip.index and frame in left_ankle.index:
                metrics["leg_lift"] = float(left_ankle.loc[frame, "y"] - left_hip.loc[frame, "y"])
            else:
                metrics["leg_lift"] = np.nan
            if frame in left_foot.index and not np.isnan(left_baseline):
                metrics["foot_sway"] = float(abs(left_foot.loc[frame, "x"] - left_baseline))
            else:
                metrics["foot_sway"] = np.nan
        else:
            if frame in right_hip.index and frame in right_ankle.index:
                metrics["leg_lift"] = float(right_ankle.loc[frame, "y"] - right_hip.loc[frame, "y"])
            else:
                metrics["leg_lift"] = np.nan
            if frame in right_foot.index and not np.isnan(right_baseline):
                metrics["foot_sway"] = float(abs(right_foot.loc[frame, "x"] - right_baseline))
            else:
                metrics["foot_sway"] = np.nan

        arm_vals: List[float] = []
        if frame in left_shoulder.index and frame in left_elbow.index:
            arm_vals.append(
                float(abs(left_shoulder.loc[frame, "y"] - left_elbow.loc[frame, "y"]))
            )
        if frame in right_shoulder.index and frame in right_elbow.index:
            arm_vals.append(
                float(abs(right_shoulder.loc[frame, "y"] - right_elbow.loc[frame, "y"]))
            )
        metrics["arm_sag"] = float(np.nanmean(arm_vals)) if arm_vals else np.nan

        records.append(metrics)

    result_df = pd.DataFrame(records)
    return result_df


def plot_frame_metrics(df: pd.DataFrame, title: str = "Frame-wise Motion Dynamics") -> None:
    plt.figure(figsize=(10, 5))
    for col in df.columns:
        if col in {"frame", "action"}:
            continue
        plt.plot(df["frame"], df[col], label=col)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def resample_series(values: pd.Series, length: int = 100) -> np.ndarray:
    cleaned = values.dropna()
    if cleaned.empty or length <= 0:
        return np.array([])

    mask = values.notna().to_numpy()
    original_idx = np.linspace(0, 1, num=len(values))
    xp = original_idx[mask]
    fp = values[mask].to_numpy(dtype=float)

    if xp.size == 0:
        return np.array([])

    target_idx = np.linspace(0, 1, num=length)
    return np.interp(target_idx, xp, fp)


def compare_motion_profiles(
    user_metrics: pd.DataFrame,
    reference_metrics: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    if columns is None:
        columns = SCORE_COLUMNS

    similarities: Dict[str, float] = {}

    for col in columns:
        if col not in user_metrics.columns or col not in reference_metrics.columns:
            similarities[col] = np.nan
            continue

        user_series = user_metrics[col].astype(float)
        ref_series = reference_metrics[col].astype(float)

        user_resampled = resample_series(user_series)
        ref_resampled = resample_series(ref_series)

        if user_resampled.size == 0 or ref_resampled.size == 0:
            similarities[col] = np.nan
            continue

        min_len = min(len(user_resampled), len(ref_resampled))
        user_vec = user_resampled[:min_len]
        ref_vec = ref_resampled[:min_len]

        user_norm = np.linalg.norm(user_vec)
        ref_norm = np.linalg.norm(ref_vec)
        if user_norm == 0 or ref_norm == 0:
            similarities[col] = np.nan
            continue

        cosine_sim = float(np.dot(user_vec, ref_vec) / (user_norm * ref_norm))
        cosine_sim = max(min(cosine_sim, 1.0), -1.0)
        similarities[col] = (cosine_sim + 1) / 2 * 100

    valid_scores = [v for v in similarities.values() if not np.isnan(v)]
    similarities["overall_similarity"] = float(np.mean(valid_scores)) if valid_scores else np.nan

    return similarities
def _determine_vertical_direction(df: pd.DataFrame) -> str:
    """
    Detect whether larger y-values are lower (\"top-down\") or higher (\"bottom-up\").
    """
    subset = df[df["landmark_index"].isin([0, 28])]
    if subset.empty or "frame" not in subset:
        return "top_down"

    pivot = subset.pivot_table(index="frame", columns="landmark_index", values="y", aggfunc="first")
    pivot = pivot.dropna()
    if pivot.empty or 0 not in pivot.columns or 28 not in pivot.columns:
        return "top_down"

    diff = pivot[28] - pivot[0]
    if diff.mean() < 0:
        return "bottom_up"
    return "top_down"
