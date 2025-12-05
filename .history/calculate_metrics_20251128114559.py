import json
import numpy as np
import pandas as pd
from math import acos, degrees
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

SCORE_COLUMNS: List[str] = [
    "head_movement",
    "shoulder_tilt",
    "torso_tilt",
    "leg_lift",
    "foot_sway",
    "arm_sag",
    "banzai_score",
]

DEFAULT_SCORE_RANGES_RIGHT: Dict[str, Tuple[float, float]] = {
    # Head stability はすでにあなたが p95 使用済み
    "head_movement": (0.005516483219872959, 0.053272474857389424),

    "shoulder_tilt": (0.021946430769230765, 0.12384272330300272),
    "torso_tilt":    (0.007595807017543859, 0.06966647748410838),
    "leg_lift":      (0.0,                     0.61519865),   # マイナスは0に補正
    "foot_sway":     (0.008381150395652065,    0.10307820541316964),
    "arm_sag":       (0.01860632456140351,     0.18818743437990576),

    "banzai_score": (0.0, 100.0),
}

DEFAULT_SCORE_RANGES_LEFT: Dict[str, Tuple[float, float]] = {
    # Head stability はすでにあなたが p95 使用済み
    "head_movement": (0.004994789133149868, 0.031311599045372396),

    "shoulder_tilt": (0.007831507462686568, 0.10730938518439757),
    "torso_tilt":    (0.006634071823204422, 0.05333447362480539),
    "leg_lift":      (0.0,                     0.6520887225),   # マイナスは0に補正
    "foot_sway":     (0.007534867280661313,    0.0624402957123336),
    "arm_sag":       (0.02557877976190476,     0.21530692104473537),

    "banzai_score": (0.0, 100.0),
}

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_SCORE_RANGE_FILES: Dict[str, Path] = {
    "right_leg": _MODULE_DIR / "score_ranges_right_leg.json",
    "left_leg": _MODULE_DIR / "score_ranges_left_leg.json",
}
_DEFAULT_ACTION_KEY = "right_leg"

LEG_PHASE_BASE_MAP = {
    "right_leg_1": "right_leg",
    "right_leg_2": "right_leg",
    "left_leg_1": "left_leg",
    "left_leg_2": "left_leg",
}


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
    for alias, base_key in LEG_PHASE_BASE_MAP.items():
        source_key = base_key if base_key in ranges else _DEFAULT_ACTION_KEY
        ranges[alias] = ranges[source_key].copy()
    ranges["unknown"] = ranges["right_leg"].copy()
    ranges["default"] = ranges["right_leg"].copy()
    return ranges


SCORE_RANGES = load_score_ranges()


def get_score_range(metric: str, action: Optional[str] = None) -> Tuple[float, float]:
    """
    Return the (low, high) range for the given metric and action.
    """
    normalized_action = LEG_PHASE_BASE_MAP.get(action, action) if action else None
    action_key = normalized_action if normalized_action in SCORE_RANGES else _DEFAULT_ACTION_KEY
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
    (15, 29, "right_leg_1"),
    (51, 65, "left_leg_1"),
    (86, 100, "right_leg_2"),
    (120, 134, "left_leg_2"),
]

RIGHT_LEG_PHASES = {"right_leg_1", "right_leg_2"}
LEFT_LEG_PHASES = {"left_leg_1", "left_leg_2"}
REFERENCE_MAX_FRAME = max(end for _, end, _ in REFERENCE_ACTION_PHASES)
BANZAI_REQUIRED_LANDMARKS = [11, 12, 15, 16, 23, 24]


def _empty_metric_dict() -> Dict[str, float]:
    return {k: np.nan for k in SCORE_COLUMNS}


def _source_label(source: Union[str, Path, pd.DataFrame]) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return "dataframe"


def classify_action(frame_idx: int, fps: float = 30.0) -> str:
    if frame_idx < 0 or frame_idx > REFERENCE_MAX_FRAME:
        return "unknown"
    for start, end, label in REFERENCE_ACTION_PHASES:
        if start <= frame_idx <= end:
            return label
    return "unknown"


def _is_left_leg_phase(action: Optional[str]) -> bool:
    return action in LEFT_LEG_PHASES


def _is_right_leg_phase(action: Optional[str]) -> bool:
    return action in RIGHT_LEG_PHASES


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


def vertical_angle(px: float, py: float) -> float:
    """
    Return the angle in degrees between (px, py) and the vertical upward direction (0, -1).
    Smaller values indicate a more vertical (arms-up) alignment.
    """
    norm = np.hypot(px, py)
    if norm == 0 or not np.isfinite(norm):
        return np.nan
    cos_theta = np.clip(-py / norm, -1.0, 1.0)
    return float(degrees(acos(cos_theta)))


def contiguous_windows(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return inclusive index intervals for contiguous True regions in the mask.
    """
    if mask.size == 0:
        return []
    true_idx = np.where(mask)[0]
    if true_idx.size == 0:
        return []
    windows: List[Tuple[int, int]] = []
    start = prev = int(true_idx[0])
    for idx in true_idx[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        windows.append((start, prev))
        start = prev = idx
    windows.append((start, prev))
    return windows


def up_score(left_up: np.ndarray, right_up: np.ndarray) -> np.ndarray:
    """
    Compute the bilateral arms-up score as the minimum elevation of both arms.
    """
    return np.minimum(left_up, right_up)


def evaluate_banzai_pose_auto(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect arms-up posture segments (Banzai) across all frames,
    compute Peak (p90), Consistency (mean), and blended score.
    Phase-independent: does not rely on pre-defined timing windows.
    """
    fps_raw = df.attrs.get("fps", 30)
    try:
        fps = int(fps_raw)
    except (TypeError, ValueError):
        fps = 30
    if fps <= 0:
        fps = 30

    VIS_THRESH = 0.3
    HEIGHT_THRESH = 0.08
    VERTICAL_DEG_MAX = 55
    MIN_WINDOW_FRAMES = max(1, int(0.18 * fps))

    base_result: Dict[str, Any] = {
        "file": str(df.attrs.get("source", "")),
        "has_window": False,
        "rep_frame_num": np.nan,
        "window_start_frame": np.nan,
        "window_end_frame": np.nan,
        "window_len_frames": 0,
        "window_len_sec": 0.0,
        "p90_upscore": np.nan,
        "mean_upscore": np.nan,
        "final_score": np.nan,
        "selected_window_index": None,
        "windows": [],
    }

    if df.empty or "frame" not in df or "landmark_index" not in df:
        base_result["note"] = "Insufficient data for evaluation"
        return base_result

    work_df = df.copy()
    if work_df["frame"].dtype == object:
        frame_num = work_df["frame"].astype(str).str.extract(r"(\\d+)")[0]
        work_df["_frame_num"] = pd.to_numeric(frame_num, errors="coerce")
    else:
        work_df["_frame_num"] = pd.to_numeric(work_df["frame"], errors="coerce")

    work_df = work_df.dropna(subset=["_frame_num"])
    if work_df.empty:
        base_result["note"] = "No valid frame indices"
        return base_result

    subset = work_df[work_df["landmark_index"].isin(BANZAI_REQUIRED_LANDMARKS)].copy()
    if subset.empty:
        base_result["note"] = "Required landmarks missing"
        return base_result

    if "visibility" in subset.columns:
        low_vis = subset["visibility"] < VIS_THRESH
        subset.loc[low_vis, ["x", "y"]] = np.nan

    try:
        pivot = subset.pivot_table(
            index="_frame_num",
            columns="landmark_index",
            values=["x", "y"],
            aggfunc="first",
        ).sort_index()
    except Exception:
        base_result["note"] = "Failed to pivot landmark data"
        return base_result

    if pivot.empty:
        base_result["note"] = "No pivoted data for evaluation"
        return base_result

    frame_numbers = pivot.index.to_numpy(dtype=int, copy=False)

    def get_xy(lm: int) -> Tuple[np.ndarray, np.ndarray]:
        x_vals = pivot.get(("x", lm))
        y_vals = pivot.get(("y", lm))
        if x_vals is None or y_vals is None:
            return np.full(len(pivot), np.nan), np.full(len(pivot), np.nan)
        return x_vals.to_numpy(dtype=float, copy=False), y_vals.to_numpy(dtype=float, copy=False)

    l_sh_x, l_sh_y = get_xy(11)
    r_sh_x, r_sh_y = get_xy(12)
    l_wr_x, l_wr_y = get_xy(15)
    r_wr_x, r_wr_y = get_xy(16)
    l_hip_x, l_hip_y = get_xy(23)
    r_hip_x, r_hip_y = get_xy(24)

    sh_x = np.nanmean(np.vstack([l_sh_x, r_sh_x]), axis=0)
    sh_y = np.nanmean(np.vstack([l_sh_y, r_sh_y]), axis=0)
    hip_x = np.nanmean(np.vstack([l_hip_x, r_hip_x]), axis=0)
    hip_y = np.nanmean(np.vstack([l_hip_y, r_hip_y]), axis=0)

    torso_len = np.hypot(sh_x - hip_x, sh_y - hip_y)
    invalid_torso = (~np.isfinite(torso_len)) | (torso_len <= 0)
    torso_len[invalid_torso] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        left_up = (sh_y - l_wr_y) / torso_len
        right_up = (sh_y - r_wr_y) / torso_len

    def _arm_angle(wr_x: np.ndarray, wr_y: np.ndarray, sh_x_vals: np.ndarray, sh_y_vals: np.ndarray) -> np.ndarray:
        angles = np.full(len(wr_x), np.nan)
        finite_mask = np.isfinite(wr_x) & np.isfinite(wr_y) & np.isfinite(sh_x_vals) & np.isfinite(sh_y_vals)
        idx = np.where(finite_mask)[0]
        for i in idx:
            delta_x = wr_x[i] - sh_x_vals[i]
            delta_y = wr_y[i] - sh_y_vals[i]
            angles[i] = vertical_angle(delta_x, delta_y)
        return angles

    l_ang = _arm_angle(l_wr_x, l_wr_y, l_sh_x, l_sh_y)
    r_ang = _arm_angle(r_wr_x, r_wr_y, r_sh_x, r_sh_y)

    cond_left = (left_up >= HEIGHT_THRESH) & (l_ang <= VERTICAL_DEG_MAX)
    cond_right = (right_up >= HEIGHT_THRESH) & (r_ang <= VERTICAL_DEG_MAX)
    bilateral_mask = cond_left & cond_right & np.isfinite(up_score(left_up, right_up))

    windows = [
        (start, end)
        for start, end in contiguous_windows(bilateral_mask)
        if (end - start + 1) >= MIN_WINDOW_FRAMES
    ]

    if not windows:
        base_result["note"] = "No valid 'arms up' window found"
        return base_result

    ups = up_score(left_up, right_up)
    window_summaries: List[Dict[str, Any]] = []
    for start, end in windows:
        seg = ups[start : end + 1]
        finite_mask = np.isfinite(seg)
        if not finite_mask.any():
            continue
        p90 = float(np.nanpercentile(seg, 90))
        mean = float(np.nanmean(seg))
        target_val = p90 if np.isfinite(p90) else mean
        if not np.isfinite(target_val):
            target_val = float(seg[finite_mask][0])
        abs_diff = np.abs(seg - target_val)
        try:
            idx_rel = int(np.nanargmin(abs_diff))
        except ValueError:
            idx_rel = int(np.where(finite_mask)[0][0])

        rep_idx = start + idx_rel
        rep_frame = int(frame_numbers[rep_idx])
        length_frames = int(end - start + 1)
        length_sec = length_frames / fps if fps > 0 else np.nan
        final_raw = np.nan
        if np.isfinite(p90) and np.isfinite(mean):
            final_raw = 0.6 * p90 + 0.4 * mean
        elif np.isfinite(p90):
            final_raw = float(p90)
        elif np.isfinite(mean):
            final_raw = float(mean)

        window_summaries.append(
            {
                "window_id": len(window_summaries),
                "start_idx": int(start),
                "end_idx": int(end),
                "window_start_frame": int(frame_numbers[int(start)]),
                "window_end_frame": int(frame_numbers[int(end)]),
                "window_len_frames": length_frames,
                "window_len_sec": float(length_sec),
                "p90_upscore": round(float(p90), 5),
                "mean_upscore": round(float(mean), 5),
                "final_score": round(float(final_raw), 5) if np.isfinite(final_raw) else np.nan,
                "rep_frame_num": rep_frame,
            }
        )

    if not window_summaries:
        base_result["note"] = "No valid 'arms up' window found"
        return base_result
    base_result["windows"] = window_summaries

    def _sort_key(item: Dict[str, Any]) -> Tuple[float, float, int]:
        final_val_raw = item.get("final_score")
        final_val = float(final_val_raw) if final_val_raw is not None else float("-inf")
        if not np.isfinite(final_val):
            final_val = float("-inf")

        p90_val_raw = item.get("p90_upscore")
        p90_val = float(p90_val_raw) if p90_val_raw is not None else float("-inf")
        if not np.isfinite(p90_val):
            p90_val = float("-inf")

        mean_val_raw = item.get("mean_upscore")
        mean_val = float(mean_val_raw) if mean_val_raw is not None else float("-inf")
        if not np.isfinite(mean_val):
            mean_val = float("-inf")

        length_frames = int(item.get("window_len_frames") or 0)
        return (final_val, p90_val, mean_val, length_frames)

    best = max(window_summaries, key=_sort_key)

    return {
        **base_result,
        "has_window": True,
        "rep_frame_num": int(best["rep_frame_num"]),
        "window_start_frame": int(best["window_start_frame"]),
        "window_end_frame": int(best["window_end_frame"]),
        "window_len_frames": int(best["window_len_frames"]),
        "window_len_sec": round(float(best["window_len_sec"]), 3),
        "p90_upscore": float(best["p90_upscore"]),
        "mean_upscore": float(best["mean_upscore"]),
        "final_score": float(best["final_score"]),
        "selected_window_index": int(best["window_id"]),
    }


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
            if _is_left_leg_phase(action):
                if frame in left_hip.index and frame in left_ankle.index:
                    value = float(left_ankle.loc[frame, "y"] - left_hip.loc[frame, "y"])
                    leg_samples.append(value)
            elif _is_right_leg_phase(action):
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
            if _is_left_leg_phase(action):
                if frame in left_stance.index:
                    foot_samples.append(float(left_stance.loc[frame, "x"]))
            elif _is_right_leg_phase(action):
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

    banzai_eval = evaluate_banzai_pose_auto(df)
    window_score_map: Dict[int, float] = {}
    window_candidates: List[Tuple[int, int, float]] = []
    raw_windows = banzai_eval.get("windows") if isinstance(banzai_eval, dict) else []
    if isinstance(raw_windows, list):
        for win in raw_windows:
            if not isinstance(win, dict):
                continue
            start_raw = win.get("window_start_frame")
            end_raw = win.get("window_end_frame")
            final_raw = win.get("final_score")
            try:
                start_frame = int(start_raw)
                end_frame = int(end_raw)
            except (TypeError, ValueError):
                continue
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame
            final_score = float(final_raw) if final_raw is not None else np.nan
            if not np.isfinite(final_score):
                continue
            window_candidates.append((start_frame, end_frame, final_score))

    for frame in frames:
        score_val = np.nan
        for start_frame, end_frame, final_score in window_candidates:
            if start_frame <= frame <= end_frame:
                if np.isnan(score_val) or final_score > score_val:
                    score_val = final_score
        window_score_map[frame] = score_val

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

        if _is_left_leg_phase(action):
            if frame in left_hip.index and frame in left_ankle.index:
                metrics["leg_lift"] = float(left_ankle.loc[frame, "y"] - left_hip.loc[frame, "y"])
            else:
                metrics["leg_lift"] = np.nan
            if frame in left_foot.index and not np.isnan(left_baseline):
                metrics["foot_sway"] = float(abs(left_foot.loc[frame, "x"] - left_baseline))
            else:
                metrics["foot_sway"] = np.nan
        elif _is_right_leg_phase(action):
            if frame in right_hip.index and frame in right_ankle.index:
                metrics["leg_lift"] = float(right_ankle.loc[frame, "y"] - right_hip.loc[frame, "y"])
            else:
                metrics["leg_lift"] = np.nan
            if frame in right_foot.index and not np.isnan(right_baseline):
                metrics["foot_sway"] = float(abs(right_foot.loc[frame, "x"] - right_baseline))
            else:
                metrics["foot_sway"] = np.nan
        else:
            metrics["leg_lift"] = np.nan
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
        score_val = window_score_map.get(frame, np.nan)
        metrics["banzai_score"] = float(score_val) if np.isfinite(score_val) else np.nan

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


# ============================================================
# BANZAI EVALUATION MODULE
# ------------------------------------------------------------
# Evaluate "arms-up" (Banzai) motion segments from pose CSVs.
# This function detects valid arms-up windows and computes
# Peak (p90) and Consistency (mean) scores, combining them
# into a final score (0.6 * Peak + 0.4 * Consistency).
# ============================================================


def evaluate_banzai_from_csv(csv_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a Mediapipe pose CSV, evaluate the Banzai pose quality, and return summary metrics.
    """
    path = Path(csv_path)
    base: Dict[str, Any] = {
        "file": str(path),
        "has_window": False,
        "rep_frame_num": np.nan,
        "window_start_frame": np.nan,
        "window_end_frame": np.nan,
        "window_len_frames": 0,
        "window_len_sec": 0.0,
        "p90_upscore": np.nan,
        "mean_upscore": np.nan,
        "final_score": np.nan,
        "selected_window_index": None,
        "windows": [],
    }

    if not path.is_file():
        base["note"] = "File not found"
        return base

    try:
        pose_df = load_pose_dataframe(path)
        pose_df.attrs["source"] = str(path)
    except Exception as exc:
        base["note"] = f"Failed to load CSV: {exc}"
        return base

    evaluation = evaluate_banzai_pose_auto(pose_df)
    merged = {**base, **evaluation}

    return merged


def batch_evaluate_banzai(input_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Evaluate all CSV files in a directory for the Banzai pose and return a DataFrame summary.
    """
    dir_path = Path(input_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"input_dir must be an existing directory: {input_dir}")

    records: List[Dict[str, Any]] = []
    for csv_file in sorted(dir_path.glob("*.csv")):
        records.append(evaluate_banzai_from_csv(csv_file))

    return pd.DataFrame(records)
