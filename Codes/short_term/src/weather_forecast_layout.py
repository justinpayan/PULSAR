from typing import Any, Dict, Optional, Tuple

import numpy as np

RMS_LOOKAHEAD_MEAN_KEY = "rms_mean_by_lookahead"
RMS_LOOKAHEAD_STD_KEY = "rms_std_by_lookahead"
RMS_LAYOUT_KEY = "rms_layout"
RMS_LAYOUT_ISSUE_LOOKAHEAD = "issuance_lookahead"


def build_global_rms_arrays(
    rms_mean_by_lookahead: np.ndarray,
    rms_std_by_lookahead: np.ndarray,
    issuance_idx: int,
    total_time_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rms_mean = np.full(total_time_steps, np.nan, dtype=float)
    rms_std = np.full(total_time_steps, np.nan, dtype=float)
    max_len = min(len(rms_mean_by_lookahead), total_time_steps - issuance_idx)
    if max_len > 0:
        rms_mean[issuance_idx:issuance_idx + max_len] = np.asarray(
            rms_mean_by_lookahead[:max_len], dtype=float
        )
        rms_std[issuance_idx:issuance_idx + max_len] = np.asarray(
            rms_std_by_lookahead[:max_len], dtype=float
        )
    return rms_mean, rms_std


def rms_uses_issuance_lookahead(forecast_state: Optional[Dict[str, Any]]) -> bool:
    if not forecast_state:
        return False
    return (
        forecast_state.get(RMS_LAYOUT_KEY) == RMS_LAYOUT_ISSUE_LOOKAHEAD
        or RMS_LOOKAHEAD_MEAN_KEY in forecast_state
    )


def get_rms_forecast_value(
    forecast_state: Optional[Dict[str, Any]],
    issuance_idx: int,
    target_idx: int,
) -> Tuple[float, float]:
    if forecast_state is None or target_idx < issuance_idx:
        return float("nan"), float("nan")

    if rms_uses_issuance_lookahead(forecast_state):
        lookahead = target_idx - issuance_idx
        rms_mean = forecast_state.get(RMS_LOOKAHEAD_MEAN_KEY)
        rms_std = forecast_state.get(RMS_LOOKAHEAD_STD_KEY)
        if rms_mean is None or rms_std is None or lookahead < 0 or lookahead >= len(rms_mean):
            return float("nan"), float("nan")
        return float(rms_mean[lookahead]), float(rms_std[lookahead])

    rms_mean = forecast_state.get("rms_mean")
    rms_std = forecast_state.get("rms_std")
    if rms_mean is None or rms_std is None or target_idx >= len(rms_mean):
        return float("nan"), float("nan")
    return float(rms_mean[target_idx]), float(rms_std[target_idx])


def get_rms_forecast_slice(
    forecast_state: Optional[Dict[str, Any]],
    issuance_idx: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if end_idx <= start_idx:
        return np.array([], dtype=float), np.array([], dtype=float)

    if forecast_state is None:
        n = end_idx - start_idx
        return np.full(n, np.nan, dtype=float), np.full(n, np.nan, dtype=float)

    if rms_uses_issuance_lookahead(forecast_state):
        start_lookahead = max(0, start_idx - issuance_idx)
        end_lookahead = max(start_lookahead, end_idx - issuance_idx)
        rms_mean = np.asarray(forecast_state.get(RMS_LOOKAHEAD_MEAN_KEY, []), dtype=float)
        rms_std = np.asarray(forecast_state.get(RMS_LOOKAHEAD_STD_KEY, []), dtype=float)
        n = end_idx - start_idx
        out_mean = np.full(n, np.nan, dtype=float)
        out_std = np.full(n, np.nan, dtype=float)
        src_start = max(0, start_lookahead)
        src_end = min(len(rms_mean), end_lookahead)
        if src_end > src_start:
            dst_start = max(0, issuance_idx - start_idx)
            dst_end = dst_start + (src_end - src_start)
            out_mean[dst_start:dst_end] = rms_mean[src_start:src_end]
            out_std[dst_start:dst_end] = rms_std[src_start:src_end]
        return out_mean, out_std

    rms_mean = np.asarray(forecast_state.get("rms_mean", []), dtype=float)
    rms_std = np.asarray(forecast_state.get("rms_std", []), dtype=float)
    return rms_mean[start_idx:end_idx], rms_std[start_idx:end_idx]


def slice_weather_forecast_for_period(
    forecast_state: Dict[str, Any],
    issuance_idx: int,
    start_idx: int,
    end_idx: int,
) -> Dict[str, Any]:
    sliced: Dict[str, Any] = {}
    local_issuance_idx = issuance_idx - start_idx
    for key, value in forecast_state.items():
        if key in (RMS_LOOKAHEAD_MEAN_KEY, RMS_LOOKAHEAD_STD_KEY):
            remaining = max(0, end_idx - issuance_idx)
            sliced[key] = np.asarray(value[:remaining], dtype=float).copy()
        elif key in {"pwv_mean", "pwv_std", "rms_mean", "rms_std"}:
            sliced[key] = np.asarray(value[start_idx:end_idx], dtype=float).copy()
        else:
            sliced[key] = value
    sliced["forecast_issue_idx"] = local_issuance_idx
    return sliced
