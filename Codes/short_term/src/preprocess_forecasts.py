"""
Preprocess weather forecasts and save them for consumption by full_year.py.

Real mode builds:
1. Time-evolving GFS PWV forecasts with climatology fallback.
2. RMS forecasts from a rolling UnobservedComponents model fit on the immediately
   prior cycle, plus a cross-validated historical-mean baseline.

Perfect mode overwrites both PWV and RMS forecasts with realized weather while
retaining the same forecast dictionary layout.
"""

import argparse
import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit

try:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
except Exception:
    UnobservedComponents = None

from weather_forecast_layout import (
    RMS_LAYOUT_ISSUE_LOOKAHEAD,
    RMS_LAYOUT_KEY,
    RMS_LOOKAHEAD_MEAN_KEY,
    RMS_LOOKAHEAD_STD_KEY,
    build_global_rms_arrays,
)

TIME_INTERVAL_MINUTES = 30
MIN_TRAINING_SLOTS_FOR_CYCLE = 1000
RMS_TRANSFORM_SCALE = 1200.0
RMS_TRANSFORM_EPS = 1e-6
RMS_UCM_SEASONAL_PERIOD = 48
RMS_UCM_SEASONAL_HARMONICS = 5
RMS_UCM_REFIT_EVERY_STEPS_DEFAULT = 48
FORCED_RMS_SOURCE_CYCLE_BY_START_YEAR = {
    2017: 2023,
    2021: 2023,
}

HARDCODED_SHUTDOWNS: dict = {
    2018: (2, 1, 3, 5),
    2019: (1, 29, 2, 28),
    2022: (1, 31, 2, 28),
    2023: (2, 1, 2, 28),
    2024: (1, 30, 2, 29),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess weather forecasts (real or perfect/oracle)."
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Directory with phase tracker, GFS, shifts, and climatology CSVs",
    )
    p.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--output", required=True, help="Path to save the output pickle file")
    p.add_argument(
        "--perfect_forecast",
        action="store_true",
        help="Replace forecasts with realized weather (oracle mode)",
    )
    p.add_argument(
        "--forecast_noise_std",
        type=float,
        default=0.0,
        help="Additive Gaussian noise std (only used with --perfect_forecast)",
    )
    p.add_argument(
        "--preprocessed_weather",
        type=str,
        default=None,
        help="Path to realized-weather pickle from preprocess_weather.py "
        "(required when --perfect_forecast is set)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rms_forecast_horizon_hours", type=float, default=16.0)
    p.add_argument("--rms_ar_order", type=int, default=4, help=argparse.SUPPRESS)
    p.add_argument("--rms_diff_order", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--rms_ma_order", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--rms_daily_fourier_order", type=int, default=4, help=argparse.SUPPRESS)
    p.add_argument("--rms_yearly_fourier_order", type=int, default=2, help=argparse.SUPPRESS)
    p.add_argument(
        "--rms_ucm_refit_every_steps",
        type=int,
        default=RMS_UCM_REFIT_EVERY_STEPS_DEFAULT,
        help="Refit the RMS UnobservedComponents model every N sequential updates.",
    )
    return p.parse_args()


def build_timeline(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.DatetimeIndex, Dict[int, pd.Timestamp]]:
    all_slots = pd.date_range(
        start=start_date,
        end=end_date,
        freq=f"{TIME_INTERVAL_MINUTES}min",
    )
    idx_to_timestamp = {i: ts for i, ts in enumerate(all_slots)}
    return all_slots, idx_to_timestamp


def load_stats_by_day(path: str, mean_col: str, std_col: str) -> pd.DataFrame:
    stats_by_day = pd.read_csv(path, index_col=["month", "day", "time_interval_utc"])
    stats_by_day.index = pd.MultiIndex.from_arrays(
        [
            stats_by_day.index.get_level_values("month"),
            stats_by_day.index.get_level_values("day"),
            pd.to_datetime(
                stats_by_day.index.get_level_values("time_interval_utc"),
                format="%H:%M:%S",
            ).time,
        ],
        names=["month", "day", "time_interval_utc"],
    )
    return stats_by_day[[mean_col, std_col]].copy()


def stats_to_timeline_arrays(
    stats_by_day: pd.DataFrame,
    all_slots: pd.DatetimeIndex,
    mean_col: str,
    std_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    mean_arr = np.full(len(all_slots), np.nan, dtype=float)
    std_arr = np.full(len(all_slots), np.nan, dtype=float)
    for idx, ts in enumerate(all_slots):
        try:
            stats = stats_by_day.loc[(ts.month, ts.day, ts.time())]
            mean_arr[idx] = float(stats[mean_col])
            std_arr[idx] = float(stats[std_col])
        except KeyError:
            pass
    return mean_arr, std_arr


def indices_from_intervals(intervals_df: pd.DataFrame, all_slots: pd.DatetimeIndex) -> Set[int]:
    indices: Set[int] = set()
    for _, row in intervals_df.iterrows():
        mask = (all_slots >= row["START_TIME"]) & (all_slots < row["END_TIME"])
        indices.update(np.where(mask)[0].tolist())
    return indices


def build_known_unavailable_mask(
    data_dir: str,
    all_slots: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> np.ndarray:
    shifts_path = os.path.join(data_dir, "shifts_dimensions.csv")
    shifts_df = pd.read_csv(shifts_path)
    shifts_df["START_TIME"] = pd.to_datetime(shifts_df["START_TIME"], utc=True)
    shifts_df["END_TIME"] = pd.to_datetime(shifts_df["END_TIME"], utc=True)
    engineering_intervals = shifts_df[shifts_df["SHIFT_ACTIVITY"].isin(["Engineering", "EOC"])]
    engineering_indices = indices_from_intervals(engineering_intervals, all_slots)

    shutdown_indices: Set[int] = set()
    for year, (sm, sd, em, ed) in HARDCODED_SHUTDOWNS.items():
        shutdown_start = pd.Timestamp(year=year, month=sm, day=sd, tz="UTC")
        try:
            shutdown_end = pd.Timestamp(year=year, month=em, day=ed + 1, tz="UTC")
        except ValueError:
            shutdown_end = pd.Timestamp(year=year, month=em + 1, day=1, tz="UTC")
        overlap_start = max(shutdown_start, start_date)
        overlap_end = min(shutdown_end, end_date + pd.Timedelta(days=1))
        if overlap_start >= overlap_end:
            continue
        mask = (all_slots >= shutdown_start) & (all_slots < shutdown_end)
        shutdown_indices.update(np.where(mask)[0].tolist())

    mask_arr = np.zeros(len(all_slots), dtype=bool)
    for idx in engineering_indices | shutdown_indices:
        if 0 <= idx < len(mask_arr):
            mask_arr[idx] = True
    return mask_arr


def load_phase_tracker_with_freqrms(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "phase_tracker_data_distinct_prepared.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    allowed_phase_err_lower = 45.83662
    rms_microns = df["array_characteristic_phase_rms_microns"].replace(0, np.nan)
    df["freqrms"] = (3.0e8 / (rms_microns * 1.0e-6 * 360.0 / allowed_phase_err_lower)) / 1e9
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df[["timestamp", "freqrms"]]


def realized_rms_on_timeline(phase_df: pd.DataFrame, all_slots: pd.DatetimeIndex) -> np.ndarray:
    rms_series = phase_df.set_index("timestamp")["freqrms"]
    tolerance = pd.Timedelta(minutes=30)
    rms_aligned = rms_series.reindex(all_slots, method="nearest", tolerance=tolerance)
    return rms_aligned.to_numpy(dtype=float)


def build_train_climatology(train_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        train_df.groupby(["month", "day", "time_interval_utc"], dropna=False)["actual_rms"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_rms_microns",
                "std": "std_rms_microns",
                "count": "data_points_across_years",
            }
        )
    )
    grouped["time_interval_utc"] = pd.to_datetime(
        grouped["time_interval_utc"], format="%H:%M:%S"
    ).dt.time
    grouped = grouped.set_index(["month", "day", "time_interval_utc"]).sort_index()
    return grouped


def fill_rms_climatology_from_training(
    train_climatology: pd.DataFrame,
    all_slots: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    mean_arr = np.full(len(all_slots), np.nan, dtype=float)
    std_arr = np.full(len(all_slots), np.nan, dtype=float)
    for idx, ts in enumerate(all_slots):
        key = (ts.month, ts.day, ts.time())
        if key in train_climatology.index:
            stats = train_climatology.loc[key]
            mean_arr[idx] = float(stats["mean_rms_microns"])
            std_arr[idx] = float(stats["std_rms_microns"])
    return mean_arr, std_arr


def _clip_rms_for_transform(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        arr[finite_mask] = np.clip(
            arr[finite_mask],
            RMS_TRANSFORM_EPS,
            RMS_TRANSFORM_SCALE - RMS_TRANSFORM_EPS,
        )
    return arr


def fwd_trans(values: np.ndarray) -> np.ndarray:
    arr = _clip_rms_for_transform(values)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        out[finite_mask] = logit(arr[finite_mask] / RMS_TRANSFORM_SCALE)
    return out


def back_trans(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(arr)
    if finite_mask.any():
        out[finite_mask] = RMS_TRANSFORM_SCALE * expit(arr[finite_mask])
    return out


def build_prior_cycle_training_series(
    full_rms_df: pd.DataFrame,
    full_slots: pd.DatetimeIndex,
    full_actual_rms: np.ndarray,
    train_climatology: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    current_cycle_year = start_date.year
    desired_cycle_year = start_date.year - 1
    prior_start = start_date - pd.DateOffset(years=1)
    prior_end = end_date - pd.DateOffset(years=1)

    target_mask = np.asarray((full_slots >= prior_start) & (full_slots <= prior_end), dtype=bool)
    selected_cycle_year = desired_cycle_year
    selected_start = prior_start
    selected_end = prior_end
    train_mask = target_mask
    used_fallback_cycle = False
    selection_mode = "direct_prior_window"
    forced_source_cycle_year = FORCED_RMS_SOURCE_CYCLE_BY_START_YEAR.get(current_cycle_year)

    if forced_source_cycle_year is not None:
        offset_years = forced_source_cycle_year - desired_cycle_year
        selected_cycle_year = forced_source_cycle_year
        selected_start = prior_start + pd.DateOffset(years=offset_years)
        selected_end = prior_end + pd.DateOffset(years=offset_years)
        train_mask = np.asarray((full_slots >= selected_start) & (full_slots <= selected_end), dtype=bool)
        used_fallback_cycle = selected_cycle_year != desired_cycle_year
        selection_mode = "forced_source_cycle"
        if int(train_mask.sum()) == 0:
            raise RuntimeError(
                f"Forced RMS source cycle {selected_cycle_year}-{selected_cycle_year + 1} "
                "has no slots available for UCM training."
            )
    elif int(target_mask.sum()) < MIN_TRAINING_SLOTS_FOR_CYCLE:
        min_year = int(full_slots.min().year) - 1
        max_year = int(full_slots.max().year)
        candidate_years: List[Tuple[int, int]] = []
        for cycle_year in range(min_year, max_year + 1):
            if cycle_year == current_cycle_year:
                continue
            offset_years = cycle_year - desired_cycle_year
            candidate_start = prior_start + pd.DateOffset(years=offset_years)
            candidate_end = prior_end + pd.DateOffset(years=offset_years)
            candidate_mask = np.asarray(
                (full_slots >= candidate_start) & (full_slots <= candidate_end),
                dtype=bool,
            )
            slot_count = int(candidate_mask.sum())
            if slot_count > 0:
                candidate_years.append((cycle_year, slot_count))

        if not candidate_years:
            raise RuntimeError("No non-current-cycle RMS slots available for UCM training.")

        viable_years = [item for item in candidate_years if item[1] >= MIN_TRAINING_SLOTS_FOR_CYCLE]
        ranked_years = viable_years if viable_years else candidate_years
        selected_cycle_year, _ = min(
            ranked_years,
            key=lambda item: (abs(item[0] - desired_cycle_year), -item[1], item[0]),
        )
        offset_years = selected_cycle_year - desired_cycle_year
        selected_start = prior_start + pd.DateOffset(years=offset_years)
        selected_end = prior_end + pd.DateOffset(years=offset_years)
        train_mask = np.asarray((full_slots >= selected_start) & (full_slots <= selected_end), dtype=bool)
        used_fallback_cycle = selected_cycle_year != desired_cycle_year
        selection_mode = "automatic_fallback"

    train_slots_source = pd.DatetimeIndex(full_slots[train_mask])
    if len(train_slots_source) == 0:
        raise RuntimeError("No usable RMS slots available for UCM training.")

    shift_years = desired_cycle_year - selected_cycle_year
    train_slots_prior_cycle = pd.DatetimeIndex(
        [ts + pd.DateOffset(years=shift_years) for ts in train_slots_source]
    )
    source_cycle_label = f"{selected_cycle_year}-{selected_cycle_year + 1}"
    forced_source_label = (
        f"{forced_source_cycle_year}-{forced_source_cycle_year + 1}"
        if forced_source_cycle_year is not None
        else "none"
    )

    full_hist_mean, _ = fill_rms_climatology_from_training(train_climatology, full_slots)
    train_rms_prior_cycle = np.asarray(full_actual_rms[train_mask], dtype=float).copy()
    train_hist_mean = np.asarray(full_hist_mean[train_mask], dtype=float)

    valid_train_values = full_rms_df.loc[
        train_mask & full_rms_df["is_valid"].to_numpy(dtype=bool),
        "actual_rms",
    ].to_numpy(dtype=float)
    valid_train_values = valid_train_values[np.isfinite(valid_train_values)]
    global_fallback = float(np.nanmedian(valid_train_values)) if len(valid_train_values) else 0.0

    train_hist_mean[~np.isfinite(train_hist_mean)] = global_fallback
    missing_mask = ~np.isfinite(train_rms_prior_cycle)
    train_rms_prior_cycle[missing_mask] = np.nan
    train_rms_prior_cycle[~np.isfinite(train_rms_prior_cycle)] = np.nan

    print("\n--- Step 1B: Build prior-cycle RMS training series ---", flush=True)
    print(f"  Desired prior cycle window     : {prior_start} -> {prior_end}", flush=True)
    print(f"  Desired prior cycle year       : {desired_cycle_year}", flush=True)
    print(f"  Target prior-window slots      : {int(target_mask.sum())}", flush=True)
    print(f"  Selected source cycle window   : {selected_start} -> {selected_end}", flush=True)
    print(f"  Source cycle year              : {selected_cycle_year}", flush=True)
    print(f"  Source cycle label             : {source_cycle_label}", flush=True)
    print(f"  Source selection mode          : {selection_mode}", flush=True)
    print(f"  Forced source cycle            : {forced_source_label}", flush=True)
    print(f"  Used fallback cycle            : {'yes' if used_fallback_cycle else 'no'}", flush=True)
    print(f"  Shift years applied            : {shift_years}", flush=True)
    print(f"  Prior-cycle training slots     : {len(train_slots_prior_cycle)}", flush=True)
    print(f"  Training slot range            : {train_slots_prior_cycle.min()} -> {train_slots_prior_cycle.max()}", flush=True)
    print(f"  Global RMS fallback             : {global_fallback:.6f}", flush=True)
    print(f"  Finite prior-cycle observations : {int(np.isfinite(full_actual_rms[train_mask]).sum())}", flush=True)

    return train_slots_prior_cycle, train_rms_prior_cycle


def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    if not y_true:
        return {
            "n_eval_slots": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias_pred_minus_actual": float("nan"),
        }
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    residual = pred - actual
    return {
        "n_eval_slots": int(len(actual)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_actual": float(np.mean(residual)),
    }


def evaluate_rms_forecasts_upto_issuance(
    per_issuance_preds: Dict[int, np.ndarray],
    hist_cycle_mean: np.ndarray,
    actual_rms_cycle: np.ndarray,
    horizon_steps: int,
    max_issuance_idx_inclusive: int,
    *,
    verbose: bool = False,
) -> Dict[str, Any]:
    model_by_lookahead: Dict[str, Dict[str, float]] = {}
    historical_by_lookahead: Dict[str, Dict[str, float]] = {}
    all_model_actuals: List[float] = []
    all_model_preds: List[float] = []
    all_hist_actuals: List[float] = []
    all_hist_preds: List[float] = []

    if max_issuance_idx_inclusive < 0:
        return {
            "ucm_by_lookahead": model_by_lookahead,
            "historical_mean_by_lookahead": historical_by_lookahead,
            "ucm_overall_on_all_lookaheads": compute_metrics([], []),
            "historical_overall_on_all_lookaheads": compute_metrics([], []),
        }

    max_issuance_idx_inclusive = min(
        int(max_issuance_idx_inclusive),
        len(actual_rms_cycle) - 1,
    )

    for lookahead in range(1, horizon_steps + 1):
        model_actuals: List[float] = []
        model_preds: List[float] = []
        hist_actuals: List[float] = []
        hist_preds: List[float] = []

        max_issuance_for_lookahead = min(
            max_issuance_idx_inclusive,
            len(actual_rms_cycle) - lookahead - 1,
        )
        if max_issuance_for_lookahead < 0:
            model_metrics = compute_metrics([], [])
            hist_metrics = compute_metrics([], [])
            model_by_lookahead[str(lookahead)] = model_metrics
            historical_by_lookahead[str(lookahead)] = hist_metrics
            if verbose:
                print(
                    f"  Lookahead {lookahead:2d}: "
                    f"UCM RMSE={model_metrics['rmse']:.6f} (n={model_metrics['n_eval_slots']}), "
                    f"Hist RMSE={hist_metrics['rmse']:.6f} (n={hist_metrics['n_eval_slots']})",
                    flush=True,
                )
            continue

        for issuance_idx in range(max_issuance_for_lookahead + 1):
            target_idx = issuance_idx + lookahead
            actual = actual_rms_cycle[target_idx]
            issuance_preds = per_issuance_preds.get(issuance_idx)
            model_pred = (
                float(issuance_preds[lookahead])
                if issuance_preds is not None and lookahead < len(issuance_preds)
                else float("nan")
            )
            hist_pred = hist_cycle_mean[target_idx]
            if np.isfinite(actual) and np.isfinite(model_pred):
                model_actuals.append(float(actual))
                model_preds.append(float(model_pred))
            if np.isfinite(actual) and np.isfinite(hist_pred):
                hist_actuals.append(float(actual))
                hist_preds.append(float(hist_pred))

        model_metrics = compute_metrics(model_actuals, model_preds)
        hist_metrics = compute_metrics(hist_actuals, hist_preds)
        model_by_lookahead[str(lookahead)] = model_metrics
        historical_by_lookahead[str(lookahead)] = hist_metrics

        all_model_actuals.extend(model_actuals)
        all_model_preds.extend(model_preds)
        all_hist_actuals.extend(hist_actuals)
        all_hist_preds.extend(hist_preds)

        if verbose:
            print(
                f"  Lookahead {lookahead:2d}: "
                f"UCM RMSE={model_metrics['rmse']:.6f} (n={model_metrics['n_eval_slots']}), "
                f"Hist RMSE={hist_metrics['rmse']:.6f} (n={hist_metrics['n_eval_slots']})",
                flush=True,
            )

    return {
        "ucm_by_lookahead": model_by_lookahead,
        "historical_mean_by_lookahead": historical_by_lookahead,
        "ucm_overall_on_all_lookaheads": compute_metrics(all_model_actuals, all_model_preds),
        "historical_overall_on_all_lookaheads": compute_metrics(all_hist_actuals, all_hist_preds),
    }


def fit_ucm_rms_model(
    train_slots_prior_cycle: pd.DatetimeIndex,
    train_rms_prior_cycle: np.ndarray,
    args: argparse.Namespace,
):
    if UnobservedComponents is None:
        raise ImportError(
            "statsmodels is required for rolling UnobservedComponents RMS forecasts. "
            "Install with: pip install statsmodels"
        )

    if len(train_slots_prior_cycle) == 0:
        raise RuntimeError("No prior-cycle RMS training rows available for UCM.")

    transformed_train_rms = fwd_trans(np.asarray(train_rms_prior_cycle, dtype=float))
    finite_train_count = int(np.isfinite(train_rms_prior_cycle).sum())
    if finite_train_count == 0:
        raise RuntimeError("No finite prior-cycle RMS observations available for UCM.")

    print("\n--- Step 1C: Fit rolling UCM RMS model ---", flush=True)
    print(f"  Training rows used             : {len(train_rms_prior_cycle)}", flush=True)
    print(f"  Training ds range              : {train_slots_prior_cycle.min()} -> {train_slots_prior_cycle.max()}", flush=True)
    print(
        f"  Training y min/max             : "
        f"{float(np.nanmin(train_rms_prior_cycle)):.6f} / {float(np.nanmax(train_rms_prior_cycle)):.6f}",
        flush=True,
    )
    print(f"  Training finite observations   : {finite_train_count}", flush=True)
    print(
        "  UCM config                    : "
        f"level=llevel, seasonal_period={RMS_UCM_SEASONAL_PERIOD}, "
        f"seasonal_harmonics={RMS_UCM_SEASONAL_HARMONICS}, "
        f"refit_every_steps={int(args.rms_ucm_refit_every_steps)}",
        flush=True,
    )

    model = UnobservedComponents(
        endog=transformed_train_rms,
        level="llevel",
        freq_seasonal=[
            {
                "period": RMS_UCM_SEASONAL_PERIOD,
                "harmonics": RMS_UCM_SEASONAL_HARMONICS,
            }
        ],
    )
    result = model.fit(disp=False)
    print("  UCM fit complete.", flush=True)
    print(f"  UCM converged                  : {bool(result.mle_retvals.get('converged', False))}", flush=True)
    return result


def evaluate_rms_forecasts(
    per_issuance_preds: Dict[int, np.ndarray],
    hist_cycle_mean: np.ndarray,
    actual_rms_cycle: np.ndarray,
    horizon_steps: int,
) -> Dict[str, Any]:
    print("\n--- Step 1E: RMS forecast evaluation by lookahead ---", flush=True)
    return evaluate_rms_forecasts_upto_issuance(
        per_issuance_preds=per_issuance_preds,
        hist_cycle_mean=hist_cycle_mean,
        actual_rms_cycle=actual_rms_cycle,
        horizon_steps=horizon_steps,
        max_issuance_idx_inclusive=len(actual_rms_cycle) - 1,
        verbose=True,
    )


def build_ucm_rms_forecasts(
    all_slots: pd.DatetimeIndex,
    hist_rms_mean: np.ndarray,
    hist_rms_std: np.ndarray,
    actual_rms_cycle: np.ndarray,
    unavailable_mask: np.ndarray,
    fit_result,
    args: argparse.Namespace,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    horizon_steps = int(round(args.rms_forecast_horizon_hours * 60.0 / TIME_INTERVAL_MINUTES))
    horizon_with_now = max(1, horizon_steps + 1)
    time_steps = len(all_slots)
    refit_every_steps = max(1, int(args.rms_ucm_refit_every_steps))
    progress_every_steps = max(1, int(round(24.0 * 60.0 / TIME_INTERVAL_MINUTES)))
    weekly_eval_every_steps = 7 * progress_every_steps
    std_fallback = float(np.nanmedian(hist_rms_std[np.isfinite(hist_rms_std)])) if np.isfinite(hist_rms_std).any() else 0.1
    rms_fallback = float(np.nanmedian(hist_rms_mean[np.isfinite(hist_rms_mean)])) if np.isfinite(hist_rms_mean).any() else 0.0
    refit_fit_kwargs = {"disp": False}

    print("\n--- Step 1D: Build rolling UCM RMS cycle forecasts ---", flush=True)
    print(f"  RMS forecast horizon (hours)   : {args.rms_forecast_horizon_hours}", flush=True)
    print(f"  RMS forecast horizon (steps)   : {horizon_steps}", flush=True)
    print(f"  Lookahead array includes t_now : yes", flush=True)
    print(f"  UCM refit cadence (steps)      : {refit_every_steps}", flush=True)
    print(f"  Progress log cadence (steps)   : {progress_every_steps}", flush=True)
    print(f"  Weekly RMSE cadence (steps)    : {weekly_eval_every_steps}", flush=True)
    print(f"  RMS std fallback               : {std_fallback:.6f}", flush=True)
    print(f"  RMS mean fallback              : {rms_fallback:.6f}", flush=True)

    hist_std_safe = hist_rms_std.copy()
    hist_std_safe[~np.isfinite(hist_std_safe)] = std_fallback
    hist_mean_safe = hist_rms_mean.copy()
    hist_mean_safe[~np.isfinite(hist_mean_safe)] = rms_fallback

    weather_rms_forecasts: Dict[int, Dict[str, Any]] = {}
    per_issuance_preds: Dict[int, np.ndarray] = {}
    rolling_result = fit_result
    ucm_slot_count = 0
    extended_fallback_count = 0
    refit_count = 0
    updates_since_refit = 0
    rolling_weekly_evaluation: List[Dict[str, Any]] = []

    for issuance_idx in range(time_steps):
        remaining_steps = time_steps - issuance_idx
        forecast_count = min(remaining_steps, horizon_with_now)
        pred_trans = np.asarray(
            rolling_result.forecast(steps=forecast_count),
            dtype=float,
        )
        rms_mean_rel = np.maximum(back_trans(pred_trans), 0.0)
        rms_std_rel = hist_std_safe[issuance_idx:issuance_idx + forecast_count].copy()
        mask_rel = unavailable_mask[issuance_idx:issuance_idx + forecast_count]
        rms_mean_rel[mask_rel] = np.nan
        rms_std_rel[mask_rel] = np.nan
        per_issuance_preds[issuance_idx] = rms_mean_rel.copy()
        rms_mean_full, rms_std_full = build_global_rms_arrays(
            rms_mean_rel, rms_std_rel, issuance_idx, time_steps
        )
        weather_rms_forecasts[issuance_idx] = {
            "rms_mean": rms_mean_full,
            "rms_std": rms_std_full,
            RMS_LOOKAHEAD_MEAN_KEY: rms_mean_rel,
            RMS_LOOKAHEAD_STD_KEY: rms_std_rel,
            RMS_LAYOUT_KEY: RMS_LAYOUT_ISSUE_LOOKAHEAD,
            "forecast_issue_idx": issuance_idx,
            "rms_prophet_horizon_steps": horizon_steps,
        }
        ucm_slot_count += forecast_count

        observed = float(actual_rms_cycle[issuance_idx])
        if not np.isfinite(observed):
            observed = float(hist_mean_safe[issuance_idx])
            extended_fallback_count += 1
        transformed_observed = fwd_trans(np.array([observed], dtype=float))
        updates_since_refit += 1
        should_refit = updates_since_refit >= refit_every_steps
        if should_refit:
            rolling_result = rolling_result.append(
                transformed_observed,
                refit=True,
                fit_kwargs=refit_fit_kwargs,
            )
        else:
            rolling_result = rolling_result.append(
                transformed_observed,
                refit=False,
            )
        if should_refit:
            refit_count += 1
            updates_since_refit = 0

        should_log_progress = (
            issuance_idx == 0
            or issuance_idx == time_steps - 1
            or ((issuance_idx + 1) % progress_every_steps == 0)
        )
        if should_log_progress:
            pct_complete = 100.0 * float(issuance_idx + 1) / float(max(1, time_steps))
            last_refit_str = "yes" if should_refit else "no"
            print(
                f"  [UCM-RMS] progress: "
                f"t={issuance_idx + 1}/{time_steps} "
                f"({pct_complete:.1f}%) "
                f"ts={all_slots[issuance_idx]} "
                f"forecasts_built={issuance_idx + 1} "
                f"refits={refit_count} "
                f"fallbacks={extended_fallback_count} "
                f"last_refit={last_refit_str}",
                flush=True,
            )

        should_run_weekly_eval = (
            ((issuance_idx + 1) % weekly_eval_every_steps == 0)
            or issuance_idx == time_steps - 1
        )
        if should_run_weekly_eval:
            snapshot_eval = evaluate_rms_forecasts_upto_issuance(
                per_issuance_preds=per_issuance_preds,
                hist_cycle_mean=hist_rms_mean,
                actual_rms_cycle=actual_rms_cycle,
                horizon_steps=horizon_steps,
                max_issuance_idx_inclusive=issuance_idx,
                verbose=False,
            )
            pct_complete = 100.0 * float(issuance_idx + 1) / float(max(1, time_steps))
            snapshot_entry = {
                "issuance_idx": int(issuance_idx),
                "timestamp": all_slots[issuance_idx],
                "pct_complete": float(pct_complete),
                "ucm_overall_on_all_lookaheads": snapshot_eval["ucm_overall_on_all_lookaheads"],
                "historical_overall_on_all_lookaheads": snapshot_eval["historical_overall_on_all_lookaheads"],
                "ucm_by_lookahead": snapshot_eval["ucm_by_lookahead"],
                "historical_mean_by_lookahead": snapshot_eval["historical_mean_by_lookahead"],
            }
            rolling_weekly_evaluation.append(snapshot_entry)
            ucm_overall = snapshot_entry["ucm_overall_on_all_lookaheads"]
            hist_overall = snapshot_entry["historical_overall_on_all_lookaheads"]
            print(
                f"  [UCM-RMS] weekly_eval: "
                f"t={issuance_idx + 1}/{time_steps} "
                f"ts={all_slots[issuance_idx]} "
                f"ucm_rmse={ucm_overall['rmse']:.6f} "
                f"hist_rmse={hist_overall['rmse']:.6f} "
                f"n={ucm_overall['n_eval_slots']}",
                flush=True,
            )

    print(f"  UCM-issued RMS slots            : {ucm_slot_count}", flush=True)
    print(f"  Extended fallback observations  : {extended_fallback_count}", flush=True)
    print(f"  Sequential parameter refits     : {refit_count}", flush=True)

    rms_eval = evaluate_rms_forecasts(
        per_issuance_preds=per_issuance_preds,
        hist_cycle_mean=hist_rms_mean,
        actual_rms_cycle=actual_rms_cycle,
        horizon_steps=horizon_steps,
    )
    rms_eval["rolling_weekly_evaluation"] = rolling_weekly_evaluation
    rms_eval["model_config"] = {
        "engine": "unobserved_components",
        "forecast_horizon_hours": float(args.rms_forecast_horizon_hours),
        "forecast_horizon_steps": int(horizon_steps),
        "lookahead_array_includes_current_timestep": True,
        "refit_every_steps": int(refit_every_steps),
        "transform": {
            "forward": "logit(x / 1200)",
            "backward": "1200 * expit(x)",
            "scale": RMS_TRANSFORM_SCALE,
            "epsilon": RMS_TRANSFORM_EPS,
        },
        "level": "llevel",
        "freq_seasonal": [
            {
                "period": RMS_UCM_SEASONAL_PERIOD,
                "harmonics": RMS_UCM_SEASONAL_HARMONICS,
            }
        ],
        "n_training_slots": int(len(fit_result.model.endog)),
        "training_window": "prior_cycle_only",
        "fit_converged": bool(fit_result.mle_retvals.get("converged", False)),
        "rms_layout": RMS_LAYOUT_ISSUE_LOOKAHEAD,
        "rms_mean_source": "rolling_ucm_forecast",
        "rms_std_source": "historical_climatology_for_each_issuance_lookahead",
        "beyond_horizon_behavior": "not_populated_beyond_requested_horizon",
    }
    return weather_rms_forecasts, rms_eval


def main():
    args = parse_args()
    data_dir = args.data_dir

    if args.perfect_forecast and args.preprocessed_weather is None:
        raise ValueError("--preprocessed_weather is required when --perfect_forecast is set.")

    start_date = pd.Timestamp(args.start_date, tz="UTC")
    end_date = pd.Timestamp(args.end_date, tz="UTC")

    print("=" * 70, flush=True)
    print("WEATHER FORECAST PREPROCESSING", flush=True)
    print("=" * 70, flush=True)
    print(f"Date range : {start_date.date()} to {end_date.date()}", flush=True)
    print(f"Mode       : {'PERFECT (oracle)' if args.perfect_forecast else 'REAL (GFS-based)'}", flush=True)

    all_slots, idx_to_timestamp = build_timeline(start_date, end_date)
    time_steps = len(all_slots)
    print(f"Timeline   : {time_steps} slots ({TIME_INTERVAL_MINUTES}-min bins)", flush=True)

    unavailable_mask = build_known_unavailable_mask(data_dir, all_slots, start_date, end_date)
    print(f"Known unavailable slots in cycle: {int(unavailable_mask.sum())}", flush=True)

    # ---- Step 1: RMS historical baseline + rolling UCM forecasts ----
    rms_eval: Dict[str, Any] = {}
    weather_rms_forecasts: Dict[int, Dict[str, Any]] = {}
    rms_hist_mean = np.full(time_steps, np.nan, dtype=float)
    rms_hist_std = np.full(time_steps, np.nan, dtype=float)

    if not args.perfect_forecast:
        print("\n--- Step 1A: Build cross-validated RMS training data ---", flush=True)
        phase_df = load_phase_tracker_with_freqrms(data_dir)
        full_start = phase_df["timestamp"].min().floor(f"{TIME_INTERVAL_MINUTES}min")
        full_end = phase_df["timestamp"].max().ceil(f"{TIME_INTERVAL_MINUTES}min")
        full_slots, _ = build_timeline(full_start, full_end)
        full_actual_rms = realized_rms_on_timeline(phase_df, full_slots)
        full_unavailable_mask = build_known_unavailable_mask(data_dir, full_slots, full_start, full_end)

        full_rms_df = pd.DataFrame(
            {
                "timestamp": full_slots,
                "actual_rms": full_actual_rms,
                "is_unavailable": full_unavailable_mask,
            }
        )
        full_rms_df["is_current_cycle"] = (
            (full_rms_df["timestamp"] >= start_date) & (full_rms_df["timestamp"] <= end_date)
        )
        full_rms_df["is_valid"] = (~full_rms_df["is_unavailable"]) & np.isfinite(full_rms_df["actual_rms"])

        train_rms_df = full_rms_df[(~full_rms_df["is_current_cycle"]) & full_rms_df["is_valid"]].copy()
        train_rms_df["month"] = train_rms_df["timestamp"].dt.month
        train_rms_df["day"] = train_rms_df["timestamp"].dt.day
        train_rms_df["time_interval_utc"] = train_rms_df["timestamp"].dt.strftime("%H:%M:%S")
        print(f"  Full RMS rows on snapped timeline : {len(full_rms_df)}", flush=True)
        print(f"  Current-cycle rows excluded       : {int(full_rms_df['is_current_cycle'].sum())}", flush=True)
        print(f"  Unavailable rows excluded         : {int(full_rms_df['is_unavailable'].sum())}", flush=True)
        print(f"  Valid out-of-cycle rows used      : {len(train_rms_df)}", flush=True)

        train_climatology = build_train_climatology(train_rms_df)
        rms_hist_mean, rms_hist_std = fill_rms_climatology_from_training(train_climatology, all_slots)
        rms_hist_mean[unavailable_mask] = np.nan
        rms_hist_std[unavailable_mask] = np.nan
        print(f"  Cross-validated RMS climatology   : {int(np.isfinite(rms_hist_mean).sum())}/{time_steps} filled", flush=True)

        actual_rms_cycle = realized_rms_on_timeline(phase_df, all_slots)
        actual_rms_cycle[unavailable_mask] = np.nan
        train_slots_prior_cycle, train_rms_prior_cycle = build_prior_cycle_training_series(
            full_rms_df=full_rms_df,
            full_slots=full_slots,
            full_actual_rms=full_actual_rms,
            train_climatology=train_climatology,
            start_date=start_date,
            end_date=end_date,
        )
        fit_result = fit_ucm_rms_model(train_slots_prior_cycle, train_rms_prior_cycle, args)
        weather_rms_forecasts, rms_eval = build_ucm_rms_forecasts(
            all_slots=all_slots,
            hist_rms_mean=rms_hist_mean,
            hist_rms_std=rms_hist_std,
            actual_rms_cycle=actual_rms_cycle,
            unavailable_mask=unavailable_mask,
            fit_result=fit_result,
            args=args,
        )
    else:
        print("\n--- Step 1: RMS UCM forecast build skipped in perfect mode ---", flush=True)

    # ---- Step 2: Historical PWV forecasts (climatological fallback) ----
    print("\n--- Step 2: Historical PWV forecasts (fallback) ---", flush=True)
    pwv_stats_path = os.path.join(data_dir, "pwv_stats_by_day.csv")
    pwv_stats_by_day = load_stats_by_day(pwv_stats_path, "mean_pwv", "std_pwv")
    pwv_mean_historical, pwv_std_historical = stats_to_timeline_arrays(
        pwv_stats_by_day, all_slots, "mean_pwv", "std_pwv"
    )
    print(f"  PWV historical mean filled: {int(np.isfinite(pwv_mean_historical).sum())}/{time_steps}", flush=True)

    # ---- Step 3: GFS PWV forecasts ----
    print("\n--- Step 3: GFS PWV forecasts ---", flush=True)
    gfs_path = os.path.join(data_dir, "gfs_pwv_combined_data.csv")
    gfs_pwv_forecasts = pd.read_csv(gfs_path)
    gfs_pwv_forecasts["run_time"] = pd.to_datetime(gfs_pwv_forecasts["run_time"], utc=True)
    gfs_pwv_forecasts["valid_time"] = pd.to_datetime(gfs_pwv_forecasts["valid_time"], utc=True)

    min_schedule_time = all_slots[0]
    max_schedule_time = all_slots[-1]
    gfs_pwv_prepared = gfs_pwv_forecasts[
        (gfs_pwv_forecasts["run_time"] <= max_schedule_time)
        & (gfs_pwv_forecasts["valid_time"] >= min_schedule_time)
        & (gfs_pwv_forecasts["valid_time"] <= max_schedule_time)
    ].copy()
    print(f"  Filtered GFS records for simulation window: {len(gfs_pwv_prepared)}", flush=True)

    all_gfs_interpolated = pd.DataFrame(index=all_slots)
    for run_time, group in gfs_pwv_prepared.groupby("run_time"):
        print(f"  Interpolating GFS run {run_time}", flush=True)
        run_series = group.set_index("valid_time")["pwv_value"]
        interpolated = run_series.reindex(all_slots).interpolate(
            method="linear", limit_direction="forward", limit=6
        )
        all_gfs_interpolated[run_time] = interpolated

    # ---- Step 4: Pre-calculate combined PWV forecasts at each GFS run time ----
    print("\n--- Step 4: Pre-calculate combined PWV forecasts ---", flush=True)
    precalculated_pwv_forecasts = {}
    unique_run_times = sorted(all_gfs_interpolated.columns)
    print(f"  Number of GFS run times: {len(unique_run_times)}", flush=True)
    for run_time in unique_run_times:
        cols_to_use = [rt for rt in unique_run_times if rt <= run_time]
        relevant_df = all_gfs_interpolated[cols_to_use]
        precalculated_pwv_forecasts[run_time] = {
            "pwv_mean": relevant_df.mean(axis=1, skipna=True).values,
            "pwv_mean_latest": all_gfs_interpolated[run_time].values.copy(),
            "pwv_std": relevant_df.std(axis=1, skipna=True).values,
        }

    # ---- Step 5: Assemble final forecast dict ----
    print("\n--- Step 5: Assemble final forecast dict ---", flush=True)
    print("  PWV mean mode: LATEST single GFS run", flush=True)
    weather_forecasts: Dict[int, Dict[str, Any]] = {}
    last_known_gfs_forecast = {
        "pwv_mean": np.full(time_steps, np.nan),
        "pwv_mean_latest": np.full(time_steps, np.nan),
        "pwv_std": np.full(time_steps, np.nan),
    }
    run_times_array = np.array(unique_run_times)
    n_gfs_used = 0
    n_climatology_fallback = 0

    for t_now_idx in range(time_steps):
        t_now_ts = idx_to_timestamp[t_now_idx]
        available_indices = np.where(run_times_array <= t_now_ts)[0]
        if available_indices.size > 0:
            most_recent_run_time = run_times_array[available_indices[-1]]
            last_known_gfs_forecast = precalculated_pwv_forecasts[most_recent_run_time]

        final_pwv_mean = last_known_gfs_forecast.get(
            "pwv_mean_latest", last_known_gfs_forecast["pwv_mean"]
        ).copy()
        final_pwv_std = last_known_gfs_forecast["pwv_std"].copy()
        nan_mask_mean = np.isnan(final_pwv_mean)
        nan_mask_std = np.isnan(final_pwv_std)
        n_climatology_fallback += int(np.sum(nan_mask_mean))
        n_gfs_used += int(np.sum(~nan_mask_mean))
        final_pwv_mean[nan_mask_mean] = pwv_mean_historical[nan_mask_mean]
        final_pwv_std[nan_mask_std] = pwv_std_historical[nan_mask_std]

        final_pwv_mean = final_pwv_mean.copy()
        final_pwv_std = final_pwv_std.copy()
        final_pwv_mean[unavailable_mask] = np.nan
        final_pwv_std[unavailable_mask] = np.nan

        weather_forecasts[t_now_idx] = {
            "pwv_mean": final_pwv_mean,
            "pwv_std": final_pwv_std,
        }
        if not args.perfect_forecast:
            weather_forecasts[t_now_idx].update(weather_rms_forecasts[t_now_idx])

    print(f"  Total slot-forecast entries: {len(weather_forecasts)}", flush=True)
    print(f"  GFS-filled entries (across all slots): {n_gfs_used}", flush=True)
    print(f"  Climatology-fallback entries:          {n_climatology_fallback}", flush=True)

    # ---- Step 6 (optional): Perfect forecast override ----
    if args.perfect_forecast:
        print("\n--- Step 6: Perfect (oracle) forecast override ---", flush=True)
        print(f"  Loading realized weather from {args.preprocessed_weather}", flush=True)
        with open(args.preprocessed_weather, "rb") as f:
            pw_data = pickle.load(f)
        realized_weather = pw_data["realized_weather"]
        pwv_actual = np.array(
            [realized_weather.get(i, (np.nan, np.nan))[0] for i in range(time_steps)],
            dtype=float,
        )
        rms_actual = np.array(
            [realized_weather.get(i, (np.nan, np.nan))[1] for i in range(time_steps)],
            dtype=float,
        )
        nan_mask = np.isnan(pwv_actual) | np.isnan(rms_actual)
        std_arr = np.full(time_steps, max(args.forecast_noise_std, 0.001), dtype=float)
        std_arr[nan_mask] = np.nan
        for t_idx in range(time_steps):
            rms_mean_rel = rms_actual[t_idx:].copy()
            rms_std_rel = std_arr[t_idx:].copy()
            rms_mean_full, rms_std_full = build_global_rms_arrays(
                rms_mean_rel, rms_std_rel, t_idx, time_steps
            )
            weather_forecasts[t_idx] = {
                "pwv_mean": pwv_actual.copy(),
                "pwv_std": std_arr.copy(),
                "rms_mean": rms_mean_full,
                "rms_std": rms_std_full,
                RMS_LOOKAHEAD_MEAN_KEY: rms_mean_rel,
                RMS_LOOKAHEAD_STD_KEY: rms_std_rel,
                RMS_LAYOUT_KEY: RMS_LAYOUT_ISSUE_LOOKAHEAD,
                "forecast_issue_idx": t_idx,
                "rms_prophet_horizon_steps": time_steps - t_idx - 1,
            }
        rms_eval = {
            "model_config": {
                "engine": "perfect_forecast",
                "rms_layout": RMS_LAYOUT_ISSUE_LOOKAHEAD,
            }
        }
        print(f"  Overwrote all {time_steps} forecast entries with oracle values.", flush=True)

    # ---- Final summary ----
    print("\n" + "=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Total forecast entries: {len(weather_forecasts)}", flush=True)
    print(f"  Mode: {'PERFECT (oracle)' if args.perfect_forecast else 'REAL (GFS-based)'}", flush=True)
    step = max(1, time_steps // 10)
    print("  Sample forecast slots (pwv_mean[slot], rms_mean_by_lookahead[0]):", flush=True)
    for t in range(0, time_steps, step):
        fc = weather_forecasts[t]
        pwv_val = fc["pwv_mean"][t] if t < len(fc["pwv_mean"]) else np.nan
        rms_rel = fc.get(RMS_LOOKAHEAD_MEAN_KEY, np.array([], dtype=float))
        rms_val = rms_rel[0] if len(rms_rel) else np.nan
        ts = idx_to_timestamp[t]
        if np.isnan(pwv_val) or np.isnan(rms_val):
            print(f"    slot {t:5d}  {ts}  pwv_mean=NaN  rms_now=NaN", flush=True)
        else:
            print(f"    slot {t:5d}  {ts}  pwv_mean={pwv_val:.4f}  rms_now={rms_val:.4f}", flush=True)

    output_data = {
        "weather_forecasts": weather_forecasts,
        "start_date": start_date,
        "end_date": end_date,
        "perfect_forecast": args.perfect_forecast,
        "rms_evaluation": rms_eval,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)
    print(f"\nSaved preprocessed forecasts to {args.output}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
