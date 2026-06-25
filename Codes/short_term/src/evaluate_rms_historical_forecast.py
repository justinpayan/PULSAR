"""
Build and evaluate holdout RMS forecasts.

This script compares historical-mean baseline against a selectable model:
1) Historical mean climatology baseline.
2) Forecast model selected via --forecast_model:
   - prophet (default)
   - sarimax (statsmodels)

Cycle-year convention:
  year Y means [Y-10-01, (Y+1)-09-30], inclusive at 30-minute cadence.
"""

import argparse
import itertools
import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None

try:
    from prophet import Prophet
except Exception:
    Prophet = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


TIME_INTERVAL_MINUTES = 30
HARDCODED_SHUTDOWNS = {
    2018: (2, 1, 3, 5),
    2019: (1, 29, 2, 28),
    2022: (1, 31, 2, 28),
    2023: (2, 1, 2, 28),
    2024: (1, 30, 2, 29),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RMS historical-mean forecast on holdout cycle year."
    )
    parser.add_argument("--data_dir", required=True, help="Directory with source CSV files")
    parser.add_argument("--train_start_year", type=int, default=2017)
    parser.add_argument("--train_end_year", type=int, default=2023)
    parser.add_argument("--test_year", type=int, default=2024)
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs")
    parser.add_argument(
        "--forecast_model",
        type=str,
        default="prophet",
        choices=["prophet", "sarimax"],
        help="Forecast engine used for model-based predictions",
    )

    # SARIMAX options
    parser.add_argument("--fourier_order", type=int, default=3,
                        help="Number of yearly Fourier harmonics for SARIMAX exogenous terms")
    parser.add_argument("--aic_search_max_points", type=int, default=20000,
                        help="Max trailing train points to use for AIC grid search")
    parser.add_argument("--aic_search_stride", type=int, default=4,
                        help="Subsample stride for AIC search points (e.g., 4 keeps every 4th point)")
    parser.add_argument("--aic_search_time_limit_sec", type=float, default=300.0,
                        help="Wall-clock time limit (seconds) for AIC grid search; <=0 disables")
    parser.add_argument("--aic_maxiter", type=int, default=50,
                        help="Max optimizer iterations for each AIC grid-search fit")
    parser.add_argument("--final_fit_maxiter", type=int, default=100,
                        help="Max optimizer iterations for final SARIMAX fit")

    # Prophet options
    parser.add_argument("--prophet_yearly_seasonality", action="store_true", default=True,
                        help="Enable Prophet yearly seasonality (default: on)")
    parser.add_argument("--prophet_daily_seasonality", action="store_true", default=True,
                        help="Enable Prophet daily seasonality (default: on)")
    parser.add_argument("--prophet_weekly_seasonality", action="store_true", default=False,
                        help="Enable Prophet weekly seasonality (default: off)")
    parser.add_argument("--prophet_changepoint_prior_scale", type=float, default=0.05,
                        help="Prophet changepoint_prior_scale")
    parser.add_argument("--prophet_fit_verbose", action="store_true",
                        help="Enable verbose Stan/CmdStan console logging during Prophet fit")
    parser.add_argument("--prophet_fit_iter", type=int, default=0,
                        help="Optional max optimization iterations passed to Prophet fit backend (0 = backend default)")
    parser.add_argument("--make_plots", action="store_true",
                        help="Generate comparison plots of actual vs historical mean vs selected model")
    parser.add_argument("--plot_days_per_figure", type=int, default=3,
                        help="Number of days per comparison plot")
    parser.add_argument("--max_plots", type=int, default=0,
                        help="Maximum number of plots to generate (0 = all windows)")
    return parser.parse_args()


def cycle_window(year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{year}-10-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-09-30 23:30:00", tz="UTC")
    return start, end


def build_timeline(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=f"{TIME_INTERVAL_MINUTES}min")


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


def indices_from_intervals(intervals_df: pd.DataFrame, all_slots: pd.DatetimeIndex) -> Set[int]:
    indices: Set[int] = set()
    for _, row in intervals_df.iterrows():
        mask = (all_slots >= row["START_TIME"]) & (all_slots < row["END_TIME"])
        indices.update(np.where(mask)[0].tolist())
    return indices


def build_engineering_and_shutdown_mask(
    data_dir: str,
    all_slots: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Set[int]:
    shifts_path = os.path.join(data_dir, "shifts_dimensions.csv")
    if not os.path.isfile(shifts_path):
        raise FileNotFoundError(f"Missing {shifts_path}")

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

    return engineering_indices | shutdown_indices


def realized_rms_on_timeline(
    phase_df: pd.DataFrame,
    all_slots: pd.DatetimeIndex,
) -> np.ndarray:
    rms_series = phase_df.set_index("timestamp")["freqrms"]
    tolerance = pd.Timedelta(minutes=30)
    rms_aligned = rms_series.reindex(all_slots, method="nearest", tolerance=tolerance)
    return rms_aligned.to_numpy()


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
    return grouped


def build_yearly_fourier_features(
    timestamps: pd.Series,
    fourier_order: int,
) -> pd.DataFrame:
    ts = pd.DatetimeIndex(timestamps)
    year_days = np.where(ts.is_leap_year, 366.0, 365.0)
    frac_of_year = (
        (ts.dayofyear.to_numpy() - 1.0)
        + (ts.hour.to_numpy() / 24.0)
        + (ts.minute.to_numpy() / 1440.0)
    ) / year_days
    features = {}
    for k in range(1, fourier_order + 1):
        angle = 2.0 * np.pi * k * frac_of_year
        features[f"sin_year_{k}"] = np.sin(angle)
        features[f"cos_year_{k}"] = np.cos(angle)
    return pd.DataFrame(features, index=np.arange(len(ts)))


def compute_metrics(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    mask_col: str,
) -> Dict[str, float]:
    eval_df = df[df[mask_col]].copy()
    if len(eval_df) == 0:
        return {
            "n_eval_slots": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias_pred_minus_actual": float("nan"),
        }
    residual = eval_df[pred_col] - eval_df[actual_col]
    return {
        "n_eval_slots": int(len(eval_df)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_actual": float(np.mean(residual)),
    }


def run_sarimax_aic_grid_search(
    endog: np.ndarray,
    exog: pd.DataFrame,
    seasonal_period: int,
    max_points: int,
    stride: int,
    time_limit_sec: float,
    maxiter: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float]:
    n = len(endog)
    if max_points > 0 and n > max_points:
        endog_search = endog[-max_points:]
        exog_search = exog.iloc[-max_points:, :].reset_index(drop=True)
    else:
        endog_search = endog
        exog_search = exog.reset_index(drop=True)

    # Optional coarse subsampling to speed up AIC model selection.
    if stride > 1:
        endog_search = endog_search[::stride]
        exog_search = exog_search.iloc[::stride, :].reset_index(drop=True)

    order_candidates: List[Tuple[int, int, int]] = [
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (2, 0, 1),
    ]
    seasonal_candidates: List[Tuple[int, int, int, int]] = [
        (0, 1, 1, seasonal_period),
        (1, 0, 1, seasonal_period),
        (1, 1, 0, seasonal_period),
        (1, 1, 1, seasonal_period),
    ]

    best_order: Optional[Tuple[int, int, int]] = None
    best_seasonal: Optional[Tuple[int, int, int, int]] = None
    best_aic = np.inf

    print("\n--- SARIMAX AIC grid search ---")
    print(f"  Search points: {len(endog_search)}")
    total_models = len(order_candidates) * len(seasonal_candidates)
    search_iter = itertools.product(order_candidates, seasonal_candidates)
    if tqdm is not None:
        search_iter = tqdm(search_iter, total=total_models, desc="AIC grid", unit="model")
    else:
        print(f"  Progress: evaluating {total_models} model combinations")

    t_start = time.time()
    for idx, (order, seasonal_order) in enumerate(search_iter, start=1):
        if time_limit_sec > 0 and (time.time() - t_start) > time_limit_sec:
            print(
                f"  Reached AIC search time limit ({time_limit_sec:.1f}s). "
                f"Stopping early at model {idx - 1}/{total_models}."
            )
            break
        if tqdm is None:
            print(f"  [{idx}/{total_models}] testing order={order}, seasonal={seasonal_order}")
        try:
            model = SARIMAX(
                endog=endog_search,
                exog=exog_search,
                order=order,
                seasonal_order=seasonal_order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit_res = model.fit(disp=False, maxiter=maxiter)
            aic_val = float(fit_res.aic)
            converged = bool(fit_res.mle_retvals.get("converged", True))
            print(
                f"  order={order}, seasonal={seasonal_order}, "
                f"AIC={aic_val:.3f}, converged={converged}"
            )
            if converged and np.isfinite(aic_val) and aic_val < best_aic:
                best_aic = aic_val
                best_order = order
                best_seasonal = seasonal_order
        except Exception as exc:
            print(f"  order={order}, seasonal={seasonal_order} failed: {exc}")

    if best_order is None or best_seasonal is None:
        raise RuntimeError("No SARIMAX model converged during AIC grid search.")

    print(
        f"  Selected by AIC: order={best_order}, seasonal={best_seasonal}, "
        f"AIC={best_aic:.3f}"
    )
    return best_order, best_seasonal, float(best_aic)


def run_sarimax_pipeline(
    train_timestamps: pd.Series,
    train_endog: np.ndarray,
    test_timestamps: pd.Series,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if SARIMAX is None:
        raise ImportError(
            "statsmodels is required for --forecast_model sarimax. "
            "Install with: pip install statsmodels"
        )
    train_exog = build_yearly_fourier_features(train_timestamps, args.fourier_order)
    test_exog = build_yearly_fourier_features(test_timestamps, args.fourier_order)

    best_order, best_seasonal_order, best_aic = run_sarimax_aic_grid_search(
        endog=train_endog,
        exog=train_exog,
        seasonal_period=48,
        max_points=args.aic_search_max_points,
        stride=args.aic_search_stride,
        time_limit_sec=args.aic_search_time_limit_sec,
        maxiter=args.aic_maxiter,
    )

    print("\n--- Fitting final SARIMAX ---")
    final_model = SARIMAX(
        endog=train_endog,
        exog=train_exog,
        order=best_order,
        seasonal_order=best_seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_fit = final_model.fit(disp=False, maxiter=args.final_fit_maxiter)
    print(f"  Final model AIC: {float(final_fit.aic):.3f}")

    forecast = final_fit.get_forecast(steps=len(test_timestamps), exog=test_exog)
    pred = np.asarray(forecast.predicted_mean, dtype=float)
    meta = {
        "engine": "sarimax",
        "selected_order": list(best_order),
        "selected_seasonal_order": list(best_seasonal_order),
        "grid_search_best_aic": float(best_aic),
        "final_fit_aic": float(final_fit.aic),
    }
    return pred, meta


def run_prophet_pipeline(
    train_timestamps: pd.Series,
    train_endog: np.ndarray,
    test_timestamps: pd.Series,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if Prophet is None:
        raise ImportError(
            "prophet is required for --forecast_model prophet. "
            "Install with: pip install prophet"
        )

    train_df = pd.DataFrame({
        "ds": pd.to_datetime(train_timestamps).dt.tz_convert("UTC").dt.tz_localize(None),
        "y": train_endog,
    })
    n_train_raw = len(train_df)
    # Prophet cannot train on NaN y values.
    train_df = train_df[np.isfinite(train_df["y"])].copy()
    n_train_finite = len(train_df)
    if train_df.empty:
        raise RuntimeError("No finite training points available for Prophet fit.")

    cap_value = float(max(1.0, train_df["y"].max() * 1.2))
    train_df["floor"] = 0.0
    train_df["cap"] = cap_value
    test_df = pd.DataFrame({
        "ds": pd.to_datetime(test_timestamps).dt.tz_convert("UTC").dt.tz_localize(None),
    })
    test_df["floor"] = 0.0
    test_df["cap"] = cap_value

    print("\n--- Fitting Prophet ---")
    print(f"  Train rows (raw): {n_train_raw}")
    print(f"  Train rows (finite y): {n_train_finite}")
    print(f"  Train ds range: {train_df['ds'].min()} -> {train_df['ds'].max()}")
    print(f"  Test rows: {len(test_df)}")
    print(f"  Test ds range:  {test_df['ds'].min()} -> {test_df['ds'].max()}")
    print(
        "  Prophet config: "
        f"growth=logistic, yearly={bool(args.prophet_yearly_seasonality)}, "
        f"daily={bool(args.prophet_daily_seasonality)}, "
        f"weekly={bool(args.prophet_weekly_seasonality)}, "
        f"cps={args.prophet_changepoint_prior_scale}"
    )
    print(
        f"  Logistic bounds: floor={float(train_df['floor'].iloc[0])}, cap={cap_value:.6f}; "
        f"train y min/max={float(train_df['y'].min()):.6f}/{float(train_df['y'].max()):.6f}"
    )
    fit_kwargs: Dict[str, object] = {
        "show_console": bool(args.prophet_fit_verbose),
    }
    if args.prophet_fit_iter and args.prophet_fit_iter > 0:
        fit_kwargs["iter"] = int(args.prophet_fit_iter)
    print(f"  Prophet fit kwargs: {fit_kwargs}")
    model = Prophet(
        growth="logistic",
        yearly_seasonality=bool(args.prophet_yearly_seasonality),
        weekly_seasonality=bool(args.prophet_weekly_seasonality),
        daily_seasonality=bool(args.prophet_daily_seasonality),
        changepoint_prior_scale=args.prophet_changepoint_prior_scale,
    )
    model.fit(train_df, **fit_kwargs)
    print("  Prophet fit complete.")
    pred_df = model.predict(test_df)
    pred = pred_df["yhat"].to_numpy(dtype=float)
    neg_before_clamp = int(np.sum(pred < 0))
    print(
        f"  Prediction stats before clamp: min={float(np.min(pred)):.6f}, "
        f"max={float(np.max(pred)):.6f}, negatives={neg_before_clamp}"
    )
    # Safety clamp: logistic+floor should already enforce this.
    pred = np.maximum(pred, 0.0)
    print(
        f"  Prediction stats after clamp:  min={float(np.min(pred)):.6f}, "
        f"max={float(np.max(pred)):.6f}, negatives={int(np.sum(pred < 0))}"
    )
    meta = {
        "engine": "prophet",
        "growth": "logistic",
        "nonnegative_constraint": "floor_0_logistic",
        "cap_value": cap_value,
        "yearly_seasonality": bool(args.prophet_yearly_seasonality),
        "daily_seasonality": bool(args.prophet_daily_seasonality),
        "weekly_seasonality": bool(args.prophet_weekly_seasonality),
        "changepoint_prior_scale": float(args.prophet_changepoint_prior_scale),
        "fit_verbose": bool(args.prophet_fit_verbose),
        "fit_iter": int(args.prophet_fit_iter) if args.prophet_fit_iter and args.prophet_fit_iter > 0 else None,
    }
    return pred, meta


def generate_comparison_plots(
    merged: pd.DataFrame,
    output_dir: str,
    model_label: str,
    days_per_figure: int = 3,
    max_plots: int = 0,
) -> List[str]:
    if days_per_figure <= 0:
        raise ValueError("plot_days_per_figure must be > 0")

    df = merged.copy().sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    start_ts = timestamps.iloc[0]
    end_ts = timestamps.iloc[-1]
    window = pd.Timedelta(days=days_per_figure)

    saved_paths: List[str] = []
    chunk_start = start_ts
    chunk_idx = 0
    while chunk_start <= end_ts:
        if max_plots > 0 and len(saved_paths) >= max_plots:
            break
        chunk_end = min(chunk_start + window, end_ts + pd.Timedelta(minutes=TIME_INTERVAL_MINUTES))
        mask = (timestamps >= chunk_start) & (timestamps < chunk_end)
        chunk = df[mask]
        if chunk.empty:
            chunk_start = chunk_end
            continue

        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(chunk["timestamp"], chunk["actual_rms"], color="black", linewidth=1.4, label="Actual RMS")
        ax.plot(chunk["timestamp"], chunk["pred_hist_mean"], color="tab:blue", linewidth=1.2, label="Historical mean")
        ax.plot(chunk["timestamp"], chunk["pred_model"], color="tab:orange", linewidth=1.2, label=f"{model_label} prediction")

        title_start = pd.Timestamp(chunk["timestamp"].iloc[0]).strftime("%Y-%m-%d %H:%M")
        title_end = pd.Timestamp(chunk["timestamp"].iloc[-1]).strftime("%Y-%m-%d %H:%M")
        ax.set_title(f"RMS Comparison: {title_start} to {title_end} UTC")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("RMS")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()

        filename = (
            f"rms_comparison_{model_label.lower()}_{chunk_idx:03d}_"
            f"{pd.Timestamp(chunk['timestamp'].iloc[0]).strftime('%Y%m%d_%H%M')}_"
            f"{pd.Timestamp(chunk['timestamp'].iloc[-1]).strftime('%Y%m%d_%H%M')}.png"
        )
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

        chunk_idx += 1
        chunk_start = chunk_end

    return saved_paths


def main():
    args = parse_args()
    if args.train_end_year < args.train_start_year:
        raise ValueError("train_end_year must be >= train_start_year")

    train_years = list(range(args.train_start_year, args.train_end_year + 1))
    if args.test_year in train_years:
        raise ValueError("test_year must not be part of training years")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("RMS HOLDOUT EVALUATION")
    print("=" * 70)
    print(f"Train cycle years: {train_years}")
    print(f"Test cycle year  : {args.test_year}")
    print(f"Forecast model   : {args.forecast_model}")

    phase_df = load_phase_tracker_with_freqrms(args.data_dir)

    train_frames = []
    train_arima_frames = []
    train_timestamps = set()
    total_train_slots = 0
    total_train_valid = 0

    for year in train_years:
        start, end = cycle_window(year)
        slots = build_timeline(start, end)
        total_train_slots += len(slots)
        rms = realized_rms_on_timeline(phase_df, slots)
        masked_idx = build_engineering_and_shutdown_mask(args.data_dir, slots, start, end)

        frame = pd.DataFrame({
            "timestamp": slots,
            "actual_rms": rms,
        })
        frame["is_masked"] = False
        if masked_idx:
            frame.loc[list(masked_idx), "is_masked"] = True

        frame["is_valid"] = (~frame["is_masked"]) & np.isfinite(frame["actual_rms"])
        frame["rms_for_model"] = frame["actual_rms"].where(frame["is_valid"], np.nan)
        train_arima_frames.append(frame[["timestamp", "rms_for_model"]])
        total_train_valid += int(frame["is_valid"].sum())
        frame = frame[frame["is_valid"]].copy()
        frame["month"] = frame["timestamp"].dt.month
        frame["day"] = frame["timestamp"].dt.day
        frame["time_interval_utc"] = frame["timestamp"].dt.strftime("%H:%M:%S")
        train_frames.append(frame[["timestamp", "actual_rms", "month", "day", "time_interval_utc"]])
        train_timestamps.update(frame["timestamp"].tolist())

    train_df = pd.concat(train_frames, ignore_index=True)
    climatology_df = build_train_climatology(train_df)
    train_arima_df = pd.concat(train_arima_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    test_start, test_end = cycle_window(args.test_year)
    test_slots = build_timeline(test_start, test_end)
    test_rms = realized_rms_on_timeline(phase_df, test_slots)
    test_masked_idx = build_engineering_and_shutdown_mask(args.data_dir, test_slots, test_start, test_end)

    test_df = pd.DataFrame({
        "timestamp": test_slots,
        "actual_rms": test_rms,
    })
    test_df["is_masked"] = False
    if test_masked_idx:
        test_df.loc[list(test_masked_idx), "is_masked"] = True
    test_df["is_valid_actual"] = (~test_df["is_masked"]) & np.isfinite(test_df["actual_rms"])
    test_df["month"] = test_df["timestamp"].dt.month
    test_df["day"] = test_df["timestamp"].dt.day
    test_df["time_interval_utc"] = test_df["timestamp"].dt.strftime("%H:%M:%S")

    # Leak check: training and holdout timestamps must be disjoint.
    overlap = set(test_df.loc[test_df["is_valid_actual"], "timestamp"].tolist()) & train_timestamps
    if overlap:
        raise RuntimeError(f"Train/test overlap detected: {len(overlap)} timestamps")

    merged = test_df.merge(
        climatology_df[["month", "day", "time_interval_utc", "mean_rms_microns", "std_rms_microns", "data_points_across_years"]],
        on=["month", "day", "time_interval_utc"],
        how="left",
    )
    merged["predicted_rms"] = merged["mean_rms_microns"]
    merged["pred_hist_mean"] = merged["predicted_rms"]
    merged["has_hist_prediction"] = np.isfinite(merged["pred_hist_mean"])

    # --- Selected model fit and holdout forecast ---
    train_endog = train_arima_df["rms_for_model"].to_numpy()
    if args.forecast_model == "sarimax":
        pred_model, model_meta = run_sarimax_pipeline(
            train_timestamps=train_arima_df["timestamp"],
            train_endog=train_endog,
            test_timestamps=test_df["timestamp"],
            args=args,
        )
    else:
        print(
            "  Note: SARIMAX AIC options are ignored in prophet mode "
            "(--aic_search_max_points, --aic_search_stride, "
            "--aic_search_time_limit_sec, --aic_maxiter, --final_fit_maxiter)."
        )
        pred_model, model_meta = run_prophet_pipeline(
            train_timestamps=train_arima_df["timestamp"],
            train_endog=train_endog,
            test_timestamps=test_df["timestamp"],
            args=args,
        )

    merged["pred_model"] = np.asarray(pred_model, dtype=float)
    merged["has_model_prediction"] = np.isfinite(merged["pred_model"])
    # Backward-compatible aliases from earlier SARIMAX-only output naming.
    merged["pred_arima"] = merged["pred_model"]
    negative_pred_count = int((merged["pred_model"] < 0).sum())

    merged["is_eval_hist"] = merged["is_valid_actual"] & merged["has_hist_prediction"]
    merged["is_eval_model"] = merged["is_valid_actual"] & merged["has_model_prediction"]
    merged["is_eval_common"] = merged["is_valid_actual"] & merged["has_hist_prediction"] & merged["has_model_prediction"]
    merged["is_eval_arima"] = merged["is_eval_model"]

    merged["residual_hist"] = merged["pred_hist_mean"] - merged["actual_rms"]
    merged["residual_model"] = merged["pred_model"] - merged["actual_rms"]
    merged["residual_arima"] = merged["residual_model"]
    merged["abs_error_hist"] = np.abs(merged["residual_hist"])
    merged["abs_error_model"] = np.abs(merged["residual_model"])
    merged["abs_error_arima"] = merged["abs_error_model"]
    merged["squared_error_hist"] = merged["residual_hist"] ** 2
    merged["squared_error_model"] = merged["residual_model"] ** 2
    merged["squared_error_arima"] = merged["squared_error_model"]
    merged["has_arima_prediction"] = merged["has_model_prediction"]

    missing_hist_among_valid = int((merged["is_valid_actual"] & ~merged["has_hist_prediction"]).sum())
    missing_model_among_valid = int((merged["is_valid_actual"] & ~merged["has_model_prediction"]).sum())

    hist_metrics = compute_metrics(merged, "pred_hist_mean", "actual_rms", "is_eval_hist")
    model_metrics = compute_metrics(merged, "pred_model", "actual_rms", "is_eval_model")
    hist_common_metrics = compute_metrics(merged, "pred_hist_mean", "actual_rms", "is_eval_common")
    model_common_metrics = compute_metrics(merged, "pred_model", "actual_rms", "is_eval_common")

    summary = {
        "train_start_year": args.train_start_year,
        "train_end_year": args.train_end_year,
        "test_year": args.test_year,
        "train_years": train_years,
        "train_total_slots": int(total_train_slots),
        "train_valid_slots_used": int(total_train_valid),
        "test_total_slots": int(len(test_df)),
        "test_valid_actual_slots": int(test_df["is_valid_actual"].sum()),
        "selected_model": args.forecast_model,
        "historical_mean": {
            "missing_prediction_slots_among_valid_actual": missing_hist_among_valid,
            **hist_metrics,
        },
        "selected_model_metrics": {
            "missing_prediction_slots_among_valid_actual": missing_model_among_valid,
            "negative_prediction_count": negative_pred_count,
            **model_meta,
            **model_metrics,
        },
        "common_eval_slots_for_comparison": int(merged["is_eval_common"].sum()),
        "historical_mean_on_common_slots": hist_common_metrics,
        "selected_model_on_common_slots": model_common_metrics,
    }

    # Keep explicit model-named entries for convenience and backward compatibility.
    if args.forecast_model == "sarimax":
        summary["arima_sarimax"] = summary["selected_model_metrics"]
        summary["arima_on_common_slots"] = summary["selected_model_on_common_slots"]
    else:
        summary["prophet"] = summary["selected_model_metrics"]
        summary["prophet_nonnegative_mode"] = "logistic_floor_0"

    plot_paths: List[str] = []
    if args.make_plots:
        model_label = "Prophet" if args.forecast_model == "prophet" else "SARIMAX"
        plot_paths = generate_comparison_plots(
            merged=merged,
            output_dir=args.output_dir,
            model_label=model_label,
            days_per_figure=args.plot_days_per_figure,
            max_plots=args.max_plots,
        )
    summary["plots"] = {
        "enabled": bool(args.make_plots),
        "days_per_figure": int(args.plot_days_per_figure),
        "max_plots": int(args.max_plots),
        "num_plots_written": int(len(plot_paths)),
        "plot_files": plot_paths[:20],
    }

    summary_path = os.path.join(
        args.output_dir,
        f"rms_holdout_summary_train_{args.train_start_year}_{args.train_end_year}_test_{args.test_year}.json",
    )
    details_path = os.path.join(
        args.output_dir,
        f"rms_holdout_predictions_train_{args.train_start_year}_{args.train_end_year}_test_{args.test_year}.csv",
    )
    climatology_path = os.path.join(
        args.output_dir,
        f"rms_climatology_train_{args.train_start_year}_{args.train_end_year}.csv",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    merged[
        [
            "timestamp",
            "actual_rms",
            "pred_hist_mean",
            "pred_model",
            "pred_arima",
            "residual_hist",
            "residual_model",
            "residual_arima",
            "abs_error_hist",
            "abs_error_model",
            "abs_error_arima",
            "squared_error_hist",
            "squared_error_model",
            "squared_error_arima",
            "is_masked",
            "is_valid_actual",
            "has_hist_prediction",
            "has_model_prediction",
            "has_arima_prediction",
            "is_eval_hist",
            "is_eval_model",
            "is_eval_arima",
            "is_eval_common",
            "data_points_across_years",
        ]
    ].to_csv(details_path, index=False)

    climatology_df.to_csv(climatology_path, index=False)

    print("\n--- Summary ---")
    print(f"Train valid slots used                 : {summary['train_valid_slots_used']}")
    print(f"Test valid actual slots                : {summary['test_valid_actual_slots']}")
    print(f"Historical mean eval slots             : {summary['historical_mean']['n_eval_slots']}")
    print(f"{args.forecast_model.upper()} eval slots                   : {summary['selected_model_metrics']['n_eval_slots']}")
    print(f"Common eval slots                      : {summary['common_eval_slots_for_comparison']}")
    print(f"Historical mean RMSE                   : {summary['historical_mean']['rmse']}")
    print(f"{args.forecast_model.upper()} RMSE                         : {summary['selected_model_metrics']['rmse']}")
    print(f"Historical mean RMSE (common slots)    : {summary['historical_mean_on_common_slots']['rmse']}")
    print(f"{args.forecast_model.upper()} RMSE (common slots)          : {summary['selected_model_on_common_slots']['rmse']}")
    print(f"{args.forecast_model.upper()} negative predictions         : {summary['selected_model_metrics']['negative_prediction_count']}")
    if args.forecast_model == "sarimax":
        print(
            f"Selected ARIMA orders                  : "
            f"{tuple(summary['selected_model_metrics']['selected_order'])}, "
            f"{tuple(summary['selected_model_metrics']['selected_seasonal_order'])}"
        )
    else:
        print(
            "Prophet config                         : "
            f"yearly={summary['selected_model_metrics']['yearly_seasonality']}, "
            f"daily={summary['selected_model_metrics']['daily_seasonality']}, "
            f"weekly={summary['selected_model_metrics']['weekly_seasonality']}, "
            f"cps={summary['selected_model_metrics']['changepoint_prior_scale']}"
        )
    print(f"Saved summary JSON                     : {summary_path}")
    print(f"Saved per-slot CSV                     : {details_path}")
    print(f"Saved train climatology CSV            : {climatology_path}")
    if args.make_plots:
        print(f"Saved comparison plots                 : {len(plot_paths)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
