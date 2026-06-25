from scipy.stats import norm, uniform
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


def forecast_distributions_factory(
    pwv_means,
    rms_means,
    STD_PWV,
    STD_RMS
):
    """
    Returns a forecast function forecast_fn(t, T) that outputs forecasted distributions
    for PWV and RMS from time t to T - 1.

    PWV: Gaussian with known mean and fixed std
    RMS: Uniform with known mean and fixed width
    """
    def forecast_fn(t, T):
        forecasts = []
        for i in range(t, T):
            mean_pwv = pwv_means[i]
            mean_rms = rms_means[i]
            forecasts.append({
                "PWV": {
                    "mean": mean_pwv,
                    "std": STD_PWV,
                    "dist": norm(loc=mean_pwv, scale=STD_PWV)
                },
                "RMS": {
                    "mean": mean_rms,
                    "width": STD_RMS,
                    "dist": norm(loc=mean_rms, scale=STD_RMS)
                }
            })
        return forecasts

    return forecast_fn


def forecast_distributions_mc_factory(
        pwv_means,
        rms_means,
        STD_PWV,
        STD_RMS,
        NUM_WEATHER_BINS
):
    """
    Returns a forecast function forecast_fn(t, T) that outputs forecasted distributions
    for PWV and RMS from time t to T - 1.

    PWV: Gaussian with known mean and fixed std
    RMS: Uniform with known mean and fixed width
    """

    def forecast_fn(t, T):
        forecasts = []
        for i in range(t, T):
            mean_pwv = pwv_means[i]
            mean_rms = rms_means[i]

            n_samples = 1000

            pwv_samples = np.round(np.random.normal(size=n_samples, loc=mean_pwv, scale=STD_PWV))
            rms_samples = np.round(np.random.normal(loc=mean_rms, scale=STD_RMS, size=n_samples))
            pwv_samples = np.clip(pwv_samples, 1, NUM_WEATHER_BINS)
            rms_samples = np.clip(rms_samples, 1, NUM_WEATHER_BINS)

            mean_pwv = np.mean(pwv_samples)
            mean_rms = np.mean(rms_samples)

            forecasts.append({
                "PWV": {
                    "mean": mean_pwv,
                    "bottom_quartile": sorted(pwv_samples)[n_samples // 4],
                    "top_quartile": sorted(pwv_samples)[3 * n_samples // 4]
                    # "std": STD_PWV,
                    # "dist": norm(loc=mean_pwv, scale=STD_PWV)
                },
                "RMS": {
                    "mean": mean_rms,
                    "bottom_quartile": sorted(rms_samples)[n_samples // 4],
                    "top_quartile": sorted(rms_samples)[3 * n_samples // 4]
                    # "width": STD_RMS,
                    # "dist": norm(loc=mean_rms, scale=STD_RMS)
                }
            })
        return forecasts

    return forecast_fn


def forecast_sampler_factory(
        pwv_means,
        rms_means,
        std_pwv,
        width_rms,
        seed=None
):
    """
    Returns a function (t, T) -> sampled weather_df from time t to T-1,
    using:
      - PWV ~ Normal(mean=pwv_means[t], std=std_pwv)
      - RMS ~ Uniform(mean ± width_rms)

    Assumes known means for each future timestep.
    """
    rng = np.random.default_rng(seed)

    def sampler(t, T):
        pwv_vals = rng.normal(loc=pwv_means[t:T], scale=std_pwv)
        rms_vals = rng.normal(loc=rms_means[t:T], scale=width_rms)

        return pd.DataFrame({
            "PWV_realized": np.round(np.clip(pwv_vals, 1, 5)),  # Optional: round and clip to match binning
            "RMS_realized": np.round(np.clip(rms_vals, 1, 5))
        }, index=range(t, T))

    return sampler


def generate_noisy_forecast_from_realized(
        realized_weather: Dict[int, Any],
        current_time_idx: int,  # The current global time step index
        base_std_pwv: float = 0.5,  # Initial std dev for PWV noise
        base_std_rms: float = 0.1,  # Initial std dev for RMS noise
        std_increase_factor_pwv: float = 0.05,  # How much PWV std increases per step into future
        std_increase_factor_rms: float = 0.02  # How much RMS std increases per step into future
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a noisy forecast based on future actual values.
    The 'forecasted mean' is the true future value. Noise is added around it.
    Standard deviation of the noise increases for steps further in the future.

    Args:
        realized_weather: the true pwv and rms observed over the episode
        current_time_idx: The current time step from which to forecast.
        forecast_horizon: How many steps into the future to forecast.
        base_std_*: Initial standard deviation for the noise.
        std_increase_factor_*: Factor by which std dev increases per forecast step.

    Returns:
        Tuple of (fc_pwv_means, fc_pwv_stds, fc_rms_means, fc_rms_stds)
        Each is a numpy array of length `forecast_horizon`.
        These are for t+1, t+2, ..., t+forecast_horizon relative to current_time_idx.
    """
    forecast_horizon = len(realized_weather) - current_time_idx

    fc_pwv_means = []
    fc_pwv_stds = []
    fc_rms_means = []
    fc_rms_stds = []

    for k in range(1, forecast_horizon + 1):  # For t+1, t+2, ...
        future_time_idx = current_time_idx + k

        # Standard deviation increases with k (how far into the future)
        current_std_pwv = base_std_pwv + (k * std_increase_factor_pwv)
        current_std_rms = base_std_rms + (k * std_increase_factor_rms)
        fc_pwv_stds.append(current_std_pwv)
        fc_rms_stds.append(current_std_rms)

        # Get the true future value to base the "mean" of the forecast on
        # pwv_base = realized_pwv_series.get(future_time_idx, np.nan)
        # rms_base = realized_rms_series.get(future_time_idx, np.nan)
        pwv_base, rms_base = realized_weather.get(future_time_idx, (np.nan, np.nan))

        if not np.isnan(pwv_base):
            fc_pwv_means.append(max(0, pwv_base + np.random.normal(0, current_std_pwv)))
            fc_rms_means.append(max(0, rms_base + np.random.normal(0, current_std_rms)))
        else:
            fc_pwv_means.append(np.nan)
            fc_rms_means.append(np.nan)

    return (
        np.array(fc_pwv_means), np.array(fc_pwv_stds),
        np.array(fc_rms_means), np.array(fc_rms_stds)
    )

def calculate_forecast_statistics(
    fc_pwv_means: np.ndarray, fc_pwv_stds: np.ndarray,
    fc_rms_means: np.ndarray, fc_rms_stds: np.ndarray
) -> List[Dict[str, Dict[str, float]]]:
    """
    Calculates mean, bottom quartile (Q1), and top quartile (Q3)
    for PWV and RMS forecasts, assuming a normal distribution.

    Args:
        fc_pwv_means, fc_pwv_stds: Forecasted means and stds for PWV.
        fc_rms_means, fc_rms_stds: Forecasted means and stds for RMS.

    Returns:
        A list of dictionaries, one for each forecast step. Each dict has
        keys "PWV" and "RMS", and sub-dictionaries with "mean",
        "bottom_quartile", "top_quartile".
    """
    forecast_stats_list = []
    num_forecast_steps = len(fc_pwv_means)

    for i in range(num_forecast_steps):
        step_stats = {"PWV": {}, "RMS": {}}

        # PWV statistics
        pwv_m, pwv_s = fc_pwv_means[i], fc_pwv_stds[i]
        if np.isnan(pwv_m):
            step_stats["PWV"]["mean"] = np.nan
            step_stats["PWV"]["bottom_quartile"] = np.nan
            step_stats["PWV"]["top_quartile"] = np.nan
        else:

            step_stats["PWV"]["mean"] = pwv_m
            # step_stats["PWV"]["bottom_quartile"] = norm.ppf(0.25, loc=pwv_m, scale=pwv_s)
            # step_stats["PWV"]["top_quartile"] = norm.ppf(0.75, loc=pwv_m, scale=pwv_s)
            # We flip these, because lower is better for PWV. So the worse case is when PWV is at its 75th percentile.
            step_stats["PWV"]["bottom_quartile"] = norm.ppf(0.75, loc=pwv_m, scale=pwv_s)
            step_stats["PWV"]["top_quartile"] = norm.ppf(0.25, loc=pwv_m, scale=pwv_s)
            # Optional: Clip to physically realistic bounds if necessary, e.g., PWV > 0
            step_stats["PWV"]["bottom_quartile"] = max(0, step_stats["PWV"]["bottom_quartile"])
            step_stats["PWV"]["top_quartile"] = max(0, step_stats["PWV"]["top_quartile"])


        # RMS statistics
        rms_m, rms_s = fc_rms_means[i], fc_rms_stds[i]
        if np.isnan(rms_m):
            step_stats["RMS"]["mean"] = np.nan
            step_stats["RMS"]["bottom_quartile"] = np.nan
            step_stats["RMS"]["top_quartile"] = np.nan
        else:
            step_stats["RMS"]["mean"] = rms_m
            step_stats["RMS"]["bottom_quartile"] = norm.ppf(0.25, loc=rms_m, scale=rms_s)
            step_stats["RMS"]["top_quartile"] = norm.ppf(0.75, loc=rms_m, scale=rms_s)
            step_stats["RMS"]["bottom_quartile"] = max(0, step_stats["RMS"]["bottom_quartile"])
            step_stats["RMS"]["top_quartile"] = max(0, step_stats["RMS"]["top_quartile"])


        forecast_stats_list.append(step_stats)

    return forecast_stats_list


def sample_weather_path_from_forecast(
        forecast_for_future: Dict[str, np.ndarray],
        rng: np.random.Generator
) -> Dict[int, Tuple[float, float]]:
    """
    Generates a single sampled path of future weather (PWV, RMS)
    based on pre-computed forecast distributions (mean and std).

    Args:
        forecast_for_future (Dict[str, np.ndarray]): A dictionary containing forecast arrays
            ('pwv_mean', 'pwv_std', 'rms_mean', 'rms_std') for future time steps.
        rng (np.random.Generator): A random number generator instance for reproducibility.

    Returns:
        Dict[int, Tuple[float, float]]: A dictionary mapping relative time index
            (0, 1, 2...) to the sampled (PWV, RMS) tuple for that step.
    """
    sampled_weather_path: Dict[int, Tuple[float, float]] = {}

    # The horizon is determined by the length of the provided forecast arrays
    forecast_horizon = len(forecast_for_future['pwv_mean'])

    for k_relative in range(forecast_horizon):
        # Get the mean and std for this future step from the forecast
        pwv_mean = forecast_for_future['pwv_mean'][k_relative]
        pwv_std = forecast_for_future['pwv_std'][k_relative]
        rms_mean = forecast_for_future['rms_mean'][k_relative]
        rms_std = forecast_for_future['rms_std'][k_relative]

        # If the forecast is invalid (NaN), the sample is also NaN
        if pd.isna(pwv_mean) or pd.isna(pwv_std):
            sampled_pwv = np.nan
        else:
            # Sample from the normal distribution defined by the forecast
            sampled_pwv = rng.normal(loc=pwv_mean, scale=max(0.001, pwv_std))
            sampled_pwv = max(0.0, sampled_pwv)  # Clip to be non-negative

        if pd.isna(rms_mean):
            sampled_rms = np.nan
        elif pd.isna(rms_std):
            sampled_rms = rms_mean
        else:
            sampled_rms = rng.normal(loc=rms_mean, scale=max(0.001, rms_std))
            sampled_rms = max(0.0, sampled_rms)  # Clip to be non-negative

        sampled_weather_path[k_relative] = (sampled_pwv, sampled_rms)

    return sampled_weather_path