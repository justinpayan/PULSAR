import os
import argparse
import glob
from copy import deepcopy

from planning_implementations import (solve_prophet_configurable,
                                      planning_loop_eb_greedy,
                                      dsa_eb_selector_factory,
                                      compute_paper_objective_value,
                                      planning_loop_pulsar)
from fixed_params import *
# Capture the module-level EXECUTIVE_QUOTAS before any function-local reassignment shadows it.
_FIXED_EXECUTIVE_QUOTAS = EXECUTIVE_QUOTAS
from job_and_weather_simulation import load_and_prepare_data
from util import convert_quotas
from evaluation import run_eval_real, _is_execution_valid
from weather_forecast_layout import (
    RMS_LAYOUT_ISSUE_LOOKAHEAD,
    RMS_LAYOUT_KEY,
    RMS_LOOKAHEAD_MEAN_KEY,
    RMS_LOOKAHEAD_STD_KEY,
    build_global_rms_arrays,
)
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import pickle
import json
import sys
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict, Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
long_term_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'long_term'))
if long_term_dir not in sys.path:
    sys.path.insert(0, long_term_dir)
from long_term_optim import build_config_calendar, solve_long_term_schedule_weekly
import itertools


def load_config_calendar(cycle_start_date: pd.Timestamp, base_calendar: Optional[pd.DataFrame] = None):
    """
    Load and adjust configuration calendar for the given cycle start date.
    
    Args:
        cycle_start_date: The start date of the cycle
        base_calendar: Optional calendar DataFrame from long_term_optim. If None, will use hard-coded calendar.
    
    Returns:
        Adjusted configuration calendar with dates shifted to the correct year
    """
    config_calendar = pd.DataFrame()
    if base_calendar is not None:
        # Use the calendar from long_term_optim
        config_calendar = base_calendar.copy()
        
        # The calendar from long_term_optim has dates in the base year
        # We need to shift them to the cycle_start_date year
        base_year_start = config_calendar['Start'].iloc[0]
        time_offset = cycle_start_date - base_year_start
        
        # Apply the offset to shift all dates to the correct year
        config_calendar['Start'] = config_calendar['Start'] + time_offset
        config_calendar['End'] = config_calendar['End'] + time_offset

        # Ensure timezone-aware timestamps
        config_calendar['End'] = config_calendar['End'] + pd.to_timedelta('23h 59m 59s')

    print("--- Configuration Calendar ---")
    print(config_calendar)
    print("-" * 30)
    return config_calendar


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _normalize_grade_label(value: Any) -> str:
    grade = str(value).strip().upper()
    if grade not in {"A", "B", "C"}:
        raise ValueError(f"Unsupported grade '{value}'. Expected one of A/B/C.")
    return grade


def _extract_unique_weight_by_grade(
        records: List[Dict[str, Any]],
        *,
        record_label: str,
        id_key: str,
) -> Dict[str, float]:
    grade_to_weight: Dict[str, float] = {}
    for record in records:
        grade = _normalize_grade_label(record.get("grade"))
        try:
            weight = float(record.get("weight"))
        except (TypeError, ValueError):
            raise ValueError(
                f"{record_label} '{record.get(id_key)}' is missing a valid numeric weight."
            ) from None

        if grade in grade_to_weight and not np.isclose(grade_to_weight[grade], weight):
            raise ValueError(
                f"Inconsistent {record_label} weights for grade {grade}: "
                f"saw both {grade_to_weight[grade]} and {weight}."
            )
        grade_to_weight.setdefault(grade, weight)

    missing_grades = {"A", "B", "C"} - set(grade_to_weight)
    if missing_grades:
        raise ValueError(
            f"Missing {record_label} weight entries for grades: {sorted(missing_grades)}."
        )
    return grade_to_weight


def _derive_objective_weights_from_loaded_data(
        *,
        base_weights: Dict[str, float],
        jobs: List[Dict[str, Any]],
        projects: List[Dict[str, Any]],
) -> Dict[str, float]:
    sb_grade_weights = _extract_unique_weight_by_grade(
        jobs,
        record_label="SB",
        id_key="job_id",
    )
    project_grade_weights = _extract_unique_weight_by_grade(
        projects,
        record_label="project",
        id_key="project_id",
    )

    w_sb = float(base_weights.get("obs_completion", 0.0))
    w_proj = float(base_weights.get("proj_completion", 0.0))
    derived_weights = dict(base_weights)
    derived_weights.update({
        "sb_A": sb_grade_weights["A"] * w_sb,
        "sb_B": sb_grade_weights["B"] * w_sb,
        "sb_C": sb_grade_weights["C"] * w_sb,
        "proj_A": project_grade_weights["A"] * w_proj,
        "proj_B": project_grade_weights["B"] * w_proj,
        "proj_C": project_grade_weights["C"] * w_proj,
    })
    return derived_weights


def _format_counter(counter_like: Counter) -> Dict[str, int]:
    return {str(key): int(counter_like[key]) for key in sorted(counter_like, key=lambda item: str(item))}


def _describe_config_calendar(config_calendar: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if config_calendar is None or len(config_calendar) == 0:
        return {"rows": 0, "first": None, "last": None}
    first_row = config_calendar.iloc[0]
    last_row = config_calendar.iloc[-1]
    return {
        "rows": int(len(config_calendar)),
        "first": {
            "Configuration": first_row.get("Configuration"),
            "Start": str(first_row.get("Start")),
            "End": str(first_row.get("End")),
        },
        "last": {
            "Configuration": last_row.get("Configuration"),
            "Start": str(last_row.get("Start")),
            "End": str(last_row.get("End")),
        },
    }


def log_dsa_eb_run_configuration(
        source_label: str,
        algorithm_name: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        seed: int,
        weights: Dict[str, float],
        executive_quotas: Dict[str, Tuple[float, float]],
        eb_ramp_exponent: float,
        extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    print(f"\n{'=' * 80}", flush=True)
    print(f"DSA_EB COMPARISON RUN CONFIG [{source_label}]", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(f"algorithm_name: {algorithm_name}", flush=True)
    print(f"start_date: {start_date}", flush=True)
    print(f"end_date: {end_date}", flush=True)
    print(f"seed: {seed}", flush=True)
    print(f"eb_ramp_exponent: {eb_ramp_exponent}", flush=True)
    print(f"weights: {_stable_json(weights)}", flush=True)
    print(f"executive_quotas: {_stable_json(executive_quotas)}", flush=True)
    if extra_fields:
        for key in sorted(extra_fields):
            print(f"{key}: {_stable_json(extra_fields[key])}", flush=True)
    print(f"{'=' * 80}\n", flush=True)


def log_dsa_eb_dataset_summary(
        source_label: str,
        stage_label: str,
        jobs: List[Dict[str, Any]],
        projects: List[Dict[str, Any]],
        time_steps: int,
        idx_to_timestamp: Dict[int, pd.Timestamp],
        realized_weather: Dict[int, Tuple[float, float]],
        observable_time: Optional[int] = None,
        config_calendar: Optional[pd.DataFrame] = None,
) -> None:
    grade_counts = Counter()
    executive_counts = Counter()
    total_execs_by_grade = Counter()
    total_execs_by_executive = Counter()
    available_lengths = []
    forecast_available_lengths = []
    condition_lengths = []
    base_ha_lengths = []
    pwv_threshold_lengths = []
    rms_thresholds = []
    job_lengths = []
    remaining_execs = []
    cycle_grade_scores = []

    for job in jobs:
        grade_counts[str(job.get("grade", "UNKNOWN")).strip().upper() or "UNKNOWN"] += 1
        executive = job.get("executive", "UNKNOWN")
        if isinstance(executive, dict):
            executive_label = "MULTI_EXEC"
        else:
            executive_label = str(executive)
        executive_counts[executive_label] += 1
        total_execs_value = float(job.get("total_execs", 0))
        total_execs_by_grade[str(job.get("grade", "UNKNOWN")).strip().upper() or "UNKNOWN"] += int(total_execs_value)
        total_execs_by_executive[executive_label] += int(total_execs_value)
        available_lengths.append(len(job.get("available", [])))
        forecast_available_lengths.append(len(job.get("forecast_available", [])))
        condition_lengths.append(len(job.get("condition_scores", {})))
        base_ha_lengths.append(len(job.get("base_ha_scores", {})))
        pwv_threshold_lengths.append(len(job.get("pwv_thresholds", {})))
        job_lengths.append(float(job.get("length", 0)))
        remaining_execs.append(float(job.get("remaining_execs", 0)))
        cycle_grade_score = job.get("cycle_grade_score")
        if cycle_grade_score is not None and np.isfinite(cycle_grade_score):
            cycle_grade_scores.append(float(cycle_grade_score))
        rms_threshold = job.get("rms_threshold")
        if rms_threshold is not None and np.isfinite(rms_threshold):
            rms_thresholds.append(float(rms_threshold))

    sorted_indices = sorted(idx_to_timestamp)
    first_timestamp = str(idx_to_timestamp[sorted_indices[0]]) if sorted_indices else None
    last_timestamp = str(idx_to_timestamp[sorted_indices[-1]]) if sorted_indices else None

    valid_pwv = []
    valid_rms = []
    finite_weather_slots = 0
    for t in range(time_steps):
        pwv, rms = realized_weather.get(t, (np.nan, np.nan))
        if np.isfinite(pwv):
            valid_pwv.append(float(pwv))
        if np.isfinite(rms):
            valid_rms.append(float(rms))
        if np.isfinite(pwv) and np.isfinite(rms):
            finite_weather_slots += 1

    project_grade_counts = Counter(
        str(project.get("grade", "UNKNOWN")).strip().upper() or "UNKNOWN"
        for project in projects
    )
    project_job_count_stats = _summary_stats(
        [len(project.get("job_ids", [])) for project in projects]
    )

    print(f"\n{'=' * 80}", flush=True)
    print(f"DSA_EB COMPARISON DATASET [{source_label}] [{stage_label}]", flush=True)
    print(f"{'=' * 80}", flush=True)
    print(
        f"time_steps={time_steps} | idx_to_timestamp={len(idx_to_timestamp)} | "
        f"first_timestamp={first_timestamp} | last_timestamp={last_timestamp}",
        flush=True,
    )
    print(
        f"jobs={len(jobs)} | projects={len(projects)} | observable_time={observable_time}",
        flush=True,
    )
    print(
        f"weather_slots_with_both_channels={finite_weather_slots} | "
        f"pwv_stats={_stable_json(_summary_stats(valid_pwv))} | "
        f"rms_stats={_stable_json(_summary_stats(valid_rms))}",
        flush=True,
    )
    print(
        f"job_grade_counts={_stable_json(_format_counter(grade_counts))} | "
        f"project_grade_counts={_stable_json(_format_counter(project_grade_counts))}",
        flush=True,
    )
    print(
        f"total_execs_by_grade={_stable_json(_format_counter(total_execs_by_grade))} | "
        f"total_execs_by_executive={_stable_json(_format_counter(total_execs_by_executive))}",
        flush=True,
    )
    print(
        f"executive_counts={_stable_json(_format_counter(executive_counts))}",
        flush=True,
    )
    print(
        f"job_length_stats={_stable_json(_summary_stats(job_lengths))} | "
        f"remaining_execs_stats={_stable_json(_summary_stats(remaining_execs))}",
        flush=True,
    )
    print(
        f"available_slots_stats={_stable_json(_summary_stats(available_lengths))} | "
        f"forecast_available_slots_stats={_stable_json(_summary_stats(forecast_available_lengths))}",
        flush=True,
    )
    print(
        f"condition_score_slots_stats={_stable_json(_summary_stats(condition_lengths))} | "
        f"base_ha_slots_stats={_stable_json(_summary_stats(base_ha_lengths))}",
        flush=True,
    )
    print(
        f"pwv_threshold_slots_stats={_stable_json(_summary_stats(pwv_threshold_lengths))} | "
        f"rms_threshold_stats={_stable_json(_summary_stats(rms_thresholds))}",
        flush=True,
    )
    print(
        f"cycle_grade_score_stats={_stable_json(_summary_stats(cycle_grade_scores))} | "
        f"project_job_count_stats={_stable_json(project_job_count_stats)}",
        flush=True,
    )
    print(
        f"jobs_with_available={sum(1 for job in jobs if job.get('available'))}/{len(jobs)} | "
        f"jobs_without_available={sum(1 for job in jobs if not job.get('available'))}/{len(jobs)} | "
        f"jobs_with_condition_scores={sum(1 for job in jobs if job.get('condition_scores'))}/{len(jobs)} | "
        f"jobs_with_base_ha_scores={sum(1 for job in jobs if job.get('base_ha_scores'))}/{len(jobs)} | "
        f"jobs_with_pwv_thresholds={sum(1 for job in jobs if job.get('pwv_thresholds'))}/{len(jobs)}",
        flush=True,
    )
    print(
        f"config_calendar={_stable_json(_describe_config_calendar(config_calendar))}",
        flush=True,
    )
    print(f"{'=' * 80}\n", flush=True)


def build_time_to_lst_bin_map(
        data_dir: str,
        year: int,
        idx_to_timestamp: Dict[int, pd.Timestamp],
) -> Dict[int, float]:
    """Map each operational time index to the long-term LST-bin convention."""
    print("\n--- Building time -> LST_bin map for strategic LP sampling ---")
    primary_pressure_path = os.path.join(data_dir, f"sb_12m_pressure_{year}.csv")
    pressure_path = primary_pressure_path
    pressure_df = pd.read_csv(primary_pressure_path)
    pressure_df['Date'] = pd.to_datetime(pressure_df['Date'], utc=True)
    lst_col = next((col for col in pressure_df.columns if str(col).strip().lower() == 'lst'), None)
    using_reference_cycle = False
    print(f"Primary pressure path: {primary_pressure_path}")
    print(f"Primary pressure columns: {list(pressure_df.columns)}")

    if lst_col is None:
        reference_pressure_path = os.path.join(data_dir, 'sb_12m_pressure.csv')
        pressure_path = reference_pressure_path
        pressure_df = pd.read_csv(reference_pressure_path)
        pressure_df['Date'] = pd.to_datetime(pressure_df['Date'], utc=True)
        lst_col = next((col for col in pressure_df.columns if str(col).strip().lower() == 'lst'), None)
        using_reference_cycle = True
        print(f"Falling back to reference LST file: {reference_pressure_path}")
        print(f"Reference pressure columns: {list(pressure_df.columns)}")

    if lst_col is None:
        raise KeyError(
            f"No LST column found in either {primary_pressure_path} or {pressure_path}. "
            f"Available columns in fallback file: {list(pressure_df.columns)}"
        )
    print(f"Using pressure path for LST lookup: {pressure_path}")
    print(f"Using LST column: {lst_col}")
    print(f"Using reference-cycle normalization: {using_reference_cycle}")
    pressure_df = pressure_df[['Date', lst_col]].drop_duplicates(subset=['Date'], keep='first').sort_values('Date')
    lst_by_date = pressure_df.set_index('Date')[lst_col]
    print(f"Unique Date -> LST rows available: {len(lst_by_date)}")

    def _normalize_to_reference_cycle(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        normalized_year = 2023 if ts.month >= 10 else 2024
        return pd.Timestamp(
            normalized_year,
            ts.month,
            ts.day,
            ts.hour,
            ts.minute,
            ts.second,
            tz='UTC',
        )

    normalized_times = []
    for ts in idx_to_timestamp.values():
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        if using_reference_cycle:
            ts = _normalize_to_reference_cycle(ts)
        normalized_times.append(ts)
    requested_times = pd.DatetimeIndex(normalized_times)
    print(f"Operational timestamps to map: {len(requested_times)}")
    if len(requested_times) > 0:
        print(f"First 5 normalized timestamps: {[str(ts) for ts in requested_times[:5]]}")
    aligned_lst = lst_by_date.reindex(
        requested_times,
        method='nearest',
        tolerance=pd.Timedelta(minutes=30),
    )
    matched_count = int(pd.Series(aligned_lst).notna().sum())
    print(f"Timestamps matched to an LST value within 30 minutes: {matched_count}/{len(requested_times)}")

    bins_as_labels = np.arange(0, 24., 0.5).astype(float)
    bin_edges = np.arange(0, 24.5, 0.5)
    aligned_df = pd.DataFrame({'timestamp': requested_times, 'lst': aligned_lst.values})
    aligned_df['lst_bin'] = pd.cut(aligned_df['lst'], bin_edges, labels=bins_as_labels).astype(float)
    lst_bin_count = int(aligned_df['lst_bin'].notna().sum())
    print(f"Timestamps assigned a valid LST bin: {lst_bin_count}/{len(aligned_df)}")
    if lst_bin_count == 0 and len(aligned_df) > 0:
        missing_preview = aligned_df[['timestamp', 'lst', 'lst_bin']].head(10)
        print("No valid LST bins were produced. Sample rows:")
        print(missing_preview.to_string(index=False))
    elif lst_bin_count < len(aligned_df):
        missing_preview = aligned_df[aligned_df['lst_bin'].isna()][['timestamp', 'lst', 'lst_bin']].head(10)
        print("Sample timestamps that still have lst_bin=None:")
        print(missing_preview.to_string(index=False))

    lst_bin_by_timestamp = {
        row['timestamp']: row['lst_bin']
        for _, row in aligned_df.iterrows()
        if pd.notna(row['lst_bin'])
    }
    time_to_lst_bin = {}
    for idx, ts in idx_to_timestamp.items():
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        if using_reference_cycle:
            ts = _normalize_to_reference_cycle(ts)
        lst_bin = lst_bin_by_timestamp.get(ts)
        if pd.notna(lst_bin):
            time_to_lst_bin[idx] = float(lst_bin)
    print(f"Final time_to_lst_bin entries: {len(time_to_lst_bin)}/{len(idx_to_timestamp)}")
    if len(time_to_lst_bin) > 0:
        preview_items = list(time_to_lst_bin.items())[:10]
        print(f"First mapped entries: {preview_items}")
    print("--- End time -> LST_bin map build ---\n")
    return time_to_lst_bin


def _build_perfect_weather_forecasts(
        realized_weather: Dict[int, Tuple[float, float]],
        time_steps: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Build a perfect-forecast view of realized weather for oracle-style selectors."""
    perfect_weather_forecasts = {}
    pwv_arr = np.array([realized_weather.get(i, (np.nan, np.nan))[0] for i in range(time_steps)])
    rms_arr = np.array([realized_weather.get(i, (np.nan, np.nan))[1] for i in range(time_steps)])
    std_arr = np.full(time_steps, 0.0)
    for t_idx in range(time_steps):
        rms_mean_rel = rms_arr[t_idx:].copy()
        rms_std_rel = std_arr[t_idx:].copy()
        rms_mean_full, rms_std_full = build_global_rms_arrays(
            rms_mean_rel, rms_std_rel, t_idx, time_steps
        )
        perfect_weather_forecasts[t_idx] = {
            'pwv_mean': pwv_arr.copy(),
            'pwv_std': std_arr.copy(),
            'rms_mean': rms_mean_full,
            'rms_std': rms_std_full,
            RMS_LOOKAHEAD_MEAN_KEY: rms_mean_rel,
            RMS_LOOKAHEAD_STD_KEY: rms_std_rel,
            RMS_LAYOUT_KEY: RMS_LAYOUT_ISSUE_LOOKAHEAD,
            'forecast_issue_idx': t_idx,
        }
    return perfect_weather_forecasts

def map_sbs_by_time_distance(df1, df2):
    """
    Matches SB_UIDs from two dataframes based on nearest time value
    within groups defined by project code and execution count.

    Args:
        df1 (pd.DataFrame): The first dataframe (c10_old_master_list).
                            Must contain ['SB_UID', 'PRJ_CODE', 'NUMBER_OF_EXECUTIONS', 'SB_TOTAL_ESTIMATED_TIME'].
        df2 (pd.DataFrame): The second dataframe (sbs_df_raw).
                            Must contain ['SB_UID', 'CODE', 'execount', 'estimatedTime'].

    Returns:
        dict: A dictionary mapping SB_UID from df1 to the best matching SB_UID from df2.
    """
    # 1. Prepare DataFrames for merging
    # Select necessary columns and rename for consistency
    df1_prep = df1[['SB_UID', 'PRJ_CODE', 'NUMBER_OF_EXECUTIONS', 'SB_TOTAL_ESTIMATED_TIME']].copy()

    df2_prep = df2[['SB_UID', 'CODE', 'execount', 'estimatedTime']].copy()
    df2_prep.rename(columns={
        'SB_UID': 'SB_UID_df2',
        'CODE': 'PRJ_CODE',
        'execount': 'NUMBER_OF_EXECUTIONS',
        'estimatedTime': 'time_df2'
    }, inplace=True)

    # 2. Cross-merge on the exact matching keys.
    # This creates a DataFrame with all possible pairs within each (PRJ_CODE, NUMBER_OF_EXECUTIONS) group.
    merged_df = pd.merge(df1_prep, df2_prep, on=['PRJ_CODE', 'NUMBER_OF_EXECUTIONS'])

    # 3. Calculate the absolute time difference for each potential pair
    merged_df['time_diff'] = np.abs(merged_df['SB_TOTAL_ESTIMATED_TIME'] - merged_df['time_df2'])

    # 4. Find the best match for each SB from the original df1
    # We group by the original SB_UID and find the index of the minimum time difference.
    # The .loc[] indexer then selects the full row for each of these best matches.
    # This is a highly efficient way to perform this operation.
    best_matches_idx = merged_df.groupby('SB_UID')['time_diff'].idxmin()
    best_matches_df = merged_df.loc[best_matches_idx]

    # 5. Create the final mapping dictionary
    # We set the index to be the SB_UID from df1 and select the corresponding SB_UID from df2.
    final_map = pd.Series(
        best_matches_df['SB_UID_df2'].values,
        index=best_matches_df['SB_UID']
    ).to_dict()

    return final_map


def get_basic_strategy_quotas(final_schedule, job_lookup, realized_weather):
    # Calculate the EB you would want based on what's been completed so far.
    ct_so_far = {executive: 0 for executive in EXECUTIVE_QUOTAS}
    for entry in final_schedule:
        job_id = entry.split("@")[0]
        job = job_lookup[job_id]
        if isinstance(job['executive'], str):
            ct_so_far[job['executive']] += job['length']
        else:
            for executive in job['executive']:
                ct_so_far[executive] += job['length'] * job['executive'][executive]
    # Now you need to calculate the total time you would need to spend in each executive and subtract that off
    total_time = 0
    for x in realized_weather.values():
        if not np.isnan(x[0]) and not np.isnan(x[1]):
            total_time += 1
    exec_targets_frac = {'NA': .3375, 'EA': .225, 'CL': .1, 'EU': .3375}
    target_per_exec = {executive: exec_targets_frac[executive] * total_time for executive in exec_targets_frac}
    updated_target_per_exec = {executive: max(target_per_exec[executive] - ct_so_far[executive], 0) for executive in
                               target_per_exec}
    total_remaining = sum(updated_target_per_exec.values())
    basic_strategy_quotas = {executive: (updated_target_per_exec[executive] / total_remaining, 1.0) for executive in
                            updated_target_per_exec}
    return basic_strategy_quotas

def _add_job_time_to_exec_balance(
    exec_time_dict: Dict[str, float], job: Dict[str, Any]
) -> None:
    """
    Accumulates job execution time, handling both single and fractional executives.
    """
    executive_info = job.get("executive")
    job_length = job.get("length", 0)
    if isinstance(executive_info, str):
        exec_time_dict[executive_info] += job_length
    elif isinstance(executive_info, dict):
        for exec_name, fraction in executive_info.items():
            exec_time_dict[exec_name] += job_length * fraction


def save_checkpoint(checkpoint_path: str, final_schedule: list, jobs_copy: list, total_value: float,
                    config_index: int, realized_weather: dict, weather_forecasts: dict = None,
                    idx_to_timestamp: dict = None):
    """
    Saves a checkpoint of the simulation state.
    
    Args:
        checkpoint_path: Path where checkpoint will be saved
        final_schedule: Current schedule state
        jobs_copy: Current jobs state with remaining_execs
        total_value: Cumulative value so far
        config_index: Index of the last completed configuration
        realized_weather: Weather data dictionary
        weather_forecasts: Forecast data dictionary (optional)
        idx_to_timestamp: Time index mapping (optional)
    """
    checkpoint_data = {
        'final_schedule': final_schedule,
        'jobs_copy': jobs_copy,
        'total_value': total_value,
        'config_index': config_index,
        'realized_weather': realized_weather,
        'weather_forecasts': weather_forecasts,
        'idx_to_timestamp': idx_to_timestamp
    }

    # First save new checkpoint to a temporary file, then delete the old one
    temp_path = checkpoint_path + '.tmp'
    with open(temp_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    os.rename(temp_path, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path} (after configuration {config_index})")


def _load_forecast_availability_frames(data_path: str, forecast_availability_dir: str) -> List[pd.DataFrame]:
    rolling_dir = os.path.join(data_path, forecast_availability_dir)
    rolling_paths = sorted(glob.glob(os.path.join(rolling_dir, "dsa_sim_issue_*_df.csv")))
    if rolling_paths:
        print(
            f"Loading rolling forecast availability from {rolling_dir} "
            f"({len(rolling_paths)} issuance files).",
            flush=True,
        )
        return [
            pd.read_csv(file_path)
            for file_path in tqdm(rolling_paths, desc="Loading rolling forecast availability")
        ]

    legacy_dir = os.path.join(data_path, "dsa_sim_for_forecast")
    legacy_paths = sorted(glob.glob(os.path.join(legacy_dir, "dsa_sim_*_df.csv")))
    if legacy_paths:
        print(
            f"Rolling forecast directory {rolling_dir} is empty; falling back to legacy "
            f"daily forecast availability in {legacy_dir}.",
            flush=True,
        )
        return [pd.read_csv(file_path) for file_path in legacy_paths]

    print(
        f"WARNING: No forecast availability files found in either {rolling_dir} or {legacy_dir}.",
        flush=True,
    )
    return []


def _load_saved_algorithm_result(pkl_path: str, result_key: str) -> Dict[str, Any]:
    """Load one algorithm's saved result dict from a full_year pickle."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if result_key in data:
        return data[result_key]
    if len(data) == 1:
        return next(iter(data.values()))
    raise KeyError(f"Could not find result_key='{result_key}' in {pkl_path}; keys={list(data.keys())}")


def load_checkpoint(checkpoint_path: str):
    """
    Loads a checkpoint of the simulation state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint data, or None if checkpoint doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    print(f"Checkpoint loaded: resuming from configuration {checkpoint_data['config_index']}")
    return checkpoint_data


def cleanup_checkpoint(checkpoint_path: str):
    """
    Removes checkpoint file if it exists.
    
    Args:
        checkpoint_path: Path to checkpoint file
    """
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint cleaned up: {checkpoint_path}")


def print_cumulative_schedule_summary(config_name: str, period_schedule: list, final_schedule: list,
                                      jobs_for_period: list, job_lookup: dict, 
                                      all_projects: list,
                                      period_weather: dict, realized_weather: dict,
                                      start_idx: int, end_idx: int, 
                                      period_time_steps: int, total_time_steps: int,
                                      weights: dict = None,
                                      executive_quotas_frac: dict = None,
                                      total_observable_time: int = None):
    """
    Prints a detailed summary of the schedule, distinguishing between attempted
    and successful executions.
    """
    print(f"\n{'=' * 15} CUMULATIVE SUMMARY AFTER PERIOD: {config_name.upper()} {'=' * 15}")

    # Create lookup for period jobs (which have local time indices)
    period_job_lookup = {j['job_id']: j for j in jobs_for_period}

    # --- Period-specific stats ---
    period_success_count = 0
    for entry in period_schedule:
        job_id, t_start_str = entry.split('@')
        job = period_job_lookup[job_id]  # Use period job with local time indices
        t_start = int(t_start_str)  # This is local time
        # was_successful, reason = _is_execution_successful(job, t_start, period_weather, period_time_steps)
        was_successful = True
        reason = "Success"
        if was_successful:
            period_success_count += 1
        else:
            global_t = t_start + start_idx
            print(f"  [local_t={t_start_str}, global_t={global_t}] FAILURE: Scheduled {job_id} failed due to {reason}.")

    print(f"\n--- Period '{config_name}' Performance ---")
    # print(f"  SBs Attempted: {len(period_schedule)}")
    #
    # if len(period_schedule) > 0:
    #     success_rate_period = (period_success_count / len(period_schedule)) * 100
    #     print(f"  Successful Executions: {period_success_count} ({success_rate_period:.1f}% success rate)")
    # else:
    #     print(f"  Successful Executions: 0 (N/A success rate)")

    # --- Cumulative stats ---
    cumulative_successful_execs = []
    cumulative_exec_time_successful = defaultdict(float)
    
    # Track by grade
    grade_hours = {'A': 0, 'B': 0, 'C': 0}
    grade_job_count = {'A': 0, 'B': 0, 'C': 0}
    successful_job_ids_by_grade = {'A': set(), 'B': set(), 'C': set()}

    for entry in final_schedule:
        job_id, t_start_str = entry.split('@')
        job = job_lookup[job_id]
        t_start = int(t_start_str)
        # was_successful, reason = _is_execution_successful(job, t_start, realized_weather, total_time_steps)
        was_successful = True
        reason = "Success"
        if was_successful:
            cumulative_successful_execs.append(entry)
            _add_job_time_to_exec_balance(cumulative_exec_time_successful, job)
            
            # Track by grade
            grade = job.get('grade', 'C')
            grade_hours[grade] += job['length']
            grade_job_count[grade] += 1
            successful_job_ids_by_grade[grade].add(job_id)
        else:
            print(f"  [t={t_start_str}] FAILURE: Scheduled {job_id} failed due to {reason}.")


    total_time_scheduled_successfully = sum(cumulative_exec_time_successful.values())
    total_usable_time_so_far = sum(
        1 for t in range(end_idx)
        if not np.isnan(realized_weather.get(t, (np.nan, np.nan))[0])
    )

    print("\n--- Cumulative Performance So Far ---")
    print(f"  Total SBs Attempted: {len(final_schedule)}")
    if len(final_schedule) > 0:
        success_rate_cumulative = (len(cumulative_successful_execs) / len(final_schedule)) * 100
        print(
            f"  Total Successful Executions: {len(cumulative_successful_execs)} ({success_rate_cumulative:.1f}% overall success rate)")
    else:
        print(f"  Total Successful Executions: 0 (N/A overall success rate)")
    print(f"  Total Time from Successful SBs: {total_time_scheduled_successfully:.0f} slots")

    if total_usable_time_so_far > 0:
        utilization = (total_time_scheduled_successfully / total_usable_time_so_far) * 100
        print(
            f"  Cumulative Telescope Utilization: {utilization:.2f}% (based on {total_usable_time_so_far} usable slots so far)")

    # --- Calculate completed projects ---
    completed_projects_by_grade = {'A': 0, 'B': 0, 'C': 0}
    for project in all_projects:
        project_grade = project.get('grade', 'C')
        # Check if all jobs in this project have remaining_execs == 0
        all_jobs_complete = all(
            job_lookup[job_id]['remaining_execs'] == 0 
            for job_id in project.get('job_ids', [])
            if job_id in job_lookup
        )
        if all_jobs_complete and len(project.get('job_ids', [])) > 0:
            completed_projects_by_grade[project_grade] += 1

    # --- Print grade statistics ---
    print("\n  Cumulative Statistics by Grade:")
    print(f"    {'Grade':<7} | {'Hours':>8} | {'Jobs':>8} | {'Projects Completed':>20}")
    print(f"    {'-' * 7} | {'-' * 8} | {'-' * 8} | {'-' * 20}")
    for grade in ['A', 'B', 'C']:
        print(f"    {grade:<7} | {grade_hours[grade]:>8.0f} | {grade_job_count[grade]:>8} | {completed_projects_by_grade[grade]:>20}")
    
    total_grade_hours = sum(grade_hours.values())
    total_grade_jobs = sum(grade_job_count.values())
    total_completed_projects = sum(completed_projects_by_grade.values())
    print(f"    {'-' * 7} | {'-' * 8} | {'-' * 8} | {'-' * 20}")
    print(f"    {'TOTAL':<7} | {total_grade_hours:>8.0f} | {total_grade_jobs:>8} | {total_completed_projects:>20}")

    print("\n  Cumulative Executive Balance (from successful SBs):")
    if total_time_scheduled_successfully > 0:
        print(f"    {'Executive':<10} | {'Time Slots':>10} | {'Fraction':>10}")
        print(f"    {'-' * 10} | {'-' * 10} | {'-' * 10}")
        sorted_execs = sorted(cumulative_exec_time_successful.items(), key=lambda item: item[1], reverse=True)
        for exec_name, time in sorted_execs:
            fraction = (time / total_time_scheduled_successfully) * 100
            print(f"    {exec_name:<10} | {time:>10.0f} | {fraction:>9.2f}%")
    else:
        print("    No successful executions yet.")
    
    # --- Calculate paper objective value if weights provided ---
    # Use observable time through end_idx (time so far) so value is comparable across algorithms
    if weights is not None and executive_quotas_frac is not None:
        observable_time_so_far = sum(
            1 for t in range(end_idx)
            if not np.isnan(realized_weather.get(t, (np.nan, np.nan))[0])
        )
        if observable_time_so_far > 0:
            objective_value = compute_paper_objective_value(
                successful_schedule_log=cumulative_successful_execs,
                jobs=list(job_lookup.values()),
                projects=all_projects,
                exec_time_used=dict(cumulative_exec_time_successful),
                total_observable_time=observable_time_so_far,
                weights=weights,
                executive_quotas_frac=executive_quotas_frac,
                verbose=True
            )
            print(f"\n  Paper Objective Value: {objective_value:.6f}")

    print("=" * (42 + len(config_name)), flush=True)


def print_greedy_cumulative_summary(
    config_name: str,
    final_schedule: list,
    jobs: list,
    projects: list,
    realized_weather: dict,
    total_time_steps: int,
    weights: dict,
    executive_quotas_frac: dict,
    total_observable_time: int,
    end_idx: int
):
    """
    Prints a cumulative summary for greedy algorithm after each configuration,
    including the paper objective value.
    """
    print(f"\n{'=' * 15} CUMULATIVE SUMMARY AFTER CONFIGURATION: {config_name.upper()} {'=' * 15}")
    
    job_lookup = {j['job_id']: j for j in jobs}
    
    # --- Cumulative stats ---
    cumulative_successful_execs = []
    cumulative_exec_time_successful = defaultdict(float)
    
    # Track by grade
    grade_hours = {'A': 0, 'B': 0, 'C': 0}
    grade_job_count = {'A': 0, 'B': 0, 'C': 0}
    successful_job_ids_by_grade = {'A': set(), 'B': set(), 'C': set()}
    
    for entry in final_schedule:
        job_id, t_start_str = entry.split('@')
        job = job_lookup[job_id]
        t_start = int(t_start_str)
        # was_successful, reason = _is_execution_successful(job, t_start, realized_weather, total_time_steps)
        was_successful = True
        reason = "Success"
        if was_successful:
            cumulative_successful_execs.append(entry)
            _add_job_time_to_exec_balance(cumulative_exec_time_successful, job)
            
            # Track by grade
            grade = job.get('grade', 'C')
            grade_hours[grade] += job['length']
            grade_job_count[grade] += 1
            successful_job_ids_by_grade[grade].add(job_id)
    
    total_time_scheduled_successfully = sum(cumulative_exec_time_successful.values())
    total_usable_time_so_far = sum(
        1 for t in range(end_idx)
        if not np.isnan(realized_weather.get(t, (np.nan, np.nan))[0])
    )
    
    print("\n--- Cumulative Performance So Far ---")
    print(f"  Total SBs Attempted: {len(final_schedule)}")
    if len(final_schedule) > 0:
        success_rate_cumulative = (len(cumulative_successful_execs) / len(final_schedule)) * 100
        print(
            f"  Total Successful Executions: {len(cumulative_successful_execs)} ({success_rate_cumulative:.1f}% overall success rate)")
    else:
        print(f"  Total Successful Executions: 0 (N/A overall success rate)")
    print(f"  Total Time from Successful SBs: {total_time_scheduled_successfully:.0f} slots")
    
    if total_usable_time_so_far > 0:
        utilization = (total_time_scheduled_successfully / total_usable_time_so_far) * 100
        print(
            f"  Cumulative Telescope Utilization: {utilization:.2f}% (based on {total_usable_time_so_far} usable slots so far)")
    
    # --- Calculate completed projects ---
    completed_projects_by_grade = {'A': 0, 'B': 0, 'C': 0}
    for project in projects:
        project_grade = project.get('grade', 'C')
        # Check if all jobs in this project have remaining_execs == 0
        all_jobs_complete = all(
            job_lookup[job_id]['remaining_execs'] == 0 
            for job_id in project.get('job_ids', [])
            if job_id in job_lookup
        )
        if all_jobs_complete and len(project.get('job_ids', [])) > 0:
            completed_projects_by_grade[project_grade] += 1
    
    # --- Print grade statistics ---
    print("\n  Cumulative Statistics by Grade:")
    print(f"    {'Grade':<7} | {'Hours':>8} | {'Jobs':>8} | {'Projects Completed':>20}")
    print(f"    {'-' * 7} | {'-' * 8} | {'-' * 8} | {'-' * 20}")
    for grade in ['A', 'B', 'C']:
        print(f"    {grade:<7} | {grade_hours[grade]:>8.0f} | {grade_job_count[grade]:>8} | {completed_projects_by_grade[grade]:>20}")
    
    total_grade_hours = sum(grade_hours.values())
    total_grade_jobs = sum(grade_job_count.values())
    total_completed_projects = sum(completed_projects_by_grade.values())
    print(f"    {'-' * 7} | {'-' * 8} | {'-' * 8} | {'-' * 20}")
    print(f"    {'TOTAL':<7} | {total_grade_hours:>8.0f} | {total_grade_jobs:>8} | {total_completed_projects:>20}")
    
    print("\n  Cumulative Executive Balance (from successful SBs):")
    if total_time_scheduled_successfully > 0:
        print(f"    {'Executive':<10} | {'Time Slots':>10} | {'Fraction':>10}")
        print(f"    {'-' * 10} | {'-' * 10} | {'-' * 10}")
        sorted_execs = sorted(cumulative_exec_time_successful.items(), key=lambda item: item[1], reverse=True)
        for exec_name, time in sorted_execs:
            fraction = (time / total_time_scheduled_successfully) * 100
            print(f"    {exec_name:<10} | {time:>10.0f} | {fraction:>9.2f}%")
    else:
        print("    No successful executions yet.")
    
    # --- Calculate paper objective value ---
    # Use observable time through end_idx (time so far) so value is comparable with strategic_* summaries
    observable_time_so_far = sum(
        1 for t in range(end_idx)
        if not np.isnan(realized_weather.get(t, (np.nan, np.nan))[0])
    )
    if observable_time_so_far <= 0:
        observable_time_so_far = 1
    objective_value = compute_paper_objective_value(
        successful_schedule_log=cumulative_successful_execs,
        jobs=jobs,
        projects=projects,
        exec_time_used=dict(cumulative_exec_time_successful),
        total_observable_time=observable_time_so_far,
        weights=weights,
        executive_quotas_frac=executive_quotas_frac,
        verbose=True
    )
    
    print(f"\n  Paper Objective Value: {objective_value:.6f}")
    print("=" * (42 + len(config_name)), flush=True)


def run_sim_full_year(start_date_str: str, end_date_str: str, seed: int, data_dir: str,
                      long_term_schedule_path: str, weights:dict, save_path:str, osco_n_threads: int,
                      algorithm_name: str, osco_num_samples: int = None,
                      sequence_horizon_steps: int = 16,
                      eb_ramp_exponent: float = 50.0,
                      counter_bonus_a_multiplier: float = 100.0,
                      counter_bonus_b_multiplier: float = 25.0,
                      debug: bool = False,
                      preprocessed_weather_path: str = None,
                      preprocessed_forecasts_path: str = None,
                      forecast_availability_dir: str = "dsa_sim_for_forecast_rolling",
                      sequence_gurobi_use_quadratic_eb: bool = False,
                      sequence_gurobi_log_dir: Optional[str] = None,
                      osco_inner_gurobi_time_limit_seconds: float = 10.0,
                      osco_log_sub_timings: bool = False,
                      osco_gurobi_use_actual_job_metadata_forecast: bool = False,
                      osco_gurobi_use_realized_pwv_forecast: bool = False,
                      osco_gurobi_use_realized_rms_forecast: bool = False):
    """
    Runs the full simulation over a specified date range.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Use provided osco_num_samples or fall back to default
    if osco_num_samples is None:
        osco_num_samples = OSCO_SAMP_SIZE
    print(f"OSCO evaluation budget (num_samples): {osco_num_samples}")

    start_date = pd.to_datetime(start_date_str, utc=True)
    end_date = pd.to_datetime(end_date_str, utc=True)
    simulation_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    num_days = len(simulation_dates)
    max_time_steps = num_days * 48

    log_dsa_eb_run_configuration(
        source_label="full_year",
        algorithm_name=algorithm_name,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
        weights=weights,
        executive_quotas=_FIXED_EXECUTIVE_QUOTAS,
        eb_ramp_exponent=eb_ramp_exponent,
        extra_fields={
            "data_dir": data_dir,
            "forecast_availability_dir": forecast_availability_dir,
            "preprocessed_weather_path": preprocessed_weather_path,
            "preprocessed_forecasts_path": preprocessed_forecasts_path,
            "debug": debug,
            "max_time_steps": max_time_steps,
            "counter_bonus_a_multiplier": counter_bonus_a_multiplier,
            "counter_bonus_b_multiplier": counter_bonus_b_multiplier,
        },
    )

    print(f"Running simulation from {start_date.date()} to {end_date.date()} ({num_days} days).", flush=True)
    print(f"Total time steps to simulate: {max_time_steps}", flush=True)

    daily_dfs = []
    daily_score_dfs = []
    data_path = data_dir
    for current_date in tqdm(simulation_dates, desc="Loading daily data"):
        file_month, file_day, file_year = current_date.month, current_date.day, current_date.year
        file_path = os.path.join(data_path, "dsa_sim", f"dsa_sim_{file_month}_{file_day}_{file_year}_df.csv")
        print(f"Loading daily data from {file_path}", flush=True)
        if os.path.exists(file_path):
            daily_dfs.append(pd.read_csv(file_path))
        else:
            print(f"WARNING: Availability file not found for {current_date.strftime('%Y-%m-%d')}. Using dummy.", flush=True)

        score_file_path = os.path.join(
            data_path, "dsa_sim_scores",
            f"dsa_sim_scores_{file_month}_{file_day}_{file_year}_df.csv"
        )
        if os.path.exists(score_file_path):
            daily_score_dfs.append(pd.read_csv(score_file_path))
        else:
            print(f"WARNING: DSA scores file not found for {current_date.strftime('%Y-%m-%d')}.", flush=True)

    forecast_daily_dfs = None
    if preprocessed_forecasts_path is not None:
        forecast_daily_dfs = _load_forecast_availability_frames(
            data_path=data_path,
            forecast_availability_dir=forecast_availability_dir,
        )
        print(
            "DSA_EB raw inputs [full_year]: "
            f"daily_availability_frames={len(daily_dfs)}, "
            f"daily_score_frames={len(daily_score_dfs)}, "
            f"forecast_availability_frames={len(forecast_daily_dfs)}",
            flush=True,
        )
    else:
        print(
            "Skipping forecast-availability CSV scan "
            "(--preprocessed_forecasts not provided; not required for this algorithm).",
            flush=True,
        )

    daily_score_dfs = daily_score_dfs if daily_score_dfs else None
    # fractional_exec_df = pd.read_csv(os.path.join(data_path, "fractional_execs.csv"))
    fractional_exec_df = pd.read_csv(os.path.join(data_path, "proposals_time_share_mod.csv"))

    # already_executed = pd.read_csv(os.path.join(data_path, "cycle_10_data_for_comp_s01.csv"))
    # remaining_execution_counts = {}
    # for _, row in already_executed.iterrows():
    #     time_per_execution = row['actual_execution_time'] / row['execution_count']
    #     remaining_executions = row['actual_execution_time_c10_start'] / time_per_execution
    #     sb_uid = row['sb_uid']
    #     remaining_execution_counts[sb_uid] = int(np.round(remaining_executions))

    # Fields are CODE,CYCLE,SB_UID,array,execution_count,estimated_execution_time_prj_hours,estimated_eb_time_hours,total_eb_time_before_c10,total_ef_sum_before_c10,execution_count_start_c10
    already_executed = pd.read_csv(os.path.join(data_path, "cycle_10_sb_active_time_to_complete_at_c10_start.csv"))
    remaining_execution_counts = {}
    for _, row in already_executed.iterrows():
        sb_uid = row['SB_UID']
        remaining_execution_counts[sb_uid] = int(np.round(row['execution_count_start_c10']))

    print("Loading shifts and downtime data...", flush=True)
    shifts_df = pd.read_csv(os.path.join(data_path, "shifts_dimensions.csv"))
    downtimes_df = pd.read_csv(os.path.join(data_path, "downtimes_dimensions.csv"))
    print("Shifts and downtime data loaded.", flush=True)

    sbs_df_raw = pd.read_csv(os.path.join(data_path, "schedblocks_c10.csv"))
    projects_df_raw = pd.read_csv(os.path.join(data_path, "projects_c10.csv"))

    # Load preprocessed realized weather (produced by preprocess_weather.py)
    print(f"Loading preprocessed weather from {preprocessed_weather_path} ...", flush=True)
    with open(preprocessed_weather_path, "rb") as f:
        pw_data = pickle.load(f)
    preprocessed_realized_weather = pw_data["realized_weather"]
    downtime_index_sets = {
        "weather": pw_data.get("weather_downtime_indices", set()),
        "technical": pw_data.get("technical_downtime_indices", set()),
        "scheduling": pw_data.get("scheduling_downtime_indices", set()),
        "engineering": pw_data.get("engineering_indices", set()),
    }
    print(f"  Loaded {len(preprocessed_realized_weather)} weather slots from preprocessed file.", flush=True)
    print(f"  Downtime sets: weather={len(downtime_index_sets['weather'])}, "
          f"technical={len(downtime_index_sets['technical'])}, "
          f"scheduling={len(downtime_index_sets['scheduling'])}, "
          f"engineering={len(downtime_index_sets['engineering'])}", flush=True)

    preprocessed_weather_forecasts = None
    if preprocessed_forecasts_path is not None:
        # Load preprocessed forecasts (produced by preprocess_forecasts.py)
        print(f"Loading preprocessed forecasts from {preprocessed_forecasts_path} ...", flush=True)
        with open(preprocessed_forecasts_path, "rb") as f:
            pf_data = pickle.load(f)
        preprocessed_weather_forecasts = pf_data["weather_forecasts"]
        is_perfect = pf_data.get("perfect_forecast", False)
        print(f"  Loaded {len(preprocessed_weather_forecasts)} forecast entries "
              f"(mode: {'PERFECT' if is_perfect else 'REAL'}).", flush=True)
        rms_eval = pf_data.get("rms_evaluation", {})
        rms_model_config = rms_eval.get("model_config", {})
        if rms_model_config:
            print(
                "  RMS forecast layout/model: "
                f"{rms_model_config.get('rms_layout', 'legacy')} / "
                f"{rms_model_config.get('engine', 'unknown')}",
                flush=True,
            )
    else:
        print(
            "Skipping preprocessed forecast pickle load "
            "(--preprocessed_forecasts not provided; not required for this algorithm).",
            flush=True,
        )

    jobs, projects, time_steps, idx_to_timestamp, realized_weather, weather_forecasts, dsa_scores = (
        load_and_prepare_data(
            max_time_steps, sbs_df_raw, projects_df_raw, daily_dfs, daily_score_dfs,
            fractional_exec_df, remaining_execution_counts, shifts_df=shifts_df,
            downtimes_df=downtimes_df, fraction_jobs_to_drop=0,
            add_fillers=False, seed=seed,
            cached_realized_weather=preprocessed_realized_weather,
            cached_weather_forecasts=preprocessed_weather_forecasts,
            availability_df_forecast=forecast_daily_dfs if forecast_daily_dfs else None,
            weight_data_dir=data_path,
        ))

    weights = _derive_objective_weights_from_loaded_data(
        base_weights=weights,
        jobs=jobs,
        projects=projects,
    )
    print(
        "Derived CSV-backed grade weights: "
        f"sb_A={weights['sb_A']:.6g}, sb_B={weights['sb_B']:.6g}, sb_C={weights['sb_C']:.6g}, "
        f"proj_A={weights['proj_A']:.6g}, proj_B={weights['proj_B']:.6g}, proj_C={weights['proj_C']:.6g}",
        flush=True,
    )

    print(f"Ended up with {len(projects)} projects and {len(jobs)} jobs.", flush=True)
    log_dsa_eb_dataset_summary(
        source_label="full_year",
        stage_label="post_load_and_prepare_data",
        jobs=jobs,
        projects=projects,
        time_steps=time_steps,
        idx_to_timestamp=idx_to_timestamp,
        realized_weather=realized_weather,
    )
    if time_steps != max_time_steps:
        print(
            f"Warning: Calculated time steps ({max_time_steps}) differs from loaded data time steps ({time_steps}). Using loaded value.")
    # --- END DATA LOADING ---

    # --- Filter by date range once for ALL algorithms (so greedy and strategic_* use same jobs/projects/time_steps) ---
    limited_indices = {i for i, ts in idx_to_timestamp.items() if start_date <= ts <= end_date}
    if limited_indices:
        max_time_idx = max(limited_indices)
        time_steps_limited = max_time_idx + 1
        print(f"\n--- Filtering to date range {start_date.date()} to {end_date.date()} ---")
        print(f"  time_steps: {time_steps} -> {time_steps_limited}")
        time_steps = time_steps_limited

        realized_weather = {t: w for t, w in realized_weather.items() if t < time_steps}
        idx_to_timestamp = {i: ts for i, ts in idx_to_timestamp.items() if i < time_steps}

        jobs_before = len(jobs)
        jobs_filtered = []
        for job in jobs:
            job_available = job.get("available", [])
            job_available_filtered = [t for t in job_available if t < time_steps]
            if job_available_filtered:
                j = job.copy()
                j["available"] = job_available_filtered
                if "pwv_thresholds" in j:
                    j["pwv_thresholds"] = {t: v for t, v in j["pwv_thresholds"].items() if t < time_steps}
                if "forecast_available" in j:
                    j["forecast_available"] = [t for t in j["forecast_available"] if t < time_steps]
                if "forecast_pwv_thresholds" in j:
                    j["forecast_pwv_thresholds"] = {t: v for t, v in j["forecast_pwv_thresholds"].items() if t < time_steps}
                if "condition_scores" in j:
                    j["condition_scores"] = {t: v for t, v in j["condition_scores"].items() if t < time_steps}
                if "base_ha_scores" in j:
                    j["base_ha_scores"] = {t: v for t, v in j["base_ha_scores"].items() if t < time_steps}
                jobs_filtered.append(j)
        jobs = jobs_filtered
        print(f"  jobs: {jobs_before} -> {len(jobs)}")

        job_ids_remaining = {j["job_id"] for j in jobs}
        projects_before = len(projects)
        projects_filtered = []
        for project in projects:
            if any(pid in job_ids_remaining for pid in project.get("job_ids", [])):
                p = project.copy()
                p["job_ids"] = [pid for pid in project.get("job_ids", []) if pid in job_ids_remaining]
                projects_filtered.append(p)
        projects = projects_filtered
        print(f"  projects: {projects_before} -> {len(projects)}")
        print("--- End date-range filter ---\n")
        log_dsa_eb_dataset_summary(
            source_label="full_year",
            stage_label="post_date_filter",
            jobs=jobs,
            projects=projects,
            time_steps=time_steps,
            idx_to_timestamp=idx_to_timestamp,
            realized_weather=realized_weather,
        )

    vals = {}
    scheds = {}

    EXECUTIVE_QUOTAS = {  # Only used for prophet over full cycle
        'CL': (0.1, 1.0), 'EA': (0.225, 1.0), 'EU': (0.3375, 1.0), 'NA': (0.3375, 1.0), 'OTHER': (0.0, 1.0)
    }

    if algorithm_name == 'dsa_eb':
        ##############################
        ## DSA_EB OVER FULL CYCLE
        ##############################
        print(f"\n{'=' * 20} RUNNING STRATEGY: DSA_EB (FULL CYCLE) {'=' * 20}", flush=True)

        start_year = start_date.year
        base_calendar = build_config_calendar(
            data_dir=data_dir,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date,
            year=start_year,
        )
        config_calendar = load_config_calendar(start_date, base_calendar)
        config_calendar = config_calendar[config_calendar['Start'] <= end_date].copy()
        if len(config_calendar) > 0 and config_calendar.iloc[-1]['End'] > end_date:
            config_calendar.iloc[-1, config_calendar.columns.get_loc('End')] = end_date

        observable_time = int(np.sum(
            [1 for i in range(time_steps) if not np.isnan(realized_weather.get(i, (np.nan, np.nan))[0])]))
        log_dsa_eb_run_configuration(
            source_label="full_year",
            algorithm_name=algorithm_name,
            start_date=start_date,
            end_date=end_date,
            seed=seed,
            weights=weights,
            executive_quotas=EXECUTIVE_QUOTAS,
            eb_ramp_exponent=eb_ramp_exponent,
            extra_fields={
                "observable_time": observable_time,
                "config_calendar_rows": len(config_calendar),
            },
        )
        log_dsa_eb_dataset_summary(
            source_label="full_year",
            stage_label="pre_planning_loop",
            jobs=jobs,
            projects=projects,
            time_steps=time_steps,
            idx_to_timestamp=idx_to_timestamp,
            realized_weather=realized_weather,
            observable_time=observable_time,
            config_calendar=config_calendar,
        )
        dsa_selector = dsa_eb_selector_factory(
            all_jobs_in_period=jobs,
            all_projects=projects,
            weights=weights,
            executive_quotas_frac=EXECUTIVE_QUOTAS,
            eb_ramp_exponent=eb_ramp_exponent,
        )

        vals[algorithm_name], scheds[algorithm_name] = planning_loop_eb_greedy(
            jobs=deepcopy(jobs), projects=projects, realized_weather=realized_weather,
            time_steps=time_steps, job_selector_fn=dsa_selector, executive_quotas_frac=EXECUTIVE_QUOTAS,
            idx_to_timestamp=idx_to_timestamp,
            config_calendar=config_calendar,
            debug=debug,
            weights=weights,
            total_observable_time=observable_time
        )
    elif algorithm_name == 'prophet':
        ##############################
        ## PROPHET OVER FULL CYCLE
        ##############################
        print(f"\n{'=' * 20} RUNNING STRATEGY: PROPHET (FULL CYCLE) {'=' * 20}", flush=True)

        use_quadratic = False
        vals['prophet'], scheds['prophet'] = solve_prophet_configurable(
            deepcopy(jobs),
            deepcopy(projects),
            deepcopy(realized_weather),
            time_steps,
            convert_quotas(EXECUTIVE_QUOTAS, realized_weather),
            weights,
            [],
            time_limit=6*3600,
            output_flag=1,
            executive_quotas_frac=EXECUTIVE_QUOTAS,
            prophet_only_mode=True,
            use_quadratic_eb=use_quadratic,
            warm_start_schedule=None,
            validation_schedule=None
        )

    elif algorithm_name == 'pulsar':
        ##############################
        ## PULSAR
        ##############################
        print(
            f"\n{'=' * 20} RUNNING STRATEGY: PULSAR {'=' * 20}",
            flush=True,
        )

        use_actual_rollout_metadata = osco_gurobi_use_actual_job_metadata_forecast
        use_realized_rollout_pwv = osco_gurobi_use_realized_pwv_forecast
        use_realized_rollout_rms = osco_gurobi_use_realized_rms_forecast
        if weather_forecasts is None and not (use_realized_rollout_pwv and use_realized_rollout_rms):
            raise ValueError(f"'{algorithm_name}' requires weather_forecasts but none were loaded.")

        c10_old_master_list = pd.read_csv(os.path.join(data_dir, "sb12m_master_prepared_c10.csv"))
        sb_map = {uid: uid for uid in c10_old_master_list['SB_UID']}
        start_year = start_date.year
        preprocessed_root = None
        if preprocessed_weather_path:
            preprocessed_root = os.path.dirname(os.path.dirname(preprocessed_weather_path))

        def pulsar_weekly_solver_wrapper(jobs, projects, exec_time_used, cumulative_observable_time, weights, **kwargs):
            """Wrapper to call the anchored weekly strategic scheduler and return planned SB/week rows."""
            try:
                elapsed_bins = cumulative_observable_time
                week_start_date = kwargs.get('week_start_date', start_date.strftime('%Y-%m-%d'))
                output_basename = (
                    f"pulsar_schedule_{pd.Timestamp(week_start_date).strftime('%Y%m%d')}"
                    f"_elapsed_{elapsed_bins}.csv"
                )
                output_parent = os.path.dirname(save_path)
                temp_output = os.path.join(output_parent, output_basename)
                os.makedirs(output_parent, exist_ok=True)
                print(f"  PULSAR weekly solver output path: {temp_output}", flush=True)

                solve_long_term_schedule_weekly(
                    weights=weights,
                    output_path=temp_output,
                    data_dir=data_dir,
                    jobs=deepcopy(jobs),
                    config_start_date=week_start_date,
                    cycle_anchor_date=start_date.strftime('%Y-%m-%d'),
                    sb_map=sb_map,
                    preprocessed_root=preprocessed_root,
                    calendar_only=False,
                    year=start_year,
                    exec_time_used=exec_time_used,
                    elapsed_bins=elapsed_bins,
                    end_date=end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date),
                    time_limit=kwargs.get('time_limit', 20 * 60),
                    use_lp_relaxation=False,
                )

                strategic_df = pd.read_csv(temp_output)
                prioritized_schedule = []
                for _, row in strategic_df.iterrows():
                    sb_uid = row.get('SB_UID', row.get('sb_uid', None))
                    week_label = row.get('Week_Label', row.get('Configuration', row.get('configuration', None)))
                    if sb_uid and pd.notna(sb_uid) and week_label and pd.notna(week_label):
                        prioritized_schedule.append({
                            'job_id': str(sb_uid),
                            'week_label': str(week_label),
                        })
                return prioritized_schedule
            except Exception as e:
                print(f"  Weekly strategic solver error: {e}", flush=True)
                return []

        vals[algorithm_name], scheds[algorithm_name] = planning_loop_pulsar(
            jobs=deepcopy(jobs),
            projects=projects,
            realized_weather=realized_weather,
            weather_forecasts=weather_forecasts,
            time_steps=time_steps,
            executive_quotas_frac=EXECUTIVE_QUOTAS,
            weights=weights,
            idx_to_timestamp=idx_to_timestamp,
            cycle_start_timestamp=start_date,
            weekly_solver_fn=pulsar_weekly_solver_wrapper,
            weekly_solver_kwargs={},
            replan_every_n_weeks=2,
            sequence_horizon_steps=sequence_horizon_steps,
            oracle_use_actual_job_metadata=use_actual_rollout_metadata,
            use_realized_pwv_forecast=use_realized_rollout_pwv,
            use_realized_rms_forecast=use_realized_rollout_rms,
            inner_gurobi_time_limit_seconds=osco_inner_gurobi_time_limit_seconds,
            max_candidates_per_executive_by_grade=5,
            fill_to_total_candidates=20,
            use_quadratic_eb=sequence_gurobi_use_quadratic_eb,
            gurobi_log_dir=sequence_gurobi_log_dir,
            osco_num_samples=osco_num_samples if osco_num_samples else 5,
            osco_n_threads=osco_n_threads,
            osco_random_seed=seed,
            osco_log_sub_timings=osco_log_sub_timings,
            osco_debug=debug,
            eb_ramp_exponent=eb_ramp_exponent,
            counter_bonus_a_multiplier=counter_bonus_a_multiplier,
            counter_bonus_b_multiplier=counter_bonus_b_multiplier,
        )
    
    # Add Eval.
    print(f"\n--- Evaluating Results ---", flush=True)

    # Calculate total observable time for objective value calculation
    total_observable_time = sum(
        1 for t in range(time_steps)
        if not np.isnan(realized_weather.get(t, (np.nan, np.nan))[0])
    )
    
    final_evaluation_results = run_eval_real(
        vals, scheds, jobs, projects, time_steps,
        idx_to_timestamp, realized_weather, EXECUTIVE_QUOTAS,
        weights=weights, total_observable_time=total_observable_time,
        downtime_index_sets=downtime_index_sets,
    )
    return final_evaluation_results


def main(args):
    output_dir = args.output_dir
    data_dir = args.data_dir
    seed = args.seed
    start_date = args.start_date
    end_date = args.end_date
    task_id = args.slurm_task_id
    algorithm_name = args.algorithm_name
    w_adherence = args.w_adherence
    osco_n_threads = args.osco_n_threads
    osco_num_samples = args.osco_num_samples
    eb_ramp_exponent = args.eb_ramp_exponent
    osco_gurobi_use_actual_job_metadata_forecast = (
        args.osco_gurobi_use_actual_job_metadata_forecast
    )
    osco_gurobi_use_realized_pwv_forecast = (
        args.osco_gurobi_use_realized_pwv_forecast
    )
    osco_gurobi_use_realized_rms_forecast = (
        args.osco_gurobi_use_realized_rms_forecast
    )

    os.makedirs(output_dir, exist_ok=True)

    if all(arg is not None for arg in [args.w_sb, args.w_proj, args.w_util, args.w_ebp]):
        print("--- Setting weights ---")
        w_sb = args.w_sb
        w_proj = args.w_proj
        w_util = args.w_util
        w_ebp = args.w_ebp

        total_weight = w_sb + w_proj + w_util + w_ebp
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"The main weights (sb, proj, util, ebp) must sum to 1.0, but they sum to {total_weight}")

        print(f"Using Fixed Main Weights: SB={w_sb:.2f}, Proj={w_proj:.2f}, Util={w_util:.2f}, EBP={w_ebp:.2f}")

        weights = {
            "adherence": w_adherence,
            "utilization": w_util,
            "eb_penalty": w_ebp,
            # Paper-style combined weights (triggers paper objective path in prophet/score_schedule)
            "obs_completion": w_sb,
            "proj_completion": w_proj,
        }

        weights_str = json.dumps(weights)

        result_filename = f"{algorithm_name}_fixed_weights_seed_{seed}.pkl"

    elif task_id is not None:
        print("--- Running in SWEEP mode (using slurm_task_id) ---")

        criteria = ['w_sb', 'w_proj', 'w_util', 'w_ebp']
        criteria_pairs = list(itertools.combinations(criteria, 2))
        weight_levels = np.array([0.3, 0.6, 0.9])

        num_combinations = len(criteria_pairs) * len(weight_levels)
        if not (0 <= task_id < num_combinations):
            raise ValueError(f"task_id must be between 0 and {num_combinations - 1}, but got {task_id}")

        pair_index = task_id // len(weight_levels)
        weight_index = task_id % len(weight_levels)
        criterion1, criterion2 = criteria_pairs[pair_index]

        epsilon = 1e-6
        weight1 = weight_levels[weight_index]
        weight2 = 1.0 - weight1 - (2 * epsilon)

        main_weights = {'w_sb': epsilon, 'w_proj': epsilon, 'w_util': epsilon, 'w_ebp': epsilon}
        main_weights[criterion1] = weight1
        main_weights[criterion2] = weight2

        w_sb = main_weights['w_sb']
        w_proj = main_weights['w_proj']
        w_util = main_weights['w_util']
        w_ebp = main_weights['w_ebp']

        weights = {
            "adherence": w_adherence,
            "utilization": w_util,
            "eb_penalty": w_ebp,
            # Paper-style combined weights (triggers paper objective path in prophet/score_schedule)
            "obs_completion": w_sb,
            "proj_completion": w_proj
        }

        weights_str = json.dumps(weights)

        print(f"--- Running Experiment for Task ID: {task_id} ---")
        print(f"Testing Pair: ({criterion1}, {criterion2})")
        print(f"Calculated Main Weights: SB={w_sb:.2f}, Proj={w_proj:.2f}, Util={w_util:.2f}, EBP={w_ebp:.2f}")

        result_filename = (
            f"{args.setting_name}_"
            f"task{task_id}_"
            f"{criterion1.replace('w_', '')}{weight1:.2f}_vs_{criterion2.replace('w_', '')}{weight2:.2f}_"
            f"seed_{args.seed}.pkl"
        )
    else:
        raise ValueError(
            "Invalid arguments. You must provide EITHER --slurm_task_id (for sweep mode) "
            "OR all of --w_sb, --w_proj, --w_util, and --w_ebp (for fixed-weight mode)."
        )

    # --- Run the Long-Term Scheduler ---

    long_term_filepath = None
    if algorithm_name.startswith("strategic"):
        print(f"\n{'=' * 80}")
        print("STEP 1: RUNNING LONG-TERM SCHEDULER")
        print(f"Full Weights Dict: {weights_str}")
        print(f"{'=' * 80}\n")

        long_term_filename = f"temp_long_term_schedule_task_{task_id}.csv"
        long_term_filepath = os.path.join(output_dir, long_term_filename)

    ## tk what do i replace jobs_copy with here - might not need anything at all
    ## we don't need sb_map on the first run
    #solve_long_term_schedule(weights, long_term_filepath, data_dir, jobs_copy, start_date, sb_map)

    # --- Run the Full-Year Simulation ---
    print(f"\n{'=' * 80}")
    print("STEP 2: RUNNING FULL-YEAR SIMULATION")
    print(f"{'=' * 80}\n")

    save_path = os.path.join(output_dir, result_filename)

    result = run_sim_full_year(
        start_date_str=start_date,
        end_date_str=end_date,
        seed=seed,
        data_dir=data_dir,
        long_term_schedule_path=long_term_filepath,
        weights=weights,
        save_path=save_path,
        osco_n_threads=osco_n_threads,
        algorithm_name=algorithm_name,
        osco_num_samples=osco_num_samples,
        sequence_horizon_steps=args.sequence_horizon_steps,
        eb_ramp_exponent=eb_ramp_exponent,
        counter_bonus_a_multiplier=args.counter_bonus_a_multiplier,
        counter_bonus_b_multiplier=args.counter_bonus_b_multiplier,
        debug=args.debug,
        preprocessed_weather_path=args.preprocessed_weather,
        preprocessed_forecasts_path=args.preprocessed_forecasts,
        forecast_availability_dir=args.forecast_availability_dir,
        sequence_gurobi_use_quadratic_eb=args.sequence_gurobi_use_quadratic_eb,
        sequence_gurobi_log_dir=args.sequence_gurobi_log_dir,
        osco_inner_gurobi_time_limit_seconds=args.osco_inner_gurobi_time_limit_seconds,
        osco_log_sub_timings=args.osco_log_sub_timings,
        osco_gurobi_use_actual_job_metadata_forecast=osco_gurobi_use_actual_job_metadata_forecast,
        osco_gurobi_use_realized_pwv_forecast=osco_gurobi_use_realized_pwv_forecast,
        osco_gurobi_use_realized_rms_forecast=osco_gurobi_use_realized_rms_forecast,
    )

    print(f"Saving final evaluation results to {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)

    print("\nRun complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run telescope scheduling simulation for a given date range.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output pickle file.")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory where input data CSVs are located.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the simulation.")

    parser.add_argument("--start_date", type=str, default="2023-09-30", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, default="2024-10-01", help="End date in YYYY-MM-DD format (inclusive).")

    parser.add_argument("--w_sb", type=float, default=None,
                        help="Overall weight for completing SBs (for fixed-weight mode).")
    parser.add_argument("--w_proj", type=float, default=None,
                        help="Overall weight for completing Projects (for fixed-weight mode).")
    parser.add_argument("--w_util", type=float, default=None,
                        help="Weight for the utilization ratio (for fixed-weight mode).")
    parser.add_argument("--w_ebp", type=float, default=None,
                        help="Weight for the executive balance L1 penalty (for fixed-weight mode).")
    parser.add_argument("--slurm_task_id", type=int, default=None, help="SLURM task ID for weight sweep mode.")
    parser.add_argument("--w_adherence", type=float, default=0,
                        help="Weight for adherence to long term schedule.")

    parser.add_argument("--algorithm_name", type=str, required=True,
                        choices=[
                            'dsa_eb',
                            'prophet',
                            'pulsar',
                        ])

    parser.add_argument("--osco_n_threads", type=int, default=-1, help="Number of threads to use for OSCO. "
                                                                       "Each thread will run a separate optimization problem, running"
                                                                       "for each pair of scheduling block and sample of future weather."
                                                                       "-1 will just use the number of CPUs.")

    parser.add_argument("--osco_num_samples", type=int, default=None, 
                        help="Number of samples per job for OSCO evaluation budget. "
                             "If not specified, uses OSCO_SAMP_SIZE from fixed_params.py")
    
    parser.add_argument("--planning_horizon_steps", type=int, default=16,
                        help="Planning horizon in number of 30-minute intervals. Default is 16 (8 hours). "
                             "For 3 days, use 144 (3 days * 24 hours * 2 intervals/hour).")
    parser.add_argument("--sequence_horizon_steps", type=int, default=16,
                        help="Number of future 30-minute intervals to search when building job sequences.")
    parser.add_argument("--eb_ramp_exponent", type=float, default=50.0,
                        help="Exponent used by ramped executive-balance penalty schedules. "
                             "Default is 50.0.")
    parser.add_argument("--counter_bonus_a_multiplier", type=float, default=100.0,
                        help="Multiplier applied to an A-grade SB's payoff score when it appears in the "
                             "weekly counter bonus for the PULSAR algorithm.")
    parser.add_argument("--counter_bonus_b_multiplier", type=float, default=25.0,
                        help="Multiplier applied to a B-grade SB's payoff score when it appears in the "
                             "weekly counter bonus for the PULSAR algorithm.")
    parser.add_argument("--osco_inner_gurobi_time_limit_seconds", type=float, default=10.0,
                        help="Per-(arm, sample) Gurobi time limit (seconds) for the "
                             "pulsar algorithm. "
                             "Each decision point runs num_samples x feasible_jobs solves with this limit each.")
    parser.add_argument("--osco_log_sub_timings", action="store_true",
                        help="Log wall-clock start/end times and elapsed seconds for every "
                             "individual Gurobi sub-solve inside the gurobi OSCO rollout algorithm.")
    parser.add_argument("--osco_gurobi_use_actual_job_metadata_forecast", action="store_true",
                        help="For the gurobi OSCO rollout, use actual job metadata "
                             "(availability/PWV thresholds/RMS thresholds) during suffix rollout evaluation.")
    parser.add_argument("--osco_gurobi_use_realized_pwv_forecast", action="store_true",
                        help="For the gurobi OSCO rollout, use realized PWV values in the sampled/suffix forecast inputs.")
    parser.add_argument("--osco_gurobi_use_realized_rms_forecast", action="store_true",
                        help="For the gurobi OSCO rollout, use realized RMS values in the sampled/suffix forecast inputs.")
    parser.add_argument("--sequence_gurobi_use_quadratic_eb", action="store_true",
                        help="Use the direct quadratic EB formulation for the gurobi OSCO rollout "
                             "instead of the default piecewise approximation.")
    parser.add_argument("--sequence_gurobi_log_dir", type=str, default=None,
                        help="Optional directory for per-solve Gurobi log files from the gurobi OSCO rollout.")

    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode to print detailed information about strategic schedules and greedy completion.")
    parser.add_argument("--preprocessed_weather", type=str, required=True,
                        help="Path to pickle file produced by preprocess_weather.py")
    parser.add_argument("--preprocessed_forecasts", type=str, default=None,
                        help="Path to pickle file produced by preprocess_forecasts.py. "
                             "Optional; only required for algorithms that consume weather forecasts "
                             "(e.g. pulsar). "
                             "Skipping it avoids the slow forecast-availability CSV scan and pickle load.")
    parser.add_argument("--forecast_availability_dir", type=str, default="dsa_sim_for_forecast_rolling",
                        help="Directory containing issuance-based forecast availability CSVs. "
                             "Falls back to legacy `dsa_sim_for_forecast` when empty.")

    args = parser.parse_args()
    main(args)
