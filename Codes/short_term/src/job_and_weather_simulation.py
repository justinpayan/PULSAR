import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional
import math
import os
import random
import pandas as pd

from fixed_params import *


def _get_first_matching_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"Missing {label}. Expected one of: {candidates}")


def _resolve_weight_file_paths(
        sbs_df_raw: pd.DataFrame,
        projects_df_raw: pd.DataFrame,
        weight_data_dir: Optional[str],
        sb_weights_path: Optional[str],
        project_weights_path: Optional[str],
) -> Tuple[str, str]:
    resolved_data_dir = (
        weight_data_dir
        or sbs_df_raw.attrs.get("data_dir")
        or projects_df_raw.attrs.get("data_dir")
        or os.getenv("AOOSP_DATA_DIR")
    )

    if sb_weights_path is None or project_weights_path is None:
        if resolved_data_dir is None:
            raise ValueError(
                "Weight files are required. Pass weight_data_dir or explicit "
                "sb_weights_path/project_weights_path so the loader can find "
                "sb_weights.csv and project_weights.csv."
            )
        resolved_data_dir = os.path.abspath(os.path.expanduser(resolved_data_dir))

    resolved_sb_weights_path = sb_weights_path or os.path.join(resolved_data_dir, "sb_weights.csv")
    resolved_project_weights_path = project_weights_path or os.path.join(resolved_data_dir, "project_weights.csv")

    return (
        os.path.abspath(os.path.expanduser(resolved_sb_weights_path)),
        os.path.abspath(os.path.expanduser(resolved_project_weights_path)),
    )


def _load_required_weight_maps(
        sbs_df_raw: pd.DataFrame,
        projects_df_raw: pd.DataFrame,
        weight_data_dir: Optional[str],
        sb_weights_path: Optional[str],
        project_weights_path: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    resolved_sb_weights_path, resolved_project_weights_path = _resolve_weight_file_paths(
        sbs_df_raw=sbs_df_raw,
        projects_df_raw=projects_df_raw,
        weight_data_dir=weight_data_dir,
        sb_weights_path=sb_weights_path,
        project_weights_path=project_weights_path,
    )

    if not os.path.exists(resolved_sb_weights_path):
        raise FileNotFoundError(f"Required SB weight file not found: {resolved_sb_weights_path}")
    if not os.path.exists(resolved_project_weights_path):
        raise FileNotFoundError(f"Required project weight file not found: {resolved_project_weights_path}")

    sb_weights_df = pd.read_csv(resolved_sb_weights_path)
    project_weights_df = pd.read_csv(resolved_project_weights_path)

    sb_uid_col = _get_first_matching_column(sb_weights_df, ["SB_UID", "sb_uid", "job_id"], "SB weight key column")
    project_id_col = _get_first_matching_column(
        project_weights_df,
        ["CODE", "project_id", "project_code"],
        "project weight key column"
    )

    if "weight" not in sb_weights_df.columns:
        raise ValueError(f"SB weights file must contain a 'weight' column: {resolved_sb_weights_path}")
    if "weight" not in project_weights_df.columns:
        raise ValueError(f"Project weights file must contain a 'weight' column: {resolved_project_weights_path}")

    if sb_weights_df[sb_uid_col].duplicated().any():
        duplicates = sorted(sb_weights_df.loc[sb_weights_df[sb_uid_col].duplicated(), sb_uid_col].astype(str).unique())
        raise ValueError(f"Duplicate SB IDs in SB weights file: {duplicates[:10]}")
    if project_weights_df[project_id_col].duplicated().any():
        duplicates = sorted(project_weights_df.loc[project_weights_df[project_id_col].duplicated(), project_id_col].astype(str).unique())
        raise ValueError(f"Duplicate project IDs in project weights file: {duplicates[:10]}")

    sb_weight_map = {
        str(row[sb_uid_col]): float(row["weight"])
        for _, row in sb_weights_df[[sb_uid_col, "weight"]].iterrows()
    }
    project_weight_map = {
        str(row[project_id_col]): float(row["weight"])
        for _, row in project_weights_df[[project_id_col, "weight"]].iterrows()
    }

    return sb_weight_map, project_weight_map


# Generate autoregressive fixed means
def generate_fixed_means(time_steps: int,
                         std_pwv: float,
                         std_rms: float,
                         seed: int) -> Tuple[List[float], List[float]]:
    rng = np.random.default_rng(seed)

    pwv_means = [np.round(rng.uniform(1, NUM_WEATHER_BINS))]
    rms_means = [np.round(rng.uniform(1, NUM_WEATHER_BINS))]

    for _ in range(1, time_steps):
        pwv_means.append(
            np.clip(np.round(pwv_means[-1] + rng.normal(0, std_pwv)), 1, NUM_WEATHER_BINS)
        )
        rms_means.append(
            np.clip(np.round(rms_means[-1] + rng.normal(0, std_rms)), 1, NUM_WEATHER_BINS)
        )

    return pwv_means, rms_means

def generate_fixed_means_diff(time_steps: int,
                         std_pwv: float,
                         std_rms: float,
                         seed: int) -> Tuple[List[float], List[float]]:
    pwv_means = []
    rms_means = []
    for t in range(time_steps):
        if t % 2 == 0:
            pwv_means.append(1)
            rms_means.append(1)
        else:
            pwv_means.append(1)
            rms_means.append(3.3)
    return pwv_means, rms_means

def generate_fixed_means_diff2(time_steps: int,
                         std_pwv: float,
                         std_rms: float,
                         seed: int) -> Tuple[List[float], List[float]]:
    pwv_means = [1.0] * (time_steps // 2)
    rms_means = [1.0] * (time_steps // 2)
    for _ in range((time_steps // 2), time_steps):
        pwv_means.append(2.8)
        rms_means.append(2.8)
    return pwv_means, rms_means


# Generate weather given means and noise parameters
def generate_realized_weather(
    pwv_means: List[float],
    rms_means: List[float],
    std_pwv: float,
    width_rms: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    pwv = np.round(rng.normal(loc=pwv_means, scale=std_pwv))
    rms = np.round(np.random.normal(loc=rms_means, scale=width_rms))

    return np.clip(pwv, 1, NUM_WEATHER_BINS), np.clip(rms, 1, NUM_WEATHER_BINS)


# Generates non-contiguous valid start times
def generate_eligible_times(job_length: int, time_steps: int, chunks: int = 2) -> List[int]:
    starts = []
    chunk_size = time_steps // (chunks + 1)
    for i in range(chunks):
        start = i * chunk_size + random.randint(0, chunk_size - job_length)
        starts.extend(range(start, min(start + random.randint(1, chunk_size), time_steps - job_length + 1)))
    return sorted(list(set(starts)))


def add_fillers(projects, jobs, time_steps):
    job_id = len(jobs)
    for executive in list(EXECUTIVE_QUOTAS.keys()):
        for _ in range(FILLERS_PER_EXEC):
            weight = 0
            # num_jobs = random.randint(*jobs_per_project_range)
            num_jobs = 1
            project_jobs = []

            for _ in range(num_jobs):
                job_length = 1
                valid_starts = list(range(time_steps))
                pwv_thresh = NUM_WEATHER_BINS
                rms_thresh = NUM_WEATHER_BINS

                job = {
                    "job_id": f"j{job_id}",
                    "project_id": f"p{len(projects)}",
                    "length": job_length,
                    "valid_starts": valid_starts,
                    "pwv_thresh": pwv_thresh,
                    "rms_thresh": rms_thresh,
                    "executive": executive,
                    "weight": weight
                }
                jobs.append(job)
                project_jobs.append(job["job_id"])
                job_id += 1

            projects.append({
                "project_id": "p%d" % len(projects),
                "executive": executive,
                "weight": weight,
                "job_ids": project_jobs
            })
    return projects, jobs

def generate_ranked_projects_and_jobs(
        num_projects: int,
        time_steps: int,
        seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate `num_projects` ranked projects and their jobs.
    Project weights are assigned based on rank (highest gets weight=num_projects-1).
    """
    random.seed(seed)

    projects = []
    jobs = []
    job_id = 0

    # Random ranking of project IDs
    project_ids = [f"p{i}" for i in range(num_projects)]
    ranked_projects = random.sample(project_ids, len(project_ids))

    for rank, project_id in enumerate(ranked_projects[::-1]):  # Highest rank gets highest weight
        weight = rank
        executive = random.choices(list(EXECUTIVE_QUOTAS.keys()), weights=[0.35, 0.35, 0.2, 0.1])[0]
        num_jobs = random.randint(*JOBS_PER_PROJECT_RANGE)
        project_jobs = []

        for _ in range(num_jobs):
            job_length = random.randint(*JOB_LENGTH_RANGE)
            valid_starts = generate_eligible_times(job_length, time_steps)
            pwv_thresh = random.randint(1, NUM_WEATHER_BINS)
            rms_thresh = random.randint(1, NUM_WEATHER_BINS)

            job = {
                "job_id": f"j{job_id}",
                "project_id": project_id,
                "length": job_length,
                "valid_starts": valid_starts,
                "pwv_thresh": pwv_thresh,
                "rms_thresh": rms_thresh,
                "executive": executive,
                "weight": weight
            }
            jobs.append(job)
            project_jobs.append(job["job_id"])
            job_id += 1

        projects.append({
            "project_id": project_id,
            "executive": executive,
            "weight": weight,
            "job_ids": project_jobs
        })

    # Add lots of fillers
    projects, jobs = add_fillers(projects, jobs, time_steps)

    return projects, jobs

def generate_ranked_projects_and_jobs_diff(
        num_projects: int,
        time_steps: int,
        seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate ranked projects where each project has jobs that either:
    - Require good weather but can start at any time (strict)
    - Allow bad weather but must start early (flexible)

    All jobs in a project have the same weight equal to the project's weight.
    """
    random.seed(seed)

    projects = []
    jobs = []
    job_id = 0

    project_ids = [f"p{i}" for i in range(num_projects)]
    # ranked_projects = random.sample(project_ids, len(project_ids))
    ranked_projects = project_ids

    for rank, project_id in enumerate(ranked_projects[::-1]):  # Higher rank = higher weight
        weight = rank
        executive = random.choices(list(EXECUTIVE_QUOTAS.keys()), weights=[0.35, 0.35, 0.2, 0.1])[0]

        num_jobs = random.randint(*JOBS_PER_PROJECT_RANGE)
        project_jobs = []

        # is_strict = random.random() < 0.5  # 50% strict-weather, 50% flexible-timing
        is_strict = (int(project_id.split("p")[1]) % 2 == 0)
        # is_strict = (rank > 3)

        for _ in range(num_jobs):

            job_length = random.randint(*JOB_LENGTH_RANGE)

            if is_strict:
                # valid_starts = list(range(time_steps - job_length + 1))  # any time
                valid_starts = [(rank - 1) % time_steps, rank % time_steps]
                # pwv_thresh = random.randint(1, 2)  # needs decent weather
                # rms_thresh = random.randint(1, 2)
                pwv_thresh = 5
                rms_thresh = 3
            else:
                # max_start = max(2, time_steps // 2)
                # max_start = time_steps // 3
                # valid_starts = list(range(min(max_start, time_steps - job_length + 1)))
                valid_starts = [(rank % time_steps)]
                pwv_thresh = NUM_WEATHER_BINS
                rms_thresh = NUM_WEATHER_BINS

            job = {
                "job_id": f"j{job_id}",
                "project_id": project_id,
                "length": job_length,
                "valid_starts": valid_starts,
                "pwv_thresh": pwv_thresh,
                "rms_thresh": rms_thresh,
                "executive": executive,
                "weight": weight  # match project weight
            }
            jobs.append(job)
            project_jobs.append(job["job_id"])
            job_id += 1

        projects.append({
            "project_id": project_id,
            "executive": executive,
            "weight": weight,
            "job_ids": project_jobs
        })

    # Optionally pad with fillers
    projects, jobs = add_fillers(projects, jobs, time_steps)

    return projects, jobs


def generate_ranked_projects_and_jobs_diff2(
        num_projects: int,
        time_steps: int,
        seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate ranked projects where each project has jobs that either:
    - Require good weather but can start at any time (strict)
    - Allow bad weather but must start early (flexible)

    All jobs in a project have the same weight equal to the project's weight.
    """
    random.seed(seed)

    projects = []
    jobs = []
    job_id = 0

    project_ids = [f"p{i}" for i in range(num_projects)]
    # ranked_projects = random.sample(project_ids, len(project_ids))
    ranked_projects = project_ids

    for rank, project_id in enumerate(ranked_projects[::-1]):  # Higher rank = higher weight
        weight = rank
        executive = random.choices(list(EXECUTIVE_QUOTAS.keys()), weights=[0.35, 0.35, 0.2, 0.1])[0]

        num_jobs = random.randint(*JOBS_PER_PROJECT_RANGE)
        project_jobs = []

        # is_strict = random.random() < 0.5  # 50% strict-weather, 50% flexible-timing
        # is_strict = (int(project_id.split("p")[1]) % 2 == 0)
        is_strict = (rank > num_projects // 2)

        for _ in range(num_jobs):

            job_length = random.randint(*JOB_LENGTH_RANGE)

            if is_strict:
                # valid_starts = list(range(time_steps - job_length + 1))  # any time
                max_start = max(2, time_steps // 2)
                # max_start = time_steps // 3
                valid_starts = list(range(min(max_start, time_steps - job_length + 1)))
                # valid_starts = [rank // 2]
                valid_starts.append(rank)
                # valid_starts = [(rank-1) % time_steps, rank % time_steps]
                # pwv_thresh = random.randint(1, 2)  # needs decent weather
                # rms_thresh = random.randint(1, 2)
                pwv_thresh = 3
                rms_thresh = 3
            else:
                max_start = max(2, time_steps // 2)
                # max_start = time_steps // 3
                valid_starts = list(range(min(max_start, time_steps - job_length + 1)))
                # valid_starts = [(rank % time_steps)]
                pwv_thresh = NUM_WEATHER_BINS
                rms_thresh = NUM_WEATHER_BINS

            job = {
                "job_id": f"j{job_id}",
                "project_id": project_id,
                "length": job_length,
                "valid_starts": valid_starts,
                "pwv_thresh": pwv_thresh,
                "rms_thresh": rms_thresh,
                "executive": executive,
                "weight": weight  # match project weight
            }
            jobs.append(job)
            project_jobs.append(job["job_id"])
            job_id += 1

        projects.append({
            "project_id": project_id,
            "executive": executive,
            "weight": weight,
            "job_ids": project_jobs
        })

    # Optionally pad with fillers
    projects, jobs = add_fillers(projects, jobs, time_steps)

    return projects, jobs

def job_sampler_with_weather(num_projects, time_steps, std_pwv, std_rms, seed, diff=False, diff2=False):
    if diff:
        projects, jobs = generate_ranked_projects_and_jobs_diff(num_projects, time_steps, seed)
        pwv_means, rms_means = generate_fixed_means_diff(time_steps, std_pwv, std_rms, seed)
    elif diff2:
        projects, jobs = generate_ranked_projects_and_jobs_diff2(num_projects, time_steps, seed)
        pwv_means, rms_means = generate_fixed_means_diff2(time_steps, std_pwv, std_rms, seed)
    else:
        projects, jobs = generate_ranked_projects_and_jobs(num_projects, time_steps, seed)
        pwv_means, rms_means = generate_fixed_means(time_steps, std_pwv, std_rms, seed)

    pwv_series, rms_series = generate_realized_weather(pwv_means, rms_means, std_pwv, std_rms, seed)

    # Show in a DataFrame
    weather_df = pd.DataFrame({
        "Time": list(range(time_steps)),
        "PWV_mean": pwv_means,
        "RMS_mean": rms_means,
        "PWV_realized": pwv_series,
        "RMS_realized": rms_series
    })

    return jobs, projects, weather_df, pwv_means, rms_means


def load_and_prepare_data(
        max_time_steps: int,
        sbs_df_raw: pd.DataFrame,
        projects_df_raw: pd.DataFrame,
        availability_df_raw: Union[pd.DataFrame, List[pd.DataFrame]],
        scores_df_raw: Optional[Union[pd.DataFrame, List[pd.DataFrame]]],
        fractional_execs: pd.DataFrame,
        remaining_execution_counts: dict,
        shifts_df: pd.DataFrame,
        downtimes_df: pd.DataFrame,
        fraction_jobs_to_drop: float = 0.0,
        add_fillers: bool = True,
        filler_weight: float = 0.0,
        filler_pwv_thresh: float = 8.0,
        filler_rms_thresh: float = 0.0,
        seed: int = None,
        cached_realized_weather: Optional[Dict[int, Tuple[float, float]]] = None,
        cached_weather_forecasts: Optional[Dict[int, Dict[str, Any]]] = None,
        availability_df_forecast: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        weight_data_dir: Optional[str] = None,
        sb_weights_path: Optional[str] = None,
        project_weights_path: Optional[str] = None,
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], int, Dict[int, Any], Dict[int, Any], Dict[int, Dict[str, Any]],
pd.DataFrame]:
    """
    Processes raw DataFrames and loads pre-computed weather data and forecasts.
    """
    if isinstance(availability_df_raw, list):
        availability_df_raw = pd.concat(availability_df_raw)

    if isinstance(scores_df_raw, list):
        scores_df_raw = pd.concat(scores_df_raw)

    # --- All initial setup, timeline generation, and realized weather processing ---
    # --- This part is correct and remains unchanged. ---
    sbs_df = sbs_df_raw.copy()
    sbs_df['execount'] = sbs_df['execount'].astype(int)
    sbs_df = sbs_df[sbs_df['execount'] > 0]
    sbs_df['estimatedTime'] = sbs_df['estimatedTime'].astype(float)
    merged_df = pd.merge(sbs_df, projects_df_raw, on="OBSPROJECT_UID", how="left")
    merged_df['PRJ_SCIENTIFIC_RANK'] = merged_df['PRJ_SCIENTIFIC_RANK'].fillna(1000.0)
    merged_df['EXEC'] = merged_df['EXEC'].fillna('NA')
    sb_weight_map, project_weight_map = _load_required_weight_maps(
        sbs_df_raw=sbs_df_raw,
        projects_df_raw=projects_df_raw,
        weight_data_dir=weight_data_dir,
        sb_weights_path=sb_weights_path,
        project_weights_path=project_weights_path,
    )
    ts_col_name = ('timestamp',) if isinstance(availability_df_raw.columns, pd.MultiIndex) else 'timestamp'
    if availability_df_raw.empty or availability_df_raw[ts_col_name].isna().all():
        return [], [], 0, {}, {}, {}
    availability_df_raw[ts_col_name] = pd.to_datetime(availability_df_raw[ts_col_name], utc=True)
    min_schedule_time = availability_df_raw[ts_col_name].min()
    max_schedule_time = min_schedule_time + pd.Timedelta(minutes=(max_time_steps - 1) * TIME_INTERVAL_MINUTES)

    all_time_slots_timestamps = pd.to_datetime(pd.date_range(
        start=min_schedule_time, end=max_schedule_time, freq=f"{TIME_INTERVAL_MINUTES}min"
    ))
    timestamp_to_idx = {ts: i for i, ts in enumerate(all_time_slots_timestamps)}
    idx_to_timestamp = {i: ts for ts, i in timestamp_to_idx.items()}
    time_steps = len(all_time_slots_timestamps)

    print("Processing shift and downtime data to create accurate weather timeline...", flush=True)

    # A. Prepare downtime/shift data
    shifts_df['START_TIME'] = pd.to_datetime(shifts_df['START_TIME'], utc=True)
    shifts_df['END_TIME'] = pd.to_datetime(shifts_df['END_TIME'], utc=True)
    downtimes_df['START_TIME'] = pd.to_datetime(downtimes_df['START_TIME'], utc=True)
    downtimes_df['END_TIME'] = pd.to_datetime(downtimes_df['END_TIME'], utc=True)

    # B. Identify intervals for each downtime type
    engineering_intervals = shifts_df[shifts_df['SHIFT_ACTIVITY'].isin(['Engineering', 'EOC'])]
    weather_downtime_intervals = downtimes_df[downtimes_df['DOWNTIME_TYPE'] == 'Weather']
    scheduling_downtime_intervals = downtimes_df[downtimes_df['DOWNTIME_TYPE'] == 'Scheduling']

    # C. Map datetime intervals to simulation time indices
    def get_indices_in_intervals(intervals_df, idx_map):
        indices = set()
        for _, row in intervals_df.iterrows():
            # Find all timestamps in our simulation timeline that fall within this interval
            mask = (all_time_slots_timestamps >= row['START_TIME']) & (all_time_slots_timestamps < row['END_TIME'])
            matching_indices = np.where(mask)[0]
            indices.update(matching_indices)
        return indices

    anticipated_downtime_indices = get_indices_in_intervals(engineering_intervals, idx_to_timestamp)
    weather_downtime_indices = get_indices_in_intervals(weather_downtime_intervals, idx_to_timestamp)
    scheduling_downtime_indices = get_indices_in_intervals(scheduling_downtime_intervals, idx_to_timestamp)

    print(f"Found {len(anticipated_downtime_indices)} anticipated downtime slots (Engineering/EOC).", flush=True)
    print(f"Found {len(weather_downtime_indices)} weather downtime slots.", flush=True)
    print(f"Found {len(scheduling_downtime_indices)} scheduling downtime slots.", flush=True)

    # D. Load pre-processed realized weather (produced by preprocess_weather.py)
    if cached_realized_weather is None:
        raise ValueError(
            "realized_weather must be provided via cached_realized_weather. "
            "Run preprocess_weather.py first."
        )
    realized_weather = cached_realized_weather.copy()

    sb_uid_col_name = ('sbuid',) if isinstance(availability_df_raw.columns, pd.MultiIndex) else 'sbuid'
    availability_map, pwv_thresholds, rms_thresholds = {}, {}, {}
    for _, row in availability_df_raw.iterrows():
        sb_uid, timestamp_val = row[sb_uid_col_name], row[ts_col_name]
        ts_idx = timestamp_to_idx.get(timestamp_val)
        if ts_idx is not None:
            if sb_uid not in rms_thresholds: rms_thresholds[sb_uid] = row['rms_thresh']
            if sb_uid not in pwv_thresholds: pwv_thresholds[sb_uid] = {}
            pwv_thresholds[sb_uid][ts_idx] = row['pwv_thresh']
            if sb_uid not in availability_map: availability_map[sb_uid] = []
            availability_map[sb_uid].append(ts_idx)
    for sb_uid_key in availability_map:
        availability_map[sb_uid_key] = sorted(list(set(availability_map[sb_uid_key])))

    # Build forecast availability lookups (from planned-config DSA runs)
    forecast_avail_map, forecast_pwv_thresholds, forecast_rms_thresholds = None, None, None
    forecast_issue_times_by_sb = None
    forecast_avail_map_by_issue = None
    forecast_pwv_thresholds_by_issue = None
    forecast_rms_thresholds_by_issue = None
    if availability_df_forecast is not None:
        if isinstance(availability_df_forecast, list):
            availability_df_forecast = pd.concat(availability_df_forecast)
        fc_ts_col = ('timestamp',) if isinstance(availability_df_forecast.columns, pd.MultiIndex) else 'timestamp'
        fc_sb_col = ('sbuid',) if isinstance(availability_df_forecast.columns, pd.MultiIndex) else 'sbuid'
        if not availability_df_forecast.empty:
            availability_df_forecast = availability_df_forecast.copy()
            availability_df_forecast[fc_ts_col] = pd.to_datetime(availability_df_forecast[fc_ts_col], utc=True)
            fc_issue_col = (
                ('forecast_issue_time',)
                if isinstance(availability_df_forecast.columns, pd.MultiIndex)
                else 'forecast_issue_time'
            )
            has_issue_col = fc_issue_col in availability_df_forecast.columns
            if has_issue_col:
                availability_df_forecast[fc_issue_col] = pd.to_datetime(
                    availability_df_forecast[fc_issue_col], utc=True, errors='coerce'
                )
                availability_df_forecast = availability_df_forecast.sort_values([fc_issue_col, fc_ts_col])
                forecast_issue_times_by_sb = {}
                forecast_avail_map_by_issue = {}
                forecast_pwv_thresholds_by_issue = {}
                forecast_rms_thresholds_by_issue = {}
                skipped_issue_rows = 0
                for _, row in availability_df_forecast.iterrows():
                    sb_uid = str(row[fc_sb_col]).strip()
                    timestamp_val = row[fc_ts_col]
                    issue_time_val = row[fc_issue_col]
                    ts_idx = timestamp_to_idx.get(timestamp_val)
                    issue_idx = timestamp_to_idx.get(issue_time_val)
                    if ts_idx is None or issue_idx is None or pd.isna(issue_time_val):
                        skipped_issue_rows += 1
                        continue

                    forecast_issue_times_by_sb.setdefault(sb_uid, set()).add(issue_idx)
                    forecast_avail_map_by_issue.setdefault(sb_uid, {}).setdefault(issue_idx, []).append(ts_idx)
                    forecast_pwv_thresholds_by_issue.setdefault(sb_uid, {}).setdefault(issue_idx, {})[ts_idx] = row['pwv_thresh']
                    forecast_rms_thresholds_by_issue.setdefault(sb_uid, {})[issue_idx] = row['rms_thresh']

                for sb_uid_key, issue_times in forecast_issue_times_by_sb.items():
                    forecast_issue_times_by_sb[sb_uid_key] = sorted(issue_times)
                for sb_uid_key, avail_by_issue in forecast_avail_map_by_issue.items():
                    for issue_idx, availability in avail_by_issue.items():
                        avail_by_issue[issue_idx] = sorted(set(availability))

                print(
                    f"Loaded rolling forecast availability for {len(forecast_avail_map_by_issue)} SBs "
                    f"from issuance-based forecast files. Skipped rows without an in-range issue/timestamp: "
                    f"{skipped_issue_rows}",
                    flush=True,
                )
            else:
                forecast_avail_map, forecast_pwv_thresholds, forecast_rms_thresholds = {}, {}, {}
                for _, row in availability_df_forecast.iterrows():
                    sb_uid, timestamp_val = row[fc_sb_col], row[fc_ts_col]
                    ts_idx = timestamp_to_idx.get(timestamp_val)
                    if ts_idx is not None:
                        if sb_uid not in forecast_rms_thresholds: forecast_rms_thresholds[sb_uid] = row['rms_thresh']
                        if sb_uid not in forecast_pwv_thresholds: forecast_pwv_thresholds[sb_uid] = {}
                        forecast_pwv_thresholds[sb_uid][ts_idx] = row['pwv_thresh']
                        if sb_uid not in forecast_avail_map: forecast_avail_map[sb_uid] = []
                        forecast_avail_map[sb_uid].append(ts_idx)
                for sb_uid_key in forecast_avail_map:
                    forecast_avail_map[sb_uid_key] = sorted(list(set(forecast_avail_map[sb_uid_key])))
                print(f"Loaded forecast availability for {len(forecast_avail_map)} SBs from dsa_sim_for_forecast.")

    shared_projects = {}
    for _, row in fractional_execs.iterrows():
        shared_projects[row['CODE']] = {executive: row[executive] for executive in ['NA', 'EU', 'EA', 'CL']}

    # =========================================================================
    # --- Map out the scores that will be used for DSA scorer ---
    # 'timestamp', 'sbuid', 'condition_score', 'science_rank_score', 'cycle_grade_score',
    #                        'array_score_no', 'array_score_yes', 'base_ha_score'
    # =========================================================================
    condition_scores, science_rank_scores, cycle_grade_scores, array_scores_no, array_scores_yes, base_ha_scores = {}, {}, {}, {}, {}, {}
    score_attachment_stats = None
    if scores_df_raw is not None:
        score_ts_col = ('timestamp',) if isinstance(scores_df_raw.columns, pd.MultiIndex) else 'timestamp'
        score_sb_col = ('sbuid',) if isinstance(scores_df_raw.columns, pd.MultiIndex) else 'sbuid'
        scores_df_raw = scores_df_raw.copy()
        total_score_rows = len(scores_df_raw)
        if not scores_df_raw.empty:
            scores_df_raw[score_ts_col] = pd.to_datetime(scores_df_raw[score_ts_col], utc=True, errors='coerce')
            scores_df_raw[score_sb_col] = scores_df_raw[score_sb_col].astype(str)

            matched_rows = 0
            invalid_timestamp_rows = 0
            malformed_sbuid_rows = 0
            timestamp_miss_rows = 0
            sample_successes = []
            sample_timestamp_misses = []
            sample_bad_sbuid = []

            print("[DSA SCORE LOAD] Starting score attachment into jobs.", flush=True)
            print(f"[DSA SCORE LOAD] Score rows read: {total_score_rows}", flush=True)
            print(
                f"[DSA SCORE LOAD] Keying on columns: timestamp={score_ts_col}, sbuid={score_sb_col}",
                flush=True,
            )

            first_valid_ts = scores_df_raw[score_ts_col].dropna().iloc[0] if scores_df_raw[score_ts_col].notna().any() else None
            first_valid_sb = scores_df_raw[score_sb_col].iloc[0] if len(scores_df_raw) > 0 else None
            print(
                f"[DSA SCORE LOAD] Normalized key samples: "
                f"timestamp={first_valid_ts!r} ({type(first_valid_ts).__name__}), "
                f"sbuid={first_valid_sb!r} ({type(first_valid_sb).__name__})",
                flush=True,
            )

            for _, row in scores_df_raw.iterrows():
                sb_uid = str(row[score_sb_col]).strip()
                timestamp_val = row[score_ts_col]

                if not sb_uid or sb_uid.lower() == 'nan':
                    malformed_sbuid_rows += 1
                    if len(sample_bad_sbuid) < 5:
                        sample_bad_sbuid.append({
                            "raw_sbuid": row[score_sb_col],
                            "timestamp": timestamp_val,
                        })
                    continue

                if pd.isna(timestamp_val):
                    invalid_timestamp_rows += 1
                    if len(sample_timestamp_misses) < 5:
                        sample_timestamp_misses.append({
                            "sbuid": sb_uid,
                            "timestamp": timestamp_val,
                            "reason": "timestamp_parse_failed",
                        })
                    continue

                ts_idx = timestamp_to_idx.get(timestamp_val)
                if ts_idx is None:
                    timestamp_miss_rows += 1
                    if len(sample_timestamp_misses) < 5:
                        sample_timestamp_misses.append({
                            "sbuid": sb_uid,
                            "timestamp": timestamp_val.isoformat(),
                            "reason": "timestamp_not_in_timeline",
                        })
                    continue

                matched_rows += 1
                if len(sample_successes) < 5:
                    sample_successes.append({
                        "sbuid": sb_uid,
                        "timestamp": timestamp_val.isoformat(),
                        "ts_idx": int(ts_idx),
                    })

                if sb_uid not in science_rank_scores: science_rank_scores[sb_uid] = row['science_rank_score']
                if sb_uid not in cycle_grade_scores: cycle_grade_scores[sb_uid] = row['cycle_grade_score']
                if sb_uid not in array_scores_no: array_scores_no[sb_uid] = row['array_score_no']
                if sb_uid not in array_scores_yes: array_scores_yes[sb_uid] = row['array_score_yes']

                if sb_uid not in condition_scores: condition_scores[sb_uid] = {}
                if sb_uid not in base_ha_scores: base_ha_scores[sb_uid] = {}

                condition_scores[sb_uid][ts_idx] = row['condition_score']
                base_ha_scores[sb_uid][ts_idx] = row['base_ha_score']

            print(
                f"[DSA SCORE LOAD] Rows matched to timeline: {matched_rows}/{total_score_rows}",
                flush=True,
            )
            print(
                f"[DSA SCORE LOAD] Unique SBs with static scores: {len(science_rank_scores)} | "
                f"condition maps: {len(condition_scores)} | HA maps: {len(base_ha_scores)}",
                flush=True,
            )
            print(
                f"[DSA SCORE LOAD] Skipped rows: invalid_timestamp={invalid_timestamp_rows}, "
                f"malformed_sbuid={malformed_sbuid_rows}, timestamp_not_in_timeline={timestamp_miss_rows}",
                flush=True,
            )
            if sample_successes:
                print(f"[DSA SCORE LOAD] Sample successful attachments: {sample_successes}", flush=True)
            if sample_timestamp_misses:
                print(f"[DSA SCORE LOAD] Sample skipped rows (timestamp issues): {sample_timestamp_misses}", flush=True)
            if sample_bad_sbuid:
                print(f"[DSA SCORE LOAD] Sample skipped rows (sbuid issues): {sample_bad_sbuid}", flush=True)

            score_attachment_stats = {
                "total_score_rows": total_score_rows,
                "matched_rows": matched_rows,
                "invalid_timestamp_rows": invalid_timestamp_rows,
                "malformed_sbuid_rows": malformed_sbuid_rows,
                "timestamp_miss_rows": timestamp_miss_rows,
            }

    # =========================================================================
    # --- Load pre-processed weather forecasts (produced by preprocess_forecasts.py)
    # =========================================================================
    # May be None for algorithms (e.g. dsa_eb, prophet) that don't consume forecasts.
    # Downstream dispatch blocks raise their own errors if a forecast-dependent
    # algorithm is selected without forecasts loaded.
    weather_forecasts = cached_weather_forecasts
    if weather_forecasts is None:
        print(
            "No cached weather forecasts provided; weather_forecasts=None. "
            "Forecast-dependent algorithms will fail later if selected.",
            flush=True,
        )
    else:
        print(
            "Loaded cached weather forecasts. RMS forecasts may include both legacy "
            "global arrays and issuance/lookahead arrays for downstream planners.",
            flush=True,
        )

    # --- All Job and Project Processing remains unchanged. ---
    jobs = []
    project_job_map = {}
    for _, sb_row in merged_df.iterrows():
        sb_uid = str(sb_row['SB_UID'])
        project_id = str(sb_row['CODE_x']) # It's ok, I double-checked they're all the same anyway.
        gous_id = sb_row['GOUS_ID']
        time_per_exec_hours = sb_row['estimatedTime'] / sb_row['execount']
        job_length_intervals = math.ceil(time_per_exec_hours * (60 / TIME_INTERVAL_MINUTES))
        if job_length_intervals <= 0: continue
        grade = sb_row['PRJ_LETTER_GRADE']
        if sb_uid not in sb_weight_map:
            raise ValueError(f"Missing SB weight for SB_UID '{sb_uid}' in sb_weights.csv")
        if project_id not in project_weight_map:
            raise ValueError(f"Missing project weight for project '{project_id}' in project_weights.csv")
        job_weight = sb_weight_map[sb_uid]
        project_weight = project_weight_map[project_id]
        executive = sb_row['EXEC']
        if executive not in EXECUTIVE_QUOTAS: executive = 'OTHER'
        if project_id in shared_projects:
            executive = shared_projects[project_id]

        rem_execs = int(sb_row['execount'])
        if sb_uid in remaining_execution_counts:
            rem_execs = min(rem_execs, remaining_execution_counts[sb_uid])
        else:
            print(f"Warning: SB {sb_uid} not found in remaining execution counts.")
    

        avail = availability_map.get(sb_uid, [])
        pwv_t = pwv_thresholds.get(sb_uid, {})
        rms_t = rms_thresholds.get(sb_uid, 1000)

        forecast_issue_times = []
        forecast_avail_by_issue = {}
        forecast_pwv_by_issue = {}
        forecast_rms_by_issue = {}
        if forecast_avail_map_by_issue is not None:
            forecast_issue_times = forecast_issue_times_by_sb.get(sb_uid, [])
            forecast_avail_by_issue = forecast_avail_map_by_issue.get(sb_uid, {})
            forecast_pwv_by_issue = forecast_pwv_thresholds_by_issue.get(sb_uid, {})
            forecast_rms_by_issue = forecast_rms_thresholds_by_issue.get(sb_uid, {})
            fc_avail = []
            fc_pwv_t = {}
            fc_rms_t = rms_t
        elif forecast_avail_map is not None:
            fc_avail = forecast_avail_map.get(sb_uid, [])
            fc_pwv_t = forecast_pwv_thresholds.get(sb_uid, {})
            fc_rms_t = forecast_rms_thresholds.get(sb_uid, 1000)
        else:
            fc_avail = avail
            fc_pwv_t = pwv_t
            fc_rms_t = rms_t

        job = {
            "job_id": sb_uid, "project_id": project_id, "gous_id": gous_id,
            "length": job_length_intervals,
            "available": avail,
            "grade": grade,
            "condition_scores": condition_scores.get(sb_uid, {}),
            "science_rank_score": science_rank_scores.get(sb_uid, 0.0),
            "cycle_grade_score": cycle_grade_scores.get(sb_uid, 0.0),
            "array_score_no": array_scores_no.get(sb_uid, 0.0),
            "array_score_yes": array_scores_yes.get(sb_uid, 0.0),
            "base_ha_scores": base_ha_scores.get(sb_uid, {}),
            "rms_threshold": rms_t,
            "pwv_thresholds": pwv_t,
            "forecast_available": fc_avail,
            "forecast_pwv_thresholds": fc_pwv_t,
            "forecast_rms_threshold": fc_rms_t,
            "forecast_issue_times": forecast_issue_times,
            "forecast_available_by_issue": forecast_avail_by_issue,
            "forecast_pwv_thresholds_by_issue": forecast_pwv_by_issue,
            "forecast_rms_thresholds_by_issue": forecast_rms_by_issue,
            "executive": executive, "weight": job_weight,
            "remaining_execs": rem_execs, "type": "science",
            "total_execs": rem_execs
        }
        jobs.append(job)
        if project_id not in project_job_map:
            project_job_map[project_id] = {"executive": executive,
                                           "weight": project_weight,
                                           "grade": grade,
                                           "job_ids": []}
        project_job_map[project_id]["job_ids"].append(sb_uid)

    if score_attachment_stats is not None:
        jobs_with_condition_scores = sum(1 for job in jobs if job["condition_scores"])
        jobs_with_base_ha_scores = sum(1 for job in jobs if job["base_ha_scores"])
        jobs_with_static_scores = sum(
            1 for job in jobs
            if (
                job["science_rank_score"] != 0.0 or
                job["cycle_grade_score"] != 0.0 or
                job["array_score_no"] != 0.0 or
                job["array_score_yes"] != 0.0
            )
        )
        job_id_set = {job["job_id"] for job in jobs}
        score_sbuid_set = set(science_rank_scores) | set(cycle_grade_scores) | set(array_scores_no) | set(array_scores_yes)
        unmatched_score_sbuids = sorted(score_sbuid_set - job_id_set)[:10]
        print(
            f"[DSA SCORE LOAD] Jobs with non-empty condition scores: {jobs_with_condition_scores}/{len(jobs)}",
            flush=True,
        )
        print(
            f"[DSA SCORE LOAD] Jobs with non-empty base HA scores: {jobs_with_base_ha_scores}/{len(jobs)}",
            flush=True,
        )
        print(
            f"[DSA SCORE LOAD] Jobs with non-default static DSA scores: {jobs_with_static_scores}/{len(jobs)}",
            flush=True,
        )
        if unmatched_score_sbuids:
            print(
                f"[DSA SCORE LOAD] Sample score SBUIDs with no matching job_id: {unmatched_score_sbuids}",
                flush=True,
            )

    all_jobs = list(jobs)

    final_projects_list = []
    for pid, data in project_job_map.items():
        if data["job_ids"]:
            final_projects_list.append({"project_id": pid, "executive": data["executive"],
                                        "weight": data["weight"], "grade": data["grade"],
                                        "job_ids": data["job_ids"]})
    final_projects_list.sort(key=lambda p: p['project_id'])
    all_jobs.sort(key=lambda j: j['job_id'])

    return all_jobs, final_projects_list, time_steps, idx_to_timestamp, realized_weather, weather_forecasts, scores_df_raw
