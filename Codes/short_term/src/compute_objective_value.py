"""
Function to compute the overall weighted objective value for a schedule
based on the marginal gain greedy selector's objective function.

This function takes the evaluation dictionary output by full_year.py and
computes the same objective value that the marginal_gain_greedy_selector_factory
would have optimized for.
"""

from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import numpy as np
from copy import deepcopy
import os
import pickle
import argparse
import pandas as pd
from tqdm import tqdm

from evaluation import _is_execution_valid
from planning_implementations import _calculate_eb_l1_penalty


def compute_weighted_objective_value(
    evaluation_dict: Dict[str, Any],
    weights: Dict[str, float],
    executive_quotas_frac: Dict[str, Tuple[float, float]],
    total_observable_time: int,
    priority_job_ids: set = None,
    schedule_log: List[str] = None
) -> float:
    """
    Computes the overall weighted objective value for a schedule using pre-computed statistics.
    
    This function directly uses the completion percentages, utilization ratio, and executive balance
    statistics from the evaluation dictionary without calculating normalization factors.
    
    Args:
        evaluation_dict: Dictionary output by full_year.py's run_eval_real function.
                        Should contain:
                        - 'completion_pct_sb_A', 'completion_pct_sb_B', 'completion_pct_sb_C' (percentages)
                        - 'completion_pct_proj_A', 'completion_pct_proj_B', 'completion_pct_proj_C' (percentages)
                        - 'usage_ratio' (as percentage)
                        - 'exec_time_fractions_real' (dictionary of fractions)
        weights: Dictionary of weights for different objective components:
                 - 'adherence': weight for priority job adherence
                 - 'sb_A', 'sb_B', 'sb_C': weights for SB completion by grade
                 - 'proj_A', 'proj_B', 'proj_C': weights for project completion by grade
                 - 'utilization': weight for telescope utilization
                 - 'eb_penalty': weight for executive balance penalty
        executive_quotas_frac: Dictionary mapping executive name to (min_frac, max_frac) tuple
        total_observable_time: Total number of observable time steps (non-NaN weather)
        priority_job_ids: Set of job IDs that are priority jobs (for adherence calculation).
                         If None and schedule_log is provided, will try to count from schedule.
        schedule_log: Optional schedule log for adherence calculation if priority_job_ids is None
    
    Returns:
        Total weighted objective value (float)
    """
    # Initialize objective value
    objective_value = 0.0
    
    # # 1. Adherence component (count of priority job executions)
    # if priority_job_ids is not None and schedule_log is not None:
    #     # Count how many times priority jobs were executed
    #     priority_job_executions = 0
    #     for entry in schedule_log:
    #         try:
    #             job_id, _ = entry.split("@")
    #             if job_id in priority_job_ids:
    #                 priority_job_executions += 1
    #         except (ValueError, IndexError):
    #             continue
    #     objective_value += weights.get('adherence', 0) * priority_job_executions
    
    # 2. SB completion by grade - use percentages directly
    completion_pct_sb_A = evaluation_dict.get('completion_pct_sb_A', 0) / 100.0
    completion_pct_sb_B = evaluation_dict.get('completion_pct_sb_B', 0) / 100.0
    completion_pct_sb_C = evaluation_dict.get('completion_pct_sb_C', 0) / 100.0
    
    objective_value += weights.get('sb_A', 0) * completion_pct_sb_A
    objective_value += weights.get('sb_B', 0) * completion_pct_sb_B
    objective_value += weights.get('sb_C', 0) * completion_pct_sb_C
    
    # 3. Project completion by grade - use percentages directly
    completion_pct_proj_A = evaluation_dict.get('completion_pct_proj_A', 0) / 100.0
    completion_pct_proj_B = evaluation_dict.get('completion_pct_proj_B', 0) / 100.0
    completion_pct_proj_C = evaluation_dict.get('completion_pct_proj_C', 0) / 100.0
    
    objective_value += weights.get('proj_A', 0) * completion_pct_proj_A
    objective_value += weights.get('proj_B', 0) * completion_pct_proj_B
    objective_value += weights.get('proj_C', 0) * completion_pct_proj_C
    
    # 4. Utilization component
    usage_ratio_pct = evaluation_dict.get('usage_ratio', 0)
    utilization_ratio = usage_ratio_pct / 100.0  # Convert percentage to ratio
    objective_value += weights.get('utilization', 0) * utilization_ratio
    
    # 5. Executive balance penalty component
    # Extract executive time fractions and convert to absolute times
    exec_time_fractions = evaluation_dict.get('exec_time_fractions_real', {})
    
    # Calculate total time used from usage_ratio
    # usage_ratio = (total_time_used / total_observable_time) * 100
    # So: total_time_used = (usage_ratio / 100) * total_observable_time
    total_time_used = (usage_ratio_pct / 100.0) * total_observable_time
    
    # Convert fractions to absolute times
    exec_time_used = {}
    for exec_name, fraction in exec_time_fractions.items():
        # fraction = exec_time / total_time_used
        # So: exec_time = fraction * total_time_used
        exec_time_used[exec_name] = fraction * total_time_used
    
    # Calculate EB penalty
    eb_penalty = _calculate_eb_l1_penalty(exec_time_used, executive_quotas_frac, total_observable_time)
    objective_value += weights.get('eb_penalty', 0) * (-eb_penalty)  # Negative because penalty should reduce value
    
    return objective_value


def load_total_observable_time(
    data_dir: str,
    start_date_str: str,
    end_date_str: str,
    seed: int = 1
) -> int:
    """
    Load only the total observable time (needed for EB penalty calculation).
    
    Returns:
        Total number of observable time steps
    """
    # Import here to avoid circular import
    from job_and_weather_simulation import load_and_prepare_data
    from full_year import get_weather_cache_path, load_cached_weather
    
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    simulation_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    num_days = len(simulation_dates)
    max_time_steps = num_days * 48
    
    print(f"Loading total observable time for period: {start_date.date()} to {end_date.date()} ({num_days} days)")
    
    # Load minimal data needed for normalization factors
    daily_dfs = []
    for current_date in tqdm(simulation_dates, desc="Loading daily data"):
        file_month, file_day, file_year = current_date.month, current_date.day, current_date.year
        file_path = os.path.join(data_dir, f"dsa_sim_{file_month}_{file_day}_{file_year}_df.csv")
        if os.path.exists(file_path):
            daily_dfs.append(pd.read_csv(file_path))
        else:
            print(f"WARNING: Availability file not found for {current_date.strftime('%Y-%m-%d')}. Using dummy.")
    
    daily_score_dfs = None
    fractional_exec_df = pd.read_csv(os.path.join(data_dir, "proposals_time_share_mod.csv"))
    
    # Load remaining execution counts
    already_executed = pd.read_csv(os.path.join(data_dir, "cycle_10_sb_active_time_to_complete_at_c10_start.csv"))
    remaining_execution_counts = {}
    for _, row in already_executed.iterrows():
        sb_uid = row['SB_UID']
        remaining_execution_counts[sb_uid] = int(np.round(row['execution_count_start_c10']))
    
    shifts_df = pd.read_csv(os.path.join(data_dir, "shifts_dimensions.csv"))
    downtimes_df = pd.read_csv(os.path.join(data_dir, "downtimes_dimensions.csv"))
    
    sbs_df_raw = pd.read_csv(os.path.join(data_dir, "schedblocks_c10.csv"))
    projects_df_raw = pd.read_csv(os.path.join(data_dir, "projects_c10.csv"))
    gfs_pwv_forecasts = pd.read_csv(os.path.join(data_dir, "gfs_pwv_combined_data.csv"))
    
    rms_stats_by_day = pd.read_csv(os.path.join(data_dir, 'rms_stats_by_day.csv'),
                                   index_col=["month", "day", "time_interval_utc"])
    rms_stats_by_day.index = pd.MultiIndex.from_arrays(
        [rms_stats_by_day.index.get_level_values('month'), 
         rms_stats_by_day.index.get_level_values('day'),
         pd.to_datetime(rms_stats_by_day.index.get_level_values('time_interval_utc'), format='%H:%M:%S').time],
        names=['month', 'day', 'time_interval_utc'])
    
    pwv_stats_by_day = pd.read_csv(os.path.join(data_dir, 'pwv_stats_by_day.csv'),
                                   index_col=["month", "day", "time_interval_utc"])
    pwv_stats_by_day.index = pd.MultiIndex.from_arrays(
        [pwv_stats_by_day.index.get_level_values('month'),
         pwv_stats_by_day.index.get_level_values('day'),
         pd.to_datetime(pwv_stats_by_day.index.get_level_values('time_interval_utc'), format='%H:%M:%S').time],
        names=['month', 'day', 'time_interval_utc'])
    
    # Check for cached weather data
    cache_path = get_weather_cache_path(data_dir, start_date, end_date)
    cache_result = load_cached_weather(cache_path)
    
    if cache_result is not None:
        cached_realized_weather, cached_weather_forecasts, cached_precalculated_pwv_forecasts = cache_result
    else:
        cached_realized_weather, cached_weather_forecasts, cached_precalculated_pwv_forecasts = None, None, None
    
    # Load and prepare data (we need this to get jobs/projects for counting)
    if cached_realized_weather is not None and cached_weather_forecasts is not None:
        print("Using fully cached weather data!")
        jobs, projects, time_steps, idx_to_timestamp, realized_weather, weather_forecasts, dsa_scores = (
            load_and_prepare_data(
                max_time_steps, sbs_df_raw, projects_df_raw, daily_dfs, daily_score_dfs, gfs_pwv_forecasts,
                rms_stats_by_day, pwv_stats_by_day, fractional_exec_df, remaining_execution_counts,
                shifts_df=shifts_df, downtimes_df=downtimes_df, fraction_jobs_to_drop=0,
                add_fillers=False, seed=seed,
                cached_realized_weather=cached_realized_weather,
                cached_weather_forecasts=cached_weather_forecasts,
                cached_precalculated_pwv_forecasts=cached_precalculated_pwv_forecasts,
                cache_path=cache_path,
                weight_data_dir=data_dir,
            ))
    else:
        jobs, projects, time_steps, idx_to_timestamp, realized_weather, weather_forecasts, dsa_scores = (
            load_and_prepare_data(
                max_time_steps, sbs_df_raw, projects_df_raw, daily_dfs, daily_score_dfs, gfs_pwv_forecasts,
                rms_stats_by_day, pwv_stats_by_day, fractional_exec_df, remaining_execution_counts,
                shifts_df=shifts_df, downtimes_df=downtimes_df, fraction_jobs_to_drop=0,
                add_fillers=False, seed=seed,
                cached_realized_weather=cached_realized_weather,
                cached_precalculated_pwv_forecasts=cached_precalculated_pwv_forecasts,
                cache_path=cache_path,
                weight_data_dir=data_dir,
            ))
    
    # Calculate total observable time
    total_observable_time = sum(
        1 for t in range(time_steps)
        if not (np.isnan(realized_weather.get(t, (np.nan, np.nan))[0]) or
                np.isnan(realized_weather.get(t, (np.nan, np.nan))[1]))
    )
    
    print(f"Total observable time: {total_observable_time}")
    
    return total_observable_time


def main():
    parser = argparse.ArgumentParser(
        description="Compute weighted objective value for a schedule from full_year.py output"
    )
    parser.add_argument(
        "--results_pickle",
        type=str,
        required=True,
        help="Path to the pickle file containing evaluation results from full_year.py"
    )
    parser.add_argument(
        "--algorithm_name",
        type=str,
        required=True,
        help="Name of the algorithm in the results dictionary (e.g., 'strategic_greedy', 'greedy', etc.)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the data files (same as used in full_year.py)"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2023-09-30",
        help="Start date in YYYY-MM-DD format (must match the date used in full_year.py)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2024-10-01",
        help="End date in YYYY-MM-DD format (must match the date used in full_year.py)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (must match the seed used in full_year.py)"
    )
    parser.add_argument(
        "--weights_json",
        type=str,
        default=None,
        help="Path to JSON file containing weights dictionary. If not provided, will try to infer from pickle file."
    )
    parser.add_argument(
        "--priority_job_ids",
        type=str,
        default=None,
        help="Comma-separated list of priority job IDs (optional, for adherence calculation)"
    )
    
    args = parser.parse_args()
    
    # Load evaluation results
    print(f"Loading evaluation results from: {args.results_pickle}")
    with open(args.results_pickle, 'rb') as f:
        all_results = pickle.load(f)
    
    if args.algorithm_name not in all_results:
        print(f"Error: Algorithm '{args.algorithm_name}' not found in results.")
        print(f"Available algorithms: {list(all_results.keys())}")
        return
    
    evaluation_dict = all_results[args.algorithm_name]
    print(f"Found results for algorithm: {args.algorithm_name}")
    print(f"Schedule log contains {len(evaluation_dict.get('schedule_log', []))} entries")
    
    # Load weights
    import json
    if args.weights_json:
        print(f"Loading weights from: {args.weights_json}")
        with open(args.weights_json, 'r') as f:
            weights = json.load(f)
    else:
        # Try to get weights from the results if available
        # Note: weights might not be saved in the pickle file
        print("Warning: No weights file provided. Using default weights.")
        print("You should provide --weights_json with the same weights used in full_year.py")
        weights = {
            "sb_A": 0.9, "sb_B": 0.07, "sb_C": 0.03,
            "proj_A": 0.9, "proj_B": 0.07, "proj_C": 0.03,
            "adherence": 0.0,
            "utilization": 0.0,
            "eb_penalty": 0.0
        }
    
    print(f"Weights: {weights}")
    
    # Load only total_observable_time for EB penalty calculation
    # (We can get this from cached weather or calculate it, but for now we'll still need it)
    # Actually, we can extract it from the usage_ratio if we know total_time_used
    # But for EB penalty we need it. Let's keep a minimal load or make it optional.
    # For now, we'll still load it but the function doesn't need the normalization factors
    
    # Load total_observable_time (minimal - just need weather count)
    total_observable_time = load_total_observable_time(
        data_dir=args.data_dir,
        start_date_str=args.start_date,
        end_date_str=args.end_date,
        seed=args.seed
    )
    
    # Executive quotas (standard values)
    EXECUTIVE_QUOTAS = {
        'CL': (0.1, 1.0),
        'EA': (0.225, 1.0),
        'EU': (0.3375, 1.0),
        'NA': (0.3375, 1.0),
        'OTHER': (0.0, 1.0)
    }
    
    # Parse priority job IDs if provided
    priority_job_ids = None
    if args.priority_job_ids:
        priority_job_ids = set(args.priority_job_ids.split(','))
        print(f"Priority job IDs: {priority_job_ids}")
    
    # Get schedule log for adherence calculation if needed
    schedule_log = evaluation_dict.get('schedule_log', [])
    
    # Compute objective value
    print("\n" + "="*80)
    print("Computing weighted objective value from pre-computed statistics...")
    print("="*80)
    
    objective_value = compute_weighted_objective_value(
        evaluation_dict=evaluation_dict,
        weights=weights,
        executive_quotas_frac=EXECUTIVE_QUOTAS,
        total_observable_time=total_observable_time,
        priority_job_ids=priority_job_ids,
        schedule_log=schedule_log
    )
    
    print(f"\n{'='*80}")
    print(f"RESULT: Weighted Objective Value = {objective_value:.6f}")
    print(f"{'='*80}\n")
    
    # Also print the planner-reported value for comparison
    planner_value = evaluation_dict.get('planner_value', None)
    if planner_value is not None:
        print(f"Planner-reported value (from evaluation): {planner_value:.6f}")
        print(f"Difference: {objective_value - planner_value:.6f}\n")


if __name__ == "__main__":
    main()

