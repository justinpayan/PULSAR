import argparse
from copy import deepcopy
from collections import Counter
import os
import pickle
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation import run_eval_real
from fixed_params import EXECUTIVE_QUOTAS
from full_year import load_config_calendar, log_dsa_eb_dataset_summary, log_dsa_eb_run_configuration
from job_and_weather_simulation import load_and_prepare_data
from planning_implementations import dsa_eb_selector_factory, planning_loop_eb_greedy

current_dir = os.path.dirname(os.path.abspath(__file__))
long_term_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "long_term"))
if long_term_dir not in sys.path:
    sys.path.insert(0, long_term_dir)

from long_term_optim import solve_long_term_schedule


DEFAULT_SB_GRADE_SPLITS = {"A": 0.9, "B": 0.07, "C": 0.03}
DEFAULT_PROJ_GRADE_SPLITS = {"A": 0.9, "B": 0.07, "C": 0.03}


def _resolve_cycle_year(timestamp: pd.Timestamp) -> int:
    return timestamp.year if timestamp.month >= 10 else timestamp.year - 1


def _resolve_preprocessed_weather_path(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        preprocessed_weather_path: Optional[str],
        preprocessed_root: Optional[str],
) -> str:
    if preprocessed_weather_path is not None:
        return os.path.abspath(os.path.expanduser(preprocessed_weather_path))

    if preprocessed_root is None:
        raise ValueError(
            "Pass either --preprocessed_weather or --preprocessed_root so realized weather can be loaded."
        )

    start_cycle_year = _resolve_cycle_year(start_date)
    end_cycle_year = _resolve_cycle_year(end_date)
    if start_cycle_year != end_cycle_year:
        raise ValueError(
            f"Start/end dates span multiple cycle years ({start_cycle_year} vs {end_cycle_year}). "
            "Please run a single cycle at a time."
        )

    return os.path.join(
        os.path.abspath(os.path.expanduser(preprocessed_root)),
        f"year_{start_cycle_year}",
        "realized_weather.pkl",
    )


def _build_fixed_weights(args: argparse.Namespace) -> Dict[str, float]:
    total_weight = args.w_sb + args.w_proj + args.w_util + args.w_ebp
    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            f"The main weights (sb, proj, util, ebp) must sum to 1.0, but they sum to {total_weight}."
        )

    return {
        "sb_A": DEFAULT_SB_GRADE_SPLITS["A"] * args.w_sb,
        "sb_B": DEFAULT_SB_GRADE_SPLITS["B"] * args.w_sb,
        "sb_C": DEFAULT_SB_GRADE_SPLITS["C"] * args.w_sb,
        "proj_A": DEFAULT_PROJ_GRADE_SPLITS["A"] * args.w_proj,
        "proj_B": DEFAULT_PROJ_GRADE_SPLITS["B"] * args.w_proj,
        "proj_C": DEFAULT_PROJ_GRADE_SPLITS["C"] * args.w_proj,
        "adherence": 0.0,
        "utilization": args.w_util,
        "eb_penalty": args.w_ebp,
        "obs_completion": args.w_sb,
        "proj_completion": args.w_proj,
    }


def _apply_cycle_grade_score_override(
        jobs: List[Dict],
        override_enabled: bool,
        grade_scores: Dict[str, float],
) -> None:
    if not override_enabled:
        print("Cycle-grade override disabled; using CSV-derived DSA cycle_grade_score values.", flush=True)
        return

    grade_counts: Counter = Counter()
    updated_counts: Counter = Counter()
    sample_updates = []
    untouched_count = 0

    for job in jobs:
        grade = str(job.get("grade", "")).strip().upper()
        grade_counts[grade] += 1
        if grade in grade_scores:
            old_score = job.get("cycle_grade_score", 0.0)
            new_score = grade_scores[grade]
            job["cycle_grade_score"] = new_score
            updated_counts[grade] += 1
            if len(sample_updates) < 8:
                sample_updates.append(
                    {
                        "job_id": job.get("job_id"),
                        "grade": grade,
                        "old_score": old_score,
                        "new_score": new_score,
                    }
                )
        else:
            untouched_count += 1

    print(
        "Cycle-grade override ACTIVE: "
        f"A->{grade_scores['A']}, B->{grade_scores['B']}, C->{grade_scores['C']}",
        flush=True,
    )
    print(
        "Cycle-grade override counts: "
        f"A={updated_counts.get('A', 0)}, "
        f"B={updated_counts.get('B', 0)}, "
        f"C={updated_counts.get('C', 0)}, "
        f"untouched_other={untouched_count}",
        flush=True,
    )
    print(
        "Job grade distribution before override: "
        f"A={grade_counts.get('A', 0)}, "
        f"B={grade_counts.get('B', 0)}, "
        f"C={grade_counts.get('C', 0)}, "
        f"other={sum(count for grade, count in grade_counts.items() if grade not in grade_scores)}",
        flush=True,
    )
    if sample_updates:
        print(f"Cycle-grade override sample updates: {sample_updates}", flush=True)


def _load_daily_simulation_frames(
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        data_dir: str,
) -> Tuple[List[pd.DataFrame], Optional[List[pd.DataFrame]]]:
    simulation_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    daily_dfs: List[pd.DataFrame] = []
    daily_score_dfs: List[pd.DataFrame] = []

    for current_date in simulation_dates:
        month, day, year = current_date.month, current_date.day, current_date.year
        availability_path = os.path.join(data_dir, "dsa_sim", f"dsa_sim_{month}_{day}_{year}_df.csv")
        if os.path.exists(availability_path):
            daily_dfs.append(pd.read_csv(availability_path))
        else:
            print(
                f"WARNING: Availability file not found for {current_date.strftime('%Y-%m-%d')}: {availability_path}",
                flush=True,
            )

        score_path = os.path.join(data_dir, f"dsa_sim_scores_{month}_{day}_{year}_df.csv")
        if os.path.exists(score_path):
            daily_score_dfs.append(pd.read_csv(score_path))
        else:
            print(
                f"WARNING: DSA scores file not found for {current_date.strftime('%Y-%m-%d')}: {score_path}",
                flush=True,
            )

    if not daily_dfs:
        raise ValueError("No daily dsa_sim availability files were found for the requested date range.")

    return daily_dfs, (daily_score_dfs if daily_score_dfs else None)


def _load_operational_inputs(
        data_dir: str,
        preprocessed_weather_path: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        seed: int,
):
    print("Running forecast-free DSA+EB setup (no forecast pickle required).", flush=True)
    daily_dfs, daily_score_dfs = _load_daily_simulation_frames(
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
    )
    print(
        "DSA_EB raw inputs [dashboard]: "
        f"daily_availability_frames={len(daily_dfs)}, "
        f"daily_score_frames={0 if daily_score_dfs is None else len(daily_score_dfs)}, "
        "forecast_availability_frames=0",
        flush=True,
    )

    fractional_exec_df = pd.read_csv(os.path.join(data_dir, "proposals_time_share_mod.csv"))
    already_executed = pd.read_csv(os.path.join(data_dir, "cycle_10_sb_active_time_to_complete_at_c10_start.csv"))
    remaining_execution_counts = {
        row["SB_UID"]: int(np.round(row["execution_count_start_c10"]))
        for _, row in already_executed.iterrows()
    }

    shifts_df = pd.read_csv(os.path.join(data_dir, "shifts_dimensions.csv"))
    downtimes_df = pd.read_csv(os.path.join(data_dir, "downtimes_dimensions.csv"))
    sbs_df_raw = pd.read_csv(os.path.join(data_dir, "schedblocks_c10.csv"))
    projects_df_raw = pd.read_csv(os.path.join(data_dir, "projects_c10.csv"))

    print(f"Loading preprocessed weather from {preprocessed_weather_path} ...", flush=True)
    with open(preprocessed_weather_path, "rb") as handle:
        pw_data = pickle.load(handle)

    preprocessed_realized_weather = pw_data["realized_weather"]
    downtime_index_sets = {
        "weather": pw_data.get("weather_downtime_indices", set()),
        "technical": pw_data.get("technical_downtime_indices", set()),
        "scheduling": pw_data.get("scheduling_downtime_indices", set()),
        "engineering": pw_data.get("engineering_indices", set()),
    }
    print(
        "DSA_EB weather inputs [dashboard]: "
        f"realized_weather_slots={len(preprocessed_realized_weather)}, "
        f"downtime_sizes={{'weather': {len(downtime_index_sets['weather'])}, "
        f"'technical': {len(downtime_index_sets['technical'])}, "
        f"'scheduling': {len(downtime_index_sets['scheduling'])}, "
        f"'engineering': {len(downtime_index_sets['engineering'])}}}",
        flush=True,
    )

    num_days = len(pd.date_range(start=start_date, end=end_date, freq="D"))
    max_time_steps = num_days * 48

    jobs, projects, time_steps, idx_to_timestamp, realized_weather, _unused_weather_forecasts, _unused_scores = (
        load_and_prepare_data(
            max_time_steps=max_time_steps,
            sbs_df_raw=sbs_df_raw,
            projects_df_raw=projects_df_raw,
            availability_df_raw=daily_dfs,
            scores_df_raw=daily_score_dfs,
            fractional_execs=fractional_exec_df,
            remaining_execution_counts=remaining_execution_counts,
            shifts_df=shifts_df,
            downtimes_df=downtimes_df,
            fraction_jobs_to_drop=0,
            add_fillers=False,
            seed=seed,
            cached_realized_weather=preprocessed_realized_weather,
            # DSA+EB greedy does not use forecast-based planning inputs, so we can
            # reuse the existing loader without reading any forecast pickle.
            cached_weather_forecasts={},
            availability_df_forecast=None,
            weight_data_dir=data_dir,
        )
    )

    return jobs, projects, time_steps, idx_to_timestamp, realized_weather, downtime_index_sets


def _build_config_calendar(
        data_dir: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        jobs: List[Dict],
        weights: Dict[str, float],
        temp_long_term_path: str,
) -> pd.DataFrame:
    c10_old_master_list = pd.read_csv(os.path.join(data_dir, "sb12m_master_prepared_c10.csv"))
    sb_map = {uid: uid for uid in c10_old_master_list["SB_UID"]}
    start_year = _resolve_cycle_year(start_date)

    base_calendar_from_long_term = solve_long_term_schedule(
        weights=weights,
        output_path=temp_long_term_path,
        data_dir=data_dir,
        jobs=deepcopy(jobs),
        config_start_date=start_date.strftime("%Y-%m-%d"),
        sb_map=sb_map,
        calendar_only=True,
        year=start_year,
        end_date=end_date,
    )

    config_calendar = load_config_calendar(start_date, base_calendar_from_long_term)
    config_calendar = config_calendar[config_calendar["Start"] <= end_date].copy()
    if len(config_calendar) > 0 and config_calendar.iloc[-1]["End"] > end_date:
        config_calendar.iloc[-1, config_calendar.columns.get_loc("End")] = end_date
    return config_calendar


def run_dashboard_dsa_eb(args: argparse.Namespace) -> Dict[str, Dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    start_date = pd.to_datetime(args.start_date, utc=True)
    end_date = pd.to_datetime(args.end_date, utc=True)
    if end_date < start_date:
        raise ValueError("--end_date must be on or after --start_date")

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    output_path = os.path.abspath(os.path.expanduser(args.output_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    preprocessed_weather_path = _resolve_preprocessed_weather_path(
        start_date=start_date,
        end_date=end_date,
        preprocessed_weather_path=args.preprocessed_weather,
        preprocessed_root=args.preprocessed_root,
    )
    if not os.path.exists(preprocessed_weather_path):
        raise FileNotFoundError(f"Preprocessed realized weather file not found: {preprocessed_weather_path}")

    weights = _build_fixed_weights(args)
    log_dsa_eb_run_configuration(
        source_label="dashboard",
        algorithm_name="dsa_eb",
        start_date=start_date,
        end_date=end_date,
        seed=args.seed,
        weights=weights,
        executive_quotas=EXECUTIVE_QUOTAS,
        eb_ramp_exponent=args.eb_ramp_exponent,
        extra_fields={
            "data_dir": data_dir,
            "output_path": output_path,
            "preprocessed_weather_path": preprocessed_weather_path,
            "preprocessed_root": args.preprocessed_root,
            "override_cycle_grade_score": args.override_cycle_grade_score,
            "cycle_grade_score_override_values": {
                "A": args.cycle_grade_score_A,
                "B": args.cycle_grade_score_B,
                "C": args.cycle_grade_score_C,
            },
            "internal_grade_splits": {
                "sb": DEFAULT_SB_GRADE_SPLITS,
                "project": DEFAULT_PROJ_GRADE_SPLITS,
            },
            "forecast_mode": "forecast-free",
        },
    )

    jobs, projects, time_steps, idx_to_timestamp, realized_weather, downtime_index_sets = _load_operational_inputs(
        data_dir=data_dir,
        preprocessed_weather_path=preprocessed_weather_path,
        start_date=start_date,
        end_date=end_date,
        seed=args.seed,
    )
    log_dsa_eb_dataset_summary(
        source_label="dashboard",
        stage_label="post_load_and_prepare_data",
        jobs=jobs,
        projects=projects,
        time_steps=time_steps,
        idx_to_timestamp=idx_to_timestamp,
        realized_weather=realized_weather,
    )
    limited_indices = {i for i, ts in idx_to_timestamp.items() if start_date <= ts <= end_date}
    if limited_indices:
        max_time_idx = max(limited_indices)
        time_steps_limited = max_time_idx + 1
        print(f"\n--- [dashboard] Filtering to date range {start_date.date()} to {end_date.date()} ---", flush=True)
        print(f"  time_steps: {time_steps} -> {time_steps_limited}", flush=True)
        time_steps = time_steps_limited

        realized_weather = {t: w for t, w in realized_weather.items() if t < time_steps}
        idx_to_timestamp = {i: ts for i, ts in idx_to_timestamp.items() if i < time_steps}

        jobs_before = len(jobs)
        jobs_filtered = []
        for job in jobs:
            job_available_filtered = [t for t in job.get("available", []) if t < time_steps]
            if job_available_filtered:
                j = job.copy()
                j["available"] = job_available_filtered
                if "pwv_thresholds" in j:
                    j["pwv_thresholds"] = {t: v for t, v in j["pwv_thresholds"].items() if t < time_steps}
                if "forecast_available" in j:
                    j["forecast_available"] = [t for t in j["forecast_available"] if t < time_steps]
                if "forecast_pwv_thresholds" in j:
                    j["forecast_pwv_thresholds"] = {
                        t: v for t, v in j["forecast_pwv_thresholds"].items() if t < time_steps
                    }
                if "condition_scores" in j:
                    j["condition_scores"] = {t: v for t, v in j["condition_scores"].items() if t < time_steps}
                if "base_ha_scores" in j:
                    j["base_ha_scores"] = {t: v for t, v in j["base_ha_scores"].items() if t < time_steps}
                jobs_filtered.append(j)
        jobs = jobs_filtered
        print(f"  jobs: {jobs_before} -> {len(jobs)}", flush=True)

        job_ids_remaining = {j["job_id"] for j in jobs}
        projects_before = len(projects)
        projects_filtered = []
        for project in projects:
            if any(pid in job_ids_remaining for pid in project.get("job_ids", [])):
                p = project.copy()
                p["job_ids"] = [pid for pid in project.get("job_ids", []) if pid in job_ids_remaining]
                projects_filtered.append(p)
        projects = projects_filtered
        print(f"  projects: {projects_before} -> {len(projects)}", flush=True)
        print("--- [dashboard] End date-range filter ---\n", flush=True)
        log_dsa_eb_dataset_summary(
            source_label="dashboard",
            stage_label="post_date_filter",
            jobs=jobs,
            projects=projects,
            time_steps=time_steps,
            idx_to_timestamp=idx_to_timestamp,
            realized_weather=realized_weather,
        )
    _apply_cycle_grade_score_override(
        jobs=jobs,
        override_enabled=args.override_cycle_grade_score,
        grade_scores={
            "A": args.cycle_grade_score_A,
            "B": args.cycle_grade_score_B,
            "C": args.cycle_grade_score_C,
        },
    )
    print(f"Loaded {len(jobs)} jobs and {len(projects)} projects.", flush=True)
    log_dsa_eb_dataset_summary(
        source_label="dashboard",
        stage_label="post_cycle_grade_override",
        jobs=jobs,
        projects=projects,
        time_steps=time_steps,
        idx_to_timestamp=idx_to_timestamp,
        realized_weather=realized_weather,
    )

    temp_long_term_path = os.path.join(
        os.path.dirname(output_path),
        f"temp_long_term_schedule_dashboard_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
    )
    config_calendar = _build_config_calendar(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        jobs=jobs,
        weights=weights,
        temp_long_term_path=temp_long_term_path,
    )

    observable_time = int(
        np.sum(
            [
                1
                for i in range(time_steps)
                if not np.isnan(realized_weather.get(i, (np.nan, np.nan))[0])
            ]
        )
    )
    log_dsa_eb_dataset_summary(
        source_label="dashboard",
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
        eb_ramp_exponent=args.eb_ramp_exponent,
    )

    planner_value, schedule_log = planning_loop_eb_greedy(
        jobs=deepcopy(jobs),
        projects=projects,
        realized_weather=realized_weather,
        time_steps=time_steps,
        job_selector_fn=dsa_selector,
        executive_quotas_frac=EXECUTIVE_QUOTAS,
        idx_to_timestamp=idx_to_timestamp,
        config_calendar=config_calendar,
        debug=args.debug,
        weights=weights,
        total_observable_time=observable_time,
    )

    results = run_eval_real(
        algo_values={"dsa_eb": planner_value},
        algo_schedules={"dsa_eb": schedule_log},
        all_jobs=jobs,
        all_projects=projects,
        total_time_steps=time_steps,
        idx_to_timestamp_map=idx_to_timestamp,
        realized_weather_dict=realized_weather,
        fractional_executive_quotas=EXECUTIVE_QUOTAS,
        weights=weights,
        total_observable_time=observable_time,
        downtime_index_sets=downtime_index_sets,
    )

    with open(output_path, "wb") as handle:
        pickle.dump(results, handle)
    print(f"Saved DSA+EB dashboard result to {output_path}", flush=True)

    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full-cycle DSA+EB scheduler without forecast inputs and save the standard result pickle."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="AOOSP data directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output pickle file.")
    parser.add_argument("--start_date", type=str, required=True, help="Simulation start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, required=True, help="Simulation end date (YYYY-MM-DD).")
    parser.add_argument("--seed", type=int, default=31415, help="Random seed.")
    parser.add_argument(
        "--preprocessed_weather",
        type=str,
        default=None,
        help="Path to realized_weather.pkl. If omitted, pass --preprocessed_root instead.",
    )
    parser.add_argument(
        "--preprocessed_root",
        type=str,
        default=None,
        help="Root containing year_<cycle>/realized_weather.pkl.",
    )
    parser.add_argument("--w_sb", type=float, required=True, help="Overall SB-completion weight.")
    parser.add_argument("--w_proj", type=float, required=True, help="Overall project-completion weight.")
    parser.add_argument("--w_util", type=float, required=True, help="Overall utilization weight.")
    parser.add_argument("--w_ebp", type=float, required=True, help="Overall executive-balance penalty weight.")
    parser.add_argument(
        "--eb_ramp_exponent",
        type=float,
        default=50.0,
        help="Exponent for the DSA+EB executive-balance ramp schedule.",
    )
    parser.add_argument(
        "--override_cycle_grade_score",
        action="store_true",
        help="Override the DSA cycle_grade_score component using the explicit A/B/C values below.",
    )
    parser.add_argument(
        "--cycle_grade_score_A",
        type=float,
        default=10.0,
        help="Cycle-grade score to use for A-grade jobs when --override_cycle_grade_score is enabled.",
    )
    parser.add_argument(
        "--cycle_grade_score_B",
        type=float,
        default=4.0,
        help="Cycle-grade score to use for B-grade jobs when --override_cycle_grade_score is enabled.",
    )
    parser.add_argument(
        "--cycle_grade_score_C",
        type=float,
        default=-100.0,
        help="Cycle-grade score to use for C-grade jobs when --override_cycle_grade_score is enabled.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose planner debug logging.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dashboard_dsa_eb(args)


if __name__ == "__main__":
    main()
