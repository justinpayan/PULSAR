from fixed_params import *
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import OrderedDict, Counter, defaultdict

def _is_execution_valid(
    job: Dict[str, Any],
    t_start: int,
    realized_weather_dict: Dict[int, Tuple[float, float]],
    total_time_steps: int
) -> bool:
    """
    Checks if a single scheduled execution of a job is valid from start to finish.

    A job is valid if:
    1. It starts within its general availability window.
    2. It fits entirely within the simulation horizon.
    3. For its ENTIRE duration, the realized weather is known (not NaN) and
       meets the job's specific PWV and RMS thresholds for each time step.

    Args:
        job: The job dictionary.
        t_start: The global start time index of the execution.
        realized_weather_dict: The dictionary of true weather conditions.
        total_time_steps: The total number of steps in the simulation.

    Returns:
        True if the execution is valid, False otherwise.
    """
    # 1. Basic placement checks
    if t_start not in job.get("available", []):
        return False
    if t_start + job["length"] > total_time_steps:
        return False

    # 2. Check weather conditions are not nan for entire duration, and that the
    # job had the right starting conditions.
    for t_offset in range(job["length"]):
        current_t = t_start + t_offset
        real_pwv, real_rms = realized_weather_dict.get(current_t, (np.nan, np.nan))

        # Weather must be known (not NaN)
        if pd.isna(real_pwv) or pd.isna(real_rms):
            return False

        # Get the specific thresholds for this time step
        pwv_thresh = job["pwv_thresholds"].get(current_t, np.inf)
        rms_thresh = job["rms_threshold"]

        # Check if weather conditions are met
        if t_offset == 0 and not (real_pwv <= pwv_thresh and real_rms >= rms_thresh):
            return False

    # If all checks passed for the entire duration
    return True

def evaluate_against_observed_weather(
        jobs,
        projects,
        sched,
        weather_df,
        time_steps
):
    """
    Compare two schedules by evaluating their objective values under observed weather.

    Args:
        schedule_1, schedule_2: lists of "job_id@t" strings
        name_1, name_2: names to tag each result
        weather_df: must contain "PWV_realized" and "RMS_realized"
    Returns:
        Dictionary with values for schedules
    """

    def compute_objective(schedule):
        job_start = {}
        for entry in schedule:
            jid, t = entry.split("@")
            job_start[jid] = int(t)

        scheduled_jobs = set()
        busy_times = set()
        exec_time = {k: 0 for k in EXECUTIVES}

        for job in jobs:
            jid = job["job_id"]
            if jid not in job_start:
                continue
            t_start = job_start[jid]
            t_end = t_start + job["length"]

            if t_end > time_steps or weather_df.loc[t_start, "PWV_realized"] > job["pwv_thresh"] or weather_df.loc[
                t_start, "RMS_realized"] > job["rms_thresh"]:
                continue  # invalid
            if any(t in busy_times for t in range(t_start, t_end)):
                continue  # overlap
            if t_start not in job["valid_starts"]:
                continue
            # Valid job
            scheduled_jobs.add(jid)
            for t_busy in range(t_start, t_end):
                busy_times.add(t_busy)
            exec_time[job["executive"]] += job["length"]

        base_reward = sum(job["weight"] for job in jobs if job["job_id"] in scheduled_jobs)
        completed_projects = [
            p for p in projects if all(jid in scheduled_jobs for jid in p["job_ids"])
        ]
        bonus = sum(2 * p["weight"] for p in completed_projects)
        total_reward = base_reward + bonus
        return total_reward

    value = compute_objective(sched)
    return value


def check_executive_balance(schedule, jobs, executive_quotas, time_steps):
    """
    Given a schedule (list of "job_id@t" strings), job list, and quotas, check if
    the executive balance constraints are satisfied.

    Returns: (is_valid: bool, exec_time_fraction: dict)
    """
    job_lookup = {job["job_id"]: job for job in jobs}
    exec_time = {exec_name: 0 for exec_name in executive_quotas}

    for entry in schedule:
        job_id, t = entry.split("@")
        job = job_lookup[job_id]
        exec_time[job["executive"]] += job["length"]

    exec_time_fraction = {
        k: exec_time[k] / time_steps for k in executive_quotas
    }

    is_valid = True
    for exec_name, (lb, ub) in executive_quotas.items():
        frac = exec_time_fraction[exec_name]
        if not (lb <= frac <= ub):
            is_valid = False
            break

    return is_valid, exec_time_fraction

def check_schedule_validity(schedule, jobs, projects, weather_df, executive_quotas, time_steps):
    """
    Check if a schedule is valid:
    - All jobs start within valid time slots
    - No job overlaps
    - Jobs stay within the time horizon
    - Job can start only if weather conditions are met
    - Executive balance is respected

    Returns: (is_valid: bool, failure_reasons: list, exec_time_fraction: dict)
    """
    job_lookup = {job["job_id"]: job for job in jobs}
    exec_time = {exec_name: 0 for exec_name in executive_quotas}
    occupied_time = set()
    failure_reasons = []

    for entry in schedule:
        try:
            job_id, t_str = entry.split("@")
            t = int(t_str)
        except Exception:
            failure_reasons.append(f"Malformed entry '{entry}'")
            continue

        if job_id not in job_lookup:
            failure_reasons.append(f"Unknown job_id {job_id}")
            continue

        job = job_lookup[job_id]
        job_len = job["length"]

        # Check start time validity
        if t not in job["valid_starts"]:
            failure_reasons.append(f"{job_id} starts at invalid time {t}")

        # Check schedule doesn't exceed time horizon
        if t + job_len > time_steps:
            failure_reasons.append(f"{job_id} exceeds horizon: {t} + {job_len} > {time_steps}")

        # Check overlap
        for dt in range(job_len):
            if (t + dt) in occupied_time:
                failure_reasons.append(f"{job_id} overlaps with another job at time {t + dt}")
                break

        # Check weather conditions at start time
        if weather_df.loc[t, "PWV_realized"] > job["pwv_thresh"]:
            failure_reasons.append(f"{job_id} violates PWV threshold at time {t}")
        if weather_df.loc[t, "RMS_realized"] > job["rms_thresh"]:
            failure_reasons.append(f"{job_id} violates RMS threshold at time {t}")

        # Book time and add executive time
        for dt in range(job_len):
            occupied_time.add(t + dt)
        exec_time[job["executive"]] += job_len

    # Executive balance
    exec_time_fraction = {
        k: exec_time[k] / time_steps for k in executive_quotas
    }
    for exec_name, (lb, ub) in executive_quotas.items():
        frac = exec_time_fraction[exec_name]
        if not (lb <= frac <= ub):
            failure_reasons.append(f"Executive {exec_name} has {frac:.2f}, outside bounds ({lb}, {ub})")

    is_valid = len(failure_reasons) == 0
    return is_valid, failure_reasons


def _add_job_time_to_exec_balance(
    exec_time_dict: Dict[str, float], job: Dict[str, Any]
) -> None:
    """
    Accumulates job execution time into the executive time dictionary.

    This function correctly handles jobs assigned to a single executive (str)
    and jobs fractionally split between executives (dict).

    Args:
        exec_time_dict: A dictionary to be updated in-place. Maps executive name to total time.
        job: The job dictionary, which contains 'length' and 'executive' keys.
    """
    executive_info = job.get("executive")
    job_length = job.get("length", 0)

    if isinstance(executive_info, str):
        # Case 1: Standard job with a single executive
        if executive_info in exec_time_dict:
            exec_time_dict[executive_info] += job_length
    elif isinstance(executive_info, dict):
        # Case 2: Fractionally-split job
        for exec_name, fraction in executive_info.items():
            if exec_name in exec_time_dict:
                exec_time_dict[exec_name] += job_length * fraction

def evaluate_value_real(
        all_jobs_list: List[Dict[str, Any]],
        all_projects_list: List[Dict[str, Any]],
        schedule_log: List[str],  # List of "job_id@global_time_idx"
        realized_weather_dict: Dict[int, Tuple[float, float]],  # {global_time_idx: (pwv, rms)}
        total_time_steps: int
) -> float:
    """
    Calculates the objective value of a schedule given the true realized weather.
    This re-simulates the schedule execution to verify its value.
    """
    job_lookup = {job["job_id"]: job for job in all_jobs_list}

    actual_scheduled_job_ids = []
    # No need to check for overlaps if the schedule comes from a valid planner,
    # but good for a strict re-evaluation. For simplicity, we'll trust planner on overlaps for now.
    # busy_times = set()

    for entry in schedule_log:
        try:
            job_id, t_start_str = entry.split("@")
            t_start = int(t_start_str)
        except ValueError:
            print(f"Warning (evaluate_value_real): Malformed schedule entry '{entry}'. Skipping.")
            continue

        if job_id not in job_lookup:
            print(f"Warning (evaluate_value_real): Job '{job_id}' from schedule not in all_jobs_list. Skipping.")
            continue

        job = job_lookup[job_id]

        # Check if job could have actually started at t_start based on REAL weather
        current_pwv_real, current_rms_real = realized_weather_dict.get(t_start, (np.nan, np.nan))

        if pd.isna(current_pwv_real) or pd.isna(current_rms_real):
            print(
                f"Warning (evaluate_value_real): Job '{job_id}' scheduled at t={t_start} but real weather is NaN. Considered invalid for value calc.")
            continue  # Cannot start if real weather is unknown

        # Get PWV threshold for the job at this specific start time
        pwv_threshold_at_t_start = 0.0  # Default to strict limit if not specified
        if isinstance(job.get("pwv_thresholds"), dict):
            pwv_threshold_at_t_start = job["pwv_thresholds"].get(t_start, 0.0)
        elif job.get("pwv_thresholds") is not None:  # Scalar threshold
            pwv_threshold_at_t_start = job["pwv_thresholds"]

        rms_threshold_job = job.get("rms_threshold", 1000)  # Default to very strict if not available

        # Weather check based on your solve_prophet_real logic:
        # PWV <= threshold (good)
        # RMS >= threshold (good) -> your prophet had realized_rms < job_rms_thresh as BAD
        weather_ok = (current_pwv_real <= pwv_threshold_at_t_start) and \
                     (current_rms_real >= rms_threshold_job)

        # General availability check (e.g., LST, elevation - pre-weather)
        # This check might be redundant if the planner already honored it, but good for robust eval.
        generally_available = t_start in job.get("available", [])

        fits_in_horizon = (t_start + job["length"] <= total_time_steps)

        if weather_ok and generally_available and fits_in_horizon:
            actual_scheduled_job_ids.append(job_id)
            # Add to busy_times if checking overlaps:
            # for t_busy in range(t_start, t_start + job["length"]):
            #     if t_busy in busy_times: # Overlap detected! This schedule is inherently flawed.
            #          print(f"FATAL ERROR (evaluate_value_real): Overlap detected for {job_id} at {t_busy}. Schedule invalid.")
            #          return -float('inf') # Or handle error appropriately
            #     busy_times.add(t_busy)
        else:
            print(
                f"Info (evaluate_value_real): Job {job_id}@{t_start} not considered for value. WeatherOK: {weather_ok}, GenAvail: {generally_available}, FitsHorizon: {fits_in_horizon}")
            print(
                f"  Details: Real PWV={current_pwv_real:.2f} (Thresh={pwv_threshold_at_t_start:.2f}), Real RMS={current_rms_real:.2f} (Thresh={rms_threshold_job:.2f})")

            pass

    base_reward = sum(job_lookup[jid]["weight"] for jid in actual_scheduled_job_ids)

    bonus = 0
    for project in all_projects_list:
        required_execs = {
            jid: job_lookup[jid]['remaining_execs']
            for jid in project["job_ids"]
        }
        if sum(required_execs.values()):
            actual_execs = Counter()
            for elt in schedule_log:
                job_id, _ = elt.split("@")
                if job_id in required_execs:
                    actual_execs[job_id] += 1

            project_complete = all(actual_execs[jid] >= required_execs[jid] for jid in project["job_ids"])
            if project_complete:
                print("Project complete: ", project['project_id'])
                # for jid in project['job_ids']:
                #     print(job_lookup[jid])
                # print()
                # bonus += PROJECT_WEIGHT * project["weight"]
                bonus += project['weight']

    total_value = base_reward + bonus
    return total_value


def check_executive_balance_real(
        schedule_log: List[str],
        all_jobs_list: List[Dict[str, Any]],
        fractional_executive_quotas: Dict[str, Tuple[float, float]],  # e.g., {'EU': (0.1, 0.3)}
        total_time_steps: int,
        # For re-validating schedule against real weather to count only validly used time
        realized_weather_dict: Dict[int, Tuple[float, float]]
) -> Tuple[bool, Dict[str, float]]:
    """
    Checks if executive balance (fractional quotas) is met by a schedule.
    Only counts time for jobs that were validly schedulable under REAL weather.
    """
    job_lookup = {job["job_id"]: job for job in all_jobs_list}
    exec_time_used_validly = {exec_name: 0 for exec_name in fractional_executive_quotas}
    all_execs_in_jobs = set()

    for j in all_jobs_list:
        if isinstance(j['executive'], str):
            all_execs_in_jobs.add(j['executive'])
        elif isinstance(j['executive'], dict):
            all_execs_in_jobs.update(j['executive'].keys())

    for exec_j in all_execs_in_jobs:  # Ensure all possible execs are tracked
        if exec_j not in exec_time_used_validly:
            exec_time_used_validly[exec_j] = 0


    for entry in schedule_log:
        try:
            job_id, t_start_str = entry.split("@")
            t_start = int(t_start_str)
        except ValueError:
            continue  # Skip malformed

        if job_id not in job_lookup: continue
        job = job_lookup[job_id]

        if _is_execution_valid(job, t_start, realized_weather_dict, total_time_steps):
            _add_job_time_to_exec_balance(exec_time_used_validly, job)

    exec_time_fraction = {}
    total_time_steps = sum([exec_time_used_validly.get(exec_name_q, 0) for exec_name_q in fractional_executive_quotas])
    for exec_name_q in fractional_executive_quotas.keys():  # Iterate over execs WITH quotas
        exec_time_fraction[exec_name_q] = exec_time_used_validly.get(exec_name_q, 0) / total_time_steps \
            if total_time_steps > 0 else 0

    # Also report fractions for execs without explicit quotas, if any were used
    for exec_name_j in all_execs_in_jobs:
        if exec_name_j not in exec_time_fraction:
            exec_time_fraction[exec_name_j] = exec_time_used_validly.get(exec_name_j, 0) / total_time_steps \
                if total_time_steps > 0 else 0

    is_overall_valid = True
    for exec_name_q, (lb_frac, ub_frac) in fractional_executive_quotas.items():
        frac = exec_time_fraction.get(exec_name_q, 0)  # Get fraction, default to 0 if exec had no usage
        if not (lb_frac <= frac <= ub_frac):
            is_overall_valid = False
            # No break, collect all fractions

    return is_overall_valid, exec_time_fraction


def check_schedule_validity_real(
        schedule_log: List[str],
        all_jobs_list: List[Dict[str, Any]],
        realized_weather_dict: Dict[int, Tuple[float, float]],
        fractional_executive_quotas: Dict[str, Tuple[float, float]],
        total_time_steps: int
) -> Tuple[bool, List[str]]:
    """
    Checks if a schedule is structurally valid and meets constraints.
    - Execution counts are now only incremented for SUCCESSFUL executions.
    - A schedule is only invalid for over-scheduling if it successfully completes
      a job more times than required.
    """
    job_lookup = {job["job_id"]: job for job in all_jobs_list}
    occupied_time_slots = set()
    failure_reasons = []

    parsed_schedule = []
    for entry_idx, entry in enumerate(schedule_log):
        try:
            job_id, t_start_str = entry.split("@")
            t_start = int(t_start_str)
            parsed_schedule.append({"job_id": job_id, "t_start": t_start})
        except ValueError:
            failure_reasons.append(f"Malformed schedule entry '{entry}' at index {entry_idx}")

    parsed_schedule.sort(key=lambda x: x["t_start"])

    # --- This counter will now ONLY track successful executions ---
    valid_job_counts = Counter()

    for item in parsed_schedule:
        job_id, t_start = item["job_id"], item["t_start"]

        if job_id not in job_lookup:
            failure_reasons.append(f"Job '{job_id}' from schedule not found.")
            continue
        job = job_lookup[job_id]
        job_len = job["length"]

        # Structural checks (availability, horizon, overlaps)
        if t_start not in job.get("available", []):
            failure_reasons.append(f"Job '{job_id}' @{t_start} is outside its general availability.")
        if t_start + job_len > total_time_steps:
            failure_reasons.append(f"Job '{job_id}' @{t_start} exceeds horizon {total_time_steps}.")
        for t_busy in range(t_start, t_start + job_len):
            if t_busy in occupied_time_slots:
                failure_reasons.append(f"Job '{job_id}' @{t_start} overlaps at time {t_busy}.")
                break
        for t_b in range(t_start, t_start + job_len): occupied_time_slots.add(t_b)

        # --- Check for success and conditionally increment counter ---
        # The _is_execution_valid check implicitly checks weather for the full duration.
        # We still add failure reasons for weather to be informative, but it won't
        # impact the over-scheduling check unless the execution was actually valid.
        is_successful = _is_execution_valid(job, t_start, realized_weather_dict, total_time_steps)

        if is_successful:
            valid_job_counts[job_id] += 1
        else:
            pass

    for job_id, count in valid_job_counts.items():
        required_execs = job_lookup[job_id].get('total_execs', 1)
        if count > required_execs:
            failure_reasons.append(
                f"Job '{job_id}' was successfully executed {count} times, but only requires {required_execs}.")

    # Executive Balance check (correctly uses its own internal logic based on valid executions)
    exec_balance_ok, _ = check_executive_balance_real(
        schedule_log, all_jobs_list, fractional_executive_quotas, total_time_steps, realized_weather_dict
    )
    if not exec_balance_ok:
        failure_reasons.append("Executive balance constraints violated based on validly completed SBs.")

    is_overall_valid = (len(failure_reasons) == 0)
    return is_overall_valid, failure_reasons


def analyze_schedule_performance(
        schedule_log: List[str],
        all_jobs: List[Dict[str, Any]],
        all_projects: List[Dict[str, Any]],
        realized_weather_dict: Dict[int, Tuple[float, float]],
        total_time_steps: int,
        downtime_index_sets: Dict[str, set] = None,
) -> Dict[str, Any]:
    """
    Performs a deep analysis of a schedule, calculating completion rates by grade,
    large project completion, and telescope usage ratio.
    """
    job_lookup = {job['job_id']: job for job in all_jobs}
    project_lookup = {p['project_id']: p for p in all_projects}

    # --- 1. First, determine which scheduled executions are valid ---
    successful_execution_job_ids = []
    total_time_observed = 0.0

    for entry in schedule_log:
        try:
            job_id, t_start_str = entry.split("@")
            t_start = int(t_start_str)
        except (ValueError, IndexError):
            continue

        if job_id not in job_lookup:
            continue

        job = job_lookup[job_id]

        if _is_execution_valid(job, t_start, realized_weather_dict, total_time_steps):
            # Instead of appending the raw entry, append the parsed job_id
            successful_execution_job_ids.append(job_id)
            total_time_observed += job['length']

    # Count how many times each SB was validly executed
    # This is now much simpler as we have a direct list of job_ids
    valid_execution_counts = Counter(successful_execution_job_ids)

    # --- 2. Calculate totals and completion counts ---
    # Total counts for normalization
    total_sbs_by_grade = Counter(j.get('grade', 'N/A') for j in all_jobs if j.get('type') != 'filler')
    total_projects_by_grade = Counter()
    for p in all_projects:
        # Find the grade from the first SB in the project
        first_job_id = p['job_ids'][0]
        grade = job_lookup[first_job_id].get('grade', 'N/A')
        if job_lookup[first_job_id].get('type') != 'filler':
            total_projects_by_grade[grade] += 1

    total_large_projects = sum(1 for p in all_projects if p['project_id'].endswith(".L"))

    # Counts of completed items
    total_required_execs_by_grade = Counter()
    successful_execs_by_grade = Counter()

    for job in all_jobs:
        if job.get('type') == 'filler':
            continue
        grade = job.get('grade', 'N/A')
        required_execs = job.get('total_execs', 1)
        job_id = job['job_id']

        total_required_execs_by_grade[grade] += required_execs
        successful_execs_by_grade[grade] += valid_execution_counts.get(job_id, 0)

    print(f"total_required_execs_by_grade: {total_required_execs_by_grade}")
    print(f"successful_execs_by_grade: {successful_execs_by_grade}")

    total_projects_by_grade = Counter()
    total_large_projects = sum(1 for p in all_projects if p['project_id'].endswith(".L"))
    completed_projects_by_grade = Counter()
    completed_large_projects = 0

    # Check Project completion
    for project in all_projects:
        grade = project.get('grade', 'N/A')
        total_projects_by_grade[grade] += 1

        # print(f"Checking project completion for: {project['project_id']}")
        is_complete = True
        for job_id in project['job_ids']:
            job_in_proj = job_lookup[job_id]
            required_execs = job_in_proj.get('total_execs', 1)

            # print(f"Job: {job_id}, required_execs: {required_execs}, valid_execution_counts.get(job_id, 0): {valid_execution_counts.get(job_id, 0)}", flush=True)

            if valid_execution_counts.get(job_id, 0) < required_execs:
                is_complete = False
                break

        if is_complete:
            completed_projects_by_grade[grade] += 1
            if project['project_id'].endswith(".L"):
                completed_large_projects += 1

    # --- 3. Calculate percentages ---
    results = {
        # --- MODIFIED: Use fractional execution completion for SBs ---
        'completion_pct_sb_A': (successful_execs_by_grade['A'] / total_required_execs_by_grade['A'] * 100) if
        total_required_execs_by_grade['A'] > 0 else 0,
        'completion_pct_sb_B': (successful_execs_by_grade['B'] / total_required_execs_by_grade['B'] * 100) if
        total_required_execs_by_grade['B'] > 0 else 0,
        'completion_pct_sb_C': (successful_execs_by_grade['C'] / total_required_execs_by_grade['C'] * 100) if
        total_required_execs_by_grade['C'] > 0 else 0,
        # --- END MODIFICATION ---

        # Project completion percentages remain the same
        'completion_pct_proj_A': (completed_projects_by_grade['A'] / total_projects_by_grade['A'] * 100) if
        total_projects_by_grade['A'] > 0 else 0,
        'completion_pct_proj_B': (completed_projects_by_grade['B'] / total_projects_by_grade['B'] * 100) if
        total_projects_by_grade['B'] > 0 else 0,
        'completion_pct_proj_C': (completed_projects_by_grade['C'] / total_projects_by_grade['C'] * 100) if
        total_projects_by_grade['C'] > 0 else 0,
        'completion_pct_proj_large': (
                    completed_large_projects / total_large_projects * 100) if total_large_projects > 0 else 0,
    }
    print(f"results: {results}")

    # --- 4. Calculate Usage Ratio ---
    total_valid_time_steps = sum(
        1 for t, (pwv, rms) in realized_weather_dict.items()
        if not (pd.isna(pwv) or pd.isna(rms))
    )
    results['usage_ratio'] = (total_time_observed / total_valid_time_steps * 100) if total_valid_time_steps > 0 else 0
    results['total_slots_used'] = total_time_observed
    results['total_valid_slots'] = total_valid_time_steps

    # --- 5. Scheduling-downtime and non-downtime utilization ---
    if downtime_index_sets is not None:
        scheduling_indices = downtime_index_sets.get("scheduling", set())
        all_downtime_indices = (
            downtime_index_sets.get("weather", set())
            | downtime_index_sets.get("technical", set())
            | scheduling_indices
            | downtime_index_sets.get("engineering", set())
        )

        # Build set of time slots occupied by successfully scheduled jobs
        occupied_slots = set()
        for entry in schedule_log:
            try:
                job_id, t_start_str = entry.split("@")
                t_start = int(t_start_str)
            except (ValueError, IndexError):
                continue
            if job_id not in job_lookup:
                continue
            job = job_lookup[job_id]
            if _is_execution_valid(job, t_start, realized_weather_dict, total_time_steps):
                for dt in range(job["length"]):
                    occupied_slots.add(t_start + dt)

        # 5a. How many scheduling-downtime slots now have something scheduled?
        sched_dt_used = occupied_slots & scheduling_indices
        results['scheduling_downtime_slots_used'] = len(sched_dt_used)
        results['scheduling_downtime_slots_total'] = len(scheduling_indices)

        # 5b. Non-downtime utilization: denominator = all slots not in any downtime
        non_downtime_slots = set(range(total_time_steps)) - all_downtime_indices
        non_dt_used = occupied_slots & non_downtime_slots
        results['non_downtime_slots_used'] = len(non_dt_used)
        results['non_downtime_slots_total'] = len(non_downtime_slots)
        results['non_downtime_utilization'] = (
            (len(non_dt_used) / len(non_downtime_slots) * 100)
            if non_downtime_slots else 0
        )

        # 5c. Schedulable utilization: denominator excludes weather, technical, and
        #     engineering downtime but includes scheduling downtime (those slots
        #     could potentially be used for observations).
        hard_downtime_indices = (
            downtime_index_sets.get("weather", set())
            | downtime_index_sets.get("technical", set())
            | downtime_index_sets.get("engineering", set())
        )
        schedulable_slots = set(range(total_time_steps)) - hard_downtime_indices
        schedulable_used = occupied_slots & schedulable_slots
        results['schedulable_slots_used'] = len(schedulable_used)
        results['schedulable_slots_total'] = len(schedulable_slots)
        results['schedulable_utilization'] = (
            (len(schedulable_used) / len(schedulable_slots) * 100)
            if schedulable_slots else 0
        )

        print(f"  Scheduling downtime slots used: {len(sched_dt_used)} / {len(scheduling_indices)}")
        print(f"  Non-downtime utilization:  {results['non_downtime_utilization']:.2f}% "
              f"({len(non_dt_used)} / {len(non_downtime_slots)} slots)")
        print(f"  Schedulable utilization:   {results['schedulable_utilization']:.2f}% "
              f"({len(schedulable_used)} / {len(schedulable_slots)} slots)")
        print(f"  Overall utilization:       {results['usage_ratio']:.2f}% "
              f"({total_time_observed} / {total_valid_time_steps} slots)")

    return results


def run_eval_real(
        algo_values: Dict[str, float],
        algo_schedules: Dict[str, List[str]],
        all_jobs: List[Dict[str, Any]],
        all_projects: List[Dict[str, Any]],
        total_time_steps: int,
        idx_to_timestamp_map: Dict[int, pd.Timestamp],
        realized_weather_dict: Dict[int, Tuple[float, float]],
        fractional_executive_quotas: Dict[str, Tuple[float, float]],
        weights: Dict[str, float] = None,
        total_observable_time: int = None,
        downtime_index_sets: Dict[str, set] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluates schedules from different algorithms against real weather and calculates performance metrics.
    """
    evaluation_results = OrderedDict()

    for alg_name, schedule_log in algo_schedules.items():
        print(f"\n--- Evaluating Algorithm: {alg_name} ---")

        # --- Block 1: Existing Value and Validity Checks (No changes) ---
        # true_value_on_real_weather = evaluate_value_real(
        #     all_jobs, all_projects, schedule_log, realized_weather_dict, total_time_steps
        # )
        is_structurally_valid, failure_reasons = check_schedule_validity_real(
            schedule_log, all_jobs, realized_weather_dict,
            fractional_executive_quotas, total_time_steps
        )
        exec_balance_met, exec_time_fractions = check_executive_balance_real(
            schedule_log, all_jobs, fractional_executive_quotas,
            total_time_steps, realized_weather_dict
        )
        planner_reported_value = algo_values.get(alg_name, -float('inf'))

        # --- Block 2: NEW - Call the performance analysis function ---
        performance_metrics = analyze_schedule_performance(
            schedule_log, all_jobs, all_projects, realized_weather_dict, total_time_steps,
            downtime_index_sets=downtime_index_sets,
        )

        # --- Block 3: Combine all results into the final dictionary ---
        alg_result = {
            "planner_value": planner_reported_value,
            # "evaluated_value_real": true_value_on_real_weather,
            "is_overall_valid": is_structurally_valid,
            "failure_reasons": failure_reasons if not is_structurally_valid else "None",
            "exec_balance_met": exec_balance_met,
            "exec_time_fractions_real": exec_time_fractions,
        }

        # Merge the new performance metrics into the results
        alg_result.update(performance_metrics)

        # Add the schedule log at the end for completeness
        alg_result["schedule_log"] = schedule_log

        evaluation_results[alg_name] = alg_result

        # --- Block 4: Enhanced Printout ---
        print(f"  Planner Reported Value: {planner_reported_value:.2f}")
        # print(f"  Evaluated Value (Real Weather): {true_value_on_real_weather:.2f}")
        print(f"  Overall Schedule Valid: {is_structurally_valid}")
        if not is_structurally_valid:
            for reason_idx, reason in enumerate(failure_reasons):
                print(f"    - Reason {reason_idx + 1}: {reason}")
        print(f"  Executive Balance Met: {exec_balance_met}")
        print(f"  Executive Time Fractions (Real): { {k: f'{v:.3f}' for k, v in exec_time_fractions.items()} }")

        # Print new metrics
        print(f"  Usage Ratio: {performance_metrics['usage_ratio']:.2f}%")
        print("  Completion Rates (by Grade):")
        print(
            f"    - SBs:      A: {performance_metrics['completion_pct_sb_A']:.1f}%, B: {performance_metrics['completion_pct_sb_B']:.1f}%, C: {performance_metrics['completion_pct_sb_C']:.1f}%")
        print(
            f"    - Projects: A: {performance_metrics['completion_pct_proj_A']:.1f}%, B: {performance_metrics['completion_pct_proj_B']:.1f}%, C: {performance_metrics['completion_pct_proj_C']:.1f}%")
        print(f"  Large Project Completion Rate: {performance_metrics['completion_pct_proj_large']:.1f}%", flush=True)
        
        # --- Calculate and print paper objective value (same formula as greedy Algorithm 2) ---
        if weights is not None and total_observable_time is not None:
            try:
                from planning_implementations import compute_paper_objective_value
                job_lookup = {j["job_id"]: j for j in all_jobs}
                valid_schedule_log = []
                exec_time_used = {k: 0.0 for k in fractional_executive_quotas}
                for j in all_jobs:
                    if isinstance(j.get("executive"), str):
                        exec_time_used.setdefault(j["executive"], 0.0)
                    elif isinstance(j.get("executive"), dict):
                        for exec_name in j["executive"]:
                            exec_time_used.setdefault(exec_name, 0.0)
                for entry in schedule_log:
                    try:
                        job_id, t_start_str = entry.split("@")
                        t_start = int(t_start_str)
                    except (ValueError, AttributeError):
                        continue
                    if job_id not in job_lookup:
                        continue
                    job = job_lookup[job_id]
                    if not _is_execution_valid(job, t_start, realized_weather_dict, total_time_steps):
                        continue
                    valid_schedule_log.append(entry)
                    _add_job_time_to_exec_balance(exec_time_used, job)
                paper_objective_value = compute_paper_objective_value(
                    valid_schedule_log, all_jobs, all_projects, exec_time_used,
                    total_observable_time, weights, fractional_executive_quotas
                )
                alg_result["paper_objective_value"] = paper_objective_value
                print(f"  Paper Objective Value (same as greedy): {paper_objective_value:.6f}")
            except Exception as e:
                print(f"  Warning: Could not calculate paper objective value: {e}")

        print_detailed_schedule(schedule_log, all_jobs, max_lines=20000)

    return evaluation_results


def _calculate_eb_l1_penalty(
        exec_time_used: Dict[str, float],
        quotas_frac: Dict[str, Tuple[float, float]],
        total_observable_time: int
) -> float:
    """Helper to calculate the normalized L1 penalty for a given state of executive time usage."""
    penalty = 0
    if total_observable_time == 0:
        return 0.0

    total_time_with_quotas = sum(exec_time_used.get(exec_name, 0) for exec_name in quotas_frac)
    if total_time_with_quotas == 0:
        return 0.0  # No penalty if no time was used in relevant executives

    for exec_name, (min_frac, max_frac) in quotas_frac.items():
        time_used = exec_time_used.get(exec_name, 0)
        frac = time_used / total_time_with_quotas

        if frac < min_frac:
            penalty += (min_frac - frac)
        elif frac > max_frac:
            penalty += (frac - max_frac)

    return penalty


def score_schedule(
        schedule: List[str],
        weather_dict: Dict[int, Tuple[float, float]],
        all_jobs: List[Dict],
        all_projects: List[Dict],
        weights: Dict,
        total_time_steps: int,
        normalization_totals: Dict,
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        cumulative_exec_time_offset: Optional[Dict[str, float]] = None,
        cumulative_observable_time_offset: int = 0
) -> float:
    """
    Calculates the full, weighted objective value of a hypothetical schedule against a
    single weather path (real or sampled).

    When cumulative offsets are provided, normalization constants (eta1, eta2, eta3) use
    cumulative observable time (from the beginning of the cycle), and executive balance
    is computed as the marginal improvement:
        obj4 = EB_before - EB_after
    where EB_before is the squared penalty from cumulative exec time before this horizon,
    and EB_after uses cumulative + schedule exec time. This makes obj4 positive when
    the schedule improves the overall executive balance.
    """
    if not schedule:
        return 0.0

    job_lookup = {j['job_id']: j for j in all_jobs}

    # 1. Determine successful executions
    successful_exec_counts = Counter()
    # Pre-populate exec_time_used with all known executive names so that
    # _add_job_time_to_exec_balance's "if key in dict" checks succeed.
    exec_time_used = defaultdict(float)
    for exec_name in executive_quotas_frac:
        exec_time_used[exec_name] = 0.0
    for job in all_jobs:
        if isinstance(job.get('executive'), str):
            exec_time_used.setdefault(job['executive'], 0.0)
        elif isinstance(job.get('executive'), dict):
            for en in job['executive']:
                exec_time_used.setdefault(en, 0.0)

    for entry in schedule:
        job_id, t_start_str = entry.split('@')
        t_start = int(t_start_str)
        job = job_lookup[job_id]

        if _is_execution_valid(job, t_start, weather_dict, total_time_steps):
            successful_exec_counts[job_id] += 1
            _add_job_time_to_exec_balance(exec_time_used, job)

    # 2. Calculate objective components
    total_observable_time = sum(
        1 for t in range(total_time_steps) if not np.isnan(weather_dict.get(t, (np.nan, np.nan))[0]))
    if total_observable_time == 0: return 0.0

    # Cumulative n' for normalization (from beginning of cycle through end of this horizon)
    n_prime_cumulative = cumulative_observable_time_offset + total_observable_time

    # --- Paper-style objective: combined normalization + squared EB ---
    # obj1: weighted observation completion
    sum_obs_weight = sum(
        job_lookup[jid].get('weight', 0) * count
        for jid, count in successful_exec_counts.items()
        if jid in job_lookup
    )

    # obj2: weighted project completion
    # Use remaining_execs (not total_execs) because successful_exec_counts
    # only tallies executions in this hypothetical schedule, not past ones.
    sum_proj_weight = 0.0
    for proj in all_projects:
        if not proj['job_ids']:
            continue
        is_complete = all(
            successful_exec_counts[jid] >= job_lookup[jid].get('remaining_execs', job_lookup[jid].get('total_execs', 1))
            for jid in proj['job_ids']
            if jid in job_lookup
        )
        if is_complete:
            sum_proj_weight += proj.get('weight', 0)

    # obj3: utilization (total time used in this horizon)
    total_time_used = sum(
        job_lookup[jid]['length'] * count
        for jid, count in successful_exec_counts.items()
        if jid in job_lookup
    )

    # Normalization: η1 = sum of top min(B, n') obs weights, η2 = sum of top min(P, n') project weights
    # Negative per-job/project weights are clamped to 0 here so that a negative
    # grade weight cannot make the denominator negative. The job/project weight
    # fields themselves are unaffected; only the normalization constant changes.
    n_prime = n_prime_cumulative
    obs_weights = []
    for j in all_jobs:
        w = max(0.0, j.get('weight', 0))
        rem = j.get('remaining_execs', j.get('total_execs', 1))
        obs_weights.extend([w] * max(0, rem))
    obs_weights.sort(reverse=True)
    topk_obs = min(len(obs_weights), max(1, n_prime))
    eta1 = max(sum(obs_weights[:topk_obs]), 1e-12) if obs_weights else 1.0

    proj_weights = sorted((max(0.0, p.get('weight', 0)) for p in all_projects), reverse=True)
    topk_proj = min(len(proj_weights), max(1, n_prime))
    eta2 = max(sum(proj_weights[:topk_proj]), 1e-12) if proj_weights else 1.0

    eta3 = n_prime

    obj1 = sum_obs_weight / eta1 if eta1 > 0 else 0
    obj2 = sum_proj_weight / eta2 if eta2 > 0 else 0
    obj3 = total_time_used / eta3 if eta3 > 0 else 0

    # obj4: executive balance
    # Paper: fraction_i = exec_time_i / total_exec_time (fractions sum to 1).
    if cumulative_exec_time_offset is not None:
        # Marginal improvement: EB_before - EB_after (positive when improving)
        eta4_before = sum(cumulative_exec_time_offset.values())
        eb_before = 0.0
        if eta4_before > 0:
            for exec_name, (min_frac, _max_frac) in executive_quotas_frac.items():
                frac_before = cumulative_exec_time_offset.get(exec_name, 0) / eta4_before
                eb_before += (min_frac - frac_before) ** 2

        combined_exec = dict(cumulative_exec_time_offset)
        for exec_name, time_val in exec_time_used.items():
            combined_exec[exec_name] = combined_exec.get(exec_name, 0) + time_val
        eta4_after = sum(combined_exec.values())
        eb_after = 0.0
        if eta4_after > 0:
            for exec_name, (min_frac, _max_frac) in executive_quotas_frac.items():
                frac_after = combined_exec.get(exec_name, 0) / eta4_after
                eb_after += (min_frac - frac_after) ** 2

        obj4 = eb_before - eb_after
    else:
        # No cumulative offsets: use total exec time as denominator
        eta4 = sum(exec_time_used.values())
        if eta4 <= 0:
            eta4 = 1.0
        eb_sq_penalty = 0.0
        for exec_name, (min_frac, _max_frac) in executive_quotas_frac.items():
            time_used = exec_time_used.get(exec_name, 0)
            fraction_i = time_used / eta4
            shortfall = min_frac - fraction_i
            eb_sq_penalty += shortfall ** 2
        obj4 = -eb_sq_penalty

    score = (
        weights.get('obs_completion', 0) * obj1 +
        weights.get('proj_completion', 0) * obj2 +
        weights.get('utilization', 0) * obj3 +
        weights.get('eb_penalty', 0) * obj4 +
        weights.get('adherence', 0) * 0  # no adherence in scoring
    )

    return score

def print_detailed_schedule(
    schedule_log: List[str],
    all_jobs: List[Dict[str, Any]],
    max_lines: int = 25
):
    """
    Parses and prints a schedule log in a detailed, chronological, and readable format.

    Args:
        schedule_log (List[str]): The list of "job_id@time" strings.
        all_jobs (List[Dict[str, Any]]): The master list of all jobs to look up details.
        max_lines (int): The maximum number of schedule entries to print to avoid flooding the console.
                         Set to None to print everything.
    """
    if not schedule_log:
        print("  Schedule is empty.")
        return

    # Create a quick lookup map for job details
    job_lookup = {job['job_id']: job for job in all_jobs}

    # 1. Parse the schedule log into a more useful structure
    parsed_schedule = []
    for entry in schedule_log:
        try:
            job_id, start_time_str = entry.split('@')
            start_time = int(start_time_str)
            parsed_schedule.append({'job_id': job_id, 'start_time': start_time})
        except (ValueError, IndexError):
            print(f"  Warning: Could not parse schedule entry: {entry}")
            continue

    # 2. Sort the schedule chronologically by start time
    parsed_schedule.sort(key=lambda x: x['start_time'])

    # 3. Print the formatted header
    print("\n  --- Schedule Details (chronological) ---")
    header = (
        f"{'Start':<8} | {'End':<8} | {'Duration':<8} | {'Job ID':<25} | "
        f"{'Grade':<5} | {'Rank':<5} | {'Project':<15} | {'Executive'}"
    )
    print(header)
    print("-" * len(header))

    last_end_time = 0
    lines_printed = 0

    # 4. Iterate and print each entry with details
    for entry in parsed_schedule:
        if max_lines is not None and lines_printed >= max_lines:
            print(f"  ... (omitting remaining {len(parsed_schedule) - lines_printed} entries) ...")
            break

        start_time = entry['start_time']
        job_id = entry['job_id']
        job_details = job_lookup.get(job_id)

        if not job_details:
            print(f"  Warning: Job ID '{job_id}' not found in job master list.")
            continue

        # Check for and print idle time
        idle_time = start_time - last_end_time
        if idle_time > 0:
            idle_header = f"{last_end_time:<8} | {start_time:<8} | {f'({idle_time})':<8} | "
            print(f"{idle_header}{'--- IDLE TIME ---':<25}")

        # Gather job details for printing
        duration = job_details.get('length', 'N/A')
        end_time = start_time + duration
        grade = job_details.get('grade', '?')
        weight = job_details.get('weight', 'N/A')
        project_code = job_details.get('project_id', 'N/A')
        executive = job_details.get('executive', 'N/A')

        # Handle fractional executives for cleaner printing
        if isinstance(executive, dict):
            exec_str = ', '.join([f"{k}:{v*100:.0f}%" for k, v in executive.items()])
        else:
            exec_str = str(executive)

        # Format and print the line for the scheduled job
        line = (
            f"{start_time:<8} | {end_time:<8} | {duration:<8} | {job_id:<25} | "
            f"{grade:<5} | {weight:<5} | {project_code:<15} | {exec_str}"
        )
        print(line)

        last_end_time = end_time
        lines_printed += 1

    print("-" * len(header))