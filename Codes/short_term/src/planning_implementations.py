from gurobipy import Model, GRB, quicksum, Env

import concurrent.futures
import inspect
import os

from fixed_params import *
import numpy as np
import pandas as pd
import random
from typing import Callable, List, Dict, Tuple, Optional, Any, Set
from collections import Counter
from copy import deepcopy
import math
from collections import defaultdict
from bisect import bisect_right
import gurobipy
import time
from datetime import datetime

from weather_forecast_layout import get_rms_forecast_slice, get_rms_forecast_value

from forecast import calculate_forecast_statistics, sample_weather_path_from_forecast


def _resolve_forecast_metadata_for_time(
        job: Dict[str, Any],
        current_time: int,
) -> Tuple[List[int], Dict[int, float], float, Optional[int]]:
    issue_times = job.get("forecast_issue_times") or []
    if issue_times:
        issue_pos = bisect_right(issue_times, current_time) - 1
        if issue_pos >= 0:
            issue_idx = issue_times[issue_pos]
            return (
                job.get("forecast_available_by_issue", {}).get(issue_idx, []),
                job.get("forecast_pwv_thresholds_by_issue", {}).get(issue_idx, {}),
                job.get("forecast_rms_thresholds_by_issue", {}).get(
                    issue_idx,
                    job.get("forecast_rms_threshold", job.get("rms_threshold", -np.inf)),
                ),
                issue_idx,
            )
        return [], {}, job.get("forecast_rms_threshold", job.get("rms_threshold", -np.inf)), None

    return (
        job.get("forecast_available", job.get("available", [])),
        job.get("forecast_pwv_thresholds", job.get("pwv_thresholds", {})),
        job.get("forecast_rms_threshold", job.get("rms_threshold", -np.inf)),
        None,
    )


def _all_forecast_available_times(job: Dict[str, Any]) -> List[int]:
    issue_times = job.get("forecast_issue_times") or []
    if issue_times:
        all_times = set()
        for issue_idx in issue_times:
            all_times.update(job.get("forecast_available_by_issue", {}).get(issue_idx, []))
        return sorted(all_times)
    return sorted(set(job.get("forecast_available", [])) | set(job.get("available", [])))


def _compute_eta1_topk(jobs, n_prime, execs_key='remaining_execs'):
    """η1 = sum of the top min(B, n') observation weights.

    Each remaining execution of a job contributes one copy of that job's weight
    to a pool of size B.  We sort descending and sum the top n' entries.
    """
    weights = []
    for j in jobs:
        # Clamp negative weights to 0 so a negative-grade weight cannot make
        # the denominator negative (or smaller). The job weight itself is
        # unaffected; this only changes the normalization constant.
        w = max(0.0, j.get('weight', 0))
        rem = j.get(execs_key, j.get('total_execs', 1))
        weights.extend([w] * max(0, rem))
    if not weights:
        return 1.0
    weights.sort(reverse=True)
    top_k = min(len(weights), max(1, n_prime))
    return max(sum(weights[:top_k]), 1e-12)


def _compute_eta2_topk(projects, n_prime):
    """η2 = sum of the top min(P, n') project weights."""
    # Clamp negative weights to 0 so a negative-grade weight cannot make the
    # denominator negative. Project weights themselves are unaffected.
    weights = sorted((max(0.0, p.get('weight', 0)) for p in projects), reverse=True)
    if not weights:
        return 1.0
    top_k = min(len(weights), max(1, n_prime))
    return max(sum(weights[:top_k]), 1e-12)


def _compute_geometric_project_bonus(
        project: Dict[str, Any],
        job_id: str,
        completed_exec_counts: Counter,
        job_info_map: Dict[str, Dict],
        eta2: float,
        project_bonus_ramp_ratio: float,
) -> float:
    """Reward the k-th completed job in a project using a geometric split."""
    if eta2 <= 0:
        return 0.0

    project_job_ids = [sb_id for sb_id in project.get('job_ids', []) if sb_id in job_info_map]
    if not project_job_ids or job_id not in project_job_ids:
        return 0.0

    total_execs_for_job = job_info_map[job_id].get('total_execs', 1)
    completed_before = completed_exec_counts.get(job_id, 0)
    if completed_before >= total_execs_for_job:
        return 0.0
    if completed_before + 1 < total_execs_for_job:
        return 0.0

    completed_jobs_after = 0
    for sb_id in project_job_ids:
        completed_after = completed_exec_counts.get(sb_id, 0) + (1 if sb_id == job_id else 0)
        if completed_after >= job_info_map[sb_id].get('total_execs', 1):
            completed_jobs_after += 1

    if completed_jobs_after <= 0:
        return 0.0

    total_project_bonus = project.get('weight', 0.0) / eta2
    num_project_jobs = len(project_job_ids)
    if math.isclose(project_bonus_ramp_ratio, 1.0):
        base_bonus = total_project_bonus / num_project_jobs
    else:
        geometric_sum = sum(project_bonus_ramp_ratio ** i for i in range(num_project_jobs))
        base_bonus = total_project_bonus / geometric_sum
    return base_bonus * (project_bonus_ramp_ratio ** (completed_jobs_after - 1))


PROPHET_VERBOSE_LOGGING = True  # Set True to enable detailed prophet breakdown (very verbose when used from OSCO/OSCOCSH)


def solve_prophet_configurable(jobs, projects, realized_weather, time_steps,
                                  executive_quotas, weights: dict, priority_job_ids=None, time_limit=60, output_flag=0,
                               force_job_at_t0: bool = False,
                               cumulative_exec_time_offset: Optional[Dict[str, float]] = None,
                               cumulative_observable_time_offset: int = 0,
                               executive_quotas_frac: Optional[Dict[str, Tuple[float, float]]] = None,
                               prophet_only_mode: bool = False,
                               use_quadratic_eb: bool = False,
                               warm_start_schedule: Optional[List[str]] = None,
                               validation_schedule: Optional[List[str]] = None,
                               use_forecast_availability: bool = False):
    """
    Solve the prophet (perfect-foresight) scheduling problem using the objective from Section 2 of the paper.
    
    The objective is: obj(π) = α1·obj1 + α2·obj2 + α3·obj3 + α4·obj4
    
    Where:
    - obj1 (weighted observation completion): (1/η1) Σ wρb · πb,t
      η1 = min(B, n') · max_b(wρb)
    - obj2 (weighted project completion): (1/η2) Σ wp · zp
      η2 = min(P, n') · max_p(wp)
    - obj3 (utilization): (1/n') Σ ℓb · πb,t
    - obj4 (executive balance): -Σ (ei - fraction_i)²
      where fraction_i = (exec_time_i) / (total_exec_time)
    
    The weights dict maps:
    - 'obs_completion' → α1 (weighted observation completion)
    - 'proj_completion' → α2 (weighted project completion)
    - 'utilization' → α3 (utilization rate)
    - 'eb_penalty' → α4 (executive balance - applied to squared deviation)
    - 'adherence' → optional bonus for following strategic schedule (not in paper's Section 2)
    
    For backwards compatibility, also supports legacy grade-based weights:
    - 'sb_A', 'sb_B', 'sb_C' → observation completion by grade
    - 'proj_A', 'proj_B', 'proj_C' → project completion by grade
    
    Args:
        cumulative_exec_time_offset: Cumulative exec time from previous periods (V^(c-1)).
            When provided, EB calculations use cumulative time as a base, making EB penalties
            smaller as the cycle progresses.
        cumulative_observable_time_offset: Cumulative observable (non-NaN) time bins from 
            previous periods. Used as base for EB denominator calculations.
    """
    # Store cumulative offsets
    if cumulative_exec_time_offset is None:
        cumulative_exec_time_offset = {}
    _cumulative_exec_offset = dict(cumulative_exec_time_offset)
    _cumulative_obs_offset = cumulative_observable_time_offset

    job_lookup = {j['job_id']: j for j in jobs}

    # Handle priority_job_ids: accept Counter, list, set, or None
    if priority_job_ids is None:
        priority_job_ids_counter = Counter()
    elif isinstance(priority_job_ids, Counter):
        priority_job_ids_counter = priority_job_ids
    else:
        # Convert list/set to Counter (each item counted once)
        priority_job_ids_counter = Counter(priority_job_ids)

    model = None
    max_retries = 10  # Try up to 10 times before giving up
    base_delay = 1.0  # Start with a 1-second delay

    for attempt in range(max_retries):
        try:
            # This is the line that can fail due to license issues
            model = Model("prophet")
            # If we successfully create the model, break out of the retry loop
            break
        except gurobipy.GurobiError as e:
            # Check if it's the specific license error we expect
            if "Failed to connect to token server" in str(e):
                if attempt < max_retries - 1:
                    # Use exponential backoff with jitter:
                    # wait a bit longer each time, with randomness to avoid all processes retrying at once
                    delay = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(
                        f"Gurobi license error (Attempt {attempt + 1}/{max_retries}). Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    # If we've exhausted all retries, re-raise the exception
                    print("Gurobi license error: All retries failed. Raising exception.")
                    raise e
            else:
                # If it's a different Gurobi error, we should fail immediately
                raise e

    # If the model is still None after the loop, something went wrong.
    if model is None:
        raise RuntimeError("Failed to initialize Gurobi model after all retries.")

    # --- Calculate n' (total available time for scientific observations) ---
    n_prime_period = 0
    for t in range(time_steps):
        if not (np.isnan(realized_weather[t][0]) or np.isnan(realized_weather[t][1])):
            n_prime_period += 1

    if n_prime_period <= 0:
        return 0.0, []

    # Use cumulative n' for normalization (eta1, eta2, eta3) so that each observation/project
    # contributes the same amount to the objective regardless of when it happens in the cycle.
    n_prime_cumulative = _cumulative_obs_offset + n_prime_period
    
    # --- Calculate normalization constants (Section 2 of paper) ---
    # η1 = sum of top min(B, n') observation weights (one per remaining exec)
    eta1 = _compute_eta1_topk(jobs, n_prime_cumulative)
    # η2 = sum of top min(P, n') project weights
    eta2 = _compute_eta2_topk(projects, n_prime_cumulative)
    # η3 = n'_cumulative (used directly as denominator for utilization)
    eta3 = n_prime_cumulative

    # --- Create decision variables ---
    def _fc_avail(job):
        if use_forecast_availability:
            return job.get("forecast_available", job.get("available", []))
        return job.get("available", [])

    def _fc_pwv(job):
        if use_forecast_availability:
            return job.get("forecast_pwv_thresholds", job.get("pwv_thresholds", {}))
        return job.get("pwv_thresholds", {})

    def _fc_rms(job):
        if use_forecast_availability:
            return job.get("forecast_rms_threshold", job.get("rms_threshold", 1000))
        return job.get("rms_threshold", 1000)

    x = {}
    n_weather_ok = sum(1 for t in range(time_steps) if not (pd.isna(realized_weather.get(t, (np.nan, np.nan))[0]) or pd.isna(realized_weather.get(t, (np.nan, np.nan))[1])))
    for job in jobs:
        for t in _fc_avail(job):
            if job["remaining_execs"] == 0:
                continue
            if t + job["length"] > time_steps:
                continue
            feasible = True
            for t_prime in range(t, t + job["length"]):
                if t_prime >= time_steps:
                    feasible = False
                    break
                pwv, rms = realized_weather.get(t_prime, (np.nan, np.nan))

                if pd.isna(pwv) or pd.isna(rms):
                    feasible = False
                    break

                if t_prime == t:
                    pwv_thresh = _fc_pwv(job).get(t_prime, np.inf)
                    rms_thresh = _fc_rms(job)
                    if pwv > pwv_thresh or rms < rms_thresh:
                        feasible = False
                        break
            if feasible:
                if use_quadratic_eb:
                    x[job["job_id"], t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{job['job_id']}_{t}", lb=0, ub=1)
                else:
                    x[job["job_id"], t] = model.addVar(vtype=GRB.BINARY, name=f"x_{job['job_id']}_{t}")

    n_vars = len(x)
    n_jobs_with_vars = len(set(k[0] for k in x))
    n_jobs_with_available = sum(1 for j in jobs if _fc_avail(j))
    total_available_slots = sum(len(_fc_avail(j)) for j in jobs)
    if PROPHET_VERBOSE_LOGGING:
        print(f"\n  [PROPHET] time_steps={time_steps}, observable_weather_slots={n_weather_ok}")
        print(f"  [PROPHET] Jobs with non-empty 'available': {n_jobs_with_available}/{len(jobs)}, total (job,t) slots: {total_available_slots}")
        print(f"  [PROPHET] Decision variables created: {n_vars} (jobs with ≥1 feasible start: {n_jobs_with_vars})", flush=True)
        if n_vars == 0:
            print(f"  [PROPHET] WARNING: No decision variables. Check: job['available'] within [0, time_steps), weather non-NaN, pwv≤threshold, rms≥threshold.", flush=True)
        elif n_vars < 10 and n_jobs_with_vars <= 3:
            print(f"  [PROPHET] WARNING: Very few feasible (job, start) pairs. Prophet can only schedule up to {n_vars} executions.", flush=True)

    model.update()

    # --- Apply warm-start hints from a previous schedule ---
    if warm_start_schedule:
        warm_start_set = set()
        for entry in warm_start_schedule:
            try:
                jid, tstr = entry.split("@")
                warm_start_set.add((jid, int(tstr)))
            except (ValueError, TypeError):
                continue
        n_hints = 0
        for key, var in x.items():
            if key in warm_start_set:
                var.Start = 1.0
                n_hints += 1
            else:
                var.Start = 0.0
        if PROPHET_VERBOSE_LOGGING:
            print(f"  [PROPHET] Warm-start: {n_hints} variables set to 1 out of {len(warm_start_schedule)} schedule entries", flush=True)

    # --- Add constraint: force a job at t=0 if requested ---
    if force_job_at_t0:
        jobs_runnable_at_t0 = [job for job in jobs if (job['job_id'], 0) in x]
        if jobs_runnable_at_t0:
            model.addConstr(
                quicksum(x.get((job["job_id"], 0), 0) for job in jobs_runnable_at_t0) == 1,
                name="force_job_at_t0"
            )

    print("Setting up constraints", flush=True)
    # --- Constraint: each observation scheduled at most remaining_execs times ---
    for job in jobs:
        model.addConstr(quicksum(x.get((job["job_id"], t), 0) for t in _fc_avail(job)) <= job['remaining_execs'])

    # --- Constraint: no overlapping observations ---
    for t in range(time_steps):
        model.addConstr(
            quicksum(
                x.get((job["job_id"], s), 0)
                for job in jobs
                for s in _fc_avail(job)
                if s <= t < s + job["length"]
            ) <= 1
        )
    print("Constraints set up", flush=True)

    # --- Optional schedule validation against prophet hard constraints ---
    if validation_schedule:
        parsed = []
        parse_errors = []
        for entry in validation_schedule:
            try:
                jid, tstr = entry.split("@")
                t0 = int(tstr)
                parsed.append((jid, t0))
            except (ValueError, TypeError):
                parse_errors.append(entry)

        missing_vars = [(jid, t0) for (jid, t0) in parsed if (jid, t0) not in x]

        # Check per-job execution count constraints.
        count_by_job = Counter(jid for jid, _ in parsed if (jid, _) in x)
        job_cap_violations = []
        for job in jobs:
            jid = job["job_id"]
            req = job.get("remaining_execs", 0)
            got = count_by_job.get(jid, 0)
            if got > req:
                job_cap_violations.append((jid, got, req))

        # Check non-overlap constraints.
        occupancy = Counter()
        for jid, t0 in parsed:
            if (jid, t0) not in x:
                continue
            j = job_lookup.get(jid)
            if not j:
                continue
            for tt in range(t0, t0 + j.get("length", 0)):
                occupancy[tt] += 1
        overlap_violations = [(tt, c) for tt, c in occupancy.items() if c > 1]

        # Check force_job_at_t0 if enabled.
        force_t0_violation = None
        if force_job_at_t0:
            at_t0 = sum(1 for (jid, t0) in parsed if t0 == 0 and (jid, t0) in x)
            if at_t0 != 1:
                force_t0_violation = at_t0

        print("\n  [PROPHET CANDIDATE SCHEDULE CHECK]")
        print(f"    candidate size={len(validation_schedule)}, parsed={len(parsed)}, parse_errors={len(parse_errors)}")
        print(f"    missing (job,t) decision vars={len(missing_vars)}")
        print(f"    job-cap violations={len(job_cap_violations)}")
        print(f"    overlap violations={len(overlap_violations)}")
        if force_t0_violation is not None:
            print(f"    force_job_at_t0 violation: count_at_t0={force_t0_violation} (expected 1)")

        if parse_errors:
            print("    sample parse errors:", parse_errors[:10])
        if missing_vars:
            print("    sample missing vars:", missing_vars[:10])
        if job_cap_violations:
            print("    sample job-cap violations:", job_cap_violations[:10])
        if overlap_violations:
            print("    sample overlap violations:", overlap_violations[:10])

        is_feasible = (
            len(parse_errors) == 0 and
            len(missing_vars) == 0 and
            len(job_cap_violations) == 0 and
            len(overlap_violations) == 0 and
            force_t0_violation is None
        )
        print(f"    candidate schedule satisfies hard constraints: {is_feasible}\n")

    # --- Calculate executive time variables for EB objective ---
    exec_time_vars = {}
    for exec_name in executive_quotas.keys():
        time_from_single = quicksum(
            job["length"] * x.get((job["job_id"], t), 0)
            for job in jobs
            if isinstance(job["executive"], str) and job["executive"] == exec_name
            for t in _fc_avail(job)
        )
        time_from_fractional = quicksum(
            job["length"] * job["executive"][exec_name] * x.get((job["job_id"], t), 0)
            for job in jobs
            if isinstance(job["executive"], dict) and exec_name in job["executive"]
            for t in _fc_avail(job)
        )
        exec_time_vars[exec_name] = time_from_single + time_from_fractional

    # Total utilization (for computing fractions)
    total_utilization = quicksum(job["length"] * x.get((job["job_id"], t), 0) for job in jobs for t in _fc_avail(job))

    # --- Minimum utilization constraint (diagnostic: force scheduling) ---
    # If the model becomes infeasible, a constraint is likely broken. If feasible, objective was favoring under-scheduling.
    # min_utilization_frac = 0.01  # Use at least 1% of observable time
    # min_bins = max(0, int(min_utilization_frac * n_prime_period))
    # if min_bins > 0 and n_vars > 0:
    #     model.addConstr(total_utilization >= min_bins, name="min_utilization")
    #     if PROPHET_VERBOSE_LOGGING:
    #         print(f"  [PROPHET] Min utilization constraint: total_utilization >= {min_bins} bins ({min_utilization_frac*100:.0f}% of {n_prime_period} observable)", flush=True)

    # --- Project completion indicator variables ---
    print("Setting up project completion indicator variables", flush=True)
    zs = {}
    for project in projects:
        z = model.addVar(vtype=GRB.BINARY, name=f"z_{project['project_id']}")
        zs[project['project_id']] = z

        # Constraint: z == 1 only if all jobs in project are fully scheduled
        total_required = 0
        job_scheduled_counts = []
        for jid in project['job_ids']:
            if jid in job_lookup:
                job = job_lookup[jid]
                total_required += job['remaining_execs']
                scheduled_count = quicksum(x.get((jid, t), 0) for t in _fc_avail(job))
                job_scheduled_counts.append(scheduled_count)

        if job_scheduled_counts:
            sum_project_sched = quicksum(job_scheduled_counts)
            model.addConstr(
                sum_project_sched >= total_required * z,
                name=f"project_bonus_constr_{project['project_id']}"
            )
            # Reverse implication: if the project reaches full required executions,
            # force z to 1 (not just allow it). With per-job caps, hitting
            # total_required means all jobs reached their required counts.
            model.addConstr(
                z >= sum_project_sched - (total_required - 1) - 1e-4,
                name=f"project_bonus_force_{project['project_id']}"
            )
        else:
            model.addConstr(z == 0)

    # --- Define Objective Components (Section 2 of paper) ---
    w = weights  # for brevity
    print("Defining objective components", flush=True)
    # Bonus for scheduling at t=0 (encourage high-weight jobs earlier in mean/OSCO)
    T0_SLOT_BONUS = 1.0

    print("Calculating weighted observation completion", flush=True)
    # obj1: Weighted observation completion = (1/η1) Σ w_ρb · πb,t (t=0 slot gets T0_SLOT_BONUS multiplier)
    weighted_obs_completion = quicksum(
        job.get('weight', 0) * (T0_SLOT_BONUS if t == 0 else 1.0) * x.get((job['job_id'], t), 0)
        for job in jobs
        for t in _fc_avail(job)
    )
    obj1 = weighted_obs_completion / eta1 if eta1 > 0 else 0

    print("Calculating weighted project completion", flush=True)
    # obj2: Weighted project completion = (1/η2) Σ wp · zp
    weighted_proj_completion = quicksum(
        p.get('weight', 0) * zs[p['project_id']]
        for p in projects
    )
    obj2 = weighted_proj_completion / eta2 if eta2 > 0 else 0

    print("Calculating utilization", flush=True)
    # obj3: Utilization = (1/η3) Σ ℓb · πb,t
    obj3 = total_utilization / eta3 if eta3 > 0 else 0

    _eb_frac_quotas = executive_quotas_frac if executive_quotas_frac is not None else executive_quotas

    # S = actual total exec time (decision variable). Fractions are exec_i / S in
    # evaluator space, but optimizer-space normalization uses S_est (constant) so
    # piecewise mode stays MILP and quadratic mode stays convex MIQP.
    total_offset = sum(_cumulative_exec_offset.values())
    S_est = max(total_offset + n_prime_period, 1)

    # Auxiliary variables for per-executive cumul time and total time.
    # Using Gurobi variables (not raw LinExprs) keeps quadratic terms O(#executives)
    # instead of O(#decision_vars²).
    E_aux = {}
    for exec_name in _eb_frac_quotas.keys():
        exec_offset = _cumulative_exec_offset.get(exec_name, 0)
        period_exec = exec_time_vars.get(exec_name, 0)
        E_aux[exec_name] = model.addVar(name=f"E_{exec_name}", lb=0)
        model.addConstr(E_aux[exec_name] == exec_offset + period_exec,
                        name=f"E_def_{exec_name}")
    S_var = model.addVar(name="S_total_exec", lb=0)
    model.addConstr(S_var == total_offset + total_utilization, name="S_def")
    model.update()

    # EB_before: constant penalty from fractions achieved BEFORE this period
    eb_before = 0.0
    if total_offset > 0:
        for exec_name, (target_frac, _) in _eb_frac_quotas.items():
            frac_before = _cumulative_exec_offset.get(exec_name, 0) / total_offset
            eb_before += (target_frac - frac_before) ** 2
    else:
        eb_before = sum(tf ** 2 for tf, _ in _eb_frac_quotas.values())

    # EB_after in optimizer-space:
    #   shortfall_i(opt) = target_i - E_i / S_est
    # where E_i = exec_offset_i + period_exec_i (linear), S_est constant.
    # This gives:
    #   - quadratic mode: exact Σ shortfall_i(opt)^2
    #   - piecewise mode: linear surrogate of the same quantity
    eb_after_expr = 0
    shortfall = {}
    abs_shortfall = {}
    z_squared_approx = {}

    if use_quadratic_eb:
        # --- Clean quadratic penalty (MIQP) ---
        print("  [PROPHET] Using QUADRATIC EB penalty (MIQP)", flush=True)
        for exec_name, (target_frac, _) in _eb_frac_quotas.items():
            shortfall[exec_name] = model.addVar(name=f"shortfall_{exec_name}", lb=-1.0, ub=1.0)
            # S_est * shortfall = target_frac * S_est - E_i
            model.addConstr(
                E_aux[exec_name] + S_est * shortfall[exec_name] == target_frac * S_est,
                name=f"shortfall_def_{exec_name}"
            )
        model.update()
        eb_after_expr = quicksum(shortfall[o] * shortfall[o] for o in _eb_frac_quotas.keys())
    else:
        # --- Piecewise-linear approximation (MILP) of the SAME shortfall ---
        print("  [PROPHET] Using PIECEWISE LINEAR EB penalty (MILP)", flush=True)
        for exec_name in _eb_frac_quotas.keys():
            shortfall[exec_name] = model.addVar(name=f"shortfall_{exec_name}", lb=-1.0, ub=1.0)
        model.update()

        for exec_name, (target_frac, _) in _eb_frac_quotas.items():
            model.addConstr(
                E_aux[exec_name] + S_var * shortfall[exec_name] == target_frac * S_var,
                name=f"shortfall_def_{exec_name}"
            )

        shortfall_pos = {}
        shortfall_neg = {}
        for exec_name in _eb_frac_quotas.keys():
            shortfall_pos[exec_name] = model.addVar(name=f"shortfall_pos_{exec_name}", lb=0)
            shortfall_neg[exec_name] = model.addVar(name=f"shortfall_neg_{exec_name}", lb=0)
            abs_shortfall[exec_name] = model.addVar(name=f"abs_shortfall_{exec_name}", lb=0)
        model.update()

        for exec_name in _eb_frac_quotas.keys():
            model.addConstr(shortfall[exec_name] == shortfall_pos[exec_name] - shortfall_neg[exec_name],
                           name=f"shortfall_split_{exec_name}")
            model.addConstr(abs_shortfall[exec_name] == shortfall_pos[exec_name] + shortfall_neg[exec_name],
                           name=f"abs_shortfall_{exec_name}")

        breakpoints = [0.0, 0.001, 0.003, 0.005, 0.008, 0.012, 0.02, 0.035, 0.05, 0.08, 0.12, 0.20, 0.40, 0.9]
        for exec_name in _eb_frac_quotas.keys():
            z_squared_approx[exec_name] = model.addVar(name=f"z_squared_{exec_name}", lb=0)
        model.update()

        for exec_name in _eb_frac_quotas.keys():
            for a in breakpoints:
                if a > 0:
                    model.addConstr(z_squared_approx[exec_name] >= 2 * a * abs_shortfall[exec_name] - a * a,
                                   name=f"tangent_eb_{exec_name}_{a}")

        eb_after_expr = quicksum(z_squared_approx[o] for o in _eb_frac_quotas.keys())

    if prophet_only_mode:
        obj4 = -eb_after_expr
    else:
        obj4 = eb_before - eb_after_expr
    print("Calculating adherence", flush=True)
    # Adherence to strategic plan (optional, not in paper's Section 2)
    adherence_contrib = {}
    for job_id, max_count in priority_job_ids_counter.items():
        if job_id in job_lookup:
            scheduled_count = quicksum(x.get((job_id, t), 0) for t in _fc_avail(job_lookup[job_id]))
            adherence_contrib[job_id] = model.addVar(vtype=GRB.CONTINUOUS, name=f"adherence_cap_{job_id}", lb=0, ub=max_count)
            model.addConstr(adherence_contrib[job_id] <= scheduled_count, name=f"adherence_cap_scheduled_{job_id}")
            model.addConstr(adherence_contrib[job_id] <= max_count, name=f"adherence_cap_max_{job_id}")
    
    if adherence_contrib:
        model.update()
    
    # Use cumulative n' for adherence normalization (consistency with eta1/2/3)
    adherence_score = quicksum(adherence_contrib.values()) / n_prime_cumulative if adherence_contrib else 0
    print("Calculating adherence score", flush=True)    
    # Paper's Section 2 objective: obj = α1·obj1 + α2·obj2 + α3·obj3 + α4·obj4
    # obj4 = -Σ dev_i² / S_est² (convex quadratic EB penalty)
    print("Running paper objective", flush=True)
    print("weights are: ", w, flush=True)
    model.setObjective(
        w.get('obs_completion', 0) * obj1 +
        w.get('proj_completion', 0) * obj2 +
        w.get('utilization', 0) * obj3 +
        w.get('eb_penalty', 0) * obj4 +
        w.get('adherence', 0) * adherence_score,
        GRB.MAXIMIZE
    )
    
    # Log lightweight structural debug info only (pre-opt expressions are symbolic).
    print("Logging information for debugging", flush=True)
    print(
        f"Objective terms are symbolic pre-opt: obj1={type(obj1).__name__}, "
        f"obj2={type(obj2).__name__}, obj3={type(obj3).__name__}, "
        f"obj4={type(obj4).__name__}, adherence={type(adherence_score).__name__}",
        flush=True
    )
    print(f"w={w}", flush=True)
    print(f"|x|={len(x)}, |zs|={len(zs)}, |E_aux|={len(E_aux)}", flush=True)
    print(f"S_est={S_est:.0f}, total_offset={total_offset:.0f}", flush=True)
    print(f"n_prime_period={n_prime_period:.0f}, n_prime_cumulative={n_prime_cumulative:.0f}", flush=True)
    print(f"eta1={eta1:.4f}, eta2={eta2:.4f}, eta3={eta3:.4f}", flush=True)
    print(f"time_steps={time_steps}, force_job_at_t0={force_job_at_t0}", flush=True)
    print(f"time_limit={time_limit}, output_flag={output_flag}", flush=True)
    print(f"prophet_only_mode={prophet_only_mode}", flush=True)
    print(f"use_quadratic_eb={use_quadratic_eb}", flush=True)
    print(f"adherence_contrib={adherence_contrib}", flush=True)
    print(f"adherence_score={adherence_score:.6f}", flush=True)
    print(f"priority_job_ids_counter={priority_job_ids_counter}", flush=True)
    print(f"job_lookup={job_lookup}", flush=True)
    print(f"exec_time_vars={exec_time_vars}", flush=True)
    print(f"exec_offset={exec_offset}", flush=True)

    print("Setting model parameters", flush=True)
    model.setParam('OutputFlag', output_flag)
    # Tight MIP gap so Prophet does not stop after a trivial 1-job solution
    model.setParam("MIPGap", .01)
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("NumericFocus", 1)
    model.setParam("TimeLimit", time_limit)
    model.optimize()
    print(model.printQuality())
    # print([z.X for z in zs])

    if output_flag:
        print(f"model.status: {model.status}")
    if model.status == GRB.INFEASIBLE:
        print(f"  [PROPHET] Model INFEASIBLE. If min_utilization is on, try lowering min_utilization_frac or check constraints.", flush=True)
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        schedule = [f"{k[0]}@{k[1]}" for k, var in x.items() if var.X > 0.5]
        total_value = model.objVal
        
        # --- Detailed objective component breakdown (disabled by default - set PROPHET_VERBOSE_LOGGING=True to enable) ---
        if PROPHET_VERBOSE_LOGGING:
            print(f"\n  [PROPHET OBJECTIVE BREAKDOWN]")
            print(f"    n_prime_period={n_prime_period}, cumulative_obs_offset={_cumulative_obs_offset}, n_prime_cumulative={n_prime_cumulative}")
            print(f"    eta1={eta1:.4f}, eta2={eta2:.4f}, eta3={eta3}")
            print(f"    time_steps={time_steps}, force_job_at_t0={force_job_at_t0}")
            print(f"    Weights dict: {w}", flush=True)

            # Count scheduled jobs per grade
            grade_counts = {'A': 0, 'B': 0, 'C': 0, '?': 0}
            grade_weight_sums = {'A': 0.0, 'B': 0.0, 'C': 0.0, '?': 0.0}
            scheduled_jobs_detail = []
            for k_var, var in x.items():
                if var.X > 0.5:
                    jid, t_sched = k_var
                    j = job_lookup.get(jid, {})
                    g = j.get('grade', '?')
                    grade_counts[g] = grade_counts.get(g, 0) + 1
                    grade_weight_sums[g] = grade_weight_sums.get(g, 0) + j.get('weight', 0)
                    scheduled_jobs_detail.append((jid, t_sched, j.get('weight', 0), g, j.get('length', 0)))

            print(f"    Scheduled {len(schedule)} job executions:")
            for g in ['A', 'B', 'C']:
                print(f"      Grade {g}: {grade_counts.get(g,0)} execs, weight_sum={grade_weight_sums.get(g,0):.4f}")

            # Show individual scheduled jobs
            scheduled_jobs_detail.sort(key=lambda x: x[1])
            for jid, t_sched, wgt, g, length in scheduled_jobs_detail[:20]:
                marker = " <<< T=0" if t_sched == 0 else ""
                exec_info = job_lookup.get(jid, {}).get('executive', '?')
                if isinstance(exec_info, dict):
                    exec_str = ",".join(f"{k}:{v:.0%}" for k, v in exec_info.items())
                else:
                    exec_str = str(exec_info)
                print(f"      @t={t_sched:<4} {jid:<30} w={wgt:>8.4f} grade={g} len={length} exec={exec_str}{marker}")

            # Projects completed
            completed_projects = [(p['project_id'], p.get('weight', 0), p.get('grade', '?'))
                                 for p in projects if zs[p['project_id']].X > 0.5]
            print(f"    Completed {len(completed_projects)} projects out of {len(projects)}:")
            for pid, pw, pg in completed_projects[:10]:
                print(f"      {pid:<30} w={pw:.4f} grade={pg}")

            # z-variable consistency diagnostic: compare z-project completion against
            # schedule-derived completion using remaining_execs requirements.
            # Also check the actual sum of the variables in the project.
            project_sums = {}
            for p in projects:
                pid = p.get('project_id', '')
                p_job_ids = p.get('job_ids', [])
                if not p_job_ids:
                    continue
                project_sums[pid] = 0.0
                for jid in p_job_ids:
                    for t in _fc_avail(job_lookup.get(jid, {})):
                        project_sums[pid] += x.get((jid, t), 0)

            scheduled_counts = Counter()
            for (jid_k, _t_k), var_k in x.items():
                if var_k.X > 0.5:
                    scheduled_counts[jid_k] += 1

            z_mismatches = []
            for p in projects:
                pid = p.get('project_id', '')
                p_job_ids = p.get('job_ids', [])
                if not p_job_ids:
                    continue

                schedule_complete = True
                for sb_id in p_job_ids:
                    job_info = job_lookup.get(sb_id, {})
                    req = job_info.get('remaining_execs', job_info.get('total_execs', 1))
                    if scheduled_counts.get(sb_id, 0) < req:
                        schedule_complete = False
                        break

                z_complete = zs.get(pid).X > 0.5 if pid in zs else False
                if schedule_complete != z_complete:
                    z_val = zs.get(pid).X if pid in zs else float('nan')
                    z_mismatches.append((pid, z_val, schedule_complete, z_complete))

            print(f"    z-vs-schedule mismatches: {len(z_mismatches)}")
            for pid, z_val, sched_comp, z_comp in z_mismatches[:20]:
                print(
                    f"      {pid:<30} z={z_val:.3f} sum={project_sums.get(pid, 0):.0f} "
                    f"schedule_complete={sched_comp} z_complete={z_comp}"
                )
                
            # Utilization
            util_val = sum(
                job['length'] * x[(job['job_id'], t)].X
                for job in jobs for t in _fc_avail(job)
                if (job['job_id'], t) in x and x[(job['job_id'], t)].X > 0.5
            )
            obj3_val = util_val / eta3 if eta3 > 0 else 0
            print(f"    Utilization: {util_val:.0f} bins used out of n_prime_cumul={n_prime_cumulative} → obj3={obj3_val:.6f}")

            # EB detail: optimizer-space vs evaluator-space
            _log_eb_frac = _eb_frac_quotas
            print(f"    Executive balance detail:")

            exec_period_vals = {}
            for exec_name in _log_eb_frac.keys():
                exec_period_var = exec_time_vars.get(exec_name, 0)
                try:
                    exec_period_vals[exec_name] = exec_period_var.getValue()
                except Exception:
                    exec_period_vals[exec_name] = 0

            S_actual = total_offset + util_val
            print(f"      S_est  = {S_est:.0f}, S_actual (total exec time) = {S_actual:.0f}")
            print(f"      S_var (optimizer denom) = {S_var.X:.0f}")

            eb_before_log = 0.0
            eb_after_opt = 0.0   # optimizer-space (S_est denominator)
            eb_after_eval = 0.0  # evaluator-space (S_actual denominator)
            piecewise_gaps = []
            for exec_name, (target_frac, _) in _log_eb_frac.items():
                exec_offset = _cumulative_exec_offset.get(exec_name, 0)
                exec_period = exec_period_vals.get(exec_name, 0)

                frac_before = exec_offset / total_offset if total_offset > 0 else 0.0
                sq_before = (target_frac - frac_before) ** 2
                eb_before_log += sq_before

                cumul = exec_offset + exec_period
                frac_opt = cumul / S_var.X if S_var.X > 0 else 0.0
                frac_eval = cumul / S_actual if S_actual > 0 else 0.0
                sq_opt = (target_frac - frac_opt) ** 2
                sq_eval = (target_frac - frac_eval) ** 2
                eb_after_opt += sq_opt
                eb_after_eval += sq_eval

                print(f"      {exec_name}: target={target_frac:.4f}, offset={exec_offset:.0f}, period={exec_period:.0f}")
                print(f"        frac(S_var)={frac_opt:.4f} sq={sq_opt:.6f} | frac(S_actual)={frac_eval:.4f} sq={sq_eval:.6f}")

                # Piecewise approximation diagnostic (true sq vs approximated z)
                if (not use_quadratic_eb) and exec_name in z_squared_approx and exec_name in shortfall:
                    try:
                        z_val = z_squared_approx[exec_name].X
                        sh_val = shortfall[exec_name].X
                        true_sq = sh_val * sh_val
                        piecewise_gaps.append(true_sq - z_val)
                    except Exception:
                        pass

            if prophet_only_mode:
                obj4_val = -eb_after_opt
                print(f"    EB (optimizer view, S_var): -{eb_after_opt:.6f} = obj4={obj4_val:+.6f}")
                print(f"    EB (evaluator view, S_actual): -{eb_after_eval:.6f}")
            else:
                obj4_val = eb_before_log - eb_after_opt
                print(f"    EB marginal (optimizer, S_est): {eb_before_log:.6f} - {eb_after_opt:.6f} = obj4={obj4_val:+.6f}")
                print(f"    EB marginal (evaluator, S_actual): {eb_before_log:.6f} - {eb_after_eval:.6f} = {eb_before_log - eb_after_eval:+.6f}")

            if (not use_quadratic_eb) and piecewise_gaps:
                max_gap = max(piecewise_gaps)
                avg_gap = sum(piecewise_gaps) / len(piecewise_gaps)
                print(f"    Piecewise EB gap (true_sq - approx_z): max={max_gap:.6e}, avg={avg_gap:.6e}")
            # Adherence
            adh_val = 0
            if adherence_contrib and n_prime_cumulative > 0:
                adh_val = sum(v.X for v in adherence_contrib.values()) / n_prime_cumulative
                
            print(f"    [PAPER OBJECTIVE PATH]")
            print(f"    eta1={eta1:.4f}, eta2={eta2:.4f}, eta3={eta3:.4f}")
            obs_val = sum(job.get('weight', 0) * x[(job['job_id'], t)].X
                            for job in jobs for t in _fc_avail(job)
                            if (job['job_id'], t) in x and x[(job['job_id'], t)].X > 0.5)
            proj_val = sum(p.get('weight', 0) * zs[p['project_id']].X
                            for p in projects if zs[p['project_id']].X > 0.5)
            obj1_val = obs_val / eta1 if eta1 > 0 else 0
            obj2_val = proj_val / eta2 if eta2 > 0 else 0
            a1 = w.get('obs_completion', 0)
            a2 = w.get('proj_completion', 0)
            a3 = w.get('utilization', 0)
            a4 = w.get('eb_penalty', 0)
            a5 = w.get('adherence', 0)
            computed = a1*obj1_val + a2*obj2_val + a3*obj3_val + a4*obj4_val + a5*adh_val
            print(f"    obj1(obs)={obj1_val:.6f}, obj2(proj)={obj2_val:.6f}, obj3(util)={obj3_val:.6f}, obj4(eb)={obj4_val:.6f}, adh={adh_val:.6f}")
            print(f"    a1={a1}*{obj1_val:.6f}={a1*obj1_val:.6f}, a2={a2}*{obj2_val:.6f}={a2*obj2_val:.6f}, a3={a3}*{obj3_val:.6f}={a3*obj3_val:.6f}, a4={a4}*{obj4_val:.6f}={a4*obj4_val:.6f}")
            print(f"    Computed={computed:.6f}, Gurobi objVal={total_value:.6f}")

            # Same-schedule evaluator check for objective parity diagnostics.
            try:
                exec_time_used_sched = {}
                for j in jobs:
                    for t_avail in _fc_avail(j):
                        if (j['job_id'], t_avail) in x and x[(j['job_id'], t_avail)].X > 0.5:
                            lb = j.get('length', 0)
                            if isinstance(j.get('executive'), str):
                                exec_time_used_sched[j['executive']] = exec_time_used_sched.get(j['executive'], 0.0) + lb
                            elif isinstance(j.get('executive'), dict):
                                for en, frac in j['executive'].items():
                                    exec_time_used_sched[en] = exec_time_used_sched.get(en, 0.0) + lb * frac
                evaluator_value = compute_paper_objective_value(
                    successful_schedule_log=schedule,
                    jobs=jobs,
                    projects=projects,
                    exec_time_used=exec_time_used_sched,
                    total_observable_time=n_prime_cumulative,
                    weights=w,
                    executive_quotas_frac=_eb_frac_quotas,
                    verbose=True
                )
                print(f"    Evaluator(same schedule)={evaluator_value:.6f}, diff_vs_gurobi={evaluator_value-total_value:+.6f}")
            except Exception as e:
                print(f"    Evaluator(same schedule) check failed: {e}")

            print(f"  [END PROPHET BREAKDOWN]\n")
        
    else:
        schedule = []
        total_value = -1000.0
    return total_value, schedule


def planning_loop_pulsar(
        jobs: List[Dict[str, Any]],
        projects: List[Dict[str, Any]],
        realized_weather: Dict[int, Any],
        weather_forecasts: Dict[int, Dict[str, np.ndarray]],
        time_steps: int,
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        weights: Dict[str, float],
        idx_to_timestamp: Dict[int, pd.Timestamp],
        cycle_start_timestamp: pd.Timestamp,
        weekly_solver_fn: Callable,
        weekly_solver_kwargs: Optional[Dict] = None,
        replan_every_n_weeks: int = 2,
        sequence_horizon_steps: int = 16,
        oracle_use_actual_job_metadata: bool = False,
        use_realized_pwv_forecast: bool = False,
        use_realized_rms_forecast: bool = False,
        inner_gurobi_time_limit_seconds: float = 10.0,
        max_candidates_per_executive_by_grade: Optional[int] = None,
        fill_to_total_candidates: Optional[int] = None,
        use_quadratic_eb: bool = False,
        gurobi_log_dir: Optional[str] = None,
        osco_num_samples: int = 5,
        osco_n_threads: int = -1,
        osco_random_seed: Optional[int] = None,
        osco_log_sub_timings: bool = False,
        osco_debug: bool = False,
        eb_ramp_exponent: float = 50.0,
        counter_bonus_a_multiplier: float = 100.0,
        counter_bonus_b_multiplier: float = 25.0,
) -> Tuple[float, List[str]]:
    """Weekly strategic cache + counter refresh loop using Gurobi OSCO rollout selection."""
    original_jobs = deepcopy(jobs)
    t = 0
    busy_until = 0
    successful_schedule_log: List[str] = []
    jobs = deepcopy(jobs)
    job_lookup: Dict[str, Dict] = {j['job_id']: j for j in jobs}

    exec_time_used = {k: 0.0 for k in executive_quotas_frac.keys()}
    for job in jobs:
        if isinstance(job.get('executive'), str):
            exec_time_used.setdefault(job['executive'], 0.0)
        elif isinstance(job.get('executive'), dict):
            for en in job['executive']:
                exec_time_used.setdefault(en, 0.0)

    all_unusable_indices = {
        idx for idx, (pwv, rms) in realized_weather.items()
        if pd.isna(pwv) or pd.isna(rms)
    }
    total_observable_time = sum(1 for idx in range(time_steps) if idx not in all_unusable_indices)

    running_weekly_counter = Counter()
    latest_strategic_schedule: List[Dict[str, Any]] = []
    last_replan_week_idx = None
    last_counter_feed_week_idx = None
    weekly_replan_count = 0
    weekly_counter_feed_count = 0
    completed_exec_counts: Counter = Counter()
    rng = np.random.default_rng(osco_random_seed)

    replan_every_n_weeks = max(1, int(replan_every_n_weeks))

    loop_wall_start = time.time()
    cumulative_selector_seconds = 0.0
    selector_call_count = 0
    current_schedule_log: List[str] = []

    def _get_ramped_weights(current_time: int) -> Dict[str, float]:
        return _with_ramped_eb_penalty(
            weights=weights,
            current_time=current_time,
            total_time_steps=time_steps,
            ramp_exponent=eb_ramp_exponent,
        )

    def execute_job(selected_job_id: str, reason: str) -> None:
        nonlocal busy_until
        selected_job = job_lookup[selected_job_id]
        successful_schedule_log.append(f"{selected_job_id}@{t}")
        current_schedule_log.append(f"{selected_job_id}@{t}")
        completed_exec_counts[selected_job_id] += 1
        job_lookup[selected_job_id]['remaining_execs'] -= 1
        busy_until = t + selected_job['length']
        _add_job_time_to_exec_balance(exec_time_used, selected_job)
        if running_weekly_counter.get(selected_job_id, 0) > 0:
            running_weekly_counter[selected_job_id] -= 1
            if running_weekly_counter[selected_job_id] <= 0:
                del running_weekly_counter[selected_job_id]
        print(
            f"  [WEEKLY-OSCO-GUROBI @t={t}] SCHEDULED {selected_job_id} via {reason} "
            f"(len={selected_job['length']}, rem={job_lookup[selected_job_id]['remaining_execs']})",
            flush=True,
        )

    while t < time_steps:
        if t < busy_until:
            t += 1
            continue

        week_idx = _anchored_week_index(idx_to_timestamp[t], cycle_start_timestamp)
        current_week_label = _anchored_week_label(week_idx)
        if last_replan_week_idx is None or (week_idx - last_replan_week_idx) >= replan_every_n_weeks:
            last_replan_week_idx = week_idx
            remaining_jobs = [deepcopy(j) for j in job_lookup.values() if j['remaining_execs'] > 0]
            cumulative_obs_time = sum(1 for i in range(t) if i not in all_unusable_indices)
            solver_kwargs = dict(weekly_solver_kwargs or {})
            solver_kwargs['week_start_date'] = pd.to_datetime(idx_to_timestamp[t], utc=True).normalize().strftime('%Y-%m-%d')
            solver_kwargs['time_limit'] = 20 * 60
            latest_strategic_schedule = weekly_solver_fn(
                jobs=remaining_jobs,
                projects=projects,
                exec_time_used=dict(exec_time_used),
                cumulative_observable_time=cumulative_obs_time,
                weights=weights,
                **solver_kwargs,
            )
            weekly_replan_count += 1
            print(
                f"  Weekly strategic replan for {current_week_label} "
                f"(cadence={replan_every_n_weeks}w): cached {len(latest_strategic_schedule)} planned rows.",
                flush=True,
            )

        if last_counter_feed_week_idx is None or week_idx != last_counter_feed_week_idx:
            current_week_counter = _build_current_week_ab_counter_from_schedule(
                strategic_schedule=latest_strategic_schedule,
                current_week_label=current_week_label,
                job_info_map=job_lookup,
            )
            running_weekly_counter.update(current_week_counter)
            last_counter_feed_week_idx = week_idx
            weekly_counter_feed_count += 1
            print(
                f"  Weekly strategic counter refresh for {current_week_label}: "
                f"added {sum(current_week_counter.values())} A/B planned executions to counter.",
                flush=True,
            )

        if t in all_unusable_indices:
            t += 1
            continue

        remaining_jobs_to_consider = [j for j in job_lookup.values() if j["remaining_execs"] > 0]
        if not remaining_jobs_to_consider:
            break

        observed_pwv_at_t, observed_rms_at_t = realized_weather.get(t, (np.nan, np.nan))
        if pd.isna(observed_pwv_at_t) or pd.isna(observed_rms_at_t):
            t += 1
            continue

        current_forecast_state = weather_forecasts.get(t)
        if current_forecast_state is None and not (use_realized_pwv_forecast and use_realized_rms_forecast):
            print(f"Error: No forecast for t={t}. Ending.")
            break

        _sel_start = time.time()
        ramped_weights = _get_ramped_weights(t)
        selected_job_id, sel_stats = _osco_gurobi_select_job(
            current_time=t,
            job_lookup=job_lookup,
            projects=projects,
            time_steps=time_steps,
            realized_weather=realized_weather,
            weather_forecasts=weather_forecasts,
            all_unusable_indices=all_unusable_indices,
            exec_time_used=exec_time_used,
            completed_exec_counts=completed_exec_counts,
            executive_quotas_frac=executive_quotas_frac,
            weights=ramped_weights,
            cumulative_observable_time=total_observable_time,
            sequence_horizon_steps=sequence_horizon_steps,
            oracle_use_actual_job_metadata=oracle_use_actual_job_metadata,
            use_realized_pwv_forecast=use_realized_pwv_forecast,
            use_realized_rms_forecast=use_realized_rms_forecast,
            max_candidates_per_executive_by_grade=max_candidates_per_executive_by_grade,
            fill_to_total_candidates=fill_to_total_candidates,
            use_quadratic_eb=use_quadratic_eb,
            inner_gurobi_time_limit_seconds=inner_gurobi_time_limit_seconds,
            gurobi_log_dir=gurobi_log_dir,
            num_samples=osco_num_samples,
            n_threads=osco_n_threads,
            rng=rng,
            prophet_remaining_counter=running_weekly_counter,
            counter_bonus_a_multiplier=counter_bonus_a_multiplier,
            counter_bonus_b_multiplier=counter_bonus_b_multiplier,
            log_sub_timings=osco_log_sub_timings,
            debug=osco_debug,
        )
        _sel_end = time.time()
        cumulative_selector_seconds += (_sel_end - _sel_start)
        selector_call_count += 1
        print(
            f"  [WEEKLY-OSCO-GUROBI @t={t}] selection_elapsed={_sel_end - _sel_start:.2f}s, "
            f"candidates={sel_stats.get('candidates_count', 0)}, "
            f"tasks={sel_stats.get('tasks_count', 0)}",
            flush=True,
        )

        if selected_job_id is None:
            t += 1
            continue

        if selected_job_id not in job_lookup or job_lookup[selected_job_id].get("remaining_execs", 0) <= 0:
            t += 1
            continue

        execute_job(selected_job_id, reason="weekly_osco_gurobi")
        t += 1

    if successful_schedule_log:
        final_value = compute_paper_objective_value(
            successful_schedule_log,
            original_jobs,
            projects,
            exec_time_used,
            total_observable_time,
            weights,
            executive_quotas_frac,
        )
    else:
        final_value = 0.0

    loop_wall_elapsed = time.time() - loop_wall_start
    avg_selector_ms = (cumulative_selector_seconds / max(1, selector_call_count)) * 1000.0

    print(f"\n{'='*80}")
    print("WEEKLY STRATEGIC COUNTER GUROBI OSCO ROLLOUT SUMMARY")
    print(f"{'='*80}")
    print(f"  Successful executions: {len(successful_schedule_log)}")
    print(f"  Weekly strategic replans executed: {weekly_replan_count}")
    print(f"  Weekly counter refreshes executed: {weekly_counter_feed_count}")
    print(f"  Strategic replanning cadence (weeks): {replan_every_n_weeks}")
    print(f"  Remaining strategic counter mass: {sum(running_weekly_counter.values())}")
    print(f"  Final objective value: {final_value:.6f}")
    print(
        f"  Wall time: {loop_wall_elapsed:.2f}s | "
        f"Selector time: {cumulative_selector_seconds:.2f}s over {selector_call_count} calls "
        f"(avg {avg_selector_ms:.1f} ms/call)"
    )
    print(f"{'='*80}\n")

    return final_value, current_schedule_log


def _compute_dsa_score_components(
        job: Dict[str, Any],
        current_time: int,
        completed_exec_counts: Counter,
        recent_jobs: set,
        current_project_completions: Dict[str, float],
        current_gous_completions: Dict[str, float],
        total_execs_by_project: Dict[str, int],
        total_execs_by_gous: Dict[str, int],
) -> Dict[str, float]:
    """Return the raw component scores used by the DSA selector."""
    job_id = job['job_id']
    observed = completed_exec_counts.get(job_id, 0)
    project_id = job['project_id']
    gous_id = job['gous_id']

    condition_score = job['condition_scores'].get(current_time, -1)
    science_rank_score = job['science_rank_score']
    cycle_grade_score = job['cycle_grade_score']
    array_score = job['array_score_yes'] if observed > 0 else job['array_score_no']
    ha_score = 12 if job_id in recent_jobs else job['base_ha_scores'].get(current_time, -1)

    total_project_execs = total_execs_by_project.get(project_id, 0)
    total_gous_execs = total_execs_by_gous.get(gous_id, 0)
    project_completion = current_project_completions.get(project_id, 0) / max(1, total_project_execs)
    gous_completion = current_gous_completions.get(gous_id, 0) / max(1, total_gous_execs)
    completion_score = 4 + 2 * (
        (observed / max(1, job['total_execs'])) +
        project_completion +
        gous_completion
    )

    return {
        'cond': condition_score,
        'sciencerank': science_rank_score,
        'cyclegrade': cycle_grade_score,
        'ha': ha_score,
        'sbcompletion': completion_score,
        'array': array_score,
    }


def _format_dsa_executive(executive: Any) -> str:
    if isinstance(executive, str):
        return executive
    if isinstance(executive, dict):
        parts = [f"{k}:{float(v):.3f}" for k, v in sorted(executive.items())]
        return "{" + ", ".join(parts) + "}"
    return str(executive)


def _build_dsa_log_row(
        job: Dict[str, Any],
        current_time: int,
        component_scores: Dict[str, float],
        weighted_dsa_score: float,
        final_score: float,
        eb_improvement_score: Optional[float] = None,
        scaled_eb_improvement_score: Optional[float] = None,
        eb_contribution_score: Optional[float] = None,
        counter_bonus_score: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        'job_id': job['job_id'],
        'grade': job.get('grade'),
        'executive': _format_dsa_executive(job.get('executive')),
        'project_id': job.get('project_id'),
        'gous_id': job.get('gous_id'),
        'remaining_execs': job.get('remaining_execs'),
        'total_execs': job.get('total_execs'),
        'pwv_threshold': job.get('pwv_thresholds', {}).get(current_time, np.nan),
        'rms_threshold': job.get('rms_threshold', np.nan),
        'cond': component_scores['cond'],
        'sciencerank': component_scores['sciencerank'],
        'cyclegrade': component_scores['cyclegrade'],
        'ha': component_scores['ha'],
        'sbcompletion': component_scores['sbcompletion'],
        'array': component_scores['array'],
        'dsa_score': weighted_dsa_score,
        'eb_improvement_score': eb_improvement_score,
        'scaled_eb_improvement_score': scaled_eb_improvement_score,
        'eb_contribution_score': eb_contribution_score,
        'counter_bonus_score': counter_bonus_score,
        'final_score': final_score,
    }


def _log_dsa_selector_step(
        selector_label: str,
        current_time: int,
        all_remaining_jobs: List[Dict[str, Any]],
        runnable_jobs: List[Dict[str, Any]],
        scored_rows: List[Dict[str, Any]],
        selected_job_id: Optional[str],
        dsa_share: Optional[float] = None,
        eb_share: Optional[float] = None,
        eb_improvement_scale: Optional[float] = None,
) -> None:
    print(f"\n[DSA-TRACE:{selector_label} t={current_time}]")
    print(f"  Remaining science jobs: {sum(1 for j in all_remaining_jobs if j.get('type') == 'science')}")
    print(f"  Runnable science jobs:  {len(runnable_jobs)}")
    if dsa_share is not None and eb_share is not None:
        print(f"  Blend weights: dsa_share={dsa_share:.4f}, eb_share={eb_share:.4f}")
    if eb_improvement_scale is not None:
        print(f"  EB improvement scale: {eb_improvement_scale:.4f}")

    if not scored_rows:
        print("  No runnable candidates (selector returns None)")
        return

    scored_rows = sorted(scored_rows, key=lambda r: (-r['final_score'], r['job_id']))
    print("  Candidate breakdown (sorted by final score):")
    for row in scored_rows:
        base = (
            f"    - {row['job_id']} | grade={row['grade']} | exec={row['executive']} | "
            f"proj={row['project_id']} | gous={row['gous_id']} | "
            f"remaining={row['remaining_execs']}/{row['total_execs']} | "
            f"pwv_thr={row['pwv_threshold']} | rms_thr={row['rms_threshold']} | "
            f"dsa={row['dsa_score']:.4f}"
        )
        if row['eb_improvement_score'] is not None:
            base += f" | eb_impr={row['eb_improvement_score']:.6f}"
        if row.get('scaled_eb_improvement_score') is not None:
            base += f" | eb_impr_scaled={row['scaled_eb_improvement_score']:.6f}"
        if row.get('eb_contribution_score') is not None:
            base += f" | eb_term={row['eb_contribution_score']:.6f}"
        if row.get('counter_bonus_score') is not None:
            base += f" | counter_bonus={row['counter_bonus_score']:.6f}"
        base += f" | final={row['final_score']:.6f}"
        print(base)
        print(
            f"      components: cond={row['cond']:.4f}, sciencerank={row['sciencerank']:.4f}, "
            f"cyclegrade={row['cyclegrade']:.4f}, ha={row['ha']:.4f}, "
            f"sbcompletion={row['sbcompletion']:.4f}, array={row['array']:.4f}"
        )

    if selected_job_id is None:
        print("  Selected: None")
        return
    selected_row = next((r for r in scored_rows if r['job_id'] == selected_job_id), None)
    if selected_row is None:
        print(f"  Selected: {selected_job_id} (not found in scored rows)")
        return
    print(
        f"  Selected: {selected_row['job_id']} | grade={selected_row['grade']} | "
        f"exec={selected_row['executive']} | final={selected_row['final_score']:.6f}"
    )


def _build_dsa_selector_support_tables(
        all_jobs_in_period: List[Dict[str, Any]],
        all_projects: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, int], Dict[str, int]]:
    """Build project / GOUS lookup tables shared by DSA-based selectors."""
    project_job_ids = {p['project_id']: list(p.get('job_ids', [])) for p in all_projects}
    gous_job_ids = defaultdict(list)
    total_execs_by_project = defaultdict(int)
    total_execs_by_gous = defaultdict(int)

    for job in all_jobs_in_period:
        total_execs = job.get('total_execs', 0)
        total_execs_by_project[job['project_id']] += total_execs
        total_execs_by_gous[job['gous_id']] += total_execs
        gous_job_ids[job['gous_id']].append(job['job_id'])

    return (
        project_job_ids,
        dict(gous_job_ids),
        dict(total_execs_by_project),
        dict(total_execs_by_gous),
    )
    

def _compute_counter_alignment_bonus(
        *,
        job: Dict[str, Any],
        prophet_remaining_counter: Optional[Counter],
        weights: Dict[str, float],
        counter_bonus_a_multiplier: float,
        counter_bonus_b_multiplier: float,
        normalization_factor: float = 1.0,
) -> float:
    prophet_remaining_counter = prophet_remaining_counter or Counter()
    if prophet_remaining_counter.get(job['job_id'], 0) <= 0:
        return 0.0

    grade = str(job.get('grade', '')).strip().upper()
    if grade == 'A':
        counter_bonus_multiplier = counter_bonus_a_multiplier
        sb_weight_scale = float(weights.get('sb_A', 0.0))
    elif grade == 'B':
        counter_bonus_multiplier = counter_bonus_b_multiplier
        sb_weight_scale = float(weights.get('sb_B', 0.0))
    else:
        return 0.0

    if counter_bonus_multiplier <= 0.0 or sb_weight_scale <= 0.0:
        return 0.0

    try:
        job_payoff_score = float(job.get('weight', 0.0))
    except (TypeError, ValueError):
        job_payoff_score = 0.0
    normalization_factor = max(float(normalization_factor), 1e-12)
    return (counter_bonus_multiplier * sb_weight_scale * job_payoff_score) / normalization_factor


def _compute_dsa_counter_bonus_normalizer(
        *,
        job: Dict[str, Any],
        total_execs_by_project: Dict[str, int],
        total_execs_by_gous: Dict[str, int],
) -> float:
    """Match the count denominators used by the DSA sbcompletion component."""
    return float(
        max(1, int(job.get('total_execs', 1))) +
        max(1, int(total_execs_by_project.get(job.get('project_id'), 0))) +
        max(1, int(total_execs_by_gous.get(job.get('gous_id'), 0)))
    )


def _compute_dsa_eb_scored_rows(
        candidate_jobs: List[Dict[str, Any]],
        current_time: int,
        exec_time_used: Dict[str, float],
        completed_exec_counts: Counter,
        current_schedule_log: Optional[List[str]],
        project_job_ids: Dict[str, List[str]],
        gous_job_ids: Dict[str, List[str]],
        total_execs_by_project: Dict[str, int],
        total_execs_by_gous: Dict[str, int],
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        weights: Dict[str, float],
        eb_ramp_exponent: float,
        total_time_steps: int,
        prophet_remaining_counter: Optional[Counter] = None,
        counter_bonus_a_multiplier: float = 0.0,
        counter_bonus_b_multiplier: float = 0.0,
) -> Tuple[Optional[str], List[Dict[str, Any]], float, float, float]:
    """Score candidate jobs using the shared DSA + ramped EB formula."""
    current_weights = _with_ramped_eb_penalty(
        weights,
        current_time=current_time,
        total_time_steps=total_time_steps,
        ramp_exponent=eb_ramp_exponent,
    )
    final_eb_share = weights.get('eb_penalty', 0.0)
    eb_share = min(max(current_weights.get('eb_penalty', final_eb_share), 0.0), 1.0)
    dsa_share = 1.0 - eb_share
    eb_improvement_scale = float(weights.get('dsa_eb_improvement_scale', 100000.0))

    recent_jobs = set()
    if current_schedule_log:
        for entry in current_schedule_log[-8:]:
            recent_jobs.add(entry.split("@", 1)[0])

    current_project_completions = {
        project_id: sum(completed_exec_counts.get(job_id, 0) for job_id in job_ids)
        for project_id, job_ids in project_job_ids.items()
    }
    current_gous_completions = {
        gous_id: sum(completed_exec_counts.get(job_id, 0) for job_id in job_ids)
        for gous_id, job_ids in gous_job_ids.items()
    }

    current_exec_time = dict(exec_time_used)
    eta4_current = sum(current_exec_time.values())
    if eta4_current <= 0:
        eta4_current = 1.0
    current_eb_sq_penalty = _calculate_eb_squared_penalty(
        current_exec_time,
        executive_quotas_frac,
        eta4_current,
    )

    best_job_id = None
    best_score = -float('inf')
    scored_rows: List[Dict[str, Any]] = []
    prophet_remaining_counter = prophet_remaining_counter or Counter()

    for job in candidate_jobs:
        component_scores = _compute_dsa_score_components(
            job=job,
            current_time=current_time,
            completed_exec_counts=completed_exec_counts,
            recent_jobs=recent_jobs,
            current_project_completions=current_project_completions,
            current_gous_completions=current_gous_completions,
            total_execs_by_project=total_execs_by_project,
            total_execs_by_gous=total_execs_by_gous,
        )
        dsa_score = (
            DSA_WEIGHTS['cond'] * component_scores['cond'] +
            DSA_WEIGHTS['sciencerank'] * component_scores['sciencerank'] +
            DSA_WEIGHTS['cyclegrade'] * component_scores['cyclegrade'] +
            DSA_WEIGHTS['ha'] * component_scores['ha'] +
            DSA_WEIGHTS['sbcompletion'] * component_scores['sbcompletion'] +
            DSA_WEIGHTS['array'] * component_scores['array']
        )

        future_exec_time = deepcopy(current_exec_time)
        if isinstance(job.get('executive'), str):
            future_exec_time.setdefault(job['executive'], 0)
            future_exec_time[job['executive']] += job['length']
        elif isinstance(job.get('executive'), dict):
            for exec_name, frac in job['executive'].items():
                future_exec_time.setdefault(exec_name, 0)
                future_exec_time[exec_name] += job['length'] * frac
        eta4_future = eta4_current + job['length']
        future_eb_sq_penalty = _calculate_eb_squared_penalty(
            future_exec_time,
            executive_quotas_frac,
            eta4_future,
        )
        eb_improvement_score = current_eb_sq_penalty - future_eb_sq_penalty
        scaled_eb_improvement_score = eb_improvement_scale * eb_improvement_score
        eb_contribution_score = eb_share * scaled_eb_improvement_score
        counter_bonus_score = _compute_counter_alignment_bonus(
            job=job,
            prophet_remaining_counter=prophet_remaining_counter,
            weights=weights,
            counter_bonus_a_multiplier=counter_bonus_a_multiplier,
            counter_bonus_b_multiplier=counter_bonus_b_multiplier,
            normalization_factor=_compute_dsa_counter_bonus_normalizer(
                job=job,
                total_execs_by_project=total_execs_by_project,
                total_execs_by_gous=total_execs_by_gous,
            ),
        )
        score = dsa_share * dsa_score + eb_contribution_score + counter_bonus_score
        scored_rows.append(
            _build_dsa_log_row(
                job=job,
                current_time=current_time,
                component_scores=component_scores,
                weighted_dsa_score=dsa_score,
                final_score=score,
                eb_improvement_score=eb_improvement_score,
                scaled_eb_improvement_score=scaled_eb_improvement_score,
                eb_contribution_score=eb_contribution_score,
                counter_bonus_score=counter_bonus_score,
            )
        )
        if score > best_score:
            best_score = score
            best_job_id = job['job_id']

    return best_job_id, scored_rows, dsa_share, eb_share, eb_improvement_scale


def dsa_eb_selector_factory(
        all_jobs_in_period: List[Dict[str, Any]],
        all_projects: List[Dict[str, Any]],
        weights: Dict[str, float],
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        eb_ramp_exponent: float = 50.0,
) -> Callable:
    """Build a DSA selector that blends DSA scores with a ramped EB gain term."""
    project_job_ids, gous_job_ids, total_execs_by_project, total_execs_by_gous = (
        _build_dsa_selector_support_tables(all_jobs_in_period, all_projects)
    )

    def job_selector(
            all_remaining_jobs: List[Dict[str, Any]],
            current_time: int,
            total_time_steps: int,
            observed_weather: Dict[int, Tuple[float, float]],
            exec_time_used: Dict[str, float],
            completed_exec_counts: Counter,
            unusable_future_indices: set,
            current_schedule_log: Optional[List[str]] = None
    ) -> Optional[str]:
        def is_runnable_now(job: Dict[str, Any]) -> bool:
            if current_time not in job.get("available", []):
                return False
            if current_time + job['length'] > total_time_steps:
                return False
            pwv_now, rms_now = observed_weather.get(current_time, (np.nan, np.nan))
            if np.isnan(pwv_now) or np.isnan(rms_now):
                return False
            pwv_thresh = job.get("pwv_thresholds", {}).get(current_time, np.inf)
            rms_thresh = job.get("rms_threshold", -np.inf)
            if pwv_now > pwv_thresh or rms_now < rms_thresh:
                return False
            for t_offset in range(job['length']):
                if (current_time + t_offset) in unusable_future_indices:
                    return False
            return True

        runnable_jobs = [
            job for job in all_remaining_jobs
            if job.get('type') == 'science' and is_runnable_now(job)
        ]
        if not runnable_jobs:
            return None

        best_job_id, scored_rows, dsa_share, eb_share, eb_improvement_scale = _compute_dsa_eb_scored_rows(
            candidate_jobs=runnable_jobs,
            current_time=current_time,
            exec_time_used=exec_time_used,
            completed_exec_counts=completed_exec_counts,
            current_schedule_log=current_schedule_log,
            project_job_ids=project_job_ids,
            gous_job_ids=gous_job_ids,
            total_execs_by_project=total_execs_by_project,
            total_execs_by_gous=total_execs_by_gous,
            executive_quotas_frac=executive_quotas_frac,
            weights=weights,
            eb_ramp_exponent=eb_ramp_exponent,
            total_time_steps=total_time_steps,
        )

        _log_dsa_selector_step(
            selector_label='dsa_eb',
            current_time=current_time,
            all_remaining_jobs=all_remaining_jobs,
            runnable_jobs=runnable_jobs,
            scored_rows=scored_rows,
            selected_job_id=best_job_id,
            dsa_share=dsa_share,
            eb_share=eb_share,
            eb_improvement_scale=eb_improvement_scale,
        )
        return best_job_id

    return job_selector


def fixed_selector_factory(prophet_solver_fn, statistic="mean"):
    """
    Job selector that assumes forecast means are exact for future,
    but uses observed weather at the current step.
    Solves prophet once per time step using a combined weather forecast,
    and selects the job scheduled at the current time step.
    """

    def job_selector(jobs, projects, forecast_distributions, observed_weather,
                     time_steps, forecast_start_time, executive_quotas):

        T_remain = time_steps - forecast_start_time

        # Step 1: Shift jobs' valid start times to align with t=0
        shifted_jobs = []
        for job in jobs:
            new_job = job.copy()
            new_job["valid_starts"] = [s - forecast_start_time for s in job["valid_starts"] if s >= forecast_start_time]
            if new_job["valid_starts"]:
                shifted_jobs.append(new_job)

        if not shifted_jobs:
            return None

        # Step 2: Truncate projects to only include shifted jobs
        job_ids_in_shifted = {job["job_id"] for job in shifted_jobs}
        shifted_projects = [
            {
                "project_id": p["project_id"],
                "executive": p["executive"],
                "weight": p["weight"],
                "job_ids": [jid for jid in p["job_ids"] if jid in job_ids_in_shifted]
            }
            for p in projects
            if any(jid in job_ids_in_shifted for jid in p["job_ids"])
        ]

        # Step 3: Create synthetic weather
        combined_weather = []

        # Step 3a: t=0 uses observed weather
        combined_weather.append({
            "PWV_realized": observed_weather["PWV_realized"],
            "RMS_realized": observed_weather["RMS_realized"]
        })

        # Step 3b: t=1,... uses forecasted mean
        for fcast in forecast_distributions[1:]:  # skip index 0, already used
            combined_weather.append({
                "PWV_realized": fcast["PWV"][statistic],
                "RMS_realized": fcast["RMS"][statistic]
            })

        fixed_weather_df = pd.DataFrame(combined_weather, index=range(T_remain))
        print(f"current_time: {forecast_start_time}")
        print(f"fixed_weather_df: {fixed_weather_df}")

        # print(mean_weather_df)
        # print(shifted_jobs)
        # print(executive_quotas)
        # Step 4: Solve prophet on shifted jobs and synthetic weather
        val, schedule = prophet_solver_fn(
            shifted_jobs,
            shifted_projects,
            fixed_weather_df,
            T_remain,
            executive_quotas=executive_quotas
        )
        # print("attempt schedule ", schedule)

        # return random choice that is feasible
        if len(schedule) == 0:
            feasible = []
            for job in shifted_jobs:
                if 0 in job["valid_starts"] and (not job["length"] > T_remain) and \
                        fixed_weather_df.loc[0, "PWV_realized"] <= job["pwv_thresh"] and \
                        fixed_weather_df.loc[0, "RMS_realized"] <= job["rms_thresh"] and \
                        executive_quotas[job['executive']][1] > job['length']:
                    feasible.append(job)
            # print("feasible jobs: ", feasible)
            if len(feasible):
                return sorted(feasible, key=lambda x: random.random())[0]["job_id"]
            else:
                return None

        # Step 5: Return the job scheduled at time 0 in shifted frame
        for entry in schedule:
            job_id, start = entry.split("@")
            if int(start) == 0:
                return job_id

        return None

    return job_selector


def _add_job_time_to_exec_balance(
    exec_time_dict: Dict[str, float], job: Dict[str, Any]
) -> None:
    """
    Accumulates job execution time into the executive time dictionary,
    handling both single (str) and fractional (dict) executive assignments.
    """
    executive_info = job.get("executive")
    job_length = job.get("length", 0)

    if isinstance(executive_info, str):
        exec_time_dict.setdefault(executive_info, 0.0)
        exec_time_dict[executive_info] += job_length
    elif isinstance(executive_info, dict):
        for exec_name, fraction in executive_info.items():
            exec_time_dict.setdefault(exec_name, 0.0)
            exec_time_dict[exec_name] += job_length * fraction

def planning_loop_with_precomputed_forecasts(
        all_jobs_initial: List[Dict[str, Any]],
        all_projects_initial: List[Dict[str, Any]],
        realized_weather: Dict[int, Tuple[float, float]],
        weather_forecasts: Dict[int, Dict[str, np.ndarray]],  # <-- NEW INPUT
        total_simulation_time_steps: int,
        job_selector_fn: Callable,
        initial_absolute_executive_quotas: Dict[str, Tuple[float, float]],
):
    """
    Simulation loop that uses a pre-computed, time-evolving forecast.

    This version replaces the call to `generate_noisy_forecast_from_realized`
    with a lookup into the `weather_forecasts` dictionary.
    """
    t_global = 0
    busy_until_global_idx = 0
    jobs = deepcopy(all_jobs_initial)
    scheduled_jobs = Counter()
    current_schedule_log = []

    exec_time_used = {k: 0 for k in initial_absolute_executive_quotas.keys()}
    all_job_execs = set()

    for j in all_jobs_initial:
        if isinstance(j['executive'], str):
            all_job_execs.add(j['executive'])
        elif isinstance(j['executive'], dict):
            all_job_execs.update(j['executive'].keys())

    for job_exec in all_job_execs:
        if job_exec not in exec_time_used: exec_time_used[job_exec] = 0
        if job_exec not in initial_absolute_executive_quotas:
            initial_absolute_executive_quotas[job_exec] = (0, total_simulation_time_steps)

    all_unusable_indices = {
        idx for idx, (pwv, rms) in realized_weather.items()
        if pd.isna(pwv) or pd.isna(rms)
    }

    while t_global < total_simulation_time_steps:
        print(f"Loop Time (Global Idx): {t_global}, Busy Until: {busy_until_global_idx}", flush=True)
        if t_global < busy_until_global_idx:
            t_global += 1
            continue

        jobs = [j for j in jobs if j["remaining_execs"] > 0]
        if not jobs:
            print(f"  No remaining jobs to consider at t_global={t_global}. Ending loop.")
            break

        # --- Get observed weather at the current time step ---
        observed_pwv_at_t, observed_rms_at_t = realized_weather.get(t_global, (np.nan, np.nan))

        if pd.isna(observed_pwv_at_t) or pd.isna(observed_rms_at_t):
            print(f"  No valid weather at t_global={t_global}. Advancing time.")
            t_global += 1
            continue

        realized_weather_for_current_t_dict = {"PWV": observed_pwv_at_t, "RMS": observed_rms_at_t}

        # =========================================================================
        # --- MODIFIED SECTION: Use the pre-computed forecast ---
        # =========================================================================

        # 1. Retrieve the entire forecast state as it was known at t_global
        current_forecast_state = weather_forecasts.get(t_global)

        if current_forecast_state is None:
            print(f"  Error: No pre-computed forecast found for t_global={t_global}. Ending loop.")
            break

        # 2. Slice the forecast arrays to get the outlook for the FUTURE (t+1, t+2, ...)
        fc_pwv_means = current_forecast_state['pwv_mean'][t_global + 1:]
        fc_pwv_stds = current_forecast_state['pwv_std'][t_global + 1:]
        fc_rms_means, fc_rms_stds = get_rms_forecast_slice(
            current_forecast_state,
            issuance_idx=t_global,
            start_idx=t_global + 1,
            end_idx=total_simulation_time_steps,
        )

        # =========================================================================

        # This part remains the same. It takes the forecast arrays and calculates statistics.
        forecast_stats_for_future_steps = calculate_forecast_statistics(
            fc_pwv_means, fc_pwv_stds, fc_rms_means, fc_rms_stds
        )

        # --- Quota and Job Selection Logic (Unchanged) ---
        updated_remaining_quotas = {}
        for exec_name, (global_lb, global_ub) in initial_absolute_executive_quotas.items():
            used = exec_time_used[exec_name]
            updated_remaining_quotas[exec_name] = (max(0, global_lb - used), max(0, global_ub - used))
        for job_exec_key in all_job_execs:
            if job_exec_key not in updated_remaining_quotas:
                updated_remaining_quotas[job_exec_key] = (0, max(0, total_simulation_time_steps - t_global))

        for i in range(len(forecast_stats_for_future_steps)):
            future_global_idx = t_global + 1 + i

            # Check the "true" realized weather for that slot
            future_pwv, future_rms = realized_weather.get(future_global_idx, (np.nan, np.nan))

            # If the true slot is unusable, force the forecast to be NaN
            if pd.isna(future_pwv) or pd.isna(future_rms):
                forecast_stats_for_future_steps[i]['PWV']['mean'] = np.nan
                forecast_stats_for_future_steps[i]['PWV']['bottom_quartile'] = np.nan
                forecast_stats_for_future_steps[i]['PWV']['top_quartile'] = np.nan
                forecast_stats_for_future_steps[i]['RMS']['mean'] = np.nan
                forecast_stats_for_future_steps[i]['RMS']['bottom_quartile'] = np.nan
                forecast_stats_for_future_steps[i]['RMS']['top_quartile'] = np.nan

        # The selector's signature might be different in your actual code,
        # but the inputs it receives are now from the real forecast.
        # This call assumes your selector is compatible with the `job_selector_fn` call in your original loop.
        # print(f"forecast_stats_for_future_steps: {forecast_stats_for_future_steps}")

        # To make sure we dont select jobs that run into NaNs right now. But dont look too far ahead for this.
        unusable_for_selector = {
            idx for idx in all_unusable_indices if t_global <= idx <= t_global+5*48
        }

        selected_job_id, _ = job_selector_fn(
            remaining_jobs=jobs,
            all_projects=all_projects_initial,
            realized_weather_for_current_t=realized_weather_for_current_t_dict,
            forecast_statistics_for_future=forecast_stats_for_future_steps,
            current_global_time_idx=t_global,
            current_executive_quotas=updated_remaining_quotas,
            total_time_steps=total_simulation_time_steps,
            unusable_future_global_indices = unusable_for_selector
        )

        if selected_job_id is None:
            t_global += 1
            continue

        job_to_schedule = next((j for j in jobs if j["job_id"] == selected_job_id), None)
        if job_to_schedule is None:
            print(f"Error: Selector job_id {selected_job_id} not found. Advancing.")
            t_global += 1
            continue

        current_schedule_log.append(f"{selected_job_id}@{t_global}")
        busy_until_global_idx = t_global + job_to_schedule["length"]

        was_successful = True
        reason = "Success"

        if was_successful:
            print(f"  [t={t_global}] SUCCESS: Fixed-selector scheduled {selected_job_id} successfully.")
            job_to_schedule['remaining_execs'] -= 1
            _add_job_time_to_exec_balance(exec_time_used, job_to_schedule)
        else:
            print(
                f"  [t={t_global}] FAILURE: Fixed-selector scheduled {selected_job_id} failed due to {reason}. Executions remaining unchanged.")
        t_global += 1

    print(f"Planning loop (with pre-computed forecasts) finished.")
    print(f"Final exec_time_used: {exec_time_used}")
    return 0.0, current_schedule_log


def needs_selector_factory() -> Callable:
    """
    Returns a job selector function that:
    - Computes how much more time each executive needs to meet the lower bound
    - Selects the executive with the largest deficit
    - Among eligible jobs for that executive, selects the one with highest weight that can be run now
    """
    def job_selector(
        jobs: List[Dict],
        projects: List[Dict],
        forecast_distributions,
        observed_weather: Dict,
        time_steps: int,
        forecast_start_time: int,
        executive_quotas: Dict[str, Tuple[float, float]]
    ) -> Optional[str]:
        t = forecast_start_time
        exec_deficits = {}

        # Compute required time for each executive
        for exec_name, (lb, ub) in executive_quotas.items():
            required = lb
            exec_deficits[exec_name] = required

        # Find candidate jobs that can be run now
        eligible_jobs = []
        for job in jobs:
            if t not in job["valid_starts"]:
                continue
            if observed_weather["PWV_realized"] > job["pwv_thresh"] or \
               observed_weather["RMS_realized"] > job["rms_thresh"]:
                continue
            eligible_jobs.append(job)

        if not eligible_jobs:
            return None

        # Sort executives by how much more time they need (highest first)
        sorted_execs = sorted(exec_deficits.items(), key=lambda x: -x[1])

        for exec_name, _ in sorted_execs:
            candidates = [j for j in eligible_jobs if j["executive"] == exec_name and j['length'] < executive_quotas[exec_name][1]]
            if candidates:
                # Pick job with max weight
                best_job = max(candidates, key=lambda j: j["weight"])
                return best_job["job_id"]

        return None  # No feasible job found

    return job_selector


def fixed_selector_factory_real(  # New version name
        prophet_solver_fn: Callable,
        weights: dict,
        priority_job_ids,
        statistic_to_use: str = "mean",
        planning_horizon: Optional[int] = None,
        cumulative_exec_time_offset: Optional[Dict[str, float]] = None,
        cumulative_observable_time_offset: int = 0
) -> Callable:
    """
    Job selector that:
    1. Receives observed weather for current step and forecast statistics for future.
    2. Constructs a single synthetic 'realized_weather' dictionary for Prophet's horizon.
    3. Shifts jobs' 'available' times to be relative to the current time.
    4. Calls Prophet solver with these shifted jobs and the synthetic weather.
    5. Selects the job Prophet schedules at relative time 0.
    
    Args:
        cumulative_exec_time_offset: Cumulative exec time from previous periods (V^(c-1)).
            Passed to prophet solver for cumulative EB tracking.
        cumulative_observable_time_offset: Cumulative observable (non-NaN) time bins from 
            previous periods. Passed to prophet solver for cumulative EB tracking.
    """
    # Store cumulative offsets for use in selector
    if cumulative_exec_time_offset is None:
        cumulative_exec_time_offset = {}
    _cumulative_exec_offset = dict(cumulative_exec_time_offset)
    _cumulative_obs_offset = cumulative_observable_time_offset

    def job_selector(
            remaining_jobs: List[Dict[str, Any]],
            all_projects: List[Dict[str, Any]],
            realized_weather_for_current_t: Dict[str, float],  # {"PWV": val_or_nan, "RMS": val_or_nan}
            forecast_statistics_for_future: List[Dict[str, Dict[str, float]]],  # For t+1, t+2...
            current_global_time_idx: int,
            current_executive_quotas: Dict[str, Tuple[float, float]],
            total_time_steps: int,
            unusable_future_global_indices: set
    ) -> (Optional[str], int):

        runnable_now_candidates = []
        current_pwv_obs, current_rms_obs = realized_weather_for_current_t["PWV"], realized_weather_for_current_t["RMS"]

        if pd.isna(current_pwv_obs) or pd.isna(current_rms_obs):
            return None, 0.0  # Cannot schedule if current weather is unknown

        for job in remaining_jobs:
            # Check availability, time horizon, and NaN overlaps
            if current_global_time_idx not in job.get("available", []): continue
            if current_global_time_idx + job['length'] > total_time_steps: continue

            is_blocked_by_nan = any(
                (current_global_time_idx + t_offset) in unusable_future_global_indices for t_offset in
                range(job['length']))
            if is_blocked_by_nan: continue

            # Check weather conditions
            pwv_thresh_now = job.get("pwv_thresholds", {}).get(current_global_time_idx, np.inf)
            rms_thresh_now = job.get("rms_threshold", -np.inf)
            if not (current_pwv_obs <= pwv_thresh_now and current_rms_obs >= rms_thresh_now): continue

            runnable_now_candidates.append(job)

        # If no job can be run now, skip the expensive Prophet solve.
        if not runnable_now_candidates:
            return None, 0.0

        remaining_time_in_sim = total_time_steps - current_global_time_idx
        prophet_horizon = remaining_time_in_sim
        if planning_horizon is not None:
            prophet_horizon = min(remaining_time_in_sim, planning_horizon)

        # --- 1. Prepare jobs for the Prophet sub-problem ---
        # Use forecast (planned-config) availability for the lookahead, but
        # for t=0 only allow jobs that are actually runnable now (array-aware).
        runnable_now_ids = {j["job_id"] for j in runnable_now_candidates}

        jobs_for_prophet = []
        for job in remaining_jobs:
            new_job = job.copy()

            forecast_avail, forecast_pwv_t, forecast_rms_t, _ = _resolve_forecast_metadata_for_time(
                job, current_global_time_idx
            )

            relative_available_starts = []
            for global_start_time in forecast_avail:
                is_blocked_by_nan = False
                for t_offset in range(job['length']):
                    if (global_start_time + t_offset) in unusable_future_global_indices:
                        is_blocked_by_nan = True
                        break

                if is_blocked_by_nan:
                    continue
                if global_start_time >= current_global_time_idx and global_start_time + job[
                    'length'] <= total_time_steps:
                    relative_start = global_start_time - current_global_time_idx
                    if relative_start == 0 and job["job_id"] not in runnable_now_ids:
                        continue
                    relative_available_starts.append(relative_start)

            new_job["available"] = sorted(list(set(relative_available_starts)))
            new_job["available"] = [x for x in new_job['available'] if x <= prophet_horizon]

            if isinstance(forecast_pwv_t, dict):
                shifted_pwv_thresholds = {}
                for global_time_key, threshold_val in forecast_pwv_t.items():
                    if global_time_key >= current_global_time_idx:
                        relative_time_key = global_time_key - current_global_time_idx
                        if relative_time_key < prophet_horizon:
                            shifted_pwv_thresholds[relative_time_key] = threshold_val
                new_job["pwv_thresholds"] = shifted_pwv_thresholds
            new_job["rms_threshold"] = forecast_rms_t
            new_job.pop("forecast_available", None)
            new_job.pop("forecast_pwv_thresholds", None)
            new_job.pop("forecast_rms_threshold", None)

            if new_job["available"]:
                jobs_for_prophet.append(new_job)

        if not jobs_for_prophet:
            # print(f"  [Selector t={current_global_time_idx}] No jobs have available start times for Prophet sub-problem.")
            return None, 0.0

        # --- 2. Prepare projects for Prophet ---
        job_ids_in_prophet_problem = {job["job_id"] for job in jobs_for_prophet}
        projects_for_prophet = []
        for p_orig in all_projects:
            current_p_job_ids = [
                jid for jid in p_orig["job_ids"] if jid in job_ids_in_prophet_problem
            ]
            if current_p_job_ids:
                projects_for_prophet.append({
                    "project_id": p_orig["project_id"], "executive": p_orig["executive"],
                    "weight": p_orig["weight"], "grade": p_orig["grade"], "job_ids": current_p_job_ids
                })

        # --- 3. Create synthetic 'realized_weather' dictionary for Prophet's horizon ---
        # This dictionary will be keyed by *relative* time steps (0 to end)
        synthetic_weather_for_prophet: Dict[int, Tuple[float, float]] = {}

        # Relative time 0 uses observed weather for current_global_time_idx
        synthetic_weather_for_prophet[0] = (
            realized_weather_for_current_t["PWV"],  # Can be NaN
            realized_weather_for_current_t["RMS"]  # Can be NaN
        )

        # Relative times 1 to prophet_problem_horizon-1 use forecast statistics
        # print("forecast_statistics_for_future: ", forecast_statistics_for_future)
        # print(total_time_steps, current_global_time_idx)
        for k_relative in range(1, prophet_horizon):
            # print(k_relative)
            forecast_idx = k_relative - 1
            if forecast_idx < len(forecast_statistics_for_future):
                current_fc_stats = forecast_statistics_for_future[forecast_idx]
                pwv_val = current_fc_stats["PWV"][statistic_to_use]  # Can be NaN
                rms_val = current_fc_stats["RMS"][statistic_to_use]  # Can be NaN
                synthetic_weather_for_prophet[k_relative] = (pwv_val, rms_val)
            else:
                # If forecast doesn't cover this far, assume unknown/NaN weather
                synthetic_weather_for_prophet[k_relative] = (np.nan, np.nan)

        # --- 4. Solve Prophet ---
        # print(f"  [Selector t={current_global_time_idx}] Solving Prophet with synthetic weather for rest of time.")
        # The executive_quotas are the *remaining* quotas for current_global_time_idx.

        scaled_quotas = {}
        if remaining_time_in_sim > 0:
            scaling_factor = prophet_horizon / remaining_time_in_sim
            for exec_name, (lb, ub) in current_executive_quotas.items():
                scaled_quotas[exec_name] = (lb * scaling_factor, ub * scaling_factor)
        else:  # Avoid division by zero
            scaled_quotas = {k: (0, 0) for k in current_executive_quotas}

        # print(jobs_for_prophet)
        # print(synthetic_weather_for_prophet)
        # print(total_time_steps, current_global_time_idx)
        
        # Pass cumulative offsets to prophet for cumulative EB tracking.
        # This makes EB penalties smaller as the cycle progresses.
        prophet_val, prophet_schedule = prophet_solver_fn(
            jobs_for_prophet,  # These jobs have relative 'available' times
            projects_for_prophet,
            synthetic_weather_for_prophet,  # Weather dict keyed by relative time
            time_steps=prophet_horizon,
            executive_quotas=scaled_quotas,
            weights=weights,
            priority_job_ids=priority_job_ids,
            force_job_at_t0=True,
            cumulative_exec_time_offset=_cumulative_exec_offset,
            cumulative_observable_time_offset=_cumulative_obs_offset
        )
        # print(f"PROPHET VAL: {prophet_val}, PROPHET_SCHED: {prophet_schedule}")

        # --- 5. Select job scheduled at relative time 0 ---
        if prophet_val > -1000.0 and prophet_schedule:  # Check for valid solution
            for entry in prophet_schedule:
                job_id_scheduled, start_time_str = entry.split("@")
                if int(start_time_str) == 0:  # Job scheduled at relative time 0
                    # print(f"  [Selector t={current_global_time_idx}] Prophet selected: {job_id_scheduled}")
                    return job_id_scheduled, prophet_val

        # print(f"  [Selector t={current_global_time_idx}] Prophet found no schedule or no job at relative t=0.")

        # Optional Fallback (if Prophet schedules nothing at t=0, but something *could* run now)
        # Check jobs that are available at relative time 0 and meet current weather
        if pd.isna(synthetic_weather_for_prophet[0][0]) or pd.isna(synthetic_weather_for_prophet[0][1]):
            return None, 0.0  # Cannot fallback if current weather is unknown

        if runnable_now_candidates:
            # Select the best valid fallback job by weight
            best_fallback_job = max(runnable_now_candidates, key=lambda j_cand: j_cand["weight"])
            return best_fallback_job["job_id"], 0.0  # Return 0 value as it's a heuristic choice

        return None, 0.0  # No job selected

    return job_selector


def is_job_globally_valid(
        job_to_check: Dict,
        current_time: int,
        total_time_steps: int,
        total_observable_time: int,
        observed_weather: Dict[int, Tuple[float, float]],
        exec_time_used: Dict[str, float],
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        remaining_valid_slots_from_t: np.ndarray
) -> bool:
    """
    Checks if a job is valid to start NOW, considering both local conditions
    and global executive balance feasibility for the entire episode.

    This is a standalone version of the logic in TelescopeSchedulerEnvPPO._job_is_valid_now.
    """
    # 1. Local checks: availability, weather, fits in timeline
    if current_time not in job_to_check.get("available", []):
        return False

    pwv_now, rms_now = observed_weather.get(current_time, (np.nan, np.nan))
    if np.isnan(pwv_now) or np.isnan(rms_now):
        return False

    pwv_thresh_now = job_to_check.get("pwv_thresholds", {}).get(current_time, np.inf)
    rms_thresh_now = job_to_check.get("rms_threshold", -np.inf)
    job_len = job_to_check['length']

    if pwv_now > pwv_thresh_now or rms_now < rms_thresh_now:
        return False

    if current_time + job_len > total_time_steps:
        return False

    # 2. Global Executive Balance Feasibility Checks
    executive_info = job_to_check.get("executive")

    # 2a. Upper-bound check for the executive(s) of the job being considered.
    # We must ensure that scheduling this job does not push any of its assigned
    # executives over their maximum allowed time fraction.
    if isinstance(executive_info, str):
        # Case 1: Job has a single executive
        exec_name = executive_info
        if exec_name in executive_quotas_frac:
            _, max_frac = executive_quotas_frac[exec_name]
            if exec_time_used.get(exec_name, 0) + job_len > max_frac * total_observable_time + 1e-9:
                return False
    elif isinstance(executive_info, dict):
        # Case 2: Job has fractional executives
        for exec_name, fraction in executive_info.items():
            if exec_name in executive_quotas_frac:
                _, max_frac = executive_quotas_frac[exec_name]
                time_to_add = job_len * fraction
                if exec_time_used.get(exec_name, 0) + time_to_add > max_frac * total_observable_time + 1e-9:
                    return False
    # If executive_info is not str or dict, or not in quotas (e.g., 'OTHER'), we pass this check.

    # 2b. Lower-bound feasibility check for ALL executives.
    # This is the most complex check: if we take this job now, is there still
    # enough time left in the entire episode for all other executives to meet their minimums?

    # First, calculate the time remaining in the episode *after* this job would finish.
    time_after_job_ends = current_time + job_len
    if time_after_job_ends >= len(remaining_valid_slots_from_t):
        remaining_slots_after_job = 0
    else:
        remaining_slots_after_job = remaining_valid_slots_from_t[time_after_job_ends]

    # It's impossible to meet future quotas if there's no time left.
    if remaining_slots_after_job < 0:
        return False

    # Create a *hypothetical* state of executive time usage, assuming we schedule this job.
    hypothetical_exec_time_used = exec_time_used.copy()
    if isinstance(executive_info, str):
        hypothetical_exec_time_used[executive_info] = hypothetical_exec_time_used.get(executive_info, 0) + job_len
    elif isinstance(executive_info, dict):
        for exec_name, fraction in executive_info.items():
            hypothetical_exec_time_used[exec_name] = hypothetical_exec_time_used.get(exec_name, 0) + (
                        job_len * fraction)

    # Now, calculate the total time still needed to meet all lower bounds, based on the hypothetical state.
    needed_time_for_all_LBs = 0.0
    for exec_name, (min_frac, _) in executive_quotas_frac.items():
        # Use the hypothetical time usage for the calculation
        time_used_for_this_exec = hypothetical_exec_time_used.get(exec_name, 0)

        # Calculate how much time this executive is still lacking to meet its minimum
        lacking = math.ceil(min_frac * total_observable_time - time_used_for_this_exec)
        if lacking > 0:
            needed_time_for_all_LBs += lacking

    # The final check: is the time we absolutely *need* for all lower bounds
    # less than or equal to the available time we have left?
    if needed_time_for_all_LBs > remaining_slots_after_job + 1e-9:
        return False

    # All checks passed
    return True

def _calculate_eb_l1_penalty(exec_time_used, quotas_frac, total_observable_time):
    """Helper to calculate the L1 penalty for a given state of executive time usage."""
    penalty = 0
    if total_observable_time == 0:
        return 0

    for exec_name, (min_frac, max_frac) in quotas_frac.items():
        time_used = exec_time_used.get(exec_name, 0)
        min_target = min_frac * total_observable_time
        max_target = max_frac * total_observable_time

        if time_used < min_target:
            penalty += (min_target - time_used)
        elif time_used > max_target:
            penalty += (time_used - max_target)

    return penalty / total_observable_time  # Return the normalized penalty


def _calculate_eb_squared_penalty(exec_time_used, quotas_frac, eta4_or_total_observed_time):
    """
    Executive balance penalty as in the paper: sum over executives i of
    (e_i - v_i/eta4)_+^2. No penalty for exceeding targets.
    eta4 = total time spent by all executives (observed time); fraction_i = time_used / eta4.
    """
    if eta4_or_total_observed_time <= 0:
        return 0.0
    penalty = 0.0
    for exec_name, (min_frac, _max_frac) in quotas_frac.items():
        time_used = exec_time_used.get(exec_name, 0)
        fraction_i = time_used / eta4_or_total_observed_time
        # Paper: penalty term is (e_i - fraction_i)_+^2
        shortfall = min_frac - fraction_i
        penalty += shortfall ** 2
    return penalty


def compute_paper_objective_value(
        successful_schedule_log: List[str],
        jobs: List[Dict],
        projects: List[Dict],
        exec_time_used: Dict[str, float],
        total_observable_time: int,
        weights: Dict[str, float],
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        verbose: bool = False,
        project_bonus_mode: str = "final_job",
        project_bonus_ramp_ratio: float = 1.2,
) -> float:
    """
    Compute the paper's objective value obj(π) from a schedule of successful
    executions. Uses same normalization (η1, η2, η3, η4) and weights (α1..α4).
    η4 = total time spent by all executives (observed time), not total observable time.
    If verbose=True, prints a full breakdown of job/project weights, the 4 objectives, and the combination.
    """
    n_prime = total_observable_time
    if n_prime <= 0:
        n_prime = 1
    # eta4 = total time spent by all executives (observed time), not total observable time
    total_observed_time = sum(exec_time_used.values())
    eta4 = total_observed_time if total_observed_time > 0 else 1.0

    job_lookup = {j['job_id']: j for j in jobs}
    project_lookup = {p['project_id']: p for p in projects}
    sb_to_project = {sb_id: p['project_id'] for p in projects for sb_id in p.get('job_ids', [])}
    eta1 = _compute_eta1_topk(jobs, n_prime, execs_key='total_execs')
    eta2 = _compute_eta2_topk(projects, n_prime)
    eta3 = n_prime

    # Support both paper-style and legacy weight keys
    if 'obs_completion' in weights or 'proj_completion' in weights:
        alpha1 = weights.get('obs_completion', 0)
        alpha2 = weights.get('proj_completion', 0)
    else:
        alpha1 = weights.get('sb_A', 0) + weights.get('sb_B', 0) + weights.get('sb_C', 0)
        alpha2 = weights.get('proj_A', 0) + weights.get('proj_B', 0) + weights.get('proj_C', 0)
    alpha3 = weights.get('utilization', 0)
    alpha4 = weights.get('eb_penalty', 0)

    # obj1: weighted observation completion
    sum_obs_weight = 0.0
    obj2_raw = 0.0
    total_time_used = 0
    completed_job_counts = Counter()
    job_weights_contributed = []  # (job_id, weight, count) for verbose
    for entry in successful_schedule_log:
        try:
            job_id, _ = entry.split("@")
        except (ValueError, AttributeError):
            continue
        if job_id not in job_lookup:
            continue
        j = job_lookup[job_id]
        w = j.get('weight', 0)
        sum_obs_weight += w
        total_time_used += j['length']
        if project_bonus_mode == "geometric_ramp":
            project_id = sb_to_project.get(job_id)
            if project_id:
                obj2_increment = _compute_geometric_project_bonus(
                    project=project_lookup[project_id],
                    job_id=job_id,
                    completed_exec_counts=completed_job_counts,
                    job_info_map=job_lookup,
                    eta2=eta2,
                    project_bonus_ramp_ratio=project_bonus_ramp_ratio,
                )
                obj2_raw += obj2_increment * eta2
        completed_job_counts[job_id] = completed_job_counts.get(job_id, 0) + 1
    if verbose:
        for job_id, count in completed_job_counts.items():
            j = job_lookup.get(job_id, {})
            job_weights_contributed.append((job_id, j.get('weight', 0), count))

    obj1 = (1.0 / eta1) * sum_obs_weight if eta1 > 0 else 0.0
    completed_projects_with_weights = []  # (project_id, weight) for verbose
    if project_bonus_mode == "final_job":
        obj2_raw = 0.0
        for p in projects:
            job_ids_p = p.get('job_ids', [])
            if not job_ids_p:
                continue

            # Align with prophet's horizon semantics: completion requirements are based on
            # remaining_execs in the provided jobs list for this solve/evaluation window.
            is_complete = True
            for sb_id in job_ids_p:
                j = job_lookup.get(sb_id, {})
                required_execs = j.get('remaining_execs', j.get('total_execs', 1))
                if completed_job_counts.get(sb_id, 0) < required_execs:
                    is_complete = False
                    break

            if is_complete:
                wp = p.get('weight', 0)
                obj2_raw += wp
                if verbose:
                    completed_projects_with_weights.append((p.get('project_id', ''), wp))
    elif project_bonus_mode != "geometric_ramp":
        raise ValueError(f"Unknown project_bonus_mode: {project_bonus_mode}")
    obj2 = (1.0 / eta2) * obj2_raw if eta2 > 0 else 0.0
    obj3 = (1.0 / eta3) * total_time_used if eta3 > 0 else 0.0
    obj4_penalty = _calculate_eb_squared_penalty(exec_time_used, executive_quotas_frac, eta4)
    obj4 = -obj4_penalty  # paper: obj4 is negative penalty

    value = alpha1 * obj1 + alpha2 * obj2 + alpha3 * obj3 + alpha4 * obj4

    if verbose:
        B = sum(j.get('total_execs', 1) for j in jobs)
        P = len(projects)
        _print_paper_objective_breakdown(
            n_prime=n_prime, B=B, P=P, max_w=0, max_proj_w=0,
            eta1=eta1, eta2=eta2, eta3=eta3, eta4=eta4,
            alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, alpha4=alpha4,
            job_weights_contributed=job_weights_contributed,
            completed_projects_with_weights=completed_projects_with_weights,
            sum_obs_weight=sum_obs_weight, total_time_used=total_time_used,
            obj2_raw=obj2_raw,
            exec_time_used=exec_time_used, executive_quotas_frac=executive_quotas_frac,
            total_observable_time=total_observable_time,
            obj1=obj1, obj2=obj2, obj3=obj3, obj4_penalty=obj4_penalty, obj4=obj4,
            value=value
        )

    return value


def _print_paper_objective_breakdown(
        n_prime: int, B: int, P: int, max_w: float,
        eta1: float, eta2: float, eta3: float, eta4: float,
        alpha1: float, alpha2: float, alpha3: float, alpha4: float,
        job_weights_contributed: list,
        completed_projects_with_weights: list,
        sum_obs_weight: float, total_time_used: float, obj2_raw: float,
        exec_time_used: dict, executive_quotas_frac: dict, total_observable_time: int,
        obj1: float, obj2: float, obj3: float, obj4_penalty: float, obj4: float,
        value: float,
        max_proj_w: float = None
):
    """Print a human-readable breakdown of the paper objective value."""
    print("\n  --- Paper objective value breakdown ---")
    print("  Normalization constants (from total_observable_time n' and job/project sets):")
    print(f"    n' (total_observable_time) = {n_prime}")
    print(f"    B (sum of job total_execs) = {B},  P (number of projects) = {P}")
    print(f"    η1 = sum(top-k obs weights) = {eta1:.4f}   [for obj1]")
    print(f"    η2 = sum(top-k proj weights) = {eta2:.4f}   [for obj2]")
    print(f"    η3 = n' = {eta3}   [for obj3]")
    print(f"    η4 = total observed time (sum exec time) = {eta4:.1f}   [for obj4]")
    print("  Weights (α) from config:")
    print(f"    α1 (obs_completion) = {alpha1:.4f}")
    print(f"    α2 (proj_completion) = {alpha2:.4f}")
    print(f"    α3 (util) = {alpha3:.4f},   α4 (EB penalty) = {alpha4:.4f}")

    print("  Job weights contributing to obj1 (each successful execution adds job weight):")
    for job_id, w, count in sorted(job_weights_contributed, key=lambda x: -x[1]):
        print(f"    {job_id}: weight={w:.4f}, executions={count}")
    print(f"  Sum of observation weights (sum_obs_weight) = {sum_obs_weight:.4f}")

    print("  Project weights contributing to obj2 (completed projects only):")
    for proj_id, w in sorted(completed_projects_with_weights, key=lambda x: -x[1])[:20]:
        print(f"    {proj_id}: weight={w:.4f}")
    if len(completed_projects_with_weights) > 20:
        print(f"    ... and {len(completed_projects_with_weights) - 20} more projects")
    print(f"  Sum of completed project weights (obj2_raw) = {obj2_raw:.4f}")

    print("  Executive balance (for obj4 penalty; fractions use η4 = total observed time):")
    for exec_name, (min_frac, _) in executive_quotas_frac.items():
        time_used = exec_time_used.get(exec_name, 0)
        frac = (time_used / eta4) if eta4 > 0 else 0
        shortfall = min_frac - frac if min_frac > frac else 0
        sq = shortfall ** 2 if shortfall > 0 else 0
        print(f"    {exec_name}: time_used={time_used:.1f}, target_frac={min_frac:.4f}, actual_frac={frac:.4f}, shortfall={shortfall:.4f}, shortfall²={sq:.6f}")
    print(f"  EB squared penalty (sum of shortfall²) = {obj4_penalty:.6f}")

    print("  Four objectives:")
    print(f"    obj1 (weighted observation completion) = sum_obs_weight / η1 = {sum_obs_weight:.4f} / {eta1:.4f} = {obj1:.6f}")
    print(f"    obj2 (weighted project completion)      = obj2_raw / η2     = {obj2_raw:.4f} / {eta2:.4f} = {obj2:.6f}")
    print(f"    obj3 (utilization)                     = total_time_used / η3 = {total_time_used:.0f} / {eta3} = {obj3:.6f}")
    print(f"    obj4 (negative EB penalty)             = -obj4_penalty     = -{obj4_penalty:.6f} = {obj4:.6f}")

    print("  Combined (value = α1*obj1 + α2*obj2 + α3*obj3 + α4*obj4):")
    t1 = alpha1 * obj1
    t2 = alpha2 * obj2
    t3 = alpha3 * obj3
    t4 = alpha4 * obj4
    print(f"    α1*obj1 = {alpha1:.4f} * {obj1:.6f} = {t1:.6f}")
    print(f"    α2*obj2 = {alpha2:.4f} * {obj2:.6f} = {t2:.6f}")
    print(f"    α3*obj3 = {alpha3:.4f} * {obj3:.6f} = {t3:.6f}")
    print(f"    α4*obj4 = {alpha4:.4f} * {obj4:.6f} = {t4:.6f}")
    print(f"    Paper objective value = {t1:.6f} + {t2:.6f} + {t3:.6f} + {t4:.6f} = {value:.6f}")
    print("  --- End paper objective breakdown ---\n")


def marginal_gain_greedy_selector_factory(
        weights: dict,
        all_jobs_in_period: list,
        all_projects: list,
        executive_quotas_frac: dict,
        total_observable_time: int,
        priority_job_ids,
        cumulative_exec_time_offset: Optional[Dict[str, float]] = None,
        cumulative_observable_time_offset: int = 0,
        suppress_grade_c_if_ab_available: bool = False,
        use_ramped_eb_penalty: bool = False,
        eb_ramp_exponent: float = 50.0,
        project_bonus_mode: str = "final_job",
        project_bonus_ramp_ratio: float = 1.2,
) -> Callable:
    """
    Returns a greedy selector that picks the job maximizing the marginal gain
    of the objective, as in Algorithm 2 (Greedy Value Routine) of the paper:
    Value(b) = alpha1*Delta_obj1 + alpha2*Delta_obj2 + alpha3*Delta_obj3 + alpha4*Delta_obj4.
    - obj1: weighted observation completion (wp/eta1)
    - obj2: weighted project completion (wp/eta2 if b completes project p, else 0)
      or a geometric project bonus ramp when enabled.
    - obj3: utilization (lb/eta3)
    - obj4: reduction in executive balance squared penalty.
    When priority_job_ids is non-empty (strategic_greedy), adds an optional adherence bonus.
    
    Args:
        cumulative_exec_time_offset: Cumulative exec time from previous periods (V^(c-1)).
            When provided, EB calculations use cumulative time instead of period-only time.
            This makes EB penalties smaller as the cycle progresses.
        cumulative_observable_time_offset: Cumulative observable (non-NaN) time bins from 
            previous periods. Used as base for eta4 denominator in EB calculations.
    """
    # Store cumulative offsets for use in selector
    if cumulative_exec_time_offset is None:
        cumulative_exec_time_offset = {}
    _cumulative_exec_offset = dict(cumulative_exec_time_offset)
    _cumulative_obs_offset = cumulative_observable_time_offset
    # Handle priority_job_ids: accept Counter, dict, set, or None
    if priority_job_ids is None:
        priority_job_ids_counter = Counter()
    elif isinstance(priority_job_ids, Counter):
        priority_job_ids_counter = priority_job_ids
    else:
        priority_job_ids_counter = Counter(priority_job_ids)

    job_info_map = {j['job_id']: j for j in all_jobs_in_period}
    project_lookup = {p['project_id']: p for p in all_projects}
    sb_to_project_map = {sb_id: p['project_id'] for p in all_projects for sb_id in p.get('job_ids', [])}

    # Use cumulative n' for normalization (eta1, eta2, eta3) so that each observation/project
    # contributes the same amount to the objective regardless of when it happens in the cycle.
    # This makes strategic_greedy (with adherence=0) behave like greedy over the full cycle.
    n_prime_period = total_observable_time
    n_prime_cumulative = _cumulative_obs_offset + n_prime_period
    
    eta1 = _compute_eta1_topk(all_jobs_in_period, n_prime_cumulative, execs_key='total_execs')
    eta2 = _compute_eta2_topk(all_projects, n_prime_cumulative)
    eta3 = max(n_prime_cumulative, 1)

    # Alpha weights: support both paper-style and legacy keys
    if 'obs_completion' in weights or 'proj_completion' in weights:
        alpha1 = weights.get('obs_completion', 0)
        alpha2 = weights.get('proj_completion', 0)
    else:
        alpha1 = weights.get('sb_A', 0) + weights.get('sb_B', 0) + weights.get('sb_C', 0)
        alpha2 = weights.get('proj_A', 0) + weights.get('proj_B', 0) + weights.get('proj_C', 0)
    alpha3 = weights.get('utilization', 0)
    base_alpha4 = weights.get('eb_penalty', 0)

    def job_selector(
            all_remaining_jobs: list,
            current_time: int,
            total_time_steps: int,
            observed_weather: dict,
            exec_time_used: dict,
            completed_exec_counts: Counter,
            unusable_future_indices: set,
            current_schedule_log: Optional[List[str]] = None
    ) -> Optional[str]:
        del current_schedule_log

        current_weights = (
            _with_ramped_eb_penalty(
                weights,
                current_time=current_time,
                total_time_steps=total_time_steps,
                ramp_exponent=eb_ramp_exponent,
            )
            if use_ramped_eb_penalty else
            weights
        )
        alpha4 = current_weights.get('eb_penalty', base_alpha4)

        def is_runnable_now(job: dict) -> bool:
            if current_time not in job.get("available", []):
                return False
            if current_time + job['length'] > total_time_steps:
                return False
            pwv_now, rms_now = observed_weather.get(current_time, (np.nan, np.nan))
            if np.isnan(pwv_now) or np.isnan(rms_now):
                return False
            pwv_thresh = job.get("pwv_thresholds", {}).get(current_time, np.inf)
            rms_thresh = job.get("rms_threshold", -np.inf)
            if pwv_now > pwv_thresh or rms_now < rms_thresh:
                return False
            for t_offset in range(job['length']):
                if (current_time + t_offset) in unusable_future_indices:
                    return False
            return True

        runnable_jobs = [job for job in all_remaining_jobs if is_runnable_now(job)]
        if not runnable_jobs:
            return None

        candidate_jobs = runnable_jobs
        if suppress_grade_c_if_ab_available:
            has_ab_runnable = any(str(job.get('grade', '')).strip().upper() in ('A', 'B') for job in runnable_jobs)
            if has_ab_runnable:
                candidate_jobs = [
                    job for job in runnable_jobs
                    if str(job.get('grade', '')).strip().upper() != 'C'
                ]
                if not candidate_jobs:
                    candidate_jobs = runnable_jobs

        # === DEBUG: Track runnable jobs by executive at each time step ===
        # Only log every 100 time steps to avoid spam
        if current_time % 100 == 0:
            _runnable_by_exec = {'NA': 0, 'EA': 0, 'EU': 0, 'CL': 0, 'OTHER': 0}
            for _j in runnable_jobs:
                _exec = _j.get('executive', 'OTHER')
                if isinstance(_exec, str):
                    _runnable_by_exec[_exec] = _runnable_by_exec.get(_exec, 0) + 1
                elif isinstance(_exec, dict):
                    _primary = max(_exec, key=_exec.get)
                    _runnable_by_exec[_primary] = _runnable_by_exec.get(_primary, 0) + 1
            _current_eb_fracs = {k: v/sum(exec_time_used.values()) if sum(exec_time_used.values()) > 0 else 0 for k, v in exec_time_used.items()}
            print(f"  [t={current_time}] Runnable: {_runnable_by_exec}, Current EB: {{{', '.join(f'{k}:{v:.2%}' for k,v in _current_eb_fracs.items())}}}")

        best_job_id = None
        max_marginal_gain = -float('inf')

        # --- Cumulative EB tracking ---
        # Combine period exec_time_used with cumulative offset from previous periods
        cumulative_exec_time = {}
        for k in set(exec_time_used.keys()) | set(_cumulative_exec_offset.keys()):
            cumulative_exec_time[k] = exec_time_used.get(k, 0) + _cumulative_exec_offset.get(k, 0)
        
        # eta4 = total exec time (matching paper's definition: fraction_i = exec_i / total_exec)
        eta4_current = sum(cumulative_exec_time.values())
        if eta4_current <= 0:
            eta4_current = 1.0
        
        current_eb_sq_penalty = _calculate_eb_squared_penalty(
            cumulative_exec_time, executive_quotas_frac, eta4_current)
        
        # === DEBUG: Log EB state and penalty at start of period ===
        if current_time == 0:
            print(f"\n  === GREEDY SELECTOR EB DEBUG (t=0) ===")
            print(f"  cumulative_obs_offset = {_cumulative_obs_offset}")
            print(f"  eta4_current (total exec time) = {eta4_current}")
            print(f"  cumulative_exec_time: {cumulative_exec_time}")
            print(f"  current_eb_sq_penalty = {current_eb_sq_penalty:.6f}")
            print(f"  EB Quotas (targets): {executive_quotas_frac}")
            print(f"  alpha4 (EB penalty weight) = {alpha4}")
            print(f"  ===================================\n")

        for job in candidate_jobs:
            job_id = job['job_id']
            wp = job.get('weight', 0)  # project weight for this observation
            lb = job['length']

            # Paper Algorithm 2: Delta obj1 = (1/eta1) * wp
            delta_obj1 = (1.0 / eta1) * wp if eta1 > 0 else 0.0

            # Delta obj2: final-job completion bonus or geometric project ramp bonus
            project_id = sb_to_project_map.get(job_id)
            delta_obj2 = 0.0
            if project_id:
                project = project_lookup[project_id]
                if project_bonus_mode == "final_job":
                    is_project_completed_after = all(
                        (completed_exec_counts.get(sb_id, 0) + (1 if sb_id == job_id else 0))
                        >= job_info_map[sb_id].get('total_execs', 1)
                        for sb_id in project['job_ids']
                    )
                    if is_project_completed_after:
                        delta_obj2 = (1.0 / eta2) * wp if eta2 > 0 else 0.0
                elif project_bonus_mode == "geometric_ramp":
                    delta_obj2 = _compute_geometric_project_bonus(
                        project=project,
                        job_id=job_id,
                        completed_exec_counts=completed_exec_counts,
                        job_info_map=job_info_map,
                        eta2=eta2,
                        project_bonus_ramp_ratio=project_bonus_ramp_ratio,
                    )
                else:
                    raise ValueError(f"Unknown project_bonus_mode: {project_bonus_mode}")

            # Delta obj3: (1/eta3) * lb
            delta_obj3 = (1.0 / eta3) * lb if eta3 > 0 else 0.0

            # Delta obj4: reduction in squared EB penalty (current - future)
            # Use cumulative time for EB calculations
            future_cumulative_exec_time = deepcopy(cumulative_exec_time)
            if isinstance(job.get('executive'), str):
                future_cumulative_exec_time.setdefault(job['executive'], 0)
                future_cumulative_exec_time[job['executive']] += lb
            elif isinstance(job.get('executive'), dict):
                for exec_name, frac in job['executive'].items():
                    future_cumulative_exec_time.setdefault(exec_name, 0)
                    future_cumulative_exec_time[exec_name] += lb * frac
            eta4_future = eta4_current + lb
            future_eb_sq_penalty = _calculate_eb_squared_penalty(
                future_cumulative_exec_time, executive_quotas_frac, eta4_future)
            delta_obj4 = current_eb_sq_penalty - future_eb_sq_penalty

            value = alpha1 * delta_obj1 + alpha2 * delta_obj2 + alpha3 * delta_obj3 + alpha4 * delta_obj4

            # Optional adherence bonus for strategic_greedy (when priority_job_ids is non-empty)
            planned_count = priority_job_ids_counter.get(job_id, 0)
            if planned_count > 0:
                completed_count = completed_exec_counts.get(job_id, 0)
                if completed_count < planned_count:
                    # Use cumulative n' for adherence normalization (consistency with eta1/2/3)
                    value += weights.get('adherence', 0) / n_prime_cumulative

            if value > max_marginal_gain:
                max_marginal_gain = value
                best_job_id = job_id
                # Store details for debug logging
                _best_delta_obj1 = delta_obj1
                _best_delta_obj2 = delta_obj2
                _best_delta_obj3 = delta_obj3
                _best_delta_obj4 = delta_obj4
                _best_job_exec = job.get('executive', 'UNKNOWN')

        # === DEBUG: Log best job selection details every 100 time steps ===
        if current_time % 100 == 0 and best_job_id is not None:
            _exec_str = _best_job_exec if isinstance(_best_job_exec, str) else max(_best_job_exec, key=_best_job_exec.get)
            _is_priority = priority_job_ids_counter.get(best_job_id, 0) > 0
            print(f"    Selected: {best_job_id} (exec={_exec_str}, priority={_is_priority})")
            print(f"      Δobj1={alpha1}*{_best_delta_obj1:.6f}={alpha1*_best_delta_obj1:.6f}, Δobj2={alpha2}*{_best_delta_obj2:.6f}={alpha2*_best_delta_obj2:.6f}")
            print(f"      Δobj3={alpha3}*{_best_delta_obj3:.6f}={alpha3*_best_delta_obj3:.6f}, Δobj4={alpha4}*{_best_delta_obj4:.6f}={alpha4*_best_delta_obj4:.6f}")
            print(f"      Total gain: {max_marginal_gain:.6f}")

        return best_job_id

    return job_selector


def _build_rollout_forecast_state(
        job_lookup: Dict[str, Dict],
        time_steps: int,
        weather_forecasts: Dict[int, Dict[str, np.ndarray]],
        realized_weather: Dict[int, Any],
        forecast_origin_t: int,
        preserve_actual_job_metadata: bool = False,
        use_realized_pwv_forecast: bool = False,
        use_realized_rms_forecast: bool = False,
) -> Tuple[Dict[int, Tuple[float, float]], List[Dict], Dict[str, Dict]]:
    """Build synthetic weather and rollout job copies for rollout policies."""
    forecast = weather_forecasts.get(forecast_origin_t, {})
    synthetic_weather: Dict[int, Tuple[float, float]] = {}
    pwv_arr = forecast.get('pwv_mean')
    for k in range(time_steps):
        if k <= forecast_origin_t:
            synthetic_weather[k] = realized_weather.get(k, (np.nan, np.nan))
        else:
            realized_pwv, realized_rms = realized_weather.get(k, (np.nan, np.nan))
            forecast_pwv = float(pwv_arr[k]) if pwv_arr is not None and k < len(pwv_arr) else np.nan
            forecast_rms, _ = get_rms_forecast_value(forecast, forecast_origin_t, k)
            pwv_val = realized_pwv if use_realized_pwv_forecast else forecast_pwv
            rms_val = realized_rms if use_realized_rms_forecast else forecast_rms
            synthetic_weather[k] = (float(pwv_val), float(rms_val))

    sim_jobs: List[Dict] = []
    sim_job_info_map: Dict[str, Dict] = {}
    for jid, job in job_lookup.items():
        sim_job = dict(job)
        if not preserve_actual_job_metadata:
            forecast_available, forecast_pwv, forecast_rms, _ = _resolve_forecast_metadata_for_time(
                job, forecast_origin_t
            )
            sim_job['available'] = forecast_available
            sim_job['pwv_thresholds'] = forecast_pwv
            sim_job['rms_threshold'] = forecast_rms
        sim_jobs.append(sim_job)
        sim_job_info_map[jid] = sim_job

    return synthetic_weather, sim_jobs, sim_job_info_map


def _is_job_runnable_at_time(
        job: Dict,
        current_time: int,
        time_steps: int,
        weather_state: Dict[int, Tuple[float, float]],
        unusable_indices: set,
) -> bool:
    """Check whether a job is executable at a specific time under the provided weather."""
    if current_time not in job.get('available', []):
        return False
    if current_time + job['length'] > time_steps:
        return False
    pwv_now, rms_now = weather_state.get(current_time, (np.nan, np.nan))
    if np.isnan(pwv_now) or np.isnan(rms_now):
        return False
    pwv_thresh = job.get("pwv_thresholds", {}).get(current_time, np.inf)
    rms_thresh = job.get("rms_threshold", -np.inf)
    if pwv_now > pwv_thresh or rms_now < rms_thresh:
        return False
    if any((current_time + offset) in unusable_indices for offset in range(job['length'])):
        return False
    return True


def _forecast_runnable_start_times_allow_horizon_overrun(
        job: Dict,
        earliest_start: int,
        local_horizon_end: int,
        total_time_steps: int,
        synthetic_weather: Dict[int, Tuple[float, float]],
        unusable_indices: set,
        occupied_indices: set,
) -> List[int]:
    """Enumerate future start times that begin within the local horizon and finish within the full cycle."""
    feasible_starts: List[int] = []
    for start_t in sorted(job.get('available', [])):
        if start_t < earliest_start:
            continue
        if start_t >= local_horizon_end:
            continue
        if start_t + job['length'] > total_time_steps:
            continue
        if any((start_t + offset) in occupied_indices for offset in range(job['length'])):
            continue
        blocked = False
        for offset in range(job['length']):
            tt = start_t + offset
            if tt in unusable_indices:
                blocked = True
                break
        if blocked:
            continue

        pwv_now, rms_now = synthetic_weather.get(start_t, (np.nan, np.nan))
        if np.isnan(pwv_now) or np.isnan(rms_now):
            continue
        pwv_thresh = job.get("pwv_thresholds", {}).get(start_t, np.inf)
        rms_thresh = job.get("rms_threshold", -np.inf)
        if pwv_now > pwv_thresh or rms_now < rms_thresh:
            continue
        feasible_starts.append(start_t)
    return feasible_starts


def _job_executive_memberships(job: Dict) -> List[str]:
    """Return every executive bucket a job should count toward for pruning."""
    executive_info = job.get('executive')
    if isinstance(executive_info, str):
        return [executive_info]
    if isinstance(executive_info, dict):
        return sorted(
            exec_name
            for exec_name, frac in executive_info.items()
            if float(frac) > 0.0
        )
    return []


def _prune_gurobi_candidates_by_exec_grade(
        ordered_candidate_ids: List[str],
        job_lookup: Dict[str, Dict],
        max_candidates_per_executive_by_grade: Optional[int],
        fill_to_total_candidates: Optional[int] = None,
) -> List[str]:
    """Keep the union of the top-k candidate jobs per executive, optionally filled toward a target total."""
    if max_candidates_per_executive_by_grade is None:
        return list(ordered_candidate_ids)
    if max_candidates_per_executive_by_grade <= 0:
        return []

    job_ids_by_exec: Dict[str, List[str]] = defaultdict(list)
    selected_ids = set()
    unbucketed_ids: List[str] = []
    for job_id in ordered_candidate_ids:
        memberships = _job_executive_memberships(job_lookup[job_id])
        if not memberships:
            unbucketed_ids.append(job_id)
            continue
        for exec_name in memberships:
            job_ids_by_exec[exec_name].append(job_id)

    for job_ids in job_ids_by_exec.values():
        ranked_ids = sorted(
            job_ids,
            key=lambda jid: (-_grade_rank(job_lookup[jid].get('grade')), jid),
        )
        selected_ids.update(ranked_ids[:max_candidates_per_executive_by_grade])

    if fill_to_total_candidates is None:
        selected_ids.update(unbucketed_ids)
    elif fill_to_total_candidates > len(selected_ids):
        ranked_leftovers = sorted(
            (jid for jid in ordered_candidate_ids if jid not in selected_ids),
            key=lambda jid: (-_grade_rank(job_lookup[jid].get('grade')), jid),
        )
        needed = fill_to_total_candidates - len(selected_ids)
        selected_ids.update(ranked_leftovers[:needed])

    return [job_id for job_id in ordered_candidate_ids if job_id in selected_ids]


def _compute_single_job_marginal_gain(
        job: Dict,
        remaining_jobs: List[Dict],
        projects: List[Dict],
        cumulative_exec_time: Dict[str, float],
        completed_exec_counts: Counter,
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        weights: Dict[str, float],
        cumulative_observable_time: int,
        job_info_map: Dict[str, Dict],
        project_bonus_mode: str = "final_job",
        project_bonus_ramp_ratio: float = 1.2,
        job_weight_bonus_by_id: Optional[Dict[str, float]] = None,
) -> float:
    """Compute the standard greedy marginal gain for a single candidate job."""
    n_prime = cumulative_observable_time if cumulative_observable_time > 0 else 1
    eta1 = _compute_eta1_topk(remaining_jobs, n_prime, execs_key='total_execs')
    eta2 = _compute_eta2_topk(projects, n_prime)
    eta3 = n_prime

    if 'obs_completion' in weights or 'proj_completion' in weights:
        alpha1 = weights.get('obs_completion', 0)
        alpha2 = weights.get('proj_completion', 0)
    else:
        alpha1 = weights.get('sb_A', 0) + weights.get('sb_B', 0) + weights.get('sb_C', 0)
        alpha2 = weights.get('proj_A', 0) + weights.get('proj_B', 0) + weights.get('proj_C', 0)
    alpha3 = weights.get('utilization', 0)
    alpha4 = weights.get('eb_penalty', 0)

    eta4_eb = sum(cumulative_exec_time.values()) or 1.0
    current_eb_penalty = _calculate_eb_squared_penalty(
        cumulative_exec_time, executive_quotas_frac, eta4_eb
    )

    project_lookup = {p['project_id']: p for p in projects}
    sb_to_project = {sb_id: p['project_id'] for p in projects for sb_id in p.get('job_ids', [])}

    job_id = job['job_id']
    if job_weight_bonus_by_id is None:
        job_weight_bonus_by_id = {}
    wp = job.get('weight', 0) + job_weight_bonus_by_id.get(job_id, 0.0)
    lb = job['length']

    delta_obj1 = wp / eta1 if eta1 > 0 else 0.0
    delta_obj2 = 0.0
    project_id = sb_to_project.get(job_id)
    if project_id:
        project = project_lookup[project_id]
        if project_bonus_mode == "final_job":
            would_complete = all(
                (completed_exec_counts.get(sb_id, 0) + (1 if sb_id == job_id else 0))
                >= job_info_map[sb_id].get('total_execs', 1)
                for sb_id in project['job_ids']
                if sb_id in job_info_map
            )
            if would_complete:
                delta_obj2 = wp / eta2 if eta2 > 0 else 0.0
        elif project_bonus_mode == "geometric_ramp":
            delta_obj2 = _compute_geometric_project_bonus(
                project=project,
                job_id=job_id,
                completed_exec_counts=completed_exec_counts,
                job_info_map=job_info_map,
                eta2=eta2,
                project_bonus_ramp_ratio=project_bonus_ramp_ratio,
            )
        else:
            raise ValueError(f"Unknown project_bonus_mode: {project_bonus_mode}")

    delta_obj3 = lb / eta3 if eta3 > 0 else 0.0

    future_exec_time = dict(cumulative_exec_time)
    if isinstance(job.get('executive'), str):
        future_exec_time[job['executive']] = future_exec_time.get(job['executive'], 0) + lb
    elif isinstance(job.get('executive'), dict):
        for exec_name, frac in job['executive'].items():
            future_exec_time[exec_name] = future_exec_time.get(exec_name, 0) + lb * frac
    future_eb_penalty = _calculate_eb_squared_penalty(
        future_exec_time, executive_quotas_frac, sum(future_exec_time.values()) or 1.0
    )
    delta_obj4 = current_eb_penalty - future_eb_penalty

    return alpha1 * delta_obj1 + alpha2 * delta_obj2 + alpha3 * delta_obj3 + alpha4 * delta_obj4


def _paper_objective_weight_terms(weights: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Return the paper objective coefficients regardless of legacy/new key style."""
    if 'obs_completion' in weights or 'proj_completion' in weights:
        alpha1 = weights.get('obs_completion', 0.0)
        alpha2 = weights.get('proj_completion', 0.0)
    else:
        alpha1 = weights.get('sb_A', 0.0) + weights.get('sb_B', 0.0) + weights.get('sb_C', 0.0)
        alpha2 = weights.get('proj_A', 0.0) + weights.get('proj_B', 0.0) + weights.get('proj_C', 0.0)
    alpha3 = weights.get('utilization', 0.0)
    alpha4 = weights.get('eb_penalty', 0.0)
    return alpha1, alpha2, alpha3, alpha4


def _piecewise_square_surrogate(abs_shortfall: float) -> float:
    """Match the Prophet rollout tangent-envelope surrogate for shortfall^2."""
    breakpoints = [0.001, 0.003, 0.005, 0.008, 0.012, 0.02, 0.035, 0.05, 0.08, 0.12, 0.20, 0.40, 0.9]
    return max(
        0.0,
        max((2.0 * a * abs_shortfall - a * a for a in breakpoints), default=0.0),
    )


def _evaluate_direct_rollout_objective(
        sequence: List[Tuple[str, int]],
        job_lookup: Dict[str, Dict],
        projects: List[Dict],
        exec_time_used: Dict[str, float],
        completed_exec_counts: Counter,
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        weights: Dict[str, float],
        cumulative_observable_time: int,
        horizon_observable_slots: int,
        horizon_end: int,
        use_quadratic_eb: bool = False,
        start_tie_break_scale: float = 1e-9,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate the direct rollout objective on a decoded schedule."""
    alpha1, alpha2, alpha3, alpha4 = _paper_objective_weight_terms(weights)
    n_prime = cumulative_observable_time if cumulative_observable_time > 0 else 1
    remaining_jobs = [job for job in job_lookup.values() if job.get('remaining_execs', 0) > 0]
    eta1 = _compute_eta1_topk(remaining_jobs, n_prime, execs_key='total_execs')
    eta2 = _compute_eta2_topk(projects, n_prime)
    eta3 = n_prime

    selected_counts = Counter(job_id for job_id, _start_t in sequence)
    weighted_obs_completion = 0.0
    total_utilization = 0.0
    period_exec_time: Dict[str, float] = {exec_name: 0.0 for exec_name in executive_quotas_frac.keys()}
    for job_id, _start_t in sequence:
        job = job_lookup[job_id]
        weighted_obs_completion += float(job.get('weight', 0.0))
        total_utilization += float(job.get('length', 0.0))
        if isinstance(job.get('executive'), str):
            period_exec_time[job['executive']] = period_exec_time.get(job['executive'], 0.0) + job['length']
        elif isinstance(job.get('executive'), dict):
            for exec_name, frac in job['executive'].items():
                period_exec_time[exec_name] = period_exec_time.get(exec_name, 0.0) + job['length'] * frac

    obj1 = (weighted_obs_completion / eta1) if eta1 > 0 else 0.0
    obj3 = (total_utilization / eta3) if eta3 > 0 else 0.0

    project_completion_weight = 0.0
    for project in projects:
        remaining_required = sum(
            int(job_lookup[jid].get('remaining_execs', 0))
            for jid in project.get('job_ids', [])
            if jid in job_lookup
        )
        if remaining_required <= 0:
            continue
        if sum(selected_counts.get(jid, 0) for jid in project.get('job_ids', []) if jid in job_lookup) >= remaining_required:
            project_completion_weight += float(project.get('weight', 0.0))
    obj2 = (project_completion_weight / eta2) if eta2 > 0 else 0.0

    total_offset = sum(exec_time_used.values())
    before_eb = 0.0
    if total_offset > 0:
        for exec_name, (target_frac, _) in executive_quotas_frac.items():
            frac_before = exec_time_used.get(exec_name, 0.0) / total_offset
            before_eb += (target_frac - frac_before) ** 2
    else:
        before_eb = sum(target_frac ** 2 for target_frac, _ in executive_quotas_frac.values())

    after_exec_time = {exec_name: float(value) for exec_name, value in exec_time_used.items()}
    for exec_name, value in period_exec_time.items():
        after_exec_time[exec_name] = after_exec_time.get(exec_name, 0.0) + float(value)

    S_actual = sum(after_exec_time.values())
    S_est = max(total_offset + horizon_observable_slots, 1)
    if use_quadratic_eb:
        eb_after_model = sum(
            (
                target_frac
                - (after_exec_time.get(exec_name, 0.0) / S_est if S_est > 0 else 0.0)
            ) ** 2
            for exec_name, (target_frac, _) in executive_quotas_frac.items()
        )
    else:
        eb_after_model = 0.0
        for exec_name, (target_frac, _) in executive_quotas_frac.items():
            if S_actual > 0:
                shortfall = (target_frac * S_actual - after_exec_time.get(exec_name, 0.0)) / S_actual
            else:
                shortfall = 0.0
            shortfall = max(-1.0, min(1.0, shortfall))
            eb_after_model += _piecewise_square_surrogate(abs(shortfall))
    eb_after_eval = _calculate_eb_squared_penalty(
        after_exec_time,
        executive_quotas_frac,
        S_actual or 1.0,
    )
    obj4_model = before_eb - eb_after_model
    obj4_eval = before_eb - eb_after_eval

    tie_break = start_tie_break_scale * sum(max(horizon_end - start_t, 0) for _job_id, start_t in sequence)
    model_score = alpha1 * obj1 + alpha2 * obj2 + alpha3 * obj3 + alpha4 * obj4_model + tie_break
    evaluator_score = alpha1 * obj1 + alpha2 * obj2 + alpha3 * obj3 + alpha4 * obj4_eval + tie_break

    return model_score, {
        "alpha1": alpha1,
        "alpha2": alpha2,
        "alpha3": alpha3,
        "alpha4": alpha4,
        "eta1": eta1,
        "eta2": eta2,
        "eta3": eta3,
        "obj1": obj1,
        "obj2": obj2,
        "obj3": obj3,
        "obj4_model": obj4_model,
        "obj4_eval": obj4_eval,
        "before_eb": before_eb,
        "eb_after_model": eb_after_model,
        "eb_after_eval": eb_after_eval,
        "tie_break": tie_break,
        "model_score": model_score,
        "evaluator_score": evaluator_score,
        "weighted_obs_completion": weighted_obs_completion,
        "project_completion_weight": project_completion_weight,
        "total_utilization": total_utilization,
        "S_actual": S_actual,
        "S_est": float(S_est),
        "horizon_observable_slots": int(horizon_observable_slots),
        "sequence_length": len(sequence),
    }


def _solve_gurobi_job_sequence_rollout(
        job_lookup: Dict[str, Dict],
        projects: List[Dict],
        time_steps: int,
        weather_forecasts: Dict[int, Dict[str, np.ndarray]],
        realized_weather: Dict[int, Any],
        unusable_indices: set,
        forecast_origin_t: int,
        sequence_horizon_steps: int,
        exec_time_used: Dict[str, float],
        completed_exec_counts: Counter,
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        weights: Dict[str, float],
        cumulative_observable_time: int,
        oracle_use_actual_job_metadata: bool = False,
        use_realized_pwv_forecast: bool = False,
        use_realized_rms_forecast: bool = False,
        time_limit_seconds: Optional[float] = None,
        max_candidates_per_executive_by_grade: Optional[int] = None,
        fill_to_total_candidates: Optional[int] = None,
        scarcity_weight_bonus_scale: float = 0.0,
        use_quadratic_eb: bool = False,
        gurobi_verbose: bool = False,
        gurobi_log_dir: Optional[str] = None,
) -> Tuple[Optional[List[Tuple[str, int]]], float, Dict[str, Any]]:
    """Solve a short-horizon rollout with a direct Gurobi scheduling model."""
    solve_started = time.perf_counter()
    forecast_state_started = time.perf_counter()
    synthetic_weather, sim_jobs, _ = _build_rollout_forecast_state(
        job_lookup=job_lookup,
        time_steps=time_steps,
        weather_forecasts=weather_forecasts,
        realized_weather=realized_weather,
        forecast_origin_t=forecast_origin_t,
        preserve_actual_job_metadata=oracle_use_actual_job_metadata,
        use_realized_pwv_forecast=use_realized_pwv_forecast,
        use_realized_rms_forecast=use_realized_rms_forecast,
    )
    forecast_state_elapsed = time.perf_counter() - forecast_state_started

    def _base_solver_stats(status: Any, terminated_early: bool = False) -> Dict[str, Any]:
        return {
            "candidate_jobs": 0,
            "candidate_jobs_before_prune": 0,
            "total_feasible_starts_before_prune": 0,
            "total_feasible_starts": 0,
            "state_count": 0,
            "action_arc_count": 0,
            "arc_count": 0,
            "model_var_count": 0,
            "model_constr_count": 0,
            "status": status,
            "sol_count": 0,
            "objective_value": None,
            "terminated_early": terminated_early,
            "forecast_state_elapsed": forecast_state_elapsed,
            "candidate_enumeration_elapsed": 0.0,
            "candidate_prune_elapsed": 0.0,
            "graph_build_elapsed": 0.0,
            "model_build_elapsed": 0.0,
            "objective_build_elapsed": 0.0,
            "optimize_elapsed": 0.0,
            "gurobi_runtime": 0.0,
            "postprocess_elapsed": 0.0,
            "decision_var_count": 0,
            "project_var_count": 0,
            "eb_var_count": 0,
            "overlap_constr_count": 0,
            "solve_elapsed": time.perf_counter() - solve_started,
        }

    horizon_end = min(time_steps, forecast_origin_t + max(1, sequence_horizon_steps))
    if horizon_end <= forecast_origin_t:
        return None, -float('inf'), _base_solver_stats("EMPTY_HORIZON")

    remaining_execs_by_job = {jid: job.get('remaining_execs', 0) for jid, job in job_lookup.items()}
    sim_job_by_id = {job['job_id']: job for job in sim_jobs}
    feasible_starts_by_job: Dict[str, Tuple[int, ...]] = {}
    ordered_candidate_ids: List[str] = []
    candidate_enumeration_started = time.perf_counter()
    for job in sorted(sim_jobs, key=lambda item: item['job_id']):
        jid = job['job_id']
        if remaining_execs_by_job.get(jid, 0) <= 0:
            continue
        feasible_starts = _forecast_runnable_start_times_allow_horizon_overrun(
            job=job,
            earliest_start=forecast_origin_t,
            local_horizon_end=horizon_end,
            total_time_steps=time_steps,
            synthetic_weather=synthetic_weather,
            unusable_indices=unusable_indices,
            occupied_indices=set(),
        )
        if feasible_starts and feasible_starts[0] == forecast_origin_t and not _is_job_runnable_at_time(
                job=job_lookup[jid],
                current_time=forecast_origin_t,
                time_steps=time_steps,
                weather_state=realized_weather,
                unusable_indices=unusable_indices,
        ):
            feasible_starts = [start_t for start_t in feasible_starts if start_t != forecast_origin_t]
        if feasible_starts:
            ordered_candidate_ids.append(jid)
            feasible_starts_by_job[jid] = tuple(sorted(feasible_starts))
    candidate_enumeration_elapsed = time.perf_counter() - candidate_enumeration_started
    total_feasible_starts_before_prune = int(sum(len(starts) for starts in feasible_starts_by_job.values()))

    candidate_jobs_before_prune = len(ordered_candidate_ids)
    candidate_prune_started = time.perf_counter()
    ordered_candidate_ids = _prune_gurobi_candidates_by_exec_grade(
        ordered_candidate_ids=ordered_candidate_ids,
        job_lookup=job_lookup,
        max_candidates_per_executive_by_grade=max_candidates_per_executive_by_grade,
        fill_to_total_candidates=fill_to_total_candidates,
    )
    feasible_starts_by_job = {
        jid: feasible_starts_by_job[jid]
        for jid in ordered_candidate_ids
    }
    candidate_prune_elapsed = time.perf_counter() - candidate_prune_started
    total_feasible_starts = int(sum(len(starts) for starts in feasible_starts_by_job.values()))

    if not ordered_candidate_ids:
        stats = _base_solver_stats("NO_CANDIDATES")
        stats.update({
            "candidate_jobs_before_prune": candidate_jobs_before_prune,
            "total_feasible_starts_before_prune": total_feasible_starts_before_prune,
            "candidate_enumeration_elapsed": candidate_enumeration_elapsed,
            "candidate_prune_elapsed": candidate_prune_elapsed,
        })
        return None, -float('inf'), stats

    if gurobi_verbose and scarcity_weight_bonus_scale > 0:
        print(
            f"  [ROLLOUT-DIRECT @t={forecast_origin_t}] "
            f"scarcity_weight_bonus_scale={scarcity_weight_bonus_scale} is ignored by the direct objective.",
            flush=True,
        )

    horizon_observable_slots = sum(
        1
        for tt in range(forecast_origin_t, horizon_end)
        if tt not in unusable_indices
        and not (
            np.isnan(synthetic_weather.get(tt, (np.nan, np.nan))[0])
            or np.isnan(synthetic_weather.get(tt, (np.nan, np.nan))[1])
        )
    )

    model_build_started = time.perf_counter()
    model = Model("greedy_sequence_gurobi_rollout_direct")
    model.setParam("OutputFlag", 1 if gurobi_verbose else 0)
    model.setParam("LogToConsole", 1 if gurobi_verbose else 0)
    if gurobi_verbose:
        model.setParam("DisplayInterval", 5)
    if gurobi_log_dir:
        os.makedirs(gurobi_log_dir, exist_ok=True)
        model.setParam("LogFile", os.path.join(
            gurobi_log_dir,
            f"greedy_sequence_gurobi_t{forecast_origin_t}.log",
        ))
    model.setParam("MIPGap", 0.0)
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("NumericFocus", 1)
    if not use_quadratic_eb:
        model.setParam("NonConvex", 2)
    if time_limit_seconds is not None and time_limit_seconds > 0:
        model.setParam("TimeLimit", float(time_limit_seconds))

    x: Dict[Tuple[str, int], Any] = {}
    for jid in ordered_candidate_ids:
        for start_t in feasible_starts_by_job.get(jid, ()):
            x[(jid, int(start_t))] = model.addVar(vtype=GRB.BINARY, name=f"roll_x_{jid}_{int(start_t)}")
    model.update()

    if not x:
        stats = _base_solver_stats("NO_DECISION_VARS")
        stats.update({
            "candidate_jobs": len(ordered_candidate_ids),
            "candidate_jobs_before_prune": candidate_jobs_before_prune,
            "total_feasible_starts_before_prune": total_feasible_starts_before_prune,
            "total_feasible_starts": total_feasible_starts,
            "candidate_enumeration_elapsed": candidate_enumeration_elapsed,
            "candidate_prune_elapsed": candidate_prune_elapsed,
            "decision_var_count": 0,
        })
        return None, -float('inf'), stats

    if gurobi_verbose:
        print(
            f"  [ROLLOUT-DIRECT @t={forecast_origin_t}] "
            f"candidates={len(ordered_candidate_ids)}/{candidate_jobs_before_prune}, "
            f"feasible_starts={total_feasible_starts}/{total_feasible_starts_before_prune}, "
            f"decision_vars={len(x)}, use_quadratic_eb={use_quadratic_eb}, "
            f"horizon_end={horizon_end}, horizon_observable_slots={horizon_observable_slots}",
            flush=True,
        )

    selected_count_exprs: Dict[str, Any] = {}
    for jid in ordered_candidate_ids:
        starts = feasible_starts_by_job.get(jid, ())
        selected_count_exprs[jid] = quicksum(x[(jid, start_t)] for start_t in starts if (jid, start_t) in x)
        model.addConstr(
            selected_count_exprs[jid] <= remaining_execs_by_job.get(jid, 0),
            name=f"rollout_job_cap_{jid}",
        )

    overlap_constr_count = 0
    for tt in range(forecast_origin_t, time_steps):
        covering_vars = [
            x[(jid, start_t)]
            for jid in ordered_candidate_ids
            for start_t in feasible_starts_by_job.get(jid, ())
            if (jid, start_t) in x and start_t <= tt < start_t + int(sim_job_by_id[jid]['length'])
        ]
        if covering_vars:
            model.addConstr(quicksum(covering_vars) <= 1, name=f"rollout_overlap_{tt}")
            overlap_constr_count += 1

    objective_build_started = time.perf_counter()
    alpha1, alpha2, alpha3, alpha4 = _paper_objective_weight_terms(weights)
    n_prime = cumulative_observable_time if cumulative_observable_time > 0 else 1
    remaining_job_snapshot = [job for job in job_lookup.values() if job.get('remaining_execs', 0) > 0]
    eta1 = _compute_eta1_topk(remaining_job_snapshot, n_prime, execs_key='total_execs')
    eta2 = _compute_eta2_topk(projects, n_prime)
    eta3 = n_prime

    weighted_obs_completion = quicksum(
        float(job_lookup[jid].get('weight', 0.0)) * x[(jid, start_t)]
        for (jid, start_t) in x.keys()
    )
    obj1 = weighted_obs_completion / eta1 if eta1 > 0 else 0.0

    total_utilization = quicksum(
        float(sim_job_by_id[jid]['length']) * x[(jid, start_t)]
        for (jid, start_t) in x.keys()
    )
    obj3 = total_utilization / eta3 if eta3 > 0 else 0.0

    zs: Dict[str, Any] = {}
    weighted_project_delta = 0
    for project in projects:
        pid = project.get('project_id')
        total_required = sum(
            int(job_lookup[jid].get('remaining_execs', 0))
            for jid in project.get('job_ids', [])
            if jid in job_lookup
        )
        if total_required <= 0:
            continue
        z = model.addVar(vtype=GRB.BINARY, name=f"rollout_z_{pid}")
        zs[pid] = z
        sum_project_sched = quicksum(
            selected_count_exprs.get(jid, 0)
            for jid in project.get('job_ids', [])
            if jid in job_lookup
        )
        model.addConstr(
            sum_project_sched >= total_required * z,
            name=f"rollout_project_complete_lb_{pid}",
        )
        model.addConstr(
            z >= sum_project_sched - (total_required - 1) - 1e-4,
            name=f"rollout_project_complete_force_{pid}",
        )
        weighted_project_delta += float(project.get('weight', 0.0)) * z
    obj2 = weighted_project_delta / eta2 if eta2 > 0 else 0.0

    exec_time_exprs: Dict[str, Any] = {}
    for exec_name in executive_quotas_frac.keys():
        time_from_single = quicksum(
            float(sim_job_by_id[jid]['length']) * x[(jid, start_t)]
            for (jid, start_t) in x.keys()
            if isinstance(job_lookup[jid].get('executive'), str) and job_lookup[jid].get('executive') == exec_name
        )
        time_from_fractional = quicksum(
            float(sim_job_by_id[jid]['length']) * float(job_lookup[jid]['executive'][exec_name]) * x[(jid, start_t)]
            for (jid, start_t) in x.keys()
            if isinstance(job_lookup[jid].get('executive'), dict) and exec_name in job_lookup[jid]['executive']
        )
        exec_time_exprs[exec_name] = time_from_single + time_from_fractional

    total_offset = sum(exec_time_used.values())
    E_aux: Dict[str, Any] = {}
    for exec_name in executive_quotas_frac.keys():
        exec_offset = exec_time_used.get(exec_name, 0.0)
        E_aux[exec_name] = model.addVar(name=f"rollout_E_{exec_name}", lb=0.0)
        model.addConstr(
            E_aux[exec_name] == exec_offset + exec_time_exprs.get(exec_name, 0.0),
            name=f"rollout_E_def_{exec_name}",
        )
    S_var = model.addVar(name="rollout_S_total_exec", lb=0.0)
    model.addConstr(S_var == total_offset + total_utilization, name="rollout_S_def")

    eb_before = 0.0
    if total_offset > 0:
        for exec_name, (target_frac, _) in executive_quotas_frac.items():
            frac_before = exec_time_used.get(exec_name, 0.0) / total_offset
            eb_before += (target_frac - frac_before) ** 2
    else:
        eb_before = sum(target_frac ** 2 for target_frac, _ in executive_quotas_frac.values())

    eb_var_count = len(E_aux) + 1
    if use_quadratic_eb:
        shortfall = {}
        for exec_name, (target_frac, _) in executive_quotas_frac.items():
            shortfall[exec_name] = model.addVar(name=f"rollout_shortfall_{exec_name}", lb=-1.0, ub=1.0)
            model.addConstr(
                E_aux[exec_name] + max(total_offset + horizon_observable_slots, 1) * shortfall[exec_name]
                == target_frac * max(total_offset + horizon_observable_slots, 1),
                name=f"rollout_shortfall_def_{exec_name}",
            )
        eb_after_expr = quicksum(shortfall[exec_name] * shortfall[exec_name] for exec_name in executive_quotas_frac.keys())
        eb_var_count += len(shortfall)
    else:
        shortfall = {}
        shortfall_pos = {}
        shortfall_neg = {}
        abs_shortfall = {}
        z_sq = {}
        breakpoints = [0.0, 0.001, 0.003, 0.005, 0.008, 0.012, 0.02, 0.035, 0.05, 0.08, 0.12, 0.20, 0.40, 0.9]
        for exec_name, (target_frac, _) in executive_quotas_frac.items():
            shortfall[exec_name] = model.addVar(name=f"rollout_shortfall_{exec_name}", lb=-1.0, ub=1.0)
            shortfall_pos[exec_name] = model.addVar(name=f"rollout_shortfall_pos_{exec_name}", lb=0.0)
            shortfall_neg[exec_name] = model.addVar(name=f"rollout_shortfall_neg_{exec_name}", lb=0.0)
            abs_shortfall[exec_name] = model.addVar(name=f"rollout_abs_shortfall_{exec_name}", lb=0.0)
            z_sq[exec_name] = model.addVar(name=f"rollout_z_sq_{exec_name}", lb=0.0)
            model.addConstr(
                E_aux[exec_name] + S_var * shortfall[exec_name] == target_frac * S_var,
                name=f"rollout_shortfall_def_{exec_name}",
            )
            model.addConstr(
                shortfall[exec_name] == shortfall_pos[exec_name] - shortfall_neg[exec_name],
                name=f"rollout_shortfall_split_{exec_name}",
            )
            model.addConstr(
                abs_shortfall[exec_name] == shortfall_pos[exec_name] + shortfall_neg[exec_name],
                name=f"rollout_abs_shortfall_def_{exec_name}",
            )
            for a in breakpoints:
                if a > 0:
                    model.addConstr(
                        z_sq[exec_name] >= 2 * a * abs_shortfall[exec_name] - a * a,
                        name=f"rollout_tangent_eb_{exec_name}_{a}",
                    )
        eb_after_expr = quicksum(z_sq[exec_name] for exec_name in executive_quotas_frac.keys())
        eb_var_count += len(shortfall) + len(shortfall_pos) + len(shortfall_neg) + len(abs_shortfall) + len(z_sq)

    obj4 = eb_before - eb_after_expr
    tie_break_expr = quicksum(
        1e-9 * float(horizon_end - start_t) * x[(jid, start_t)]
        for (jid, start_t) in x.keys()
    )
    model.setObjective(
        alpha1 * obj1 +
        alpha2 * obj2 +
        alpha3 * obj3 +
        alpha4 * obj4 +
        tie_break_expr,
        GRB.MAXIMIZE,
    )
    model.update()
    objective_build_elapsed = time.perf_counter() - objective_build_started
    model_build_elapsed = time.perf_counter() - model_build_started

    if gurobi_verbose:
        print(
            f"  [ROLLOUT-DIRECT @t={forecast_origin_t}] "
            f"eta1={eta1:.4f}, eta2={eta2:.4f}, eta3={eta3:.4f}, "
            f"eb_before={eb_before:.6f}, objective_build={objective_build_elapsed:.3f}s",
            flush=True,
        )

    optimize_started = time.perf_counter()
    model.optimize()
    optimize_elapsed = time.perf_counter() - optimize_started

    postprocess_started = time.perf_counter()
    sequence: List[Tuple[str, int]] = []
    score_breakdown: Dict[str, Any] = {}
    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        sequence = sorted(
            [
                (jid, start_t)
                for (jid, start_t), var in x.items()
                if var.X > 0.5
            ],
            key=lambda item: (item[1], item[0]),
        )

    best_score = -float('inf')
    if sequence:
        best_score, score_breakdown = _evaluate_direct_rollout_objective(
            sequence=sequence,
            job_lookup=job_lookup,
            projects=projects,
            exec_time_used=exec_time_used,
            completed_exec_counts=completed_exec_counts,
            executive_quotas_frac=executive_quotas_frac,
            weights=weights,
            cumulative_observable_time=cumulative_observable_time,
            horizon_observable_slots=horizon_observable_slots,
            horizon_end=horizon_end,
            use_quadratic_eb=use_quadratic_eb,
        )
    postprocess_elapsed = time.perf_counter() - postprocess_started

    if gurobi_verbose and model.SolCount > 0:
        print(
            f"  [ROLLOUT-DIRECT @t={forecast_origin_t}] "
            f"status={int(model.Status)}, sol_count={int(model.SolCount)}, "
            f"selected_jobs={len(sequence)}, obj={float(model.ObjVal):.6f}",
            flush=True,
        )
        if score_breakdown:
            print(
                f"  [ROLLOUT-DIRECT @t={forecast_origin_t}] "
                f"obj1={score_breakdown['obj1']:.6f}, obj2={score_breakdown['obj2']:.6f}, "
                f"obj3={score_breakdown['obj3']:.6f}, obj4_model={score_breakdown['obj4_model']:.6f}, "
                f"obj4_eval={score_breakdown['obj4_eval']:.6f}, tie_break={score_breakdown['tie_break']:.6e}",
                flush=True,
            )
            print(
                f"  [ROLLOUT-DIRECT @t={forecast_origin_t}] "
                f"score(model)={score_breakdown['model_score']:.6f}, "
                f"score(eval)={score_breakdown['evaluator_score']:.6f}, "
                f"S_est={score_breakdown['S_est']:.1f}, S_actual={score_breakdown['S_actual']:.1f}",
                flush=True,
            )

    return (sequence or None), best_score, {
        "candidate_jobs": len(ordered_candidate_ids),
        "candidate_jobs_before_prune": candidate_jobs_before_prune,
        "total_feasible_starts_before_prune": total_feasible_starts_before_prune,
        "total_feasible_starts": total_feasible_starts,
        "state_count": 0,
        "action_arc_count": 0,
        "arc_count": 0,
        "model_var_count": int(model.NumVars),
        "model_constr_count": int(model.NumConstrs),
        "status": int(model.Status),
        "sol_count": int(model.SolCount),
        "objective_value": float(model.ObjVal) if model.SolCount > 0 else None,
        "terminated_early": bool(model.Status == GRB.TIME_LIMIT),
        "forecast_state_elapsed": forecast_state_elapsed,
        "candidate_enumeration_elapsed": candidate_enumeration_elapsed,
        "candidate_prune_elapsed": candidate_prune_elapsed,
        "graph_build_elapsed": 0.0,
        "model_build_elapsed": model_build_elapsed,
        "objective_build_elapsed": objective_build_elapsed,
        "optimize_elapsed": optimize_elapsed,
        "gurobi_runtime": float(getattr(model, "Runtime", 0.0) or 0.0),
        "postprocess_elapsed": postprocess_elapsed,
        "decision_var_count": len(x),
        "project_var_count": len(zs),
        "eb_var_count": eb_var_count,
        "overlap_constr_count": overlap_constr_count,
        "solve_elapsed": time.perf_counter() - solve_started,
    }


def planning_loop_eb_greedy(
        jobs: List[Dict[str, Any]],
        projects: List[Dict[str, Any]],
        realized_weather: Dict[int, Any],
        time_steps: int,
        job_selector_fn: Callable,
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        idx_to_timestamp: Optional[Dict[int, pd.Timestamp]] = None,
        config_calendar: Optional[pd.DataFrame] = None,
        debug: bool = False,
        weights: Optional[Dict[str, float]] = None,
        total_observable_time: Optional[int] = None,
        initial_exec_time_used: Optional[Dict[str, float]] = None,
        selector_mode: str = 'greedy',
        weather_forecasts: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        planning_horizon: int = 72,
        project_bonus_ramp_ratio: float = 1.2,
        prophet_solver_fn: Optional[Callable] = None,
        prophet_daily_job_counters: Optional[Dict[pd.Timestamp, Counter]] = None,
        osco_num_samples: int = 5,
        osco_n_threads: int = -1,
        osco_random_seed: int = None
):
    """
    Planning loop that supports greedy, forecast-aware urgency selectors, and prophet/OSCO-style selectors.
    All modes track cumulative EB and normalization over the full cycle.
    
    When selector_mode='greedy' (default), uses the provided job_selector_fn (factory pattern).
    When selector_mode='mean', 'osco', 'oscocsh', 'oscocshfl', 'oscoucb', or 'oscoucbbetter',
    bypasses job_selector_fn and uses internal helpers with cumulative state.
    
    When integrating with strategic scheduler (Appendix A), pass initial_exec_time_used = V^(c-1)
    (cumulative operational time through previous chunks) and total_observable_time = N_avail^(c).
    
    Args:
        selector_mode: Job selection method ('greedy', 'mean', 'multiplicative', 'additive', 'scarcity_override', 'scarcity_override_no_c_if_ab', 'scarcity_override_project_ramp', 'scarcity_override_ramped_eb', 'executive_scarcity', 'osco', 'oscocsh', 'oscocshfl', 'oscoucb', 'oscoucbbetter')
        weather_forecasts: Weather forecast data (required for forecast-based selectors)
        planning_horizon: Lookahead horizon in time steps (for forecast-based selectors)
        project_bonus_ramp_ratio: Geometric ratio for ramped per-project completion bonus
        prophet_solver_fn: Prophet solver function (for forecast-based selectors that solve subproblems)
        osco_num_samples: Number of Monte Carlo samples (for 'osco')
        osco_n_threads: Number of parallel workers (for 'osco', -1 = auto)
        osco_random_seed: Random seed for OSCO sampling
    """
    # Validate selector mode
    if selector_mode in ('mean', 'multiplicative', 'additive', 'scarcity_override', 'scarcity_override_no_c_if_ab', 'scarcity_override_project_ramp', 'scarcity_override_ramped_eb', 'executive_scarcity', 'osco', 'oscocsh', 'oscocshfl', 'oscoscreen', 'oscoucb', 'oscoucbbetter'):
        if weather_forecasts is None:
            raise ValueError(f"selector_mode='{selector_mode}' requires weather_forecasts")
        if selector_mode in ('mean', 'osco', 'oscocsh', 'oscocshfl', 'oscoscreen', 'oscoucb', 'oscoucbbetter') and prophet_solver_fn is None:
            raise ValueError(f"selector_mode='{selector_mode}' requires prophet_solver_fn")
    if project_bonus_ramp_ratio < 1.0:
        raise ValueError(f"project_bonus_ramp_ratio must be >= 1.0, got {project_bonus_ramp_ratio}")
    
    print(f"\n  Selector mode: {selector_mode}")
    if selector_mode in ('mean', 'multiplicative', 'additive', 'scarcity_override', 'scarcity_override_no_c_if_ab', 'scarcity_override_project_ramp', 'scarcity_override_ramped_eb', 'executive_scarcity', 'osco', 'oscocsh', 'oscocshfl', 'oscoscreen', 'oscoucb', 'oscoucbbetter'):
        print(f"  Planning horizon: {planning_horizon} time steps")
    if selector_mode == 'scarcity_override_project_ramp':
        print(f"  Project bonus ramp ratio: {project_bonus_ramp_ratio}")
    if selector_mode == 'osco':
        print(f"  OSCO samples: {osco_num_samples}, threads: {osco_n_threads}")
    if selector_mode == 'oscocsh':
        print(f"  OSCOCSH samples (per job): {osco_num_samples}, threads: {osco_n_threads}")
        print(f"  (Total budget = osco_num_samples * num_candidates, computed per time step)")
    if selector_mode == 'oscocshfl':
        print(f"  OSCOCSH-FL samples (per job): {osco_num_samples}, threads: {osco_n_threads}")
        print(f"  (Frontload {osco_num_samples // 2}/arm, then SH on top half with remainder)")
    if selector_mode == 'oscoscreen':
        print(f"  SCREEN samples (per job): {osco_num_samples}, threads: {osco_n_threads}")
        print(f"  (Screen n0=5 → shortlist 10 → verify → quantile tie-break)")
    if selector_mode == 'oscoucb':
        print(f"  OSCOUCB samples (per job): {osco_num_samples}, threads: {osco_n_threads}")
        print(f"  (Total budget = osco_num_samples * num_candidates, UCB arm selection)")
    if selector_mode == 'oscoucbbetter':
        print(f"  OSCOUCB-LUCB samples (per job): {osco_num_samples}, threads: {osco_n_threads}")
        print(f"  (Total budget = osco_num_samples * num_candidates, Correlated-Batch LUCB)")
    
    t = 0
    busy_until = 0
    current_schedule_log = []
    successful_schedule_log = []  # Only executions that passed weather check (for value computation)

    # --- Track state: use initial_exec_time_used (Appendix A) when provided ---
    exec_time_used = {k: float(initial_exec_time_used.get(k, 0)) for k in executive_quotas_frac.keys()} \
        if initial_exec_time_used is not None else {k: 0.0 for k in executive_quotas_frac.keys()}
    for j in jobs:  # Ensure all executives are initialized
        if isinstance(j['executive'], str):
            exec_time_used.setdefault(j['executive'], 0.0)
        elif isinstance(j['executive'], dict):
            for exec_name in j['executive']: exec_time_used.setdefault(exec_name, 0.0)

    completed_exec_counts = Counter()
    job_info_map = {j['job_id']: j for j in jobs}  # For quick lookups

    all_unusable_indices = {idx for idx, (pwv, rms) in realized_weather.items() if pd.isna(pwv) or pd.isna(rms)}
    
    # --- OSCO / OSCOCSH random state ---
    osco_rng = np.random.default_rng(osco_random_seed) if selector_mode in ('osco', 'oscocsh', 'oscocshfl', 'oscoscreen', 'oscoucb', 'oscoucbbetter') else None
    
    # --- Timing metrics ---
    loop_wall_start = time.time()
    cumulative_selector_seconds = 0.0
    selector_call_count = 0
    
    # --- Configuration tracking for greedy (always enabled if calendar provided) ---
    config_completions = {}  # Maps config_name -> {job_ids: set, grade_counts: dict}
    current_config = None
    time_to_config = {}  # Initialize empty dict
    config_end_indices = {}  # Track end index for each configuration
    if idx_to_timestamp is not None and config_calendar is not None:
        # Build a mapping from time index to configuration
        for idx, timestamp in idx_to_timestamp.items():
            for _, row in config_calendar.iterrows():
                if row['Start'] <= timestamp <= row['End']:
                    config_name = row['Configuration']
                    time_to_config[idx] = config_name
                    # Track the maximum end index for each configuration
                    if config_name not in config_end_indices:
                        config_end_indices[config_name] = idx
                    else:
                        config_end_indices[config_name] = max(config_end_indices[config_name], idx)
                    break

    greedy_selector_accepts_schedule_log = (
        selector_mode == 'greedy' and
        job_selector_fn is not None and
        'current_schedule_log' in inspect.signature(job_selector_fn).parameters
    )
    greedy_selector_accepts_prophet_counter = (
        selector_mode == 'greedy' and
        job_selector_fn is not None and
        'prophet_remaining_counter' in inspect.signature(job_selector_fn).parameters
    )
    prophet_remaining_counter = Counter()
    loaded_prophet_days = set()

    while t < time_steps:
        if t < busy_until:
            t += 1
            continue

        remaining_jobs_to_consider = [j for j in jobs if j["remaining_execs"] > 0]
        if not remaining_jobs_to_consider: break

        if idx_to_timestamp is not None and prophet_daily_job_counters:
            current_day = idx_to_timestamp[t].normalize()
            if current_day not in loaded_prophet_days:
                prophet_remaining_counter.update(prophet_daily_job_counters.get(current_day, Counter()))
                loaded_prophet_days.add(current_day)

        # --- Select job based on selector_mode ---
        _sel_start = time.time()
        
        if selector_mode == 'greedy':
            selector_kwargs = dict(
                all_remaining_jobs=remaining_jobs_to_consider,
                current_time=t,
                total_time_steps=time_steps,
                observed_weather=realized_weather,
                exec_time_used=exec_time_used,
                completed_exec_counts=completed_exec_counts,
                unusable_future_indices=all_unusable_indices,
            )
            if greedy_selector_accepts_schedule_log:
                selector_kwargs['current_schedule_log'] = current_schedule_log
            if greedy_selector_accepts_prophet_counter:
                selector_kwargs['prophet_remaining_counter'] = prophet_remaining_counter
            selected_job_id = job_selector_fn(**selector_kwargs)
        else:
            raise ValueError(f"Unknown selector_mode: {selector_mode}")
        
        _sel_end = time.time()
        cumulative_selector_seconds += (_sel_end - _sel_start)
        selector_call_count += 1

        if selected_job_id is None:
            t += 1
            continue

        job_to_schedule = job_info_map[selected_job_id]

        # --- Schedule the ATTEMPT, then check for SUCCESS ---
        # 1. Log the attempt and set the telescope to busy
        current_schedule_log.append(f"{selected_job_id}@{t}")
        busy_until = t + job_to_schedule["length"]

        # 2. Check for configuration changes (before checking success)
        # --- Track configuration changes and print summaries ---
        if idx_to_timestamp is not None and config_calendar is not None:
            # Determine which configuration this time slot belongs to
            if t in time_to_config:
                config_name = time_to_config[t]
                if config_name != current_config:
                    # New configuration started - print summary of previous config
                    if current_config is not None:
                        # Import here to avoid circular import
                        from full_year import print_greedy_cumulative_summary
                        if weights is not None and total_observable_time is not None:
                            # Use the last time index of the previous configuration
                            end_idx = config_end_indices.get(current_config, t - 1) + 1
                            print_greedy_cumulative_summary(
                                config_name=current_config,
                                final_schedule=current_schedule_log.copy(),
                                jobs=jobs,
                                projects=projects,
                                realized_weather=realized_weather,
                                total_time_steps=time_steps,
                                weights=weights,
                                executive_quotas_frac=executive_quotas_frac,
                                total_observable_time=total_observable_time,
                                end_idx=end_idx
                            )
                    
                    current_config = config_name
                    if config_name not in config_completions:
                        config_completions[config_name] = {
                            'job_ids': set(),
                            'grade_counts': {'A': 0, 'B': 0, 'C': 0}
                        }

        # 3. After the attempt, perform a retrospective check on the real weather
        # was_successful, reason = _is_execution_successful(job_to_schedule, t, realized_weather, time_steps)
        was_successful = True
        reason = "Success"

        # 4. Only update state if the execution was successful
        if was_successful:
            print(f"  [t={t}] SUCCESS: Scheduled {selected_job_id} was successful.")
            successful_schedule_log.append(f"{selected_job_id}@{t}")
            completed_exec_counts[selected_job_id] += 1
            if prophet_remaining_counter.get(selected_job_id, 0) > 0:
                prophet_remaining_counter[selected_job_id] -= 1
                if prophet_remaining_counter[selected_job_id] <= 0:
                    del prophet_remaining_counter[selected_job_id]
            if isinstance(job_to_schedule["executive"], str):
                exec_time_used[job_to_schedule["executive"]] += job_to_schedule["length"]
            elif isinstance(job_to_schedule["executive"], dict):
                for exec_name, frac in job_to_schedule["executive"].items():
                    exec_time_used[exec_name] += job_to_schedule['length'] * frac
            job_to_schedule['remaining_execs'] -= 1
            
            # --- Track configuration completions (only for successful jobs) ---
            if idx_to_timestamp is not None and config_calendar is not None and t in time_to_config:
                config_name = time_to_config[t]
                if config_name in config_completions:
                    config_completions[config_name]['job_ids'].add(selected_job_id)
                    grade = job_to_schedule.get('grade', 'C')
                    config_completions[config_name]['grade_counts'][grade] += 1
        else:
            print(
                f"  [t={t}] FAILURE: Scheduled {selected_job_id} failed due to {reason}. Executions remaining unchanged.")

        t += 1
    
    # --- Print final configuration summary ---
    if idx_to_timestamp is not None and config_calendar is not None and current_config is not None:
        # Import here to avoid circular import
        from full_year import print_greedy_cumulative_summary
        if weights is not None and total_observable_time is not None:
            end_idx = config_end_indices.get(current_config, time_steps) + 1
            print_greedy_cumulative_summary(
                config_name=current_config,
                final_schedule=current_schedule_log.copy(),
                jobs=jobs,
                projects=projects,
                realized_weather=realized_weather,
                total_time_steps=time_steps,
                weights=weights,
                executive_quotas_frac=executive_quotas_frac,
                total_observable_time=total_observable_time,
                end_idx=end_idx
            )

    # === Summary ===
    loop_wall_elapsed = time.time() - loop_wall_start
    avg_selector_ms = (cumulative_selector_seconds / max(1, selector_call_count)) * 1000.0
    
    print("\n" + "=" * 80)
    print(f"PLANNING LOOP EB SUMMARY (selector_mode={selector_mode})")
    print("=" * 80)
    _total_exec_time = sum(exec_time_used.values())
    print(f"  Total exec time used: {_total_exec_time:.1f} bins")
    print(f"  Total attempts: {len(current_schedule_log)}")
    print(f"  Successful executions: {len(successful_schedule_log)}")
    print(f"  Wall time: {loop_wall_elapsed:.2f}s | "
          f"Selector time: {cumulative_selector_seconds:.2f}s over {selector_call_count} calls "
          f"(avg {avg_selector_ms:.1f} ms/call)")
    print(f"  EB Quotas (targets): {executive_quotas_frac}")
    print(f"\n  {'Exec':<8} | {'Time Used':<12} | {'Actual %':<12} | {'Target Min':<12} | {'Target Max':<12} | {'Status':<10}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10}")
    for _exec in ['NA', 'EA', 'EU', 'CL', 'OTHER']:
        if _exec in executive_quotas_frac:
            _time = exec_time_used.get(_exec, 0)
            _frac = _time / _total_exec_time if _total_exec_time > 0 else 0
            _min_tgt, _max_tgt = executive_quotas_frac[_exec]
            if _frac < _min_tgt - 0.02:
                _status = "UNDER"
            elif _frac > _max_tgt + 0.02:
                _status = "OVER"
            else:
                _status = "OK"
            print(f"  {_exec:<8} | {_time:>12.1f} | {_frac:>12.2%} | {_min_tgt:>12.2%} | {_max_tgt:>12.2%} | {_status:<10}")
    print("=" * 80 + "\n")

    selector_summary_fn = getattr(job_selector_fn, "print_holdout_summary", None)
    if selector_mode == 'greedy' and callable(selector_summary_fn):
        selector_summary_fn(
            completed_exec_counts=completed_exec_counts,
            current_schedule_log=current_schedule_log,
            successful_schedule_log=successful_schedule_log,
        )

    # Compute and return the paper objective value for successful executions
    total_obs_time = total_observable_time if total_observable_time is not None else time_steps
    if weights is not None and successful_schedule_log:
        final_value = compute_paper_objective_value(
            successful_schedule_log, jobs, projects, exec_time_used,
            total_obs_time, weights, executive_quotas_frac
        )
        return final_value, current_schedule_log
    return 0.0, current_schedule_log


def _anchored_week_index(timestamp: pd.Timestamp, cycle_start_timestamp: pd.Timestamp) -> int:
    ts = pd.to_datetime(timestamp, utc=True).normalize()
    cycle_start = pd.to_datetime(cycle_start_timestamp, utc=True).normalize()
    delta_days = max(0, (ts - cycle_start).days)
    return delta_days // 7


def _anchored_week_label(week_index: int) -> str:
    return f"{int(week_index) + 1}A"


def _build_current_week_ab_counter_from_schedule(
        strategic_schedule: List[Dict[str, Any]],
        current_week_label: str,
        job_info_map: Dict[str, Dict[str, Any]],
) -> Counter:
    current_week_counter = Counter()
    for row in strategic_schedule:
        job_id = str(row.get('job_id', ''))
        week_label = str(row.get('week_label', ''))
        if week_label != current_week_label:
            continue
        job = job_info_map.get(job_id)
        if job is None:
            continue
        if str(job.get('grade', '')).strip().upper() not in ('A', 'B'):
            continue
        current_week_counter[job_id] += 1
    return current_week_counter


def _grade_rank(grade: Any) -> int:
    grade_str = str(grade).strip().upper()
    return {"A": 3, "B": 2, "C": 1}.get(grade_str, 0)


def _with_ramped_eb_penalty(
        weights: Dict[str, float],
        current_time: int,
        total_time_steps: int,
        ramp_exponent: float = 50.0,
) -> Dict[str, float]:
    """
    Return a shallow copy of weights with eb_penalty ramped over the cycle.
    """
    ramped_weights = dict(weights)
    final_eb_penalty = weights.get('eb_penalty', 0.0)
    if total_time_steps <= 1:
        progress = 1.0
    else:
        progress = min(max(current_time / (total_time_steps - 1), 0.0), 1.0)
    ramped_weights['eb_penalty'] = final_eb_penalty * (progress ** ramp_exponent)
    return ramped_weights


def _filter_to_highest_available_grade(jobs: List[Dict]) -> List[Dict]:
    """Keep only the highest grade currently available (A over B over C)."""
    if not jobs:
        return jobs
    best_rank = max(_grade_rank(job.get('grade')) for job in jobs)
    filtered_jobs = [job for job in jobs if _grade_rank(job.get('grade')) == best_rank]
    return filtered_jobs or jobs

# ============================================================================
# OSCO + Gurobi Rollout
# ============================================================================

def _slim_job_for_rollout_window(job: Dict, lo: int, hi: int) -> Dict:
    """Return a shallow copy of ``job`` whose per-timestep fields are sliced to ``[lo, hi]``.

    Used by ``_osco_gurobi_select_job`` so the per-task payload shipped to OSCO
    workers contains only the slots the rollout solver can actually consult,
    cutting pickle size by orders of magnitude on full-cycle runs.

    The shallow copy is safe because the rollout solver and its helpers never
    write back into the job dict.
    """
    slim = dict(job)
    avail = job.get('available')
    if avail is not None:
        slim['available'] = [t for t in avail if lo <= t <= hi]
    thresh = job.get('pwv_thresholds')
    if thresh:
        slim['pwv_thresholds'] = {t: v for t, v in thresh.items() if lo <= t <= hi}
    fa_by_issue = job.get('forecast_available_by_issue')
    if fa_by_issue:
        slim['forecast_available_by_issue'] = {
            issue_idx: [t for t in lst if lo <= t <= hi]
            for issue_idx, lst in fa_by_issue.items()
        }
    fp_by_issue = job.get('forecast_pwv_thresholds_by_issue')
    if fp_by_issue:
        slim['forecast_pwv_thresholds_by_issue'] = {
            issue_idx: {t: v for t, v in d.items() if lo <= t <= hi}
            for issue_idx, d in fp_by_issue.items()
        }
    fa = job.get('forecast_available')
    if fa is not None:
        slim['forecast_available'] = [t for t in fa if lo <= t <= hi]
    fp = job.get('forecast_pwv_thresholds')
    if fp:
        slim['forecast_pwv_thresholds'] = {t: v for t, v in fp.items() if lo <= t <= hi}
    return slim


def _evaluate_gurobi_arm_job_batch(args_tuple):
    """Batched OSCO arm evaluator: fixes one candidate job and runs Gurobi on the
    suffix horizon under a *list* of sampled weather scenarios.

    The heavy per-task payload (`job_lookup_post`, `projects`, `unusable_indices`,
    `realized_weather_prefix`) is shipped once per fixed job rather than once per
    (job, sample) pair, which is the main reason this batched variant exists.

    Returns ``(job_id, [(payoff_0, timing_dict_0), (payoff_1, timing_dict_1), ...])``.

    Must be a top-level function so it can be pickled by ProcessPoolExecutor.
    """
    (
        job_id,
        job_length,
        sampled_paths_abs_list,
        realized_weather_prefix,
        job_lookup_post_fix,
        projects,
        time_steps,
        forecast_origin_t,
        sequence_horizon_steps,
        all_unusable_indices,
        exec_time_used_post,
        completed_exec_counts_post,
        executive_quotas_frac,
        ramped_weights,
        cumulative_observable_time,
        oracle_use_actual_job_metadata,
        use_realized_pwv_forecast,
        use_realized_rms_forecast,
        max_candidates_per_executive_by_grade,
        fill_to_total_candidates,
        use_quadratic_eb,
        inner_gurobi_time_limit_seconds,
        gurobi_log_dir,
        immediate_marginal_gain,
        log_timing,
        debug,
    ) = args_tuple

    results: List[Tuple[float, Dict[str, Any]]] = []
    for sampled_weather_abs in sampled_paths_abs_list:
        per_sample_args = (
            job_id,
            job_length,
            sampled_weather_abs,
            realized_weather_prefix,
            job_lookup_post_fix,
            projects,
            time_steps,
            forecast_origin_t,
            sequence_horizon_steps,
            all_unusable_indices,
            exec_time_used_post,
            completed_exec_counts_post,
            executive_quotas_frac,
            ramped_weights,
            cumulative_observable_time,
            oracle_use_actual_job_metadata,
            use_realized_pwv_forecast,
            use_realized_rms_forecast,
            max_candidates_per_executive_by_grade,
            fill_to_total_candidates,
            use_quadratic_eb,
            inner_gurobi_time_limit_seconds,
            gurobi_log_dir,
            immediate_marginal_gain,
            log_timing,
            debug,
        )
        _, payoff, timing_dict = _evaluate_gurobi_arm_sample(per_sample_args)
        results.append((payoff, timing_dict))

    return (job_id, results)


def _evaluate_gurobi_arm_sample(args_tuple):
    """OSCO arm evaluator: fixes a candidate job at the current decision time and
    runs Gurobi on the suffix horizon under one sampled weather scenario.

    Returns ``(job_id, total_payoff, timing_dict)`` where ``total_payoff`` is
    the immediate marginal gain of the fixed job plus the Gurobi suffix score
    for the rest of the horizon. ``timing_dict`` records wall-clock start/end
    timestamps and elapsed seconds around the Gurobi solve so the caller can
    surface per-sub-optimization timing information when requested.

    Must be a top-level function so it can be pickled by ProcessPoolExecutor.
    """
    setup_start = time.perf_counter()
    (
        job_id,
        job_length,
        sampled_weather_abs,
        realized_weather_prefix,
        job_lookup_post_fix,
        projects,
        time_steps,
        forecast_origin_t,
        sequence_horizon_steps,
        all_unusable_indices,
        exec_time_used_post,
        completed_exec_counts_post,
        executive_quotas_frac,
        ramped_weights,
        cumulative_observable_time,
        oracle_use_actual_job_metadata,
        use_realized_pwv_forecast,
        use_realized_rms_forecast,
        max_candidates_per_executive_by_grade,
        fill_to_total_candidates,
        use_quadratic_eb,
        inner_gurobi_time_limit_seconds,
        gurobi_log_dir,
        immediate_marginal_gain,
        log_timing,
        debug,
    ) = args_tuple

    combined_weather: Dict[int, Tuple[float, float]] = dict(realized_weather_prefix)
    for abs_k, pair in sampled_weather_abs.items():
        combined_weather[abs_k] = pair

    fake_pwv = np.full(time_steps, np.nan, dtype=float)
    fake_rms = np.full(time_steps, np.nan, dtype=float)
    for abs_k, (pwv_val, rms_val) in sampled_weather_abs.items():
        if 0 <= abs_k < time_steps:
            fake_pwv[abs_k] = pwv_val
            fake_rms[abs_k] = rms_val
    fake_forecast = {
        'pwv_mean': fake_pwv,
        'pwv_std': np.zeros(time_steps, dtype=float),
        'rms_mean': fake_rms,
        'rms_std': np.zeros(time_steps, dtype=float),
    }
    fake_weather_forecasts = {forecast_origin_t: fake_forecast}

    if forecast_origin_t >= time_steps:
        timing_dict = {
            "wall_start_ts": 0.0,
            "wall_end_ts": 0.0,
            "elapsed_s": 0.0,
            "forecast_origin_t": forecast_origin_t,
            "suffix_score": 0.0,
            "immediate_gain": float(immediate_marginal_gain),
            "skipped": True,
            "setup_elapsed_s": time.perf_counter() - setup_start,
            "solver_forecast_state_s": 0.0,
            "solver_candidate_enum_s": 0.0,
            "solver_candidate_prune_s": 0.0,
            "solver_model_build_s": 0.0,
            "solver_optimize_s": 0.0,
            "solver_gurobi_runtime_s": 0.0,
            "solver_postprocess_s": 0.0,
            "model_var_count": 0,
            "model_constr_count": 0,
            "candidate_jobs": 0,
            "total_feasible_starts": 0,
            "time_steps": time_steps,
        }
        if debug:
            timing_dict["debug_schedule"] = []
        return (job_id, float(immediate_marginal_gain), timing_dict)

    wall_start_ts = time.time()
    perf_start = time.perf_counter()
    setup_elapsed_s = perf_start - setup_start
    raw_schedule, best_score, solver_stats = _solve_gurobi_job_sequence_rollout(
        job_lookup=job_lookup_post_fix,
        projects=projects,
        time_steps=time_steps,
        weather_forecasts=fake_weather_forecasts,
        realized_weather=combined_weather,
        unusable_indices=all_unusable_indices,
        forecast_origin_t=forecast_origin_t,
        sequence_horizon_steps=sequence_horizon_steps,
        exec_time_used=exec_time_used_post,
        completed_exec_counts=completed_exec_counts_post,
        executive_quotas_frac=executive_quotas_frac,
        weights=ramped_weights,
        cumulative_observable_time=cumulative_observable_time,
        oracle_use_actual_job_metadata=oracle_use_actual_job_metadata,
        use_realized_pwv_forecast=use_realized_pwv_forecast,
        use_realized_rms_forecast=use_realized_rms_forecast,
        time_limit_seconds=inner_gurobi_time_limit_seconds,
        max_candidates_per_executive_by_grade=max_candidates_per_executive_by_grade,
        fill_to_total_candidates=fill_to_total_candidates,
        scarcity_weight_bonus_scale=0.0,
        use_quadratic_eb=use_quadratic_eb,
        gurobi_verbose=False,
        gurobi_log_dir=gurobi_log_dir,
    )
    elapsed_s = time.perf_counter() - perf_start
    wall_end_ts = time.time()

    if best_score is None or best_score == -float('inf') or (
            isinstance(best_score, float) and math.isnan(best_score)
    ):
        suffix_score = 0.0
    else:
        suffix_score = float(best_score)

    timing_dict = {
        "wall_start_ts": wall_start_ts,
        "wall_end_ts": wall_end_ts,
        "elapsed_s": elapsed_s,
        "forecast_origin_t": forecast_origin_t,
        "suffix_score": suffix_score,
        "immediate_gain": float(immediate_marginal_gain),
        "skipped": False,
        "setup_elapsed_s": setup_elapsed_s,
        "solver_forecast_state_s": float(solver_stats.get("forecast_state_elapsed", 0.0) or 0.0),
        "solver_candidate_enum_s": float(solver_stats.get("candidate_enumeration_elapsed", 0.0) or 0.0),
        "solver_candidate_prune_s": float(solver_stats.get("candidate_prune_elapsed", 0.0) or 0.0),
        "solver_model_build_s": float(solver_stats.get("model_build_elapsed", 0.0) or 0.0),
        "solver_optimize_s": float(solver_stats.get("optimize_elapsed", 0.0) or 0.0),
        "solver_gurobi_runtime_s": float(solver_stats.get("gurobi_runtime", 0.0) or 0.0),
        "solver_postprocess_s": float(solver_stats.get("postprocess_elapsed", 0.0) or 0.0),
        "model_var_count": int(solver_stats.get("model_var_count", 0) or 0),
        "model_constr_count": int(solver_stats.get("model_constr_count", 0) or 0),
        "candidate_jobs": int(solver_stats.get("candidate_jobs", 0) or 0),
        "total_feasible_starts": int(solver_stats.get("total_feasible_starts", 0) or 0),
        "time_steps": int(time_steps),
    }

    if debug:
        timing_dict["debug_schedule"] = list(raw_schedule) if raw_schedule else []

    # Suppress unused-variable lint when log_timing is False; the flag is consumed
    # by the caller, but we keep it in the args tuple for symmetry / future use.
    _ = log_timing

    return (job_id, float(immediate_marginal_gain) + suffix_score, timing_dict)


def _osco_gurobi_select_job(
        *,
        current_time: int,
        job_lookup: Dict[str, Dict],
        projects: List[Dict],
        time_steps: int,
        realized_weather: Dict[int, Tuple[float, float]],
        weather_forecasts: Dict[int, Dict[str, np.ndarray]],
        all_unusable_indices: set,
        exec_time_used: Dict[str, float],
        completed_exec_counts: Counter,
        executive_quotas_frac: Dict[str, Tuple[float, float]],
        weights: Dict[str, float],
        cumulative_observable_time: int,
        sequence_horizon_steps: int,
        oracle_use_actual_job_metadata: bool,
        use_realized_pwv_forecast: bool,
        use_realized_rms_forecast: bool,
        max_candidates_per_executive_by_grade: Optional[int],
        fill_to_total_candidates: Optional[int],
        use_quadratic_eb: bool,
        inner_gurobi_time_limit_seconds: float,
        gurobi_log_dir: Optional[str],
        num_samples: int,
        n_threads: int,
        rng: np.random.Generator,
        prophet_remaining_counter: Optional[Counter] = None,
        counter_bonus_a_multiplier: float = 0.0,
        counter_bonus_b_multiplier: float = 0.0,
        log_sub_timings: bool = False,
        debug: bool = False,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Select a job at current_time using OSCO with a Gurobi suffix solver per arm/sample.

    When ``log_sub_timings`` is True, prints a per-sub-solve timing table after
    all worker results have been collected so we can investigate why some
    Gurobi sub-optimizations take longer than others.

    When ``debug`` is True, prints (per decision step):
      A) candidate jobs with full availability lists and PWV/RMS thresholds
      B) the forecast (mean/std for PWV and RMS) over the lookahead horizon
      C) all sampled weather trajectories (every step of every sample)
      D) the schedule produced by each (job, sample) pair
      E) a perfect-weather lookahead schedule computed from realized_weather
    """
    selection_started = time.perf_counter()
    t0 = time.perf_counter()
    prophet_remaining_counter = prophet_remaining_counter or Counter()

    feasible_now: List[Dict] = []
    for jid, job in job_lookup.items():
        if job.get('remaining_execs', 0) <= 0:
            continue
        if _is_job_runnable_at_time(
                job=job,
                current_time=current_time,
                time_steps=time_steps,
                weather_state=realized_weather,
                unusable_indices=all_unusable_indices,
        ):
            feasible_now.append(job)

    if max_candidates_per_executive_by_grade is not None:
        ordered_ids = [j['job_id'] for j in feasible_now]
        ordered_ids = _prune_gurobi_candidates_by_exec_grade(
            ordered_candidate_ids=ordered_ids,
            job_lookup=job_lookup,
            max_candidates_per_executive_by_grade=max_candidates_per_executive_by_grade,
            fill_to_total_candidates=fill_to_total_candidates,
        )
        ordered_set = set(ordered_ids)
        feasible_now = [j for j in feasible_now if j['job_id'] in ordered_set]
    t1 = time.perf_counter()
    t_feasibility_prune_s = t1 - t0

    stats: Dict[str, Any] = {
        "candidates_count": len(feasible_now),
        "samples_count": 0,
        "tasks_count": 0,
        "selection_elapsed": 0.0,
    }

    if not feasible_now:
        stats["selection_elapsed"] = time.perf_counter() - selection_started
        return None, stats

    if debug:
        print(
            f"\n[DEBUG-OSCO @t={current_time}] Candidate jobs under consideration "
            f"({len(feasible_now)} after feasibility/prune):",
            flush=True,
        )
        for job in feasible_now:
            jid = job.get('job_id')
            jlen = job.get('length', '?')
            jgrade = job.get('grade', 'N/A')
            jexec = job.get('executive', 'N/A')
            avail_list = sorted(job.get('available', []) or [])
            pwv_thr_dict = job.get('pwv_thresholds', {}) or {}
            pwv_thr_items = sorted(pwv_thr_dict.items())
            rms_thr = job.get('rms_threshold', None)
            print(
                f"  JOB {jid} | length={jlen} | grade={jgrade} | executive={jexec} | "
                f"rms_threshold={rms_thr}",
                flush=True,
            )
            print(
                f"    available ({len(avail_list)} entries): {avail_list}",
                flush=True,
            )
            print(
                f"    pwv_thresholds ({len(pwv_thr_items)} entries): "
                f"{[(int(k), float(v)) for k, v in pwv_thr_items]}",
                flush=True,
            )

    horizon_end = min(time_steps, current_time + max(1, sequence_horizon_steps))
    horizon_steps = horizon_end - current_time
    current_forecast = weather_forecasts.get(current_time) if weather_forecasts else None

    if horizon_steps <= 0:
        forecast_for_future = None
    else:
        pwv_mean_arr = (
            np.asarray(current_forecast.get('pwv_mean', []), dtype=float)
            if current_forecast is not None else np.array([], dtype=float)
        )
        pwv_std_arr = (
            np.asarray(current_forecast.get('pwv_std', []), dtype=float)
            if current_forecast is not None else np.array([], dtype=float)
        )
        if pwv_mean_arr.size:
            pwv_mean_slice = pwv_mean_arr[current_time:current_time + horizon_steps]
        else:
            pwv_mean_slice = np.full(horizon_steps, np.nan)
        if pwv_std_arr.size:
            pwv_std_slice = pwv_std_arr[current_time:current_time + horizon_steps]
        else:
            pwv_std_slice = np.full(horizon_steps, np.nan)

        if current_forecast is not None:
            rms_mean_slice, rms_std_slice = get_rms_forecast_slice(
                current_forecast,
                issuance_idx=current_time,
                start_idx=current_time,
                end_idx=current_time + horizon_steps,
            )
        else:
            rms_mean_slice = np.full(horizon_steps, np.nan)
            rms_std_slice = np.full(horizon_steps, np.nan)

        if use_realized_pwv_forecast:
            pwv_mean_slice = np.asarray([
                realized_weather.get(abs_t, (np.nan, np.nan))[0]
                for abs_t in range(current_time, current_time + horizon_steps)
            ], dtype=float)
            pwv_std_slice = np.zeros(horizon_steps, dtype=float)
        if use_realized_rms_forecast:
            rms_mean_slice = np.asarray([
                realized_weather.get(abs_t, (np.nan, np.nan))[1]
                for abs_t in range(current_time, current_time + horizon_steps)
            ], dtype=float)
            rms_std_slice = np.zeros(horizon_steps, dtype=float)

        forecast_for_future = {
            'pwv_mean': np.asarray(pwv_mean_slice, dtype=float),
            'pwv_std': np.asarray(pwv_std_slice, dtype=float),
            'rms_mean': np.asarray(rms_mean_slice, dtype=float),
            'rms_std': np.asarray(rms_std_slice, dtype=float),
        }

    if forecast_for_future is None:
        stats["selection_elapsed"] = time.perf_counter() - selection_started
        return None, stats

    if (
            (not use_realized_pwv_forecast and current_forecast is None)
            or (not use_realized_rms_forecast and current_forecast is None)
    ):
        stats["selection_elapsed"] = time.perf_counter() - selection_started
        return None, stats

    def _debug_true_pair(abs_t: int) -> Tuple[float, float]:
        pair = realized_weather.get(abs_t, (np.nan, np.nan))
        pw_raw, rm_raw = pair
        pw_v = float(pw_raw) if pw_raw is not None else float('nan')
        rm_v = float(rm_raw) if rm_raw is not None else float('nan')
        return pw_v, rm_v

    if debug:
        pwv_mean_arr_dbg = np.asarray(forecast_for_future['pwv_mean'], dtype=float)
        pwv_std_arr_dbg = np.asarray(forecast_for_future['pwv_std'], dtype=float)
        rms_mean_arr_dbg = np.asarray(forecast_for_future['rms_mean'], dtype=float)
        rms_std_arr_dbg = np.asarray(forecast_for_future['rms_std'], dtype=float)
        h_dbg = len(pwv_mean_arr_dbg)
        print(
            f"\n[DEBUG-OSCO @t={current_time}] Forecast for lookahead horizon "
            f"({h_dbg} steps):",
            flush=True,
        )
        print(
            f"  {'step':>4} | {'abs_t':>6} | {'pwv_mean':>10} | {'pwv_std':>10} | "
            f"{'true_pwv':>10} | {'rms_mean':>10} | {'rms_std':>10} | {'true_rms':>10}",
            flush=True,
        )
        for k_dbg in range(h_dbg):
            true_pwv_dbg, true_rms_dbg = _debug_true_pair(current_time + k_dbg)
            print(
                f"  {k_dbg:>4d} | {current_time + k_dbg:>6d} | "
                f"{float(pwv_mean_arr_dbg[k_dbg]):>10.4f} | "
                f"{float(pwv_std_arr_dbg[k_dbg]):>10.4f} | "
                f"{true_pwv_dbg:>10.4f} | "
                f"{float(rms_mean_arr_dbg[k_dbg]):>10.4f} | "
                f"{float(rms_std_arr_dbg[k_dbg]):>10.4f} | "
                f"{true_rms_dbg:>10.4f}",
                flush=True,
            )

        def _safe_stats(arr: np.ndarray) -> Tuple[float, float]:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return float('nan'), float('nan')
            return float(np.mean(finite)), float(np.std(finite))

        pwv_m_mean, pwv_m_std = _safe_stats(pwv_mean_arr_dbg)
        pwv_s_mean, pwv_s_std = _safe_stats(pwv_std_arr_dbg)
        rms_m_mean, rms_m_std = _safe_stats(rms_mean_arr_dbg)
        rms_s_mean, rms_s_std = _safe_stats(rms_std_arr_dbg)
        print(
            f"  Forecast summary (mean +/- std across the horizon, ignoring NaN):",
            flush=True,
        )
        print(
            f"    pwv_mean: {pwv_m_mean:.4f} +/- {pwv_m_std:.4f} | "
            f"pwv_std:  {pwv_s_mean:.4f} +/- {pwv_s_std:.4f}",
            flush=True,
        )
        print(
            f"    rms_mean: {rms_m_mean:.4f} +/- {rms_m_std:.4f} | "
            f"rms_std:  {rms_s_mean:.4f} +/- {rms_s_std:.4f}",
            flush=True,
        )

    t_weather_sample_start = time.perf_counter()
    sampled_paths_abs: List[Dict[int, Tuple[float, float]]] = []
    for _ in range(max(1, num_samples)):
        sample_seed = int(rng.integers(0, 2 ** 32 - 1))
        child_rng = np.random.default_rng(sample_seed)
        sampled_path = sample_weather_path_from_forecast(forecast_for_future, child_rng)
        sampled_abs: Dict[int, Tuple[float, float]] = {}
        for k_rel, pair in sampled_path.items():
            sampled_abs[k_rel + current_time] = pair
        sampled_paths_abs.append(sampled_abs)
    t_weather_sample_s = time.perf_counter() - t_weather_sample_start

    if debug:
        n_samples_dbg = len(sampled_paths_abs)
        if n_samples_dbg > 0:
            n_steps_dbg = len(sampled_paths_abs[0])
        else:
            n_steps_dbg = 0
        print(
            f"\n[DEBUG-OSCO @t={current_time}] Sampled weather trajectories "
            f"({n_samples_dbg} samples x {n_steps_dbg} steps):",
            flush=True,
        )
        per_step_pwv: Dict[int, List[float]] = {}
        per_step_rms: Dict[int, List[float]] = {}
        for s_idx, sampled_abs in enumerate(sampled_paths_abs):
            print(f"  --- Sample {s_idx} ---", flush=True)
            print(
                f"    {'abs_t':>6} | {'pwv':>10} | {'rms':>10} | "
                f"{'true_pwv':>10} | {'true_rms':>10}",
                flush=True,
            )
            for abs_t in sorted(sampled_abs.keys()):
                pwv_v, rms_v = sampled_abs[abs_t]
                true_pwv_dbg, true_rms_dbg = _debug_true_pair(int(abs_t))
                print(
                    f"    {int(abs_t):>6d} | {float(pwv_v):>10.4f} | "
                    f"{float(rms_v):>10.4f} | "
                    f"{true_pwv_dbg:>10.4f} | {true_rms_dbg:>10.4f}",
                    flush=True,
                )
                per_step_pwv.setdefault(abs_t, []).append(float(pwv_v))
                per_step_rms.setdefault(abs_t, []).append(float(rms_v))
        if per_step_pwv:
            print(
                f"\n  [DEBUG-OSCO @t={current_time}] Per-step PWV/RMS sample "
                f"mean and std (across {n_samples_dbg} samples):",
                flush=True,
            )
            print(
                f"    {'abs_t':>6} | {'pwv_mean':>10} | {'pwv_std':>10} | "
                f"{'rms_mean':>10} | {'rms_std':>10} | "
                f"{'true_pwv':>10} | {'true_rms':>10}",
                flush=True,
            )
            for abs_t in sorted(per_step_pwv.keys()):
                pwv_arr = np.asarray(per_step_pwv[abs_t], dtype=float)
                rms_arr = np.asarray(per_step_rms[abs_t], dtype=float)
                pwv_finite = pwv_arr[np.isfinite(pwv_arr)]
                rms_finite = rms_arr[np.isfinite(rms_arr)]
                pwv_m = float(np.mean(pwv_finite)) if pwv_finite.size else float('nan')
                pwv_s = (
                    float(np.std(pwv_finite)) if pwv_finite.size > 1 else 0.0
                ) if pwv_finite.size else float('nan')
                rms_m = float(np.mean(rms_finite)) if rms_finite.size else float('nan')
                rms_s = (
                    float(np.std(rms_finite)) if rms_finite.size > 1 else 0.0
                ) if rms_finite.size else float('nan')
                true_pwv_dbg, true_rms_dbg = _debug_true_pair(int(abs_t))
                print(
                    f"    {int(abs_t):>6d} | {pwv_m:>10.4f} | {pwv_s:>10.4f} | "
                    f"{rms_m:>10.4f} | {rms_s:>10.4f} | "
                    f"{true_pwv_dbg:>10.4f} | {true_rms_dbg:>10.4f}",
                    flush=True,
                )

    t_marginal_gain_start = time.perf_counter()
    remaining_jobs_for_marginal = [
        j for j in job_lookup.values() if j.get('remaining_execs', 0) > 0
    ]
    n_prime = cumulative_observable_time if cumulative_observable_time > 0 else 1
    eta1_counter_bonus = _compute_eta1_topk(
        remaining_jobs_for_marginal,
        n_prime,
        execs_key='total_execs',
    )
    marginal_gain_map: Dict[str, float] = {}
    for job in feasible_now:
        marginal_gain_map[job['job_id']] = _compute_single_job_marginal_gain(
            job=job,
            remaining_jobs=remaining_jobs_for_marginal,
            projects=projects,
            cumulative_exec_time=exec_time_used,
            completed_exec_counts=completed_exec_counts,
            executive_quotas_frac=executive_quotas_frac,
            weights=weights,
            cumulative_observable_time=cumulative_observable_time,
            job_info_map=job_lookup,
        )
        marginal_gain_map[job['job_id']] += _compute_counter_alignment_bonus(
            job=job,
            prophet_remaining_counter=prophet_remaining_counter,
            weights=weights,
            counter_bonus_a_multiplier=counter_bonus_a_multiplier,
            counter_bonus_b_multiplier=counter_bonus_b_multiplier,
            normalization_factor=eta1_counter_bonus,
        )
    t_marginal_gain_s = time.perf_counter() - t_marginal_gain_start

    t_args_build_start = time.perf_counter()
    max_job_length_in_pool = max(
        (int(j.get('length', 1)) for j in job_lookup.values()),
        default=1,
    )
    window_lo = current_time
    window_hi = current_time + sequence_horizon_steps + 2 * max_job_length_in_pool

    realized_prefix: Dict[int, Tuple[float, float]] = {}

    trimmed_unusable_indices = {
        idx for idx in all_unusable_indices
        if window_lo <= idx <= window_hi
    }

    slim_job_lookup_base = {
        jid: _slim_job_for_rollout_window(j, window_lo, window_hi)
        for jid, j in job_lookup.items()
    }

    args_list: List[tuple] = []
    for job in feasible_now:
        jid = job['job_id']
        job_length = int(job['length'])
        forecast_origin_t = current_time + job_length
        if forecast_origin_t > time_steps:
            continue

        job_lookup_post = dict(slim_job_lookup_base)
        job_lookup_post[jid] = dict(slim_job_lookup_base[jid])
        job_lookup_post[jid]['remaining_execs'] = max(
            0, job_lookup_post[jid].get('remaining_execs', 0) - 1
        )
        completed_post = Counter(completed_exec_counts)
        completed_post[jid] += 1
        exec_time_used_post = dict(exec_time_used)
        if isinstance(job.get('executive'), str):
            exec_time_used_post[job['executive']] = (
                exec_time_used_post.get(job['executive'], 0.0) + job_length
            )
        elif isinstance(job.get('executive'), dict):
            for en, frac in job['executive'].items():
                exec_time_used_post[en] = exec_time_used_post.get(en, 0.0) + job_length * frac

        immediate_gain = float(marginal_gain_map.get(jid, 0.0))
        args_list.append((
            jid,
            job_length,
            sampled_paths_abs,
            realized_prefix,
            job_lookup_post,
            projects,
            time_steps,
            forecast_origin_t,
            sequence_horizon_steps,
            trimmed_unusable_indices,
            exec_time_used_post,
            completed_post,
            executive_quotas_frac,
            weights,
            cumulative_observable_time,
            oracle_use_actual_job_metadata,
            use_realized_pwv_forecast,
            use_realized_rms_forecast,
            max_candidates_per_executive_by_grade,
            fill_to_total_candidates,
            use_quadratic_eb,
            inner_gurobi_time_limit_seconds,
            gurobi_log_dir,
            immediate_gain,
            log_sub_timings,
            debug,
        ))

    t_args_build_s = time.perf_counter() - t_args_build_start

    n_samples_planned = max(1, num_samples)
    n_total_evals = len(args_list) * n_samples_planned

    stats["samples_count"] = n_samples_planned
    stats["tasks_count"] = n_total_evals

    if not args_list:
        stats["selection_elapsed"] = time.perf_counter() - selection_started
        print(
            f"  [OSCO-GUROBI @t={current_time}] Setup phases: "
            f"feasibility+prune={t_feasibility_prune_s:.3f}s | "
            f"weather_sample={t_weather_sample_s:.3f}s | "
            f"marginal_gain={t_marginal_gain_s:.3f}s | "
            f"args_build(slim)={t_args_build_s:.3f}s | "
            f"dispatch+collect=0.000s (no tasks)",
            flush=True,
        )
        print(
            f"  [OSCO-GUROBI @t={current_time}] Problem size: "
            f"time_steps={time_steps}, candidates={len(feasible_now)}, tasks=0, workers=0",
            flush=True,
        )
        return None, stats

    if n_threads == -1:
        max_workers = int(os.getenv('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    else:
        max_workers = max(1, n_threads)

    print(
        f"  [OSCO-GUROBI @t={current_time}] Evaluating {n_total_evals} (job, sample) pairs "
        f"({len(args_list)} jobs x {n_samples_planned} samples, batched per-job) "
        f"across {max_workers} workers...",
        flush=True,
    )

    t_dispatch_collect_start = time.perf_counter()
    payoffs_by_job: Dict[str, List[float]] = {j['job_id']: [] for j in feasible_now}
    timing_records: List[Tuple[str, Dict[str, Any]]] = []
    if max_workers > 1 and len(args_list) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for jid, batch_results in executor.map(_evaluate_gurobi_arm_job_batch, args_list):
                for payoff, timing_dict in batch_results:
                    payoffs_by_job[jid].append(payoff)
                    timing_records.append((jid, timing_dict))
    else:
        for args in args_list:
            jid, batch_results = _evaluate_gurobi_arm_job_batch(args)
            for payoff, timing_dict in batch_results:
                payoffs_by_job[jid].append(payoff)
                timing_records.append((jid, timing_dict))
    t_dispatch_collect_s = time.perf_counter() - t_dispatch_collect_start

    print(
        f"  [OSCO-GUROBI @t={current_time}] Setup phases: "
        f"feasibility+prune={t_feasibility_prune_s:.3f}s | "
        f"weather_sample={t_weather_sample_s:.3f}s | "
        f"marginal_gain={t_marginal_gain_s:.3f}s | "
        f"args_build(slim)={t_args_build_s:.3f}s | "
        f"dispatch+collect={t_dispatch_collect_s:.3f}s",
        flush=True,
    )
    print(
        f"  [OSCO-GUROBI @t={current_time}] Problem size: "
        f"time_steps={time_steps}, candidates={len(feasible_now)}, "
        f"jobs_dispatched={len(args_list)}, samples_per_job={n_samples_planned}, "
        f"total_evals={n_total_evals}, workers={max_workers}",
        flush=True,
    )

    if debug and timing_records:
        print(
            f"\n[DEBUG-OSCO @t={current_time}] Per-(candidate, sample) Gurobi "
            f"suffix schedules (selected job at t={current_time} is "
            f"prepended implicitly):",
            flush=True,
        )
        per_job_sample_idx_dbg: Dict[str, int] = {}
        schedules_by_job: Dict[str, List[Tuple[int, List[Tuple[str, int]]]]] = {}
        for jid, td in timing_records:
            sample_idx = per_job_sample_idx_dbg.get(jid, 0)
            per_job_sample_idx_dbg[jid] = sample_idx + 1
            sched = td.get("debug_schedule", []) or []
            schedules_by_job.setdefault(jid, []).append((sample_idx, sched))
        for jid in sorted(schedules_by_job.keys()):
            j_obj = next((j for j in feasible_now if j.get('job_id') == jid), None)
            j_len = j_obj.get('length', '?') if j_obj else '?'
            print(
                f"  CANDIDATE {jid} (length={j_len}): "
                f"prepend ({jid}@{current_time}) then suffix below",
                flush=True,
            )
            for sample_idx, sched in sorted(schedules_by_job[jid], key=lambda x: x[0]):
                if not sched:
                    print(
                        f"    sample {sample_idx:>3}: <empty suffix>",
                        flush=True,
                    )
                else:
                    sched_sorted = sorted(sched, key=lambda it: (int(it[1]), str(it[0])))
                    sched_str = ", ".join(
                        f"{sjid}@{int(stt)}" for sjid, stt in sched_sorted
                    )
                    print(
                        f"    sample {sample_idx:>3} ({len(sched_sorted)} entries): "
                        f"{sched_str}",
                        flush=True,
                    )

    if debug:
        perfect_pwv = np.full(time_steps, np.nan, dtype=float)
        perfect_rms = np.full(time_steps, np.nan, dtype=float)
        for abs_k in range(time_steps):
            pair = realized_weather.get(abs_k)
            if pair is None:
                continue
            pw, rm = pair
            if not pd.isna(pw):
                perfect_pwv[abs_k] = float(pw)
            if not pd.isna(rm):
                perfect_rms[abs_k] = float(rm)
        perfect_fake_forecast = {
            'pwv_mean': perfect_pwv,
            'pwv_std': np.zeros(time_steps, dtype=float),
            'rms_mean': perfect_rms,
            'rms_std': np.zeros(time_steps, dtype=float),
        }
        perfect_weather_forecasts = {current_time: perfect_fake_forecast}

        perfect_horizon = max(1, sequence_horizon_steps)
        try:
            # Force oracle_use_actual_job_metadata=True so the lookahead uses the
            # job's real `available` / `pwv_thresholds` / `rms_threshold` instead
            # of the forecast-issued metadata. Without this, the rollout solver
            # builds its job copies via _resolve_forecast_metadata_for_time(job,
            # current_time), whose forecast `available` list typically excludes
            # `current_time` itself, so the perfect-weather schedule would
            # effectively start one timestep ahead. The oracle interpretation
            # is "if I knew the weather perfectly", so the job-feasibility
            # constraints should also be the actual ones.
            perfect_schedule, perfect_score, _perfect_stats = _solve_gurobi_job_sequence_rollout(
                job_lookup=slim_job_lookup_base,
                projects=projects,
                time_steps=time_steps,
                weather_forecasts=perfect_weather_forecasts,
                realized_weather=realized_weather,
                unusable_indices=all_unusable_indices,
                forecast_origin_t=current_time,
                sequence_horizon_steps=perfect_horizon,
                exec_time_used=exec_time_used,
                completed_exec_counts=completed_exec_counts,
                executive_quotas_frac=executive_quotas_frac,
                weights=weights,
                cumulative_observable_time=cumulative_observable_time,
                oracle_use_actual_job_metadata=True,
                use_realized_pwv_forecast=True,
                use_realized_rms_forecast=True,
                time_limit_seconds=inner_gurobi_time_limit_seconds,
                max_candidates_per_executive_by_grade=max_candidates_per_executive_by_grade,
                fill_to_total_candidates=fill_to_total_candidates,
                scarcity_weight_bonus_scale=0.0,
                use_quadratic_eb=use_quadratic_eb,
                gurobi_verbose=False,
                gurobi_log_dir=gurobi_log_dir,
            )
        except Exception as exc:  # noqa: BLE001
            perfect_schedule = None
            perfect_score = None
            print(
                f"\n[DEBUG-OSCO @t={current_time}] Perfect-weather lookahead "
                f"raised an exception: {exc!r}",
                flush=True,
            )

        if perfect_schedule is not None:
            score_str = (
                f"{float(perfect_score):.6f}"
                if perfect_score is not None and perfect_score != -float('inf')
                else "n/a"
            )
            print(
                f"\n[DEBUG-OSCO @t={current_time}] Perfect-weather lookahead "
                f"schedule (horizon={perfect_horizon} steps, score={score_str}):",
                flush=True,
            )
            if not perfect_schedule:
                print("  <empty schedule>", flush=True)
            else:
                sched_sorted = sorted(
                    perfect_schedule, key=lambda it: (int(it[1]), str(it[0]))
                )
                for sjid, stt in sched_sorted:
                    print(f"  {sjid}@{int(stt)}", flush=True)

    mean_payoffs: Dict[str, float] = {
        jid: float(np.mean(p_list)) for jid, p_list in payoffs_by_job.items() if p_list
    }
    if not mean_payoffs:
        stats["selection_elapsed"] = time.perf_counter() - selection_started
        return None, stats

    best_mean = max(mean_payoffs.values())
    tol = 1e-9
    top_tier = [jid for jid, mean_val in mean_payoffs.items() if mean_val >= best_mean - tol]
    if len(top_tier) == 1:
        best_id = top_tier[0]
    else:
        best_id = max(top_tier, key=lambda jid: marginal_gain_map.get(jid, -float('inf')))

    if log_sub_timings and timing_records:
        per_job_sample_idx: Dict[str, int] = {}
        ordered_records: List[Tuple[str, int, Dict[str, Any]]] = []
        for jid, td in timing_records:
            sample_idx = per_job_sample_idx.get(jid, 0)
            per_job_sample_idx[jid] = sample_idx + 1
            ordered_records.append((jid, sample_idx, td))

        ordered_records.sort(key=lambda rec: rec[2].get("wall_start_ts", 0.0))

        n_jobs = len(payoffs_by_job)
        n_samples = max(1, num_samples)
        n_tasks = len(timing_records)

        def _summarize(field_name: str) -> Tuple[float, float, float, float, float]:
            vals = [
                float(rec[2].get(field_name, 0.0) or 0.0)
                for rec in ordered_records
                if not rec[2].get("skipped", False)
            ]
            if not vals:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            return (
                float(sum(vals)),
                float(np.mean(vals)),
                float(np.median(vals)),
                float(min(vals)),
                float(max(vals)),
            )

        elapsed_total, elapsed_mean, elapsed_median, elapsed_min, elapsed_max = _summarize("elapsed_s")
        fcast_total, fcast_mean, fcast_median, fcast_min, fcast_max = _summarize("solver_forecast_state_s")
        model_total, model_mean, model_median, model_min, model_max = _summarize("solver_model_build_s")
        opt_total, opt_mean, opt_median, opt_min, opt_max = _summarize("solver_optimize_s")

        print(
            f"\n  [OSCO-GUROBI @t={current_time}] Sub-solve timing "
            f"({n_jobs} jobs x {n_samples} samples = {n_tasks} tasks):",
            flush=True,
        )
        print(
            f"    elapsed_s     : total={elapsed_total:.2f}s | mean={elapsed_mean:.2f}s | "
            f"median={elapsed_median:.2f}s | min={elapsed_min:.2f}s | max={elapsed_max:.2f}s",
            flush=True,
        )
        print(
            f"    fcast_state_s : total={fcast_total:.2f}s | mean={fcast_mean:.3f}s | "
            f"median={fcast_median:.3f}s | min={fcast_min:.3f}s | max={fcast_max:.3f}s",
            flush=True,
        )
        print(
            f"    model_build_s : total={model_total:.2f}s | mean={model_mean:.3f}s | "
            f"median={model_median:.3f}s | min={model_min:.3f}s | max={model_max:.3f}s",
            flush=True,
        )
        print(
            f"    optimize_s    : total={opt_total:.2f}s | mean={opt_mean:.3f}s | "
            f"median={opt_median:.3f}s | min={opt_min:.3f}s | max={opt_max:.3f}s",
            flush=True,
        )
        header = (
            f"    {'job_id':<22} | {'samp':>4} | {'wall_start':<19} | "
            f"{'elapsed':>7} | {'setup':>6} | {'fcast':>6} | {'enum':>6} | "
            f"{'model':>6} | {'opt':>6} | {'grb_rt':>6} | {'vars':>6} | "
            f"{'constrs':>7} | {'cands':>5}"
        )
        print(header, flush=True)
        print("    " + "-" * (len(header) - 4), flush=True)
        for jid, sample_idx, td in ordered_records:
            if td.get("skipped", False):
                start_str = "-- skipped --"
            else:
                ws = td.get("wall_start_ts", 0.0)
                start_str = datetime.fromtimestamp(ws).strftime("%Y-%m-%d %H:%M:%S")
            elapsed = float(td.get("elapsed_s", 0.0) or 0.0)
            setup_s = float(td.get("setup_elapsed_s", 0.0) or 0.0)
            fcast_s = float(td.get("solver_forecast_state_s", 0.0) or 0.0)
            enum_s = float(td.get("solver_candidate_enum_s", 0.0) or 0.0)
            model_s = float(td.get("solver_model_build_s", 0.0) or 0.0)
            opt_s = float(td.get("solver_optimize_s", 0.0) or 0.0)
            grb_rt = float(td.get("solver_gurobi_runtime_s", 0.0) or 0.0)
            n_vars = int(td.get("model_var_count", 0) or 0)
            n_constrs = int(td.get("model_constr_count", 0) or 0)
            n_cands = int(td.get("candidate_jobs", 0) or 0)
            jid_short = jid if len(jid) <= 22 else jid[:21] + "~"
            print(
                f"    {jid_short:<22} | {sample_idx:>4} | {start_str:<19} | "
                f"{elapsed:>7.2f} | {setup_s:>6.3f} | {fcast_s:>6.3f} | {enum_s:>6.3f} | "
                f"{model_s:>6.3f} | {opt_s:>6.3f} | {grb_rt:>6.3f} | {n_vars:>6d} | "
                f"{n_constrs:>7d} | {n_cands:>5d}",
                flush=True,
            )
        print("", flush=True)

        payoff_stats: List[Tuple[str, float, float, float, float, float, float, float, int]] = []
        for jid, p_list in payoffs_by_job.items():
            if not p_list:
                continue
            arr = np.asarray(p_list, dtype=float)
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr)) if arr.size > 1 else 0.0
            min_v = float(np.min(arr))
            max_v = float(np.max(arr))
            if arr.size >= 2:
                p25_v, med_v, p75_v = (float(v) for v in np.percentile(arr, [25, 50, 75]))
            else:
                p25_v = med_v = p75_v = mean_v
            payoff_stats.append((jid, mean_v, std_v, min_v, p25_v, med_v, p75_v, max_v, int(arr.size)))

        if payoff_stats:
            payoff_stats.sort(key=lambda r: -r[1])
            n_per_job = max(p[8] for p in payoff_stats)
            print(
                f"  [OSCO-GUROBI @t={current_time}] Payoff distribution "
                f"(N={n_per_job} samples/job):",
                flush=True,
            )
            payoff_header = (
                f"    {'job_id':<22} | {'mean':>11} | {'std':>11} | "
                f"{'min':>11} | {'p25':>11} | {'median':>11} | {'p75':>11} | {'max':>11} | sel"
            )
            print(payoff_header, flush=True)
            print("    " + "-" * (len(payoff_header) - 4), flush=True)
            for (jid, mean_v, std_v, min_v, p25_v, med_v, p75_v, max_v, _n) in payoff_stats:
                marker = ">>" if jid == best_id else "  "
                jid_short = jid if len(jid) <= 22 else jid[:21] + "~"
                print(
                    f"    {jid_short:<22} | {mean_v:>11.3e} | {std_v:>11.3e} | "
                    f"{min_v:>11.3e} | {p25_v:>11.3e} | {med_v:>11.3e} | "
                    f"{p75_v:>11.3e} | {max_v:>11.3e} | {marker}",
                    flush=True,
                )
            print("", flush=True)

    sorted_decisions = sorted(mean_payoffs.items(), key=lambda x: -x[1])
    candidate_map = {j['job_id']: j for j in feasible_now}
    print(f"\n--- OSCO-GUROBI Evaluation Results (t={current_time}) ---")
    for jid, mean_val in sorted_decisions[:15]:
        job_details = candidate_map.get(jid)
        if not job_details:
            continue
        grade = job_details.get('grade', 'N/A')
        length = job_details.get('length', '?')
        exec_info = job_details.get('executive', 'Unknown')
        if isinstance(exec_info, dict):
            exec_str = ", ".join([f"{k}:{v:.1%}" for k, v in exec_info.items()])
        else:
            exec_str = str(exec_info)
        gain = float(marginal_gain_map.get(jid, 0.0))
        payoff_list = payoffs_by_job.get(jid, [])
        n_pulls = len(payoff_list)
        std_val = float(np.std(payoff_list)) if n_pulls > 1 else 0.0
        min_val = float(np.min(payoff_list)) if payoff_list else 0.0
        max_val = float(np.max(payoff_list)) if payoff_list else 0.0
        choice = ">>" if jid == best_id else "  "
        print(
            f"{choice} Job: {jid:<25} | Mean: {mean_val:>12.3e} | Std: {std_val:>12.3e} | "
            f"Min: {min_val:>12.3e} | Max: {max_val:>12.3e} | "
            f"N: {n_pulls:>3} | Gain: {gain:>12.3e} | Grade: {grade} | Len: {length:<3} | Exec: {exec_str}"
        )
    print(f"  Final Choice: {best_id}\n", flush=True)

    stats["selection_elapsed"] = time.perf_counter() - selection_started
    return best_id, stats


