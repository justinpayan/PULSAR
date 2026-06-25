"""
Preprocess realized weather from phase-tracker data, applying downtime rules
and linear interpolation, then save to a pickle for consumption by full_year.py.

Pipeline (in order):
  1. Load phase-tracker readings and snap to 30-min timeline slots.
  2. Overwrite Weather downtime slots with (inf, -inf).
  3. Overwrite Technical downtime slots with (inf, -inf).
  4. Context-aware gap fill for remaining NaN slots not in any downtime/engineering
     index: linear interpolation if both boundaries are valid, copy from the single
     valid boundary, or fill as weather downtime if both boundaries are bad.
  5. Overwrite Engineering/EOC slots with (NaN, NaN) — last, so interpolation
     does not bridge through them.

Output pickle keys:
  realized_weather   : dict {slot_idx: (pwv, rms)}
  idx_to_timestamp   : dict {slot_idx: pd.Timestamp}
  start_date         : pd.Timestamp
  end_date           : pd.Timestamp

Usage:
    python preprocess_weather.py --data_dir /path/to/data \
        --start_date 2023-10-01 --end_date 2024-10-01 \
        --output preprocessed_weather.pkl
"""

import argparse
import os
import pickle
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

TIME_INTERVAL_MINUTES = 30

# Hard-coded February/March array shutdowns by calendar year.
# Each entry is (start_month, start_day, end_month, end_day) where end is INCLUSIVE.
# These shutdowns are decided exogenously and are not always reflected in the
# downtimes_dimensions or shifts_dimensions CSVs, so they must be defined here.
HARDCODED_SHUTDOWNS: dict = {
    2018: (2, 1,  3, 5),
    2019: (1, 29, 2, 28),
    2022: (1, 31, 2, 28),
    2023: (2, 1,  2, 28),
    2024: (1, 30, 2, 29),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess realized weather from phase-tracker data with downtime rules."
    )
    p.add_argument("--data_dir", required=True,
                   help="Directory with phase_tracker_data_distinct_prepared.csv, "
                        "downtimes_dimensions.csv, shifts_dimensions.csv")
    p.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    p.add_argument("--output", required=True, help="Path to save the output pickle file")
    return p.parse_args()


def load_downtimes_and_shifts(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    downtimes_path = os.path.join(data_dir, "downtimes_dimensions.csv")
    shifts_path = os.path.join(data_dir, "shifts_dimensions.csv")
    if not os.path.isfile(downtimes_path):
        raise FileNotFoundError(f"Missing {downtimes_path}")
    if not os.path.isfile(shifts_path):
        raise FileNotFoundError(f"Missing {shifts_path}")
    downtimes_df = pd.read_csv(downtimes_path)
    shifts_df = pd.read_csv(shifts_path)
    downtimes_df["START_TIME"] = pd.to_datetime(downtimes_df["START_TIME"], utc=True)
    downtimes_df["END_TIME"] = pd.to_datetime(downtimes_df["END_TIME"], utc=True)
    shifts_df["START_TIME"] = pd.to_datetime(shifts_df["START_TIME"], utc=True)
    shifts_df["END_TIME"] = pd.to_datetime(shifts_df["END_TIME"], utc=True)
    return downtimes_df, shifts_df


def build_timeline(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Tuple[pd.DatetimeIndex, Dict[pd.Timestamp, int], Dict[int, pd.Timestamp]]:
    all_slots = pd.date_range(
        start=start_date, end=end_date,
        freq=f"{TIME_INTERVAL_MINUTES}min",
    )
    timestamp_to_idx = {ts: i for i, ts in enumerate(all_slots)}
    idx_to_timestamp = {i: ts for ts, i in timestamp_to_idx.items()}
    return all_slots, timestamp_to_idx, idx_to_timestamp


def get_indices_in_intervals(
    intervals_df: pd.DataFrame,
    all_time_slots: pd.DatetimeIndex,
) -> Set[int]:
    indices: Set[int] = set()
    for _, row in intervals_df.iterrows():
        mask = (all_time_slots >= row["START_TIME"]) & (all_time_slots < row["END_TIME"])
        indices.update(np.where(mask)[0].tolist())
    return indices


def load_phase_tracker(
    data_dir: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    all_slots: pd.DatetimeIndex,
    time_steps: int,
) -> Tuple[Dict[int, Tuple[float, float]], int]:
    """Load phase-tracker CSV and snap to 30-min slots.

    Returns the raw realized_weather dict and count of slots with data.
    """
    path = os.path.join(data_dir, "phase_tracker_data_distinct_prepared.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")

    pt_df = pd.read_csv(path)
    pt_df["timestamp"] = pd.to_datetime(pt_df["timestamp_utc"], utc=True)
    pt_df = pt_df[
        (pt_df["timestamp"] >= start_date - pd.Timedelta(hours=6))
        & (pt_df["timestamp"] <= end_date + pd.Timedelta(hours=6))
    ].drop_duplicates(subset="timestamp").copy()

    allowed_phase_err_lower = 45.83662
    rms_microns = pt_df["array_characteristic_phase_rms_microns"].replace(0, np.nan)
    pt_df["freqrms"] = (3.0e8 / (rms_microns * 1.0e-6 * 360.0 / allowed_phase_err_lower)) / 1e9

    pwv_series = pt_df.set_index("timestamp")["averaged_pwv"]
    rms_series = pt_df.set_index("timestamp")["freqrms"]
    tolerance = pd.Timedelta(minutes=30)
    pwv_reindexed = pwv_series.reindex(all_slots, method="nearest", tolerance=tolerance)
    rms_reindexed = rms_series.reindex(all_slots, method="nearest", tolerance=tolerance)

    realized_weather: Dict[int, Tuple[float, float]] = {}
    n_with = 0
    for i in range(time_steps):
        pwv = pwv_reindexed.iloc[i] if i < len(pwv_reindexed) else np.nan
        rms = rms_reindexed.iloc[i] if i < len(rms_reindexed) else np.nan
        realized_weather[i] = (float(pwv), float(rms))
        if not (pd.isna(pwv) or pd.isna(rms)):
            n_with += 1
    return realized_weather, n_with


def print_sample(realized_weather: Dict[int, Tuple[float, float]],
                 idx_to_timestamp: Dict[int, pd.Timestamp],
                 time_steps: int, n: int = 10):
    """Print a few sample slots for spot-checking."""
    step = max(1, time_steps // n)
    print("  Sample slots:")
    for i in range(0, time_steps, step):
        pwv, rms = realized_weather[i]
        ts = idx_to_timestamp[i]
        if pd.isna(pwv) or pd.isna(rms):
            print(f"    slot {i:5d}  {ts}  PWV=NaN  RMS=NaN")
        elif np.isinf(pwv) or np.isinf(rms):
            print(f"    slot {i:5d}  {ts}  PWV={pwv}  RMS={rms}")
        else:
            print(f"    slot {i:5d}  {ts}  PWV={pwv:.4f}  RMS={rms:.4f}")


def count_nan(realized_weather: Dict[int, Tuple[float, float]]) -> int:
    return sum(1 for pwv, rms in realized_weather.values()
               if pd.isna(pwv) or pd.isna(rms))


def main():
    args = parse_args()
    data_dir = args.data_dir

    start_date = pd.Timestamp(args.start_date, tz="UTC")
    end_date = pd.Timestamp(args.end_date, tz="UTC")

    print("=" * 70)
    print("WEATHER PREPROCESSING")
    print("=" * 70)
    print(f"Date range : {start_date.date()} to {end_date.date()}")

    # ---- Timeline ----
    all_slots, timestamp_to_idx, idx_to_timestamp = build_timeline(start_date, end_date)
    time_steps = len(all_slots)
    print(f"Timeline   : {time_steps} slots ({TIME_INTERVAL_MINUTES}-min bins)")

    # ---- Downtimes / shifts ----
    print("\nLoading downtimes and shifts...")
    downtimes_df, shifts_df = load_downtimes_and_shifts(data_dir)

    downtime_types = downtimes_df["DOWNTIME_TYPE"].dropna().unique().tolist()
    print(f"Downtime types found in CSV: {sorted(downtime_types)}")

    weather_dt_intervals = downtimes_df[downtimes_df["DOWNTIME_TYPE"] == "Weather"]
    technical_dt_intervals = downtimes_df[downtimes_df["DOWNTIME_TYPE"] == "Technical"]
    scheduling_dt_intervals = downtimes_df[downtimes_df["DOWNTIME_TYPE"] == "Scheduling"]
    engineering_intervals = shifts_df[shifts_df["SHIFT_ACTIVITY"].isin(["Engineering", "EOC"])]

    weather_dt_indices = get_indices_in_intervals(weather_dt_intervals, all_slots)
    technical_dt_indices = get_indices_in_intervals(technical_dt_intervals, all_slots)
    scheduling_dt_indices = get_indices_in_intervals(scheduling_dt_intervals, all_slots)
    engineering_indices = get_indices_in_intervals(engineering_intervals, all_slots)

    print(f"  Weather downtime slots      : {len(weather_dt_indices)}")
    print(f"  Technical downtime slots     : {len(technical_dt_indices)}")
    print(f"  Scheduling downtime slots   : {len(scheduling_dt_indices)}")
    print(f"  Engineering/EOC slots        : {len(engineering_indices)}")

    # ---- Hard-coded annual February/March shutdowns ----
    print("\n--- Applying hard-coded annual shutdowns ---")
    shutdown_indices: Set[int] = set()
    for year, (sm, sd, em, ed) in HARDCODED_SHUTDOWNS.items():
        # Inclusive end: add 1 day so the mask uses < (start of next day)
        shutdown_start = pd.Timestamp(year=year, month=sm, day=sd, tz="UTC")
        # end is inclusive, so the exclusive upper bound is the next day at midnight
        end_day_plus1 = ed + 1
        end_month = em
        end_year = year
        # Handle month overflow (e.g., March 5 + 1 day = March 6, fine;
        # Feb 28 + 1 = Feb 29 in leap year or March 1 otherwise)
        try:
            shutdown_end = pd.Timestamp(year=end_year, month=end_month,
                                        day=end_day_plus1, tz="UTC")
        except ValueError:
            # Day overflowed the month (e.g., Feb 28 in non-leap year: +1 → March 1)
            shutdown_end = pd.Timestamp(year=end_year, month=end_month + 1,
                                        day=1, tz="UTC")

        # Only add slots that fall within the simulation window
        overlap_start = max(shutdown_start, start_date)
        overlap_end = min(shutdown_end, end_date + pd.Timedelta(days=1))
        if overlap_start >= overlap_end:
            print(f"  Year {year}: shutdown {shutdown_start.date()} – "
                  f"{(shutdown_end - pd.Timedelta(days=1)).date()} "
                  f"is outside simulation window — skipping.")
            continue

        mask = (all_slots >= shutdown_start) & (all_slots < shutdown_end)
        year_shutdown_idx = set(np.where(mask)[0].tolist())
        new_idx = year_shutdown_idx - engineering_indices
        shutdown_indices |= year_shutdown_idx
        print(f"  Year {year}: shutdown {shutdown_start.date()} – "
              f"{(shutdown_end - pd.Timedelta(days=1)).date()} "
              f"-> {len(year_shutdown_idx)} slot(s) total, "
              f"{len(new_idx)} not already in Engineering/EOC")

    engineering_indices = engineering_indices | shutdown_indices
    print(f"  Engineering/EOC slots after merging shutdowns: {len(engineering_indices)}")

    # ---- Overlap check: log any slot that belongs to more than one type ----
    type_sets = [
        ("Weather", weather_dt_indices),
        ("Technical", technical_dt_indices),
        ("Scheduling", scheduling_dt_indices),
        ("Engineering/EOC", engineering_indices),
    ]
    no_downtime_indices = set(range(time_steps)) - (
        weather_dt_indices | technical_dt_indices
        | scheduling_dt_indices | engineering_indices
    )
    type_sets_with_no_dt = type_sets + [("No downtime", no_downtime_indices)]

    print("\n--- Overlap check (downtime / no-downtime types) ---")
    overlap_found = False
    for i, (name_a, set_a) in enumerate(type_sets_with_no_dt):
        for name_b, set_b in type_sets_with_no_dt[i + 1:]:
            if name_a == name_b:
                continue
            overlap = set_a & set_b
            if overlap:
                overlap_found = True
                sorted_idx = sorted(overlap)
                times = [idx_to_timestamp[idx] for idx in sorted_idx]
                print(f"  OVERLAP: '{name_a}' and '{name_b}' -> {len(overlap)} slot(s)")
                print(f"    First: slot {sorted_idx[0]}  {times[0]}")
                print(f"    Last:  slot {sorted_idx[-1]}  {times[-1]}")
                if len(overlap) <= 10:
                    for idx in sorted_idx:
                        print(f"      slot {idx}  {idx_to_timestamp[idx]}")
                else:
                    for idx in sorted_idx[:5]:
                        print(f"      slot {idx}  {idx_to_timestamp[idx]}")
                    print(f"      ... and {len(overlap) - 5} more")

    # Slots in 3+ types (among the four downtime types only)
    for idx in range(time_steps):
        count = (
            (1 if idx in weather_dt_indices else 0)
            + (1 if idx in technical_dt_indices else 0)
            + (1 if idx in scheduling_dt_indices else 0)
            + (1 if idx in engineering_indices else 0)
        )
        if count >= 3:
            overlap_found = True
            types_here = [
                n for n, s in type_sets
                if idx in s
            ]
            print(f"  SLOT IN 3+ TYPES: slot {idx}  {idx_to_timestamp[idx]}  -> {types_here}")
    if not overlap_found:
        print("  No overlaps between types (each slot belongs to at most one category).")
    print("\n--- Step 1: Load phase-tracker readings ---")
    realized_weather, n_with = load_phase_tracker(
        data_dir, start_date, end_date, all_slots, time_steps
    )
    n_missing = time_steps - n_with
    print(f"  Slots with phase-tracker data: {n_with}")
    print(f"  Slots missing data           : {n_missing}")
    print_sample(realized_weather, idx_to_timestamp, time_steps)

    # ---- Step 2: Overwrite Weather downtime → (inf, -inf) ----
    print("\n--- Step 2: Overwrite Weather downtime → (inf, -inf) ---")
    n_overwritten = 0
    for idx in weather_dt_indices:
        if idx in realized_weather:
            realized_weather[idx] = (np.inf, -np.inf)
            n_overwritten += 1
    print(f"  Slots overwritten: {n_overwritten}")

    # ---- Step 3: Overwrite Technical downtime → (inf, -inf) ----
    print("\n--- Step 3: Overwrite Technical downtime → (inf, -inf) ---")
    n_overwritten = 0
    for idx in technical_dt_indices:
        if idx in realized_weather:
            realized_weather[idx] = (np.inf, -np.inf)
            n_overwritten += 1
    print(f"  Slots overwritten: {n_overwritten}")

    # ---- Step 4: Context-aware gap fill ----
    print("\n--- Step 4: Context-aware gap fill for remaining NaN slots ---")
    nan_before = count_nan(realized_weather)
    print(f"  NaN slots before gap fill: {nan_before}")

    bad_indices = weather_dt_indices | technical_dt_indices | engineering_indices

    def _is_valid_boundary(idx: int) -> bool:
        if idx < 0 or idx >= time_steps:
            return False
        if idx in bad_indices:
            return False
        pwv, rms = realized_weather[idx]
        return not (pd.isna(pwv) or pd.isna(rms) or np.isinf(pwv) or np.isinf(rms))

    # Collect contiguous runs of fillable NaN slots (not in any bad set)
    gaps: list = []
    i = 0
    while i < time_steps:
        pwv, rms = realized_weather[i]
        is_nan = pd.isna(pwv) or pd.isna(rms)
        if is_nan and i not in bad_indices:
            gap_start = i
            while i < time_steps:
                p, r = realized_weather[i]
                if not (pd.isna(p) or pd.isna(r)) or i in bad_indices:
                    break
                i += 1
            gaps.append((gap_start, i))  # [gap_start, i) are the NaN slots to fill
        else:
            i += 1

    n_linear = 0
    n_copy = 0
    n_bad_fill = 0

    for gap_start, gap_end in gaps:
        left_idx = gap_start - 1
        right_idx = gap_end
        left_valid = _is_valid_boundary(left_idx)
        right_valid = _is_valid_boundary(right_idx)
        gap_len = gap_end - gap_start

        if left_valid and right_valid:
            left_pwv, left_rms = realized_weather[left_idx]
            right_pwv, right_rms = realized_weather[right_idx]
            for k, slot in enumerate(range(gap_start, gap_end)):
                frac = (k + 1) / (gap_len + 1)
                interp_pwv = left_pwv + frac * (right_pwv - left_pwv)
                interp_rms = left_rms + frac * (right_rms - left_rms)
                realized_weather[slot] = (interp_pwv, interp_rms)
            n_linear += gap_len
        elif left_valid or right_valid:
            source_idx = left_idx if left_valid else right_idx
            val = realized_weather[source_idx]
            for slot in range(gap_start, gap_end):
                realized_weather[slot] = val
            n_copy += gap_len
        else:
            for slot in range(gap_start, gap_end):
                realized_weather[slot] = (np.inf, -np.inf)
            n_bad_fill += gap_len

    nan_after = count_nan(realized_weather)
    total_filled = n_linear + n_copy + n_bad_fill
    print(f"  Gaps found           : {len(gaps)}")
    print(f"  Slots filled (total) : {total_filled}")
    print(f"    Linear interpolated: {n_linear}")
    print(f"    Copied from 1 side : {n_copy}")
    print(f"    Filled as downtime : {n_bad_fill}")
    print(f"  NaN slots after fill : {nan_after}")
    nan_in_engineering = sum(1 for i in engineering_indices
                            if pd.isna(realized_weather[i][0]) or pd.isna(realized_weather[i][1]))
    nan_in_weather_dt = sum(1 for i in weather_dt_indices
                            if pd.isna(realized_weather[i][0]) or pd.isna(realized_weather[i][1]))
    nan_in_technical_dt = sum(1 for i in technical_dt_indices
                              if pd.isna(realized_weather[i][0]) or pd.isna(realized_weather[i][1]))
    nan_unexpected = nan_after - nan_in_engineering - nan_in_weather_dt - nan_in_technical_dt
    print(f"  Breakdown of remaining NaN slots:")
    print(f"    In engineering_indices  : {nan_in_engineering}  (expected; Step 5 sets these to (NaN,NaN))")
    print(f"    In weather_dt_indices   : {nan_in_weather_dt}   (should be 0; Step 2 sets these to inf)")
    print(f"    In technical_dt_indices : {nan_in_technical_dt} (should be 0; Step 3 sets these to inf)")
    print(f"    In no downtime category : {nan_unexpected}       *** should be 0; non-zero = bug ***")
    if nan_unexpected > 0:
        print(f"  WARNING: Step 4 left {nan_unexpected} NaN slots not in any downtime category!")
    print_sample(realized_weather, idx_to_timestamp, time_steps)

    # ---- Step 5: Overwrite Engineering/EOC → (NaN, NaN) ----
    print("\n--- Step 5: Overwrite Engineering/EOC → (NaN, NaN) ---")
    n_overwritten = 0
    for idx in engineering_indices:
        if idx in realized_weather:
            realized_weather[idx] = (np.nan, np.nan)
            n_overwritten += 1
    print(f"  Slots overwritten: {n_overwritten}")

    # ---- Final summary ----
    nan_final = count_nan(realized_weather)
    inf_final = sum(1 for p, r in realized_weather.values() if np.isinf(p) or np.isinf(r))
    finite_final = time_steps - nan_final - inf_final

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total slots         : {time_steps}")
    print(f"  Finite weather      : {finite_final}")
    print(f"  Inf (bad weather)   : {inf_final}")
    print(f"  NaN (engineering)   : {nan_final}")
    print_sample(realized_weather, idx_to_timestamp, time_steps)

    # ---- Save ----
    output_data = {
        "realized_weather": realized_weather,
        "idx_to_timestamp": idx_to_timestamp,
        "start_date": start_date,
        "end_date": end_date,
        "weather_downtime_indices": weather_dt_indices,
        "technical_downtime_indices": technical_dt_indices,
        "scheduling_downtime_indices": scheduling_dt_indices,
        "engineering_indices": engineering_indices,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)
    print(f"\nSaved preprocessed weather to {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
