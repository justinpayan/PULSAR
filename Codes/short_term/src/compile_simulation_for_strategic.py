"""
Compile simulation data for strategic scheduling.

This script processes dsa_sim_*_df.csv files and creates cycle-specific
simulation files that exclude dates within each cycle.

For each cycle (2017, 2018, 2021, 2022, 2023), it collects all SB_UID, timestamp
pairs from dates that are NOT part of that cycle. Timestamps are normalized
to the 2023-2024 cycle, and configuration information is included.
"""

import pandas as pd
import numpy as np
import os
import glob
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import tqdm
import time
from collections import defaultdict


def get_cycle_dates(year):
    """
    Get the start and end dates for a cycle.
    Cycles run from October 1 of year Y to September 30 of year Y+1.
    
    Args:
        year: The starting year of the cycle (e.g., 2017)
    
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    start_date = datetime(year, 10, 1, 12, 0, 0)
    end_date = datetime(year + 1, 9, 30, 23, 59, 59)
    return start_date, end_date


def is_date_in_cycle(date, cycle_start, cycle_end):
    """
    Check if a date falls within a cycle.
    
    Args:
        date: datetime object to check
        cycle_start: datetime object for cycle start
        cycle_end: datetime object for cycle end
    
    Returns:
        bool: True if date is in cycle, False otherwise
    """
    return cycle_start <= date <= cycle_end


def parse_filename(filename):
    """
    Parse a dsa_sim filename to extract date information.
    
    Filename format: dsa_sim_{month}_{day}_{year}_df.csv
    
    Args:
        filename: The filename to parse
    
    Returns:
        tuple: (year, month, day) or None if parsing fails
    """
    try:
        # Extract base name without path
        basename = os.path.basename(filename)
        # Remove extension
        basename = basename.replace('_df.csv', '').replace('.csv', '')
        # Split by underscores
        parts = basename.split('_')
        # Format: dsa_sim_{month}_{day}_{year}
        if len(parts) >= 5 and parts[0] == 'dsa' and parts[1] == 'sim':
            month = int(parts[2])
            day = int(parts[3])
            year = int(parts[4])
            return year, month, day
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse filename {filename}: {e}")
    return None


def normalize_timestamp_to_2023_cycle(timestamp_str, original_date):
    """
    Normalize a timestamp to the 2023-2024 cycle.
    
    The timestamp is normalized by keeping the same month, day, and time,
    but changing the year to 2023 (or 2024 if the original date was after
    the cycle boundary in the original year).
    
    Args:
        timestamp_str: The timestamp string from the CSV
        original_date: The date from the filename (datetime object)
    
    Returns:
        str: Normalized timestamp string
    """
    try:
        # Parse the timestamp
        if isinstance(timestamp_str, str):
            # Try to parse various timestamp formats
            try:
                ts = pd.to_datetime(timestamp_str)
            except:
                # If parsing fails, try to extract time components
                ts = pd.to_datetime(timestamp_str, errors='coerce')
        else:
            ts = pd.to_datetime(timestamp_str)
        
        if pd.isna(ts):
            return None
        
        # Get the month, day, hour, minute, second from the original timestamp
        month = ts.month
        day = ts.day
        hour = ts.hour
        minute = ts.minute
        second = ts.second
        
        # Determine the year: if original date was Oct-Dec, use 2023; if Jan-Sep, use 2024
        # But we need to check the original date's position in its cycle
        # Actually, simpler: if month >= 10, use 2023; if month < 10, use 2024
        if month >= 10:
            normalized_year = 2023
        else:
            normalized_year = 2024
        
        # Create normalized timestamp
        normalized_ts = datetime(normalized_year, month, day, hour, minute, second)
        return normalized_ts.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Warning: Could not normalize timestamp {timestamp_str}: {e}")
        return None


def get_source_cycle_year(file_year, file_month):
    """Return the cycle year for a file date."""
    return file_year if file_month >= 10 else file_year - 1


def resolve_realized_weather_path(preprocessed_root, cycle_year):
    """Resolve the preprocessed realized weather pickle for a cycle."""
    return os.path.join(preprocessed_root, f"year_{cycle_year}", "realized_weather.pkl")


def load_realized_weather_lookup(preprocessed_root, cycle_year, weather_cache):
    """Load and cache realized weather as a timestamp -> (pwv, rms) mapping."""
    if cycle_year in weather_cache:
        return weather_cache[cycle_year]

    weather_path = resolve_realized_weather_path(preprocessed_root, cycle_year)
    if not os.path.exists(weather_path):
        raise FileNotFoundError(
            f"Could not find realized weather pickle for cycle {cycle_year}: {weather_path}"
        )

    with open(weather_path, "rb") as f:
        weather_data = pickle.load(f)

    realized_weather = weather_data.get("realized_weather")
    idx_to_timestamp = weather_data.get("idx_to_timestamp")
    if realized_weather is None or idx_to_timestamp is None:
        raise KeyError(
            f"Expected 'realized_weather' and 'idx_to_timestamp' in {weather_path}"
        )

    weather_lookup = {}
    for idx, timestamp in idx_to_timestamp.items():
        if idx not in realized_weather:
            continue
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        weather_lookup[timestamp] = realized_weather[idx]

    weather_cache[cycle_year] = {
        "path": weather_path,
        "lookup": weather_lookup,
    }
    print(
        f"Loaded realized weather for cycle {cycle_year} from {weather_path} "
        f"({len(weather_lookup)} timestamps)"
    )
    return weather_cache[cycle_year]


def load_configuration_mapping(data_dir):
    """
    Load configuration mapping from sb_12m_pressure.csv and interpolate
    for all 30-minute intervals in the 2023-2024 cycle.
    
    For missing intervals, checks the 2 closest times that have mappings,
    verifies they have the same configuration, and uses that value.
    
    Args:
        data_dir: Directory containing the reference file
    
    Returns:
        dict: Dictionary mapping Date (datetime) -> ARRAY (configuration)
    """
    reference_file = os.path.join(data_dir, 'sb_12m_pressure.csv')
 
    print(f"Loading configuration mapping from {reference_file}...")
    df_ref = pd.read_csv(reference_file)
    
    # Normalize Date column to datetime and ensure UTC timezone-aware
    df_ref['Date'] = pd.to_datetime(df_ref['Date'])
    # Ensure all dates are UTC timezone-aware pandas Timestamps
    if df_ref['Date'].dt.tz is None:
        df_ref['Date'] = df_ref['Date'].dt.tz_localize('UTC')
    else:
        df_ref['Date'] = df_ref['Date'].dt.tz_convert('UTC')
    
    # Get unique Date-ARRAY pairs (in case there are duplicates)
    df_ref_unique = df_ref[['Date', 'ARRAY']].drop_duplicates()
    
    # Create initial mapping from existing data
    config_map = {}
    for _, row in df_ref_unique.iterrows():
        date_key = row['Date']
        # Ensure it's a UTC timezone-aware pandas Timestamp
        if isinstance(date_key, pd.Timestamp):
            if date_key.tz is None:
                date_key = date_key.tz_localize('UTC')
            else:
                date_key = date_key.tz_convert('UTC')
        else:
            date_key = pd.Timestamp(date_key).tz_localize('UTC')
        config_map[date_key] = row['ARRAY']
    
    print(f"Loaded {len(config_map)} configuration mappings from reference file")
    
    # Generate all 30-minute intervals for the 2023-2024 cycle
    # Cycle: 2023-10-01 00:00:00 to 2024-09-30 23:59:59 (UTC)
    cycle_start = pd.Timestamp('2023-10-01 12:00:00', tz='UTC')
    cycle_end = pd.Timestamp('2024-09-30 23:59:59', tz='UTC')
    
    # Create all 30-minute intervals as UTC timezone-aware pandas Timestamps
    all_intervals = []
    current = cycle_start
    while current <= cycle_end:
        all_intervals.append(current)
        current = current + pd.Timedelta(minutes=30)
    
    print(f"Generated {len(all_intervals)} 30-minute intervals for 2023-2024 cycle")
    
    # Interpolate missing intervals
    for interval in all_intervals:
        print(f"Processing interval: {interval}")
        # Skip if already in map
        if interval in config_map:
            continue
        
        # Find the 2 closest times that have mappings
        # We'll look for times before and after this interval
        times_before = [t for t in config_map.keys() if t < interval]
        times_after = [t for t in config_map.keys() if t > interval]
        
        closest_before = max(times_before) if times_before else None
        closest_after = min(times_after) if times_after else None
        
        # We need the closest time before and after the interval
        config1 = config_map[closest_before]
        config2 = config_map[closest_after]

        if closest_before is None:
            config_map[interval] = config2
        elif closest_after is None:
            config_map[interval] = config1
        elif abs(interval - closest_before) < abs(interval - closest_after):
            config_map[interval] = config1
        else:
            config_map[interval] = config2

    print(f"Final configuration mapping has {len(config_map)} entries")
    return config_map


def compile_simulation_for_cycle(
        data_dir,
        output_dir,
        cycle_year,
        config_map=None,
        write_interval=500,
        preprocessed_root=None,
):
    """
    Compile simulation data for a specific cycle, excluding dates within that cycle.
    
    Args:
        data_dir: Directory containing dsa_sim_*_df.csv files
        output_dir: Directory to save output CSV
        cycle_year: The starting year of the cycle (e.g., 2017)
        config_map: Dictionary mapping (Date) -> ARRAY configuration
        write_interval: Write output file every N files processed (default: 500)
        preprocessed_root: Root directory containing year_<cycle>/realized_weather.pkl
    """
    print(f"\n{'=' * 80}")
    print(f"Processing cycle {cycle_year} (from {cycle_year}-10-01 to {cycle_year+1}-09-30)")
    print(f"{'=' * 80}")
    
    # Get cycle date range
    cycle_start, cycle_end = get_cycle_dates(cycle_year)
    print(f"Cycle dates: {cycle_start.date()} to {cycle_end.date()}")
    
    # Find all dsa_sim files
    pattern = os.path.join(data_dir, 'dsa_sim', 'dsa_sim_*_df.csv')
    all_files = glob.glob(pattern)
    print(f"Found {len(all_files)} dsa_sim files")
    
    # Collect all SB_UID, timestamp pairs with weather data from dates outside the cycle
    all_pairs = []
    files_processed = 0
    files_skipped = 0
    output_filename = os.path.join(output_dir, f'sb_12m_pressure_{cycle_year}.csv')
    last_write_time = time.time()
    config_series = pd.Series(config_map) if config_map is not None else None
    weather_cache = {}
    weather_match_stats = defaultdict(lambda: {
        'files': 0,
        'rows': 0,
        'matched_rows': 0,
        'unmatched_rows': 0,
    })
    
    def write_intermediate_output():
        """Helper function to write current accumulated data to output file"""
        if len(all_pairs) == 0:
            return
        
        # Combine all pairs so far and calculate weather fractions
        combined_df = _calculate_weather_fractions(all_pairs)
        
        # Write to CSV
        combined_df.to_csv(output_filename, index=False)
        print(f"\n  [Intermediate write] Saved {len(combined_df)} unique pairs to {output_filename}")
    
    for filename in tqdm.tqdm(all_files):
        # Parse filename to get date
        date_info = parse_filename(filename)
        if date_info is None:
            print(f"Warning: Could not parse date from {filename}")
            files_skipped += 1
            continue
        file_year, file_month, file_day = date_info
        file_date = datetime(file_year, file_month, file_day)
        
        # Skip if this date is within the cycle
        if is_date_in_cycle(file_date, cycle_start, cycle_end):
            files_skipped += 1
            continue
        
        # Read the CSV file
        try:
            df = pd.read_csv(filename)
        except Exception as e:
            print(f"Warning: Could not read {filename}: {e}")
            files_skipped += 1
            continue

        source_cycle_year = get_source_cycle_year(file_year, file_month)
        try:
            weather_info = load_realized_weather_lookup(
                preprocessed_root=preprocessed_root,
                cycle_year=source_cycle_year,
                weather_cache=weather_cache,
            )
        except Exception as e:
            print(f"Warning: Could not load realized weather for {filename}: {e}")
            files_skipped += 1
            continue

        # Check required columns exist
        required_cols = ['sbuid', 'timestamp', 'pwv_thresh', 'rms_thresh']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in {filename}: {missing_cols}")
            files_skipped += 1
            continue

        # Extract SB_UID, timestamp, and weather thresholds
        pairs_df = df[['sbuid', 'timestamp', 'pwv_thresh', 'rms_thresh']].copy()

        # Normalize timestamps to 2023-2024 cycle (fully vectorized)
        # Convert timestamp strings to datetime
        pairs_df['timestamp_dt'] = pd.to_datetime(pairs_df['timestamp'], utc=True, errors='coerce')
        invalid_timestamp_count = int(pairs_df['timestamp_dt'].isna().sum())
        if invalid_timestamp_count > 0:
            print(f"Warning: {invalid_timestamp_count} rows in {filename} had invalid timestamps and were dropped")
            pairs_df = pairs_df[pairs_df['timestamp_dt'].notna()].copy()
        if pairs_df.empty:
            print(f"Warning: No usable rows in {filename} after timestamp parsing")
            files_skipped += 1
            continue

        pairs_df['timestamp_weather_lookup'] = pairs_df['timestamp_dt'].dt.floor('30min')
        pairs_df['realized_weather'] = pairs_df['timestamp_weather_lookup'].map(weather_info['lookup'])
        pairs_df['realized_pwv'] = pairs_df['realized_weather'].apply(
            lambda x: x[0] if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 2 else np.nan
        )
        pairs_df['realized_freqrms'] = pairs_df['realized_weather'].apply(
            lambda x: x[1] if isinstance(x, (tuple, list, np.ndarray)) and len(x) >= 2 else np.nan
        )

        matched_rows = int(pairs_df['realized_weather'].notna().sum())
        unmatched_rows = int(len(pairs_df) - matched_rows)
        weather_match_stats[source_cycle_year]['files'] += 1
        weather_match_stats[source_cycle_year]['rows'] += len(pairs_df)
        weather_match_stats[source_cycle_year]['matched_rows'] += matched_rows
        weather_match_stats[source_cycle_year]['unmatched_rows'] += unmatched_rows

        # Extract month to determine target year
        orig_month = pairs_df['timestamp_dt'].dt.month
        
        # Determine target year: if month >= 10, use 2023, else 2024 (vectorized)
        target_year = (orig_month >= 10).astype(int) * 2023 + (orig_month < 10).astype(int) * 2024
        
        # Create normalized timestamps using vectorized year replacement
        pairs_df['timestamp_normalized'] = pd.to_datetime({
            'year': target_year,
            'month': pairs_df['timestamp_dt'].dt.month,
            'day': pairs_df['timestamp_dt'].dt.day,
            'hour': pairs_df['timestamp_dt'].dt.hour,
            'minute': pairs_df['timestamp_dt'].dt.minute,
            'second': pairs_df['timestamp_dt'].dt.second
        }, utc=True)
        
        # Round to nearest 30-minute interval (fully vectorized)
        pairs_df['timestamp_rounded'] = pairs_df['timestamp_normalized'].dt.floor('30min')
        
        # Check weather suitability using realized weather from the preprocessed pickle.
        # Handle NaN values: if any weather value is NaN, count as not suitable
        pairs_df['weather_suitable'] = (
            (pairs_df['realized_pwv'] <= pairs_df['pwv_thresh']) &
            (pairs_df['realized_freqrms'] >= pairs_df['rms_thresh']) &
            (pairs_df['realized_pwv'].notna()) &
            (pairs_df['realized_freqrms'].notna()) &
            (pairs_df['pwv_thresh'].notna()) &
            (pairs_df['rms_thresh'].notna())
        )
        
        # Add source year for tracking which year each entry comes from
        pairs_df['source_year'] = file_year
        
        # Add configuration if config_map is available (vectorized lookup)
        if config_series is not None:
            pairs_df['ARRAY'] = pairs_df['timestamp_rounded'].map(config_series)
        else:
            pairs_df['ARRAY'] = None
        
        # Keep needed columns
        pairs_df = pairs_df[[
            'sbuid', 'timestamp_rounded', 'ARRAY', 'weather_suitable', 'source_year'
        ]].copy()
        
        pairs_df = pairs_df.rename(columns={
            'sbuid': 'SB_UID',
            'timestamp_rounded': 'Date'
        })
        
        all_pairs.append(pairs_df)
        files_processed += 1
        
        # Periodically write output file
        if files_processed % write_interval == 0:
            print(f"\n  Processed {files_processed} files...")
            write_intermediate_output()
            last_write_time = time.time()
        elif files_processed % 100 == 0:
            print(f"  Processed {files_processed} files...")
        
        # Also write if it's been more than 5 minutes since last write
        current_time = time.time()
        if current_time - last_write_time > 300:  # 5 minutes
            print(f"\n  [Time-based write] Writing after {files_processed} files...")
            write_intermediate_output()
            last_write_time = current_time
    
    print(f"\nFiles processed: {files_processed}")
    print(f"Files skipped (within cycle or errors): {files_skipped}")
    if weather_match_stats:
        print("\nRealized weather match summary by source cycle:")
        for source_cycle in sorted(weather_match_stats):
            stats = weather_match_stats[source_cycle]
            match_pct = (stats['matched_rows'] / stats['rows'] * 100.0) if stats['rows'] else 0.0
            print(
                f"  Cycle {source_cycle}: files={stats['files']}, rows={stats['rows']}, "
                f"matched={stats['matched_rows']}, unmatched={stats['unmatched_rows']} "
                f"({match_pct:.1f}% matched)"
            )
    
    if len(all_pairs) == 0:
        print(f"WARNING: No data collected for cycle {cycle_year}")
        return
    
    # Final write (combine, calculate weather fractions, deduplicate, sort, and save)
    print("\nCalculating weather suitability fractions...")
    combined_df = _calculate_weather_fractions(all_pairs)
    
    # Save final output to CSV
    combined_df.to_csv(output_filename, index=False)
    print(f"\n[Final write] Saved {len(combined_df)} unique SB_UID, timestamp pairs to {output_filename}")
    print(f"Output columns: {list(combined_df.columns)}")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Unique SB_UIDs: {combined_df['SB_UID'].nunique()}")
    print(f"  Unique dates: {combined_df['Date'].nunique()}")
    if 'ARRAY' in combined_df.columns:
        print(f"  Rows with configuration: {combined_df['ARRAY'].notna().sum()} ({combined_df['ARRAY'].notna().sum() / len(combined_df) * 100:.1f}%)")
        print(f"  Unique configurations: {combined_df['ARRAY'].nunique()}")
    if 'weather_suitable_fraction' in combined_df.columns:
        print(f"  Weather suitability fraction statistics:")
        print(f"    Mean: {combined_df['weather_suitable_fraction'].mean():.3f}")
        print(f"    Median: {combined_df['weather_suitable_fraction'].median():.3f}")
        print(f"    Min: {combined_df['weather_suitable_fraction'].min():.3f}")
        print(f"    Max: {combined_df['weather_suitable_fraction'].max():.3f}")
    print(f"  Date range in data: {combined_df['Date'].min()} to {combined_df['Date'].max()}")


def _calculate_weather_fractions(all_pairs):
    """
    Calculate weather suitability fractions for each (SB_UID, Date) pair.
    
    For each unique (SB_UID, Date) pair, this function:
    1. Groups all entries by source year
    2. For each source year, determines if weather was suitable (at least one suitable entry)
    3. Calculates fraction: (years with suitable weather) / (total unique source years)
       If a year doesn't have data for a particular (SB_UID, Date), it counts as not suitable.
    
    Args:
        all_pairs: List of DataFrames, each containing columns:
                   SB_UID, Date, ARRAY, weather_suitable, source_year
    
    Returns:
        DataFrame with columns: SB_UID, Date, ARRAY, weather_suitable_fraction
    """
    if len(all_pairs) == 0:
        return pd.DataFrame()
    
    # Combine all pairs
    combined = pd.concat(all_pairs, ignore_index=True)
    
    # Get all unique source years (all years that could potentially have data, but the current year wasn't counted)
    all_source_years = sorted(combined['source_year'].unique())
    total_years = len(all_source_years) - 1
    print(f"Found {total_years} unique source years: {all_source_years}")
    
    # Fill NaN values with False (missing data counts as not suitable)
    combined['weather_suitable'] = combined['weather_suitable'].fillna(False)
    
    # For each (SB_UID, Date) pair, aggregate by source year
    # Group by SB_UID, Date, and source_year
    # For each group, if ANY entry has weather_suitable=True, count that year as suitable
    year_aggregated = combined.groupby(['SB_UID', 'Date', 'source_year'], as_index=False).agg({
        'weather_suitable': 'any',  # True if any entry in this year was suitable
        'ARRAY': 'first'  # Take first ARRAY value (should be same for same Date)
    })
    
    # Now for each (SB_UID, Date) pair, calculate fraction across ALL years
    # If a year doesn't have data for this (SB_UID, Date), it counts as not suitable (False)
    weather_fractions = []
    
    for (sb_uid, date), group in year_aggregated.groupby(['SB_UID', 'Date']):
        # Get years that have data for this (SB_UID, Date) pair
        years_with_data = set(group['source_year'].unique())
        
        # Count how many years had suitable weather (among years with data).
        suitable_count = group['weather_suitable'].sum() 
        
        # Years without data count as not suitable (0), so:
        # Total suitable years = years with suitable weather from data we have
        # Total years = all unique source years
        # Years missing data = total_years - len(years_with_data), all count as 0 (not suitable)
        
        fraction = suitable_count / total_years
        
        # Get ARRAY value (should be same for all entries with same Date)
        array_value = group['ARRAY'].iloc[0] if len(group) > 0 else None
        
        weather_fractions.append({
            'SB_UID': sb_uid,
            'Date': date,
            'ARRAY': array_value,
            'weather_suitable_fraction': fraction
        })
    
    weather_fractions_df = pd.DataFrame(weather_fractions)
    
    # Remove duplicates (shouldn't be any, but just in case)
    weather_fractions_df = weather_fractions_df.drop_duplicates(subset=['SB_UID', 'Date'])
    
    # Sort by Date first, then by SB_UID
    weather_fractions_df = weather_fractions_df.sort_values(['Date', 'SB_UID']).reset_index(drop=True)
    
    return weather_fractions_df


def main():
    parser = argparse.ArgumentParser(
        description="Compile simulation data for strategic scheduling by cycle"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/data/user_data/jpayan/AOOSP/data/',
        help='Directory containing dsa_sim_*_df.csv files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/user_data/jpayan/AOOSP/data/',
        help='Directory to save output CSV files'
    )
    parser.add_argument(
        '--write_interval',
        type=int,
        default=500,
        help='Write output file every N files processed (default: 500)'
    )
    parser.add_argument(
        '--cycles',
        type=int,
        nargs='+',
        default=[2017, 2018, 2021, 2022, 2023],
        help='Cycle years to process (default: 2017 2018 2021 2022 2023)'
    )
    parser.add_argument(
        '--preprocessed_root',
        type=str,
        default=None,
        help='Root directory containing year_<cycle>/realized_weather.pkl (defaults to data_dir)'
    )
    
    args = parser.parse_args()
    preprocessed_root = args.preprocessed_root or args.data_dir
    
    # Load configuration mapping once for all cycles
    print("Loading configuration mapping from reference file...")
    config_map = load_configuration_mapping(args.data_dir)
    
    # Process each cycle
    for cycle_year in args.cycles:
        compile_simulation_for_cycle(
            args.data_dir, 
            args.output_dir, 
            cycle_year, 
            config_map=config_map,
            write_interval=args.write_interval,
            preprocessed_root=preprocessed_root,
        )
    
    print(f"\n{'=' * 80}")
    print("All cycles processed successfully!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

