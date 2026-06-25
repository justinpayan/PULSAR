# long_term_optim_configurable.py
import gurobipy as gp
from gurobipy import GRB, quicksum
import pandas as pd
import argparse
from pathlib import Path
import json
import os
import shutil
import tempfile
import glob
import pickle
import hashlib

import sys
# %% Import libraries
import pandas as pd
import numpy as np
from datetime import timedelta
from random import randint
import random as random
from itertools import cycle
import matplotlib.pyplot as plt
from configuration import configurable_dict
from collections import defaultdict
import re
import tqdm
import math
import gurobipy as gp
from gurobipy import GRB


def build_config_calendar(data_dir: str, start_date: str, end_date: str = None, year: int = None) -> pd.DataFrame:
    """
    Lightweight loader that builds a (Configuration, Start, End) calendar from
    expected_times_c10_for_strategic.csv only. Mirrors the calendar-construction
    logic inside preprocess() but skips every other CSV load and debug print.

    Args:
        data_dir: Directory containing expected_times_c10_for_strategic.csv
        start_date: Cycle start date string (YYYY-MM-DD or any pd-parseable form)
        end_date: Optional cycle end date string
        year: Cycle start year, used to map cycle dates onto the availability data's
              actual year range. If None, inferred from start_date.

    Returns:
        DataFrame with columns ['Configuration', 'Start', 'End'] sorted by Start.
    """
    availability_path = os.path.join(data_dir, "expected_times_c10_for_strategic.csv")
    df_availability = pd.read_csv(availability_path, index_col=False)
    df_availability['timestamp'] = df_availability['timestamp'].apply(lambda x: pd.to_datetime(x, utc=True))

    start_date_parsed = pd.to_datetime(start_date, utc=True)
    end_date_parsed = pd.to_datetime(end_date, utc=True) if end_date else None
    cycle_start_year = year if year is not None else start_date_parsed.year

    if len(df_availability) == 0:
        raise ValueError(f"{availability_path} is empty; cannot build configuration calendar.")

    avail_min_date = df_availability['timestamp'].min()
    avail_max_date = df_availability['timestamp'].max()
    avail_start_year = avail_min_date.year
    avail_end_year = avail_max_date.year

    def replace_year(d, target_year):
        try:
            return d.replace(year=target_year)
        except ValueError:
            return d.replace(year=target_year, day=28)

    def year_for_date(d):
        if d.year == cycle_start_year:
            return avail_start_year
        if d.year == cycle_start_year + 1:
            return avail_end_year
        return d.year

    normalized_start_date = replace_year(start_date_parsed, year_for_date(start_date_parsed))
    normalized_end_date = (
        replace_year(end_date_parsed, year_for_date(end_date_parsed))
        if end_date_parsed is not None else None
    )

    after_start = df_availability[df_availability['timestamp'] > normalized_start_date].copy()
    if len(after_start) > 0:
        first_config_after_start = after_start['conf'].iloc[0]
        actual_filter_date = normalized_start_date
        for _, row in after_start.iterrows():
            if row['conf'] != first_config_after_start:
                time_until_change = row['timestamp'] - after_start['timestamp'].iloc[0]
                if time_until_change.total_seconds() < 24 * 3600:
                    actual_filter_date = row['timestamp'] - timedelta(seconds=1)
                break
    else:
        actual_filter_date = normalized_start_date

    df_availability = df_availability[df_availability['timestamp'] > actual_filter_date]
    if normalized_end_date is not None:
        df_availability = df_availability[df_availability['timestamp'] <= normalized_end_date]
    df_availability = df_availability.reset_index(drop=True)

    if len(df_availability) == 0:
        raise ValueError(
            "No availability rows left after filtering by start/end date. "
            "Check that filter dates use the correct year for the availability data."
        )

    calendar = pd.DataFrame(columns=['Configuration', 'Start', 'End'])
    begin_calendar = pd.DataFrame(
        {'Configuration': df_availability['conf'][0], 'Start': df_availability['timestamp'][0]},
        index=[0],
    )
    calendar = pd.concat([begin_calendar, calendar.loc[:]])

    actual_c = calendar['Configuration'][0]
    for row in range(len(df_availability)):
        if df_availability['conf'][row] != actual_c:
            add_row_calendar = pd.DataFrame(
                {'Configuration': df_availability['conf'][row], 'Start': df_availability['timestamp'][row]},
                index=[0],
            )
            calendar = pd.concat([add_row_calendar, calendar.loc[:]]).reset_index(drop=True)
            actual_c = df_availability['conf'][row]

    calendar['Start'] = pd.to_datetime(calendar['Start']).apply(lambda a: pd.to_datetime(a).date())
    calendar['End'] = pd.to_datetime(calendar['End']).apply(lambda a: pd.to_datetime(a).date())
    calendar['Start'] = pd.to_datetime(calendar['Start'], utc=True)
    calendar['End'] = pd.to_datetime(calendar['End'], utc=True)

    calendar = calendar.sort_values(by=['Start']).reset_index(drop=True)
    for row in range(1, len(calendar['Start'])):
        calendar['End'][row - 1] = calendar['Start'][row] - timedelta(days=1)
    last_timestamp = df_availability['timestamp'].iloc[-1]
    calendar['End'].iloc[-1] = pd.to_datetime(last_timestamp.date(), utc=True)

    calendar['Start'] = pd.to_datetime(calendar['Start'], utc=True)
    calendar['End'] = pd.to_datetime(calendar['End'], utc=True)

    return calendar


def preprocess(configurable_dict, jobs, start_date, sb_map, data_dir: str = None, output_path: str = None, end_date: str = None, cycle_start_year: int = None):
    main = configurable_dict['main']
    # name of file with available time per bin
    availability = configurable_dict['availability']
    # name of file of where can be the sb executed
    simulation = configurable_dict['simulation']
    # name of file with extra information for projects
    modes = configurable_dict['modes']
    # Get year if available (for loading pressure file with lst)
    year = configurable_dict.get('year', None)
    suppress_fraction_obs_debug = bool(configurable_dict.get('suppress_fraction_obs_debug', False))

    def resolve_input_path(path_value: str) -> str:
        if data_dir and not os.path.isabs(path_value):
            candidate = os.path.join(data_dir, path_value)
            if os.path.exists(candidate):
                return candidate
        return path_value

    # %% load dataframes
    print(f"\n--- Loading Data Files ---")
    main = resolve_input_path(main)
    availability = resolve_input_path(availability)
    simulation = resolve_input_path(simulation)
    modes = resolve_input_path(modes)
    print(f"Loading main file from: {main}")
    
    df_main = pd.read_csv(main, index_col=False)
    print(f"Initial df_main shape: {df_main.shape} (rows, cols)")
    print(f"Initial df_main columns: {list(df_main.columns)}")
    
    if len(df_main) == 0:
        print(f"ERROR: File {main} is empty or has no data rows!")
    
    # Show first few rows for debugging
    print(f"First 3 rows of df_main:\n{df_main.head(3)}")
    
    df_availability = pd.read_csv(availability, index_col=False)
    df_simulation = pd.read_csv(simulation, index_col=False)
    df_modes = pd.read_csv(modes)
    print(f"Loaded availability: {df_availability.shape}, simulation: {df_simulation.shape}, modes: {df_modes.shape}")
    # set 'timestamp' column in timestamp
    df_availability['timestamp'] = df_availability.timestamp.apply(lambda x: pd.to_datetime(x, utc = True))

    ## add lines 30-34 of re_run_code.py using period_start_date tk
    # === YEAR NORMALIZATION ===
    # Normalize input dates to the availability data's year range using the cycle's original start year.
    # When cycle_start_year is passed (e.g. 2023): dates in cycle_start_year -> first year of availability,
    # dates in cycle_start_year + 1 -> second year of availability. So Jan 2024 stays 2024, not 2023.
    
    start_date_parsed = pd.to_datetime(start_date, utc=True)
    end_date_parsed = pd.to_datetime(end_date, utc=True) if end_date else None
    
    if len(df_availability) > 0:
        avail_min_date = df_availability['timestamp'].min()
        avail_max_date = df_availability['timestamp'].max()
        avail_start_year = avail_min_date.year
        avail_end_year = avail_max_date.year
        
        def replace_year(d, target_year):
            try:
                return d.replace(year=target_year)
            except ValueError:
                return d.replace(year=target_year, day=28)
        
        if cycle_start_year is not None:
            # Use passed-in cycle start year: cycle_start_year -> avail_start_year, cycle_start_year+1 -> avail_end_year
            def year_for_date(d):
                if d.year == cycle_start_year:
                    return avail_start_year
                if d.year == cycle_start_year + 1:
                    return avail_end_year
                return d.year  # leave other years unchanged
            normalized_start_date = replace_year(start_date_parsed, year_for_date(start_date_parsed))
            normalized_end_date = replace_year(end_date_parsed, year_for_date(end_date_parsed)) if end_date_parsed is not None else None
        else:
            # Fallback: infer from start/end dates (first year of range -> avail_start, second -> avail_end)
            end_year = end_date_parsed.year if end_date_parsed is not None else start_date_parsed.year
            input_start_year = min(start_date_parsed.year, end_year)
            input_end_year = max(start_date_parsed.year, end_year)
            def year_for_date(d):
                target = avail_end_year if d.year == input_end_year else avail_start_year
                return target
            normalized_start_date = replace_year(start_date_parsed, year_for_date(start_date_parsed))
            normalized_end_date = replace_year(end_date_parsed, year_for_date(end_date_parsed)) if end_date_parsed is not None else None
        
        if (start_date_parsed != normalized_start_date) or (end_date_parsed != normalized_end_date if end_date_parsed is not None else False):
            print(f"\n--- Year Normalization ---")
            if cycle_start_year is not None:
                print(f"  Cycle start year: {cycle_start_year} -> {avail_start_year}, {cycle_start_year + 1} -> {avail_end_year}")
            print(f"  Original start_date: {start_date_parsed}")
            if end_date_parsed is not None:
                print(f"  Original end_date: {end_date_parsed}")
            print(f"  Availability data range: {avail_min_date} to {avail_max_date}")
            print(f"  Normalized start_date: {normalized_start_date}")
            if normalized_end_date is not None:
                print(f"  Normalized end_date: {normalized_end_date}")
            print(f"--- End Year Normalization ---\n")
    else:
        normalized_start_date = start_date_parsed
        normalized_end_date = end_date_parsed
    
    # === FIND ACTUAL CONFIGURATION START ===
    # Configurations typically start in the evening, so filtering from midnight 
    # might pick up a few hours of the previous config from earlier that day.
    # We'll find the first configuration change AFTER the normalized start date
    # and use that as the actual filter point.
    
    print(f"\n--- Finding actual configuration start ---")
    print(f"  Original start_date: {start_date}")
    print(f"  Normalized start_date: {normalized_start_date}")
    
    if len(df_availability) > 0:
        # Find rows after the normalized start date
        after_start = df_availability[df_availability['timestamp'] > normalized_start_date].copy()
        
        if len(after_start) > 0:
            # Find the first configuration and when it changes
            first_config_after_start = after_start['conf'].iloc[0]
            
            # Find where the configuration changes (the actual start of the NEXT config)
            config_changes_after_start = []
            prev_config = first_config_after_start
            for idx, row in after_start.iterrows():
                if row['conf'] != prev_config:
                    config_changes_after_start.append({
                        'config': row['conf'],
                        'timestamp': row['timestamp']
                    })
                    prev_config = row['conf']
                    break  # We only need the first change
            
            # If the first config after start is just a remnant of the previous day's config,
            # snap to the next configuration change
            if config_changes_after_start:
                first_change = config_changes_after_start[0]
                # Check if the first config is just a small remnant (less than 12 hours)
                time_until_change = first_change['timestamp'] - after_start['timestamp'].iloc[0]
                
                if time_until_change.total_seconds() < 24 * 3600:  # Less than 24 hours
                    print(f"  First config ({first_config_after_start}) is a remnant ({time_until_change})")
                    print(f"  Snapping to next config ({first_change['config']}) at {first_change['timestamp']}")
                    # Use the timestamp just before the config change to filter
                    actual_filter_date = first_change['timestamp'] - timedelta(seconds=1)
                else:
                    print(f"  First config ({first_config_after_start}) has significant time ({time_until_change})")
                    actual_filter_date = normalized_start_date
            else:
                print(f"  No config changes found after normalized start, using normalized_start_date")
                actual_filter_date = normalized_start_date
        else:
            print(f"  No data after normalized start date")
            actual_filter_date = normalized_start_date
    else:
        actual_filter_date = normalized_start_date
    
    print(f"  Actual filter date: {actual_filter_date}")
    print(f"--- End finding actual start ---\n")
    
    # === DEBUG: Check filtering by start_date and end_date ===
    print(f"\n--- Filtering df_availability ---")
    print(f"  Actual filter start: {actual_filter_date}")
    if normalized_end_date:
        print(f"  Normalized end_date: {normalized_end_date}")
    print(f"  df_availability rows BEFORE filter: {len(df_availability)}")
    if len(df_availability) > 0:
        print(f"  df_availability date range BEFORE filter: {df_availability['timestamp'].min()} to {df_availability['timestamp'].max()}")
        unique_configs_before = df_availability['conf'].unique()
        print(f"  Unique configurations BEFORE filter: {sorted(unique_configs_before)}")
    
    # Filter by start date
    df_availability = df_availability[df_availability['timestamp'] > actual_filter_date]
    
    # Filter by end date if provided
    if normalized_end_date:
        df_availability = df_availability[df_availability['timestamp'] <= normalized_end_date]
        print(f"  Applied end_date filter: timestamp <= {normalized_end_date}")
    
    df_availability = df_availability.reset_index(drop=True)
    
    print(f"  df_availability rows AFTER filter: {len(df_availability)}")
    if len(df_availability) > 0:
        print(f"  df_availability date range AFTER filter: {df_availability['timestamp'].min()} to {df_availability['timestamp'].max()}")
        unique_configs_after = df_availability['conf'].unique()
        print(f"  Unique configurations AFTER filter: {sorted(unique_configs_after)}")
    else:
        raise ValueError(
            "No availability data left after filtering by start/end date. "
            "Check that filter start and end dates use the correct year (e.g. 2024-01-14 not 2023-01-14 when crossing into the new year)."
        )
    print(f"--- End filtering check ---\n")
    
    # Need to rename columns of main to what they used to be.
    # Current columns are SB_UID,PRJ_CODE,PRJ_GRADE,SB_TOTAL_ESTIMATED_TIME,NUMBER_OF_EXECUTIONS,SB_TIME_BY_EXECUTION,OPTIMAL_PWV,EXTENDED,COMPACT,ISTIMECONSTRAINED,ISTOO,fraction_obs_C1,fraction_obs_C2,fraction_obs_C3,fraction_obs_C4,fraction_obs_C5,fraction_obs_C6,fraction_obs_C7,fraction_obs_C8,prj_scientific_rank,CL,EA,EU,NA
    # Old columns were: SB_UID,PRJ_CODE,PRJ_GRADE,PRJ_SCIENTIFIC_RANK,SB_TOTAL_ESTIMATED_TIME,NUMBER_OF_EXECUTIONS,SB_TIME_BY_EXECUTION,OPTIMAL_PWV,EXTENDED,COMPACT,ISTIMECONSTRAINED,ISTOO,FRACTION_OBS_C1,FRACTION_OBS_C2,FRACTION_OBS_C3,FRACTION_OBS_C4,FRACTION_OBS_C5,FRACTION_OBS_C6,FRACTION_OBS_C7,FRACTION_OBS_C8,CL,EA,EU,NA
    
    # Check which columns exist before renaming
    rename_map = {}
    columns_to_check = {
        'prj_scientific_rank': 'PRJ_SCIENTIFIC_RANK',
        'fraction_obs_C1': 'FRACTION_OBS_C1',
        'fraction_obs_C2': 'FRACTION_OBS_C2',
        'fraction_obs_C3': 'FRACTION_OBS_C3',
        'fraction_obs_C4': 'FRACTION_OBS_C4',
        'fraction_obs_C5': 'FRACTION_OBS_C5',
        'fraction_obs_C6': 'FRACTION_OBS_C6',
        'fraction_obs_C7': 'FRACTION_OBS_C7',
        'fraction_obs_C8': 'FRACTION_OBS_C8'
    }
    
    for old_col, new_col in columns_to_check.items():
        if old_col in df_main.columns:
            rename_map[old_col] = new_col
        elif new_col in df_main.columns:
            print(f"Column {new_col} already exists, skipping rename")
        else:
            print(f"WARNING: Column {old_col} (or {new_col}) not found in df_main")
    
    if rename_map:
        df_main = df_main.rename(columns=rename_map)
        print(f"After rename, df_main shape: {df_main.shape}")
    
    # # --- Load DSA data and merge status information (matching DsaAlgorithm selector) ---
    # # Define eligible statuses (matching run_dsa.py lines 186-187)
    # eligible_prj_status = ("Ready", "InProgress", "PartiallyCompleted", "ObservingTimedOut")
    # eligible_sb_status = ["Ready", "Running", "ObservingTimedOut", "Phase2Submitted", "Waiting"]
    
    # rows_before_status_filter = len(df_main)
    
    # # Load DSA data from pkl file to get status information
    # import pickle
    # import sys
    # import os
    
    # # Set up DSA environment (similar to run_dsa.py)
    # DSA = "/home/jpayan/AOOSP/Codes/short_term/src/DSA/DSA/src/"
    # if DSA not in sys.path:
    #     sys.path.append(DSA)
    
    # os.environ['DSA'] = DSA
    # os.environ['CON_STR'] = 'placeholder'
    # os.environ['POL_FILE_PATH'] = '/home/jpayan/AOOSP/Codes/short_term/src/DSA/DSA/'
    
    # import DsaAlgorithm as Dsa
    # import DsaTools as DsaTool
    # import DsaScorers as DsaScore
    # from log_config import init_loggers

    # data_path = f'/data/user_data/jpayan/AOOSP/data/'

    # data_Path = Path(data_path)
    # logs = data_Path.joinpath('logs')
    # logs.mkdir(exist_ok=True)

    # init_loggers(
    #     {"handlers": {
    #         'daily_json_logfile': {
    #             'level': 'DEBUG',
    #             'class': "logging.handlers.TimedRotatingFileHandler",
    #             'formatter': 'json',
    #             'filename': '{0}/logs/dsa.log.json'.format("/home/jpayan/AOOSP/Codes/short_term/src/DSA/DSA"),
    #             'when': 'midnight',
    #             'interval': 1,
    #             'backupCount': 3
    #         },
    #         'console': {
    #             'level': 'INFO'
    #         },
    #         'daily_logfile': {
    #             'level': 'INFO',
    #             'filename': '{0}/logs/dsa.log'.format("/home/jpayan/AOOSP/Codes/short_term/src/DSA/DSA"),
    #             'backupCount': 1
    #         },
    #     },
    #         "loggers": {
    #             "dsa": {
    #                 "handlers": ['daily_json_logfile'],
    #                 "level": 'INFO'
    #             },
    #         }})

    # cycle_fix = 'c10'

    # with open(f'{data_path}/data_active_{cycle_fix}.pkl', 'rb') as filein:
    #     data = pickle.load(filein)

    # dsa12m = Dsa.DsaAlgorithm(data, 'TWELVE-M', path=data_path, aprc=False)
    
    # # Merge SB_STATE from data.sb_status (matching DsaAlgorithm.update_status)
    # print("Merging SB_STATE from data.sb_status...")
    # sb_status_df = dsa12m.data.sb_status.reset_index(drop=True)[['SB_UID', 'SB_STATE']].copy()
    # df_main = pd.merge(
    #     df_main,
    #     sb_status_df,
    #     on='SB_UID',
    #     how='left'
    # )
    # print(f"  Merged SB_STATE: {df_main['SB_STATE'].notna().sum()} rows have SB_STATE")
    
    # # # Merge PRJ_STATUS from data.projects (matching DsaAlgorithm.__init__)
    # # print("Merging PRJ_STATUS from data.projects...")
    # # # Check what column name projects uses for project code
    # # projects_df = dsa12m.data.projects.copy()

    # # projects_status_df = projects_df[['CODE', 'PRJ_STATUS']].copy().reset_index(drop=True)
    # # df_main = pd.merge(
    # #     df_main,
    # #     projects_status_df,
    # #     left_on='PRJ_CODE',
    # #     right_on='CODE',
    # #     how='left'
    # # )
    # # print(f"  Merged PRJ_STATUS: {df_main['PRJ_STATUS'].notna().sum()} rows have PRJ_STATUS")
    
    # print("--- End DSA Data Loading ---\n")
    
    # # --- Filter by project and SB status (matching DsaAlgorithm selector) ---
    # print(f"\n--- Filtering by Project and SB Status ---")
    # print(f"Eligible project statuses: {eligible_prj_status}")
    # print(f"Eligible SB statuses: {eligible_sb_status}")
    
    # # Show breakdown by status before filtering for debugging
    # if rows_before_status_filter > 0:
    #     print(f"\nStatus breakdown before filtering:")
    #     # prj_status_counts_before = df_main['PRJ_STATUS'].value_counts()
    #     # print(f"  Project statuses: {dict(prj_status_counts_before)}")
    #     sb_status_counts_before = df_main['SB_STATE'].value_counts()
    #     print(f"  SB statuses: {dict(sb_status_counts_before)}")
    
    # # Filter to only include rows with eligible statuses
    # df_main = df_main[
    #     (df_main['PRJ_STATUS'].isin(eligible_prj_status)) & 
    #     (df_main['SB_STATE'].isin(eligible_sb_status))
    # ].copy()
    
    # rows_after_status_filter = len(df_main)
    # print(f"\nStatus filtering: {rows_before_status_filter} -> {rows_after_status_filter} rows (removed {rows_before_status_filter - rows_after_status_filter})")
    
    # # Show breakdown by status after filtering for debugging
    # if rows_after_status_filter > 0:
    #     print(f"\nStatus breakdown after filtering:")
    #     prj_status_counts_after = df_main['PRJ_STATUS'].value_counts()
    #     print(f"  Project statuses: {dict(prj_status_counts_after)}")
    #     sb_status_counts_after = df_main['SB_STATE'].value_counts()
    #     print(f"  SB statuses: {dict(sb_status_counts_after)}")
    # print("--- End Status Filtering ---\n")
    
    
    # Also, if any row has missing values for FRACTION_OBS_ columns, set those columns to 0.25
    # Handle both old format (FRACTION_OBS_C1, etc.) and new format (FRACTION_OBS_C1A, etc.)
    fraction_obs_cols = [col for col in df_main.columns if 'FRACTION_OBS_' in col and len(col) > len('FRACTION_OBS_')]
    for col in fraction_obs_cols:
        if col in df_main.columns:
            df_main[col] = df_main[col].fillna(0.25)

    # %% keep only projects from the year in df_main and df_simulation
    # if the project starts with 'year' keep in df_main
    df_main = df_main.reset_index()  # reset index
    df_main = df_main.drop('index', axis=1)  # drop index column
    df_simulation = df_simulation.reset_index()  # reset index
    df_simulation = df_simulation.drop('index', axis=1)  # drop index column

    print("Checking which projects need to be less than 45 hours")
    # %% list of projects that can not exceed 45 hrs
    proj_max_45 = []  # create empty list for projects
    for row in range(len(df_modes['CODE'])):  # for each row in df_modes
        if df_modes['MODE_NAME'][row] in ['BandToBand Interferometry',
                                          'BandwidthSwitching Interferometry']:  # if the mode of the projects match with the right options
            if df_modes['CODE'][row] not in proj_max_45:  # if the project code is not already on the list
                proj_max_45.append(df_modes['CODE'][row])  # append to the list the project code

    print("Checking which projects need to be less than 50 hours")
    # %% list of project '.P' that can not exceed 50 hrs
    proj_P_max_50 = []  # create empty list for projects
    for row in range(len(df_main['PRJ_CODE'])):  # for each row in df_main
        name = df_main['PRJ_CODE'][row]  # save the code of the project
        if name[
           -2:] == '.P' and name not in proj_P_max_50:  # if the last character of the code match with .P, and is not already on the list
            proj_P_max_50.append(df_main['PRJ_CODE'][row])  # append to the list the project code

    # %% Drop columns that are not needed and may have NA values
    columns_to_drop = ['OPTIMAL_PWV', 'EXTENDED', 'COMPACT']
    columns_existing = [col for col in columns_to_drop if col in df_main.columns]
    if columns_existing:
        print(f"Dropping columns before dropna: {columns_existing}")
        df_main = df_main.drop(columns=columns_existing)
        print(f"After dropping columns, df_main shape: {df_main.shape}")
    
    # %% ignore rows with no information in fraction_obs_cx
    print(f"Before dropna, df_main shape: {df_main.shape}")
    rows_before_dropna = len(df_main)
    
    # Check NA values before dropping
    if rows_before_dropna > 0:
        na_counts_before = df_main.isna().sum()
        cols_with_na = na_counts_before[na_counts_before > 0]
        if len(cols_with_na) > 0:
            print(f"Columns with NA values before dropna: {dict(cols_with_na)}")
    
    df_main = df_main.dropna()  # drop rows with na values
    rows_after_dropna = len(df_main)
    print(f"After dropna, df_main shape: {df_main.shape} (dropped {rows_before_dropna - rows_after_dropna} rows)")
    
    if len(df_main) == 0:
        print("ERROR: df_main is empty after dropna()!")
        print("All rows were dropped. Check the NA counts above to see which columns caused this.")
    
    df_main = df_main.reset_index()  # reset index of df_main
    print(f"After reset_index, df_main shape: {df_main.shape}")


    # %% function for dictionaries
    # function that multiplies a dictionary value for a number
    def multiply_dict_values(dictionary, mult):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                multiply_dict_values(value, mult)
            else:
                dictionary[key] = value * mult

    # function that sums up all values of a key in a dictionary
    def sum_dict_by_key(new_dict):
        res = dict()
        for sub in new_dict.values():
            for key, ele in sub.items():
                res[key] = ele + res.get(key, 0)
        return res

    # function that approximates values of a dictionary
    def approximate_dict_values(dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                approximate_dict_values(value)
            else:
                dictionary[key] = math.floor(value)

    # %% keep in df_main only rows from the top 50 projects or accepted projects
    # if problem == 'plan':  # if the problem is planning problem
    #     grade_prj = df_cycle_accepted.groupby('PRJ_CODE')[
    #         'GRADE'].first().to_dict()  # create dict with projects as keys and the grade as values
    #     cycle_accepted_prj_list = df_cycle_accepted[
    #         'PRJ_CODE'].tolist()  # create list of project codes that are accepted for the cycle
    #     df_main = df_main.loc[df_main['PRJ_CODE'].isin(
    #         cycle_accepted_prj_list)]  # keep rows that belong to code of projects that are accepted
    #     df_main['PRJ_GRADE'] = ''  # create empty column to save grades in df_main
    #     for idx, row in df_main.iterrows():  # for each row of df_main
    #         p = row['PRJ_CODE']  # save the project code
    #         df_main['PRJ_GRADE'][idx] = grade_prj[p]  # save in the right row the grade of the project
    # else:  # if the problem is not planning
    #     df_main = df_main.loc[df_main['PRJ_CODE'].isin(top50)]  # keep rows of only top 50 projects according to ranking

    # %% add index for each SB
    indice = list(range(1, len(df_main) + 1))  # create a list of numbers according to the lenght of the df_main
    df_main.insert(0, 'eb', indice)  # add eb column to df_main to identify each eb with just a number
    # print('columns eb added')

    # %% duplicate rows to create as many variable as execution may be needed
    # duplicate rows based on the 'NUMBER_OF_EXECUTIONS' column
    
    job_lookup = {j['job_id']: j for j in jobs}

    # Debug: Print mapping statistics
    print(f"\n--- SB_UID Mapping Debug ---")
    print(f"Total SBs in df_main: {len(df_main)}")
    print(f"Total jobs in job_lookup: {len(job_lookup)}")
    print(f"Total entries in sb_map: {len(sb_map)}")
    
    # Count how many SBs are mapped vs unmapped
    mapped_count = 0
    unmapped_count = 0
    total_remaining_execs = 0
    sbs_with_execs = 0
    
    for index, row in df_main.iterrows():
        sb_id = row['SB_UID']
        if sb_id not in sb_map:
            df_main.loc[index, 'NUMBER_OF_EXECUTIONS'] = 0
            unmapped_count += 1
        else:
            jobs_id = sb_map[sb_id]
            if jobs_id not in job_lookup:
                print(f"WARNING: Mapped job_id '{jobs_id}' not found in job_lookup for SB_UID '{sb_id}'")
                df_main.loc[index, 'NUMBER_OF_EXECUTIONS'] = 0
                unmapped_count += 1
            else:
                remaining = job_lookup[jobs_id]['remaining_execs']
                df_main.loc[index, 'NUMBER_OF_EXECUTIONS'] = remaining
                total_remaining_execs += remaining
                if remaining > 0:
                    sbs_with_execs += 1
                mapped_count += 1
    
    print(f"SBs successfully mapped: {mapped_count}")
    print(f"SBs not in sb_map: {unmapped_count}")
    print(f"SBs with remaining_execs > 0: {sbs_with_execs}")
    print(f"Total remaining executions: {total_remaining_execs}")
    
    # === DEBUG: Check for jobs in operational scheduler but NOT in df_main ===
    print("\n" + "=" * 80)
    print("DEBUG: JOBS IN OPERATIONAL SCHEDULER BUT NOT IN df_main")
    print("=" * 80)
    df_main_sbuids = set(df_main['SB_UID'].values)
    sb_map_sbuids = set(sb_map.keys())
    job_lookup_job_ids = set(job_lookup.keys())
    
    # SB_UIDs that are in sb_map but not in df_main
    in_sbmap_not_in_dfmain = sb_map_sbuids - df_main_sbuids
    print(f"SB_UIDs in sb_map but NOT in df_main: {len(in_sbmap_not_in_dfmain)}")
    if len(in_sbmap_not_in_dfmain) > 0:
        sample_missing = list(in_sbmap_not_in_dfmain)
        print(f"  Sample: {sample_missing}")
    
    print("=" * 80 + "\n")
    
    print("--- End SB_UID Mapping Debug ---\n")

    ## NOEMI TK look through df_main. sb id: ... num executions. modify df_main directly: replace with remaining executions.
    ## map_sbs_by_time_distance in full_year.py maps the sbs from the two different data frames



    duplicated_rows = df_main.loc[df_main.index.repeat(df_main['NUMBER_OF_EXECUTIONS'])]
    # modify the 'eb' column in the duplicated rows
    duplicated_rows['eb'] = duplicated_rows['eb'].astype(str) + '.' + (
            duplicated_rows.groupby(level=0).cumcount() + 1).astype(str)
    # reset the index of the duplicated rows
    duplicated_rows.reset_index(drop=True, inplace=True)
    # concatenate the original DataFrame with the duplicated rows
    df_main = pd.concat([df_main, duplicated_rows], ignore_index=True)

    # %% drop rows of eb that are duplicated
    df_main = df_main.loc[~df_main['eb'].apply(lambda x: isinstance(x, int))]  # ignore rows that are duplicated
    df_main.set_index('eb', inplace=True)  # set eb column as index

    # %% create a dataframe with matching eb and SB_UID and save it in a csv file
    df_match_eb_sbuid = df_main.reset_index()  # duplicate df_main and reset index
    df_match_eb_sbuid = df_match_eb_sbuid[
        ['eb', 'SB_UID', 'PRJ_CODE']]  # keep only needed columns with information about project, eb and sb_uid
    # Save to output_path if provided, otherwise current directory
    if output_path:
        match_eb_path = os.path.join(output_path, 'match_eb_sbuid.csv')
    else:
        match_eb_path = 'match_eb_sbuid.csv'
    df_match_eb_sbuid.to_csv(match_eb_path, index=False)  # export file with matching information
    print(f"Saved match_eb_sbuid.csv to: {match_eb_path}")

    eb_to_grade = df_main['PRJ_GRADE'].to_dict()
    project_to_grade = df_main.drop_duplicates('PRJ_CODE').set_index('PRJ_CODE')['PRJ_GRADE'].to_dict()

    # %% create calendar with start and end dates
    calendar = pd.DataFrame(columns=['Configuration', 'Start', 'End'])  # create df to save calendar
    begin_calendar = pd.DataFrame(
        {'Configuration': df_availability['conf'][0], 'Start': df_availability['timestamp'][0]},
        index=[0])  # create first row of calendar
    calendar = pd.concat([begin_calendar, calendar.loc[:]])  # append first row to calendar

    count = 0  # create count variable
    actual_c = calendar['Configuration'][0]  # set first configuration to the actual one
    for row in range(len(df_availability)):  # for each row of df_availability
        if df_availability['conf'][row] != actual_c:  # if the configuration is different than the actual one
            count += 1  # add one to the count variable
            add_row_calendar = pd.DataFrame(
                {'Configuration': df_availability['conf'][row], 'Start': df_availability['timestamp'][row]},
                index=[0])  # create following row of calendar
            calendar = pd.concat([add_row_calendar, calendar.loc[:]]).reset_index(drop=True)  # append row to calendar
            actual_c = df_availability['conf'][row]  # update the actual configuration
    # calendar['Start'] = calendar['Start'].apply(lambda a: pd.to_datetime(a).date()) #set start date in datetime format
    calendar['Start'] = pd.to_datetime(calendar['Start']).apply(lambda a: pd.to_datetime(a).date())
    calendar['End'] = pd.to_datetime(calendar['End']).apply(lambda a: pd.to_datetime(a).date())
    calendar['Start'] = pd.to_datetime(calendar['Start'], utc=True)
    calendar['End'] = pd.to_datetime(calendar['End'], utc=True)

    calendar = calendar.sort_values(by=['Start']).reset_index()  # sort calendar according to start date
    calendar = calendar.drop(['index'], axis=1)  # drop index column
    for row in range(1, len(calendar['Start'])):  # for the second row till the end of the calendar
        calendar['End'][row - 1] = calendar['Start'][row] - timedelta(
            days=1)  # set end date by subtracting a day to start date of the following configuration
    # Set end date for the last configuration using last row of df_availability (keep UTC timezone)
    last_timestamp = df_availability['timestamp'].iloc[-1]
    calendar['End'].iloc[-1] = pd.to_datetime(last_timestamp.date(), utc=True)
    
    # Ensure all calendar dates are timezone-aware after timedelta operations
    calendar['Start'] = pd.to_datetime(calendar['Start'], utc=True)
    calendar['End'] = pd.to_datetime(calendar['End'], utc=True)
    
    print('Calendar created')
    
    # === DEBUG: Print the calendar that was created ===
    print(f"\n--- Calendar built from filtered df_availability ---")
    print(f"  Number of configurations in calendar: {len(calendar)}")
    for idx, row in calendar.iterrows():
        print(f"    {idx}: {row['Configuration']} from {row['Start']} to {row['End']}")
    print(f"--- End calendar debug ---\n")
    
    # %% add column with just day to df_availability
    # Normalize timestamps to just dates but keep timezone-aware (UTC)
    df_availability['day'] = df_availability['timestamp'].dt.normalize()

    # %% create config array
    lista_config = calendar['Configuration'].str.split('-', n=1, expand=True)[
        1]  # get from calendar just the number of the configuration
    config = lista_config.to_list()  # create config list
    # print(config)
    duplicados = {x for x in config if config.count(x) > 1}  # get set of repeated configurations

    # use a defaultdict to keep track of element counts
    element_counts = defaultdict(int)
    new_config = []  # create new list to save configurations
    for element in config:  # for each configuration of the original list
        element_counts[element] += 1  # increment the count of the current configuration
        letter = chr(ord('A') + element_counts[element] - 1)  # determine the letter to add based on the count
        new_config.append(element + letter)  # append configuration with letter to the new list
    # print(new_config)
    config = new_config  # replace with new list
    
    # === DEBUG: Print the final config list ===
    print(f"\n--- Final config list (used for optimization) ---")
    print(f"  Configurations: {config}")
    print(f"  Total: {len(config)} configurations")
    print(f"--- End config list debug ---\n")
    # %% dict conf_duration
    durations = []  # create empty array
    for c in range(len(calendar['Start'])):  # for each row in calendar
        durations.append(0)  # append a 0 to each place in durations list
        for row in range(len(df_availability)):  # for each row in df_availability
            if bool((calendar['Start'][c] - timedelta(days=1)) < df_availability['day'][row] and (
                    df_availability['day'][row] < (calendar['End'][c] + timedelta(days=1)))):
                # if the day of df_availability is between the start and end date of the configuration in calendar
                durations[c] += df_availability['available_time'][row]  # add available time to the duration
                # durations[c] += 0.5
    conf_duration = {config[i]: durations[i] for i in range(
        len(config))}  # create a dictionary with configurations as keys and durations as values (in hours)
    print('dict conf_duration')
    
    # Print estimated time per configuration for debugging
    print("\n--- Estimated Available Time per Configuration (from expected_times file) ---")
    for conf_name, duration_hours in conf_duration.items():
        duration_bins = duration_hours * 2
        print(f"  {conf_name}: {duration_hours:.2f} hours ({duration_bins:.0f} bins)")
    print(f"  Total across all configurations: {sum(conf_duration.values()):.2f} hours ({sum(conf_duration.values()) * 2:.0f} bins)")
    print("--- End Estimated Time per Configuration ---\n")

    # %% dict conf_duration_bin
    conf_duration_bin = {key: value * 2 for key, value in
                         conf_duration.items()}  # create dictionary with configurations and durations with bins
    print('dict conf_duration_bin')

    # %% add columns to calendar
    calendar['Config'] = config  # add config column to calendar
    calendar['Days'] = (calendar['End'] - calendar[
        'Start']).dt.days + 1  # get amount of days for each configuration into a column
    calendar['Durations'] = list(conf_duration.values())  # get durations in hours into a column
    calendar['Durations_bins'] = list(conf_duration_bin.values())  # get durations in bins into a column
    
    for i, row in calendar.iterrows():
        if row['Config'] == '4a' :
            row['Days'] -= 28
    


    # calendar['Days'][5] -= 28  # subtract days of feb of configuration

    ## TK NOEMI go through and find calendar start and calendar end and check if it has february in it
    ## figure out how many days it overlaps with february and subtract that specific amount. an easier way 
    ## to do this is just check if we are in configuration 4a - do this if i get confused.
    
    # TODO: make it depend on the year, since sometimes there is a leap day.
    print(calendar)
    # %% assign value to category in a new column (inversa de ranking)
    df_main['SB_Value'] = 1 / df_main['PRJ_SCIENTIFIC_RANK']  # create sb_value column
    # if you want to add extra points to the profit equation, you can add it in the 'SB_Value' column
    # for idx, row in df_main.iterrows(): #for each row
    #     if row['PRJ_GRADE'] == 'A': #if the project is A graded
    #         df_main['SB_Value'][idx] += 1000 #add 1000 points
    #     elif row['PRJ_GRADE'] == 'B': if the project is B graded
    #         df_main['SB_Value'][idx] += 10 #add 1000 points

    # %% dict eb_duration
    eb_duration = {}  # create empty dictionary to save the hours that each EB needs
    for idx, row in df_main.iterrows():  # for each row in df_main
        eb_duration[idx] = (row[
            'SB_TIME_BY_EXECUTION'])  # save in the dictionary with EB as keys and duration in hours of the execution as values
    print('dict eb_duration')

    # %% dict bins_eb
    # Use job lengths from the operational scheduler (passed in via 'jobs' parameter) for consistency
    # This ensures both schedulers use the same execution time estimates
    job_length_map = {job['job_id']: job.get('length', 0) for job in jobs}
    
    bins_eb = {}  # create empty dictionary to save how many bins each EB needs
    jobs_with_operational_length = 0
    jobs_with_fallback_length = 0
    
    for idx, row in df_main.iterrows():  # for each row in df_main
        sb_uid = row['SB_UID']
        if sb_uid in job_length_map and job_length_map[sb_uid] > 0:
            # Use the length from the operational scheduler
            bins_eb[idx] = job_length_map[sb_uid]
            jobs_with_operational_length += 1
        else:
            # Fallback to the file-based calculation if job not found
            cant_bins = (row['SB_TIME_BY_EXECUTION']) * 2  # get amount of bins that each EB lasts
            bins_eb[idx] = round(cant_bins)
            jobs_with_fallback_length += 1
    
    print(f'dict bins_eb: {jobs_with_operational_length} jobs using operational lengths, {jobs_with_fallback_length} using fallback')

    # %% dict projects_eb
    code_count = df_main['PRJ_CODE'].nunique()  # get amount of projects
    unique_code = df_main['PRJ_CODE'].unique()  # get list of project codes
    projects_eb = {}  # create empty dictionary to save which EBs belong to each project
    grp_proj = df_main.groupby('PRJ_CODE')['PRJ_CODE']

    for i in tqdm.tqdm(list(range(code_count))):  # for each project
        projects_eb[unique_code[i]] = grp_proj.get_group(unique_code[i]).index.to_list()

    print('dict projects_eb')

    # %% get a list of the names of FRACTION_OBS_columns
    # extract columns with 'FRACTION_OBS_'
    # Handle both old format (FRACTION_OBS_C1, FRACTION_OBS_C2, etc.) and new format (FRACTION_OBS_C1A, FRACTION_OBS_C2B, etc.)
    cols = [col for col in df_main.columns if 'FRACTION_OBS_' in col and len(col) > len('FRACTION_OBS_')]
    if not suppress_fraction_obs_debug:
        print(f"Found FRACTION_OBS columns: {cols}")
    
    # Parse columns to extract configuration number and instance letter
    # Format: FRACTION_OBS_C{number}{letter} or FRACTION_OBS_C{number}
    fraction_cols_map = {}  # Maps (config_num, instance_letter) to column name
    for col in cols:
        # Extract configuration number and optional instance letter
        # Pattern: FRACTION_OBS_C followed by digits, optionally followed by A, B, or C
        match = re.match(r'FRACTION_OBS_C(\d+)([ABC]?)', col)
        if match:
            config_num = match.group(1)
            instance_letter = match.group(2) if match.group(2) else ''  # Empty string if no letter
            fraction_cols_map[(config_num, instance_letter)] = col
            if not suppress_fraction_obs_debug:
                print(f"  Mapped {col} -> Config {config_num}, Instance '{instance_letter}'")
    
    # Also create a fallback map for old format (without instance letters)
    old_format_map = {}
    for col in cols:
        match = re.match(r'FRACTION_OBS_(\d+)$', col)  # Only match if no letter suffix
        if match:
            config_num = match.group(1)
            old_format_map[config_num] = col
    # %% find max profit (pmax)

    max_value = 0  # set max_value as 0
    preferencias = df_main[cols]  # create df with just fraction obs columns
    preferencias = preferencias.dropna()  # drop na values
    if 3 in element_counts.values():  # if any configurations is scheduled 3 times in a cycle
        valor_extra_por_config = 0.002  # the extra value is 0.002
    else:  # if a configuration is just scheduled at most twice
        valor_extra_por_config = 0.001  # the extra value is 0.001
    for idx, row in preferencias.iterrows():  # for each row in 'preferencias'
        # Extract configuration number from column name (handles both old and new formats)
        max_col = pd.to_numeric(preferencias.loc[idx]).idxmax()
        # Parse column name to get config number (handles FRACTION_OBS_C1, FRACTION_OBS_C2A, etc.)
        match = re.match(r'FRACTION_OBS_C(\d+)', max_col)
        if match:
            indice = match.group(1)  # Get just the configuration number
        else:
            # Fallback to old parsing method
            indice = max_col.split('_')[2].split('C')[1] if len(max_col.split('_')) > 2 else max_col.split('C')[1]
        valor = preferencias.loc[idx].max()  # get the value of the configuration selected previously
        nota = df_main['SB_Value'][idx]  # get value of EB from df_main
        if indice in duplicados:  # if the configuration is scheduled more than once in the cycle
            valor += valor_extra_por_config  # the value gets an "extra"
        max_value += nota * valor  # add to max_value the points of executing the EB in the best option available (the biggest preference * value of eb)
    print(max_value)
    print('max_value')

    # %% create bins
    bins_as_string = np.arange(0, 24., 24. / (24 * 60. / 30.)).astype(
        str)  # create array of bins representing each time slot of the day as strings
    bins = np.arange(0, 24.5, 24. / (24 * 60. / 30.))  # create array of bins as numbers

    # %% merge dataframes to have number of sb in df_simulation
    df_sb_index = df_main[['SB_UID']]  # create df with just sb_uid column
    df_sb_index['eb'] = df_main.index  # add eb column
    df_disponibilidad = df_simulation.copy()  # copy df_simulation
    
    # Check if df_simulation has lst column, if not we'll need to load it from pressure file
    has_lst = 'lst' in df_disponibilidad.columns
    
    df_sb_index.set_index('SB_UID', inplace=True)  # SB_UID as index in df_sb_index

    def extract_date(df):
        df['Day'] = df['Date'].str.split('T', n=1, expand=True)[0]  # get just date in a new column
        df['Day'] = pd.to_datetime(df['Day']).dt.date  # set datetimedate format to day column
        return df

    def date_in_range(start_date, end_date, df_row):
        return start_date < df_row['Day'] < end_date

    def update_df_disponibilidad(df_disponibilidad, calendar, config):
        df_disponibilidad = extract_date(df_disponibilidad)

        # Use a copy of calendar to avoid modifying the original (which has UTC timezone)
        cal_copy = calendar.copy()
        cal_copy['Start'] = pd.to_datetime(cal_copy['Start']).dt.date
        cal_copy['End'] = pd.to_datetime(cal_copy['End']).dt.date

        config_avail_list = []
        for index, calendar_row in cal_copy.iterrows():  # for each row in calendar
            start_date = calendar_row['Start']
            end_date = calendar_row['End'] + pd.Timedelta(days=1)
            day_count = (end_date - start_date).days
            for single_date in (start_date + timedelta(n) for n in range(day_count)):
                config_avail_list.append([single_date, str(config[index])])
        config_avail_df = pd.DataFrame(config_avail_list)
        config_avail_df.columns = ['Day', 'array']
        df_disponibilidad = df_disponibilidad.merge(config_avail_df, on="Day", how="left")

        return df_disponibilidad

    df_disponibilidad = update_df_disponibilidad(df_disponibilidad, calendar, config)
    #
    print("Finished with update_df_disponibilidad")
    #
    # #%% sort df_disponibilidad
    df_disponibilidad = df_disponibilidad.sort_values(by=['SB_UID'])  # sort values of df_disponibilidad by SB_UID code
    df_disponibilidad = df_disponibilidad.dropna()  # drop na rows
    df_disponibilidad = df_disponibilidad.reset_index()  # reset index
    
    # Filter df_disponibilidad to only include configurations in the calendar (based on date range)
    # Note: config is derived from calendar which was already filtered by start_date and end_date
    valid_configs_set = set(config)
    print(f"Filtering df_disponibilidad to only include configurations: {sorted(valid_configs_set)}")
    rows_before = len(df_disponibilidad)
    df_disponibilidad = df_disponibilidad[df_disponibilidad['array'].isin(valid_configs_set)].reset_index(drop=True)
    print(f"Filtered df_disponibilidad: {rows_before} -> {len(df_disponibilidad)} rows (removed {rows_before - len(df_disponibilidad)})")

    # %%
    sb_uid_list_main = df_main['SB_UID'].unique().tolist()  # get list of SB_UIDs of df_main
    sb_uid_list_disp = df_disponibilidad['SB_UID'].unique().tolist()  # get list of SB_UIDs of df_disponibilidad
    # %%
    df_disponibilidad_filtered = df_disponibilidad[
        df_disponibilidad['SB_UID'].isin(
            sb_uid_list_main)]  # filter rows of df_disponibilidad if the SB_UID is in df_main
    df_disponibilidad_filtered.reset_index(inplace=True)  # reset index

    # --- Load lst from pressure file and merge into df_disponibilidad_filtered ---
    if not has_lst:
        pressure_file = os.path.join(data_dir, 'sb_12m_pressure.csv')
        print(f"Loading lst from: {pressure_file}")
        
        # Load the pressure file to get Date -> lst mapping
        pressure_df = pd.read_csv(pressure_file, index_col=False)
        pressure_df['Date'] = pd.to_datetime(pressure_df['Date'], utc=True)
 
        # Extract Date and lst columns, deduplicate
        lst_mapping = pressure_df[['Date', 'lst']].drop_duplicates(subset=['Date'])
        # If there are still duplicates (same Date, different lst), take the first one
        lst_mapping = lst_mapping.drop_duplicates(subset=['Date'], keep='first')
        
        print(f"Loaded {len(lst_mapping)} unique Date->lst mappings from pressure file")
        
        # Ensure df_disponibilidad_filtered Date is datetime for merging
        # Convert Date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df_disponibilidad_filtered['Date']):
            df_disponibilidad_filtered['Date'] = pd.to_datetime(df_disponibilidad_filtered['Date'], utc=True)
        
        # Merge lst based on Date
        df_disponibilidad_filtered = df_disponibilidad_filtered.merge(
            lst_mapping[['Date', 'lst']],
            on='Date',
            how='left'
        )
        
        # Check how many rows got lst values
        lst_filled = df_disponibilidad_filtered['lst'].notna().sum()
        print(f"Merged lst: {lst_filled} out of {len(df_disponibilidad_filtered)} rows have lst values")
        
        if lst_filled < len(df_disponibilidad_filtered):
            missing_count = len(df_disponibilidad_filtered) - lst_filled
            print(f"Warning: {missing_count} rows do not have lst values after merge")
    else:
        print("lst column already present in df_disponibilidad_filtered")

    print("Creating eb_temp")

    # %% create eb_temp
    # Check for NaN lst values before binning
    nan_lst_before = df_disponibilidad_filtered['lst'].isna().sum()
    if nan_lst_before > 0:
        print(f"WARNING: Found {nan_lst_before} rows with NaN 'lst' values before binning")
    
    df_disponibilidad_filtered['lst_bin'] = pd.cut(df_disponibilidad_filtered['lst'], bins,
                                                   labels=bins_as_string).astype(
        float) * 2
    
    # Check for NaN lst_bin values after binning
    nan_lstbin_after = df_disponibilidad_filtered['lst_bin'].isna().sum()
    if nan_lstbin_after > 0:
        print(f"WARNING: Found {nan_lstbin_after} rows with NaN 'lst_bin' after binning")
        if nan_lstbin_after != nan_lst_before:
            print(f"  This suggests some 'lst' values are outside the bin range [0, 24.5)")
            # Show sample of problematic rows
            problem_rows = df_disponibilidad_filtered[df_disponibilidad_filtered['lst_bin'].isna()][['SB_UID', 'Date', 'lst', 'array']].head(5)
            print(f"  Sample rows with NaN lst_bin:")
            print(problem_rows.to_string(index=False))
    
    df_eb_array_lstbin = df_disponibilidad_filtered[['SB_UID', 'array', 'lst_bin']].merge(df_sb_index, on='SB_UID',
                                                                                          how='left').drop_duplicates()

    eb_groups = df_eb_array_lstbin.groupby('eb')

    eb_temp = {}
    nan_lstbin_count = 0
    for eb_group in tqdm.tqdm(list(eb_groups)):
        eb, eb_group = eb_group
        for _, row in eb_group.iterrows():
            c, lstbin = row['array'], row['lst_bin']
            # Check for NaN lst_bin values
            if pd.isna(lstbin):
                nan_lstbin_count += 1
                # Skip NaN values - they shouldn't be used as time bins
                continue
            if eb not in eb_temp:
                eb_temp[eb] = {}
            if c not in eb_temp[eb]:
                eb_temp[eb][c] = {}
            eb_temp[eb][c][lstbin] = 1
    
    if nan_lstbin_count > 0:
        print(f"WARNING: Found {nan_lstbin_count} rows with NaN lst_bin values (these were skipped)")
        print(f"  This can happen if 'lst' is missing or outside the valid range in df_disponibilidad_filtered")

    # %% dict Tti
    # Ensure calendar dates are timezone-aware (UTC) for comparisons
    calendar['Start'] = pd.to_datetime(calendar['Start'], utc=True)
    calendar['End'] = pd.to_datetime(calendar['End'], utc=True)
    df_availability['array'] = 0  # create array column in df_availability
    for s in range(0, len(df_availability['day'])):  # for each row in df_availability
        for i in range(0, len(calendar['Start'])):  # for each row in calendar
            if bool((calendar['Start'][i] - timedelta(days=1)) < df_availability['day'][s] and (
                    df_availability['day'][s] < (calendar['End'][i] + timedelta(days=1)))):
                # if the date of df_availability is between the start and end date of the row in the calendar
                df_availability['array'][s] = config[i]  # add to row the configuration where can be executed

    # %% Calculo Tti_hr
    Tti_hr = {}  # create empty dictionary
    # Get set of valid configurations from the limited config list
    valid_configs = set(config)
    for c in config:  # for each configuration
        temp_dict = {}  # create a temporary dictionary
        for t in range(48):  # for each half of hour
            temp_dict[t] = 0  # create as many keys as bins with all values = 0
        Tti_hr[c] = temp_dict  # fill the dictionary with the temporary dict as values and configuration as keys

    for row in range(len(df_availability['array'])):  # for each row in df_availability
        c = df_availability['array'][row]  # get configuration that have the available time
        # Skip configurations not in the limited config list (in limited mode)
        if c not in valid_configs:
            continue
        t = df_availability['LST_bin'][row] * 2  # get bins that are available
        if c in Tti_hr and t in Tti_hr[c]:
            Tti_hr[c][t] += df_availability['available_time'][
                row]  # add to the value of the configuration and bin the amount of available time
    print('dict Tti_hr')
    # %% respaldo Tti_hr
    # Tti={}
    Tti = Tti_hr.copy()
    # %% dejar en bins Tti
    multiply_dict_values(Tti, (2))  # multiply values to get the available time in bins
    multiply_dict_values(Tti, (2))  # multiply again to overestimate the available time
    approximate_dict_values(Tti)  # aproximate amount of bins available to integers

    # %% print estimated time for each configuration, and the total estimated time
    print('Estimated time for each configuration, and the total estimated time:')
    for c in config:
        for t in Tti[c]:
            print(f'{c}: {t} - {Tti[c][t]}')
    print(f'Total estimated time: {sum([sum(x.values()) for x in Tti.values()])}')
    print('dict Tti en bins')

    # %%
    pref_conf = pd.DataFrame()  # create empty dataframe to save the preference of the configurations
    for c in config:  # for each configuration (e.g., "1A", "2B", "3C")
        # Extract configuration number and instance letter from config name
        # Config format is like "1A", "2B", etc. (number + letter)
        match = re.match(r'(\d+)([ABC]?)', c)
        if match:
            config_num = match.group(1)
            instance_letter = match.group(2) if match.group(2) else ''
        else:
            # Fallback: try to extract just the number
            config_num = ''.join([s for s in c if s.isdigit()])
            instance_letter = ''
        
        # Try to find the matching column
        col_name = None
        if (config_num, instance_letter) in fraction_cols_map:
            # Found exact match with instance letter
            col_name = fraction_cols_map[(config_num, instance_letter)]
        elif (config_num, '') in fraction_cols_map:
            # Found match without instance letter (old format)
            col_name = fraction_cols_map[(config_num, '')]
        elif config_num in old_format_map:
            # Fallback to old format column
            col_name = old_format_map[config_num]
        else:
            # Last resort: try to find any column with this config number
            # This handles cases where we might have C2A but need C2
            for (num, letter), col in fraction_cols_map.items():
                if num == config_num:
                    col_name = col
                    print(f"Warning: Using {col} for configuration {c} (no exact match found)")
                    break
        
        if col_name is None:
            print(f"ERROR: Could not find FRACTION_OBS column for configuration {c} (config_num={config_num}, instance={instance_letter})")
            print(f"Available columns: {list(fraction_cols_map.values())}")
            # Create a column of zeros as fallback
            pref_conf[c] = pd.Series(0.0, index=df_main.index)
        else:
            pref_conf[c] = df_main[col_name]  # create a column in the new df with the information for each configuration

    # === DEBUG: Print pref_conf statistics per configuration ===
    if not suppress_fraction_obs_debug:
        print("\n" + "=" * 80)
        print("DEBUG: FRACTION_OBS (pref_conf) STATISTICS")
        print("=" * 80)
        print(f"  pref_conf shape: {pref_conf.shape} (rows=EBs, columns=configurations)")
        for c in config:
            if c in pref_conf.columns:
                non_null = pref_conf[c].notna().sum()
                non_zero = (pref_conf[c].notna() & (pref_conf[c] > 0)).sum()
                total = len(pref_conf[c])
                print(f"  {c}: {non_zero} EBs with FRACTION_OBS > 0 (out of {total}, {non_null} non-null)")
            else:
                print(f"  {c}: NOT IN pref_conf columns!")
        print("=" * 80 + "\n")

    # %% dict sb_value (Pijt)
    # 0.015625 min value of fraction_obs...
    epsilon = 1e-10  # set tiny value to avoid symmetry problems in each bin

    sb_value = {}  # create empty dictionary to save the value of each eb (profit)
    # Get set of valid configurations from the limited config list
    valid_configs = set(config)
    
    for s in tqdm.tqdm(
            list(eb_temp.keys())):  # for each eb in the dictionary eb_temp (that tells us if can be scheduled)
        if s not in sb_value:  # if the eb is not in the dictionary
            sb_value[s] = {}  # create a key with the eb with empty dictionary inside
        for c in eb_temp[s].keys():  # for each configuration where the eb can be executed
            # Skip configurations that are not in the limited config list or not in pref_conf
            if c not in valid_configs or c not in pref_conf.columns:
                continue
            
            key_c = "".join([ele for ele in c if ele.isdigit()])  # get the number of the configuration
            # Initialize extra_val with a default value
            extra_val = 0  # default value
            if key_c in element_counts:
                if element_counts[key_c] == 3:  # if the configuration is scheduled 3 times throughout the cycle
                    extra_val = 0.002  # the extra value is 0.002
                elif element_counts[key_c] == 2:  # if is scheduled twice throughout the cycle
                    extra_val = 0.001  # the extra value is 0.001
                elif element_counts[key_c] == 1:  # if is scheduled just once
                    extra_val = 0  # there is no extra value
            
            sb_value[s][c] = {}  # create a empty dictionary for the EB in the configuration
            for t in eb_temp[s][c].keys():  # for each bin where can be the EB s in the configutaion c excecuted
                if np.isnan(pref_conf[c][s]) == False:  # if there is a value of preference in the dataframe
                    sb_value[s][c][t] = (df_main['SB_Value'][s] * pref_conf[c][s]) + extra_val + (
                            epsilon * (float(t) + 1))  # assign value of EB s in configuration c and bin t
    print('dict sb_value')

    # %% dict org_o (porcentaje correspondiente a cada organizacion)
    org_p = {}  # create empty dictionary to save % of executive balances per organizations
    org_p['CL'] = 0.1  # ExecBal of Chile
    org_p['EA'] = 0.225  # ExecBal of East Asia
    org_p['EU'] = 0.3375  # ExecBal of Europe
    org_p['NA'] = 0.3375  # ExecBal of North America

    # %% dict ex_bal (% de pertenencia del sb a cada organizacion)
    ex_bal = {}  # create empty dictionary to save % of executive balance per organization as values and EB as keys
    for s in df_sb_index['eb']:  # for each EB in df_main
        marks = {}  # create empty dictionary
        for subject in list(df_main[['CL', 'EA', 'EU', 'NA']].columns):  # for each organization
            marks[subject] = df_main[subject][s]  # get the % of executive balances
        ex_bal[s] = marks  # save the values for each EB
    print('dict ex_bal')

    # %% time per organization per sb
    tiempo_a = {}  # create empty dictionary to save information of how much time each organization could use
    for s in ex_bal.keys():  # for each EB
        tiempo_a[s] = {}  # create en empty dictionary
        for o in ex_bal[s].keys():  # for each organization
            tiempo_a[s][o] = ex_bal[s][o] * df_main['SB_TIME_BY_EXECUTION'][s]  # save the estimated time

    # Job weight (project weight for paper): w_b = w_ρb for objective normalization (Section 4.1)
    job_weight = df_main['SB_Value'].to_dict()

    return (eb_temp, projects_eb, sb_value,
     max_value, bins_eb, ex_bal,
     org_p, config, Tti, eb_to_grade, project_to_grade, calendar, pref_conf, job_weight)


## add dict as input. sb id: num times already been run

def _load_match_eb_mapping(output_dir: str, data_dir: str) -> pd.DataFrame:
    """Load the EB-to-SBUID mapping file written during preprocessing."""
    match_file_path = None
    if output_dir:
        candidate = os.path.join(output_dir, 'match_eb_sbuid.csv')
        if os.path.exists(candidate):
            match_file_path = candidate
    if match_file_path is None:
        candidate = os.path.join(data_dir, 'match_eb_sbuid.csv')
        if os.path.exists(candidate):
            match_file_path = candidate
    if match_file_path is None and os.path.exists('match_eb_sbuid.csv'):
        match_file_path = 'match_eb_sbuid.csv'
    if match_file_path is None:
        raise FileNotFoundError(
            f"Could not find match_eb_sbuid.csv in output_dir ({output_dir}), data_dir ({data_dir}), "
            f"or current directory. Current working directory: {os.getcwd()}"
        )
    return pd.read_csv(match_file_path)


def _format_week_label(week_index: int) -> str:
    return f"{int(week_index) + 1}A"


def _replace_year_safe(timestamp: pd.Timestamp, target_year: int) -> pd.Timestamp:
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    try:
        return timestamp.replace(year=target_year)
    except ValueError:
        return timestamp.replace(year=target_year, day=28)


def _normalize_timestamp_to_cycle_years(timestamp: pd.Timestamp, cycle_start_year: int) -> pd.Timestamp:
    target_year = cycle_start_year if timestamp.month >= 10 else cycle_start_year + 1
    return _replace_year_safe(timestamp, target_year)


def _parse_dsa_sim_filename(path: str) -> tuple[int, int, int] | None:
    match = re.search(r"dsa_sim_(\d+)_(\d+)_(\d+)_df\.csv$", os.path.basename(path))
    if match is None:
        return None
    month, day, year = map(int, match.groups())
    return year, month, day


def _load_realized_weather_lookup_from_root(
        preprocessed_root: str,
        cycle_year: int,
        weather_cache: dict,
) -> dict[pd.Timestamp, tuple[float, float]]:
    if cycle_year in weather_cache:
        return weather_cache[cycle_year]

    weather_path = os.path.join(preprocessed_root, f"year_{cycle_year}", "realized_weather.pkl")
    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"Missing realized weather pickle for cycle {cycle_year}: {weather_path}")

    with open(weather_path, "rb") as handle:
        payload = pickle.load(handle)

    realized_weather = payload.get("realized_weather")
    idx_to_timestamp = payload.get("idx_to_timestamp")
    if realized_weather is None or idx_to_timestamp is None:
        raise KeyError(
            f"Expected 'realized_weather' and 'idx_to_timestamp' in realized weather pickle: {weather_path}"
        )

    lookup: dict[pd.Timestamp, tuple[float, float]] = {}
    for idx, timestamp in idx_to_timestamp.items():
        if idx not in realized_weather:
            continue
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        lookup[ts] = realized_weather[idx]

    weather_cache[cycle_year] = lookup
    return lookup


def _build_reference_lst_bin_lookup(data_dir: str, cycle_start_year: int) -> dict[pd.Timestamp, int]:
    pressure_path = os.path.join(data_dir, "sb_12m_pressure.csv")
    if not os.path.exists(pressure_path):
        raise FileNotFoundError(f"Reference LST file not found: {pressure_path}")

    df_pressure = pd.read_csv(pressure_path, usecols=lambda col: col in {"Date", "lst", "LST"})
    lst_col = "lst" if "lst" in df_pressure.columns else "LST"
    if "Date" not in df_pressure.columns or lst_col not in df_pressure.columns:
        raise ValueError(f"Reference LST file must contain Date and lst/LST columns: {pressure_path}")

    df_pressure["Date"] = pd.to_datetime(df_pressure["Date"], utc=True, errors="coerce")
    df_pressure = df_pressure[df_pressure["Date"].notna()].copy()
    df_pressure["normalized_date"] = df_pressure["Date"].apply(
        lambda ts: _normalize_timestamp_to_cycle_years(ts, cycle_start_year).floor("30min")
    )
    df_pressure["lst_bin"] = (
        pd.to_numeric(df_pressure[lst_col], errors="coerce")
        .mul(2.0)
        .round()
        .astype("Int64")
    )
    df_pressure = df_pressure[df_pressure["lst_bin"].notna()].copy()
    df_pressure["lst_bin"] = df_pressure["lst_bin"].astype(int).clip(lower=0, upper=47)
    df_pressure = df_pressure.drop_duplicates(subset=["normalized_date"], keep="first")
    return dict(zip(df_pressure["normalized_date"], df_pressure["lst_bin"]))


def _build_weekly_direct_cache_dir(
        data_dir: str,
        cycle_anchor_date: str,
        cycle_end_date: str,
        year: int,
) -> str:
    cache_key = hashlib.md5(
        f"{year}|{pd.Timestamp(cycle_anchor_date).date()}|{pd.Timestamp(cycle_end_date).date()}|weekly_direct_v2".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return os.path.join(data_dir, "weekly_direct_cache", f"year_{year}_{cache_key}")


def _week_label_sort_key(week_label: str) -> tuple[int, str]:
    week_label = str(week_label)
    numeric_part = "".join(ch for ch in week_label if ch.isdigit())
    return (int(numeric_part) if numeric_part else 10**9, week_label)


def _log_weekly_direct_suitability_summary(cache_dir: str, max_examples: int = 8) -> None:
    weekly_path = os.path.join(cache_dir, "weekly_direct_suitability.csv")
    slot_path = os.path.join(cache_dir, "weekly_slot_availability.csv")
    if not os.path.exists(weekly_path):
        print(f"Weekly direct-suitability summary unavailable; missing {weekly_path}")
        return

    weekly_direct_df = pd.read_csv(weekly_path)
    slot_availability_df = pd.read_csv(slot_path) if os.path.exists(slot_path) else pd.DataFrame()

    print("\n" + "=" * 80)
    print("WEEKLY DIRECT-SUITABILITY SUMMARY")
    print("=" * 80)

    if weekly_direct_df.empty:
        print("No nonzero direct-suitability rows were generated.")
        if not slot_availability_df.empty:
            slot_summary = (
                slot_availability_df.groupby("Week_Label", as_index=False)
                .agg(
                    open_week_bins=("slot_open_fraction", lambda s: int((s > 0).sum())),
                    total_week_bins=("lst_bin", "size"),
                    mean_open_fraction=("slot_open_fraction", "mean"),
                    max_open_fraction=("slot_open_fraction", "max"),
                )
            )
            slot_summary = slot_summary.sort_values(
                by="Week_Label",
                key=lambda s: s.map(_week_label_sort_key),
            )
            print("Slot-open summary (used for weekly capacity):")
            for _, row in slot_summary.iterrows():
                print(
                    f"  {row['Week_Label']}: open_bins={int(row['open_week_bins'])}/{int(row['total_week_bins'])}, "
                    f"mean_open_fraction={float(row['mean_open_fraction']):.4f}, "
                    f"max_open_fraction={float(row['max_open_fraction']):.4f}"
                )
        print("=" * 80 + "\n")
        return

    week_summary = (
        weekly_direct_df.groupby("Week_Label", as_index=False)
        .agg(
            nonzero_sb_lst_pairs=("SB_UID", "size"),
            nonzero_sbs=("SB_UID", "nunique"),
            mean_suitability=("suitability_fraction", "mean"),
            median_suitability=("suitability_fraction", "median"),
            max_suitability=("suitability_fraction", "max"),
        )
    )
    if not slot_availability_df.empty:
        slot_summary = (
            slot_availability_df.groupby("Week_Label", as_index=False)
            .agg(
                open_week_bins=("slot_open_fraction", lambda s: int((s > 0).sum())),
                total_week_bins=("lst_bin", "size"),
                mean_open_fraction=("slot_open_fraction", "mean"),
                max_open_fraction=("slot_open_fraction", "max"),
            )
        )
        week_summary = week_summary.merge(slot_summary, on="Week_Label", how="left")

    week_summary = week_summary.sort_values(
        by="Week_Label",
        key=lambda s: s.map(_week_label_sort_key),
    )
    print(
        f"Nonzero direct-suitability rows: {len(weekly_direct_df)} "
        f"across {weekly_direct_df['SB_UID'].nunique()} SBs and {week_summary['Week_Label'].nunique()} weeks"
    )
    for _, row in week_summary.iterrows():
        slot_open_text = ""
        if "open_week_bins" in week_summary.columns:
            slot_open_text = (
                f", open_bins={int(row['open_week_bins'])}/{int(row['total_week_bins'])}, "
                f"mean_open_fraction={float(row['mean_open_fraction']):.4f}"
            )
        print(
            f"  {row['Week_Label']}: nonzero_sbs={int(row['nonzero_sbs'])}, "
            f"nonzero_sb_lst_pairs={int(row['nonzero_sb_lst_pairs'])}, "
            f"mean={float(row['mean_suitability']):.4f}, "
            f"median={float(row['median_suitability']):.4f}, "
            f"max={float(row['max_suitability']):.4f}{slot_open_text}"
        )

    example_rows = weekly_direct_df.copy()
    example_rows["week_sort_key"] = example_rows["Week_Label"].map(lambda label: _week_label_sort_key(label)[0])
    example_rows = example_rows.sort_values(
        by=["week_sort_key", "lst_bin", "suitability_fraction", "SB_UID"],
        ascending=[True, True, False, True],
    ).head(max_examples)
    print("\nExample direct-suitability rows:")
    for _, row in example_rows.iterrows():
        print(
            f"  week={row['Week_Label']}, lst_bin={int(row['lst_bin'])}, sb={row['SB_UID']}, "
            f"suitability={float(row['suitability_fraction']):.4f} "
            f"({int(row['suitable_slots'])}/{int(row['total_slots'])})"
        )

    top_rows = weekly_direct_df.nlargest(min(max_examples, len(weekly_direct_df)), "suitability_fraction")
    print("\nHighest-suitability examples:")
    for _, row in top_rows.iterrows():
        print(
            f"  week={row['Week_Label']}, lst_bin={int(row['lst_bin'])}, sb={row['SB_UID']}, "
            f"suitability={float(row['suitability_fraction']):.4f} "
            f"({int(row['suitable_slots'])}/{int(row['total_slots'])})"
        )
    print("=" * 80 + "\n")


def _build_weekly_direct_suitability_inputs(
        data_dir: str,
        cycle_anchor_date: str,
        cycle_end_date: str,
        year: int,
        preprocessed_root: str,
) -> str:
    cache_dir = _build_weekly_direct_cache_dir(
        data_dir=data_dir,
        cycle_anchor_date=cycle_anchor_date,
        cycle_end_date=cycle_end_date,
        year=year,
    )
    expected_files = [
        os.path.join(cache_dir, f"sb12m_master_prepared_c10_{year}.csv"),
        os.path.join(cache_dir, "expected_times_c10_for_strategic.csv"),
        os.path.join(cache_dir, f"sb_12m_pressure_{year}.csv"),
        os.path.join(cache_dir, "weekly_calendar.csv"),
        os.path.join(cache_dir, "weekly_direct_suitability.csv"),
        os.path.join(cache_dir, "weekly_slot_availability.csv"),
        os.path.join(cache_dir, "weekly_direct_metadata.json"),
        os.path.join(cache_dir, "sb12m_master_with_modes.csv"),
        os.path.join(cache_dir, "accepted_projects_cycle10.csv"),
    ]
    if all(os.path.exists(path) for path in expected_files):
        print(f"Reusing cached weekly direct-suitability artifacts from {cache_dir}")
        _log_weekly_direct_suitability_summary(cache_dir)
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)

    main_path = os.path.join(data_dir, "sb12m_master_prepared_c10.csv")
    modes_path = os.path.join(data_dir, "sb12m_master_with_modes.csv")
    accepted_path = os.path.join(data_dir, "accepted_projects_cycle10.csv")
    df_main = pd.read_csv(main_path, index_col=False)

    cycle_start_ts = pd.to_datetime(cycle_anchor_date, utc=True).normalize()
    cycle_end_ts = pd.to_datetime(cycle_end_date, utc=True).normalize() + pd.Timedelta(hours=23, minutes=30)
    lst_bin_lookup = _build_reference_lst_bin_lookup(data_dir=data_dir, cycle_start_year=year)

    all_cycle_slots = pd.date_range(start=cycle_start_ts, end=cycle_end_ts, freq="30min", tz="UTC")
    all_cycle_slots_df = pd.DataFrame({"timestamp": all_cycle_slots})
    all_cycle_slots_df["lst_bin"] = all_cycle_slots_df["timestamp"].map(lst_bin_lookup)
    all_cycle_slots_df = all_cycle_slots_df[all_cycle_slots_df["lst_bin"].notna()].copy()
    all_cycle_slots_df["lst_bin"] = all_cycle_slots_df["lst_bin"].astype(int)
    all_cycle_slots_df["day"] = all_cycle_slots_df["timestamp"].dt.normalize()
    all_cycle_slots_df["week_index"] = (
        (all_cycle_slots_df["day"] - cycle_start_ts).dt.days // 7
    ).astype(int)
    all_cycle_slots_df["Week_Label"] = all_cycle_slots_df["week_index"].map(_format_week_label)
    all_cycle_slots_df["conf"] = all_cycle_slots_df["week_index"].map(lambda idx: f"Configuration-{int(idx) + 1}")

    week_calendar = (
        all_cycle_slots_df.groupby("week_index")["day"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "Start", "max": "End"})
    )
    week_calendar["Week_Label"] = week_calendar["week_index"].map(_format_week_label)
    week_calendar["Week_Start"] = week_calendar["week_index"].map(
        lambda idx: cycle_start_ts + pd.Timedelta(days=7 * int(idx))
    )

    total_slots_by_week_bin: dict[tuple[str, int], int] = defaultdict(int)
    open_slots_by_week_bin: dict[tuple[str, int], int] = defaultdict(int)
    suitable_slots_by_sb_week_bin: dict[tuple[str, str, int], int] = defaultdict(int)
    weather_cache: dict[int, dict[pd.Timestamp, tuple[float, float]]] = {}

    processed_files = 0
    skipped_current_cycle_files = 0
    dsa_pattern = os.path.join(data_dir, "dsa_sim", "dsa_sim_*_df.csv")
    for dsa_path in sorted(glob.glob(dsa_pattern)):
        date_parts = _parse_dsa_sim_filename(dsa_path)
        if date_parts is None:
            continue
        file_year, file_month, file_day = date_parts
        file_day_ts = pd.Timestamp(year=file_year, month=file_month, day=file_day, tz="UTC")
        if cycle_start_ts <= file_day_ts <= cycle_end_ts:
            skipped_current_cycle_files += 1
            continue

        source_cycle_year = file_year if file_month >= 10 else file_year - 1
        weather_lookup = _load_realized_weather_lookup_from_root(
            preprocessed_root=preprocessed_root,
            cycle_year=source_cycle_year,
            weather_cache=weather_cache,
        )

        df_dsa = pd.read_csv(dsa_path)
        required_cols = {"sbuid", "timestamp", "pwv_thresh", "rms_thresh"}
        if not required_cols.issubset(df_dsa.columns):
            continue

        normalized_day = _normalize_timestamp_to_cycle_years(file_day_ts, year).normalize()
        day_slots = pd.date_range(
            start=normalized_day,
            end=normalized_day + pd.Timedelta(hours=23, minutes=30),
            freq="30min",
            tz="UTC",
        )
        for slot_ts in day_slots:
            lst_bin = lst_bin_lookup.get(slot_ts)
            if lst_bin is None:
                continue
            week_index = int(((slot_ts.normalize() - cycle_start_ts).days) // 7)
            week_label = _format_week_label(week_index)
            total_slots_by_week_bin[(week_label, int(lst_bin))] += 1

        pairs_df = df_dsa[["sbuid", "timestamp", "pwv_thresh", "rms_thresh"]].copy()
        pairs_df["timestamp_dt"] = pd.to_datetime(pairs_df["timestamp"], utc=True, errors="coerce").dt.floor("30min")
        pairs_df = pairs_df[pairs_df["timestamp_dt"].notna()].copy()
        if pairs_df.empty:
            continue

        pairs_df["normalized_slot"] = pairs_df["timestamp_dt"].apply(
            lambda ts: _normalize_timestamp_to_cycle_years(ts, year).floor("30min")
        )
        pairs_df = pairs_df[
            (pairs_df["normalized_slot"] >= cycle_start_ts) & (pairs_df["normalized_slot"] <= cycle_end_ts)
        ].copy()
        if pairs_df.empty:
            continue

        pairs_df["lst_bin"] = pairs_df["normalized_slot"].map(lst_bin_lookup)
        pairs_df = pairs_df[pairs_df["lst_bin"].notna()].copy()
        if pairs_df.empty:
            continue
        pairs_df["lst_bin"] = pairs_df["lst_bin"].astype(int)
        pairs_df["day"] = pairs_df["normalized_slot"].dt.normalize()
        pairs_df["week_index"] = ((pairs_df["day"] - cycle_start_ts).dt.days // 7).astype(int)
        pairs_df["Week_Label"] = pairs_df["week_index"].map(_format_week_label)

        pairs_df["realized_weather"] = pairs_df["timestamp_dt"].map(weather_lookup)
        pairs_df["realized_pwv"] = pairs_df["realized_weather"].apply(
            lambda value: value[0] if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2 else np.nan
        )
        pairs_df["realized_rms"] = pairs_df["realized_weather"].apply(
            lambda value: value[1] if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 2 else np.nan
        )
        pairs_df["weather_suitable"] = (
            (pairs_df["realized_pwv"] <= pairs_df["pwv_thresh"]) &
            (pairs_df["realized_rms"] >= pairs_df["rms_thresh"]) &
            pairs_df["realized_pwv"].notna() &
            pairs_df["realized_rms"].notna() &
            pairs_df["pwv_thresh"].notna() &
            pairs_df["rms_thresh"].notna()
        )

        slot_any = (
            pairs_df.groupby(["normalized_slot", "Week_Label", "lst_bin"], as_index=False)["weather_suitable"]
            .any()
        )
        slot_any = slot_any[slot_any["weather_suitable"]].copy()
        for _, row in slot_any.iterrows():
            open_slots_by_week_bin[(str(row["Week_Label"]), int(row["lst_bin"]))] += 1

        suitable_counts = (
            pairs_df[pairs_df["weather_suitable"]]
            .drop_duplicates(subset=["sbuid", "normalized_slot"])
            .groupby(["sbuid", "Week_Label", "lst_bin"])
            .size()
            .reset_index(name="suitable_slots")
        )
        for _, row in suitable_counts.iterrows():
            suitable_slots_by_sb_week_bin[
                (str(row["sbuid"]), str(row["Week_Label"]), int(row["lst_bin"]))
            ] += int(row["suitable_slots"])

        processed_files += 1

    slot_rows = []
    for (week_label, lst_bin), total_slots in sorted(total_slots_by_week_bin.items()):
        open_slots = open_slots_by_week_bin.get((week_label, lst_bin), 0)
        slot_rows.append({
            "Week_Label": week_label,
            "lst_bin": int(lst_bin),
            "total_slots": int(total_slots),
            "open_slots": int(open_slots),
            "slot_open_fraction": float(open_slots / total_slots) if total_slots > 0 else 0.0,
        })
    slot_availability_df = pd.DataFrame(slot_rows)
    slot_fraction_lookup = {
        (str(row["Week_Label"]), int(row["lst_bin"])): float(row["slot_open_fraction"])
        for _, row in slot_availability_df.iterrows()
    }

    suitability_rows = []
    for (sb_uid, week_label, lst_bin), suitable_slots in sorted(suitable_slots_by_sb_week_bin.items()):
        total_slots = total_slots_by_week_bin.get((week_label, lst_bin), 0)
        if total_slots <= 0:
            continue
        suitability_rows.append({
            "SB_UID": sb_uid,
            "Week_Label": week_label,
            "lst_bin": int(lst_bin),
            "suitable_slots": int(suitable_slots),
            "total_slots": int(total_slots),
            "suitability_fraction": float(suitable_slots / total_slots),
        })
    weekly_direct_df = pd.DataFrame(suitability_rows)

    synthetic_main = df_main.copy()
    for week_label in week_calendar["Week_Label"].tolist():
        week_num = int(str(week_label).rstrip("A"))
        synthetic_main[f"FRACTION_OBS_C{week_num}A"] = 0.0

    synthetic_availability = all_cycle_slots_df[["timestamp", "conf", "lst_bin"]].copy()
    synthetic_availability["available_time"] = synthetic_availability.apply(
        lambda row: slot_fraction_lookup.get((str(row["conf"]).replace("Configuration-", "") + "A", int(row["lst_bin"])), 0.0) / 4.0,
        axis=1,
    )
    synthetic_availability["LST_bin"] = synthetic_availability["lst_bin"] / 2.0
    synthetic_availability = synthetic_availability[["timestamp", "conf", "available_time", "LST_bin"]]

    week_start_lookup = {
        str(row["Week_Label"]): pd.Timestamp(row["Week_Start"]).tz_localize("UTC")
        if pd.Timestamp(row["Week_Start"]).tzinfo is None
        else pd.Timestamp(row["Week_Start"]).tz_convert("UTC")
        for _, row in week_calendar.iterrows()
    }
    if weekly_direct_df.empty:
        synthetic_simulation = pd.DataFrame(columns=["SB_UID", "Date", "lst"])
    else:
        synthetic_simulation = weekly_direct_df[["SB_UID", "Week_Label", "lst_bin", "suitability_fraction"]].copy()
        synthetic_simulation = synthetic_simulation[synthetic_simulation["suitability_fraction"] > 0].copy()
        synthetic_simulation["Date"] = synthetic_simulation.apply(
            lambda row: week_start_lookup[str(row["Week_Label"])] + pd.Timedelta(minutes=30 * int(row["lst_bin"])),
            axis=1,
        )
        synthetic_simulation["lst"] = synthetic_simulation["lst_bin"] / 2.0
        synthetic_simulation = synthetic_simulation[["SB_UID", "Date", "lst"]].drop_duplicates()

    synthetic_main.to_csv(os.path.join(cache_dir, f"sb12m_master_prepared_c10_{year}.csv"), index=False)
    synthetic_availability.to_csv(os.path.join(cache_dir, "expected_times_c10_for_strategic.csv"), index=False)
    synthetic_simulation.to_csv(os.path.join(cache_dir, f"sb_12m_pressure_{year}.csv"), index=False)
    week_calendar.to_csv(os.path.join(cache_dir, "weekly_calendar.csv"), index=False)
    weekly_direct_df.to_csv(os.path.join(cache_dir, "weekly_direct_suitability.csv"), index=False)
    slot_availability_df.to_csv(os.path.join(cache_dir, "weekly_slot_availability.csv"), index=False)
    shutil.copy2(modes_path, os.path.join(cache_dir, "sb12m_master_with_modes.csv"))
    shutil.copy2(accepted_path, os.path.join(cache_dir, "accepted_projects_cycle10.csv"))
    pressure_with_lst_path = os.path.join(data_dir, "sb_12m_pressure.csv")
    if os.path.exists(pressure_with_lst_path):
        shutil.copy2(pressure_with_lst_path, os.path.join(cache_dir, "sb_12m_pressure.csv"))

    with open(os.path.join(cache_dir, "weekly_direct_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cycle_anchor_date": str(pd.Timestamp(cycle_anchor_date).date()),
                "cycle_end_date": str(pd.Timestamp(cycle_end_date).date()),
                "year": int(year),
                "processed_non_current_cycle_files": int(processed_files),
                "skipped_current_cycle_files": int(skipped_current_cycle_files),
                "unique_week_bin_pairs": int(len(total_slots_by_week_bin)),
                "nonzero_sb_week_bin_pairs": int(len(weekly_direct_df)),
            },
            handle,
            indent=2,
        )

    print(
        f"Created weekly direct-suitability artifacts in {cache_dir} "
        f"from {processed_files} non-current-cycle dsa_sim files"
    )
    _log_weekly_direct_suitability_summary(cache_dir)
    return cache_dir


def _load_match_eb_mapping_for_weekly_direct(output_dir: str) -> dict[str, list[str]]:
    match_path = os.path.join(output_dir, "match_eb_sbuid.csv")
    if not os.path.exists(match_path):
        raise FileNotFoundError(f"Expected match_eb mapping was not written: {match_path}")
    match_df = pd.read_csv(match_path)
    if not {"eb", "SB_UID"}.issubset(match_df.columns):
        raise ValueError(f"Malformed match_eb mapping file: {match_path}")
    grouped = match_df.groupby("SB_UID")["eb"].apply(list)
    return {str(sb_uid): [str(eb) for eb in eb_list] for sb_uid, eb_list in grouped.items()}


def _build_weekly_direct_suitability_tensor(
        weekly_direct_df: pd.DataFrame,
        match_eb_by_sb_uid: dict[str, list[str]],
) -> dict[str, dict[str, dict[int, float]]]:
    suitability: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for _, row in weekly_direct_df.iterrows():
        sb_uid = str(row["SB_UID"])
        eb_ids = match_eb_by_sb_uid.get(sb_uid, [])
        if not eb_ids:
            continue
        week_label = str(row["Week_Label"])
        lst_bin = int(row["lst_bin"])
        suitability_fraction = float(row["suitability_fraction"])
        for eb_id in eb_ids:
            suitability[eb_id][week_label][lst_bin] = suitability_fraction
    return suitability


def _log_weekly_direct_preprocess_state(
        x: dict,
        projects_eb: dict,
        bins_eb: dict,
        ex_bal: dict,
        config: list,
        Tti: dict,
        job_weight: dict,
        max_value,
) -> None:
    print("\n" + "=" * 80)
    print("WEEKLY DIRECT PREPROCESS STATE")
    print("=" * 80)
    print(
        f"x_jobs={len(x)}, projects={len(projects_eb)}, configs={len(config)}, "
        f"bins_eb_type={type(bins_eb).__name__}, ex_bal_type={type(ex_bal).__name__}, "
        f"job_weight_type={type(job_weight).__name__}, max_value_type={type(max_value).__name__}"
    )

    sample_job = next(iter(x.keys()), None)
    if sample_job is not None:
        sample_configs = sorted(list(x[sample_job].keys()))[:3]
        sample_bins = []
        for conf_name in sample_configs[:1]:
            sample_bins = sorted(list(x[sample_job][conf_name].keys()))[:5]
        print(
            f"sample_job={sample_job}, sample_job_bins={bins_eb.get(sample_job)}, "
            f"sample_job_weight={job_weight.get(sample_job)}, "
            f"sample_job_ex_bal={ex_bal.get(sample_job)}, "
            f"sample_job_configs={sample_configs}, sample_start_bins={sample_bins}"
        )

    sample_project = next(iter(projects_eb.keys()), None)
    if sample_project is not None:
        print(f"sample_project={sample_project}, sample_project_jobs={projects_eb[sample_project][:5]}")

    sample_config = config[0] if config else None
    if sample_config is not None:
        sample_tti = list(sorted(Tti.get(sample_config, {}).items()))[:5]
        print(f"sample_config={sample_config}, sample_tti={sample_tti}")

    if not isinstance(max_value, dict):
        print(
            "NOTE: preprocess() returned scalar max_value normalization; "
            "weekly direct solver will use per-job assignment cap <= 1."
        )
    print("=" * 80 + "\n")


def _solve_long_term_schedule_weekly_direct(
        weights: dict,
        output_path: str,
        synthetic_data_dir: str,
        jobs: list,
        config_start_date: str,
        sb_map: dict,
        year: int,
        exec_time_used: dict,
        elapsed_bins: int,
        end_date: str,
        eb_objective_type: str,
        time_limit: int,
        use_lp_relaxation: bool,
        calendar_only: bool,
):
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    configurable_dict_local = configurable_dict.copy()
    configurable_dict_local.update({
        "year": year,
        "main": f"sb12m_master_prepared_c10_{year}.csv",
        "availability": "expected_times_c10_for_strategic.csv",
        "simulation": f"sb_12m_pressure_{year}.csv",
        "modes": "sb12m_master_with_modes.csv",
        "suppress_fraction_obs_debug": True,
    })

    (
        eb_temp,
        projects_eb,
        sb_value,
        max_value,
        bins_eb,
        ex_bal,
        org_p,
        config,
        Tti,
        eb_to_grade,
        project_to_grade,
        calendar,
        pref_conf,
        job_weight,
    ) = preprocess(
        configurable_dict=configurable_dict_local,
        jobs=jobs,
        start_date=config_start_date,
        sb_map=sb_map,
        data_dir=synthetic_data_dir,
        output_path=output_dir,
        end_date=end_date,
        cycle_start_year=year,
    )

    if calendar_only:
        weekly_calendar_path = os.path.join(synthetic_data_dir, "weekly_calendar.csv")
        return pd.read_csv(weekly_calendar_path)

    weekly_direct_df = pd.read_csv(os.path.join(synthetic_data_dir, "weekly_direct_suitability.csv"))
    match_eb_by_sb_uid = _load_match_eb_mapping_for_weekly_direct(output_dir=output_dir or synthetic_data_dir)
    direct_suitability = _build_weekly_direct_suitability_tensor(
        weekly_direct_df=weekly_direct_df,
        match_eb_by_sb_uid=match_eb_by_sb_uid,
    )

    remaining_bins = sum(sum(Tti[c].values()) for c in config)
    if exec_time_used is not None:
        actual_exec_time_used = exec_time_used
    else:
        actual_exec_time_used = {'CL': 0.0, 'EA': 0.0, 'EU': 0.0, 'NA': 0.0}

    total_observed_time = sum(actual_exec_time_used.values())
    eb_denominator = total_observed_time + remaining_bins
    adjusted_org_p = {}
    for org in org_p.keys():
        original_target_time = org_p[org] * eb_denominator
        completed_time = actual_exec_time_used.get(org, 0.0)
        remaining_target_time = original_target_time - completed_time
        adjusted_org_p[org] = (
            max(0.0, min(1.0, remaining_target_time / remaining_bins))
            if remaining_bins > 0
            else org_p[org]
        )

    m = gp.Model("GAPM_Weekly_Direct")
    assignment_vtype = GRB.CONTINUOUS if use_lp_relaxation else GRB.BINARY
    project_vtype = GRB.CONTINUOUS if use_lp_relaxation else GRB.BINARY

    x = {}
    for eb_id in eb_temp.keys():
        x[eb_id] = {}
        for conf_name in eb_temp[eb_id].keys():
            x[eb_id][conf_name] = {}
            for lst_bin, feasible in eb_temp[eb_id][conf_name].items():
                suitability_value = direct_suitability.get(eb_id, {}).get(conf_name, {}).get(lst_bin, 0.0)
                if feasible == 1 and suitability_value > 0.0:
                    x[eb_id][conf_name][lst_bin] = m.addVar(
                        vtype=assignment_vtype,
                        lb=0.0,
                        ub=1.0,
                        name=f"x[{eb_id},{conf_name},{lst_bin}]",
                    )

    y = {
        project_id: m.addVar(vtype=project_vtype, lb=0.0, ub=1.0, name=f"y[{project_id}]")
        for project_id in projects_eb.keys()
    }

    feasible_config_bin_to_vars = defaultdict(list)
    feasible_slots_by_job = defaultdict(list)
    for eb_id in x:
        for conf_name in x[eb_id]:
            for lst_bin, var in x[eb_id][conf_name].items():
                feasible_config_bin_to_vars[(conf_name, lst_bin)].append(var)
                feasible_slots_by_job[eb_id].append((conf_name, lst_bin, var))

    _log_weekly_direct_preprocess_state(
        x=x,
        projects_eb=projects_eb,
        bins_eb=bins_eb,
        ex_bal=ex_bal,
        config=config,
        Tti=Tti,
        job_weight=job_weight,
        max_value=max_value,
    )

    w_r = weights.get('sb_A', 0) + weights.get('sb_B', 0) + weights.get('sb_C', 0)
    w_p = weights.get('proj_A', 0) + weights.get('proj_B', 0) + weights.get('proj_C', 0)
    w_t = weights.get('utilization', 0)
    w_e = weights.get('eb_penalty', 0)

    zeta1 = max(
        1e-9,
        sum(
            max(
                direct_suitability.get(eb_id, {}).get(conf_name, {}).get(lst_bin, 0.0)
                for conf_name, lst_bin, _ in feasible_slots_by_job.get(eb_id, [])
            ) * job_weight.get(eb_id, 0.0)
            for eb_id in feasible_slots_by_job.keys()
        ),
    )
    zeta2 = max(1e-9, len(projects_eb))
    zeta3 = max(1e-9, sum(bins_eb.values()))

    obj1_raw = quicksum(
        var * direct_suitability.get(eb_id, {}).get(conf_name, {}).get(lst_bin, 0.0) * job_weight.get(eb_id, 0.0)
        for eb_id in x
        for conf_name in x[eb_id]
        for lst_bin, var in x[eb_id][conf_name].items()
    )
    obj1 = (w_r / zeta1) * obj1_raw

    obj2_terms = []
    for project_id, project_jobs in projects_eb.items():
        denom = max(1, len(project_jobs))
        suitability_sum = sum(
            direct_suitability.get(eb_id, {}).get(conf_name, {}).get(lst_bin, 0.0)
            for eb_id in project_jobs
            for conf_name in x.get(eb_id, {})
            for lst_bin in x[eb_id][conf_name].keys()
        )
        coeff = (w_p / zeta2) * (suitability_sum / denom)
        obj2_terms.append(coeff * y[project_id])
    obj2 = quicksum(obj2_terms)

    obj3_raw = quicksum(
        var * direct_suitability.get(eb_id, {}).get(conf_name, {}).get(lst_bin, 0.0) * bins_eb.get(eb_id, 0.0)
        for eb_id in x
        for conf_name in x[eb_id]
        for lst_bin, var in x[eb_id][conf_name].items()
    )
    obj3 = (w_t / zeta3) * obj3_raw

    time_per_exec = {}
    for organization in adjusted_org_p.keys():
        time_per_exec[organization] = quicksum(
            x[eb_id][conf_name][lst_bin] * ex_bal.get(eb_id, {}).get(organization, 0) * bins_eb.get(eb_id, 0)
            for eb_id in x
            for conf_name in x[eb_id]
            for lst_bin in x[eb_id][conf_name]
        )

    shortfall = {org: m.addVar(name=f"shortfall_{org}", lb=-GRB.INFINITY) for org in adjusted_org_p.keys()}
    for org in adjusted_org_p.keys():
        m.addConstr(time_per_exec[org] / eb_denominator - adjusted_org_p[org] == shortfall[org], name=f"shortfall_def_{org}")

    if eb_objective_type == "quadratic":
        obj4_penalty = quicksum(shortfall[org] * shortfall[org] for org in adjusted_org_p.keys())
    else:
        breakpoints = np.linspace(-1.0, 1.0, 11)
        z_squared_approx = {org: m.addVar(lb=0.0, name=f"z_sq_{org}") for org in adjusted_org_p.keys()}
        for org in adjusted_org_p.keys():
            for point in breakpoints:
                slope = 2 * point
                intercept = -(point ** 2)
                m.addConstr(
                    z_squared_approx[org] >= slope * shortfall[org] + intercept,
                    name=f"pwlin_{org}_{point:.2f}".replace("-", "m"),
                )
        obj4_penalty = quicksum(z_squared_approx[org] for org in adjusted_org_p.keys())
    obj4 = -w_e * obj4_penalty

    m.setObjective(obj1 + obj2 + obj3 + obj4, GRB.MAXIMIZE)

    for conf_name in config:
        for lst_bin in Tti[conf_name].keys():
            overlapping_vars = []
            for eb_id in x:
                duration = int(bins_eb.get(eb_id, 0))
                for start_bin, var in x[eb_id].get(conf_name, {}).items():
                    if start_bin <= lst_bin < start_bin + duration:
                        overlapping_vars.append(var)
            m.addConstr(
                quicksum(overlapping_vars) <= Tti[conf_name][lst_bin],
                name=f"time_capacity_{conf_name}_{lst_bin}",
            )

    for eb_id in x:
        m.addConstr(
            quicksum(
                x[eb_id][conf_name][lst_bin]
                for conf_name in x[eb_id]
                for lst_bin in x[eb_id][conf_name]
            ) <= 1,
            name=f"job_assign_{eb_id}",
        )

    for project_id, project_jobs in projects_eb.items():
        lhs = quicksum(
            x[eb_id][conf_name][lst_bin]
            for eb_id in project_jobs
            for conf_name in x.get(eb_id, {})
            for lst_bin in x[eb_id][conf_name]
        )
        m.addConstr(lhs >= y[project_id], name=f"project_complete_lb_{project_id}")
        m.addConstr(lhs <= max(1, len(project_jobs)) * y[project_id], name=f"project_complete_ub_{project_id}")

    m.Params.OutputFlag = 1
    m.Params.TimeLimit = time_limit
    if not use_lp_relaxation:
        m.Params.MIPGap = 0.0005
        print("Weekly direct solver Gurobi params: TimeLimit=%ss, MIPGap=0.0005" % int(time_limit))
    else:
        print("Weekly direct solver Gurobi params: TimeLimit=%ss, LP relaxation mode" % int(time_limit))
    if use_lp_relaxation:
        m.Params.Method = 2
        m.Params.Crossover = 0

    m.optimize()

    if output_path and m.SolCount > 0:
        assignments, parsing_errors = _parse_x_assignments_from_model(m, 0.5 if not use_lp_relaxation else 1e-6)
        if parsing_errors:
            print(f"Weekly direct solver parsing warnings: {len(parsing_errors)}")
        if assignments:
            df_solution = pd.DataFrame(assignments)
            eb_to_sb_uid = {
                eb_id: sb_uid
                for sb_uid, eb_ids in match_eb_by_sb_uid.items()
                for eb_id in eb_ids
            }
            df_solution = df_solution.rename(columns={
                "eb": "EB",
                "configuration": "Configuration",
                "lst_bin": "LST_bin",
                "value": "Assignment_Value",
            })
            df_solution["SB_UID"] = df_solution["EB"].map(eb_to_sb_uid)
            df_solution["Week_Label"] = df_solution["Configuration"]
            df_solution.to_csv(output_path, index=False)
        else:
            pd.DataFrame(columns=["var_name", "Assignment_Value", "EB", "SB_UID", "Configuration", "LST_bin", "Week_Label"]).to_csv(
                output_path, index=False
            )

    return None


def _parse_x_assignments_from_model(model: gp.Model, threshold: float) -> tuple[list, list]:
    """Extract x[j,c,t] assignments above the provided threshold."""
    assignments = []
    parsing_errors = []
    for var in model.getVars():
        if not var.varName.startswith('x') or var.x <= threshold:
            continue
        cleaned = var.varName.replace('x[', '').replace(']', '')
        parts = cleaned.split(',')
        if len(parts) < 2:
            parsing_errors.append(f"Unexpected format: {var.varName} -> parts: {parts}")
            continue

        eb = parts[0].strip()
        conf = parts[1].strip()
        time_bin = None
        if len(parts) >= 3:
            time_bin_str = parts[2].strip()
            try:
                if time_bin_str.lower() not in ['nan', 'none', '']:
                    time_bin = float(time_bin_str)
            except (ValueError, TypeError):
                if time_bin_str.lower() not in ['nan', 'none']:
                    time_bin = time_bin_str
        else:
            parsing_errors.append(f"Missing time bin in: {var.varName}")

        assignments.append({
            'var_name': var.varName,
            'value': float(var.x),
            'eb': eb,
            'configuration': conf,
            'lst_bin': time_bin,
        })
    return assignments, parsing_errors


def solve_long_term_schedule_weekly(
        weights: dict,
        output_path: str,
        data_dir: str,
        jobs: list,
        config_start_date: str,
        sb_map: dict,
        cycle_anchor_date: str,
        preprocessed_root: str = None,
        calendar_only: bool = False,
        year: int = None,
        exec_time_used: dict = None,
        elapsed_bins: int = 0,
        end_date: str = None,
        eb_objective_type: str = "piecewise_linear",
        time_limit: int = 6000,
        use_lp_relaxation: bool = False,
):
    """Solve a weekly strategic schedule using anchored 7-day week buckets and LST bins."""
    if year is None:
        try:
            year = pd.to_datetime(config_start_date).year
        except Exception:
            year = 2023

    if preprocessed_root is None:
        raise ValueError("solve_long_term_schedule_weekly requires preprocessed_root for historical realized weather.")

    synthetic_data_dir = _build_weekly_direct_suitability_inputs(
        data_dir=data_dir,
        cycle_anchor_date=cycle_anchor_date,
        cycle_end_date=end_date,
        year=year,
        preprocessed_root=preprocessed_root,
    )

    return _solve_long_term_schedule_weekly_direct(
        weights=weights,
        output_path=output_path,
        synthetic_data_dir=synthetic_data_dir,
        jobs=jobs,
        config_start_date=config_start_date,
        sb_map=sb_map,
        year=year,
        exec_time_used=exec_time_used,
        elapsed_bins=elapsed_bins,
        end_date=end_date,
        eb_objective_type=eb_objective_type,
        time_limit=time_limit,
        use_lp_relaxation=use_lp_relaxation,
        calendar_only=calendar_only,
    )


def solve_long_term_schedule(weights: dict, output_path: str, data_dir: str, jobs: list, config_start_date: str, sb_map: dict, calendar_only: bool = False, year: int = None, exec_time_used: dict = None, elapsed_bins: int = 0, end_date: str = None, eb_objective_type: str = "piecewise_linear", time_limit: int = 6000, use_lp_relaxation: bool = False):
    """
    Solve the long-term strategic scheduling problem (Section 4.1 of paper).
    
    Args:
        weights: Dictionary of objective weights (alpha1-4 components)
        output_path: File path for output CSV
        data_dir: Directory containing input data files
        jobs: List of job dictionaries with remaining_execs, etc.
        config_start_date: Start date for strategic scheduling
        sb_map: Mapping of SB UIDs to job IDs
        calendar_only: If True, only load calendar without optimization
        year: Year for file selection (extracted from config_start_date if None)
        exec_time_used: Cumulative executive time from prior chunks
        elapsed_bins: Number of time bins elapsed before this scheduling point
        end_date: End date for filtering configurations
        eb_objective_type: Type of executive balance objective. Options:
            - "quadratic": Original exact quadratic objective -Σ(e_i - fraction_i)²
              More accurate but slower due to quadratic programming.
            - "piecewise_linear": Piecewise linear approximation of the quadratic.
              Uses tangent line approximations for faster LP solving.
              Default option for better performance.
    
    Returns:
        Calendar DataFrame if calendar_only=True, otherwise None (writes to output_path)
    """
    # output_path is the file path for the output CSV
    # Extract the directory from output_path for intermediate files like match_eb_sbuid.csv
    # NOTE: total_year_bins is calculated as elapsed_bins + remaining_bins (from Tti)
    # This combines actual observed past with estimated future

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    if calendar_only:
        print("--- Loading Long-Term Calendar (skipping optimization) ---")
    else:
        print("--- Starting Long-Term Optimization with Configurable Objective ---")
        print(f"Objective Weights: {weights}")

    print(f"\n--- File Path Configuration ---")
    print(f"data_dir: {data_dir}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Determine year for file selection
    if year is None:
        # Extract year from config_start_date if not provided
        try:
            year = pd.to_datetime(config_start_date).year
        except:
            year = 2023  # Default fallback
            print(f"Warning: Could not extract year from {config_start_date}, using default {year}")
    
    print(f"Using year: {year} for file selection")
    
    # Use year-specific files if available, otherwise fall back to default
    main_path = os.path.join(data_dir, f"sb12m_master_prepared_c10_{year}.csv")
    simulation_path = os.path.join(data_dir, f"sb_12m_pressure_{year}.csv")
    
    # Check if year-specific files exist, otherwise use defaults
    if not os.path.exists(main_path):
        print(f"Year-specific main file not found: {main_path}")
    
    if not os.path.exists(simulation_path):
        print(f"Year-specific simulation file not found: {simulation_path}")
    
    print(f"Main file path: {main_path}")
    print(f"Simulation file path: {simulation_path}")
    
    configurable_dict = dict(
        main=main_path,
        availability=os.path.join(data_dir, "expected_times_c10_for_strategic.csv"),
        simulation=simulation_path,
        modes=os.path.join(data_dir, "sb12m_master_with_modes.csv"),
        cycle_accepted=os.path.join(data_dir, 'accepted_projects_cycle10.csv'),
        borderconf_1='3A',
        borderconf_2='1A',
        d=0,
        total_bins=8600,
        model_gap=0.01,
        problem='plan',
        out_file=os.path.join(data_dir, "cycle10_med_term.csv"),
        year=year  # Pass year to preprocess for A/B/C parsing
    )
    print("--- End File Path Configuration ---\n")
    
    # === DEBUG: Print info about input jobs list ===
    print("\n" + "=" * 80)
    print("DEBUG: INPUT JOBS LIST INFO (from operational scheduler)")
    print("=" * 80)
    print(f"  Total jobs passed to strategic scheduler: {len(jobs)}")
    if jobs:
        unique_job_ids = set(j.get('job_id', 'unknown') for j in jobs)
        print(f"  Unique job_ids: {len(unique_job_ids)}")
        # Show distribution of remaining_execs
        remaining_execs_dist = {}
        for j in jobs:
            re = j.get('remaining_execs', 0)
            remaining_execs_dist[re] = remaining_execs_dist.get(re, 0) + 1
        print(f"  Remaining executions distribution: {dict(sorted(remaining_execs_dist.items()))}")
    print("=" * 80 + "\n")
    
    # === DEBUG: Verify cycle progress tracking ===
    print("\n" + "=" * 80)
    print("DEBUG: CYCLE PROGRESS TRACKING")
    print("=" * 80)
    print(f"  Config start date: {config_start_date}")
    print(f"  (Strategic schedule will be calculated for configurations starting from this date)")
    
    # Count jobs with some executions already completed
    jobs_with_completions = 0
    total_completions = 0
    total_remaining = 0
    jobs_fully_completed = 0
    
    for job in jobs:
        total = job.get('total_execs', 0)
        remaining = job.get('remaining_execs', 0)
        completed = total - remaining
        
        if completed > 0:
            jobs_with_completions += 1
            total_completions += completed
        if remaining == 0 and total > 0:
            jobs_fully_completed += 1
        total_remaining += remaining
    
    print(f"\n  Jobs with at least one completed execution: {jobs_with_completions}")
    print(f"  Jobs fully completed (remaining=0): {jobs_fully_completed}")
    print(f"  Total completed executions: {total_completions}")
    print(f"  Total remaining executions: {total_remaining}")
    
    if exec_time_used:
        print(f"\n  Executive time used so far (from simulation):")
        for org, time_used in exec_time_used.items():
            print(f"    {org}: {time_used:.1f} time slots")
        print(f"    Total: {sum(exec_time_used.values()):.1f} time slots")
    else:
        print(f"\n  NOTE: No exec_time_used provided - will calculate from job completions")
    
    print("=" * 80 + "\n")
    
    (eb_temp, projects_eb, sb_value,
     max_value, bins_eb, ex_bal,
     org_p, config, Tti, eb_to_grade, project_to_grade, calendar, pref_conf, job_weight) = (
        preprocess(configurable_dict, jobs, config_start_date, sb_map, data_dir=data_dir, output_path=output_dir, end_date=end_date, cycle_start_year=year))
    ## pass sb_map, sb_executed, period_start_date

    # === DEBUG: Print which configurations are included in the strategic schedule ===
    print("\n" + "=" * 80)
    print("DEBUG: CONFIGURATIONS IN STRATEGIC SCHEDULE")
    print("=" * 80)
    print(f"  Config start date: {config_start_date}")
    print(f"  Configurations to schedule: {config}")
    print(f"  Number of configurations: {len(config)}")
    if len(calendar) > 0:
        print(f"\n  Calendar details:")
        for idx, row in calendar.iterrows():
            print(f"    {row.get('Config', row.get('Configuration', 'unknown'))}: {row['Start']} to {row['End']}")
    print("=" * 80 + "\n")

    # If calendar_only flag is set, return early with just the calendar
    if calendar_only:
        print("Calendar loaded. Skipping optimization and returning calendar.")
        return calendar

    # === CALCULATE REMAINING TIME AND ADJUSTED EXECUTIVE BALANCE TARGETS ===
    # 
    # Issue: The original code used hardcoded total_bins (8600) and the original org_p fractions.
    # This doesn't account for:
    # 1. We're partway through the year - we should use the remaining available time
    # 2. Some executive time has already been used - we need adjusted targets
    #
    # The goal is: if we hit the adjusted targets for the remaining time,
    # combined with what's already been completed, we'll hit the original org_p targets for the whole year.
    #
    # NOTE: exec_time_used and elapsed_bins should be passed from full_year.py based on:
    # - exec_time_used: cumulative executive time from valid executions in the current cycle's schedule
    # - elapsed_bins: actual number of non-NaN time steps that have passed in the simulation
    # We do NOT calculate these from total_execs - remaining_execs because some executions
    # may have occurred before the current cycle started.
    
    # Calculate remaining time from Tti (this is the expected available time for the rest of the cycle)
    remaining_bins = sum(sum(Tti[c].values()) for c in config)
    
    # Total estimated overall time: elapsed (observed available steps) + remaining available (from Tti)
    total_year_bins = elapsed_bins + remaining_bins
    
    # Use exec_time_used from the simulation (passed from full_year.py)
    # Default to zeros if not provided (e.g., at start of cycle)
    if exec_time_used is not None:
        actual_exec_time_used = exec_time_used
    else:
        actual_exec_time_used = {'CL': 0.0, 'EA': 0.0, 'EU': 0.0, 'NA': 0.0}
    
    # EB is computed as time_used_per_exec / total_time_used_by_all. We may not use full available time,
    # so the denominator for "year" should be: total OBSERVED time so far + estimated remaining time.
    total_observed_time = sum(actual_exec_time_used.values())
    eb_denominator = total_observed_time + remaining_bins  # used for org_p target calculations
    
    # === DEBUG: Print time tracking ===
    print("\n" + "=" * 80)
    print("DEBUG: TIME TRACKING FOR EXECUTIVE BALANCE")
    print("=" * 80)
    print(f"  Total observed time (time actually used by all executives so far): {total_observed_time:.1f} bins")
    print(f"  Estimated denominator for EB (observed + estimated remaining): {eb_denominator:.1f} bins")
    print(f"  Total elapsed time (actual non-NaN time steps passed): {elapsed_bins}")
    print(f"  Total estimated overall time (elapsed + remaining from Tti): {total_year_bins}")
    print(f"  Remaining bins (estimated from Tti): {remaining_bins}")
    print(f"  Config start date: {config_start_date}")
    print("")
    print(f"  Executive time used so far (from simulation):")
    for org, time_used in actual_exec_time_used.items():
        print(f"    {org}: {time_used:.1f} time slots")
    print(f"    Total: {sum(actual_exec_time_used.values()):.1f} time slots")
    print("=" * 80 + "\n")
    
    # Calculate adjusted executive balance targets
    # Goal: (completed_time + new_time) / eb_denominator = org_p[o], with eb_denominator = observed + remaining
    # So: remaining_target_time = org_p[o] * eb_denominator - completed_time
    #     adjusted_frac = remaining_target_time / remaining_bins (share of remaining time this exec needs)
    adjusted_org_p = {}
    print("\n" + "=" * 80)
    print("DEBUG: ADJUSTED EXECUTIVE BALANCE TARGETS")
    print("=" * 80)
    print(f"  Original targets (org_p): {org_p}")
    print(f"  Adjusting for completed work and remaining time...")
    print("")
    
    for o in org_p.keys():
        original_target_time = org_p[o] * eb_denominator
        completed_time = actual_exec_time_used.get(o, 0)
        remaining_target_time = original_target_time - completed_time
        
        if remaining_bins > 0:
            adjusted_frac = remaining_target_time / remaining_bins
            # Clamp to [0, 1] range
            adjusted_frac = max(0.0, min(1.0, adjusted_frac))
        else:
            adjusted_frac = org_p[o]  # Fallback if no remaining time
        
        adjusted_org_p[o] = adjusted_frac
        
        status = ""
        if adjusted_frac > org_p[o] + 0.01:
            status = "(NEED TO CATCH UP)"
        elif adjusted_frac < org_p[o] - 0.01:
            status = "(CAN REDUCE)"
        else:
            status = "(ON TRACK)"
        
        print(f"  {o}:")
        print(f"    Original target: {org_p[o]*100:.2f}% of EB denominator = {original_target_time:.1f} bins")
        print(f"    Already completed: {completed_time:.1f} bins")
        print(f"    Remaining needed: {remaining_target_time:.1f} bins")
        print(f"    Adjusted target for remaining time: {adjusted_frac*100:.2f}% {status}")
    
    print(f"\n  Adjusted targets sum: {sum(adjusted_org_p.values())*100:.2f}%")
    print("=" * 80 + "\n")

    total_sbs_A = sum(1 for grade in eb_to_grade.values() if grade == 'A')
    total_sbs_B = sum(1 for grade in eb_to_grade.values() if grade == 'B')
    total_sbs_C = sum(1 for grade in eb_to_grade.values() if grade == 'C')

    total_projects_A = sum(1 for grade in project_to_grade.values() if grade == 'A')
    total_projects_B = sum(1 for grade in project_to_grade.values() if grade == 'B')
    total_projects_C = sum(1 for grade in project_to_grade.values() if grade == 'C')

    print(f"total_sbs_A: {total_sbs_A}, total_sbs_B: {total_sbs_B}, total_sbs_C: {total_sbs_C}")
    print(f"total_projects_A: {total_projects_A}, total_projects_B: {total_projects_B}, total_projects_C: {total_projects_C}")

    # === DEBUG: Print summary of available time per configuration ===
    print("\n" + "=" * 80)
    print("DEBUG: AVAILABLE TIME PER CONFIGURATION (from Tti)")
    print("=" * 80)
    for c in config:
        total_bins_for_config = sum(Tti[c].values())
        print(f"  {c}: {total_bins_for_config} total bins across all LST slots")
    print(f"  TOTAL across all configs: {sum(sum(Tti[c].values()) for c in config)} bins")
    print("=" * 80 + "\n")

    # === DEBUG: Print summary of jobs available per configuration in eb_temp ===
    print("\n" + "=" * 80)
    print("DEBUG: JOBS AVAILABLE PER CONFIGURATION (from eb_temp)")
    print("=" * 80)
    jobs_per_config = {c: 0 for c in config}
    for j in eb_temp.keys():
        for c in eb_temp[j].keys():
            if any(eb_temp[j][c].get(t, 0) == 1 for t in eb_temp[j][c]):
                jobs_per_config[c] += 1
    for c in config:
        print(f"  {c}: {jobs_per_config[c]} jobs have at least one available bin")
    print("=" * 80 + "\n")

    # === DEBUG: Print full list of jobs available in 8A ===
    print("\n" + "=" * 80)
    print("DEBUG: FULL LIST OF JOBS AVAILABLE IN 8A")
    print("=" * 80)
    for j in eb_temp.keys():
        for c in eb_temp[j].keys():
            if c == '8A':
                print(f"  {j}")
    print("=" * 80 + "\n")

    # === DEBUG: Print summary of job durations (bins_eb) ===
    print("\n" + "=" * 80)
    print("DEBUG: JOB DURATION DISTRIBUTION (bins_eb)")
    print("=" * 80)
    duration_counts = {}
    for j, dur in bins_eb.items():
        duration_counts[dur] = duration_counts.get(dur, 0) + 1
    for dur in sorted(duration_counts.keys()):
        print(f"  Duration {dur} bins: {duration_counts[dur]} jobs")
    avg_duration = sum(bins_eb.values()) / len(bins_eb) if bins_eb else 0
    print(f"  Average duration: {avg_duration:.2f} bins")
    print(f"  Total jobs in bins_eb: {len(bins_eb)}")
    print("=" * 80 + "\n")

    m = gp.Model('GAPM_Configurable')
    lp_config_bin_spread_weight = float(weights.get('lp_config_bin_spread_weight', 0.01))
    lp_sb_slot_spread_weight = float(weights.get('lp_sb_slot_spread_weight', 0.005))
    lp_spread_epsilon = float(weights.get('lp_spread_epsilon', 0.05))
    if lp_spread_epsilon <= 0:
        lp_spread_epsilon = 0.05

    # --- 1. Define Variables (largely the same as your original model) ---
    x = {}
    assignment_vtype = GRB.CONTINUOUS if use_lp_relaxation else GRB.BINARY
    project_vtype = GRB.CONTINUOUS if use_lp_relaxation else GRB.BINARY
    for j in eb_temp.keys():
        x[j] = {}
        for i in eb_temp[j].keys():
            x[j][i] = {}
            for bin_t in eb_temp[j][i].keys():
                if eb_temp[j][i][bin_t] == 1:
                    x[j][i][bin_t] = m.addVar(
                        vtype=assignment_vtype,
                        lb=0.0,
                        ub=1.0,
                        name=f'x[{j},{i},{bin_t}]'
                    )

    y = {
        k: m.addVar(vtype=project_vtype, lb=0.0, ub=1.0, name=f'y[{k}]')
        for k in projects_eb.keys()
    }

    feasible_config_bin_to_vars = defaultdict(list)
    feasible_slots_by_job = defaultdict(list)
    for j in x:
        for c in x[j]:
            for bin_t, var in x[j][c].items():
                feasible_config_bin_to_vars[(c, bin_t)].append(var)
                feasible_slots_by_job[j].append((c, bin_t, var))

    config_bin_mass = {}
    config_bin_support = {}
    sb_slot_support = {}
    if use_lp_relaxation:
        for (c, bin_t), vars_for_pair in feasible_config_bin_to_vars.items():
            key_name = f"{c}_{bin_t}".replace(".", "p")
            config_bin_mass[(c, bin_t)] = m.addVar(lb=0.0, name=f"mass_config_bin[{key_name}]")
            config_bin_support[(c, bin_t)] = m.addVar(lb=0.0, ub=1.0, name=f"u_config_bin[{key_name}]")
        for j, slots in feasible_slots_by_job.items():
            for c, bin_t, _ in slots:
                key_name = f"{j}_{c}_{bin_t}".replace(".", "p")
                sb_slot_support[(j, c, bin_t)] = m.addVar(lb=0.0, ub=1.0, name=f"u_sb_slot[{key_name}]")
    m.update()

    # === DEBUG: Print summary of variables created per configuration ===
    print("\n" + "=" * 80)
    print("DEBUG: VARIABLES CREATED PER CONFIGURATION (x[j][c][t])")
    print("=" * 80)
    vars_per_config = {c: 0 for c in config}
    jobs_with_vars_per_config = {c: set() for c in config}
    for j in x:
        for c in x[j]:
            num_bins = len(x[j][c])
            vars_per_config[c] += num_bins
            if num_bins > 0:
                jobs_with_vars_per_config[c].add(j)
    for c in config:
        print(f"  {c}: {len(jobs_with_vars_per_config[c])} unique jobs, {vars_per_config[c]} total (job, bin) variables")
    print("=" * 80 + "\n")

    # === DEBUG: Decision variables by executive (scheduling opportunities) ===
    print("\n" + "=" * 80)
    print("DEBUG: DECISION VARIABLES BY EXECUTIVE (Scheduling Opportunities)")
    print("=" * 80)
    _vars_by_exec = {'NA': 0, 'EA': 0, 'EU': 0, 'CL': 0}
    _bins_by_exec = {'NA': 0, 'EA': 0, 'EU': 0, 'CL': 0}
    _jobs_by_exec = {'NA': 0, 'EA': 0, 'EU': 0, 'CL': 0}
    for j in x:
        _primary_exec = max(['NA', 'EA', 'EU', 'CL'], key=lambda k: ex_bal.get(j, {}).get(k, 0))
        _job_bins = bins_eb.get(j, 0)
        _jobs_by_exec[_primary_exec] += 1
        for c in x[j]:
            _vars_by_exec[_primary_exec] += len(x[j][c])
            _bins_by_exec[_primary_exec] += len(x[j][c]) * _job_bins
    print(f"  {'Exec':<6} | {'Jobs':<8} | {'Variables':<12} | {'Potential Bins':<15} | {'Frac of Total':<15}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*12} | {'-'*15} | {'-'*15}")
    _total_bins = sum(_bins_by_exec.values())
    for org in ['NA', 'EA', 'EU', 'CL']:
        _frac = _bins_by_exec[org] / _total_bins if _total_bins > 0 else 0
        print(f"  {org:<6} | {_jobs_by_exec[org]:>8} | {_vars_by_exec[org]:>12} | {_bins_by_exec[org]:>15.0f} | {_frac:>15.2%}")
    print(f"\n  INSIGHT: If one exec has disproportionately many scheduling opportunities,")
    print(f"  the optimizer may favor them when maximizing utilization (obj3).")
    print("=" * 80 + "\n")

    # === DIAGNOSTIC: Available job time by executive ===
    # This helps identify if EB violations are due to skewed job availability
    print("\n" + "=" * 80)
    print("DEBUG: AVAILABLE JOB TIME BY EXECUTIVE (Jobs with decision variables)")
    print("=" * 80)
    available_time_by_exec = {'NA': 0.0, 'EA': 0.0, 'EU': 0.0, 'CL': 0.0}
    available_weighted_value_by_exec = {'NA': 0.0, 'EA': 0.0, 'EU': 0.0, 'CL': 0.0}
    jobs_by_exec = {'NA': 0, 'EA': 0, 'EU': 0, 'CL': 0}
    
    for j in x:
        job_bins = bins_eb.get(j, 0)
        job_value = job_weight.get(j, 0)
        for org in ['NA', 'EA', 'EU', 'CL']:
            org_frac = ex_bal.get(j, {}).get(org, 0)
            if org_frac > 0:
                available_time_by_exec[org] += org_frac * job_bins
                available_weighted_value_by_exec[org] += org_frac * job_bins * job_value
                if org_frac >= 0.5:  # Count as belonging to this exec if majority ownership
                    jobs_by_exec[org] += 1
    
    total_available_time = sum(available_time_by_exec.values())
    print(f"\n  If ALL jobs with decision variables were scheduled:")
    print(f"  {'Executive':<12} | {'Avail Time (bins)':<20} | {'Fraction':<12} | {'Target':<12} | {'Status':<15} | {'Majority Jobs':<15}")
    print(f"  {'-'*12} | {'-'*20} | {'-'*12} | {'-'*12} | {'-'*15} | {'-'*15}")
    for org in ['NA', 'EA', 'EU', 'CL']:
        avail_frac = available_time_by_exec[org] / total_available_time if total_available_time > 0 else 0
        target = adjusted_org_p.get(org, org_p.get(org, 0))
        diff = avail_frac - target
        if abs(diff) < 0.02:
            status = "OK"
        elif diff > 0:
            status = f"EXCESS +{diff*100:.1f}%"
        else:
            status = f"SHORTAGE {diff*100:.1f}%"
        print(f"  {org:<12} | {available_time_by_exec[org]:>20.1f} | {avail_frac:>12.2%} | {target:>12.2%} | {status:<15} | {jobs_by_exec[org]:>15}")
    print(f"  {'-'*12} | {'-'*20} | {'-'*12} | {'-'*12} | {'-'*15} | {'-'*15}")
    print(f"  {'TOTAL':<12} | {total_available_time:>20.1f} | {'100.00%':>12} | {'100.00%':>12} | {'':<15} | {sum(jobs_by_exec.values()):>15}")
    
    # Check if EB is achievable
    eb_achievable = True
    eb_issues = []
    for org in ['NA', 'EA', 'EU', 'CL']:
        avail_frac = available_time_by_exec[org] / total_available_time if total_available_time > 0 else 0
        target = adjusted_org_p.get(org, org_p.get(org, 0))
        if avail_frac < target - 0.02:
            eb_achievable = False
            eb_issues.append(f"{org}: only {avail_frac*100:.1f}% available but target is {target*100:.1f}%")
    
    if not eb_achievable:
        print(f"\n  *** WARNING: EB TARGETS MAY BE UNACHIEVABLE ***")
        print(f"  The available job pool is skewed. Issues:")
        for issue in eb_issues:
            print(f"    - {issue}")
        print(f"  The optimizer cannot hit EB targets if there aren't enough jobs for certain executives!")
    else:
        print(f"\n  EB targets appear achievable based on available job pool.")
    print("=" * 80 + "\n")

    # --- 2. Paper Section 4.1: Strategic scheduler objective (mirrors operational) ---
    # Normalization: n'' = remaining_bins; ζ1 = (B ∧ n'') max_b w_b, ζ2 = (P ∧ n'') max_p w_p, ζ3 = ζ4 = n''
    n_prime_prime = remaining_bins
    if n_prime_prime <= 0:
        n_prime_prime = 1
    B_count = len(x)
    P_count = len(projects_eb)
    max_w = max((job_weight.get(j, 0) for j in x), default=1.0)
    if max_w <= 0:
        max_w = 1.0
    zeta1 = min(B_count, n_prime_prime) * max_w
    zeta2 = min(P_count, n_prime_prime) * max_w
    zeta3 = n_prime_prime
    zeta4 = n_prime_prime
    if zeta1 <= 0:
        zeta1 = 1.0
    if zeta2 <= 0:
        zeta2 = 1.0

    # Alpha weights (same as operational greedy)
    alpha1 = weights.get('sb_A', 0) + weights.get('sb_B', 0) + weights.get('sb_C', 0)
    alpha2 = weights.get('proj_A', 0) + weights.get('proj_B', 0) + weights.get('proj_C', 0)
    alpha3 = weights.get('utilization', 0)
    alpha4 = weights.get('eb_penalty', 0)
    
    # --- Appendix A: Log strategic objective normalization constants ---
    print("\n" + "=" * 90)
    print("APPENDIX A: STRATEGIC SCHEDULER NORMALIZATION CONSTANTS")
    print("=" * 90)
    print(f"  Per paper Section 4.1, the strategic scheduler uses the following normalization:")
    print(f"\n  === Raw Values ===")
    print(f"    n'' (remaining_bins, estimated available time for rest of cycle) = {n_prime_prime} bins ({n_prime_prime/2.0:.1f} hours)")
    print(f"    B (number of SB execution blocks with decision variables) = {B_count}")
    print(f"    P (number of projects) = {P_count}")
    print(f"    max_w (maximum job/project weight) = {max_w:.6f}")
    print(f"\n  === Normalization Constants (ζ) ===")
    print(f"    ζ1 = min(B, n'') × max_w = min({B_count}, {n_prime_prime}) × {max_w:.6f} = {zeta1:.4f}")
    print(f"    ζ2 = min(P, n'') × max_w = min({P_count}, {n_prime_prime}) × {max_w:.6f} = {zeta2:.4f}")
    print(f"    ζ3 = n'' = {zeta3} (for utilization objective)")
    print(f"    ζ4 = n'' = {zeta4} (for executive balance penalty)")
    print(f"\n  === Objective Weights (α) ===")
    print(f"    α1 (SB completion) = sb_A + sb_B + sb_C = {weights.get('sb_A', 0)} + {weights.get('sb_B', 0)} + {weights.get('sb_C', 0)} = {alpha1}")
    print(f"    α2 (project completion) = proj_A + proj_B + proj_C = {weights.get('proj_A', 0)} + {weights.get('proj_B', 0)} + {weights.get('proj_C', 0)} = {alpha2}")
    print(f"    α3 (utilization) = {alpha3}")
    print(f"    α4 (EB penalty) = {alpha4}")
    print(f"\n  === Comparison with Operational Scheduler ===")
    print(f"    The operational scheduler (greedy) uses:")
    print(f"      η1 = min(B_total_execs, n') × max_w  where n' = total observable time")
    print(f"      η2 = min(P, n') × max_w")
    print(f"      η3 = n'")
    print(f"      η4 = total observed time (actual time used)")
    print(f"    The strategic scheduler uses:")
    print(f"      ζ1, ζ2, ζ3, ζ4 as above (based on estimated remaining time)")
    print(f"\n  === Key Insight ===")
    print(f"    The strategic schedule is optimizing over expected future time (n'' = {n_prime_prime}).")
    print(f"    The EB targets have been adjusted based on:")
    print(f"      - Elapsed bins (actual past): {elapsed_bins}")
    print(f"      - Executive time already used: {sum(actual_exec_time_used.values()):.1f} bins")
    print(f"      - Remaining time estimate: {remaining_bins} bins")
    print("=" * 90 + "\n")

    # Helper: g_b(c,θ) from paper — use pref_conf (FRACTION_OBS) per (job, config); same for all θ in config
    def g_bc(j, c):
        try:
            if c not in pref_conf.columns:
                return 0.0
            s = pref_conf[c]
            if j not in s.index:
                return 0.0
            v = s[j]
            return float(v) if pd.notna(v) and v > 0 else 0.0
        except Exception:
            return 0.0

    # === DEBUG: FRACTION_OBS (g_bc) by executive ===
    print("\n" + "=" * 80)
    print("DEBUG: FRACTION_OBS (g_bc) VALUES BY EXECUTIVE")
    print("=" * 80)
    _g_bc_by_exec = {'NA': [], 'EA': [], 'EU': [], 'CL': []}
    _g_bc_sum_by_exec = {'NA': 0.0, 'EA': 0.0, 'EU': 0.0, 'CL': 0.0}
    for j in x:
        _primary_exec = max(['NA', 'EA', 'EU', 'CL'], key=lambda k: ex_bal.get(j, {}).get(k, 0))
        for c in x[j]:
            _g_val = g_bc(j, c)
            if _g_val > 0:
                _g_bc_by_exec[_primary_exec].append(_g_val)
                _g_bc_sum_by_exec[_primary_exec] += _g_val
    _g_bc_avg_by_exec = {k: (sum(v)/len(v) if v else 0) for k, v in _g_bc_by_exec.items()}
    _g_bc_count_by_exec = {k: len(v) for k, v in _g_bc_by_exec.items()}
    print(f"  {'Exec':<6} | {'Count (job,config pairs)':<25} | {'Avg g_bc':<12} | {'Sum g_bc':<12}")
    print(f"  {'-'*6} | {'-'*25} | {'-'*12} | {'-'*12}")
    for org in ['NA', 'EA', 'EU', 'CL']:
        print(f"  {org:<6} | {_g_bc_count_by_exec[org]:>25} | {_g_bc_avg_by_exec[org]:>12.4f} | {_g_bc_sum_by_exec[org]:>12.2f}")
    print(f"\n  INSIGHT: If one exec has much higher Sum g_bc, their jobs are more 'observable'")
    print(f"  and contribute more to obj1 (SB completion) and obj3 (utilization).")
    print("=" * 80 + "\n")

    # obj1(ψ): weighted observation completion = (1/ζ1) Σ ψ_{b,c,θ} g_b(c,θ) w_b
    obj1_raw = quicksum(
        x[j][i][t] * g_bc(j, i) * job_weight.get(j, 0)
        for j in x for i in x[j] for t in x[j][i]
    )
    obj1 = obj1_raw / zeta1 if zeta1 > 0 else 0

    # obj2(ψ): weighted project completion — coeff_p = (w_p/ζ2)*(1/|p|)*Σ_{b∈p,c,θ} g_b(c,θ); obj2 = Σ_p coeff_p * y[p]
    # Sum over (c,θ) for fixed b: g_b(c) same for all θ in c, so Σ_{c,θ} g_b(c,θ) = Σ_c g_b(c)*|bins for b in c|
    project_obj2_coeff = {}
    for p, job_list in projects_eb.items():
        w_p = job_weight.get(job_list[0], 0) if job_list else 0
        sum_g_p = 0.0
        for b in job_list:
            for c in (x[b].keys() if b in x else []):
                g_val = g_bc(b, c)
                num_bins = len(eb_temp.get(b, {}).get(c, {}))
                sum_g_p += g_val * num_bins
        n_p = len(job_list) if job_list else 1
        project_obj2_coeff[p] = (w_p / zeta2) * (1.0 / n_p) * sum_g_p if zeta2 > 0 and n_p > 0 else 0.0
    obj2 = quicksum(project_obj2_coeff.get(p, 0) * y[p] for p in y)

    # obj3(ψ): utilization = (1/ζ3) Σ ψ_{b,c,θ} g_b(c,θ) ℓ_b
    obj3_raw = quicksum(
        x[j][i][t] * g_bc(j, i) * bins_eb.get(j, 0)
        for j in x for i in x[j] for t in x[j][i]
    )
    obj3 = obj3_raw / zeta3 if zeta3 > 0 else 0

    # obj4(ψ): executive balance = - Σ_i (e_i - (1/ζ4) Σ ψ τ_{b,i} g_b ℓ_b)^2
    # Expected time per exec: fraction_o = (1/ζ4) Σ x[j][i][t] * g(j,i) * ex_bal[j][o] * bins_eb[j]
    # shortfall_o >= e_o - fraction_o, add -alpha4 * Σ shortfall_o^2
    time_per_exec = {}
    for organization in adjusted_org_p.keys():
        time_per_exec[organization] = quicksum(
            x[j][i][t] * ex_bal.get(j, {}).get(organization, 0) * bins_eb.get(j, 0)
            for j in x for i in x[j] for t in x[j][i]
        )
    shortfall = {o: m.addVar(name=f"shortfall_{o}", lb=-GRB.INFINITY) for o in adjusted_org_p.keys()}
    m.update()
    for o in adjusted_org_p.keys():
        e_o = adjusted_org_p[o]
        fraction_o = time_per_exec[o] / zeta4 if zeta4 > 0 else 0
        m.addConstr(shortfall[o] == e_o - fraction_o, name=f"shortfall_{o}")
    
    # Choose between quadratic or piecewise linear EB objective
    print(f"\n  Using EB objective type: {eb_objective_type}")
    
    if eb_objective_type == "quadratic":
        # Original quadratic objective: -Σ shortfall²
        obj4_penalty = quicksum(shortfall[o] * shortfall[o] for o in adjusted_org_p.keys())
        obj4 = -obj4_penalty  # paper: obj4 is negative (squared) penalty
    
    elif eb_objective_type == "piecewise_linear":
        # Piecewise linear approximation of shortfall²
        # We approximate x² using tangent lines at breakpoints.
        # Since x² is convex, z ≥ x² can be modeled as: z ≥ 2*a*x - a² for each breakpoint a.
        # For shortfall which can be positive or negative, we use |shortfall|² approximation.
        
        # First, create absolute value variables: abs_shortfall = |shortfall|
        # Using: abs_shortfall = s_pos + s_neg, shortfall = s_pos - s_neg, s_pos,s_neg >= 0
        abs_shortfall = {}
        shortfall_pos = {}
        shortfall_neg = {}
        
        for o in adjusted_org_p.keys():
            shortfall_pos[o] = m.addVar(name=f"shortfall_pos_{o}", lb=0)
            shortfall_neg[o] = m.addVar(name=f"shortfall_neg_{o}", lb=0)
            abs_shortfall[o] = m.addVar(name=f"abs_shortfall_{o}", lb=0)
        
        m.update()
        
        for o in adjusted_org_p.keys():
            # shortfall = s_pos - s_neg
            m.addConstr(shortfall[o] == shortfall_pos[o] - shortfall_neg[o], name=f"shortfall_split_{o}")
            # abs_shortfall = s_pos + s_neg
            m.addConstr(abs_shortfall[o] == shortfall_pos[o] + shortfall_neg[o], name=f"abs_shortfall_{o}")
        
        # Now approximate abs_shortfall² using piecewise linear function
        # Breakpoints for the approximation (covers range 0 to 0.5, which is max possible deviation)
        breakpoints = [0.0, 0.002, 0.005, 0.05, 0.10, 0.40]
        
        # For each executive, create auxiliary variable z_o that approximates abs_shortfall²
        z_squared_approx = {}
        for o in adjusted_org_p.keys():
            z_squared_approx[o] = m.addVar(name=f"z_squared_{o}", lb=0)
        
        m.update()
        
        # Add tangent constraints: z ≥ 2*a*x - a² for each breakpoint a
        # This gives a lower bound on z that approaches x² as breakpoints get denser
        for o in adjusted_org_p.keys():
            for a in breakpoints:
                if a > 0:  # Skip a=0 which gives z ≥ 0 (already enforced by lb=0)
                    # Tangent to x² at x=a: y = 2ax - a²
                    m.addConstr(z_squared_approx[o] >= 2 * a * abs_shortfall[o] - a * a,
                               name=f"tangent_{o}_{a}")
        
        obj4_penalty = quicksum(z_squared_approx[o] for o in adjusted_org_p.keys())
        obj4 = -obj4_penalty  # Negative because we want to minimize penalty
        
    else:
        raise ValueError(f"Unknown eb_objective_type: {eb_objective_type}. Use 'quadratic' or 'piecewise_linear'.")

    # === DEBUG: Print target summary for verification ===
    print("\n" + "=" * 80)
    print("DEBUG: STRATEGIC OBJECTIVE (Section 4.1) — Executive balance targets")
    print("=" * 80)
    for o in adjusted_org_p.keys():
        print(f"  {o}: e_o = {adjusted_org_p[o]*100:.2f}%")
    print(f"  ζ1={zeta1:.1f}, ζ2={zeta2:.1f}, ζ3=ζ4={zeta3}")
    print("=" * 80 + "\n")

    lp_config_bin_spread_obj = 0
    lp_sb_slot_spread_obj = 0
    if use_lp_relaxation:
        for (c, bin_t), vars_for_pair in feasible_config_bin_to_vars.items():
            mass_var = config_bin_mass[(c, bin_t)]
            support_var = config_bin_support[(c, bin_t)]
            m.addConstr(
                mass_var == quicksum(vars_for_pair),
                name=f"mass_config_bin[{c},{bin_t}]"
            )
            m.addConstr(
                support_var <= mass_var / lp_spread_epsilon,
                name=f"support_config_bin[{c},{bin_t}]"
            )

        for j, slots in feasible_slots_by_job.items():
            for c, bin_t, var in slots:
                m.addConstr(
                    sb_slot_support[(j, c, bin_t)] <= var / lp_spread_epsilon,
                    name=f"support_sb_slot[{j},{c},{bin_t}]"
                )

        feasible_config_bin_count = len(feasible_config_bin_to_vars)
        feasible_job_count = len(feasible_slots_by_job)
        lp_config_bin_spread_obj = (
            quicksum(config_bin_support.values()) / feasible_config_bin_count
            if feasible_config_bin_count > 0 else 0
        )
        lp_sb_slot_spread_obj = (
            quicksum(
                (
                    quicksum(sb_slot_support[(j, c, bin_t)] for c, bin_t, _ in slots)
                    / max(1, len(slots))
                )
                for j, slots in feasible_slots_by_job.items()
            ) / feasible_job_count
            if feasible_job_count > 0 else 0
        )

        print("\n" + "=" * 80)
        print("DEBUG: LP SPREAD REGULARIZATION")
        print("=" * 80)
        print(f"  epsilon threshold: {lp_spread_epsilon:.6f}")
        print(f"  config-bin spread weight: {lp_config_bin_spread_weight:.6f}")
        print(f"  sb-slot spread weight: {lp_sb_slot_spread_weight:.6f}")
        print(f"  feasible config-bin pairs: {feasible_config_bin_count}")
        print(f"  jobs with feasible slots: {feasible_job_count}")
        print("=" * 80 + "\n")

    # --- 3. Combined objective: obj(ψ) = Σ_k α_k obj_k(ψ) ---
    m.setObjective(
        alpha1 * obj1
        + alpha2 * obj2
        + alpha3 * obj3
        + alpha4 * obj4
        + lp_config_bin_spread_weight * lp_config_bin_spread_obj
        + lp_sb_slot_spread_weight * lp_sb_slot_spread_obj,
        GRB.MAXIMIZE
    )

    # --- 4. Define Constraints (unchanged from your model) ---
    print("Adding model constraints...")
    
    # === DEBUG: Print info about the Assign constraint (each job can only be scheduled once across ALL configs) ===
    print("\n" + "=" * 80)
    print("DEBUG: ASSIGN CONSTRAINT INFO")
    print("=" * 80)
    jobs_with_assign_constraint = len(sb_value.keys())
    jobs_with_variables = len(x)
    print(f"  Jobs in sb_value (with Assign constraint): {jobs_with_assign_constraint}")
    print(f"  Jobs with variables created (in x): {jobs_with_variables}")
    print(f"  NOTE: Assign constraint limits each job to AT MOST 1 assignment across ALL configurations")
    print("=" * 80 + "\n")
    
    # Assign constraint
    for j in sb_value.keys():
        m.addConstr(quicksum(x[j][i][t] for i in x[j] for t in x[j][i]) <= 1, name=f'Assign[{j}]')
    
    # Constraint: Prevent assignment when FRACTION_OBS is 0
    # If pref_conf[c][s] == 0 or NaN, then x[j][i][t] == 0 for all bins t in that configuration
    fraction_obs_exclusions_per_config = {c: 0 for c in config}
    fraction_obs_excluded_jobs_per_config = {c: set() for c in config}
    for j in eb_temp.keys():
        for i in eb_temp[j].keys():
            # Check if FRACTION_OBS is 0 or NaN for this job-configuration combination
            if i in pref_conf.columns and j in pref_conf.index:
                pref_val = pref_conf[i][j]
                # If FRACTION_OBS is 0 or NaN, prevent assignment
                if np.isnan(pref_val) or pref_val == 0:
                    if j in x and i in x[j]:
                        fraction_obs_exclusions_per_config[i] += len(x[j][i])
                        fraction_obs_excluded_jobs_per_config[i].add(j)
                        for bin_t in x[j][i].keys():
                            m.addConstr(x[j][i][bin_t] == 0, name=f'NoAssign_FRACTION_OBS_0[{j},{i},{bin_t}]')

    # === DEBUG: Print FRACTION_OBS exclusions per configuration ===
    print("\n" + "=" * 80)
    print("DEBUG: FRACTION_OBS EXCLUSIONS PER CONFIGURATION")
    print("=" * 80)
    for c in config:
        print(f"  {c}: {len(fraction_obs_excluded_jobs_per_config[c])} jobs excluded ({fraction_obs_exclusions_per_config[c]} variables set to 0)")
    print("=" * 80 + "\n")

    # Time constraint
    # Helper function to check if bin t falls within a job's execution window (with wraparound)
    def bin_in_job_window(t, t_prime, duration, num_bins=48):
        """Check if bin t is occupied by a job starting at t_prime with given duration.
        Handles wraparound for LST bins (cyclic over 24 hours = 48 half-hour bins)."""
        for offset in range(duration):
            if (t_prime + offset) % num_bins == t:
                return True
        return False
    
    for c in config:
        for t in Tti[c]:
            lexp = quicksum(x[s][c][t_prime] for s in eb_temp if c in eb_temp.get(s, {}) for t_prime in eb_temp[s][c] if
                            bin_in_job_window(t, t_prime, bins_eb.get(s, 0)))
            m.addConstr(lexp <= Tti[c][t], name=f'time_bin[{t},{c}]')
            # Note: Removed flexibility term `(Tti_mod[c][t]*extra_duration[c])` for simplicity. Add it back if needed.

    # Project completion constraint
    for p in projects_eb.keys():
        for j in projects_eb[p]:
            if j in x:
                m.addConstr(y[p] <= quicksum(x[j][i][t] for i in x[j] for t in x[j][i]),
                            name=f'All_sb_per_proj[{p}]_and_eb[{j}]')

    m.update()

    # --- 5. Optimize ---
    print("Optimizing model...")
    if not use_lp_relaxation:
        m.Params.MIPGap = 0.1
    m.Params.OutputFlag = 1
    m.Params.TimeLimit = time_limit # Optional: set a time limit
    m.optimize()

    print(f"Model status after solving: {m.status}")

    # --- 6. Save Results ---
    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        print("Optimization finished. Saving results...")
        
        # === DEBUG: OBJECTIVE COMPONENT BREAKDOWN ===
        print("\n" + "=" * 80)
        print("DEBUG: OBJECTIVE COMPONENT BREAKDOWN (Strategic Schedule)")
        print("=" * 80)
        
        # Compute actual objective component values
        # obj1: weighted SB completion
        obj1_val = sum(
            x[j][i][t].x * g_bc(j, i) * job_weight.get(j, 0)
            for j in x for i in x[j] for t in x[j][i]
        ) / zeta1 if zeta1 > 0 else 0
        
        # obj2: weighted project completion
        obj2_val = sum(
            project_obj2_coeff.get(p, 0) * y[p].x for p in y
        )
        
        # obj3: utilization
        obj3_val = sum(
            x[j][i][t].x * g_bc(j, i) * bins_eb.get(j, 0)
            for j in x for i in x[j] for t in x[j][i]
        ) / zeta3 if zeta3 > 0 else 0
        
        # obj4: EB penalty (negative)
        obj4_val = -sum(shortfall[o].x ** 2 for o in shortfall)
        
        print(f"\n  === Raw Objective Values (before alpha weighting) ===")
        print(f"    obj1 (SB completion):      {obj1_val:>12.6f}  (range: 0 to 1)")
        print(f"    obj2 (Project completion): {obj2_val:>12.6f}  (range: 0 to ~1)")
        print(f"    obj3 (Utilization):        {obj3_val:>12.6f}  (range: 0 to 1)")
        print(f"    obj4 (EB penalty):         {obj4_val:>12.6f}  (range: -4 to 0)")
        
        print(f"\n  === Alpha Weights ===")
        print(f"    α1 = {alpha1}")
        print(f"    α2 = {alpha2}")
        print(f"    α3 = {alpha3}")
        print(f"    α4 = {alpha4}")
        
        print(f"\n  === Weighted Contributions (α_k × obj_k) ===")
        contrib1 = alpha1 * obj1_val
        contrib2 = alpha2 * obj2_val
        contrib3 = alpha3 * obj3_val
        contrib4 = alpha4 * obj4_val
        total_contrib = contrib1 + contrib2 + contrib3 + contrib4
        
        print(f"    α1 × obj1 = {alpha1} × {obj1_val:.6f} = {contrib1:>12.6f}")
        print(f"    α2 × obj2 = {alpha2} × {obj2_val:.6f} = {contrib2:>12.6f}")
        print(f"    α3 × obj3 = {alpha3} × {obj3_val:.6f} = {contrib3:>12.6f}")
        print(f"    α4 × obj4 = {alpha4} × {obj4_val:.6f} = {contrib4:>12.6f}")
        print(f"    {'-'*50}")
        print(f"    Total (computed):  {total_contrib:>12.6f}")
        print(f"    Gurobi objVal:     {m.objVal:>12.6f}")
        
        # Show relative contributions
        print(f"\n  === Relative Contribution Analysis ===")
        abs_contribs = [abs(contrib1), abs(contrib2), abs(contrib3), abs(contrib4)]
        total_abs = sum(abs_contribs)
        if total_abs > 0:
            print(f"    obj1 contributes: {abs(contrib1)/total_abs*100:>6.2f}% of total magnitude")
            print(f"    obj2 contributes: {abs(contrib2)/total_abs*100:>6.2f}% of total magnitude")
            print(f"    obj3 contributes: {abs(contrib3)/total_abs*100:>6.2f}% of total magnitude")
            print(f"    obj4 contributes: {abs(contrib4)/total_abs*100:>6.2f}% of total magnitude")
        
        # Identify dominant objective
        max_contrib_idx = abs_contribs.index(max(abs_contribs))
        obj_names = ['obj1 (SB completion)', 'obj2 (Project completion)', 'obj3 (Utilization)', 'obj4 (EB penalty)']
        print(f"\n  *** DOMINANT OBJECTIVE: {obj_names[max_contrib_idx]} ***")
        print(f"  The optimizer is primarily optimizing for this objective.")
        print("=" * 80 + "\n")
        
        # === DEBUG: Summary of assignments per configuration ===
        print("\n" + "=" * 80)
        print("DEBUG: ASSIGNMENTS PER CONFIGURATION (POST-OPTIMIZATION)")
        print("=" * 80)
        assignments_per_config = {c: 0 for c in config}
        jobs_assigned_per_config = {c: set() for c in config}
        total_bins_used_per_config = {c: 0 for c in config}
        for j in x:
            for c in x[j]:
                for t in x[j][c]:
                    if x[j][c][t].x > 0.5:
                        assignments_per_config[c] += 1
                        jobs_assigned_per_config[c].add(j)
                        total_bins_used_per_config[c] += bins_eb[j]
        for c in config:
            avail_bins = sum(Tti[c].values())
            print(f"  {c}: {assignments_per_config[c]} assignments ({len(jobs_assigned_per_config[c])} unique jobs)")
            print(f"       Time used: {total_bins_used_per_config[c]} bins / {avail_bins} available ({100*total_bins_used_per_config[c]/avail_bins if avail_bins > 0 else 0:.1f}%)")
        print(f"  TOTAL: {sum(assignments_per_config.values())} assignments, {sum(total_bins_used_per_config.values())} bins used")
        print("=" * 80 + "\n")
        
        # === DEBUG: EXECUTIVE BALANCE VERIFICATION ===
        # Calculate the executive time in the new schedule and compare to targets
        print("\n" + "=" * 80)
        print("DEBUG: EXECUTIVE BALANCE VERIFICATION (POST-OPTIMIZATION)")
        print("=" * 80)
        
        scheduled_exec_time = {o: 0.0 for o in adjusted_org_p.keys()}
        for j in x:
            for c in x[j]:
                for t in x[j][c]:
                    if x[j][c][t].x > 0.5:
                        job_bins = bins_eb.get(j, 0)
                        for org in adjusted_org_p.keys():
                            if org in ex_bal.get(j, {}):
                                scheduled_exec_time[org] += ex_bal[j][org] * job_bins
        
        total_scheduled_time = sum(scheduled_exec_time.values())
        print(f"\n  Time scheduled in this strategic schedule:")
        for org in adjusted_org_p.keys():
            achieved_frac = scheduled_exec_time[org] / remaining_bins if remaining_bins > 0 else 0
            target_frac = adjusted_org_p[org]
            diff = achieved_frac - target_frac
            status = "OK" if abs(diff) < 0.02 else ("OVER" if diff > 0 else "UNDER")
            print(f"    {org}: {scheduled_exec_time[org]:.1f} bins ({achieved_frac*100:.2f}%) vs target {target_frac*100:.2f}% [{status}]")
        print(f"    Total: {total_scheduled_time:.1f} bins")
        
        # Show projected overall executive balance (completed + scheduled)
        print(f"\n  Projected overall executive balance (completed + this schedule):")
        for org in adjusted_org_p.keys():
            completed = actual_exec_time_used.get(org, 0)
            scheduled = scheduled_exec_time[org]
            total_for_org = completed + scheduled
            overall_frac = total_for_org / total_year_bins if total_year_bins > 0 else 0
            original_target = org_p[org]
            diff = overall_frac - original_target
            status = "OK" if abs(diff) < 0.02 else ("OVER" if diff > 0 else "UNDER")
            print(f"    {org}: {completed:.1f} + {scheduled:.1f} = {total_for_org:.1f} bins ({overall_frac*100:.2f}%) vs original target {original_target*100:.2f}% [{status}]")
        
        
        # === CRITICAL DEBUG: Shortfall and fraction_o analysis ===
        print("\n" + "=" * 80)
        print("DEBUG: SHORTFALL AND FRACTION_O ANALYSIS (EB Penalty Internals)")
        print("=" * 80)
        print(f"  zeta4 (normalization denominator) = {zeta4}")
        print(f"  remaining_bins = {remaining_bins}")
        print(f"\n  {'Exec':<6} | {'Sched Time':<12} | {'fraction_o':<12} | {'e_o (target)':<12} | {'shortfall':<12} | {'shortfall²':<12}")
        print(f"  {'-'*6} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12}")
        _total_shortfall_sq = 0.0
        for org in adjusted_org_p.keys():
            _sched_time = scheduled_exec_time.get(org, 0)
            _fraction_o = _sched_time / zeta4 if zeta4 > 0 else 0
            _e_o = adjusted_org_p[org]
            _shortfall_val = shortfall[org].x
            _shortfall_sq = _shortfall_val ** 2
            _total_shortfall_sq += _shortfall_sq
            print(f"  {org:<6} | {_sched_time:>12.1f} | {_fraction_o:>12.4f} | {_e_o:>12.4f} | {_shortfall_val:>12.4f} | {_shortfall_sq:>12.6f}")
        print(f"  {'-'*6} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12}")
        print(f"  {'SUM':<6} | {total_scheduled_time:>12.1f} | {'':<12} | {'':<12} | {'':<12} | {_total_shortfall_sq:>12.6f}")
        print(f"\n  obj4 (EB penalty) = -Σ(shortfall²) = -{_total_shortfall_sq:.6f}")
        print(f"  α4 × obj4 = {alpha4} × (-{_total_shortfall_sq:.6f}) = {alpha4 * (-_total_shortfall_sq):.6f}")
        print(f"\n  Gurobi objective value: {m.objVal:.6f}")
        print("=" * 80 + "\n")
        
        # === DIAGNOSTIC: Why is EB being violated? ===
        print("\n" + "=" * 80)
        print("DEBUG: EB VIOLATION ANALYSIS - Why did optimizer choose these jobs?")
        print("=" * 80)
        
        # Categorize scheduled jobs by primary executive
        scheduled_jobs_by_exec = {'NA': [], 'EA': [], 'EU': [], 'CL': []}
        for j in x:
            for c in x[j]:
                for t in x[j][c]:
                    if x[j][c][t].x > 0.5:
                        job_bins = bins_eb.get(j, 0)
                        job_val = job_weight.get(j, 0)
                        # Find primary executive
                        primary_exec = max(ex_bal.get(j, {'NA': 0}), key=lambda k: ex_bal.get(j, {}).get(k, 0))
                        scheduled_jobs_by_exec[primary_exec].append({
                            'job_id': j,
                            'bins': job_bins,
                            'value': job_val,
                            'exec_fracs': ex_bal.get(j, {})
                        })
                        break  # Only count once per job
        
        # Compare average job value by executive
        print("\n  Scheduled jobs analysis by primary executive:")
        print(f"  {'Exec':<6} | {'Jobs':<6} | {'Avg Value':<12} | {'Avg Bins':<10} | {'Total Bins':<12} | {'Total Value':<12}")
        print(f"  {'-'*6} | {'-'*6} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12}")
        for org in ['NA', 'EA', 'EU', 'CL']:
            jobs = scheduled_jobs_by_exec[org]
            if jobs:
                avg_val = sum(j['value'] for j in jobs) / len(jobs)
                avg_bins = sum(j['bins'] for j in jobs) / len(jobs)
                total_bins = sum(j['bins'] for j in jobs)
                total_val = sum(j['value'] for j in jobs)
            else:
                avg_val = avg_bins = total_bins = total_val = 0
            print(f"  {org:<6} | {len(jobs):<6} | {avg_val:>12.2f} | {avg_bins:>10.1f} | {total_bins:>12.1f} | {total_val:>12.2f}")
        
        # Check if high-value jobs are skewed toward one executive
        print("\n  INSIGHT: High-value jobs may be concentrated in one executive.")
        print("  If EA jobs have much higher average value, the optimizer is trading off")
        print("  EB compliance for SB/project completion objectives.")
        print(f"\n  Current objective weights:")
        print(f"    α1 (SB completion): {alpha1}")
        print(f"    α2 (Project completion): {alpha2}")
        print(f"    α3 (Utilization): {alpha3}")
        print(f"    α4 (EB penalty): {alpha4}")
        
        # Calculate approximate penalty
        max_shortfall = max(abs(achieved_frac - adjusted_org_p[org]) 
                           for org, achieved_frac in [(o, scheduled_exec_time[o] / remaining_bins if remaining_bins > 0 else 0) 
                                                       for o in adjusted_org_p.keys()])
        approx_penalty = sum((achieved_frac - adjusted_org_p[org])**2 
                            for org, achieved_frac in [(o, scheduled_exec_time[o] / remaining_bins if remaining_bins > 0 else 0) 
                                                        for o in adjusted_org_p.keys()])
        print(f"\n  Approximate EB penalty magnitude: {approx_penalty:.6f}")
        print(f"  α4 × penalty = {alpha4} × {approx_penalty:.6f} = {alpha4 * approx_penalty:.6f}")
        print(f"\n  *** If this value is small compared to α1, α2, α3, the optimizer will")
        print(f"  *** ignore EB constraints in favor of other objectives!")
        print(f"  *** Consider increasing 'eb_penalty' weight or adding hard EB constraints.")
        print("=" * 80 + "\n")
        
        # --- Appendix A: Log expected V̂^(c) per configuration ---
        print("\n" + "=" * 90)
        print("APPENDIX A: STRATEGIC SCHEDULE V̂^(c) PER CONFIGURATION")
        print("=" * 90)
        print("  This shows the expected time by executive for each configuration in the strategic schedule.")
        print("  These values will be used to compute ê^(c) targets at the operational level.")
        
        V_hat_per_config = {}
        for c in config:
            V_hat_per_config[c] = {o: 0.0 for o in adjusted_org_p.keys()}
        
        for j in x:
            for c in x[j]:
                for t in x[j][c]:
                    if x[j][c][t].x > 0.5:
                        job_bins = bins_eb.get(j, 0)
                        for org in adjusted_org_p.keys():
                            if org in ex_bal.get(j, {}):
                                V_hat_per_config[c][org] += ex_bal[j][org] * job_bins
        
        print(f"\n  {'Configuration':<15} | {'NA':>10} | {'EA':>10} | {'EU':>10} | {'CL':>10} | {'Total':>12}")
        print(f"  {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")
        
        total_by_exec = {o: 0.0 for o in adjusted_org_p.keys()}
        for c in config:
            total_c = sum(V_hat_per_config[c].values())
            for org in adjusted_org_p.keys():
                total_by_exec[org] += V_hat_per_config[c][org]
            print(f"  {c:<15} | {V_hat_per_config[c].get('NA', 0):>10.1f} | {V_hat_per_config[c].get('EA', 0):>10.1f} | {V_hat_per_config[c].get('EU', 0):>10.1f} | {V_hat_per_config[c].get('CL', 0):>10.1f} | {total_c:>12.1f}")
        
        print(f"  {'-'*15} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12}")
        total_all = sum(total_by_exec.values())
        print(f"  {'TOTAL':<15} | {total_by_exec.get('NA', 0):>10.1f} | {total_by_exec.get('EA', 0):>10.1f} | {total_by_exec.get('EU', 0):>10.1f} | {total_by_exec.get('CL', 0):>10.1f} | {total_all:>12.1f}")
        
        # Show EB fractions for the scheduled work
        print(f"\n  Executive balance fractions in strategic schedule:")
        if total_all > 0:
            for org in ['NA', 'EA', 'EU', 'CL']:
                frac = total_by_exec.get(org, 0) / total_all
                target = org_p.get(org, 0)
                diff = frac - target
                status = "OK" if abs(diff) < 0.02 else ("OVER" if diff > 0 else "UNDER")
                print(f"    {org}: {frac*100:.2f}% (target: {target*100:.2f}%) [{status}]")
        
        print("=" * 90 + "\n")
        
        # --- DETAILED DEBUGGING: Analyze Gurobi Solution ---
        print("\n" + "=" * 80)
        print("DETAILED GUROBI SOLUTION ANALYSIS")
        print("=" * 80)
        
        # Get all assigned variables with their values
        assignment_threshold = 1e-9 if use_lp_relaxation else 0.5
        assignment_label = "x[j,i,t] > 0" if use_lp_relaxation else "x[j,i,t] > 0.5"
        parsed_assignments, parsing_errors = _parse_x_assignments_from_model(m, threshold=assignment_threshold)
        assigned_vars_with_values = [(entry['var_name'], entry['value']) for entry in parsed_assignments]
        print(f"\nTotal assigned variables ({assignment_label}): {len(assigned_vars_with_values)}")

        # Load the mapping file - look in output_dir first, then data_dir, then current directory
        df_match_eb_sbuid = _load_match_eb_mapping(output_dir=output_dir, data_dir=data_dir)
        eb_to_sbuid = pd.Series(df_match_eb_sbuid.SB_UID.values, index=df_match_eb_sbuid.eb.astype(str)).to_dict()
        
        # Also create reverse mapping: SB_UID -> list of execution blocks
        sbuid_to_ebs = {}
        for eb, sb_uid in eb_to_sbuid.items():
            if sb_uid not in sbuid_to_ebs:
                sbuid_to_ebs[sb_uid] = []
            sbuid_to_ebs[sb_uid].append(eb)
        
        # Get required executions per SB_UID
        # After duplication and filtering, df_match_eb_sbuid contains all execution blocks
        # Count how many execution blocks exist for each SB_UID (this is the required number)
        sbuid_required_execs = df_match_eb_sbuid.groupby('SB_UID').size().to_dict()
        
        # Parse assigned variables and group by execution block
        eb_assignments = {}  # eb -> list of (config, time_bin, value)
        for entry in parsed_assignments:
            eb = entry['eb']
            conf = entry['configuration']
            time_bin = entry['lst_bin']
            var_value = entry['value']
            if eb not in eb_assignments:
                eb_assignments[eb] = []
            eb_assignments[eb].append((conf, time_bin, var_value))
        
        if parsing_errors:
            print(f"\nWARNING: {len(parsing_errors)} parsing errors encountered:")
            for error in parsing_errors[:10]:  # Show first 10
                print(f"  {error}")
            if len(parsing_errors) > 10:
                print(f"  ... and {len(parsing_errors) - 10} more")
        
        # Debug: Show sample variable names
        if assigned_vars_with_values:
            print(f"\n--- SAMPLE VARIABLE NAMES (First 5) ---")
            for var_name, var_value in assigned_vars_with_values[:5]:
                print(f"  {var_name} = {var_value:.3f}")
        
        print(f"\nTotal execution blocks assigned: {len(eb_assignments)}")

        if use_lp_relaxation:
            print(f"\n--- LP SPREAD DIAGNOSTICS ---")
            config_bin_mass_values = {
                key: float(var.X) for key, var in config_bin_mass.items()
            }
            used_config_bin_mass_values = {
                key: mass for key, mass in config_bin_mass_values.items()
                if mass > assignment_threshold
            }
            used_config_bin_support_values = {
                key: float(config_bin_support[key].X)
                for key in used_config_bin_mass_values
            }
            print(f"Feasible config-bin pairs: {len(config_bin_mass_values)}")
            print(f"Used config-bin pairs (> {assignment_threshold:g} mass): {len(used_config_bin_mass_values)}")
            if used_config_bin_mass_values:
                used_mass_array = np.array(list(used_config_bin_mass_values.values()), dtype=float)
                print(
                    f"Mass per used config-bin pair: min={used_mass_array.min():.6f}, "
                    f"mean={used_mass_array.mean():.6f}, max={used_mass_array.max():.6f}"
                )
                used_support_array = np.array(list(used_config_bin_support_values.values()), dtype=float)
                print(
                    f"Support indicator per used config-bin pair: min={used_support_array.min():.6f}, "
                    f"mean={used_support_array.mean():.6f}, max={used_support_array.max():.6f}"
                )
                print("Sample used config-bin pairs by mass:")
                for (conf, time_bin), mass in sorted(
                    used_config_bin_mass_values.items(),
                    key=lambda item: item[1],
                    reverse=True
                )[:10]:
                    support_val = used_config_bin_support_values[(conf, time_bin)]
                    print(
                        f"  Config {conf}, LST {time_bin}: mass={mass:.6f}, support={support_val:.6f}"
                    )

            sb_slot_counts = []
            sb_slot_details = []
            for j, slots in feasible_slots_by_job.items():
                positive_slots = []
                total_mass = 0.0
                for c, bin_t, _ in slots:
                    mass = float(x[j][c][bin_t].X)
                    if mass > assignment_threshold:
                        positive_slots.append((c, bin_t, mass, float(sb_slot_support[(j, c, bin_t)].X)))
                        total_mass += mass
                if positive_slots:
                    sb_slot_counts.append(len(positive_slots))
                    sb_slot_details.append((j, len(positive_slots), len(slots), total_mass, positive_slots))

            print(f"Jobs with positive LP mass: {len(sb_slot_details)}")
            if sb_slot_counts:
                sb_slot_counts_array = np.array(sb_slot_counts, dtype=float)
                print(
                    f"Positive slot counts per job: min={sb_slot_counts_array.min():.0f}, "
                    f"mean={sb_slot_counts_array.mean():.3f}, max={sb_slot_counts_array.max():.0f}"
                )
                print("Sample jobs with widest LP support:")
                for j, positive_count, feasible_count, total_mass, positive_slots in sorted(
                    sb_slot_details,
                    key=lambda item: (item[1], item[3]),
                    reverse=True
                )[:10]:
                    print(
                        f"  EB {j}: positive_slots={positive_count}, feasible_slots={feasible_count}, "
                        f"total_mass={total_mass:.6f}"
                    )
                    for c, bin_t, mass, support_val in positive_slots[:5]:
                        print(
                            f"    - Config {c}, LST {bin_t}: mass={mass:.6f}, support={support_val:.6f}"
                        )
        
        # Group by SB_UID to see how many execution blocks were assigned per SB
        sbuid_assigned_counts = {}
        sbuid_assigned_details = {}
        for eb, assignments in eb_assignments.items():
            sb_uid = eb_to_sbuid.get(eb)
            if sb_uid:
                if sb_uid not in sbuid_assigned_counts:
                    sbuid_assigned_counts[sb_uid] = 0
                    sbuid_assigned_details[sb_uid] = []
                sbuid_assigned_counts[sb_uid] += 1
                # Store details: (eb, config, time_bin)
                for conf, time_bin, value in assignments:
                    sbuid_assigned_details[sb_uid].append((eb, conf, time_bin, value))
        
        # Print summary statistics
        print(f"\n--- SOLUTION SUMMARY ---")
        print(f"Unique SB_UIDs with assigned execution blocks: {len(sbuid_assigned_counts)}")
        print(f"Total execution blocks assigned: {sum(sbuid_assigned_counts.values())}")
        
        # Print detailed breakdown per SB_UID
        print(f"\n--- DETAILED BREAKDOWN BY SB_UID ---")
        print(f"{'SB_UID':<20} {'Required':<10} {'Assigned':<10} {'Configurations':<30} {'Status':<20}")
        print("-" * 90)
        
        # Sort by SB_UID for easier reading. Also keep track of total counts
        total_required = 0
        total_assigned = 0
        for sb_uid in sorted(sbuid_assigned_counts.keys()):
            required = sbuid_required_execs.get(sb_uid, 0)
            assigned = sbuid_assigned_counts[sb_uid]
            total_required += required
            total_assigned += assigned
            # Get unique configurations this SB was assigned to
            configs = set()
            for eb, conf, time_bin, value in sbuid_assigned_details[sb_uid]:
                configs.add(conf)
            configs_str = ', '.join(sorted(configs))
            if len(configs_str) > 28:
                configs_str = configs_str[:25] + "..."
            
            status = "COMPLETE" if assigned >= required else f"MISSING {required - assigned}"
            print(f"{sb_uid:<20} {required:<10} {assigned:<10} {configs_str:<30} {status:<20}")
        
        print(f"\n--- TOTAL COUNTS ---")
        print(f"Total required execution blocks: {total_required}")
        print(f"Total assigned execution blocks: {total_assigned}")
        
        # Print full solution details (first 20 SBs)
        print(f"\n--- FULL SOLUTION DETAILS (First 20 SB_UIDs) ---")
        count = 0
        for sb_uid in sorted(sbuid_assigned_counts.keys()):
            if count >= 20:
                break
            print(f"\nSB_UID: {sb_uid}")
            print(f"  Required executions: {sbuid_required_execs.get(sb_uid, 0)}")
            print(f"  Assigned execution blocks: {sbuid_assigned_counts[sb_uid]}")
            print(f"  Execution block details:")
            for eb, conf, time_bin, value in sbuid_assigned_details[sb_uid]:
                time_bin_str = f"{time_bin:.1f}" if time_bin is not None and not (isinstance(time_bin, float) and np.isnan(time_bin)) else "NaN"
                print(f"    - EB {eb} -> Config {conf}, Time bin {time_bin_str}, Value {value:.3f}")
            count += 1
        
        if len(sbuid_assigned_counts) > 20:
            print(f"\n... (showing first 20 of {len(sbuid_assigned_counts)} SB_UIDs)")
        
        # Now create the output file (keeping all entries, including duplicates)
        print(f"\n--- OUTPUT FILE GENERATION ---")
        results = []
        if use_lp_relaxation:
            aggregated_lp_results = defaultdict(float)
            for entry in parsed_assignments:
                sb_uid = eb_to_sbuid.get(entry['eb'])
                if not sb_uid:
                    continue
                key = (str(sb_uid), str(entry['configuration']), entry['lst_bin'])
                aggregated_lp_results[key] += float(entry['value'])

            for (sb_uid, conf, lst_bin), probability in aggregated_lp_results.items():
                results.append({
                    'SB_UID': sb_uid,
                    'Configuration': conf,
                    'LST_bin': lst_bin,
                    'Probability': probability,
                })
        else:
            for entry in parsed_assignments:
                sb_uid = eb_to_sbuid.get(entry['eb'])
                if sb_uid:
                    results.append({'SB_UID': sb_uid, 'Configuration': entry['configuration']})

        if results:
            df_output = pd.DataFrame(results)
            
            # Print what ends up in the output file
            if use_lp_relaxation:
                print(f"\nOutput file will contain {len(df_output)} fractional (SB_UID, Configuration, LST_bin) rows")
                print("(Aggregated by SB and current-bin support with Probability mass)")
            else:
                print(f"\nOutput file will contain {len(df_output)} (SB_UID, Configuration) pairs")
                print(f"(Keeping all entries to track repetition counts)")
            
            # Show which SBs appear in output file vs solution
            output_sbuids = set(df_output['SB_UID'].unique())
            solution_sbuids = set(sbuid_assigned_counts.keys())
            
            print(f"\nSB_UIDs in solution: {len(solution_sbuids)}")
            print(f"SB_UIDs in output file: {len(output_sbuids)}")
            
            # Check for discrepancies
            only_in_solution = solution_sbuids - output_sbuids
            only_in_output = output_sbuids - solution_sbuids
            
            if only_in_solution:
                print(f"WARNING: {len(only_in_solution)} SB_UIDs in solution but not in output file")
            if only_in_output:
                print(f"WARNING: {len(only_in_output)} SB_UIDs in output file but not in solution")
            
            # Show sample of output file
            print(f"\n--- SAMPLE OF OUTPUT FILE (First 10 rows) ---")
            print(df_output.head(10).to_string(index=False))
            
            # Count how many times each SB appears in output file
            sbuid_output_counts = df_output['SB_UID'].value_counts()
            print(f"\n--- SB_UID APPEARANCE IN OUTPUT FILE ---")
            print(f"SB_UIDs appearing once: {(sbuid_output_counts == 1).sum()}")
            print(f"SB_UIDs appearing multiple times: {(sbuid_output_counts > 1).sum()}")
            if (sbuid_output_counts > 1).sum() > 0:
                print(f"\nSB_UIDs with multiple configurations in output:")
                for sb_uid, count in sbuid_output_counts[sbuid_output_counts > 1].head(10).items():
                    configs = df_output[df_output['SB_UID'] == sb_uid]['Configuration'].tolist()
                    print(f"  {sb_uid}: {count} times -> {configs}")
            
            df_output.to_csv(output_path, index=False)
            print(f"\nSuccessfully saved long-term schedule to {output_path}")
        else:
            print("Warning: No SBs were scheduled. Output file will be empty.")
            # Create an empty file to avoid downstream errors
            empty_columns = ['SB_UID', 'Configuration', 'LST_bin', 'Probability'] if use_lp_relaxation else ['SB_UID', 'Configuration']
            pd.DataFrame(columns=empty_columns).to_csv(output_path, index=False)
        
        print("=" * 80 + "\n")

    else:
        print(f"Gurobi optimization failed with status: {m.status}")
    
    # Return the calendar that was constructed
    return calendar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run configurable long-term scheduler.")
    parser.add_argument("--weights", type=str, required=True, help="JSON string of objective weights.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output CSV schedule.")
    args = parser.parse_args()

    try:
        objective_weights = json.loads(args.weights)
        solve_long_term_schedule(objective_weights, args.output_path, ".")
    except Exception as e:
        print(f"An error occurred: {e}")