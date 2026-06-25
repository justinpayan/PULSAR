import pandas as pd, numpy as np
import sys
import os
import math
from pathlib import Path
import datetime
import pickle
import argparse

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DSA_BASE_DIR = SCRIPT_DIR / "DSA" / "DSA"
DEFAULT_DSA_SRC_DIR = DEFAULT_DSA_BASE_DIR / "src"
DEFAULT_DSA_LOG_DIR = DEFAULT_DSA_BASE_DIR / "logs"
DEFAULT_DATA_DIR = SCRIPT_DIR.parents[3]
TIME_INTERVAL_MINUTES = 30
ROLLING_FORECAST_OUTPUT_DIR = "dsa_sim_for_forecast_rolling"
ROLLING_FORECAST_ISSUE_EVERY_HOURS = 8
ROLLING_FORECAST_WINDOW_HOURS = 16


def _default_preprocessed_weather_path(start_year, start_month, preprocessed_root):
    cycle_year = start_year if start_month >= 10 else start_year - 1
    if preprocessed_root is None:
        return None
    return os.path.join(preprocessed_root, f"year_{cycle_year}", "realized_weather.pkl")


def _resolve_runtime_paths(args, start_year=None, start_month=None):
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    dsa_base_dir = os.path.abspath(os.path.expanduser(args.dsa_base_dir))
    dsa_src_dir = os.path.abspath(os.path.expanduser(args.dsa_src_dir or os.path.join(dsa_base_dir, "src")))
    pol_file_path = os.path.abspath(os.path.expanduser(args.pol_file_path or dsa_base_dir))
    dsa_log_dir = os.path.abspath(os.path.expanduser(args.dsa_log_dir or os.path.join(dsa_base_dir, "logs")))

    preprocessed_weather_path = args.preprocessed_weather
    if preprocessed_weather_path is None and start_year is not None and start_month is not None:
        preprocessed_weather_path = _default_preprocessed_weather_path(
            start_year=start_year,
            start_month=start_month,
            preprocessed_root=args.preprocessed_root,
        )
    if preprocessed_weather_path is not None:
        preprocessed_weather_path = os.path.abspath(os.path.expanduser(preprocessed_weather_path))

    return {
        "data_dir": data_dir,
        "dsa_base_dir": dsa_base_dir,
        "dsa_src_dir": dsa_src_dir + "/",
        "pol_file_path": pol_file_path,
        "dsa_log_dir": dsa_log_dir,
        "preprocessed_weather_path": preprocessed_weather_path,
    }


def _setup_dsa_runtime(args, start_month, start_day, include_scorers=False):
    paths = _resolve_runtime_paths(
        args,
        start_year=args.start_year,
        start_month=args.start_month,
    )

    if paths["dsa_src_dir"] not in sys.path:
        sys.path.append(paths["dsa_src_dir"])

    os.environ['DSA'] = paths["dsa_src_dir"]
    os.environ['CON_STR'] = 'placeholder'
    os.environ['POL_FILE_PATH'] = paths["pol_file_path"]

    import DsaAlgorithm as Dsa  # pyright: ignore[reportMissingImports]
    import DsaTools as DsaTool  # pyright: ignore[reportMissingImports]
    from log_config import init_loggers  # pyright: ignore[reportMissingImports]

    os.makedirs(paths["dsa_log_dir"], exist_ok=True)

    init_loggers(
        {"handlers": {
            'daily_json_logfile': {
                'level': 'DEBUG',
                'class': "logging.handlers.TimedRotatingFileHandler",
                'formatter': 'json',
                'filename': os.path.join(paths["dsa_log_dir"], f"dsa{start_month}_{start_day}.log.json"),
                'when': 'midnight',
                'interval': 1,
                'backupCount': 3
            },
            'console': {
                'level': 'INFO'
            },
            'daily_logfile': {
                'level': 'INFO',
                'filename': os.path.join(paths["dsa_log_dir"], f"dsa{start_month}_{start_day}.log"),
                'backupCount': 1
            },
        },
            "loggers": {
                "dsa": {
                    "handlers": ['daily_json_logfile'],
                    "level": 'INFO'
                },
            }})

    runtime = {"paths": paths, "Dsa": Dsa, "DsaTool": DsaTool}
    if include_scorers:
        import DsaScorers as DsaScore  # pyright: ignore[reportMissingImports]
        runtime["DsaScore"] = DsaScore
    return runtime


def _load_dashboard_context(data_path):
    dashboard_event_pair_df = pd.read_csv(os.path.join(data_path, "DASHBOARD_EVENT_PAIR.csv"))
    dashboard_event_df = pd.read_csv(os.path.join(data_path, "DASHBOARD_EVENT.csv"))
    dashboard_antenna_id_df = pd.read_csv(os.path.join(data_path, "DASHBOARD_ANTENNA.csv"))

    dashboard_event_pair_df["START_TIMESTAMP"] = pd.to_datetime(
        dashboard_event_pair_df["START_TIMESTAMP"], utc=True
    )
    dashboard_event_pair_df["END_TIMESTAMP"] = pd.to_datetime(
        dashboard_event_pair_df["END_TIMESTAMP"], utc=True
    )
    dashboard_event_df["EVENTTIME"] = pd.to_datetime(dashboard_event_df["EVENTTIME"], utc=True)

    dashboard_event_df = dashboard_event_df.merge(
        dashboard_antenna_id_df[["ID", "NAME"]],
        left_on="ANTENNA_ID",
        right_on="ID",
        suffixes=["", "_drop"],
    )
    dashboard_event_df.drop("ID_drop", axis=1, inplace=True)

    change_types_track = [
        "ArrayElementStatus", "B1Status", "B3Status", "B4Status", "B5Status", "B6Status", "B7Status",
        "B8Status", "B9Status", "B10Status", "pumping", "Pumping", "is", "ManualIntegration", "AE", "Pad",
    ]
    interesting_events_df = dashboard_event_df.query("CHANGE_TYPE in @change_types_track").copy()

    band_code_dict = {
        "ALMA_RB_01": "B1Status",
        "ALMA_RB_02": "B2Status",
        "ALMA_RB_03": "B3Status",
        "ALMA_RB_04": "B4Status",
        "ALMA_RB_05": "B5Status",
        "ALMA_RB_06": "B6Status",
        "ALMA_RB_07": "B7Status",
        "ALMA_RB_08": "B8Status",
        "ALMA_RB_09": "B9Status",
        "ALMA_RB_10": "B10Status",
    }

    return {
        "dashboard_event_pair_df": dashboard_event_pair_df,
        "interesting_events_df": interesting_events_df,
        "band_code_dict": band_code_dict,
        "da_antennas": dashboard_antenna_id_df[dashboard_antenna_id_df.NAME.str.startswith("DA")].NAME.tolist(),
        "dv_antennas": dashboard_antenna_id_df[dashboard_antenna_id_df.NAME.str.startswith("DV")].NAME.tolist(),
        "pm_antennas": dashboard_antenna_id_df[dashboard_antenna_id_df.NAME.str.startswith("PM")].NAME.tolist(),
        "cm_antennas": dashboard_antenna_id_df[dashboard_antenna_id_df.NAME.str.startswith("CM")].NAME.tolist(),
    }


def _get_available_pads(timestamp, band, dashboard_context, antenna_types=None):
    if antenna_types is None:
        antenna_types = ["da", "dv"]

    dashboard_event_pair_df = dashboard_context["dashboard_event_pair_df"]
    interesting_events_df = dashboard_context["interesting_events_df"]
    band_status_code = dashboard_context["band_code_dict"][band]

    def get_events_from_timestamp(query_timestamp, start_time_event, end_time_event):
        return start_time_event <= query_timestamp <= end_time_event

    filter_relevant_events = dashboard_event_pair_df.apply(
        lambda x: get_events_from_timestamp(timestamp, x["START_TIMESTAMP"], x["END_TIMESTAMP"]),
        axis=1,
    )
    relevant_events_for_timestamp = dashboard_event_pair_df[filter_relevant_events].START_EVENT_ID.values
    status_for_timestamp_df = interesting_events_df.query("ID in @relevant_events_for_timestamp")

    array_element_status_filter = status_for_timestamp_df.query(
        'CHANGE_TYPE == "ArrayElementStatus" and NEW_VALUE == ["O", "C"]'
    ).NAME.unique()
    pumping_status_filter = status_for_timestamp_df.query(
        'CHANGE_TYPE == ["pumping", "Pumping"] and NEW_VALUE == "no_pumping"'
    ).NAME.unique()
    integration_status_filter = status_for_timestamp_df.query(
        '(CHANGE_TYPE == ["is"] and NEW_VALUE == "R") or (EVENTTIME <= @pd.Timestamp("2025-02-08 15:35:52Z"))'
    ).NAME.unique()
    band_status_filter = status_for_timestamp_df.query(
        "CHANGE_TYPE == @band_status_code and NEW_VALUE == ['O', 'C']"
    ).NAME.unique()
    ant = status_for_timestamp_df.query(
        "NAME in @array_element_status_filter and NAME in @pumping_status_filter and "
        "NAME in @integration_status_filter and NAME in @band_status_filter"
    ).NAME.unique()

    antennas_types_selected = []
    for antenna_type in antenna_types:
        if antenna_type == "da":
            antennas_types_selected.extend(dashboard_context["da_antennas"])
        elif antenna_type == "dv":
            antennas_types_selected.extend(dashboard_context["dv_antennas"])
        elif antenna_type == "pm":
            antennas_types_selected.extend(dashboard_context["pm_antennas"])
        elif antenna_type == "cm":
            antennas_types_selected.extend(dashboard_context["cm_antennas"])

    pretend_year = 2024
    if timestamp.month >= 10:
        pretend_year = 2023
    timestamp_normalized_c10 = pd.to_datetime(
        pd.Timestamp(pretend_year, timestamp.month, timestamp.day, 0, 0, 0),
        utc=True,
    )

    filter_relevant_events_c10 = dashboard_event_pair_df.apply(
        lambda x: get_events_from_timestamp(timestamp_normalized_c10, x["START_TIMESTAMP"], x["END_TIMESTAMP"]),
        axis=1,
    )
    relevant_events_for_timestamp_c10 = dashboard_event_pair_df[filter_relevant_events_c10].START_EVENT_ID.values
    status_for_timestamp_df_c10 = interesting_events_df.query("ID in @relevant_events_for_timestamp_c10")
    pads = status_for_timestamp_df_c10.query(
        'NAME in @ant and CHANGE_TYPE == "Pad" and (NAME in @antennas_types_selected)'
    ).NEW_VALUE.unique()

    return len(pads), pads


def _initialize_dsa_ready_state(dsa12m):
    dsa12m.data.sb_status["SB_STATE"] = "Ready"
    dsa12m.data.qastatus["Observed"] = 0.0
    dsa12m.data.qastatus["Pass"] = 0.0
    dsa12m.data.qastatus["Observed2"] = 0.0
    dsa12m.data.qastatus["SemiPass"] = 0.0
    dsa12m.data.projects.PRJ_STATUS = "Ready"
    dsa12m.schedblocks.sbStatusXml = "Ready"


def _rolling_forecast_output_path(data_path, issue_time, output_dir):
    filename = f"dsa_sim_issue_{issue_time.strftime('%Y%m%d_%H%M')}_df.csv"
    return os.path.join(data_path, output_dir, filename)

def get_avail(args):
    start_month = args.start_month
    start_day = args.start_day
    start_year = args.start_year

    runtime = _setup_dsa_runtime(args, start_month, start_day)
    data_path = runtime["paths"]["data_dir"]
    Dsa = runtime["Dsa"]
    DsaTool = runtime["DsaTool"]
    output_dir = "dsa_sim"
    output_path = os.path.join(data_path, output_dir, f"dsa_sim_{start_month}_{start_day}_{start_year}_df.csv")
    if os.path.exists(output_path):
        print(f"Output file already exists, skipping: {output_path}", flush=True)
        return

    print(f"Creating {output_dir}/dsa_sim_{start_month}_{start_day}_{start_year}_df.csv", flush=True)

    os.makedirs(os.path.join(data_path, output_dir), exist_ok=True)

    data_Path = Path(data_path)
    logs = data_Path.joinpath('logs')
    logs.mkdir(exist_ok=True)


    dashboard_context = _load_dashboard_context(data_path)

    bands = ['ALMA_RB_01', 'ALMA_RB_03', 'ALMA_RB_04', 'ALMA_RB_05', 'ALMA_RB_06',
             'ALMA_RB_07', 'ALMA_RB_08', 'ALMA_RB_09', 'ALMA_RB_10']
    cycle_fix = 'c10'

    with open(f'{data_path}/data_active_{cycle_fix}.pkl', 'rb') as filein:
        data = pickle.load(filein)

    dsa12m = Dsa.DsaAlgorithm(data, 'TWELVE-M', path=data_path, aprc=False)

    dt_st = pd.Timestamp(start_year, start_month, start_day, 0, 0, 0, tz="utc")
    dt_end = dt_st + datetime.timedelta(hours=24)
    full_day_timeline = pd.to_datetime(
        pd.date_range(start=dt_st, end=dt_end, freq=f"{TIME_INTERVAL_MINUTES}min", inclusive="left")
    )

    # Set SBs to ready if the csv says so. Projects, all can be set to ready.
    _initialize_dsa_ready_state(dsa12m)

    sim_results = []

    for band in bands:
        for timest in full_day_timeline:
            print(timest)

            _, pads = _get_available_pads(timest, band, dashboard_context)
            array_info = pads.tolist()

            if isinstance(array_info, list) and len(array_info) < 2:
                print(f"Skipping {timest} band {band}: fewer than 2 antennas available ({len(array_info)})", flush=True)
                continue

            dsa12m.set_time(timest)
            dsa12m.write_ephem_coords()
            dsa12m.static_param()
            dsa12m.aggregate_dsa()

            # Save out the list of everything that can run at this time,
            # and include with them the threshold for PWV down to the 0.05
            # accuracy level.
            sel = dsa12m.selector(
                minha=-4., maxha=3.,
                array_info=array_info,
                pwv=0.0, horizon=20,
                bands=[band],
                freq_rms=1000.0, online=False,
                prj_status=("Ready", "InProgress", "PartiallyCompleted", "ObservingTimedOut"),
                sb_status=["Ready", "Running", "Phase2Submitted", "Waiting"], check=False, # ObservingTimedOut and FullyObserved should be removed
                with_tp_pads_observability=False)

            available = dsa12m.master_seldsa_df['SB_UID'].tolist()

            thresholds = {x: 0.00 for x in available}

            # TODO: Fix up so we stop recomputing stuff on things that have passed the threshold.
            # TODO: Better yet, use binary search.
            # TODO: Absolute best would be computing an analytic solution directly.
            for pwv_val in np.arange(0.00, 8.01, 0.05):
                print("pwv_val: ", pwv_val, flush=True)
                freq_idx = np.around(dsa12m.master_seldsa_df.repfreq, decimals=1)
                pwv_idx = np.full(
                    freq_idx.shape,
                    np.around(
                        int(pwv_val / 0.05) * 0.05 +
                        (0.05 if (int(pwv_val * 100) % 5) > 2 else 0.),
                        decimals=2
                    )
                )

                dsa12m.master_seldsa_df['tau'] = dsa12m.tau.reindex([freq_idx, pwv_idx]).values
                dsa12m.master_seldsa_df['tsky'] = dsa12m.tsky.reindex([freq_idx, pwv_idx]).values

                dsa12m.master_seldsa_df['airmass'] = DsaTool.calc_airmass(
                    dsa12m.master_seldsa_df.elev.values, False)
                dsa12m.master_seldsa_df['tsys'] = DsaTool.calc_tsys(
                    dsa12m.master_seldsa_df.g.values, dsa12m.master_seldsa_df.trx.values,
                    dsa12m.master_seldsa_df.tsky.values,
                    dsa12m.master_seldsa_df.tau.values,
                    dsa12m.master_seldsa_df.airmass.values
                )
                dsa12m.master_seldsa_df['tsys_ratio'] = DsaTool.calc_tsys_ratio(
                    dsa12m.master_seldsa_df.tsys.values,
                    dsa12m.master_seldsa_df.tsys_ot.values
                )
                dsa12m.master_seldsa_df['Exec. Frac'] = DsaTool.calc_exec_frac(
                    dsa12m.master_seldsa_df.bl_ratio.values,
                    dsa12m.master_seldsa_df.tsys_ratio.values
                )

                # dsa12m.master_seldsa_df['selCond'] = (
                #         dsa12m.master_seldsa_df['Exec. Frac'] >= 0.70)

                still_available = dsa12m.master_seldsa_df.query('`Exec. Frac` >= 0.70')['SB_UID'].tolist()
                for x in still_available:
                    # assert thresholds[x] >= pwv_val - 0.06 # Make sure we didnt skip a level. If thats true, it means the monotonicity assumption was wrong
                    # TODO: Ask Ignacio about this. The PWV can be TOO good and then the exec frac is set to 0 manually.
                    # TODO: Is this a convenience or is this a real constraint to be aware of?
                    thresholds[x] = pwv_val
                print("thresholds: ", thresholds, "\n\n")
            for sbuid in available:
                sim_results.append([timest, sbuid, thresholds[sbuid],
                                    dsa12m.master_seldsa_df.query('SB_UID == @sbuid').repfreq[0], band])
                print("sbuid: ", sbuid, "band: ", band, "timest: ", timest, "threshold: ", thresholds[sbuid], flush=True)
            print("\n\n", flush=True)

    res_df = pd.DataFrame(sim_results)
    if len(res_df) > 0:
        res_df.columns = [['timestamp', 'sbuid', 'pwv_thresh', 'rms_thresh', 'band']]
    else:
        res_df = pd.DataFrame(columns=['timestamp', 'sbuid', 'pwv_thresh', 'rms_thresh', 'band'])
    res_df.to_csv(os.path.join(data_path, output_dir, f"dsa_sim_{start_month}_{start_day}_{start_year}_df.csv"), index=False)


def get_avail_rolling_forecast(args):
    start_month = args.start_month
    start_day = args.start_day
    start_year = args.start_year
    start_hour = args.start_hour
    issue_every_hours = args.forecast_issue_every_hours
    forecast_window_hours = args.forecast_window_hours

    if start_hour % issue_every_hours != 0:
        raise ValueError(
            f"start_hour={start_hour} is invalid for issue cadence {issue_every_hours}h. "
            f"Expected a multiple of {issue_every_hours}."
        )

    issue_time = pd.Timestamp(start_year, start_month, start_day, start_hour, 0, 0, tz="UTC")
    runtime = _setup_dsa_runtime(args, start_month, start_day)
    data_path = runtime["paths"]["data_dir"]
    Dsa = runtime["Dsa"]
    DsaTool = runtime["DsaTool"]

    output_dir = args.forecast_output_dir
    output_path = _rolling_forecast_output_path(data_path, issue_time, output_dir)
    if os.path.exists(output_path):
        print(f"Rolling forecast file already exists, skipping: {output_path}", flush=True)
        return

    print(f"Creating rolling forecast availability for issue time {issue_time.isoformat()}", flush=True)
    print(f"Output path: {output_path}", flush=True)
    os.makedirs(os.path.join(data_path, output_dir), exist_ok=True)

    dashboard_context = _load_dashboard_context(data_path)

    bands = ['ALMA_RB_01', 'ALMA_RB_03', 'ALMA_RB_04', 'ALMA_RB_05', 'ALMA_RB_06',
             'ALMA_RB_07', 'ALMA_RB_08', 'ALMA_RB_09', 'ALMA_RB_10']
    cycle_fix = 'c10'
    with open(f'{data_path}/data_active_{cycle_fix}.pkl', 'rb') as filein:
        data = pickle.load(filein)

    dsa12m = Dsa.DsaAlgorithm(data, 'TWELVE-M', path=data_path, aprc=False)
    _initialize_dsa_ready_state(dsa12m)

    issue_pad_map = {}
    for band in bands:
        _, pads = _get_available_pads(issue_time, band, dashboard_context)
        issue_pad_map[band] = pads.tolist()

    forecast_steps = forecast_window_hours * (60 // TIME_INTERVAL_MINUTES)
    forecast_timeline = pd.to_datetime(pd.date_range(
        start=issue_time + pd.Timedelta(minutes=TIME_INTERVAL_MINUTES),
        periods=forecast_steps,
        freq=f"{TIME_INTERVAL_MINUTES}min",
    ))

    print(
        f"Issue cadence is handled externally; this issue covers {forecast_timeline[0]} "
        f"through {forecast_timeline[-1]} ({forecast_steps} bins).",
        flush=True,
    )

    sim_results = []
    for band in bands:
        array_info = issue_pad_map[band]
        if len(array_info) < 2:
            print(
                f"Skipping issue {issue_time} band {band}: fewer than 2 antennas available "
                f"at issuance time ({len(array_info)}).",
                flush=True,
            )
            continue

        for timest in forecast_timeline:
            print(f"{issue_time} -> {timest} [{band}]", flush=True)
            dsa12m.set_time(timest)
            dsa12m.write_ephem_coords()
            dsa12m.static_param()
            dsa12m.aggregate_dsa()

            sel = dsa12m.selector(
                minha=-4., maxha=3.,
                array_info=array_info,
                pwv=0.0, horizon=20,
                bands=[band],
                freq_rms=1000.0, online=False,
                prj_status=("Ready", "InProgress", "PartiallyCompleted", "ObservingTimedOut"),
                sb_status=["Ready", "Running", "Phase2Submitted", "Waiting"], check=False,
                with_tp_pads_observability=False,
            )

            available = dsa12m.master_seldsa_df["SB_UID"].tolist()
            thresholds = {x: 0.00 for x in available}

            for pwv_val in np.arange(0.00, 8.01, 0.05):
                print("pwv_val: ", pwv_val, flush=True)
                freq_idx = np.around(dsa12m.master_seldsa_df.repfreq, decimals=1)
                pwv_idx = np.full(
                    freq_idx.shape,
                    np.around(
                        int(pwv_val / 0.05) * 0.05 +
                        (0.05 if (int(pwv_val * 100) % 5) > 2 else 0.),
                        decimals=2
                    )
                )

                dsa12m.master_seldsa_df["tau"] = dsa12m.tau.reindex([freq_idx, pwv_idx]).values
                dsa12m.master_seldsa_df["tsky"] = dsa12m.tsky.reindex([freq_idx, pwv_idx]).values
                dsa12m.master_seldsa_df["airmass"] = DsaTool.calc_airmass(
                    dsa12m.master_seldsa_df.elev.values, False)
                dsa12m.master_seldsa_df["tsys"] = DsaTool.calc_tsys(
                    dsa12m.master_seldsa_df.g.values, dsa12m.master_seldsa_df.trx.values,
                    dsa12m.master_seldsa_df.tsky.values,
                    dsa12m.master_seldsa_df.tau.values,
                    dsa12m.master_seldsa_df.airmass.values
                )
                dsa12m.master_seldsa_df["tsys_ratio"] = DsaTool.calc_tsys_ratio(
                    dsa12m.master_seldsa_df.tsys.values,
                    dsa12m.master_seldsa_df.tsys_ot.values
                )
                dsa12m.master_seldsa_df["Exec. Frac"] = DsaTool.calc_exec_frac(
                    dsa12m.master_seldsa_df.bl_ratio.values,
                    dsa12m.master_seldsa_df.tsys_ratio.values
                )

                still_available = dsa12m.master_seldsa_df.query('`Exec. Frac` >= 0.70')["SB_UID"].tolist()
                for sb_uid in still_available:
                    thresholds[sb_uid] = pwv_val

            for sbuid in available:
                sim_results.append([
                    issue_time,
                    timest,
                    sbuid,
                    thresholds[sbuid],
                    dsa12m.master_seldsa_df.query("SB_UID == @sbuid").repfreq.iloc[0],
                    band,
                ])

    res_df = pd.DataFrame(
        sim_results,
        columns=["forecast_issue_time", "timestamp", "sbuid", "pwv_thresh", "rms_thresh", "band"],
    )
    res_df.to_csv(output_path, index=False)
    print(f"Wrote {len(res_df)} forecast rows to {output_path}", flush=True)

def calc_bifurcated_array_score(sb_name, resoption, array_kind, array_ar_sb, minar, maxar):
    if array_ar_sb == np.NaN or array_ar_sb <= 0:
        return (0., 0.)

    # resoption can be "Single", "Range", "Any"
    if resoption == 'Single':
        average_ar = np.average([minar, maxar])

        if (average_ar * 0.95) <= array_ar_sb <= (average_ar * 1.05):
            return (10., 10.)
        elif (0.8 * average_ar) <= array_ar_sb <= (1.2 * average_ar):
            return (7.0, 7.0)
        elif array_ar_sb < (0.8 * average_ar) or array_ar_sb > (1.2 * average_ar):
            # Some projects can have the average_ar outside the
            # min/max ar range that OT defines with resoption = Single.
            return (3.0, 3.0)
        else:
            return (-1., -1.)
    elif resoption in ('Range', 'Any'):
        if array_ar_sb * 1.05 > maxar or array_ar_sb * 0.95 < minar:
            # give less priority to SBs that are too close to borders of the allowed AR range
            # (this case happen to SBs with resoption = Any).
            # If has previous observations, give more priority to not have too much
            # array configuration separation between observations.
            return (5, 9)
        elif minar <= array_ar_sb <= maxar:
            # If it is within range, give a conservative score.
            # If has previous observations, give more priority to not have too much
            # array configuration separation between observations.
            return (7, 9)
        else:
            return (-1., -1.)
    else:
        return (-1., -1.)

def calc_base_ha_score(ha):
    sb_ha_scorer = ((math.cos(math.radians((ha + 0.5) * 15.)) - 0.55) /
                    (1 - 0.55)) * 10.

    return max(0, sb_ha_scorer)

def get_scores(args):
    start_month = args.start_month
    start_day = args.start_day
    start_year = args.start_year

    runtime = _setup_dsa_runtime(args, start_month, start_day, include_scorers=True)
    data_path = runtime["paths"]["data_dir"]
    Dsa = runtime["Dsa"]
    DsaScore = runtime["DsaScore"]
    output_path = os.path.join(data_path, f"dsa_sim_scores_{start_month}_{start_day}_{start_year}_df.csv")
    if os.path.exists(output_path):
        print(f"Output file already exists, skipping: {output_path}", flush=True)
        return

    data_Path = Path(data_path)
    logs = data_Path.joinpath('logs')
    logs.mkdir(exist_ok=True)

    DASHBOARD_EVENT_PAIR_df = pd.read_csv(os.path.join(data_path, "DASHBOARD_EVENT_PAIR.csv"))
    DASHBOARD_EVENT_df = pd.read_csv(os.path.join(data_path, "DASHBOARD_EVENT.csv"))
    DASHBOARD_ANTENNA_ID_df = pd.read_csv(os.path.join(data_path, "DASHBOARD_ANTENNA.csv"))

    DASHBOARD_EVENT_PAIR_df["START_TIMESTAMP"] = pd.to_datetime(DASHBOARD_EVENT_PAIR_df["START_TIMESTAMP"], utc=True)
    DASHBOARD_EVENT_PAIR_df["END_TIMESTAMP"] = pd.to_datetime(DASHBOARD_EVENT_PAIR_df["END_TIMESTAMP"], utc=True)
    DASHBOARD_EVENT_df["EVENTTIME"] = pd.to_datetime(DASHBOARD_EVENT_df["EVENTTIME"], utc=True)

    DASHBOARD_EVENT_df = DASHBOARD_EVENT_df.merge(
        DASHBOARD_ANTENNA_ID_df[['ID', 'NAME']], 
        left_on="ANTENNA_ID", right_on="ID", 
        suffixes=['', '_drop']
    )

    DASHBOARD_EVENT_df.drop("ID_drop", axis=1, inplace=True)

    change_types_track = [
        'ArrayElementStatus', 'B1Status', 'B3Status', 'B4Status', 'B5Status', 'B6Status', 'B7Status',
        'B8Status', 'B9Status', 'B10Status', 'pumping', 'Pumping', 'is', 'ManualIntegration', 'AE', 'Pad',
    ]

    dv_antennas = DASHBOARD_ANTENNA_ID_df[DASHBOARD_ANTENNA_ID_df.NAME.str.startswith('DV')].NAME
    da_antennas = DASHBOARD_ANTENNA_ID_df[DASHBOARD_ANTENNA_ID_df.NAME.str.startswith('DA')].NAME
    pm_antennas = DASHBOARD_ANTENNA_ID_df[DASHBOARD_ANTENNA_ID_df.NAME.str.startswith('PM')].NAME
    cm_antennas = DASHBOARD_ANTENNA_ID_df[DASHBOARD_ANTENNA_ID_df.NAME.str.startswith('CM')].NAME

    interesting_events_df = DASHBOARD_EVENT_df.query('CHANGE_TYPE in @change_types_track').copy()

    band_code_dict = {
        'ALMA_RB_01': 'B1Status',
        'ALMA_RB_02': 'B2Status',
        'ALMA_RB_03': 'B3Status',
        'ALMA_RB_04': 'B4Status',
        'ALMA_RB_05': 'B5Status',
        'ALMA_RB_06': 'B6Status',
        'ALMA_RB_07': 'B7Status',
        'ALMA_RB_08': 'B8Status',
        'ALMA_RB_09': 'B9Status',
        'ALMA_RB_10': 'B10Status',
    }

    def get_events_from_timestamp(timestamp, star_time_event, end_time_event):
        if star_time_event <= timestamp <= end_time_event:
            return True
        else:
            return False

    def get_available_pads(timestamp, band, antenna_types=['da', 'dv']):
        band_status_code = band_code_dict[band]
        filter_relevant_events = DASHBOARD_EVENT_PAIR_df.apply(
            lambda x: get_events_from_timestamp(timestamp, x['START_TIMESTAMP'], x['END_TIMESTAMP']), axis=1
        )
        relevant_events_for_timestamp = DASHBOARD_EVENT_PAIR_df[filter_relevant_events].START_EVENT_ID.values
        
        status_for_timestamp_df = interesting_events_df.query('ID in @relevant_events_for_timestamp')

        array_element_status_filter = status_for_timestamp_df.query(
            'CHANGE_TYPE == "ArrayElementStatus" and NEW_VALUE == ["O", "C"]'
        ).NAME.unique()

        pumping_status_filter = status_for_timestamp_df.query(
            'CHANGE_TYPE == ["pumping", "Pumping"] and NEW_VALUE == "no_pumping"'
        ).NAME.unique()

        integration_status_filter = status_for_timestamp_df.query(
            '(CHANGE_TYPE == ["is"] and NEW_VALUE == "R") or (EVENTTIME <= @pd.Timestamp("2025-02-08 15:35:52Z"))'
        ).NAME.unique()
        
        band_status_filter = status_for_timestamp_df.query(
            'CHANGE_TYPE == @band_status_code and NEW_VALUE == ["O", "C"]'
        ).NAME.unique()
        
        ant = status_for_timestamp_df.query(
            'NAME in @array_element_status_filter and NAME in @pumping_status_filter and '
            'NAME in @integration_status_filter and NAME in @band_status_filter'
        ).NAME.unique()
        
        antennas_types_selected = []
        for at in antenna_types:
            if at == 'da':
                antennas_types_selected.extend(da_antennas)
            elif at == 'dv':
                antennas_types_selected.extend(dv_antennas)
            elif at == 'pm':
                antennas_types_selected.extend(pm_antennas)
            elif at == 'cm':
                antennas_types_selected.extend(cm_antennas)

        # Get the pad locations during cycle 10 specifically
        pretend_year = 2024
        if timestamp.month >= 10:
            pretend_year = 2023
        timestamp_normalized_c10 = pd.to_datetime(pd.Timestamp(pretend_year, timestamp.month, timestamp.day, 0, 0, 0), utc=True)
        
        filter_relevant_events_c10 = DASHBOARD_EVENT_PAIR_df.apply(
            lambda x: get_events_from_timestamp(timestamp_normalized_c10, x['START_TIMESTAMP'], x['END_TIMESTAMP']), axis=1
        )
        relevant_events_for_timestamp_c10 = DASHBOARD_EVENT_PAIR_df[filter_relevant_events_c10].START_EVENT_ID.values
        
        status_for_timestamp_df_c10 = interesting_events_df.query('ID in @relevant_events_for_timestamp_c10')
        
        pads = status_for_timestamp_df_c10.query(
            'NAME in @ant and CHANGE_TYPE == "Pad" and (NAME in @antennas_types_selected)').NEW_VALUE.unique()
        
        return len(pads), pads

    bands = ['ALMA_RB_01', 'ALMA_RB_03', 'ALMA_RB_04', 'ALMA_RB_05', 'ALMA_RB_06',
             'ALMA_RB_07', 'ALMA_RB_08', 'ALMA_RB_09', 'ALMA_RB_10']
    cycle_fix = 'c10'

    with open(f'{data_path}/data_active_{cycle_fix}.pkl', 'rb') as filein:
        data = pickle.load(filein)

    dsa12m = Dsa.DsaAlgorithm(data, 'TWELVE-M', path=data_path, aprc=False)

    dt_st = pd.Timestamp(start_year, start_month, start_day, 0, 0, 0, tz="utc")
    dt_end = dt_st + datetime.timedelta(hours=24)

    preprocessed_weather_path = runtime["paths"]["preprocessed_weather_path"]
    if preprocessed_weather_path is None:
        raise ValueError(
            "A preprocessed weather input is required. Pass --preprocessed_weather "
            "or set --preprocessed_root so the script can resolve year_<cycle>/realized_weather.pkl."
        )
    print(f"Loading preprocessed weather from {preprocessed_weather_path}", flush=True)
    with open(preprocessed_weather_path, "rb") as f:
        pw_data = pickle.load(f)

    realized_weather = pw_data["realized_weather"]
    idx_to_timestamp = pw_data["idx_to_timestamp"]
    weather_rows = []
    for idx, timestamp in idx_to_timestamp.items():
        timestamp = pd.to_datetime(timestamp, utc=True)
        if dt_st <= timestamp <= dt_end:
            pwv, freqrms = realized_weather[idx]
            weather_rows.append({
                "timestamp": timestamp,
                "pwv": pwv,
                "freq30bcl": freqrms,
            })

    result_df_reindex = pd.DataFrame(weather_rows, columns=["timestamp", "pwv", "freq30bcl"])
    result_df_reindex = result_df_reindex.set_index("timestamp").sort_index()
    result_df_reindex = result_df_reindex.dropna(subset=["pwv", "freq30bcl"])

    print(result_df_reindex)

    dsa12m.data.sb_status['SB_STATE'] = "Ready"
    dsa12m.data.qastatus['Observed'] = 0.0
    dsa12m.data.qastatus['Pass'] = 0.0
    dsa12m.data.qastatus['Observed2'] = 0.0
    dsa12m.data.qastatus['SemiPass'] = 0.0
    dsa12m.data.projects.PRJ_STATUS = "Ready"
    dsa12m.schedblocks.sbStatusXml = "Ready"
    
    sim_results = []

    for band in bands:
        for r in list(result_df_reindex.iterrows()):
            timest = r[0]
            pwv = r[1]['pwv']
            freqrms = r[1]['freq30bcl']
            print(timest, pwv, freqrms, flush=True)

            _, pads = get_available_pads(timest, band)
            array_info = pads.tolist()

            if isinstance(array_info, list) and len(array_info) < 2:
                print(f"Skipping {timest} band {band}: fewer than 2 antennas available ({len(array_info)})", flush=True)
                continue

            dsa12m.set_time(timest)
            dsa12m.write_ephem_coords()
            dsa12m.static_param()
            dsa12m.aggregate_dsa()

            # Check under current conditions
            sel = dsa12m.selector(
                minha=-4., maxha=3.,
                array_info=array_info,
                pwv=pwv, horizon=20,
                bands=band,
                freq_rms=freqrms, online=False,
                prj_status=("Ready", "InProgress", "PartiallyCompleted", "ObservingTimedOut"),
                sb_status=["Ready", "Running", "Phase2Submitted", "Waiting"], check=False, # ObservingTimedOut and FullyObserved should be removed
                with_tp_pads_observability=False)

            # Now compute scores
            ranks_df = dsa12m.get_ranks()
            dsa12m.master_seldsa_df = pd.merge(
                dsa12m.master_seldsa_df.reset_index(drop=True),
                ranks_df[[
                    'SB_UID',
                    'PRJ_SCIENTIFIC_RANK_NORMALIZED',
                    'PRJ_SCIENTIFIC_RANK_NORMALIZED_MIN',
                    'PRJ_SCIENTIFIC_RANK_NORMALIZED_MAX',
                ]],
                on=['SB_UID']
            ).set_index('SB_UID', drop=False)
            scorer = dsa12m.master_seldsa_df.apply(
                lambda x: DsaScore.calc_all_scores(
                    pwv, x['maxPWVC'], x['Exec. Frac'], x['sbName'], x['EXEC'],
                    x['array'], x['band'], x['ARcordec'], x['array_ar_cond'], x['resoption'], x['minAR_ot'],
                    x['maxAR_ot'], x['Observed'], x['EXECOUNT'], x['GOUS_comp_next'],
                    x['proj_comp_next'], x['PRJ_SCIENTIFIC_RANK_NORMALIZED'],
                    x['PRJ_SCIENTIFIC_RANK_NORMALIZED_MIN'], x['PRJ_SCIENTIFIC_RANK_NORMALIZED_MAX'],
                    x['DC_LETTER_GRADE'], x['OverGrade'], x['last_obs_ago'],
                    x['HA'], x['selPol'], x['selConf'],
                    x['acaStandAlone'], x['JOINT_PROPOSAL'], x['SPECIAL_PRIORITY']), axis=1)

            dsa12m.master_seldsa_df['BestConf'] = dsa12m.master_seldsa_df['nominalConf']

            if len(dsa12m.master_seldsa_df):
                fin = pd.merge(
                    dsa12m.master_seldsa_df.reset_index(drop=True),
                    scorer.reset_index(), on='SB_UID').set_index(
                    'SB_UID', drop=False).sort_values(by='Score', ascending=0)
                # fin = dsa12m.master_seldsa_df.join(scorer).sort_values(by='Score', ascending=False)

                for x in fin.iterrows():
                    # We need to go back through and calculate the array score and ha score information that depends on the prior executions.
                    x = x[1]
                    array_score_has_not_obs, array_score_has_obs = calc_bifurcated_array_score(x['sbName'], x['resoption'],
                                                                                            x['array'], x['array_ar_cond'],
                                                                                            x['minAR_ot'], x['maxAR_ot']
                                                                                            )
                    base_ha_score = calc_base_ha_score(x['HA'])

                    sim_results.append([timest, x['SB_UID'], x['conditon score'],
                                        x['science rank score'], x['cycle grade score'],
                                        array_score_has_not_obs, array_score_has_obs, base_ha_score])
                    print(sim_results[-1], flush=True)

    res_df = pd.DataFrame(sim_results)
    res_df.columns = [['timestamp', 'sbuid', 'condition_score', 'science_rank_score', 'cycle_grade_score',
                       'array_score_no', 'array_score_yes', 'base_ha_score']]
    res_df.to_csv(os.path.join(data_path, f"dsa_sim_scores_{start_month}_{start_day}_{start_year}_df.csv"), index=False)

def main(args):
    if args.function == "avail":
        get_avail(args)
    elif args.function == "avail_rolling_forecast":
        get_avail_rolling_forecast(args)
    elif args.function == "scores":
        get_scores(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_month", type=int)
    parser.add_argument("--start_day", type=int)
    parser.add_argument("--start_year", type=int)
    parser.add_argument("--start_hour", type=int, default=0)
    parser.add_argument("--function", type=str, default="avail")
    parser.add_argument("--forecast_issue_every_hours", type=int, default=ROLLING_FORECAST_ISSUE_EVERY_HOURS)
    parser.add_argument("--forecast_window_hours", type=int, default=ROLLING_FORECAST_WINDOW_HOURS)
    parser.add_argument("--forecast_output_dir", type=str, default=ROLLING_FORECAST_OUTPUT_DIR,
                        help="Directory where rolling forecast-availability CSVs will be written.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory containing AOOSP data files and where DSA outputs will be written.")
    parser.add_argument("--dsa_base_dir", type=str, default=str(DEFAULT_DSA_BASE_DIR),
                        help="Base directory for the embedded DSA checkout; used to derive POL_FILE_PATH, src, and logs.")
    parser.add_argument("--dsa_src_dir", type=str, default=None,
                        help="Override the DSA Python source directory. Defaults to <dsa_base_dir>/src.")
    parser.add_argument("--pol_file_path", type=str, default=None,
                        help="Override POL_FILE_PATH for the DSA environment. Defaults to <dsa_base_dir>.")
    parser.add_argument("--dsa_log_dir", type=str, default=None,
                        help="Directory for DSA log files. Defaults to <dsa_base_dir>/logs.")
    parser.add_argument("--preprocessed_root", type=str, default=None,
                        help="Root directory containing preprocessed/year_<cycle>/realized_weather.pkl.")
    parser.add_argument("--preprocessed_weather", type=str, default=None,
                        help="Path to the realized-weather pickle from preprocess_weather.py. "
                             "Overrides --preprocessed_root when set.")

    args = parser.parse_args()
    main(args)
