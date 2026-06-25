# PULSAR: Predictive Uncertainty-aware Lookahead Scheduling with Adaptive Rebalancing

# Overview

This repository contains the code for the paper "Uncertainty-aware Ground-based Telescope Observation
Scheduling at ALMA", which introduces the PULSAR algorithm. The code in the repository is developed for use at ALMA. We hope to partner with other
observatories, and have worked to make this code easily adaptable beyond ALMA (more on that
below).

The repository serves two purposes: it lets you reproduce the experiments from the paper, and it
provides code to deploy PULSAR in practice.

The ALMA data is not provided, but we provide detailed explanations of the required data and its formatting.

Concretely, the code in this repository includes:

- Ability to simulate a full cycle year run of PULSAR and of the dispatch-based scheduler currently used at ALMA. We run these simulations using historical weather and array data.
- Single time-step scheduling: rather than simulating a whole cycle, you can run PULSAR or ALMA's current
  dispatcher for one 30-minute decision at the command line, producing the recommended next scheduling
  block from the current observatory state. 
- A "Prophet" oracle, which looks ahead at the realized conditions for the entire year and solves for an
  optimal full-year schedule. This is not a deployable algorithm; it provides an informative upper bound
  against which the operational schedulers can be measured.
- Helper code that builds forecasting models and precomputes rolling weather forecasts used throughout
  the yearly simulation.
- Data-packaging utilities that prepare availability and score inputs, compile per-cycle pressure
  statistics, and preprocess raw weather data into the formats the simulators expect.

All of this is structured to work with ALMA's data and scheduling constraints. That said, we have
attempted to isolate ALMA-specific code as much as possible to make it straightforward to adapt PULSAR
to other observatories. If you would like to work with us to see whether PULSAR fits your context,
please email Justin Payan at jpayan@andrew.cmu.edu.

# Table of Contents

The rest of this README walks through the workflow in order: first install the environment, then
prepare the input data. You can then run full-year simulations, or run a single scheduling time step
at the command line.

- [Installation](#installation)
- [Environment configuration](#environment-configuration)
  - [`base_env.sh` variables](#base_envsh-variables)
  - [`dsa_calls_env.sh` variables](#dsa_calls_envsh-variables)
- [Preparing the inputs](#preparing-the-inputs)
  - [Key concepts](#key-concepts)
  - [Inputs and scripts](#inputs-and-scripts)
  - [Static cycle reference data](#static-cycle-reference-data)
  - [`run_preprocess_weather_forecasts.sh` — realized weather and forecasts](#run_preprocess_weather_forecastssh--realized-weather-and-forecasts)
  - [`run_dsa_avail.sh` — daily DSA availability](#run_dsa_availsh--daily-dsa-availability)
  - [`run_dsa_scores.sh` — daily DSA scores](#run_dsa_scoressh--daily-dsa-scores)
  - [`run_dsa_rolling_forecast_availability.sh` — rolling forecast availability for OSCO rollout](#run_dsa_rolling_forecast_availabilitysh--rolling-forecast-availability-for-osco-rollout)
  - [`run_compile_simulation.sh` — pressure-aggregated simulation file](#run_compile_simulationsh--pressure-aggregated-simulation-file)
  - [`generate_grade_weight_files.py` — grade-based weights for SBs and projects](#generate_grade_weight_filespy--grade-based-weights-for-sbs-and-projects)
- [Full-year simulations](#full-year-simulations)
  - [`run_dsa_eb.sh` — DSA + executive-balance penalty](#run_dsa_ebsh--dsa--executive-balance-penalty)
  - [`run_prophet.sh` — Prophet (oracle upper bound)](#run_prophetsh--prophet-oracle-upper-bound)
  - [`run_pulsar.sh` — PULSAR (strategic weekly counter + Gurobi OSCO rollout)](#run_pulsarsh--pulsar-strategic-weekly-counter--gurobi-osco-rollout)
  - [Input dependency summary](#input-dependency-summary)
- [Interactive dashboard](#interactive-dashboard)
- [Single time-step scheduling](#single-time-step-scheduling)
  - [`run_step_pulsar.sh` — PULSAR single step](#run_step_pulsarsh--pulsar-single-step)
  - [`run_step_dsa_eb.sh` — DSA+EB single step](#run_step_dsa_ebsh--dsaeb-single-step)

# Installation

Before running anything, install and activate the `pulsar` conda environment:

```bash
conda env create --name pulsar --file pulsar.yml python=3.11
conda activate pulsar
```

# Environment configuration

All paths that depend on your machine are centralized in two small shell files in
`Codes/short_term/shell`, so you set them once rather than editing every script.

How the two files work:

- Every simulation script begins by sourcing
  `base_env.sh` via `. "$(dirname "$0")/base_env.sh"`. This pulls in the common path/config variables.
- Scripts that invoke ALMA's DSA instead source `dsa_calls_env.sh`. That file sources
  `base_env.sh` first and then adds the DSA-specific variables on top.
- In practice this means you **edit `base_env.sh` once** when setting up a new machine, and only touch
  `dsa_calls_env.sh` if your DSA installation or preprocessed outputs live somewhere non-standard.

## `base_env.sh` variables

Located at `Codes/short_term/shell/base_env.sh`. To set them, open the file and edit the assignments
near the top.

- `PYTHON_BIN` — command or full path to the Python executable (e.g. `python`, `python3`, or
  `/path/to/env/bin/python`). 
- `PROJECT_ROOT` — absolute path to the root of this repository (the directory containing `Codes/`,
  `README.md`, etc.). Set this to wherever you cloned the repo.
- `SRC_DIR` — derived automatically as `${PROJECT_ROOT}/Codes/short_term/src`. You normally do not need
  to change this; it updates automatically once `PROJECT_ROOT` is correct.
- `DATA_DIR` — absolute path to the directory holding all input data files (CSVs, pickles, DSA sim
  outputs, etc.). This is the `DATA_DIR` referenced throughout the rest of this README.
- `GRB_LICENSE_FILE` — path to your Gurobi license file. It is exported with a default of
  `/opt/gurobi/license/gurobi.lic`, but only if the variable is not already set. To override, either set
  `GRB_LICENSE_FILE` in your shell before running a script, or edit the default value in this file.

## `dsa_calls_env.sh` variables

Located at `Codes/short_term/shell/dsa_calls_env.sh`. This file sources `base_env.sh` first (so all of
the variables above are available), then sets the DSA-specific paths below. To set them, open the file
and edit the assignments.

The DSA module is ALMA's proprietary dynamic scheduling algorithm (DSA) code. This code is used to calculate expected execution fraction for observations under different conditions. If expected execution fraction is high enough, an observation will be considered feasible for scheduling. The scheduling algorithms in this repository (ALMA's dispatch based scheduler, Prophet oracle, and PULSAR) take the feasibility calculations from ALMA's DSA as input, and optimize the selection among feasible observations at each step.

- `DSA_BASE_DIR` — path to the top-level directory of the DSA submodule installation.  Derived as
  `${SRC_DIR}/DSA/DSA`; change only if the submodule is cloned elsewhere. 
- `DSA_SRC_DIR` — path to DSA's `src/` subdirectory. Derived as `${DSA_BASE_DIR}/src`; normally left as
  is.
- `POL_FILE_PATH` — directory from which DSA reads its policy files. Set to `${DSA_BASE_DIR}` by
  default.
- `DSA_LOG_DIR` — directory where DSA writes its log files. Set to `${DSA_BASE_DIR}/logs` by default.
- `PREPROCESSED_ROOT` — absolute path to the root directory containing the preprocessed outputs
  (specifically the `year_${YEAR}/realized_weather.pkl` files produced by
  `run_preprocess_weather_forecasts.sh`). **This must be set to your local path.**

# Preparing the inputs

## Key concepts

**Project.** A scientific program assigned a letter grade (A, B, or C) reflecting its scientific
priority, and an executive affiliation (CL, EA, EU, or NA) indicating which ALMA partner organization
owns the observation time. Some projects are jointly owned and carry fractional time-share weights across
multiple executives. Each project has a scientific rank used in objective weighting.

**Scheduling block (SB).** The atomic unit of scheduling. Each SB belongs to a project and specifies
what is to be observed: the target source, observing band and representative frequency, required array
configuration, estimated execution time, and observing constraints (hour-angle window, maximum
precipitable water vapor, and RMS atmospheric noise limit). Only SBs whose constraints are satisfied by
the current telescope state may be scheduled at a given time.

**Multiple executions.** An SB may need to be observed more than once to accumulate sufficient
sensitivity or for calibration purposes. The `execount` field specifies the total number of required
executions; the simulator tracks how many have been completed and treats the SB as finished only when
all `execount` executions have been performed. Each execution occupies one scheduling slot and draws
down the SB's remaining execution count.

**The selector function.** The observability of every SB at every 30-minute timestep is determined by
`dsa12m.selector()`, called inside `run_dsa.py`. Given the current time, array configuration, band,
PWV, and RMS conditions, the selector checks each SB's hour-angle visibility, frequency-specific
atmospheric transmission, array-configuration compatibility, and project/SB status flags, and returns
the subset of SBs that are legally observable. This check is the foundation of the simulation pipeline:
`run_dsa_avail.sh` calls it once per timestep to record which SBs are available, and `run_dsa_scores.sh`
calls it to compute a DSA score for each available SB. All three scheduling algorithms then consume
these precomputed availability and score tables as their primary input, rather than re-evaluating
telescope constraints at runtime.

The selector itself is part of ALMA's Dynamic Scheduling Algorithm (DSA) software, which is proprietary
ALMA code that is not redistributed in this repository. Adapting PULSAR to a different observatory
means replacing `dsa12m.selector()` — and the surrounding DSA tool calls in `run_dsa.py` — with
observatory-specific code that performs the equivalent observability check for that facility's
constraints and data formats.

## Inputs and scripts

The simulators depend on a number of generated inputs. Each one is produced by a dedicated script under
`Codes/short_term/shell/` (or in one case directly under `Codes/short_term/src/`). The same convention
applies throughout: open the script, update the paths and parameters at the top, then execute it.
Generate these inputs before running any of the full-year simulations described in the next section.

Before running any of those scripts, however, you must first obtain the static cycle reference data
described next, since most of the preparation scripts (and the simulators themselves) depend on it.

## Static cycle reference data

The following CSVs are not produced by any script in this repo. They are static cycle reference data
that you are expected to obtain or generate separately and place in `DATA_DIR`. The column descriptions
below list the fields the code actually reads; your files may contain additional columns, which are
ignored.

### Required by all three algorithms

These are loaded inside `full_year.py` when jobs and the cycle calendar are built.

**`schedblocks_c10.csv`** — master scheduling-block (SB) list. One row per SB.

| Column | Description |
|---|---|
| `SB_UID` | Unique scheduling-block identifier; becomes the internal `job_id`. |
| `OBSPROJECT_UID` | Foreign key used to join each SB to its project in `projects_c10.csv`. |
| `execount` | Planned number of executions for the SB. SBs with `execount <= 0` are filtered out; it also caps how many times the SB may be scheduled. |
| `estimatedTime` | Total estimated SB time in hours. Divided by `execount` to get the per-execution duration, then binned into 30-minute steps. |
| `GOUS_ID` | Group Observing Unit Set. Each SB is part of a group, within a project. This is only used for the baseline dispatch scheduler. |

**`projects_c10.csv`** — per-project metadata. One row per project.

| Column | Description |
|---|---|
| `OBSPROJECT_UID` | Join key to `schedblocks_c10.csv`. |
| `CODE` | Project code; becomes the internal `project_id`. |
| `PRJ_LETTER_GRADE` | Project letter grade (A/B/C); drives grade-based weights. |
| `EXEC` | Default executive affiliation (CL/EA/EU/NA). |
| `PRJ_SCIENTIFIC_RANK` | Science rank. Lower is better, with `1` being the best. |

**`proposals_time_share_mod.csv`** — executive time-share fractions for jointly owned projects. One row per shared project.

| Column | Description |
|---|---|
| `CODE` | Project code (matches `projects_c10.csv`). |
| `NA`, `EU`, `EA`, `CL` | Fractional executive shares for the project (summing to 1). When a project appears here, its job `executive` becomes this fraction dict instead of the single `EXEC` value from `projects_c10.csv`. |

**`cycle_10_sb_active_time_to_complete_at_c10_start.csv`** — remaining executions per SB at the start of the cycle.

| Column | Description |
|---|---|
| `SB_UID` | Scheduling-block identifier. |
| `execution_count_start_c10` | Remaining executions at cycle start; rounded to an integer and used to cap `remaining_execs`. |

**`shifts_dimensions.csv`** — operator shift schedule. One row per shift interval.

| Column | Description |
|---|---|
| `START_TIME` | Interval start, parsed as a UTC datetime. |
| `END_TIME` | Interval end. |
| `SHIFT_ACTIVITY` | Activity type. Rows labeled `Engineering` or `EOC` mark maintenance slots. The simulator treats them as anticipated downtime. |

**`downtimes_dimensions.csv`** — Unplanned downtime intervals. One row per downtime.

| Column | Description |
|---|---|
| `START_TIME` | Downtime interval start (UTC). |
| `END_TIME` | Downtime interval end (UTC). |
| `DOWNTIME_TYPE` | Category. `Weather` and `Technical` are unplanned downtime categories. `Scheduling` intervals are tracked as scheduling-downtime indices in the simulator, but are available for observing. |

The following three "dashboard" tables are read by `run_dsa.py` during DSA preprocessing
(`run_dsa_scores.sh`, `run_dsa_avail.sh`). They reconstruct which antennas and pads were operational
at each simulated timestamp.

**`DASHBOARD_ANTENNA.csv`** — ALMA antenna registry. One row per antenna.

| Column | Description |
|---|---|
| `ID` | Database antenna ID; join key to `DASHBOARD_EVENT.ANTENNA_ID`. |
| `NAME` | Antenna name. |

**`DASHBOARD_EVENT.csv`** — antenna status-change log. One row per status change.

| Column | Description |
|---|---|
| `ANTENNA_ID` | Links the event to an antenna via `DASHBOARD_ANTENNA.ID`. |
| `EVENTTIME` | UTC timestamp when the status change occurred. |
| `ID` | Event ID; matched against `DASHBOARD_EVENT_PAIR.START_EVENT_ID`. |
| `CHANGE_TYPE` | Status category (e.g. `ArrayElementStatus`, `B1Status`–`B10Status`, `pumping`, `Pad`). |
| `NEW_VALUE` | New status value (e.g. `"O"`/`"C"` for open/closed, or can be a pad location string). |

**`DASHBOARD_EVENT_PAIR.csv`** — validity intervals for each dashboard state. One row per state interval.

| Column | Description |
|---|---|
| `START_TIMESTAMP` | Start of the validity interval. |
| `END_TIMESTAMP` | End of the validity interval. |
| `START_EVENT_ID` | Event ID (from `DASHBOARD_EVENT`) whose state is active during `[START_TIMESTAMP, END_TIMESTAMP]`. |

### Additionally required by `pulsar`

These are used inside the long-term (weekly) planner.

**`expected_times_c10_for_strategic.csv`** — per-record estimate of available observing time, broken
down by array configuration and local sidereal time (LST) bin. The long-term optimizer sums `available_time` over each (configuration, LST bin) pair to estimate the total observing time available in each time bin. This per-bin time budget is what the optimizer uses to allocate scheduling blocks across the cycle.

| Column | Description |
|---|---|
| `timestamp` | Record timestamp. Consecutive rows sharing the same `conf` delimit the start and end dates of each array configuration period. |
| `conf` | Array configuration label (e.g. `Configuration-7`); groups records into configuration periods. |
| `available_time` | Observing time (hours) available for this record's configuration and LST bin. Summed per configuration to get each configuration's total duration, and summed per (configuration, LST bin) to estimate the observing-time budget in each time bin. |
| `LST_bin` | Local sidereal time bin index (hourly). Multiplied by 2 to index the 48 half-hour bins per day when accumulating the per-bin time budget. |

**`sb12m_master_prepared_c10.csv`** — master SB file providing per-SB metadata for the runtime
system. It is used to map SB UIDs between the scheduling-block catalogue and the long-term model
(`SB_UID`, plus `PRJ_CODE`, `NUMBER_OF_EXECUTIONS`, `SB_TOTAL_ESTIMATED_TIME` for nearest-time
matching), and as the SB-metadata template consumed by the long-term optimizer.

| Column | Description |
|---|---|
| `SB_UID` | SB identifier. |
| `PRJ_CODE` | Project code. |
| `PRJ_GRADE` | Project grade. |
| `PRJ_SCIENTIFIC_RANK` | Science rank. |
| `SB_TOTAL_ESTIMATED_TIME` | Total SB time. |
| `NUMBER_OF_EXECUTIONS` | Execution count. |
| `SB_TIME_BY_EXECUTION` | Hours per execution. |
| `OPTIMAL_PWV` | SB metadata. |
| `ISTIMECONSTRAINED`, `ISTOO` | Time-constraint and target-of-opportunity flags. |
| `CL`, `EA`, `EU`, `NA` | Executive organization fractions. |

**`sb12m_master_with_modes.csv`** — per-SB observing-mode information, read by the long-term
pre-processor.

| Column | Description |
|---|---|
| `CODE` | Project code. |
| `MODE_NAME` | Observing-mode string. Projects with `BandToBand Interferometry` or `BandwidthSwitching Interferometry` modes are subject to a 45-hour cap in the long-term optimizer. |

**`accepted_projects_cycle10.csv`** — accepted-project list for the cycle, read by the long-term
pre-processor.

| Column | Description |
|---|---|
| `PRJ_CODE` | Accepted project code; filters the master SB list. |
| `GRADE` | Accepted project grade; mapped onto `PRJ_GRADE` in the master table. |

**`sb_12m_pressure.csv`** — reference pressure file used to build the LST-bin lookup table.

| Column | Description |
|---|---|
| `Date` | 30-minute UTC timestamp; key for LST and configuration lookup. |
| `lst` / `LST` | Local sidereal time (case-insensitive column name); mapped to 0.5-hour LST bins. |
| `ARRAY` | Array configuration string (e.g. `Configuration-7`). |
| `SB_UID` | SB identifier. |
| `weather_suitable_fraction` | Fraction of historical years where weather was suitable at that timestamp. |

Note that `projects_c10.csv` and `schedblocks_c10.csv` also serve as inputs to
`generate_grade_weight_files.py`, described below.

## `run_preprocess_weather_forecasts.sh` — realized weather and forecasts

Runs two preprocessing stages back-to-back:

1. `preprocess_weather.py` ingests the raw cycle weather data and writes
   `${DATA_DIR}/preprocessed/year_${YEAR}/realized_weather.pkl`. This file is required by all three
   algorithms.
2. `preprocess_forecasts.py` fits a rolling unobserved-components model (UCM) for RMS, combines it
   with NOAA PWV forecasts, and writes `${DATA_DIR}/preprocessed/year_${YEAR}/forecasts_real.pkl`.
   This file is required only by `pulsar`.

Key parameters:
- `YEAR` — cycle starting year (default `2023`)
- `RMS_HORIZON_HOURS`, `RMS_AR_ORDER`, `RMS_DIFF_ORDER`, `RMS_MA_ORDER`,
  `RMS_DAILY_FOURIER_ORDER`, `RMS_YEARLY_FOURIER_ORDER` — RMS UCM hyperparameters (defaults
  `16`, `4`, `0`, `1`, `4`, `2`); these define how the rolling RMS forecast is structured

Run as:

```bash
bash Codes/short_term/shell/run_preprocess_weather_forecasts.sh
```

If you only intend to run `prophet` or `dsa_eb`, you only need `realized_weather.pkl`; you can comment
out or skip Step 2.

## `run_dsa_avail.sh` — daily DSA availability

Runs `run_dsa.py --function avail` for a single date, producing one
`dsa_sim/dsa_sim_${month}_${day}_${year}_df.csv` file. **This script must be run for every date in the
cycle** — Oct 1 of `START_YEAR` through Sep 30 of `START_YEAR + 1`. Required by all three algorithms,
because the resulting per-SB LST windows and PWV/RMS thresholds determine which time steps each SB is
schedulable in.

Key parameters:
- `START_YEAR`, `START_MONTH`, `START_DAY` — the date to generate availability for

For full-cycle coverage on a SLURM cluster, use `run_dsa_avail.sbatch`, which dispatches every
in-window date as a parallel array job.

## `run_dsa_scores.sh` — daily DSA scores

Runs `run_dsa.py --function scores` for a single date, producing the per-day score file used to rank
jobs in the `dsa_eb` selector. **Like `run_dsa_avail.sh`, this must be run for every date in the
cycle.** Required only by `dsa_eb`.

Key parameters:
- `START_YEAR`, `START_MONTH`, `START_DAY` — the date to generate scores for
- `DSA_BASE_DIR`, `DSA_SRC_DIR`, `POL_FILE_PATH`, `DSA_LOG_DIR` — paths into the DSA submodule used to compute scores
- `PREPROCESSED_ROOT` — points at the directory containing `year_${YEAR}/realized_weather.pkl`

For full-cycle coverage on a SLURM cluster, use `run_dsa_scores.sbatch`.

## `run_dsa_rolling_forecast_availability.sh` — rolling forecast availability for OSCO rollout

Runs `run_dsa.py --function avail_rolling_forecast` for one 8-hour-issued, 16-hour-window forecast
slot. There are roughly 1095 slots in a full cycle; the script processes a single slot indexed by
`SLOT_INDEX`. **All slots must be generated to cover the full cycle.** Output goes to
`dsa_sim_for_forecast_rolling/`. Required only by `pulsar`.

Key parameters:
- `YEAR` — cycle starting year (default `2023`)
- `SLOT_INDEX` — which 8-hour issuance slot to generate (0 maps to Oct 1 00:00 UTC)
- `ISSUE_EVERY_HOURS`, `FORECAST_WINDOW_HOURS`, `FORECAST_OUTPUT_DIR` — must match the values used
  later when running the simulator (defaults `8`, `16`, `dsa_sim_for_forecast_rolling`)

For full-cycle coverage on a SLURM cluster, use
`submit_all_dsa_rolling_forecast_availability.sbatch`, which dispatches all 1100 slots in parallel.

## `run_compile_simulation.sh` — pressure-aggregated simulation file

Runs `compile_simulation_for_strategic.py` to scan all `dsa_sim/dsa_sim_*_df.csv` files and aggregate
them into a single per-cycle pressure file:
`${DATA_DIR}/sb_12m_pressure_${YEAR}.csv`. PULSAR uses this file during its
2-week replan steps. Required only by `pulsar`.

Key parameters:
- `YEAR` — cycle starting year (default `2023`)
- `WRITE_INTERVAL` — how often (in files processed) to flush partial output (default `500`)

This step requires the daily DSA availability files (`dsa_sim/`) to already exist, so run
`run_dsa_avail.sh` first.

## `generate_grade_weight_files.py` — grade-based weights for SBs and projects

Reads `projects_c10.csv` and `schedblocks_c10.csv` from `DATA_DIR` and writes `project_weights.csv`
and `sb_weights.csv` to the same directory. Each row maps an SB UID or project code to its letter
grade and a numeric weight (A → `1.0`, B → `0.4`, C → `-10`). These weight files are read by all
three algorithms during job loading. Required by `prophet`, `dsa_eb`, and
`pulsar`.

Run as:

```bash
python Codes/short_term/src/generate_grade_weight_files.py --data_dir "${DATA_DIR}"
```

The script also accepts overrides for input and output paths via `--projects_csv`,
`--schedblocks_csv`, `--project_weights_out`, and `--sb_weights_out`.

# Full-year simulations

Once the inputs are prepared, ready-to-run shell scripts for each supported algorithm live in
`Codes/short_term/shell`. Each `run_*.sh` script is self-contained: after setting your machine paths in
`base_env.sh` (see [Environment configuration](#environment-configuration)), open the script, edit any
remaining user-configurable parameters at the top, then execute it from the project root, e.g.

```bash
bash Codes/short_term/shell/run_dsa_eb.sh
```

All three scripts call the same entry point (`Codes/short_term/src/full_year.py`) but with different
algorithm names and inputs. By default each script runs a single cycle starting in October of the year
specified by `YEAR` (default `2023`) and ending September 30 of the following year. Output is written to
`OUTPUT_ROOT/year_${YEAR}/${ALGORITHM}/` as a pickle file containing the schedule, completion percentages,
executive-balance fractions, and other metrics.

The three supported algorithms are described below, followed by a dependency table summarizing which
inputs each one needs.

## `run_dsa_eb.sh` — DSA + executive-balance penalty

A greedy operational scheduler that, at each 30-minute time step, ranks candidate scheduling blocks
using DSA visibility/score data combined with an executive-balance penalty that ramps nonlinearly over
the cycle. As the cycle progresses, the penalty for being away from per-executive time targets grows
sharply, pushing the scheduler to converge on the target distribution by the end of the cycle.

Inputs needed:
- Realized weather pickle (`realized_weather.pkl`)
- Daily DSA availability files (`dsa_sim/dsa_sim_*_df.csv`)
- Daily DSA score files (`dsa_sim_scores_*_df.csv`)
- Project and SB grade weight files (`project_weights.csv`, `sb_weights.csv`)
- Static cycle reference data (see "Static cycle reference data" above)

Key parameters in the script:
- `YEAR` — cycle starting year (default `2023`)
- `EB_RAMP_EXPONENT` — controls how steeply the executive-balance penalty grows over the cycle (default `10`); higher values make the scheduler less aggressive about hitting EB targets near the beginning
- `W_SB`, `W_PROJ`, `W_UTIL`, `W_EBP` — objective weights on SB completion, project completion, utilization, and EB penalty respectively. Must sum to 1. The defaults (`0.002, 0.002, 0.001, 0.995`) are an EB-heavy preset. The first 3 are just used for calculating metrics on the completed schedule, for this algorithm. The W_EBP is the total weight placed on executive balance penalty by the end of the cycle.
- `SEED` — RNG seed (default `31415`)
- `PYTHON_BIN`, `PROJECT_ROOT`, `SRC_DIR`, `DATA_DIR`, `OUTPUT_ROOT`, `PREPROCESSED_WEATHER_ROOT` — paths you must update to match your environment

Usually takes less than 5-10 minutes to run the entire cycle's simulation.

## `run_prophet.sh` — Prophet (oracle upper bound)

Solves a single full-year Gurobi mixed-integer program with perfect knowledge of the realized weather
for the entire cycle. This is not a deployable algorithm — it is an oracle that produces an
informative upper bound on achievable performance against which the operational algorithms can be
compared.

Inputs needed:
- Realized weather pickle (`realized_weather.pkl`)
- Daily DSA availability files (`dsa_sim/dsa_sim_*_df.csv`)
- Project and SB grade weight files
- Static cycle reference data (see "Static cycle reference data" above)

No forecast data and no DSA score files are required. The daily DSA availability files are still
required because they define when each SB is physically observable (LST window plus PWV/RMS
thresholds); without them no SB will ever be schedulable.

Key parameters in the script:
- `YEAR` — cycle starting year (default `2023`)
- `W_SB`, `W_PROJ`, `W_UTIL`, `W_EBP` — objective weights, same meaning as above. We actually do optimize for the objective as defined by these weights now.
- `SEED` — RNG seed (default `31415`)
- Same path variables as `run_dsa_eb.sh`

The internal Gurobi time limit is set to 6 hours.

## `run_pulsar.sh` — PULSAR (strategic weekly counter + Gurobi OSCO rollout)

This combines three layers:

1. A 2-week strategic replan run every two weeks.
2. A weekly strategic counter that tracks which A- and B-graded SBs should have been completed according to the strategic plan.
3. At each 30-minute step, a Gurobi-based online stochastic combinatorial optimization (OSCO) rollout that samples weather forecasts and selects the
   next job by approximately solving a short-horizon stochastic look-ahead.

Inputs needed:
- Realized weather pickle (`realized_weather.pkl`)
- Real forecast pickle (`forecasts_real.pkl`)
- Daily DSA availability files (`dsa_sim/dsa_sim_*_df.csv`)
- Rolling DSA forecast availability files (`dsa_sim_for_forecast_rolling/`)
- Compiled simulation pressure file (`sb_12m_pressure_{year}.csv`)
- Project and SB grade weight files
- Static cycle reference data, including the additional long-term files (see "Static cycle reference data" above)

Key parameters in the script:
- `YEAR` — cycle starting year (default `2023`)
- `W_SB`, `W_PROJ`, `W_UTIL`, `W_EBP` — objective weights, same meaning as above
- `EB_RAMP_EXPONENT` — same as for `dsa_eb` (default `10`)
- `COUNTER_BONUS_A_MULTIPLIER`, `COUNTER_BONUS_B_MULTIPLIER` — strength of the strategic-counter bonus for A and B graded SBs (defaults `10` and `1.5`)
- `SEQUENCE_HORIZON_STEPS` — how many 30-minute steps the OSCO rollout looks ahead (default `16`, i.e. 8 hours)
- `OSCO_NUM_SAMPLES` — number of forecast samples drawn at each rollout (default `5`)
- `OSCO_INNER_GUROBI_TIME_LIMIT_SECONDS` — per-sample Gurobi time limit during rollout (default `10`)
- `OSCO_N_THREADS` — worker threads used by OSCO sampling (default `20`); set this to roughly the number of CPU cores you can dedicate to the run
- `SEED` — RNG seed (default `31415`)
- Same path variables as the other scripts

One simulation through a full cycle should take about 24-48 hours.

## Input dependency summary

| Input | prophet | dsa_eb | pulsar |
|---|---|---|---|
| `realized_weather.pkl` | yes | yes | yes |
| `project_weights.csv`, `sb_weights.csv` | yes | yes | yes |
| Daily DSA availability (`dsa_sim/`) | yes | yes | yes |
| Daily DSA scores (`dsa_sim_scores_*`) | no | yes | no |
| `forecasts_real.pkl` | no | no | yes |
| Rolling forecast availability (`dsa_sim_for_forecast_rolling/`) | no | no | yes |
| `sb_12m_pressure_{year}.csv` | no | no | yes |

# Interactive dashboard

`Codes/short_term/src/dashboard.py` is a Plotly Dash web app for interactively comparing two scheduler
weight configurations side by side. It renders two bar charts — project/SB completion ("Projects") and
executive-balance fractions ("Executive Balance") — for one or two runs at once. It is useful for
exploring how changing the per-project grade scores and executive-balance targets affects the resulting
schedule.

There are two ways to populate each panel: running an algorithm live (the dashboard launches a
`run_dsa_eb_dashboard.py` subprocess) or uploading a previously generated results pickle.

Prerequisites:

- The `pulsar` conda environment. 
- For the live "Use Algorithm Output" mode: the same inputs `run_dsa_eb.sh` needs for the chosen cycle
  (static cycle reference data plus the `dsa_sim/` availability and score files) in the `--data_dir`
  directory, the `year_${YEAR}/realized_weather.pkl` file under `--preprocessed_root`, and a working
  Gurobi license.
- For "Upload Pickle File" mode: no preprocessed data is required — just a results pickle, such as the
  output of `run_dsa_eb.sh`.

The recommended way to launch the dashboard is the `run_dashboard.sh` shell script, which sources
`base_env.sh` (so the paths come from the same place as every other script) and passes the required
command-line arguments to `dashboard.py` for you. After setting your machine paths in `base_env.sh`
(see [Environment configuration](#environment-configuration)), open `run_dashboard.sh`, edit any of the
variables below if needed, then run it from the project root:

```bash
bash Codes/short_term/shell/run_dashboard.sh
```

Then open `http://localhost:8051` in a browser (or the port you set).

Configurable variables in `run_dashboard.sh`:

- `YEAR` — cycle starting year (default `2023`); `START_DATE` and `END_DATE` are derived from it as
  October 1 of `YEAR` through September 30 of the following year.
- `DASHBOARD_DATA_DIR` — directory containing the static cycle reference data and the `dsa_sim/`
  availability/score files (defaults to `${DATA_DIR}/dataDashboard`).
- `PREPROCESSED_ROOT` — root directory containing `year_${YEAR}/realized_weather.pkl`, produced by
  `run_preprocess_weather_forecasts.sh` (defaults to `${DATA_DIR}/preprocessed`).
- `PORT` — port for the Dash server (default `8051`).

If you prefer to invoke `dashboard.py` directly, run it from `Codes/short_term/src` (the live-run
subprocess calls `run_dsa_eb_dashboard.py` by relative path); `python dashboard.py --help` lists the
underlying `--src_dir`, `--data_dir`, `--preprocessed_root`, `--start_date`, `--end_date`, and `--port`
arguments.

The UI has two symmetric columns: the left column controls Weights Set 1 and the right column controls
Weights Set 2. Each column exposes:

- An algorithm dropdown: "Current" (`dsa`) or "Current + Penalty" (`dsa_eb`).
- Weight inputs: Project A/B/C grade scores; Executive Balance targets for Chile, East Asia, Europe,
  and North America (entered as percentages); and Executive Balance Looseness (the `eb_ramp_exponent`).
- A source dropdown: "Use Algorithm Output" (run the algorithm live via subprocess) or "Upload Pickle
  File" (drag-and-drop a results pickle).

A "Display Mode" radio selects whether to render "Compare Two Runs", "Only Weights Set 1", or "Only
Weights Set 2", and the "Run" button triggers rendering.

When a panel is set to "Use Algorithm Output", clicking "Run" invokes `run_dsa_eb_dashboard.py` as a
subprocess for the cycle specified by `--start_date`/`--end_date`. A full live run can take several
minutes.

# Single time-step scheduling

In addition to the full-year simulations above, the repository (will) provide scripts that run a single
30-minute scheduling decision at the command line. These are intended for deployment and debugging:
rather than simulating an entire cycle, they take the current observatory state and produce the
recommended next scheduling block for one time step.

## `run_step_pulsar.sh` — PULSAR single step

Runs a single 30-minute scheduling decision using PULSAR's OSCO rollout. Given the current telescope
state, the cycle progress, and a set of weather forecast samples, the script outputs the recommended
next scheduling block and the supporting look-ahead scores.

*This script is not yet available. It will accept the same path and model parameters as `run_pulsar.sh`,
plus a `--timestamp` argument specifying the decision point.*

## `run_step_dsa_eb.sh` — DSA+EB single step

Runs a single 30-minute scheduling decision using the DSA + executive-balance greedy selector. Given
the current telescope state and cycle progress, the script outputs the top-ranked scheduling block.

*This script is not yet available. It will accept the same path and model parameters as `run_dsa_eb.sh`,
plus a `--timestamp` argument specifying the decision point.*
