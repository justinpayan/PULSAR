#!/bin/bash
# ===================================================================
# ==  DSA AVAILABILITY - SINGLE DATE EXAMPLE                        ==
# ===================================================================
# This script generates DSA availability for a single date. To cover
# a full observing cycle (Oct 1 YYYY through Sep 30 YYYY+1) you must
# run it for each date in that range. On a SLURM cluster, prefer
# run_dsa_avail.sbatch which dispatches every date in parallel.
# ===================================================================

# shellcheck source=dsa_calls_env.sh
. "$(dirname "$0")/dsa_calls_env.sh"

RUN_DSA_SCRIPT="${SRC_DIR}/run_dsa.py"

START_YEAR=2023
START_MONTH=10
START_DAY=1

echo "Running DSA availability generation for ${START_YEAR}-${START_MONTH}-${START_DAY}"

"${PYTHON_BIN}" "${RUN_DSA_SCRIPT}" \
    --start_month "${START_MONTH}" \
    --start_day "${START_DAY}" \
    --start_year "${START_YEAR}" \
    --function "avail"
