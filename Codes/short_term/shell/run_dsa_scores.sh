#!/bin/bash
# ===================================================================
# ==  DSA SCORES - SINGLE DATE EXAMPLE                              ==
# ===================================================================
# This script generates DSA scores for a single date. To cover a full
# observing cycle (Oct 1 YYYY through Sep 30 YYYY+1) you must run it
# for each date in that range. On a SLURM cluster, prefer
# run_dsa_scores.sbatch which dispatches every date in parallel.
# ===================================================================

# shellcheck source=dsa_calls_env.sh
. "$(dirname "$0")/dsa_calls_env.sh"

RUN_DSA_SCRIPT="${SRC_DIR}/run_dsa.py"

START_YEAR=2023
START_MONTH=10
START_DAY=1

echo "Running DSA score generation for ${START_YEAR}-${START_MONTH}-${START_DAY}"
echo "Using data dir: ${DATA_DIR}"
echo "Using DSA base dir: ${DSA_BASE_DIR}"
echo "Using preprocessed root: ${PREPROCESSED_ROOT}"

"${PYTHON_BIN}" "${RUN_DSA_SCRIPT}" \
    --start_month "${START_MONTH}" \
    --start_day "${START_DAY}" \
    --start_year "${START_YEAR}" \
    --function "scores" \
    --data_dir "${DATA_DIR}" \
    --dsa_base_dir "${DSA_BASE_DIR}" \
    --dsa_src_dir "${DSA_SRC_DIR}" \
    --pol_file_path "${POL_FILE_PATH}" \
    --dsa_log_dir "${DSA_LOG_DIR}" \
    --preprocessed_root "${PREPROCESSED_ROOT}"
