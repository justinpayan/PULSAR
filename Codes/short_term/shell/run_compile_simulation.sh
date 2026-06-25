#!/bin/bash
# ===================================================================
# ==  COMPILE SIMULATION DATA FOR STRATEGIC SCHEDULING - SINGLE YEAR
# ===================================================================

# shellcheck source=base_env.sh
. "$(dirname "$0")/base_env.sh"

OUTPUT_DIR="${DATA_DIR}"

YEAR=2023
WRITE_INTERVAL=500

cd "${SRC_DIR}" || exit

echo "========================================================"
echo "Starting Compile Simulation for Strategic Scheduling"
echo "Cycle: ${YEAR}"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Source Directory: ${SRC_DIR}"
echo "========================================================"
date
echo "========================================================"

"${PYTHON_BIN}" compile_simulation_for_strategic.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --write_interval "${WRITE_INTERVAL}" \
    --cycles "${YEAR}"

EXIT_CODE=$?

echo "========================================================"
date
echo "Job completed with exit code: ${EXIT_CODE}"
echo "========================================================"

exit ${EXIT_CODE}
