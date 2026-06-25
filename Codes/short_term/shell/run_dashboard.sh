#!/bin/bash
# ===================================================================
# ==   INTERACTIVE DASHBOARD - LAUNCH                              ==
# ===================================================================

# shellcheck source=base_env.sh
. "$(dirname "$0")/base_env.sh"

YEAR=2023
DASHBOARD_DATA_DIR="${DATA_DIR}/dataDashboard"
PREPROCESSED_ROOT="${DATA_DIR}/preprocessed"
PORT=8051

START_DATE="${YEAR}-10-01"
END_YEAR=$((YEAR + 1))
END_DATE="${END_YEAR}-09-30"

echo "========================================================"
echo "Launching interactive dashboard"
echo "========================================================"
echo "Year: ${YEAR}"
echo "Cycle: ${START_DATE} to ${END_DATE}"
echo "Dashboard data dir: ${DASHBOARD_DATA_DIR}"
echo "Preprocessed root: ${PREPROCESSED_ROOT}"
echo "Serving on port: ${PORT}"
echo "========================================================"

cd "${SRC_DIR}"
"${PYTHON_BIN}" dashboard.py \
    --src_dir    "${SRC_DIR}" \
    --data_dir   "${DASHBOARD_DATA_DIR}" \
    --preprocessed_root "${PREPROCESSED_ROOT}" \
    --start_date "${START_DATE}" \
    --end_date   "${END_DATE}" \
    --port       "${PORT}"
