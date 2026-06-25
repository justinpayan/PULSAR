#!/bin/bash
# ===================================================================
# ==  DSA ROLLING FORECAST AVAILABILITY - SINGLE SLOT EXAMPLE      ==
# ===================================================================
# This script generates one rolling forecast slot. To cover the full
# year you must run it for every SLOT_INDEX in 0..N-1, where
#   N = ceil(8760 / ISSUE_EVERY_HOURS)  (~1095 slots for 8h cadence).
# On a SLURM cluster, prefer submit_all_dsa_rolling_forecast_availability.sbatch
# which dispatches every slot in parallel as an array job.
# ===================================================================

# shellcheck source=dsa_calls_env.sh
. "$(dirname "$0")/dsa_calls_env.sh"

YEAR=2023
SLOT_INDEX=0   # 0 maps to Oct 1 ${YEAR} 00:00 UTC; vary 0..~1094 for full year

ISSUE_EVERY_HOURS=8
FORECAST_WINDOW_HOURS=16
FORECAST_OUTPUT_DIR=dsa_sim_for_forecast_rolling

ISSUE_INFO=$("${PYTHON_BIN}" - <<'PY' "${YEAR}" "${SLOT_INDEX}" "${ISSUE_EVERY_HOURS}"
from datetime import datetime, timedelta, timezone
import sys

year = int(sys.argv[1])
slot_index = int(sys.argv[2])
issue_every_hours = int(sys.argv[3])

start = datetime(year, 10, 1, 0, 0, tzinfo=timezone.utc)
end = datetime(year + 1, 10, 1, 0, 0, tzinfo=timezone.utc)

issues = []
current = start
while current < end:
    issues.append(current)
    current += timedelta(hours=issue_every_hours)

if slot_index >= len(issues):
    print("")
else:
    issue = issues[slot_index]
    print(issue.strftime("%m %d %Y %H"))
PY
)

if [ -z "${ISSUE_INFO}" ]; then
    echo "SLOT_INDEX=${SLOT_INDEX} is outside the valid range for year ${YEAR}; exiting."
    exit 0
fi

read START_MONTH START_DAY START_YEAR START_HOUR <<< "${ISSUE_INFO}"

cd "${SRC_DIR}" || exit 1

echo "========================================================"
echo "Starting rolling DSA forecast availability generation"
echo "Year: ${YEAR}"
echo "Slot index: ${SLOT_INDEX}"
echo "Issue timestamp (UTC): ${START_YEAR}-${START_MONTH}-${START_DAY} ${START_HOUR}:00"
echo "Issue cadence: ${ISSUE_EVERY_HOURS}h"
echo "Forecast window: ${FORECAST_WINDOW_HOURS}h"
echo "Output dir: ${FORECAST_OUTPUT_DIR}"
echo "========================================================"

"${PYTHON_BIN}" run_dsa.py \
    --function avail_rolling_forecast \
    --start_year "${START_YEAR}" \
    --start_month "${START_MONTH}" \
    --start_day "${START_DAY}" \
    --start_hour "${START_HOUR}" \
    --forecast_issue_every_hours "${ISSUE_EVERY_HOURS}" \
    --forecast_window_hours "${FORECAST_WINDOW_HOURS}" \
    --forecast_output_dir "${FORECAST_OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}"

echo "========================================================"
echo "Finished rolling DSA forecast availability generation"
echo "Slot index: ${SLOT_INDEX}"
echo "========================================================"
