#!/bin/bash
#SBATCH --job-name=dsa_fc_roll
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=0-12:00:00
#SBATCH --output=/data/user_data/jpayan/logs/AOOSP/dsa_rolling_forecast_%A_%a.out
#SBATCH --error=/data/user_data/jpayan/logs/AOOSP/dsa_rolling_forecast_%A_%a.err
#SBATCH --partition=array
#SBATCH --array=0-5499

YEARS=(2017 2018 2021 2022 2023)
SLOTS_PER_YEAR=1100
ISSUE_EVERY_HOURS=8
FORECAST_WINDOW_HOURS=16
FORECAST_OUTPUT_DIR=dsa_sim_for_forecast_rolling

TASK_ID=${SLURM_ARRAY_TASK_ID}
YEAR_INDEX=$((TASK_ID / SLOTS_PER_YEAR))
SLOT_INDEX=$((TASK_ID % SLOTS_PER_YEAR))

if [ "${YEAR_INDEX}" -ge "${#YEARS[@]}" ]; then
    echo "Task ${TASK_ID} is outside configured year range; exiting."
    exit 0
fi

YEAR=${YEARS[${YEAR_INDEX}]}

ISSUE_INFO=$(python - <<'PY' "${YEAR}" "${SLOT_INDEX}" "${ISSUE_EVERY_HOURS}"
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
    echo "Task ${TASK_ID} maps to an unused slot for year ${YEAR}; exiting."
    exit 0
fi

read START_MONTH START_DAY START_YEAR START_HOUR <<< "${ISSUE_INFO}"

# shellcheck source=dsa_calls_env.sh
. "$(dirname "$0")/dsa_calls_env.sh"

cd "${SRC_DIR}" || exit 1

echo "========================================================"
echo "Starting rolling DSA forecast availability generation"
echo "Task ID: ${TASK_ID}"
echo "Year: ${YEAR}"
echo "Issue timestamp (UTC): ${START_YEAR}-${START_MONTH}-${START_DAY} ${START_HOUR}:00"
echo "Issue cadence: ${ISSUE_EVERY_HOURS}h"
echo "Forecast window: ${FORECAST_WINDOW_HOURS}h"
echo "Output dir: ${FORECAST_OUTPUT_DIR}"
echo "========================================================"

python run_dsa.py \
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
echo "Task ID: ${TASK_ID}"
echo "========================================================"
