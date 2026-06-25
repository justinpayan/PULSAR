#!/bin/bash

# ============================================
#SBATCH --job-name=run_dsa_all_years
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=0-3:00:00
#SBATCH --output=/data/user_data/jpayan/logs/AOOSP/run_dsa_%a.out
#SBATCH --error=/data/user_data/jpayan/logs/AOOSP/run_dsa_%a.err
#SBATCH --partition=array

# Years to process: 2017, 2018, 2019, 2021, 2022, 2023, 2024
# Total jobs: 7 years * 12 months * 31 days = 2604
#SBATCH --array=0-2603
# ============================================

# shellcheck source=dsa_calls_env.sh
. "$(dirname "$0")/dsa_calls_env.sh"

# Define years array
YEARS=(2017 2018 2019 2021 2022 2023 2024)

# Calculate year, month, and day from array index
# Each year has 12 months * 31 days = 372 combinations
DAYS_PER_YEAR=372
YEAR_INDEX=$(($SLURM_ARRAY_TASK_ID / $DAYS_PER_YEAR))
REMAINING_INDEX=$(($SLURM_ARRAY_TASK_ID % $DAYS_PER_YEAR))

START_YEAR=${YEARS[$YEAR_INDEX]}
START_MONTH=$(($REMAINING_INDEX / 31 + 1))
START_DAY=$(($REMAINING_INDEX % 31 + 1))

# Skip dates outside the two valid simulation windows:
#   Oct 1 2017 – Sep 30 2019  and  Oct 1 2021 – Sep 30 2024
if   { [ "$START_YEAR" -eq 2017 ] && [ "$START_MONTH" -lt 10 ]; } \
  || { [ "$START_YEAR" -eq 2019 ] && [ "$START_MONTH" -gt 9  ]; } \
  || { [ "$START_YEAR" -eq 2021 ] && [ "$START_MONTH" -lt 10 ]; } \
  || { [ "$START_YEAR" -eq 2024 ] && [ "$START_MONTH" -gt 9  ]; }
then
    echo "Skipping out-of-window date: ${START_YEAR}-${START_MONTH}-${START_DAY}"
    exit 0
fi

"${PYTHON_BIN}" "${SRC_DIR}/run_dsa.py" --start_month ${START_MONTH} --start_day ${START_DAY} --start_year ${START_YEAR} --function "avail"
