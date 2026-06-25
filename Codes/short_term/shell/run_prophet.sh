#!/bin/bash
# ===================================================================
# ==   PROPHET - SINGLE YEAR RUN                                    ==
# ===================================================================

# shellcheck source=base_env.sh
. "$(dirname "$0")/base_env.sh"

FULL_YEAR_SCRIPT="${SRC_DIR}/full_year.py"
OUTPUT_ROOT="${OUTPUTS_ROOT}/prophet"
PREPROCESSED_WEATHER_ROOT=$DATA_DIR/preprocessed

YEAR=2023
ALGORITHM="prophet"

# EB-heavy weight settings
W_UTIL=0.001
W_SB=0.002
W_PROJ=0.002
W_EBP=0.995

SEED=31415

PREPROCESSED_WEATHER_PATH="${PREPROCESSED_WEATHER_ROOT}/year_${YEAR}/realized_weather.pkl"
OUTPUT_DIR="${OUTPUT_ROOT}/year_${YEAR}/${ALGORITHM}"

mkdir -p "${OUTPUT_DIR}"

START_DATE="${YEAR}-10-01"
END_YEAR=$((YEAR + 1))
END_DATE="${END_YEAR}-09-30"

echo "========================================================"
echo "Starting Prophet"
echo "========================================================"
echo "Year: ${YEAR}"
echo "Algorithm: ${ALGORITHM}"
echo "Running simulation from ${START_DATE} to ${END_DATE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Weights: SB=${W_SB}, Proj=${W_PROJ}, EBP=${W_EBP}, Util=${W_UTIL}"
echo "========================================================"

"${PYTHON_BIN}" "${FULL_YEAR_SCRIPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --seed "${SEED}" \
    --w_sb "${W_SB}" \
    --w_proj "${W_PROJ}" \
    --w_util "${W_UTIL}" \
    --w_ebp "${W_EBP}" \
    --preprocessed_weather "${PREPROCESSED_WEATHER_PATH}" \
    --algorithm_name "${ALGORITHM}"

echo "========================================================"
echo "Prophet run finished."
echo "Year: ${YEAR}, Algorithm: ${ALGORITHM}"
echo "========================================================"
