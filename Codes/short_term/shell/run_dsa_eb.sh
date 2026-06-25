#!/bin/bash
# ===================================================================
# ==   DSA_EB - SINGLE YEAR RUN                                     ==
# ===================================================================

# shellcheck source=base_env.sh
. "$(dirname "$0")/base_env.sh"

FULL_YEAR_SCRIPT="${SRC_DIR}/full_year.py"
OUTPUT_ROOT="${OUTPUTS_ROOT}/dsa_eb_eb_heavy"
PREPROCESSED_WEATHER_ROOT=$DATA_DIR/preprocessed

YEAR=2023
ALGORITHM="dsa_eb"
EB_RAMP_EXPONENT=10

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
echo "Starting DSA_EB"
echo "========================================================"
echo "Year: ${YEAR}"
echo "Algorithm: ${ALGORITHM}"
echo "Running simulation from ${START_DATE} to ${END_DATE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Preprocessed weather: ${PREPROCESSED_WEATHER_PATH}"
echo "Weights (eb_heavy): SB=${W_SB}, Proj=${W_PROJ}, EBP=${W_EBP}, Util=${W_UTIL}"
echo "EB ramp exponent: ${EB_RAMP_EXPONENT}"
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
    --algorithm_name "${ALGORITHM}" \
    --eb_ramp_exponent "${EB_RAMP_EXPONENT}" \
    --preprocessed_weather "${PREPROCESSED_WEATHER_PATH}" \
    --debug

echo "========================================================"
echo "DSA_EB run finished."
echo "Year: ${YEAR}, Algorithm: ${ALGORITHM}, Exponent: ${EB_RAMP_EXPONENT}"
echo "========================================================"
