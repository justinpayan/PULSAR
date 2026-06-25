#!/bin/bash
# ==============================================================================
# ==   PULSAR (EB-HEAVY)                                                      ==
# ==   SINGLE YEAR RUN                                                        ==
# ==   Weekly counter refresh, 2-week strategic replans                       ==
# ==============================================================================

# shellcheck source=base_env.sh
. "$(dirname "$0")/base_env.sh"

FULL_YEAR_SCRIPT="${SRC_DIR}/full_year.py"
OUTPUT_ROOT="${OUTPUTS_ROOT}/pulsar_eb_heavy"
PREPROCESSED_ROOT=$DATA_DIR/preprocessed

YEAR=2023
ALGORITHM="pulsar"

# EB-heavy weight settings
W_UTIL=0.001
W_SB=0.002
W_PROJ=0.002
W_EBP=0.995
EB_RAMP_EXPONENT=10
COUNTER_BONUS_A_MULTIPLIER=10
COUNTER_BONUS_B_MULTIPLIER=1.5

SEQUENCE_HORIZON_STEPS=16
OSCO_NUM_SAMPLES=5
OSCO_INNER_GUROBI_TIME_LIMIT_SECONDS=10
OSCO_N_THREADS=20

SEED=31415

PREPROCESSED_WEATHER_PATH="${PREPROCESSED_ROOT}/year_${YEAR}/realized_weather.pkl"
PREPROCESSED_FORECASTS_PATH="${PREPROCESSED_ROOT}/year_${YEAR}/forecasts_real.pkl"
OUTPUT_DIR="${OUTPUT_ROOT}/year_${YEAR}/${ALGORITHM}"

mkdir -p "${OUTPUT_DIR}"

START_DATE="${YEAR}-10-01"
END_YEAR=$((YEAR + 1))
END_DATE="${END_YEAR}-09-30"

echo "========================================================"
echo "Starting PULSAR"
echo "========================================================"
echo "Year: ${YEAR}"
echo "Running simulation from ${START_DATE} to ${END_DATE}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Weights: SB=${W_SB}, Proj=${W_PROJ}, EBP=${W_EBP}, Util=${W_UTIL}"
echo "Sequence horizon: ${SEQUENCE_HORIZON_STEPS} steps"
echo "OSCO num samples: ${OSCO_NUM_SAMPLES}"
echo "OSCO inner Gurobi time limit: ${OSCO_INNER_GUROBI_TIME_LIMIT_SECONDS} seconds"
echo "OSCO worker threads: ${OSCO_N_THREADS}"
echo "Executive pruning: top 5 candidates per executive by grade (fill to 20)"
echo "Strategic replanning cadence: every 2 weeks"
echo "Strategic counter refresh cadence: every week"
echo "EB ramp exponent: ${EB_RAMP_EXPONENT}"
echo "Counter bonus multipliers: A=${COUNTER_BONUS_A_MULTIPLIER}, B=${COUNTER_BONUS_B_MULTIPLIER}"
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
    --eb_ramp_exponent "${EB_RAMP_EXPONENT}" \
    --counter_bonus_a_multiplier "${COUNTER_BONUS_A_MULTIPLIER}" \
    --counter_bonus_b_multiplier "${COUNTER_BONUS_B_MULTIPLIER}" \
    --algorithm_name "${ALGORITHM}" \
    --sequence_horizon_steps "${SEQUENCE_HORIZON_STEPS}" \
    --osco_num_samples "${OSCO_NUM_SAMPLES}" \
    --osco_inner_gurobi_time_limit_seconds "${OSCO_INNER_GUROBI_TIME_LIMIT_SECONDS}" \
    --osco_n_threads "${OSCO_N_THREADS}" \
    --preprocessed_weather "${PREPROCESSED_WEATHER_PATH}" \
    --preprocessed_forecasts "${PREPROCESSED_FORECASTS_PATH}"

echo "========================================================"
echo "PULSAR run finished."
echo "Year: ${YEAR}"
echo "========================================================"
