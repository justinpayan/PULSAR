#!/bin/bash
# ===================================================================
# ==  Preprocess weather + forecasts (real) for a single year      ==
# ==    1. preprocess_weather.py   -> realized weather pickle      ==
# ==    2. preprocess_forecasts.py -> real forecast pickle         ==
# ===================================================================

# shellcheck source=base_env.sh
. "$(dirname "$0")/base_env.sh"

YEAR=2023

OUTPUT_DIR="${DATA_DIR}/preprocessed/year_${YEAR}"

mkdir -p "${OUTPUT_DIR}"

START_DATE="${YEAR}-10-01"
END_YEAR=$((YEAR + 1))
END_DATE="${END_YEAR}-09-30"

SEED=31415
RMS_HORIZON_HOURS=16
RMS_AR_ORDER=4
RMS_DIFF_ORDER=0
RMS_MA_ORDER=1
RMS_DAILY_FOURIER_ORDER=4
RMS_YEARLY_FOURIER_ORDER=2

cd "${SRC_DIR}" || exit

echo "========================================================"
echo "Preprocessing weather & forecasts"
echo "Year: ${YEAR}"
echo "Date range: ${START_DATE} to ${END_DATE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "========================================================"

# --- Step 1: Preprocess realized weather ---
echo ""
echo "--- Step 1: preprocess_weather.py ---"
"${PYTHON_BIN}" preprocess_weather.py \
    --data_dir "${DATA_DIR}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output "${OUTPUT_DIR}/realized_weather.pkl"

echo ""
echo "--- Step 2: preprocess_forecasts.py (REAL) ---"
echo "    RMS mode: rolling UCM with prior-cycle training (${RMS_HORIZON_HOURS}h issuance/lookahead forecasts)"
"${PYTHON_BIN}" -u preprocess_forecasts.py \
    --data_dir "${DATA_DIR}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --seed ${SEED} \
    --rms_forecast_horizon_hours ${RMS_HORIZON_HOURS} \
    --rms_ar_order ${RMS_AR_ORDER} \
    --rms_diff_order ${RMS_DIFF_ORDER} \
    --rms_ma_order ${RMS_MA_ORDER} \
    --rms_daily_fourier_order ${RMS_DAILY_FOURIER_ORDER} \
    --rms_yearly_fourier_order ${RMS_YEARLY_FOURIER_ORDER} \
    --output "${OUTPUT_DIR}/forecasts_real.pkl"

echo "========================================================"
echo "Preprocessing complete for year ${YEAR}."
echo "Outputs:"
echo "  ${OUTPUT_DIR}/realized_weather.pkl"
echo "  ${OUTPUT_DIR}/forecasts_real.pkl"
echo "========================================================"
