#!/bin/bash
# ===================================================================
# ==  AOOSP - DSA calls environment configuration                  ==
# ==  Extends base_env.sh with paths specific to DSA generation.   ==
# ==  Edit DSA-specific paths below; edit base paths in            ==
# ==  base_env.sh.                                                 ==
# ===================================================================

# shellcheck source=base_env.sh
. "$(dirname "$0")/base_env.sh"

# DSA installation paths (relative to SRC_DIR)
DSA_BASE_DIR="${SRC_DIR}/DSA/DSA"
DSA_SRC_DIR="${DSA_BASE_DIR}/src"
POL_FILE_PATH="${DSA_BASE_DIR}"
DSA_LOG_DIR="${DSA_BASE_DIR}/logs"

# Root directory where preprocessed DSA score inputs are stored
PREPROCESSED_ROOT=/data/user_data/jpayan/outputs/AOOSP/preprocessed
