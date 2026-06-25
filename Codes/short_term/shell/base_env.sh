#!/bin/bash
# ===================================================================
# ==  AOOSP - Base environment configuration                       ==
# ==  Edit this file to match your system. All other env files     ==
# ==  and simulation scripts source this file.                     ==
# ===================================================================

PYTHON_BIN=python
PROJECT_ROOT=/home/jpayan/AOOSP
SRC_DIR="${PROJECT_ROOT}/Codes/short_term/src"
DATA_DIR=/data/user_data/jpayan/AOOSP/data

# Root directory under which simulation outputs are written
OUTPUTS_ROOT=/data/user_data/jpayan/outputs/AOOSP

# Gurobi license file (can also be set externally before calling a script)
export GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-/opt/gurobi/license/gurobi.lic}
