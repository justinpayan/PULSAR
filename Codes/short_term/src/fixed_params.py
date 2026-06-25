import os

NUM_WEATHER_BINS = 5
JOBS_PER_PROJECT_RANGE = (1, 1)
JOB_LENGTH_RANGE = (1, 1)
# NUM_EXECUTIVES = 4
# EXECUTIVES = ['EU', 'NA', 'EA', 'CL']
# EXECUTIVE_QUOTAS = {
#     'EU': (0.00, 1.00),
#     'NA': (0.00, 1.00),
#     'EA': (0.00, 1.00),
#     'CL': (0.00, 1.00),
# }
EXPECT_SAMP_SIZE = 50
FILLERS_PER_EXEC = 20

TIME_INTERVAL_MINUTES = 30

NUM_EXECUTIVES = 5
EXECUTIVES = ['EU', 'NA', 'EA', 'CL', 'OTHER']
EXECUTIVE_QUOTAS = {
    'EU': (0.3375, 0.34),
    'NA': (0.3375, 0.34),
    'EA': (0.225, 0.23),
    'CL': (0.1, 0.11),
    'OTHER': (0.0, 1.00)
}

OSCO_SAMP_SIZE = 10

DSA_WEIGHTS = {
    "cond":         float(os.environ.get("DSA_W_COND",         0.35)),
    "array":        float(os.environ.get("DSA_W_ARRAY",        0.10)),
    "sbcompletion": float(os.environ.get("DSA_W_SBCOMPLETION", 0.15)),
    "sciencerank":  float(os.environ.get("DSA_W_SCIENCERANK",  0.10)),
    "cyclegrade":   float(os.environ.get("DSA_W_CYCLEGRADE",   0.20)),
    "ha":           float(os.environ.get("DSA_W_HA",           0.10)),
}


# Noisier/harder setting:
# JOB_DROP_FRAC = 0.92
#
# BASE_STD_PWV_FORECAST = 0.5
# BASE_STD_RMS_FORECAST = 100
# STD_INCREASE_FACTOR_PWV_FORECAST = 0.2
# STD_INCREASE_FACTOR_RMS_FORECAST = 70

#Easier setting:
JOB_DROP_FRAC = 0.9

BASE_STD_PWV_FORECAST = 0.1
BASE_STD_RMS_FORECAST = 50
STD_INCREASE_FACTOR_PWV_FORECAST = 0.01
STD_INCREASE_FACTOR_RMS_FORECAST = 10