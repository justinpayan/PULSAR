import numpy as np


def convert_quotas(executive_quotas, realized_weather):
    total_time_quota = {}
    total_time = 0
    for t in realized_weather:
        if not np.isnan(realized_weather[t][0]) and not np.isnan(realized_weather[t][1]):
            total_time += 1
    for executive in executive_quotas:
        total_time_quota[executive] = (executive_quotas[executive][0]*total_time, executive_quotas[executive][1]*total_time)

    return total_time_quota