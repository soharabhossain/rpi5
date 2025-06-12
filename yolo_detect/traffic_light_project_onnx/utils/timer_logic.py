def compute_new_timings(base_time, count1, count2):
    delta = count1 - count2
    adjust = delta * 5  # 5 seconds per vehicle difference
    new_time_1 = max(30, base_time + adjust)
    new_time_2 = max(30, base_time - adjust)
    return new_time_1, new_time_2
