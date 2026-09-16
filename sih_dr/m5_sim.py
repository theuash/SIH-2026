"""M5 — District throughput model. Mirrors runScreeningSim.m contract.

Analytic M/M/c approximation (stdlib only — no simpy/Simulink needed for MVP).
The .slx port maps each stage to a Queue+Server block with these same params.
"""
import math

DAY = 8 * 3600  # working seconds per day


def _mmc_wait(arrival_s, service_s, c):
    """Mean queue wait (s) for M/M/c. 0 when underloaded tiny; inf when rho>=1."""
    if c < 1 or service_s <= 0:
        return 0.0
    rho = arrival_s * service_s / c
    if rho >= 1:
        return float("inf")
    if rho < 0.05:
        return 0.0
    # Erlang-C lite
    a = c * rho
    s = sum(a ** k / math.factorial(k) for k in range(c))
    p0 = 1 / (s + a ** c / (math.factorial(c) * (1 - rho)))
    pc = (a ** c / (math.factorial(c) * (1 - rho))) * p0
    return pc * service_s / (c * (1 - rho))


def runScreeningSim(patients_per_year=100_000, centers=6, image_mb=1.2,
                    bandwidth_mbps=4.0, proc_sec=3.5, review_sec=30.0,
                    referable_frac=0.28, n_servers=1, n_reviewers=3,
                    peak_factor=1.8):
    per_day = patients_per_year / 365 * peak_factor
    arrival_s = per_day / DAY  # patients/sec district-wide
    upload_s = image_mb * 8 / max(bandwidth_mbps, 0.1)
    proc_wait = _mmc_wait(arrival_s / max(centers, 1) * centers, proc_sec, max(n_servers, 1))
    rev_arrival = arrival_s * referable_frac
    rev_wait = _mmc_wait(rev_arrival, review_sec, max(n_reviewers, 1))
    turnaround_min = (upload_s + proc_sec + proc_wait + review_sec * referable_frac + rev_wait) / 60
    bottleneck = min({"acquisition": 0, "bandwidth": upload_s,
                      "processing": proc_sec + proc_wait,
                      "review": review_sec * referable_frac + rev_wait}.items(),
                     key=lambda kv: -kv[1])[0]
    capacity_day = int(DAY / max(proc_sec, 0.1)) * max(n_servers, 1)
    ok = turnaround_min < 120 and rev_wait != float("inf")
    return {"patients_per_day": round(per_day, 1),
            "arrival_per_sec": round(arrival_s, 4),
            "upload_sec": round(upload_s, 2),
            "processing_wait_sec": round(proc_wait, 1),
            "review_wait_sec": round(0 if rev_wait == float("inf") else rev_wait, 1),
            "turnaround_min": round(turnaround_min, 1),
            "capacity_per_day": capacity_day,
            "bottleneck": bottleneck,
            "meets_target": bool(ok),
            "recommendation": (f"{centers} centers, {n_servers} server(s), "
                               f"{n_reviewers} reviewer(s) -> ~{turnaround_min:.1f}min turnaround")}
