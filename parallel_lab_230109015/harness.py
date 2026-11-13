from dataclasses import dataclass
import multiprocessing as mp
import statistics
import time

@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float
    max_ms: float

def fnv1a_hash(s: str) -> int:
    h = 14695981039346656037
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def run_benchmark(fn, warmup_runs=2, timed_runs=7):
    for _ in range(warmup_runs):
        fn()
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    mid = len(samples) // 2
    median = samples[mid] if len(samples) % 2 else (samples[mid-1] + samples[mid]) / 2
    q25 = samples[len(samples)//4]
    q75 = samples[(3*len(samples))//4]
    return BenchStats(samples[0], median, q75-q25, samples[-1])

def emit_verify_tag(task_id, student_seed, primary_metric, secondary_metric):
    hw = mp.cpu_count()
    payload = f"{task_id}|{student_seed}|{hw}|{primary_metric:.4f}|{secondary_metric:.4f}"
    return f"VERIFY_HASH [{task_id}]: {fnv1a_hash(payload):016X} (P_log={hw})"

def powers_of_two_up_to(n):
    vals = [1]
    x = 2
    while x <= n:
        vals.append(x)
        x *= 2
    if n not in vals:
        vals.append(n)
    return sorted(set(vals))
