import sys, time
import multiprocessing as mp
from harness import emit_verify_tag

if __name__ == "__main__":
    student_id = int(sys.argv[1])          # numeric Student ID, e.g. python task0_cal.py 12345
    plog = mp.cpu_count()
    n = 10**6
    t0 = time.perf_counter_ns()
    for _ in range(n):
        time.perf_counter_ns()
    avg_ns = (time.perf_counter_ns() - t0) / n
    print(f"P_log={plog}  avg perf_counter_ns call latency = {avg_ns:.2f} ns")
    print(emit_verify_tag("PHASE0", student_id, float(plog), avg_ns))
