import multiprocessing as mp
import platform
import sys
import time
from harness import emit_verify_tag

STUDENT_ID = 230109015

def main():
    print("Student ID:", STUDENT_ID)
    print("CPU logical threads:", mp.cpu_count())
    print("Python:", sys.version.replace("\n", " "))
    print("OS:", platform.platform())
    reads = 2_000_000
    t0 = time.perf_counter_ns()
    for _ in range(reads):
        time.perf_counter_ns()
    elapsed = time.perf_counter_ns() - t0
    avg = elapsed / reads
    print(f"Clock reads: {reads:,}")
    print(f"Total elapsed: {elapsed:,} ns")
    print(f"Average call latency: {avg:.4f} ns")
    print(emit_verify_tag("PHASE0", STUDENT_ID, avg, mp.cpu_count()))

if __name__ == "__main__":
    mp.freeze_support()
    main()
