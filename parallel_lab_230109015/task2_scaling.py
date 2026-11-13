import math
import multiprocessing as mp
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230109015
N_STRONG = 1_600_000
N0_WEAK = 250_000
REPS = 35

def serial_stage(n):
    total = 0.0
    count = max(1, int(0.05*n))
    for i in range(count):
        x = (i % 1000) + 1.0
        for j in range(6):
            x = math.sin(x) + math.sqrt(x + 1.0)
        total += x
    return total

def parallel_chunk(bounds):
    start, end = bounds
    total = 0.0
    for i in range(start, end):
        x = (i % 1000) + 1.0
        for _ in range(REPS):
            x = x * 1.0000001 + 0.0000003
            x = math.sin(x)
        total += x
    return total

def run_pipeline(pool, n, p):
    a0 = __import__("time").perf_counter_ns()
    serial_stage(n)
    stage_a_ms = (__import__("time").perf_counter_ns()-a0)/1e6
    chunk = (n+p-1)//p
    ranges = [(i*chunk, min(n, (i+1)*chunk)) for i in range(p)]
    pool.map(parallel_chunk, ranges)
    return stage_a_ms

def main():
    logical = mp.cpu_count()
    physical = int(input("Enter physical core count: ").strip())
    strong_times = {}
    print("2A Strong scaling (N=1,600,000)")
    for p in range(1, logical+1):
        with mp.Pool(processes=p) as pool:
            stats = run_benchmark(lambda: run_pipeline(pool, N_STRONG, p), 1, 3)
        strong_times[p] = stats.median_ms
        print(f"p={p:2d}, median={stats.median_ms:.3f} ms")
    t1 = strong_times[1]
    print("\nStrong speedup and efficiency")
    for p, tp in strong_times.items():
        s = t1/tp
        print(f"p={p:2d}, S={s:.4f}, E={s/p:.4f}")
    with mp.Pool(processes=1) as pool:
        stage_a = run_pipeline(pool, N_STRONG, 1)
    serial_fraction = stage_a/t1 if t1 else 0
    print(f"\nMeasured serial fraction s={serial_fraction:.6f}")
    print(f"Amdahl S_inf={1/serial_fraction:.4f}" if serial_fraction else "Amdahl S_inf=undefined")
    print(f"Amdahl S at logical threads={1/(serial_fraction+(1-serial_fraction)/logical):.4f}" if serial_fraction else "")
    print("\n2B Weak scaling (N=p*250,000)")
    weak = {}
    for p in range(1, logical+1):
        n = p*N0_WEAK
        with mp.Pool(processes=p) as pool:
            stats = run_benchmark(lambda: run_pipeline(pool, n, p), 1, 3)
        weak[p] = stats.median_ms
        eff = weak[1]/stats.median_ms if 1 in weak else 1.0
        print(f"p={p:2d}, N={n:,}, median={stats.median_ms:.3f} ms, weak_eff={eff:.4f}")
    p = logical
    tp = strong_times[p]
    sp = t1/tp
    weak_t1 = weak[1]
    weak_tp = weak[p]
    # Stage A measured separately for weak workload at p=logical
    alpha = serial_stage(p*N0_WEAK)  # diagnostic work proxy; report measured stage-A timing separately if needed
    with mp.Pool(processes=p) as pool:
        stage_a_ms = run_pipeline(pool, p*N0_WEAK, p)
    alpha_fraction = stage_a_ms/weak_tp if weak_tp else 0
    gust = p-alpha_fraction*(p-1)
    print(f"\nAt p={logical}: strong speedup={sp:.4f}, weak efficiency={weak_t1/weak_tp:.4f}, measured alpha={alpha_fraction:.6f}, Gustafson={gust:.4f}")
    print(emit_verify_tag("PHASE2", STUDENT_ID, sp, gust))

if __name__ == "__main__":
    mp.freeze_support()
    main()
