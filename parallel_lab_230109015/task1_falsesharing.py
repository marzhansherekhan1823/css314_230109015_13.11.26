import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import time
from harness import run_benchmark, emit_verify_tag, powers_of_two_up_to

STUDENT_ID = 230109015
M = 24_000_000

def fs_worker(name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc
    del arr
    shm.close()

def run_case(p, stride):
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        workers = []
        per_worker = M // p
        t0 = time.perf_counter_ns()
        for t in range(p):
            proc = mp.Process(target=fs_worker, args=(shm.name, t*stride, per_worker))
            proc.start()
            workers.append(proc)
        for proc in workers:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker failed: exit code {proc.exitcode}")
        return (time.perf_counter_ns()-t0)/1e6
    finally:
        shm.close()
        shm.unlink()

def main():
    logical = mp.cpu_count()
    physical = int(input("Enter physical core count from Task 0 / Windows CPU info: ").strip())
    pvals = powers_of_two_up_to(logical)
    print("\nTable 1A: stride sweep at physical core count")
    stride_results = {}
    for stride in (1, 2, 4, 8, 16, 32):
        stats = run_benchmark(lambda s=stride: run_case(physical, s), 2, 7)
        stride_results[stride] = stats.median_ms
        throughput = M / (stats.median_ms/1000) / 1e6
        print(f"stride={stride:2d}, separation={stride*8:3d} B, median={stats.median_ms:.3f} ms, IQR={stats.iqr_ms:.3f} ms, throughput={throughput:.3f} Mops/s")
    print("\nTable 1B: process scaling")
    for stride in (1, 16):
        t1 = run_benchmark(lambda: run_case(1, stride), 2, 7).median_ms
        print(f"\nstride={stride}, T1={t1:.3f} ms")
        for p in pvals:
            tp = run_benchmark(lambda p=p: run_case(p, stride), 2, 7).median_ms
            speedup = t1/tp
            efficiency = speedup/p
            karp = ((1/speedup)-(1/p))/(1-(1/p)) if p >= 2 else float("nan")
            print(f"p={p:2d}, Tp={tp:.3f} ms, S={speedup:.4f}, E={efficiency:.4f}, e={karp:.6f}")
    print("\n" + emit_verify_tag("PHASE1", STUDENT_ID, stride_results[1], stride_results[16]))

if __name__ == "__main__":
    mp.freeze_support()
    main()
