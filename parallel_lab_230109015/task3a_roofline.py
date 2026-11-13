import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
import time
from harness import run_benchmark

SIZE = 96 * 1_000_000
FLOPS_PER_WORKER = 80_000_000

def copy_worker(a_name, b_name, start, end):
    a = shared_memory.SharedMemory(name=a_name)
    b = shared_memory.SharedMemory(name=b_name)
    mv_a, mv_b = a.buf, b.buf
    mv_a[start:end] = mv_b[start:end]
    del mv_a, mv_b
    a.close(); b.close()

def fma_worker(flops):
    loops = flops // 64
    a, b = 1.0000001, 0.0000003
    xs = [float(i+1) for i in range(16)]
    for _ in range(loops):
        for j in range(16):
            xs[j] = xs[j]*a+b
            xs[j] = xs[j]*a+b
    return sum(xs)

def main():
    p = int(input("Enter physical core count: ").strip())
    a = shared_memory.SharedMemory(create=True, size=SIZE)
    b = shared_memory.SharedMemory(create=True, size=SIZE)
    try:
        a.buf[:] = b'\0' * SIZE
        b.buf[:] = b'\1' * SIZE
        def copy_all():
            ps=[]
            for i in range(p):
                start = SIZE*i//p; end = SIZE*(i+1)//p
                q=mp.Process(target=copy_worker,args=(a.name,b.name,start,end))
                q.start(); ps.append(q)
            for q in ps:
                q.join()
                if q.exitcode: raise RuntimeError("Copy worker failed")
        stats=run_benchmark(copy_all,1,5)
        bw=(2*SIZE)/(stats.median_ms/1000)/1e9
        print(f"DRAM copy median={stats.median_ms:.3f} ms; estimated bandwidth={bw:.4f} GB/s")
    finally:
        a.close(); a.unlink(); b.close(); b.unlink()
    t0=time.perf_counter_ns()
    with mp.Pool(p) as pool:
        pool.map(fma_worker,[FLOPS_PER_WORKER]*p)
    elapsed=(time.perf_counter_ns()-t0)/1e9
    total_flops=FLOPS_PER_WORKER*p
    peak=total_flops/elapsed/1e9
    print(f"Python compute elapsed={elapsed:.3f} s; estimated compute ceiling={peak:.6f} GFLOP/s")
    if bw:
        print(f"Ridge point={peak/bw:.6f} FLOP/Byte")

if __name__ == "__main__":
    mp.freeze_support()
    main()
