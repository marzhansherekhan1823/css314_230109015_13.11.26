import ctypes, sys, time
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import stats_from, emit_verify_tag

M = 16_000_000
WARMUP, RUNS = 1, 5

def fs_worker(shm_name, slot_idx, iters, barrier):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    barrier.wait()                      # all workers start together
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & 0xFFFFFFFFFFFFFFFF
        arr[slot_idx] = acc
    del arr
    shm.close()

def run_once(p, stride):
    assert p * stride <= 512, "slots exceed 4096-byte block"
    shm = shared_memory.SharedMemory(create=True, size=4096)
    try:
        barrier = mp.Barrier(p + 1)
        procs = [mp.Process(target=fs_worker,
                            args=(shm.name, t * stride, M // p, barrier))
                 for t in range(p)]
        for pr in procs: pr.start()
        barrier.wait()
        t0 = time.perf_counter_ns()
        for pr in procs: pr.join()
        return (time.perf_counter_ns() - t0) / 1e6
    finally:
        shm.close(); shm.unlink()

def bench(p, stride):
    for _ in range(WARMUP): run_once(p, stride)
    return stats_from([run_once(p, stride) for _ in range(RUNS)])

if __name__ == "__main__":
    student_id, pphys = int(sys.argv[1]), int(sys.argv[2])
    plog = mp.cpu_count()

    print(f"== Stride sweep, p={pphys} ==")
    print("stride | bytes | median_ms | IQR_ms | Mops/s")
    med = {}
    for stride in (1, 2, 4, 8, 16, 32):
        if pphys * stride > 512:
            print(f"{stride:6d} | skipped (p*stride>512)"); continue
        st = bench(pphys, stride); med[stride] = st.median_ms
        print(f"{stride:6d} | {stride*8:5d} | {st.median_ms:9.1f} | {st.iqr_ms:6.1f} | {M/(st.median_ms*1e3):.2f}")

    ps, p = [], 1
    while p < plog: ps.append(p); p *= 2
    ps.append(plog)
    for stride in (1, 16):
        print(f"\n== Process scaling, stride={stride} ==")
        print("p | median_ms | Sp | Ep | Karp-Flatt e(p)")
        t1 = None
        for p in ps:
            if p * stride > 512: continue
            t = bench(p, stride).median_ms
            if t1 is None: t1 = t
            sp = t1 / t; ep = sp / p
            e = (1/sp - 1/p) / (1 - 1/p) if p >= 2 else float("nan")
            print(f"{p} | {t:9.1f} | {sp:5.2f} | {ep:5.2f} | {e:7.4f}")

    print(emit_verify_tag("PHASE1", student_id, med.get(1, 0.0), med.get(16, 0.0)))
