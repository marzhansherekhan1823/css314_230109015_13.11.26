import math
import multiprocessing as mp
import time
from harness import emit_verify_tag

STUDENT_ID=230109015
N=12000
CHUNKS=(1,4,16,64,256,1024)

def item_work(i):
    k=1+(i//60)**2
    x=(i%97)+1.0
    for _ in range(k):
        x=math.sin(x)+math.cos(x)+math.sqrt(x+1.0)
    return x

def worker_static(indices, times, wid):
    t0=time.perf_counter_ns()
    checksum=0.0
    for i in indices:
        checksum+=item_work(i)
    times[wid]=(time.perf_counter_ns()-t0)/1e6
    return checksum

def worker_dynamic(counter, lock, chunk, times, wid):
    t0=time.perf_counter_ns()
    checksum=0.0
    while True:
        with lock:
            start=counter.value
            counter.value += chunk
        if start>=N:
            break
        for i in range(start,min(N,start+chunk)):
            checksum+=item_work(i)
    times[wid]=(time.perf_counter_ns()-t0)/1e6
    return checksum

def run_static(p, cyclic):
    times=mp.Array('d',p)
    procs=[]
    t0=time.perf_counter_ns()
    for w in range(p):
        if cyclic:
            inds=list(range(w,N,p))
        else:
            lo=N*w//p; hi=N*(w+1)//p
            inds=range(lo,hi)
        q=mp.Process(target=worker_static,args=(inds,times,w))
        q.start();procs.append(q)
    for q in procs:
        q.join()
        if q.exitcode: raise RuntimeError("Static worker failed")
    wall=(time.perf_counter_ns()-t0)/1e6
    vals=list(times)
    lam=max(vals)/(sum(vals)/p) if sum(vals) else 0
    return wall,lam,vals

def run_dynamic(p,chunk):
    times=mp.Array('d',p)
    counter=mp.Value('i',0)
    lock=mp.Lock()
    procs=[]
    t0=time.perf_counter_ns()
    for w in range(p):
        q=mp.Process(target=worker_dynamic,args=(counter,lock,chunk,times,w))
        q.start();procs.append(q)
    for q in procs:
        q.join()
        if q.exitcode: raise RuntimeError("Dynamic worker failed")
    wall=(time.perf_counter_ns()-t0)/1e6
    vals=list(times)
    lam=max(vals)/(sum(vals)/p) if sum(vals) else 0
    return wall,lam,vals

def main():
    p=int(input("Enter physical core count: ").strip())
    print("Strategy 1 - contiguous static")
    s1,l1,_=run_static(p,False)
    print(f"Wall={s1:.3f} ms, lambda={l1:.4f}")
    print("Strategy 2 - cyclic static")
    s2,l2,_=run_static(p,True)
    print(f"Wall={s2:.3f} ms, lambda={l2:.4f}")
    best=None
    print("Strategy 3 - dynamic chunks")
    for c in CHUNKS:
        wall,lam,_=run_dynamic(p,c)
        print(f"chunk={c:4d}, wall={wall:.3f} ms, lambda={lam:.4f}")
        if best is None or wall<best[1]:
            best=(c,wall,lam)
    print(f"Best chunk={best[0]}, wall={best[1]:.3f} ms, lambda={best[2]:.4f}")
    print(emit_verify_tag("PHASE4",STUDENT_ID,s1,best[1]))

if __name__=="__main__":
    mp.freeze_support()
    main()
