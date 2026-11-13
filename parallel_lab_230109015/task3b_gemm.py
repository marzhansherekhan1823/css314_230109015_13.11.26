import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes, time, math
from harness import run_benchmark, emit_verify_tag

STUDENT_ID=230109015
N=384
TILES=(4,8,12,16,24,32,48,64,96,128)
A_NAME=B_NAME=C_NAME=None

def gemm_worker(args):
    variant, row0, row1, bs, a_name, b_name, c_name = args
    a_shm=shared_memory.SharedMemory(name=a_name)
    b_shm=shared_memory.SharedMemory(name=b_name)
    c_shm=shared_memory.SharedMemory(name=c_name)
    A=(ctypes.c_double*(N*N)).from_buffer(a_shm.buf)
    B=(ctypes.c_double*(N*N)).from_buffer(b_shm.buf)
    C=(ctypes.c_double*(N*N)).from_buffer(c_shm.buf)
    if variant==1:
        for i in range(row0,row1):
            for j in range(N):
                dot=0.0
                for k in range(N):
                    dot += A[i*N+k]*B[k*N+j]
                C[i*N+j]=dot
    elif variant==2:
        for i in range(row0,row1):
            base=i*N
            for k in range(N):
                aik=A[base+k]
                bk=k*N
                for j in range(N):
                    C[base+j] += aik*B[bk+j]
    else:
        for ii in range(row0,row1,bs):
            iend=min(row1,ii+bs)
            for kk in range(0,N,bs):
                kend=min(N,kk+bs)
                for jj in range(0,N,bs):
                    jend=min(N,jj+bs)
                    for i in range(ii,iend):
                        base=i*N
                        for k in range(kk,kend):
                            aik=A[base+k]
                            bk=k*N
                            for j in range(jj,jend):
                                C[base+j] += aik*B[bk+j]
    del A,B,C
    a_shm.close(); b_shm.close(); c_shm.close()

def main():
    global A_NAME,B_NAME,C_NAME
    p=int(input("Enter physical core count: ").strip())
    a=shared_memory.SharedMemory(create=True,size=N*N*8)
    b=shared_memory.SharedMemory(create=True,size=N*N*8)
    c=shared_memory.SharedMemory(create=True,size=N*N*8)
    A_NAME,B_NAME,C_NAME=a.name,b.name,c.name
    aa=(ctypes.c_double*(N*N)).from_buffer(a.buf)
    bb=(ctypes.c_double*(N*N)).from_buffer(b.buf)
    for i in range(N):
        for j in range(N):
            aa[i*N+j]=((i+j)%13)/13.0
            bb[i*N+j]=((i*2+j)%17)/17.0
    del aa,bb
    def run(v,bs=32):
        c.buf[:]=b'\0'*(N*N*8)
        jobs=[]
        for w in range(p):
            r0=N*w//p; r1=N*(w+1)//p
            jobs.append((v,r0,r1,bs,A_NAME,B_NAME,C_NAME))
        with mp.Pool(p) as pool:
            pool.map(gemm_worker,jobs)
    results={}
    try:
        for v in (1,2):
            stats=run_benchmark(lambda v=v:run(v),0,1)
            gflops=(2*N**3)/(stats.median_ms*1e6)
            results[v]=gflops
            print(f"Variant {v}: {stats.median_ms:.3f} ms, {gflops:.6f} GFLOP/s")
        best=None
        print("\nVariant 3 tile sweep")
        for bs in TILES:
            stats=run_benchmark(lambda bs=bs:run(3,bs),0,1)
            gflops=(2*N**3)/(stats.median_ms*1e6)
            footprint=3*bs*bs*8
            print(f"BS={bs:3d}, median={stats.median_ms:.3f} ms, GFLOP/s={gflops:.6f}, Wtile={footprint} bytes")
            if best is None or gflops>best[1]:
                best=(bs,gflops,stats.median_ms)
        print(f"\nBest tile: BS={best[0]}, GFLOP/s={best[1]:.6f}")
        print(emit_verify_tag("PHASE3",STUDENT_ID,results[1],best[1]))
    finally:
        a.close();a.unlink();b.close();b.unlink();c.close();c.unlink()

if __name__=="__main__":
    mp.freeze_support()
    main()
