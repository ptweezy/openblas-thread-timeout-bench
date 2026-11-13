"""Several worker processes on one machine, each with its own default-sized OpenBLAS pool
(the gunicorn / pytest-xdist / job-runner situation). Each "request" is one threaded
matmul followed by single-threaded Python work. Nothing limits the BLAS thread count,
as in an unconfigured deployment.

  python multiproc.py            sweeps process counts and timeouts
  python multiproc.py --child T  runs the request loop starting at time T, prints JSON

Environment: OUT_DIR, MULTIPROC_PROCS (default "1,2,4,8"), MULTIPROC_RUNS (default 3;
the run with the median wall time is reported), MULTIPROC_TIMEOUTS (default "default,24,22,20").
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))
REQUESTS = 200
WORK_S = 0.005  # single-threaded Python work per request
N = 256


def child():
    import numpy as np

    rng = np.random.default_rng(os.getpid())
    a, b = rng.standard_normal((N, N)), rng.standard_normal((N, N))
    a @ b
    start = float(sys.argv[2])  # all children begin together
    while time.time() < start:
        time.sleep(0.001)
    lat = []
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(REQUESTS):
        t0 = time.perf_counter()
        a @ b
        end = time.perf_counter() + WORK_S
        x = 0
        while time.perf_counter() < end:
            x += 1
        lat.append(time.perf_counter() - t0)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    lat.sort()
    print(json.dumps({"wall": wall, "cpu": cpu, "p50_ms": lat[len(lat) // 2] * 1e3, "p99_ms": lat[int(len(lat) * 0.99)] * 1e3}))


def run(procs, timeout):
    env = dict(os.environ)
    env.pop("OPENBLAS_THREAD_TIMEOUT", None)
    env.pop("OPENBLAS_NUM_THREADS", None)
    if timeout != "default":
        env["OPENBLAS_THREAD_TIMEOUT"] = timeout
    start = time.time() + 2.0
    ps = [
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", str(start)], env=env, stdout=subprocess.PIPE, text=True)
        for _ in range(procs)
    ]
    outs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
    wall = max(o["wall"] for o in outs)
    return {
        "wall": wall,
        "cpu": sum(o["cpu"] for o in outs),
        "p50_ms": sorted(o["p50_ms"] for o in outs)[len(outs) // 2],
        "p99_ms": max(o["p99_ms"] for o in outs),
        "rps": procs * REQUESTS / wall,
    }


def parent():
    timeouts = os.environ.get("MULTIPROC_TIMEOUTS", "default,24,22,20").split(",")
    proc_counts = [int(p) for p in os.environ.get("MULTIPROC_PROCS", "1,2,4,8").split(",")]
    n_runs = int(os.environ.get("MULTIPROC_RUNS", "3"))
    print(f"{REQUESTS} requests per process; each = one {N}x{N} matmul + {WORK_S * 1e3:.0f} ms single-threaded work")
    print(f"(the single-threaded work alone is {REQUESTS * WORK_S:.2f} s per process; {os.cpu_count()} CPUs; median of {n_runs} runs)\n")
    results = {}
    for procs in proc_counts:
        runs = {t: [] for t in timeouts}
        for i in range(n_runs):
            k = i % len(timeouts)
            for t in timeouts[k:] + timeouts[:k]:
                runs[t].append(run(procs, t))
        print(f"--- {procs} process(es) ---")
        print(f"{'timeout':<10}{'wall s':>9}{'req/s':>9}{'CPU s':>9}{'p50 ms':>9}{'p99 ms':>9}")
        for t in timeouts:
            mid = sorted(runs[t], key=lambda r: r["wall"])[len(runs[t]) // 2]
            results[f"{procs}:{t}"] = runs[t]
            print(f"{t:<10}{mid['wall']:>9.2f}{mid['rps']:>9.0f}{mid['cpu']:>9.1f}{mid['p50_ms']:>9.2f}{mid['p99_ms']:>9.2f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "multiproc.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    child() if "--child" in sys.argv else parent()
