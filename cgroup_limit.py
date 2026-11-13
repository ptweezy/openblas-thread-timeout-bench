"""One process in a container with a CPU quota smaller than the machine (docker --cpus=N,
Kubernetes CPU limits). OpenBLAS sizes its pool from the visible CPUs, not the quota, so
idle spinning is charged against the quota and the kernel throttles the whole container.

Run inside the limited container:  python cgroup_limit.py LABEL
Reuses the request loop of multiproc.py (one 256x256 matmul + 5 ms single-threaded work).
Environment: CGROUP_RUNS (default 3; the run with the median wall time is reported),
CGROUP_TIMEOUTS (default "default,24,22,20").
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def cgroup_dir():
    try:
        rel = open("/proc/self/cgroup").read().strip().splitlines()[-1].split("::", 1)[1]
    except (OSError, IndexError):
        rel = "/"
    return os.path.join("/sys/fs/cgroup", rel.lstrip("/"))


def cpu_stat():
    out = {}
    try:
        with open(os.path.join(cgroup_dir(), "cpu.stat")) as f:
            for line in f:
                k, v = line.split()
                out[k] = int(v)
    except OSError:
        pass
    return out


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "?"
    timeouts = os.environ.get("CGROUP_TIMEOUTS", "default,24,22,20").split(",")
    n_runs = int(os.environ.get("CGROUP_RUNS", "3"))
    try:
        quota = open(os.path.join(cgroup_dir(), "cpu.max")).read().split()
        limit = "unlimited" if quota[0] == "max" else f"{int(quota[0]) / int(quota[1]):g} CPUs"
    except OSError:
        limit = "unknown"
    print(f"--- {label}: cgroup CPU limit {limit}, {os.cpu_count()} CPUs visible, median of {n_runs} runs ---")
    print(f"{'timeout':<10}{'wall s':>9}{'req/s':>9}{'CPU s':>9}{'p50 ms':>9}{'p99 ms':>9}{'throttled':>11}{'thr. ms':>9}")
    runs = {t: [] for t in timeouts}
    for i in range(n_runs):
        k = i % len(timeouts)
        for t in timeouts[k:] + timeouts[:k]:
            env = dict(os.environ)
            env.pop("OPENBLAS_THREAD_TIMEOUT", None)
            env.pop("OPENBLAS_NUM_THREADS", None)
            if t != "default":
                env["OPENBLAS_THREAD_TIMEOUT"] = t
            s0 = cpu_stat()
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, "multiproc.py"), "--child", str(time.time() + 1.0)],
                env=env, capture_output=True, text=True, check=True,
            )
            s1 = cpu_stat()
            r = json.loads(p.stdout.strip().splitlines()[-1])
            r["throttled"] = s1.get("nr_throttled", 0) - s0.get("nr_throttled", 0)
            r["throttled_ms"] = (s1.get("throttled_usec", 0) - s0.get("throttled_usec", 0)) / 1e3
            runs[t].append(r)
    for t in timeouts:
        mid = sorted(runs[t], key=lambda r: r["wall"])[len(runs[t]) // 2]
        print(
            f"{t:<10}{mid['wall']:>9.2f}{200 / mid['wall']:>9.0f}{mid['cpu']:>9.1f}{mid['p50_ms']:>9.2f}"
            f"{mid['p99_ms']:>9.2f}{mid['throttled']:>11d}{mid['throttled_ms']:>9.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
