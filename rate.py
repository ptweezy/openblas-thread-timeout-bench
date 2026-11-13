"""Throughput of small-call loops for a fixed wall-clock budget, so that a pathological
setting cannot run for minutes. Meant for a pool as large as the CPU count (OpenBLAS
caps the pool there, so a larger THREADS behaves the same).

  python rate.py THREADS [SECONDS] [REPS]

Prints calls per second relative to the default timeout (paired within a repetition,
median over repetitions) and writes OUT_DIR/rate-T<THREADS>.json.
Environment: OUT_DIR, RATE_TIMEOUTS (default "default,24,22,20,18").
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))
TIMEOUTS = os.environ.get("RATE_TIMEOUTS", "default,24,22,20,18").split(",")


def child(budget):
    import numpy as np

    rng = np.random.default_rng(0)
    r = rng.standard_normal
    A100, b100 = r((100, 100)), r(100)
    A400, b400 = r((400, 400)), r(400)
    M1500, v0 = r((1500, 1500)) / 40, np.ones(1500)
    Wr, h0 = r((256, 256)) / 16, np.ones((256, 256))
    Xb = r((512, 784))
    W1, W2 = r((784, 512)) / 28, r((512, 512)) / 22
    A1000, B1000 = r((1000, 1000)), r((1000, 1000))

    def mlp():
        h = np.maximum(Xb @ W1, 0)
        return np.maximum(h @ W2, 0)

    work = {
        "solve 100x100": lambda: np.linalg.solve(A100, b100),
        "solve 400x400": lambda: np.linalg.solve(A400, b400),
        "matvec 1500 + norm": lambda: np.linalg.norm(M1500 @ v0),
        "tanh(W @ h) 256": lambda: np.tanh(Wr @ h0),
        "MLP forward 512x784": mlp,
        "matmul 1000x1000": lambda: A1000 @ B1000,
    }
    out = {}
    for name, fn in work.items():
        fn()
        time.sleep(0.3)  # start from an idle pool, as a fresh request would
        n = 0
        c0, w0 = time.process_time(), time.perf_counter()
        end = w0 + budget
        while time.perf_counter() < end:
            fn()
            n += 1
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        out[name] = {"rate": n / wall, "cores": cpu / wall}
    print("JSON " + json.dumps(out))


def parent():
    threads = sys.argv[1]
    budget = sys.argv[2] if len(sys.argv) > 2 else "3"
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    runs = {t: [] for t in TIMEOUTS}
    for rep_i in range(reps):
        k = rep_i % len(TIMEOUTS)
        for t in TIMEOUTS[k:] + TIMEOUTS[:k]:
            env = dict(os.environ, OPENBLAS_NUM_THREADS=threads)
            env.pop("OPENBLAS_THREAD_TIMEOUT", None)
            if t != "default":
                env["OPENBLAS_THREAD_TIMEOUT"] = t
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", budget], env=env, capture_output=True, text=True, check=True)
            runs[t].append(json.loads([l for l in p.stdout.splitlines() if l.startswith("JSON ")][-1][5:]))
        print(f"rep {rep_i + 1}/{reps} done", file=sys.stderr, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"rate-T{threads}.json"), "w") as f:
        json.dump({"threads": threads, "cpus": os.cpu_count(), "runs": runs}, f)

    names = list(runs[TIMEOUTS[0]][0])
    others = [t for t in TIMEOUTS if t != "default"]
    print(f"\n=== OPENBLAS_NUM_THREADS={threads} on {os.cpu_count()} CPUs: {reps} repetitions of {budget} s per loop ===")
    print(f"{'loop':<22}{'default calls/s':>16}{'cores':>7}" + "".join(f"{t + ' speedup':>13}{'cores':>7}" for t in others))
    for n in names:
        row = f"{n:<22}{statistics.median(r[n]['rate'] for r in runs['default']):>16.0f}{statistics.median(r[n]['cores'] for r in runs['default']):>7.1f}"
        for t in others:
            ratio = statistics.median(a[n]["rate"] / b[n]["rate"] for a, b in zip(runs[t], runs["default"]))
            row += f"{ratio:>12.2f}x{statistics.median(r[n]['cores'] for r in runs[t]):>7.1f}"
        print(row)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child(float(sys.argv[sys.argv.index("--child") + 1]))
    else:
        parent()
