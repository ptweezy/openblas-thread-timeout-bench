"""Wall time and CPU time of ordinary NumPy / SciPy / scikit-learn workloads under
different OPENBLAS_THREAD_TIMEOUT values (PyPI wheels, which bundle a pthreads OpenBLAS).

  python workloads.py THREADS [REPS]      prints tables, writes OUT_DIR/workloads-T<THREADS>.json
  python workloads.py --child             runs every workload once, prints JSON

Each repetition runs every timeout once, in an order that rotates between repetitions,
and each timeout is compared with the default of the same repetition (paired ratios).
The tables show the median over repetitions.

CPU time is process CPU (all threads) from the start of a workload until 150 ms after
it returns, so the spin that follows the last BLAS call is charged to it. A workload
that exceeds WORKLOADS_CAP_S stops early and is scaled to its full size (marked *).

Environment: OUT_DIR, WORKLOADS_TIMEOUTS (default "default,24,22,20,18"),
WORKLOADS_CAP_S (default 15), WORKLOADS_QUICK (1/8 of the work, for smoke tests).
"""
import json
import math
import os
import statistics
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))
TIMEOUTS = os.environ.get("WORKLOADS_TIMEOUTS", "default,24,22,20,18").split(",")
CAP = float(os.environ.get("WORKLOADS_CAP_S", "15"))
QUICK = bool(os.environ.get("WORKLOADS_QUICK"))


def build():
    import numpy as np
    import scipy.linalg as sla
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(0)
    r = rng.standard_normal
    W = {}

    def rep(k, fn):
        """Call fn k times; stop early past the cap and return the fraction completed."""
        k = max(1, k // 8) if QUICK else k

        def run():
            end = time.perf_counter() + CAP
            for i in range(k):
                fn()
                if i + 1 < k and time.perf_counter() > end:
                    return (i + 1) / k
            return 1.0

        return run

    # ---- dense LAPACK: long parallel regions with short serial sections inside ----
    A3000, b3000 = r((3000, 3000)), r(3000)
    W["lapack: LU solve 3000"] = rep(6, lambda: np.linalg.solve(A3000, b3000))
    S3000 = A3000 @ A3000.T + 3000 * np.eye(3000)
    W["lapack: cholesky 3000"] = rep(10, lambda: np.linalg.cholesky(S3000))
    A800 = r((800, 800))
    W["lapack: svd 800"] = rep(2, lambda: np.linalg.svd(A800))
    S1200 = A3000[:1200, :1200] + A3000[:1200, :1200].T
    W["lapack: eigh 1200"] = rep(2, lambda: np.linalg.eigh(S1200))
    A1500 = r((1500, 1500))
    W["lapack: qr 1500"] = rep(4, lambda: np.linalg.qr(A1500))
    X4000, y4000 = r((4000, 300)), r(4000)
    W["lapack: lstsq 4000x300"] = rep(4, lambda: np.linalg.lstsq(X4000, y4000, rcond=None))
    A600 = r((600, 600))
    W["lapack: inv 600"] = rep(20, lambda: np.linalg.inv(A600))
    A400e = r((400, 400)) / 20
    W["scipy: expm 400"] = rep(5, lambda: sla.expm(A400e))

    # ---- many small/medium calls with interpreter work in between ----
    A100, b100 = r((100, 100)), r(100)
    W["loop: solve 100 x20000"] = rep(20000, lambda: np.linalg.solve(A100, b100))
    A400, b400 = r((400, 400)), r(400)
    W["loop: solve 400 x1500"] = rep(1500, lambda: np.linalg.solve(A400, b400))

    M1500 = r((1500, 1500)) / 40
    pv = {"v": np.ones(1500)}

    def power_step():
        v = M1500 @ pv["v"]
        pv["v"] = v / np.linalg.norm(v)

    W["loop: power iteration 1500 x2000"] = rep(2000, power_step)

    Wr = r((256, 256)) / 16
    hs = {"h": np.ones((256, 256))}

    def rnn_step():
        hs["h"] = np.tanh(Wr @ hs["h"])

    W["loop: tanh(W @ h) 256 x4000"] = rep(4000, rnn_step)

    Xb = r((512, 784))
    W1, W2, W3 = r((784, 512)) / 28, r((512, 512)) / 22, r((512, 10)) / 22

    def mlp_forward():
        h = np.maximum(Xb @ W1, 0)
        h = np.maximum(h @ W2, 0)
        z = h @ W3
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    W["loop: numpy MLP forward x400"] = rep(400, mlp_forward)
    Xc = r((5000, 400))
    W["numpy: cov 5000x400"] = rep(10, lambda: np.cov(Xc, rowvar=False))

    # ---- scikit-learn ----
    Xl = r((20000, 100))
    yl = (Xl[:, :5].sum(axis=1) + 0.5 * r(20000) > 0).astype(int)
    W["sklearn: LogisticRegression fit"] = rep(3, lambda: LogisticRegression(max_iter=200).fit(Xl, yl))
    Xr, yr = r((50000, 200)), r(50000)
    W["sklearn: Ridge fit 50000x200"] = rep(4, lambda: Ridge().fit(Xr, yr))
    W["sklearn: LinearRegression 50000x200"] = rep(2, lambda: LinearRegression().fit(Xr, yr))
    Xp = r((5000, 300))
    W["sklearn: PCA full 5000x300"] = rep(3, lambda: PCA(n_components=20, svd_solver="full").fit(Xp))
    Xpr = r((20000, 300))
    W["sklearn: PCA randomized 20000x300"] = rep(
        3, lambda: PCA(n_components=20, svd_solver="randomized", random_state=0).fit(Xpr)
    )
    Xk = r((100000, 20))
    W["sklearn: KMeans 100000x20"] = rep(1, lambda: KMeans(n_clusters=8, n_init=1, max_iter=20, random_state=0).fit(Xk))
    Xg = r((600, 5))
    yg = np.sin(Xg).sum(axis=1)
    W["sklearn: GaussianProcess fit 600"] = rep(1, lambda: GaussianProcessRegressor(random_state=0).fit(Xg, yg))
    Xm = r((5000, 100))
    ym = (Xm[:, 0] * Xm[:, 1] > 0).astype(int)

    def mlp_fit():
        return MLPClassifier(hidden_layer_sizes=(128,), max_iter=15, random_state=0).fit(Xm, ym)

    W["sklearn: MLPClassifier fit"] = rep(1, mlp_fit)
    clf = mlp_fit()
    Xq = r((512, 100))
    W["sklearn: MLP predict x3000 back-to-back"] = rep(3000, lambda: clf.predict_proba(Xq))

    def request(pause):
        def one():
            clf.predict_proba(Xq)
            time.sleep(pause)

        return one

    W["service: MLP predict every 20 ms x100"] = rep(100, request(0.02))
    W["service: MLP predict every 200 ms x15"] = rep(15, request(0.2))
    return W


def child():
    warnings.simplefilter("ignore")
    W = build()
    time.sleep(0.3)
    out = {}
    for name, fn in W.items():
        c0, w0 = time.process_time(), time.perf_counter()
        frac = fn()
        wall = time.perf_counter() - w0
        time.sleep(0.15)
        out[name] = {"wall": wall, "cpu": time.process_time() - c0, "frac": frac}
        time.sleep(0.1)
    print("JSON " + json.dumps(out))


def parent():
    threads = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    runs = {t: [] for t in TIMEOUTS}
    for rep_i in range(reps):
        k = rep_i % len(TIMEOUTS)
        for t in TIMEOUTS[k:] + TIMEOUTS[:k]:  # rotate the order each repetition
            env = dict(os.environ, OPENBLAS_NUM_THREADS=threads)
            env.pop("OPENBLAS_THREAD_TIMEOUT", None)
            if t != "default":
                env["OPENBLAS_THREAD_TIMEOUT"] = t
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"], env=env, capture_output=True, text=True)
            if p.returncode != 0:
                print(p.stderr[-3000:], file=sys.stderr)
                sys.exit(1)
            line = [l for l in p.stdout.splitlines() if l.startswith("JSON ")][-1]
            runs[t].append(json.loads(line[5:]))
            print(f"rep {rep_i + 1}/{reps} timeout={t} done", file=sys.stderr, flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"workloads-T{threads}.json"), "w") as f:
        json.dump({"threads": threads, "reps": reps, "cpus": os.cpu_count(), "runs": runs}, f)

    names = list(runs["default"][0])
    others = [t for t in TIMEOUTS if t != "default"]

    def full(run, name, key):  # scale a capped run to its full size
        return run[name][key] / run[name]["frac"]

    print(f"\n=== OPENBLAS_NUM_THREADS={threads} on {os.cpu_count()} CPUs, {reps} repetitions ===")
    print("Per timeout: median over repetitions of (this timeout / default of the same repetition).")
    for key, title in (("wall", "WALL time"), ("cpu", "CPU time, all threads")):
        print(f"\n{title}")
        print(f"{'workload':<42}{'default s':>10}" + "".join(f"{t:>8}" for t in others) + "   noise")
        logs = {t: [] for t in others}
        for n in names:
            base = statistics.median(full(r, n, key) for r in runs["default"])
            capped = any(r[n]["frac"] < 1 for t in TIMEOUTS for r in runs[t])
            row = f"{(n + (' *' if capped else '')):<42}{base:>10.3f}"
            spreads = []
            for t in others:
                ratios = sorted(full(a, n, key) / full(b, n, key) for a, b in zip(runs[t], runs["default"]))
                med = statistics.median(ratios)
                logs[t].append(math.log(med))
                row += f"{med:>7.2f}x"
                if len(ratios) >= 4:
                    q = statistics.quantiles(ratios, n=4)
                    spreads.append((q[2] - q[0]) / med)
            row += f"   {100 * statistics.median(spreads):.0f}%" if spreads else ""
            print(row)
        print(f"{'geometric mean':<42}{'':>10}" + "".join(f"{math.exp(sum(logs[t]) / len(logs[t])):>7.2f}x" for t in others))
        print(f"{'workloads slower by more than 10%':<42}{'':>10}" + "".join(f"{sum(1 for x in logs[t] if x > math.log(1.10)):>8}" for t in others))
        print(f"{'workloads faster by more than 10%':<42}{'':>10}" + "".join(f"{sum(1 for x in logs[t] if x < math.log(0.90)):>8}" for t in others))
        tot = {t: statistics.median(sum(full(r, n, key) for n in names) for r in runs[t]) for t in TIMEOUTS}
        print(f"{'total seconds':<42}{tot['default']:>10.2f}" + "".join(f"{tot[t]:>8.2f}" for t in others))
    print("\nnoise = interquartile range of the paired ratios, relative to their median, median across timeouts")
    print("* = hit the time cap in at least one run and was scaled to full size")


if __name__ == "__main__":
    child() if "--child" in sys.argv else parent()
