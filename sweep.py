"""Drive spin_bench over library builds, timeouts, thread counts, sizes and gaps.

Settings:
  stock        unchanged build, compiled default (THREAD_TIMEOUT 28)
  patched      build with thread-timeout.patch, compiled default (22)
  stock+NN     unchanged build with OPENBLAS_THREAD_TIMEOUT=NN

Environment:
  OB_ROOT          builds from build.sh (default ./_build)
  OUT_DIR          where sweep.json goes (default ./out)
  SWEEP_THREADS    comma list (default 2,4,8)
  SWEEP_SETTINGS   comma list (default stock,patched,stock+24,stock+22,stock+20,stock+18)
  SWEEP_REPS       passes over the whole grid; tables show the median (default 1)
  SWEEP_TARGET_S   intended seconds per run (default 0.8)
  SWEEP_MAX_S      hard stop per run (default 4)
"""
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("OB_ROOT", os.path.join(HERE, "_build"))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "out"))
SETTINGS = os.environ.get("SWEEP_SETTINGS", "stock,patched,stock+24,stock+22,stock+20,stock+18").split(",")
THREADS = [int(t) for t in os.environ.get("SWEEP_THREADS", "2,4,8").split(",")]
REPS = int(os.environ.get("SWEEP_REPS", "1"))
TARGET_S = float(os.environ.get("SWEEP_TARGET_S", "0.8"))
MAX_S = os.environ.get("SWEEP_MAX_S", "4")
SIZES = {300: 300e-6, 1000: 8e-3}  # n -> rough seconds per call, used to size ITERS
GAPS = [(0, "busy"), (100, "busy"), (1000, "busy"), (3000, "busy"), (10000, "busy"), (20000, "sleep"), (200000, "sleep")]


def run(setting, threads, n, gap_us, mode):
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads))
    env.pop("OPENBLAS_THREAD_TIMEOUT", None)
    lib = "patched" if setting == "patched" else "stock"
    env["LD_LIBRARY_PATH"] = os.path.join(ROOT, f"ob-{lib}", "lib")
    if "+" in setting:
        env["OPENBLAS_THREAD_TIMEOUT"] = setting.split("+")[1]
    per = SIZES[n] + gap_us / 1e6
    iters = int(min(4000, max(10 if mode == "sleep" else 40, TARGET_S / per)))
    cmd = [os.path.join(ROOT, "spin_bench"), str(n), str(gap_us), mode, str(iters), MAX_S]
    return json.loads(subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout)


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    for rep in range(REPS):
        order = SETTINGS[rep % len(SETTINGS):] + SETTINGS[:rep % len(SETTINGS)]
        for threads in THREADS:
            for n in SIZES:
                for gap_us, mode in GAPS:
                    for setting in order:  # settings innermost, so drift hits all of them alike
                        r = run(setting, threads, n, gap_us, mode)
                        r.update(setting=setting, rep=rep)
                        results.append(r)
            print(f"rep {rep + 1}/{REPS} threads={threads} done", file=sys.stderr, flush=True)
    with open(os.path.join(OUT, "sweep.json"), "w") as f:
        json.dump(results, f)

    def agg(setting, threads, n, gap_us, mode, fn):
        vals = [fn(r) for r in results if (r["setting"], r["threads"], r["n"], r["gap_us"], r["mode"]) == (setting, threads, n, float(gap_us), mode)]
        return statistics.median(vals)

    print(f"spin_bench sweep, {os.cpu_count()} CPUs, median of {REPS} pass(es)")
    for fn, title, fmt in (
        (lambda r: r["med_us"], "median microseconds per dgemm call", "{:.0f}"),
        (lambda r: r["p99_us"], "p99 microseconds per dgemm call", "{:.0f}"),
        (lambda r: r["mean_us"], "mean microseconds per dgemm call", "{:.0f}"),
        (lambda r: r["cpu_s"] / r["wall_s"], "average cores busy (process CPU / wall)", "{:.2f}"),
    ):
        print(f"\n##### {title} #####")
        for threads in THREADS:
            for n in SIZES:
                print(f"\nthreads={threads} n={n}")
                print(f"  {'gap':<16}" + "".join(f"{s:>10}" for s in SETTINGS))
                for gap_us, mode in GAPS:
                    cells = "".join(f"{fmt.format(agg(s, threads, n, gap_us, mode, fn)):>10}" for s in SETTINGS)
                    print(f"  {str(gap_us) + ' us ' + mode:<16}" + cells)


if __name__ == "__main__":
    main()
