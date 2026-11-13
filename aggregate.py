"""Combine the artifacts of one bench workflow run into per-runner tables.

  gh run download RUN_ID --dir artifacts
  python aggregate.py artifacts

Each artifact directory is named <part>-<runner>-<rep>. Results of the VMs of one
runner label are pooled: paired ratios (timeout / default of the same repetition on
the same VM) from every VM go into one median.
"""
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict


def load(root):
    by_runner = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        m = re.match(r"(sweep|workloads|service)-(.+)-(\d+)$", os.path.basename(d))
        if m:
            by_runner[m.group(2)][m.group(1)].append(d)
    return by_runner


def machine(d):
    try:
        text = open(os.path.join(d, "machine.txt")).read()
    except OSError:
        return "?"
    model = re.search(r"Model name:\s*(.+)", text)
    cpus = re.search(r"^CPU\(s\):\s*(\d+)", text, re.M)
    return f"{model.group(1).strip() if model else '?'} ({cpus.group(1) if cpus else '?'} CPUs)"


def paired_table(title, files, key, scale, lower_is_better=True):
    """files: JSON files with runs[timeout] = [ {name: {...}} per repetition ]."""
    pools = defaultdict(lambda: defaultdict(list))  # name -> timeout -> ratios
    base = defaultdict(list)
    timeouts, names, capped = [], [], set()
    for f in files:
        runs = json.load(open(f))["runs"]
        timeouts = [t for t in runs if t != "default"]
        names = list(runs["default"][0])
        for t in timeouts:
            for a, b in zip(runs[t], runs["default"]):
                for n in names:
                    pools[n][t].append(scale(a[n], key) / scale(b[n], key))
                    if a[n].get("frac", 1) < 1 or b[n].get("frac", 1) < 1:
                        capped.add(n)
        for b in runs["default"]:
            for n in names:
                base[n].append(scale(b[n], key))
    if not names:
        return
    npairs = len(next(iter(pools[names[0]].values())))
    print(f"\n{title} ({len(files)} VM(s), {npairs} paired runs per timeout)")
    print(f"{'':<42}{'default':>10}" + "".join(f"{t:>8}" for t in timeouts))
    logs = defaultdict(list)
    for n in names:
        row = f"{(n + (' *' if n in capped else '')):<42}{statistics.median(base[n]):>10.3f}"
        for t in timeouts:
            med = statistics.median(pools[n][t])
            logs[t].append(math.log(med))
            row += f"{med:>7.2f}x"
        print(row)
    print(f"{'geometric mean':<42}{'':>10}" + "".join(f"{math.exp(sum(logs[t]) / len(logs[t])):>7.2f}x" for t in timeouts))
    worse = (lambda x: x > math.log(1.10)) if lower_is_better else (lambda x: x < math.log(0.90))
    better = (lambda x: x < math.log(0.90)) if lower_is_better else (lambda x: x > math.log(1.10))
    print(f"{'worse by more than 10%':<42}{'':>10}" + "".join(f"{sum(1 for x in logs[t] if worse(x)):>8}" for t in timeouts))
    print(f"{'better by more than 10%':<42}{'':>10}" + "".join(f"{sum(1 for x in logs[t] if better(x)):>8}" for t in timeouts))


def sweep_tables(files):
    rows = []
    for f in files:
        rows += json.load(open(f))
    if not rows:
        return
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    for r in rows:  # OpenBLAS caps the pool at the CPU count; group by what actually ran
        r["threads"] = min(r["threads"], r.get("requested", r["threads"]))
    threads = sorted({r["threads"] for r in rows})
    sizes = sorted({r["n"] for r in rows})
    gaps = list(dict.fromkeys((r["gap_us"], r["mode"]) for r in rows))
    for fn, title, fmt in (
        (lambda r: r["med_us"], "median microseconds per dgemm call", "{:.0f}"),
        (lambda r: r["mean_us"], "mean microseconds per dgemm call", "{:.0f}"),
        (lambda r: r["cpu_s"] / r["wall_s"], "average cores busy", "{:.2f}"),
    ):
        print(f"\nsweep: {title} (median over {len(files)} VM(s) and their passes)")
        for th in threads:
            for n in sizes:
                print(f"threads={th} n={n}")
                print(f"  {'gap':<16}" + "".join(f"{s:>10}" for s in settings))
                for gap, mode in gaps:
                    cells = ""
                    for s in settings:
                        vals = [fn(r) for r in rows if (r["setting"], r["threads"], r["n"], r["gap_us"], r["mode"]) == (s, th, n, gap, mode)]
                        cells += f"{fmt.format(statistics.median(vals)):>10}"
                    print(f"  {f'{gap:.0f} us {mode}':<16}" + cells)


def main():
    root = sys.argv[1]
    full = lambda run, key: run[key] / run.get("frac", 1)
    for runner, parts in load(root).items():
        print("\n" + "=" * 100)
        print(f"{runner}")
        for part in ("sweep", "workloads", "service"):
            for d in parts.get(part, []):
                print(f"  {os.path.basename(d)}: {machine(d)}")
        print("=" * 100)
        sweep_tables([f for d in parts.get("sweep", []) for f in glob.glob(os.path.join(d, "sweep.json"))])
        by_t = defaultdict(list)
        for d in parts.get("workloads", []):
            for f in glob.glob(os.path.join(d, "workloads-T*.json")):
                by_t[re.search(r"T(\d+)", os.path.basename(f)).group(1)].append(f)
        for t in sorted(by_t, key=int, reverse=True):
            paired_table(f"workloads, {t} threads: WALL time vs default", by_t[t], "wall", full)
            paired_table(f"workloads, {t} threads: CPU time vs default", by_t[t], "cpu", full)
        by_t = defaultdict(list)
        for d in parts.get("service", []):
            for f in glob.glob(os.path.join(d, "rate-T*.json")):
                by_t[re.search(r"T(\d+)", os.path.basename(f)).group(1)].append(f)
        for t in sorted(by_t, key=int):
            paired_table(f"small-call loops, {t} threads: calls per second vs default (higher is better)", by_t[t], "rate", lambda run, key: run[key], lower_is_better=False)
        for name in ("multiproc.txt", "cgroup.txt"):
            for d in parts.get("service", []):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    print(f"\n{name} from {os.path.basename(d)}")
                    print(open(p).read().rstrip())


if __name__ == "__main__":
    main()
