# Results

All numbers compare the default idle timeout (`THREAD_TIMEOUT` 28) with shorter ones. "2^22" is the value in `thread-timeout.patch`. The patched build and the unchanged build with `OPENBLAS_THREAD_TIMEOUT=22` gave the same numbers in every sweep row, so the Python parts vary the environment variable.

## Setup

- GitHub-hosted runners, workflow run [43355055147](https://github.com/ptweezy/openblas-thread-timeout-bench/actions/runs/43355055147), two VMs per runner label for each part:
  - `ubuntu-24.04`: AMD EPYC 7763 and EPYC 9V74, 4 vCPUs (2 cores, 2 threads each).
  - `ubuntu-24.04-arm`: Neoverse-N2, 4 vCPUs.
- C harness: OpenBLAS at 5809b12, built on the runner with the autodetected target.
- Python parts: NumPy 2.5.4 (bundles OpenBLAS 0.3.34), SciPy 1.18.1 (bundles 0.3.31.dev), scikit-learn 1.9.1, all from PyPI, pthreads threading layer.
- Local machine: AMD Ryzen 7 9800X3D (8 cores, 16 threads), Docker on WSL2. It was shared with other workloads during the first runs, so pools were kept at 8 threads or fewer and every timeout was run 5 to 8 times, interleaved. Pools of 12 and 16 threads were measured later, when it was quieter (see "Larger pools on the local machine").

Raw output: `results/runners/aggregate.txt` (from `aggregate.py`) and `results/local/`.

## 25 NumPy / SciPy / scikit-learn workloads

Ratio to the default, median over paired runs (6 pairs over 2 VMs on the runners, 8 pairs locally).

Pool as large as the CPU count (4 threads on 4 vCPUs):

| | x86-64 2^24 | 2^22 | 2^20 | 2^18 | arm64 2^24 | 2^22 | 2^20 | 2^18 |
|---|---|---|---|---|---|---|---|---|
| Wall time, geometric mean | 0.87x | 0.84x | 0.82x | 0.81x | 0.91x | 0.88x | 0.87x | 0.87x |
| Workloads more than 10% slower | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Workloads more than 10% faster | 4 | 4 | 4 | 5 | 4 | 4 | 4 | 4 |
| CPU time, geometric mean | 0.60x | 0.50x | 0.43x | 0.38x | 0.64x | 0.54x | 0.48x | 0.43x |

Pool half as large as the CPU count:

| | x86-64 2^22 | 2^20 | arm64 2^22 | 2^20 | local (8 of 16) 2^24 | 2^22 | 2^21 | 2^20 |
|---|---|---|---|---|---|---|---|---|
| Wall time, geometric mean | 0.98x | 0.99x | 0.99x | 0.99x | 0.94x | 0.93x | 0.93x | 0.94x |
| Workloads more than 10% slower | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| CPU time, geometric mean | 0.60x | 0.55x | 0.68x | 0.64x | 0.51x | 0.40x | 0.36x | 0.32x |

The wall-time gain comes from four workloads that use more than one thread pool in the same process (scikit-learn's OpenMP loops, and SciPy's copy of OpenBLAS next to NumPy's): `LogisticRegression` fit (0.32x on x86-64, 0.46x on arm64 at 2^22), randomized PCA (0.31x, 0.26x), KMeans (0.49x, 0.57x) and Ridge (0.48x, 0.63x). While one pool works, the other pool's idle threads are still spinning on the same CPUs. The other 21 workloads are within 7% of the default at 2^22.

The one regression: locally, 3,000 back-to-back `MLPClassifier.predict_proba` calls take 1.71x as long at 2^20, which is 0.22 ms on that machine's 4.7 GHz timer and shorter than the interpreter work between two calls. It is unchanged at 2^21 and 2^22. An earlier local run at 2^18 was 1.24x slower by geometric mean and up to 2.7x on single workloads.

## Small-call loops, pool as large as the CPU count

Calls per second for a fixed time budget (`rate.py`: `solve` 100 and 400, matvec 1500, `tanh(W @ h)`, an MLP forward pass, matmul 1000). At 2^22 the geometric mean is 1.01x on both runner architectures, and no loop moves by more than 10% at any timeout. Locally with 8 threads the loops are 1.01x to 1.13x at 2^22.

## Several worker processes

Each process has a default-sized pool and handles 200 requests of one 256x256 matmul plus 5 ms of single-threaded work (`multiproc.py`). Requests per second, default -> 2^22:

| Processes | x86-64 VM 1 | x86-64 VM 2 | arm64 VM 1 | arm64 VM 2 | local (16-thread pools) |
|---|---|---|---|---|---|
| 1 | 177 -> 175 | 174 -> 172 | 178 -> 179 | 179 -> 178 | 31 -> 47 |
| 2 | 146 -> 236 | 141 -> 238 | 133 -> 278 | 135 -> 283 | 49 -> 269 |
| 4 | 115 -> 200 | 121 -> 198 | 95 -> 176 | 92 -> 163 | 50 -> 133 |

CPU time for the single process drops from 4.5 s to 2.5 s (x86-64) and 2.0 s (arm64).

## One process in a CPU-limited container

Same request loop under `docker run --cpus=N` (`cgroup_limit.py`). OpenBLAS sizes its pool from the visible CPUs, so the idle spin is charged to the quota. Requests per second, default -> 2^22:

| Limit | x86-64 VM 1 | x86-64 VM 2 | arm64 VM 1 | arm64 VM 2 |
|---|---|---|---|---|
| 1 of 4 CPUs | 50 -> 80 | 46 -> 77 | 50 -> 101 | 50 -> 105 |
| 2 of 4 CPUs | 92 -> 161 | 91 -> 155 | 91 -> 179 | 91 -> 179 |
| 2 of 4 CPUs, p99 latency | 56 -> 15 ms | 58 -> 19 ms | 56 -> 5.8 ms | 56 -> 5.7 ms |

Locally, with 16 CPUs visible: 10 -> 33 requests per second at `--cpus=2` and 27 -> 159 at `--cpus=4`.

## Latency of one call after an idle gap

`dgemm` 300x300, median microseconds, unchanged build -> patched build (`sweep.py`):

| Gap before the call | x86-64, 4 threads | arm64, 4 threads | local, 8 threads |
|---|---|---|---|
| none | 521 -> 513 | 583 -> 583 | 119 -> 117 |
| 1 ms of single-threaded work | 522 -> 523 | 581 -> 579 | 116 -> 222 |
| 3 ms | 528 -> 535 | 581 -> 597 | 117 -> 231 |
| 10 ms | 538 -> 557 | 587 -> 605 | 118 -> 249 |
| 20 ms asleep | 558 -> 579 | 589 -> 708 | 118 -> 260 |
| 200 ms asleep | 670 -> 867 | 733 -> 733 | 279 -> 284 |

Average cores busy over the same runs, with 10 ms gaps: 3.99 -> 1.61 (x86-64), 3.99 -> 1.46 (arm64), 7.97 -> 1.71 (local). With 20 ms sleeps: 3.02 -> 0.35, 3.02 -> 0.27, 6.97 -> 0.37.

A call that arrives after the workers have gone to sleep pays for waking them: about 20 microseconds in total on the 4-vCPU runners, and about 15 microseconds per thread woken on the local machine (about 55 with 4 threads, 110 to 130 with 8). `dgemm` 1000x1000 (4 ms per call) is within 3% in every row.

## Larger pools on the local machine

A later run with the machine quiet except for one unrelated container that held one core throughout (other host processes: about half a core). A 12-thread pool fits beside that; a 16-thread pool does not. Raw output in `results/local-quiet/`.

12 threads, 5 paired repetitions:

| | 2^24 | 2^22 | 2^21 | 2^20 |
|---|---|---|---|---|
| 25 workloads, wall time, geometric mean | 0.88x | 0.84x | 0.84x | 0.86x |
| Workloads more than 10% slower | 0 | 0 | 1 | 4 |
| CPU time, geometric mean | 0.46x | 0.34x | 0.31x | 0.27x |
| Small-call loops, calls per second | 0.98x to 1.04x | 0.99x to 1.03x | 0.99x to 1.06x | 0.99x to 1.08x |

At 2^22 the largest slowdowns are `expm` 400 (1.06x) and `inv` 600 (1.05x). At 2^21 `MLPClassifier` fit is 1.30x, and at 2^20 back-to-back `predict_proba` is 2.15x. Total CPU for the suite goes from 154.9 s to 69.2 s at 2^22.

`dgemm` with 12 threads, median microseconds, unchanged -> patched: 300x300 is 80 -> 80 back-to-back, 82 -> 259 after 1 ms and 84 -> 317 after 10 ms; 1000x1000 is 2472 -> 2836 after 10 ms. Across pool sizes the wake costs about 18 microseconds per thread: about 55 with 4 threads, 105 to 130 with 8, 180 to 240 with 12.

16 threads with one of the 16 CPUs busy elsewhere:

- With the default, everything is far slower than with 12 threads: six LU solves of size 3000 take 14.4 s instead of 0.39 s, and the `solve` 100x100 loop runs 24 calls per second instead of 12,245.
- 2^22 is faster than the default on every loop: `solve` 100 2.11x, `solve` 400 2.05x, matvec 5.35x, `tanh(W @ h)` 5.11x, MLP forward 1.72x, matmul 1000 1.47x.
- The 25 workloads (3 paired repetitions, each capped at 6 s and scaled) take 0.37x the wall time at 2^22 by geometric mean. 22 are more than 10% faster and one is slower (`cov` 5000x400, 1.11x of 0.03 s).
- Back-to-back `dgemm` 300x300: median 4003 -> 980 microseconds.

An earlier run at 12 and 16 threads, made while other work was using 4 to 5 cores, had shown slowdowns at 2^22 (3.3x on the `solve` 100x100 loop at 12 threads). They did not reproduce here.

## What is not covered

- Every machine here is a virtual machine (Azure for the runners, WSL2 locally). There are no bare-metal numbers.
- No machine has more than 16 CPUs. The caller wakes sleeping workers one at a time, so the cost of a wake grows with the pool size, and the margin shrinks with it: 2^21 was clean with 8 threads and made one workload 1.3x slower with 12.
- A 16-thread pool on a 16-CPU machine with nothing else running. One core was busy during every 16-thread run.
