# openblas-thread-timeout-bench

Benchmarks for the idle-thread timeout of [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)'s pthreads thread server.

After a multithreaded BLAS call, each worker thread in `driver/others/blas_server.c` busy-waits for `1 << THREAD_TIMEOUT` timer cycles before it sleeps on a condition variable. The default is 28, about 90 ms on a 3 GHz machine. `thread-timeout.patch` changes it to 22, about 1.4 ms. The same value can be set at run time with `OPENBLAS_THREAD_TIMEOUT`, which is how the Python benchmarks here vary it, because the NumPy and SciPy wheels on PyPI bundle a pthreads OpenBLAS with the default.

Related upstream issues: [#5171](https://github.com/OpenMathLib/OpenBLAS/issues/5171), [#1560](https://github.com/OpenMathLib/OpenBLAS/issues/1560), [#1614](https://github.com/OpenMathLib/OpenBLAS/issues/1614).

## What is measured

- `spin_bench.c`, driven by `sweep.py`: the latency of a threaded `dgemm` after an idle gap, and the process CPU time, for the unchanged build, the patched build, and the unchanged build with several `OPENBLAS_THREAD_TIMEOUT` values. The gap is either single-threaded work in the calling thread or a sleep.
- `workloads.py`: 25 NumPy, SciPy and scikit-learn workloads (LAPACK factorizations, loops of small calls with interpreter work between them, model fits, a request loop with pauses). Wall time and CPU time per timeout, each compared with the default of the same repetition.
- `rate.py`: calls per second of small-call loops under a fixed time budget, for a pool as large as the CPU count.
- `multiproc.py`: several worker processes, each with a default-sized pool, each handling requests of one matmul plus 5 ms of single-threaded work.
- `cgroup_limit.py`: the same request loop in a container whose CPU quota is smaller than the number of visible CPUs.

## Running

On GitHub-hosted runners (x86-64 and arm64 by default, two VMs each):

```
gh workflow run bench.yml
```

Locally on Linux:

```
bash build.sh                 # builds both libraries and spin_bench into ./_build
python sweep.py
pip install numpy scipy scikit-learn threadpoolctl
python workloads.py 8 5       # 8 BLAS threads, 5 repetitions
python rate.py 8
python multiproc.py
```

Results go to `./out` (or `OUT_DIR`). `aggregate.py` combines the artifacts of a workflow run into per-runner tables.

See [RESULTS.md](RESULTS.md) for the numbers so far.
