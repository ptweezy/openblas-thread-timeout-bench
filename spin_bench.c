/*
 * Per-call latency of a threaded dgemm after an idle gap, plus process CPU time.
 *
 *   spin_bench N GAP_US MODE ITERS [MAX_SECONDS]
 *     N            matrix dimension (N x N dgemm)
 *     GAP_US       microseconds between calls
 *     MODE         "busy"  = the calling thread does single-threaded work during the gap
 *                  "sleep" = the calling thread sleeps during the gap (pure idle)
 *     ITERS        number of calls
 *     MAX_SECONDS  stop early once this much wall time has passed (0 = no limit)
 *
 * Thread count comes from OPENBLAS_NUM_THREADS, the idle timeout from the
 * library default or OPENBLAS_THREAD_TIMEOUT. Prints one JSON object.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

extern int openblas_get_num_threads(void);

static double now(clockid_t c) {
  struct timespec ts;
  clock_gettime(c, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s N GAP_US busy|sleep ITERS [MAX_SECONDS]\n", argv[0]);
    return 2;
  }
  int n = atoi(argv[1]);
  double gap = atof(argv[2]) * 1e-6;
  int sleep_mode = strcmp(argv[3], "sleep") == 0;
  int iters = atoi(argv[4]);
  double max_s = argc > 5 ? atof(argv[5]) : 0.0;

  size_t nn = (size_t)n * (size_t)n;
  double *a = malloc(nn * sizeof(double));
  double *b = malloc(nn * sizeof(double));
  double *c = malloc(nn * sizeof(double));
  double *lat = malloc((size_t)iters * sizeof(double));
  if (!a || !b || !c || !lat) return 1;

  unsigned long long s = 88172645463325252ULL;
  for (size_t i = 0; i < nn; i++) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    a[i] = (double)(s >> 11) / 9007199254740992.0 - 0.5;
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    b[i] = (double)(s >> 11) / 9007199254740992.0 - 0.5;
  }

  for (int i = 0; i < 5; i++)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);

  /* Let the workers reach their steady idle state before measuring. */
  struct timespec settle = {0, 400 * 1000 * 1000};
  nanosleep(&settle, NULL);

  double cpu0 = now(CLOCK_PROCESS_CPUTIME_ID), wall0 = now(CLOCK_MONOTONIC);
  double sum = 0.0;
  int done = 0;
  for (int i = 0; i < iters; i++) {
    double t0 = now(CLOCK_MONOTONIC);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
    double t1 = now(CLOCK_MONOTONIC);
    lat[done++] = t1 - t0;
    sum += t1 - t0;
    if (max_s > 0 && t1 - wall0 > max_s) break;
    if (gap > 0) {
      if (sleep_mode) {
        struct timespec ts = {(time_t)gap, (long)((gap - (double)(time_t)gap) * 1e9)};
        nanosleep(&ts, NULL);
      } else {
        double end = now(CLOCK_MONOTONIC) + gap;
        while (now(CLOCK_MONOTONIC) < end) { }
      }
    }
  }
  double cpu = now(CLOCK_PROCESS_CPUTIME_ID) - cpu0, wall = now(CLOCK_MONOTONIC) - wall0;

  qsort(lat, (size_t)done, sizeof(double), cmp);
  printf("{\"n\":%d,\"threads\":%d,\"gap_us\":%.0f,\"mode\":\"%s\",\"iters\":%d,"
         "\"med_us\":%.2f,\"p90_us\":%.2f,\"p99_us\":%.2f,\"mean_us\":%.2f,"
         "\"cpu_s\":%.4f,\"wall_s\":%.4f,\"check\":%.6g}\n",
         n, openblas_get_num_threads(), gap * 1e6, sleep_mode ? "sleep" : "busy", done,
         lat[done / 2] * 1e6, lat[(int)(done * 0.90)] * 1e6, lat[(int)(done * 0.99)] * 1e6,
         sum / done * 1e6, cpu, wall, c[nn / 2]);
  return 0;
}
