#!/bin/bash
# Build OpenBLAS twice from the same pinned commit, unchanged (THREAD_TIMEOUT 28)
# and with thread-timeout.patch (22), then compile spin_bench against it.
#
#   OB_ROOT        where builds are installed (default: ./_build)
#   OPENBLAS_REF   upstream commit to build
#   OB_TARGET      optional TARGET= for the OpenBLAS build (default: autodetect)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OB_ROOT="${OB_ROOT:-$HERE/_build}"
OPENBLAS_REF="${OPENBLAS_REF:-5809b12a4259c26185f59673f636a58414e0f413}"
FLAGS="USE_THREAD=1 USE_OPENMP=0 NO_LAPACK=1 NOFORTRAN=1 NO_AFFINITY=1 NUM_THREADS=64"
[ -n "${OB_TARGET:-}" ] && FLAGS="TARGET=$OB_TARGET $FLAGS"

mkdir -p "$OB_ROOT/src"
cd "$OB_ROOT/src"
if [ ! -d OpenBLAS/.git ]; then
  git init -q OpenBLAS
  git -C OpenBLAS remote add origin https://github.com/OpenMathLib/OpenBLAS.git
fi
cd OpenBLAS
git fetch -q --depth 1 origin "$OPENBLAS_REF"
git checkout -q -f FETCH_HEAD
echo "OpenBLAS commit: $(git log --oneline -1)"

make $FLAGS libs netlib shared > "$OB_ROOT/build-stock.log" 2>&1 || { tail -40 "$OB_ROOT/build-stock.log"; exit 1; }
make $FLAGS PREFIX="$OB_ROOT/ob-stock" install > "$OB_ROOT/install-stock.log" 2>&1
grep -m1 -E "OpenBLAS build complete|Library Name" "$OB_ROOT/build-stock.log" || true

git apply "$HERE/thread-timeout.patch"
git diff --stat | tail -1
make $FLAGS libs netlib shared > "$OB_ROOT/build-patched.log" 2>&1 || { tail -40 "$OB_ROOT/build-patched.log"; exit 1; }
make $FLAGS PREFIX="$OB_ROOT/ob-patched" install > "$OB_ROOT/install-patched.log" 2>&1

gcc -O2 -Wall -o "$OB_ROOT/spin_bench" "$HERE/spin_bench.c" -I"$OB_ROOT/ob-stock/include" -L"$OB_ROOT/ob-stock/lib" -lopenblas -lm
if cmp -s "$OB_ROOT/ob-stock/lib/libopenblas.so" "$OB_ROOT/ob-patched/lib/libopenblas.so"; then
  echo "ERROR: stock and patched libraries are identical"; exit 1
fi
echo "built: $OB_ROOT/ob-stock, $OB_ROOT/ob-patched, $OB_ROOT/spin_bench"
