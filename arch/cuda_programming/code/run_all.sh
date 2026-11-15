#!/usr/bin/env bash
# 一键编译并运行全部示例，结果保存到 results/
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${ARCH:-sm_90}"
mkdir -p bin results
for f in *.cu; do
  name="${f%.cu}"
  echo "===== build $name ($ARCH) ====="
  nvcc -O3 -arch="$ARCH" -lineinfo "$f" -o "bin/$name"
  echo "===== run $name ====="
  ./bin/"$name" | tee "results/$name.txt"
  echo
done
echo "all done. results in ./results/"
