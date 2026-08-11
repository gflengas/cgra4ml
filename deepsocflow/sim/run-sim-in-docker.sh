#!/bin/bash
# Builds and runs the deepsocflow verilator testbench inside the pinned-Verilator
# container, mirroring hardware.py::simulate()'s verilator invocation exactly.
#
# Why the worktree is mounted at its own absolute path rather than /work:
# hw.export() writes sources.txt with absolute HOST paths
# (deepsocflow/rtl/... under the worktree root). Mounting at the identical path
# makes every one of them resolve inside the container too, so nothing has to be
# rewritten and hardware.py stays untouched.
#
# Usage: run-sim-in-docker.sh <run-dir-relative-to-worktree>   (default: run)
set -euo pipefail

# Worktree root = two directories up from this script (deepsocflow/sim/ -> repo root).
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${1:-run}"
IMAGE=deepsocflow-sim:v5.024

docker run --rm -v "$W:$W" -w "$W/$RUN_DIR" "$IMAGE" bash -c "
set -euo pipefail
rm -rf build && mkdir -p build && cd build

verilator --binary -j 0 -O3 --relative-includes \
  --top top_tb \
  -I../ -I$W/deepsocflow/rtl/ \
  -F ../sources.txt --Mdir ./ \
  -CFLAGS -DSIM -CFLAGS -DTB_MODULE=top_tb -CFLAGS -DFB_MODULE=fb_axi_vip \
  -CFLAGS -I../ -CFLAGS -I$W/deepsocflow/firebridge/ -CFLAGS -g \
  $W/deepsocflow/c/sim.c $W/deepsocflow/firebridge/fb_top_verilator_wrap.cpp \
  --Wno-INITIALDLY --Wno-BLKANDNBLK --Wno-UNOPTFLAT

echo '=== BUILD OK, SIMULATING ==='
./Vtop_tb
"
