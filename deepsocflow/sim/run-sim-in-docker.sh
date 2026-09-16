#!/bin/bash
# Builds the sim image if missing, then builds and runs the deepsocflow
# verilator testbench inside it (the pinned-Verilator container). The
# verilator invocation below mirrors hardware.py::simulate()'s verilator
# invocation (deepsocflow/py/hardware.py:232-257 and its brevitas-backend
# counterpart, deepsocflow/py/brevitas/hardware.py) by hand - it is NOT
# generated from that code, so if either of those argv lists changes, update
# this one to match. Not refactored into one shared place - out of scope for
# this fix wave.
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

# Build the image if it doesn't exist yet. A fresh checkout has no image and
# no registry to pull one from - explicitly:
#   docker build -t deepsocflow-sim:v5.024 -f deepsocflow/sim/Dockerfile.sim deepsocflow/sim
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "=== Image '$IMAGE' not found locally, building it (this takes a few minutes) ==="
  docker build -t "$IMAGE" -f "$W/deepsocflow/sim/Dockerfile.sim" "$W/deepsocflow/sim"
fi

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
