# Running RTL simulation on macOS

This directory holds the tooling that lets `Hardware.simulate()` (both the
legacy `deepsocflow/py/hardware.py` and the brevitas backend's
`deepsocflow/py/brevitas/hardware.py`) actually run a Verilator simulation on
a macOS host, where the toolchain this project depends on does not build
natively. Read this before touching anything under `deepsocflow/sim/` or
debugging a simulation that won't run.

## Why this exists: two independent Verilator problems

Neither of macOS's obvious Verilator options works for this design:

1. **`brew install verilator` (currently 5.050) segfaults.**
   `deepsocflow/firebridge/fb_top_verilator_wrap.cpp` drives the simulation
   with a loop that calls `top->eval()`; a DPI task (`run()`, invoked from
   *inside* that `eval()`) itself calls back into `at_posedge_clk()` ->
   `step_time_veri()` -> `top->eval()` again, i.e. `eval()` re-enters itself.
   Verilator 5's scheduler does not support that re-entrancy: the initial
   block that calls `run()` restarts from the top on every nested `eval()`
   call, and the process dies with SIGSEGV (exit 139) after printing the
   startup banner over a thousand times.

2. **The project's own pin (`Dockerfile:22`, `VERILATOR_VERSION=v5.024`) will
   not build against Apple clang's libc++.** Tested against Apple clang 21
   (`clang --version` on this machine: `Apple clang version 21.0.0
   (clang-2100.1.1.101)`, macOS 27.0/Darwin 27.0.0). `verilated_timing.h`'s
   coroutine runtime includes `<experimental/coroutine>` unconditionally,
   which Apple clang 21 removed; patching that include guard just uncovers
   ~14 further errors inside 5.024's own `VlCoroutineHandle`/`std::multimap`
   machinery that don't compile against Apple clang 21's libc++ either.
   Verilator itself builds fine in isolation on macOS - it is compiling
   *this design's* re-entrant/DPI-heavy testbench against 5.024's headers
   that fails.

The README's "Verilator 5.014+" is misleading on macOS: the real contract is
whatever `Dockerfile:22` pins (`v5.024`), and that pin is a Linux/libstdc++
build in practice on current Apple clang.

**Resolution:** run Verilator 5.024 inside a Linux container. On
Linux/libstdc++ its coroutine runtime compiles without incident, and 5.024
(not 5.050) doesn't hit the re-entrant-`eval()` segfault. `Dockerfile.sim`
builds a minimal image (Verilator + a C++ toolchain only - no TensorFlow,
PyTorch, or RISC-V toolchain) pinned to `v5.024`.

## Prerequisites

- Docker. On macOS without Docker Desktop, `colima start` provides a Docker
  daemon; run that first if `docker info` fails.
- Everything else (Python, PyTorch/brevitas, generating vectors and
  `config_fw.h`/`config_hw.svh`) runs on the macOS host exactly as normal.
  Only the Verilator build-and-run step needs Linux - the pipeline splits
  cleanly into (1) Python generates vectors/config, (2) Verilator builds and
  runs the testbench, (3) Python diffs sim output against expected, and only
  step 2 is routed into the container.

## Running a simulation

From the worktree root:

```
python deepsocflow/sim/docker_sim.py <path/to/script.py>
```

For example:

```
python deepsocflow/sim/docker_sim.py deepsocflow/py/brevitas/main.py
python deepsocflow/sim/docker_sim.py run/xor_qkeras.py
python deepsocflow/sim/docker_sim.py run/example.py
```

`docker_sim.py` monkeypatches `Hardware.simulate` (on **both** the legacy
`deepsocflow.py.hardware.Hardware` and the brevitas backend's
`deepsocflow.py.brevitas.hardware.Hardware` - they are separate,
non-inheriting classes, kept that way so the brevitas backend doesn't pull in
the legacy TensorFlow/qkeras import chain just by importing `Hardware`) to
shell out to `run-sim-in-docker.sh` instead of invoking a local
`verilator`/`xsim` binary, then runs the target script as `__main__` with its
own directory as cwd (matching how these scripts expect to be invoked
directly).

`run-sim-in-docker.sh` builds the `deepsocflow-sim:v5.024` image if it isn't
present locally (a fresh checkout has no image and no registry to pull one
from - first run takes a few minutes; later runs reuse the cached image),
then runs the same Verilator build+simulate invocation `hardware.py::simulate()`
would run locally, inside that container. The verilator argv in the script is
a hand-kept mirror of `hardware.py`'s invocation, not generated from it - if
either changes, update the other to match (not unified into one place; see
the script's own comment for why).

A passing brevitas run ends with:

```
Bundle 0, Error: 0. Passed
Bundle 1, Error: 0. Passed
Bundle 2, Error: ... Passed
brevitas XOR: RTL simulation PASSED
```

## Traps to know about before debugging a "failure"

- **`Hardware(axi_width=...)` defaults to `64`, which silently corrupts
  output.** Building `Hardware(...)` with the default `axi_width` makes the
  simulation run to completion and report no error, but bundle 0's raw
  engine-layout output has values duplicated at some offsets and dropped at
  others - a data-alignment corruption in the legacy AXI DMA/burst-splitting
  RTL, not a rounding difference. This was root-caused by A/B comparison
  against `run/xor_qkeras.py` (which passes `axi_width=128` explicitly) and
  was not chased further into the RTL - out of scope for the brevitas port.
  **Always pass `axi_width=128`** (or whatever a known-passing reference
  script uses) when building a new model's `Hardware(...)`; don't rely on the
  default.

- **`site-packages/deepsocflow.pth` may point at a different checkout.** If
  you have more than one clone/worktree of this repo, that `.pth` file can
  point at a different one. Running a script as `cd run && python ...`
  resolves `import deepsocflow...` against whatever's on `sys.path`, and the
  `.pth` entry can silently win over your intended worktree, so you'd be
  running someone else's checkout's code without any error. `docker_sim.py`
  avoids this by inserting the worktree root (derived from its own file
  location, not hardcoded) at `sys.path[0]` before importing anything
  `deepsocflow`-shaped - always launch simulations through `docker_sim.py`
  rather than `cd`-ing into `run/` and invoking a script directly, or if you
  must run something standalone, confirm `sys.path[0]` wins by running from
  the worktree root.

- **Why the worktree is mounted at its own absolute path, not `/work`.**
  `hw.export()` writes `sources.txt` with absolute host paths (`.../rtl/...`
  under the worktree root). Mounting the worktree at the identical absolute
  path inside the container makes every one of those paths resolve unchanged,
  so nothing needs to be rewritten and `hardware.py` stays untouched. If you
  ever change the mount point in `run-sim-in-docker.sh`, you also need to
  rewrite or regenerate `sources.txt`.

## Scope of what currently passes

The brevitas adapter (`deepsocflow/py/brevitas/adapter.py`) that feeds this
simulation path supports dense (`XDense`) bundles only - no conv, pooling, or
residual connections. Those paths are exercised in this repo only by the
pre-existing qkeras `run/example.py`, not by anything brevitas-driven. A
passing brevitas RTL run validates the dense/activation/softmax path on an
XOR-sized model; it does not validate conv/pooling/residual through the
brevitas backend.

## If the machine has Vivado, skip all of the above

The Docker/Verilator setup in this directory exists because the *macOS* host it
was written on could run neither the pinned Verilator 5.024 nor a working newer
one. On a Linux box with Vivado installed there is a much shorter path: both
`Hardware` classes already have an `if SIM == 'xsim'` branch, so Vivado's own
simulator drives the same testbench directly.

```bash
export PATH=/path/to/Xilinx/Vivado/2023.2/bin:$PATH
```

```python
verify_inference(None, hw, SIM='xsim', SIM_PATH='/path/to/Xilinx/Vivado/2023.2/bin/')
```

Measured on `geonosis` (Vivado 2023.2): a full compile + run of the XOR model
takes about **6 seconds**, against several minutes for the container route. It
works for the legacy qkeras scripts too - `run/xor_qkeras.py` and `run/example.py`
pass unchanged this way, by monkeypatching `deepsocflow.py.hardware.Hardware.simulate`
to force `SIM='xsim'` the same way `docker_sim.py` forces the container.

## `import torch` failing with `GLIBCXX_3.4.31 not found`

Not a code problem: the system `libstdc++` is older than the conda `torch` build
expects. Point the loader at conda's own copy first:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Without this, roughly ten tests fail on import in a way that reads like a real
regression but is purely environmental.

## Before running the PYNQ driver harness, restore the committed model

`run/work_pynq/pynq_deploy/test_pynq_driver_xor.py` asserts that a freshly
regenerated `config.json`/`wbx.bin` matches the deployed copies byte for byte.
Running `pytest` first breaks that: the suite retrains `xor.py` as a side effect,
and the retrain is only deterministic *within* one environment - a different torch
version produces slightly different weights. Restore the committed artifacts first:

```bash
git checkout -- deepsocflow/py/brevitas/model/
```
