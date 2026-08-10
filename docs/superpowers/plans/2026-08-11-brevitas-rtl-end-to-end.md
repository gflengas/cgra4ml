# Brevitas → RTL End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the brevitas backend drive a passing verilator RTL simulation of the XOR model.

**Architecture:** brevitas owns the numbers (quantization, integer arithmetic); the legacy qkeras backend owns the format (engine layout, `config_fw.h`, buffer allocation). A new adapter makes brevitas bundles expose legacy `XBundle`'s attribute surface and registers them into the legacy `BUNDLES` global, so the legacy export and verify paths run unchanged.

**Tech Stack:** Python 3.11, PyTorch + brevitas, TensorFlow + qkeras (legacy side), numpy, pytest, verilator.

**Spec:** `docs/superpowers/specs/2026-08-11-brevitas-rtl-end-to-end-design.md`

## Global Constraints

- **Work on a new branch off `brevitas-golden-ref`.** The user asked for this explicitly. Do not implement on `brevitas-golden-ref` itself.
- **Never `git push`.** Committing locally is expected (this plan commits per task); pushing is the user's job. This overrides any skill guidance to push.
- Hardware config for XOR, used identically in every phase: `processing_elements=(8, 24)`, `bits_input=8`, `bits_weights=8`, `bits_bias=16`, `bits_sum=32`. These are the values already in `deepsocflow/py/brevitas/main.py` and are a combination `hardware.py:62-63` accepts.
- Export **all 4 XOR rows** (`batch_size=4`), not the legacy `batch_size=1` convention.
- Do not modify `deepsocflow/py/brevitas/export.py::export_inference` (the Phase 1 flat-text export) or its 20 existing tests. New RTL work goes in new functions.
- Out of scope: all six CLAUDE.md Known Issues, conv/pool/residual/flatten bundles, dropping the TensorFlow dependency.
- The legacy dense→conv reshape puts **batch in the H slot**: `(batch, features)` → `(1, batch, 1, features)`. Getting this backwards silently produces wrong runtime params.

**Environment (established during Task 1 — these override the raw commands written in later tasks):**

- **Never run `cd run && python <script>.py`.** `site-packages/deepsocflow.pth` points at the MAIN repo (`/Users/charaphat/CERN/cgra4ml`), not this worktree, so running from `run/` silently executes the wrong copy of the code. Running from the worktree root is safe because cwd wins on `sys.path`.
- **The host cannot simulate.** Verilator 5.050 (brew) breaks firebridge's re-entrant `eval()` pattern (SIGSEGV); Verilator 5.024 (which the project's own `Dockerfile:22` pins) cannot build against Apple clang 21's libc++. The simulator runs in a container instead.
- **Run every script that simulates like this, from the worktree root:**

  ```bash
  python .superpowers/sdd/2026-08-11-brevitas-rtl-end-to-end/docker_sim.py run/<script>.py
  ```

  `docker_sim.py` inserts the worktree at `sys.path[0]`, monkeypatches `Hardware.simulate()` to build and run the testbench inside `deepsocflow-sim:v5.024`, chdirs to the script's own directory, and then runs it. Both problems above are handled; nothing in the repo needs changing.
- Pure-Python work (pytest, `python -m deepsocflow.py.brevitas.main` without simulation) runs normally from the worktree root.

---

### Task 1: Phase 0 — prove the toolchain works

No project code changes. This is a gate: if it fails, the problem is the environment, not this design, and you should stop and report rather than work around it.

**Files:** none (environment only)

**Interfaces:**
- Consumes: nothing
- Produces: a working `verilator` on `PATH`; confidence that `deepsocflow/rtl/` + `deepsocflow/c/sim.c` + `deepsocflow/firebridge/` simulate correctly as-is

- [ ] **Step 1: Install verilator**

```bash
brew install verilator
verilator --version
```

Record the version in your report — it is the single most likely cause if later phases behave strangely.

- [ ] **Step 2: Run the existing legacy example end-to-end**

`run/example.py` does `sys.path.append("../../")` and expects to run from `run/`. It downloads MNIST on first run.

```bash
cd run && python example.py
```

Expected: the script prints `SIMULATING...`, then per-bundle `Bundle N, Error: 0. Passed` lines, and exits 0.

- [ ] **Step 3: If `export_vivado_tcl` fails, disable it and rerun**

`hardware.py:285` asserts a board `.tcl` file exists. That step targets Vivado synthesis and is **not** needed for verilator simulation. If it raises, comment out the `hw.export_vivado_tcl(board='zcu104')` line in your local copy and rerun. Do not commit that change — it is a local workaround for Task 1 only.

- [ ] **Step 4: Report the outcome**

If the simulation passed, continue to Task 2. **If it failed for any reason other than the `export_vivado_tcl` assert, stop and report** — everything downstream assumes a working simulator, and debugging our adapter against a broken toolchain wastes the whole exercise.

Nothing to commit in this task.

---

### Task 1.5: Fix the `ic_right` regression blocking Task 1

**Added mid-execution.** Task 1 found that `run/example.py` fails inside
`export_inference` before reaching the simulator:
`InvalidArgumentError: filter depth must be strictly positive, got 0`.

Root cause, confirmed via `git log -L 179,195:deepsocflow/py/xbundle.py`: commit
`d3091e2` ("brevitas layers added") deleted the line `ic_right += CM_p` from
`XBundle.export`'s per-pass loop while adding comments to it. `ic_left` and
`ic_right` therefore stay `0`, so every pass slices `[0:0]`. That `CM_p` is now
computed and never used is corroborating evidence the deletion was accidental.

**Files:**
- Modify: `deepsocflow/py/xbundle.py:182-191`
- Test: `deepsocflow/test/py/test_xbundle_passes.py`

**Interfaces:**
- Consumes: `get_runtime_params` from `deepsocflow.py.dataflow`
- Produces: `_pass_channel_slices(r) -> list[tuple[int, int]]` in `xbundle.py` — the `(ic_left, ic_right)` bound pair for each of `r.CP` passes

- [ ] **Step 1: Write the failing test**

Restoring one line would fix the symptom, but the bug was invisible because the
slice arithmetic is inlined in a loop that needs a full built model to exercise.
Extract it into a pure helper so it is unit-testable, then test the helper.

```python
# deepsocflow/test/py/test_xbundle_passes.py
"""Guards the per-pass input-channel slicing in XBundle.export. Commit d3091e2
dropped `ic_right += CM_p` from that loop while adding comments, making every
pass slice [0:0]; nothing caught it because the arithmetic was inlined in a loop
that needs a fully built model to reach."""
from collections import namedtuple

import pytest


def _runtime(CP, CM_0, CM, CI):
    return namedtuple('R', ['CP', 'CM_0', 'CM', 'CI'])(CP=CP, CM_0=CM_0, CM=CM, CI=CI)


def test_single_pass_covers_all_channels():
    pytest.importorskip("tensorflow")
    from deepsocflow.py.xbundle import _pass_channel_slices

    assert _pass_channel_slices(_runtime(CP=1, CM_0=3, CM=72, CI=3)) == [(0, 3)]


def test_multi_pass_slices_are_contiguous_and_cover_all_channels():
    pytest.importorskip("tensorflow")
    from deepsocflow.py.xbundle import _pass_channel_slices

    # CI=200 split as CM_0=56 then two full passes of 72
    slices = _pass_channel_slices(_runtime(CP=3, CM_0=56, CM=72, CI=200))

    assert len(slices) == 3
    assert slices[0][0] == 0, "first pass must start at channel 0"
    assert slices[-1][1] == 200, "last pass must end at CI"
    for (_, prev_right), (next_left, _) in zip(slices, slices[1:]):
        assert prev_right == next_left, "slices must be contiguous, no gaps"


def test_no_slice_is_empty():
    """The actual d3091e2 regression: every slice was [0:0], which TF's conv2d
    rejects with 'filter depth must be strictly positive, got 0'."""
    pytest.importorskip("tensorflow")
    from deepsocflow.py.xbundle import _pass_channel_slices

    for left, right in _pass_channel_slices(_runtime(CP=3, CM_0=56, CM=72, CI=200)):
        assert right > left, f"empty channel slice [{left}:{right}]"


def test_first_pass_uses_cm_0_and_rest_use_cm():
    pytest.importorskip("tensorflow")
    from deepsocflow.py.xbundle import _pass_channel_slices

    slices = _pass_channel_slices(_runtime(CP=3, CM_0=56, CM=72, CI=200))
    widths = [right - left for left, right in slices]
    assert widths == [56, 72, 72]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest deepsocflow/test/py/test_xbundle_passes.py -v`
Expected: FAIL with `ImportError: cannot import name '_pass_channel_slices'`

- [ ] **Step 3: Add the helper and use it in the loop**

Add above `class XBundle` in `deepsocflow/py/xbundle.py`:

```python
def _pass_channel_slices(r):
    """(ic_left, ic_right) input-channel bounds for each of r.CP passes.

    Pass 0 handles r.CM_0 channels (the remainder), every later pass handles a
    full r.CM. Extracted from XBundle.export so the arithmetic is unit-testable:
    commit d3091e2 silently dropped the `ic_right += CM_p` increment here and
    nothing caught it."""
    slices = []
    ic_left = ic_right = 0
    for ip in range(r.CP):
        ic_right += r.CM_0 if ip == 0 else r.CM
        slices.append((ic_left, ic_right))
        ic_left = ic_right
    return slices
```

Then replace the loop body in `XBundle.export` (currently lines 182-191) with:

```python
        self.ye_exp_p = []  # ye_exp per pass (p)
        for ic_left, ic_right in _pass_channel_slices(r):
            wp = w_int[:,:, ic_left:ic_right, :]  # weight slice (w) for this pass (p)
            xp = x_int[:,:,:, ic_left:ic_right ]  # input slice (x) for this pass (p)
            yp = tf.keras.backend.conv2d(xp.astype(np.float32), wp.astype(np.float32), padding='same').numpy().astype(np.int32)  # conv-sum (y) for this pass (p)
            self.ye_exp_p += [reorder_y_q2e_conv(yp, hw, r)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest deepsocflow/test/py/test_xbundle_passes.py -v`
Expected: 4 passed

- [ ] **Step 5: Confirm the original failure is gone**

```bash
cd run && python example.py
```

Expected: gets past `export_inference` — no `filter depth must be strictly
positive` error. It may still fail later at `hw.export_vivado_tcl` (missing board
`.tcl`); that is Task 1's known workaround and not your concern. If it fails
anywhere else, report DONE_WITH_CONCERNS with the error.

- [ ] **Step 6: Commit**

```bash
git add deepsocflow/py/xbundle.py deepsocflow/test/py/test_xbundle_passes.py
git commit -m "fix: restore per-pass channel increment dropped in d3091e2"
```

---

### Task 2: Phase 1 — qkeras XOR baseline as reference

Build a qkeras model with the same *shape* as the brevitas XOR (`XDense` 2→3→3→2, relu/relu/identity+softmax) and run the full legacy flow through RTL simulation. Weights are whatever qkeras initialises — matching brevitas's trained weights is explicitly not a goal (see spec). This produces the reference file set that Task 8 diffs against, and the regression test that protects Task 3.

**Files:**
- Create: `run/xor_qkeras.py`

**Interfaces:**
- Consumes: verilator from Task 1
- Produces: reference files in `run/vectors_qkeras_xor/` and `run/config_fw.h`; a script re-runnable as a regression test after Task 3

- [ ] **Step 1: Write the baseline script**

Modelled on `run/example.py`. Note `use_bias=True` on every layer (brevitas XOR has biases everywhere) and `type=None` on the last activation (identity) with `softmax=True` on its bundle.

```python
# run/xor_qkeras.py
"""qkeras XOR-equivalent baseline: same architecture as the brevitas XOR
(deepsocflow/py/brevitas/xor.py), independent weights. Its purpose is to produce
a reference set of RTL-facing files for a model of this shape, and to act as the
regression test for refactors of deepsocflow/py/xmodel.py."""
import os
import sys
sys.path.append("../")

import numpy as np
from tensorflow import keras
from keras.layers import Input
from keras.models import Model, save_model
from qkeras.utils import load_qmodel

from deepsocflow import *

SIM = 'xsim' if os.name == 'nt' else 'verilator'

# Matches the brevitas side: 8-bit activations/weights, 16-bit bias.
sys_bits = SYS_BITS(x=8, k=8, b=16)


@keras.saving.register_keras_serializable()
class UserModel(XModel):
    def __init__(self, sys_bits, x_int_bits, *args, **kwargs):
        super().__init__(sys_bits, x_int_bits, *args, **kwargs)

        self.b1 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=3, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)))

        self.b2 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=3, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type='relu', slope=0)))

        self.b3 = XBundle(
            core=XDense(
                k_int_bits=0, b_int_bits=0, units=2, use_bias=True,
                act=XActivation(sys_bits=sys_bits, o_int_bits=0, type=None)),
            softmax=True)

    def call(self, x):
        x = self.input_quant_layer(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x


x_in = Input((2,), name="input")
user_model = UserModel(sys_bits=sys_bits, x_int_bits=0)
model = Model(inputs=[x_in], outputs=[user_model(x_in)])

save_model(model, "xor_qkeras.h5")
loaded_model = load_qmodel("xor_qkeras.h5")

hw = Hardware(
    processing_elements=(8, 24),
    frequency_mhz=250,
    bits_input=8,
    bits_weights=8,
    bits_sum=32,
    bits_bias=16,
    max_batch_size=64,
    max_channels_in=512,
    max_kernel_size=9,
    max_image_size=512,
    max_n_bundles=64,
    ram_weights_depth=512,
    ram_edges_depth=3584,
    axi_width=128,
    config_baseaddr="B0000000",
    target_cpu_int_bits=32,
    valid_prob=1,
    ready_prob=1,
    data_dir='vectors_qkeras_xor',
)

hw.export_json()
hw = Hardware.from_json('hardware.json')
hw.export()  # config_hw.svh, config_hw.tcl, sources.txt

# batch_size=4 matches the brevitas side (all four XOR rows). The legacy exporter
# feeds random input, not the XOR truth table - fine here, since this baseline
# exists for file structure and RTL-liveness, not for XOR correctness.
export_inference(loaded_model, hw, batch_size=4)
verify_inference(loaded_model, hw, SIM=SIM)
print("qkeras XOR baseline: RTL simulation PASSED")
```

- [ ] **Step 2: Run it**

```bash
cd run && python xor_qkeras.py
```

Expected: `Bundle 0..2, Error: 0. Passed` then `qkeras XOR baseline: RTL simulation PASSED`.

- [ ] **Step 3: Preserve the reference file set**

These files are the reference Task 8 diffs against. Copy them somewhere they will not be overwritten by later runs:

```bash
cd run && mkdir -p reference_qkeras_xor && cp -r vectors_qkeras_xor config_fw.h reference_qkeras_xor/
ls reference_qkeras_xor/vectors_qkeras_xor | head -20
```

- [ ] **Step 4: Commit**

```bash
git add run/xor_qkeras.py
git commit -m "test: add qkeras XOR baseline for RTL reference and regression"
```

Do not commit `reference_qkeras_xor/`, `vectors_qkeras_xor/`, `*.h5`, `hardware.json`, or `config_fw.h` — they are generated artifacts. If `git status` shows them as untracked noise, add them to `.gitignore` in this commit.

---

### Task 3: Split the keras preamble out of legacy `export_inference`

The legacy `export_inference` mixes a keras-specific preamble (lines 49-62: `model.layers[1]`, `tf.random.uniform`, `input_quant_layer`, `sys_bits` asserts) with a backend-agnostic bundle loop (line 82 onward: `call_int`/`export`, buffer allocation, `config_fw.h`). Split them so both backends share one code path.

**Files:**
- Modify: `deepsocflow/py/xmodel.py:42-333`

**Interfaces:**
- Consumes: nothing new
- Produces: `_export_bundles(hw, x)` — runs the bundle loop over an already-populated `BUNDLES`, where `x` is the input `XTensor` passed to bundle 0's `call_int` (brevitas passes `None`). `export_inference(model, hw, batch_size=1)` keeps its exact current signature and behaviour.

- [ ] **Step 1: Perform the split**

In `deepsocflow/py/xmodel.py`, change `export_inference` so everything from the `add_buffer_map = []` line (currently line 79) to the end of the function moves verbatim into a new module-level function, and `export_inference` ends by calling it. The only edit inside the moved body is nothing at all — it already refers only to `hw`, `BUNDLES`, and `x`.

```python
def export_inference(model, hw, batch_size=1):
    # ... existing preamble unchanged, through the line that builds `x`:
    #     x = XTensor(tensor=x_qtensor, bits=hw.X_BITS, int=user_model.x_int_bits)
    #     and the DATA_DIR cleaning block
    return _export_bundles(hw, x)


def _export_bundles(hw, x):
    """Bundle loop shared by both backends. Assumes BUNDLES is already populated
    and, for the brevitas backend, that each bundle's integer tensors are already
    computed (its call_int is a no-op). `x` is the input XTensor consumed by
    bundle 0's call_int; the brevitas backend passes None."""
    # ... body moved verbatim from export_inference, starting at `add_buffer_map = []`
```

Keep the `''' Clean the data directory'''` block and the `print("\n-----------STARTING EXPORT-----------\n")` in the **preamble** side (`export_inference`), because the brevitas driver will do its own directory cleaning.

- [ ] **Step 2: Verify the qkeras backend still passes RTL simulation**

This is the regression gate. It must pass before anything else proceeds.

```bash
cd run && python xor_qkeras.py
```

Expected: identical output to Task 2 — `Bundle 0..2, Error: 0. Passed`.

- [ ] **Step 3: Verify the larger legacy example also still passes**

```bash
cd run && python example.py
```

Expected: passes as in Task 1. This catches anything the XOR-shaped baseline is too simple to exercise (conv, pooling, residual add).

- [ ] **Step 4: Commit**

```bash
git add deepsocflow/py/xmodel.py
git commit -m "refactor: split keras preamble from bundle loop in export_inference"
```

---

### Task 4: Adapter — activation parameter mapping

First slice of the adapter: translate a brevitas activation name into the three fields legacy `XActivation` exposes. Mirrors `deepsocflow/py/xlayers.py:20-24`.

**Files:**
- Create: `deepsocflow/py/brevitas/adapter.py`
- Test: `deepsocflow/test/py/test_brevitas_adapter.py`

**Interfaces:**
- Consumes: nothing
- Produces: `act_params(activation: str, negative_slope: float = 0.0) -> tuple[int, int]` returning `(non_zero, plog_slope)`

- [ ] **Step 1: Write the failing test**

```python
# deepsocflow/test/py/test_brevitas_adapter.py
import numpy as np
import pytest

from deepsocflow.py.brevitas.adapter import act_params


def test_act_params_relu():
    # legacy: slope=0 -> non_zero = 1*(0 != 0) = 0, plog_slope = 0
    assert act_params('relu') == (0, 0)


def test_act_params_identity():
    # legacy: type=None forces slope=1 -> non_zero = 1, log2(1) = 0
    assert act_params('identity') == (1, 0)


def test_act_params_leaky_relu_power_of_two():
    assert act_params('leaky_relu', negative_slope=0.125) == (1, 3)
    assert act_params('leaky_relu', negative_slope=0.5) == (1, 1)


def test_act_params_rejects_non_power_of_two_slope():
    with pytest.raises(AssertionError, match="power of two"):
        act_params('leaky_relu', negative_slope=0.1)


def test_act_params_rejects_unsupported_activation():
    with pytest.raises(NotImplementedError, match="silu"):
        act_params('silu')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'deepsocflow.py.brevitas.adapter'`

- [ ] **Step 3: Write minimal implementation**

```python
# deepsocflow/py/brevitas/adapter.py
"""Adapts brevitas FixedPointModel bundles onto the legacy XBundle attribute
surface, so the legacy engine-layout export (deepsocflow/py/xmodel.py) and RTL
verification path run over brevitas-produced numbers unchanged.

brevitas owns the numbers; the legacy backend owns the file format. This module
is the only seam between them."""
import math


def act_params(activation, negative_slope=0.0):
    """(non_zero, plog_slope) as legacy XActivation computes them
    (deepsocflow/py/xlayers.py:20-24).

    non_zero is 0 only for plain relu (slope 0); identity is modelled by legacy
    as slope=1, which makes non_zero 1 and plog_slope 0. plog_slope is the
    right-shift amount applied to negative inputs, so it is only non-zero for
    leaky_relu."""
    if activation == 'relu':
        return 0, 0
    if activation == 'identity':
        return 1, 0
    if activation == 'leaky_relu':
        log_slope = math.log2(negative_slope)
        assert log_slope == int(log_slope) and log_slope <= 0, (
            f"negative_slope={negative_slope} must be a negative power of two "
            f"(0.5, 0.25, 0.125, ...) - quant_lrelu implements it as a shift")
        return 1, -int(log_slope)
    raise NotImplementedError(
        f"activation '{activation}' has no integer-exact hardware implementation "
        f"(see CLAUDE.md Known Issues); only relu/identity/leaky_relu are deployable")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_adapter.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add deepsocflow/py/brevitas/adapter.py deepsocflow/test/py/test_brevitas_adapter.py
git commit -m "feat: map brevitas activations to legacy XActivation params"
```

---

### Task 5: Adapter — tensor layout conversion

Second slice: reshape brevitas's 2-D dense tensors into the 4-D conv layout the legacy reorder functions expect. These two functions are where a silent transpose bug would live, so they get their own tests.

**Files:**
- Modify: `deepsocflow/py/brevitas/adapter.py`
- Test: `deepsocflow/test/py/test_brevitas_adapter.py`

**Interfaces:**
- Consumes: nothing
- Produces: `to_engine_weight(weight_int) -> np.ndarray` of shape `(1, 1, in_features, out_features)`; `to_engine_activation(x_int) -> np.ndarray` of shape `(1, batch, 1, features)`

- [ ] **Step 1: Write the failing tests**

Append to `deepsocflow/test/py/test_brevitas_adapter.py`:

```python
from deepsocflow.py.brevitas.adapter import to_engine_activation, to_engine_weight


def test_to_engine_weight_shape_and_transpose():
    # torch Linear weight is (out_features, in_features); keras/legacy wants
    # (KH, KW, CI, CO) = (1, 1, in_features, out_features)
    w = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])          # (out=3, in=2)
    e = to_engine_weight(w)
    assert e.shape == (1, 1, 2, 3)
    # element (in=0, out=1) must be w[out=1][in=0] == 3
    assert e[0, 0, 0, 1] == 3
    assert e[0, 0, 1, 2] == 6


def test_to_engine_activation_puts_batch_in_h_slot():
    # (batch, features) -> (XN, XH, XW, CI) = (1, batch, 1, features).
    # Batch lands in H, NOT in N - see xbundle.py:126.
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])          # (batch=4, features=2)
    e = to_engine_activation(x)
    assert e.shape == (1, 4, 1, 2)
    assert e[0, 2, 0, 0] == 1       # row 2 is [1, 0]
    assert e[0, 2, 0, 1] == 0


def test_to_engine_roundtrip_preserves_values():
    x = np.arange(12).reshape(4, 3)
    assert to_engine_activation(x).flatten().tolist() == x.flatten().tolist()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_adapter.py -v`
Expected: FAIL with `ImportError: cannot import name 'to_engine_weight'`

- [ ] **Step 3: Write minimal implementation**

Append to `deepsocflow/py/brevitas/adapter.py` (add `import numpy as np` at the top):

```python
def to_engine_weight(weight_int):
    """torch Linear weight (out_features, in_features) -> legacy conv weight
    (KH, KW, CI, CO) = (1, 1, in_features, out_features).

    The transpose is real: torch stores (out, in), keras stores (in, out)."""
    return np.asarray(weight_int).T[None, None, :, :]


def to_engine_activation(x_int):
    """(batch, features) -> (XN, XH, XW, CI) = (1, batch, 1, features).

    Batch goes in the H slot, not the N slot - this mirrors the legacy dense
    reshape at xbundle.py:126. Getting it backwards produces wrong runtime
    params (XL, X_PAD) without any error."""
    return np.asarray(x_int)[None, :, None, :]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_adapter.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add deepsocflow/py/brevitas/adapter.py deepsocflow/test/py/test_brevitas_adapter.py
git commit -m "feat: convert brevitas dense tensors to legacy engine layout"
```

---

### Task 6: Adapter — bundle objects and topology

Third slice: the `BrevitasBundle` class itself, plus the function that builds one per `FixedPointModel` bundle and registers them into the legacy `BUNDLES` global.

The critical design point: `call_int` is a **no-op**. brevitas already computed every integer tensor, so the adapter pre-populates them and lets legacy `XBundle.export()` contribute only the reorder step.

**Files:**
- Modify: `deepsocflow/py/brevitas/adapter.py`
- Test: `deepsocflow/test/py/test_brevitas_adapter.py`

**Interfaces:**
- Consumes: `act_params`, `to_engine_weight`, `to_engine_activation` (Tasks 4-5); `FixedPointModel` from `deepsocflow.py.brevitas.sim` with `.bundle_order`, `.bundles`, `.trace`, `.pre_softmax`, `.softmax_out`, `.softmax_frac`
- Produces: `build_bundles(model, hw) -> list[BrevitasBundle]` — clears and repopulates the legacy `BUNDLES` global, returns the adapters in `ib` order. Each adapter exposes `ib`, `prev_ib`, `next_ibs`, `next_add_ibs`, `core` (with `.w/.x/.y/.b` XTensors and `.act`, `.type`, `.strides`, `.padding`), `pool=None`, `add=None`, `flatten=False`, `softmax`, `out`, `pre_softmax`, and `call_int(x, hw)`.

- [ ] **Step 1: Write the failing tests**

Append to `deepsocflow/test/py/test_brevitas_adapter.py`. These reuse the graph-JSON fixture style from `test_brevitas_sim.py`.

```python
import json

from deepsocflow.py.brevitas.sim import FixedPointModel


def _bundle_cfg(input_frac, input_bits, weight_values, weight_frac, weight_bits,
                activation, act_bits, act_frac, bias_values, bias_frac, bias_bits,
                softmax=False, input_name=None):
    return {
        "type": "linear",
        "input": input_name,
        "input_bits": input_bits,
        "input_frac": input_frac,
        "input_signed": True,
        "in_features": len(weight_values[0]),
        "out_features": len(weight_values),
        "weight": {"bits": weight_bits, "frac": weight_frac, "values": weight_values},
        "bias": {"bits": bias_bits, "frac": bias_frac, "values": bias_values},
        "activation": activation,
        "act_bits": act_bits,
        "act_frac": act_frac,
        "act_signed": activation != 'relu',
        "softmax": softmax,
    }


def _two_bundle_model(tmp_path):
    """A 2-input -> 2-hidden (relu) -> 2-output (identity+softmax) chain, run
    forward so .trace is populated."""
    layers = {
        "bundle0": _bundle_cfg(
            input_frac=7, input_bits=8,
            weight_values=[[64, 0], [0, 64]], weight_frac=6, weight_bits=8,
            bias_values=[0, 0], bias_frac=13, bias_bits=16,
            activation="relu", act_bits=8, act_frac=6),
        "bundle1": _bundle_cfg(
            input_frac=6, input_bits=8,
            weight_values=[[64, 0], [0, 64]], weight_frac=6, weight_bits=8,
            bias_values=[0, 0], bias_frac=12, bias_bits=16,
            activation="identity", act_bits=8, act_frac=6,
            softmax=True, input_name="bundle0"),
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"layers": layers}))
    model = FixedPointModel(str(path))
    model.load_int_weights(str(path))
    model.forward(model.quantize_input([[1.0, 0.0], [0.0, 1.0]]))
    return model


def test_build_bundles_sets_chain_topology(tmp_path):
    pytest.importorskip("tensorflow")
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    assert [b.ib for b in bundles] == [0, 1]
    assert bundles[0].prev_ib is None
    assert bundles[1].prev_ib == 0
    assert sorted(bundles[0].next_ibs) == [1]
    assert sorted(bundles[1].next_ibs) == []


def test_build_bundles_registers_into_legacy_bundles_global(tmp_path):
    pytest.importorskip("tensorflow")
    from deepsocflow.py.utils import BUNDLES
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    build_bundles(_two_bundle_model(tmp_path), hw)
    assert len(BUNDLES) == 2

    # building again must not accumulate
    build_bundles(_two_bundle_model(tmp_path), hw)
    assert len(BUNDLES) == 2


def test_build_bundles_shift_bits_matches_sim(tmp_path):
    pytest.importorskip("tensorflow")
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    # acc_frac = input_frac + weight_frac = 7 + 6 = 13; act_frac = 6
    # shift_bits = plog_slope + acc_frac - act_frac = 0 + 13 - 6 = 7
    assert bundles[0].core.act.shift_bits == 7
    assert bundles[0].core.act.non_zero == 0      # relu
    assert bundles[1].core.act.non_zero == 1      # identity


def test_call_int_is_a_noop(tmp_path):
    pytest.importorskip("tensorflow")
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(_two_bundle_model(tmp_path), hw)

    before = bundles[0].core.y.itensor.numpy().copy()
    bundles[0].call_int(None, hw)
    assert np.array_equal(bundles[0].core.y.itensor.numpy(), before)


def test_bias_none_when_absent(tmp_path):
    """legacy xbundle.py:135,167 tests `if self.core.b` truthiness - an absent
    bias must be None, never an empty/zero array."""
    pytest.importorskip("tensorflow")
    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.brevitas.hardware import Hardware

    layers = {
        "bundle0": _bundle_cfg(
            input_frac=7, input_bits=8,
            weight_values=[[64, 0], [0, 64]], weight_frac=6, weight_bits=8,
            bias_values=[0, 0], bias_frac=13, bias_bits=16,
            activation="identity", act_bits=8, act_frac=6),
    }
    del layers["bundle0"]["bias"]
    path = tmp_path / "nobias.json"
    path.write_text(json.dumps({"layers": layers}))
    model = FixedPointModel(str(path))
    model.load_int_weights(str(path))
    model.forward(model.quantize_input([[1.0, 0.0]]))

    hw = Hardware(processing_elements=(8, 24), bits_input=8, bits_weights=8,
                  bits_bias=16, bits_sum=32, data_dir=str(tmp_path / 'vectors'))
    bundles = build_bundles(model, hw, has_bias={"bundle0": False})
    assert bundles[0].core.b is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_adapter.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_bundles'`

- [ ] **Step 3: Write minimal implementation**

Append to `deepsocflow/py/brevitas/adapter.py`:

```python
from deepsocflow.py.utils import BUNDLES, XTensor


class _Act:
    """Stands in for legacy XActivation. Only the four attributes the export
    path reads are provided - there is no call_int, because the adapter never
    recomputes anything."""

    def __init__(self, non_zero, plog_slope, shift_bits, out):
        self.non_zero = non_zero
        self.plog_slope = plog_slope
        self.shift_bits = shift_bits
        self.out = out


class _Core:
    """Stands in for legacy XDense. type/strides/padding are what
    get_runtime_params reads off a dense core (dataflow.py:34-47)."""

    type = 'dense'
    strides = (1, 1)
    padding = 'same'

    def __init__(self, w, x, y, b, act):
        self.w, self.x, self.y, self.b, self.act = w, x, y, b, act


class BrevitasBundle:
    """One brevitas bundle wearing legacy XBundle's attribute surface."""

    def __init__(self, ib, core, softmax, out, pre_softmax, prev_ib):
        self.ib = ib
        self.core = core
        self.pool = None
        self.add = None
        self.flatten = False
        self.softmax = softmax
        self.out = out
        self.pre_softmax = pre_softmax
        self.prev_ib = prev_ib
        self.next_ibs = set()
        self.next_add_ibs = set()

    def call_int(self, x, hw):
        """No-op: brevitas already computed every integer tensor and the adapter
        pre-populated them. Legacy XBundle.call_int recomputes the bundle in
        integer arithmetic; doing that here would either duplicate sim.py or
        silently disagree with it."""
        return self.out

    def export(self, hw, is_last):
        from deepsocflow.py.xbundle import XBundle
        return XBundle.export(self, hw, is_last)


def build_bundles(model, hw, has_bias=None):
    """Builds one BrevitasBundle per FixedPointModel bundle, wires the chain
    topology, and registers them into the legacy BUNDLES global (replacing
    whatever was there).

    model must have had forward() called already - the adapter reads .trace.
    has_bias maps bundle name -> bool; defaults to True for every bundle, since
    the JSON exporter only omits "bias" when the layer genuinely has none."""
    has_bias = {} if has_bias is None else has_bias

    for b in BUNDLES:
        b.next_ibs.clear()
        b.next_add_ibs.clear()
    BUNDLES.clear()

    index_of = {name: i for i, name in enumerate(model.bundle_order)}
    adapters = []

    for ib, name in enumerate(model.bundle_order):
        cfg = model.bundles[name]
        trace = model.trace[name]

        acc_frac = cfg['input_frac'] + cfg['weight_frac']
        non_zero, plog_slope = act_params(cfg['activation'])

        act_out = XTensor(
            tensor=np.asarray(trace['out'], dtype=np.float32),
            bits=cfg['act_bits'], frac=cfg['act_frac'], from_int=True)
        act = _Act(
            non_zero=non_zero,
            plog_slope=plog_slope,
            shift_bits=plog_slope + acc_frac - cfg['act_frac'],
            out=act_out)

        w = XTensor(
            tensor=to_engine_weight(cfg['weight']).astype(np.float32),
            bits=hw.K_BITS, frac=cfg['weight_frac'], from_int=True)
        x = XTensor(
            tensor=to_engine_activation(trace['x']).astype(np.float32),
            bits=cfg['input_bits'], frac=cfg['input_frac'], from_int=True)
        y = XTensor(
            tensor=to_engine_activation(trace['y']).astype(np.float32),
            bits=hw.Y_BITS, frac=acc_frac, from_int=True)

        if has_bias.get(name, True):
            b = XTensor(tensor=np.asarray(cfg['bias'], dtype=np.float32),
                        bits=hw.B_BITS, frac=cfg['bias_frac'], from_int=True)
        else:
            b = None

        is_last = ib == len(model.bundle_order) - 1
        if is_last and cfg['softmax']:
            pre_softmax = XTensor(
                tensor=to_engine_activation(model.pre_softmax).astype(np.float32),
                bits=cfg['act_bits'], frac=model.softmax_frac, from_int=True)
            out = XTensor(
                tensor=to_engine_activation(model.softmax_out).astype(np.float32),
                bits=None, float_only=True)
        else:
            pre_softmax = None
            out = act_out

        prev_ib = index_of[cfg['input']] if cfg['input'] is not None else None
        adapter = BrevitasBundle(
            ib=ib,
            core=_Core(w=w, x=x, y=y, b=b, act=act),
            softmax=bool(cfg['softmax']),
            out=out,
            pre_softmax=pre_softmax,
            prev_ib=prev_ib)

        adapters.append(adapter)
        BUNDLES.append(adapter)

    for adapter in adapters:
        if adapter.prev_ib is not None:
            adapters[adapter.prev_ib].next_ibs.add(adapter.ib)

    return adapters
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_adapter.py -v`
Expected: 14 passed

If `XTensor` rejects `bits=None` with `float_only=True`, drop the `bits` argument entirely for that call — `float_only=True` skips the frac/int computation that needs it (`utils.py:25-27`).

- [ ] **Step 5: Confirm existing tests still pass**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_sim.py deepsocflow/test/py/test_brevitas_export_inference.py -q`
Expected: 20 passed

- [ ] **Step 6: Commit**

```bash
git add deepsocflow/py/brevitas/adapter.py deepsocflow/test/py/test_brevitas_adapter.py
git commit -m "feat: build legacy-compatible bundles from a brevitas FixedPointModel"
```

---

### Task 7: `export_rtl` driver

Wire the adapter to the refactored legacy export. This produces the engine-layout files, `.bin` blobs, and `config_fw.h`.

**Files:**
- Modify: `deepsocflow/py/brevitas/export.py`
- Modify: `deepsocflow/py/brevitas/main.py`

**Interfaces:**
- Consumes: `build_bundles` (Task 6), `_export_bundles` (Task 3)
- Produces: `export_rtl(model, hw, x_float, batch_size=4) -> dict` with key `'files'` listing everything written

- [ ] **Step 1: Write `export_rtl`**

Append to `deepsocflow/py/brevitas/export.py`:

```python
def export_rtl(model, hw, x_float, batch_size=4):
    """Exports everything the RTL testbench consumes - engine-layout text files,
    packed .bin blobs, and config_fw.h - by adapting this model's bundles onto
    the legacy XBundle surface and reusing the legacy export path.

    Unlike export_inference (which writes layout-independent golden reference
    text and is left untouched), this drives the real hardware file format.

    config_fw.h is written to the CURRENT WORKING DIRECTORY by the legacy
    exporter (xmodel.py), not into hw.DATA_DIR - run this from the directory
    where the firmware build expects it."""
    import os

    from deepsocflow.py.brevitas.adapter import build_bundles
    from deepsocflow.py.xmodel import _export_bundles

    x_int = model.quantize_input(np.asarray(x_float)[:batch_size])
    model.forward(x_int)

    check_hardware(model, hw)
    build_bundles(model, hw)

    os.makedirs(hw.DATA_DIR, exist_ok=True)
    for entry in os.scandir(hw.DATA_DIR):
        os.remove(entry.path)

    _export_bundles(hw, None)  # x=None: the adapter's call_int is a no-op

    files = sorted(entry.path for entry in os.scandir(hw.DATA_DIR))
    return {'files': files}
```

- [ ] **Step 2: Run it against the real XOR model**

Update `deepsocflow/py/brevitas/main.py` to call it after the existing `export_inference` call, using all four XOR rows:

```python
    from deepsocflow.py.brevitas.export import export_rtl

    print()
    rtl_result = export_rtl(model, hw, X, batch_size=4)
    print(f"Exported {len(rtl_result['files'])} RTL files to {hw.DATA_DIR}")
```

Run: `python -m deepsocflow.py.brevitas.main`
Expected: runs to completion, printing the runtime params per bundle and the file count.

- [ ] **Step 3: Diff the output against the Task 2 reference**

This is the gate that catches adapter bugs before RTL debugging.

```bash
ls deepsocflow/py/brevitas/vectors/ | sort > /tmp/brevitas_files.txt
ls run/reference_qkeras_xor/vectors_qkeras_xor/ | sort > /tmp/qkeras_files.txt
diff /tmp/brevitas_files.txt /tmp/qkeras_files.txt
```

Expected: no difference in file names. Then compare `config_fw.h` field by field against `run/reference_qkeras_xor/config_fw.h`. Every shape-derived field (`w_bpt`, `x_bpt`, `CP`, `IT`, `XN`/`XH`/`XW`, buffer indices) must match. Only calibration-dependent fields (`ca_shift`, and `ca_nzero`/`ca_pl_scale` where activation types differ) may differ, because the two models carry different weights and therefore calibrate to different fracs.

**Any mismatch in a shape-derived field is an adapter bug — fix it here, not after RTL fails.**

- [ ] **Step 4: Commit**

```bash
git add deepsocflow/py/brevitas/export.py deepsocflow/py/brevitas/main.py
git commit -m "feat: export RTL engine-layout files from the brevitas backend"
```

---

### Task 8: Phase 3 — RTL simulation passes

The finish line.

**Files:**
- Modify: `deepsocflow/py/brevitas/main.py`

**Interfaces:**
- Consumes: `export_rtl` (Task 7)
- Produces: a passing verilator simulation of the brevitas XOR model

- [ ] **Step 1: Add hardware config export and verification to `main.py`**

`verify_inference` reads the legacy `BUNDLES` global, which `export_rtl` has already populated, so it needs no adapter of its own.

```python
    from deepsocflow.py.xmodel import verify_inference

    hw.export_json()
    hw.export()  # config_hw.svh, config_hw.tcl, sources.txt

    verify_inference(None, hw, SIM='verilator')
    print("brevitas XOR: RTL simulation PASSED")
```

If `verify_inference`'s signature rejects `None` for `model`, note that it never uses the `model` argument (`xmodel.py:335-398` reads only `BUNDLES` and `hw`) — change the parameter to default to `None` in that case rather than fabricating a model object.

- [ ] **Step 2: Run the full pipeline**

`hw.export()` writes `sources.txt` and `config_hw.svh` relative to the current directory, and the simulator builds in `./build/`. Run from the repository root so those land consistently:

```bash
python -m deepsocflow.py.brevitas.main
```

Expected: `Bundle 0, Error: 0. Passed` through `Bundle 2`, then `brevitas XOR: RTL simulation PASSED`.

- [ ] **Step 3: Debug systematically if it fails**

`verify_inference` checks five points per bundle in this order: `y_raw` (per pass/iteration) → `y_sum` → `y_nhwc` → `y_tiled` → `y_packed`. The **first** failing check localizes the bug:

- `y_raw` fails on bundle 0 → weight or input layout is wrong (revisit `to_engine_weight` / `to_engine_activation`)
- `y_raw` passes but `y_sum` fails → bias mapping or accumulator frac
- `y_sum` passes but `y_nhwc` fails → activation params (`shift_bits`, `non_zero`, `plog_slope`)
- `y_nhwc` passes but `y_packed` fails → bit packing, which is legacy code, so suspect a bit-width field in `config_fw.h`

Use `superpowers:systematic-debugging` rather than guessing. Compare against the Task 2 reference run, which passes.

- [ ] **Step 4: Confirm no regressions**

```bash
python -m pytest deepsocflow/test/py/ -q
cd run && python xor_qkeras.py && python example.py
```

Expected: all Python tests pass; both legacy scripts still pass RTL simulation.

- [ ] **Step 5: Commit**

```bash
git add deepsocflow/py/brevitas/main.py
git commit -m "feat: verify brevitas XOR model against RTL simulation"
```

---

### Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: a passing RTL simulation (Task 8)
- Produces: documentation that lets the next session reproduce the run

- [ ] **Step 1: Replace the "Current Priority" section with a Progress Log entry**

Delete the `# Current Priority (2026-08-10)` section — it is satisfied. Remove the `*(Deprioritized - see Current Priority above...)*` line under `# Known Issues`, replacing it with a note that the issues remain deprioritized now that the pipeline is end-to-end.

Add a Progress Log entry dated 2026-08-11 covering: the adapter approach and why (reuse legacy format ownership rather than re-deriving `config_fw.h`), the `xmodel.py` preamble/bundle-loop split, the qkeras XOR baseline as reference and regression, the four-row export decision, and the exact commands to reproduce the RTL run.

- [ ] **Step 2: Document the verilator dependency**

Record the verilator version from Task 1 and the fact that `run/example.py`'s `export_vivado_tcl` step is not needed for simulation.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: record brevitas RTL end-to-end completion"
```

---

## Self-Review

**Spec coverage:**

| Spec element | Task |
|---|---|
| Phase 0 (verilator, legacy example) | 1 |
| Phase 1 (qkeras XOR baseline, independent weights) | 2 |
| Phase 2 (adapter, `xmodel.py` refactor) | 3, 4, 5, 6, 7 |
| Phase 3 (RTL sim passes) | 8 |
| Adapter mapping table (weights, activations, tensors, topology) | 4, 5, 6 |
| Activation params per type (relu/identity/leaky_relu) | 4 |
| Batch in the H slot | 5 |
| Bias `None` when absent | 6 |
| `call_int` no-op design point | 6 |
| Reuse legacy `XTensor` | 6 |
| Verification layer 1 (legacy regression) | 3 (steps 2-3), 8 (step 4) |
| Verification layer 2 (file diff) | 7 (step 3) |
| Verification layer 3 (RTL sim) | 8 |
| Adapter unit tests, RTL sim not in CI | 4, 5, 6 |
| `config_fw.h` written to cwd | 7 (docstring), 8 (step 2) |
| `NotImplementedError` for conv/pool/residual | 4 (`act_params`) |
| CLAUDE.md update | 9 |

**Gap found and closed:** the spec says the adapter raises `NotImplementedError` for conv/pool/residual/flatten bundles, but `FixedPointModel._build_topology` (`sim.py:62-65`) already rejects any non-`linear` type before the adapter ever sees it, so the adapter only needs to guard activations — which Task 4 does. No extra task needed.

**Type consistency:** `act_params(activation, negative_slope=0.0) -> (non_zero, plog_slope)`, `to_engine_weight(weight_int)`, `to_engine_activation(x_int)`, `build_bundles(model, hw, has_bias=None)`, `export_rtl(model, hw, x_float, batch_size=4)`, `_export_bundles(hw, x)` — each is used in later tasks exactly as defined.

**Known uncertainty, flagged rather than papered over:** Task 6 asserts specific attribute names on legacy `XBundle.export` (`core.w/x/y/b`, `core.act.*`, `pool`, `add`, `flatten`, `softmax`, `pre_softmax`, `out`) read from `xbundle.py:119-192`. If `XBundle.export` touches an attribute this plan missed, Task 6 Step 4 fails with a plain `AttributeError` naming it — add that attribute rather than restructuring.
