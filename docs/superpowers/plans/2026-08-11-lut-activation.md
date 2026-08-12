# Value-LUT Activations — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** carry the curved activations (SiLU, Tanh, Sigmoid, GELU, SELU) from the
working Python model down to a passing RTL simulation and a real PYNQ run, keeping
`Error: 0` bit-exactness at every layer.

**Architecture:** brevitas owns the numbers; the legacy backend owns the file
format. The activation becomes a table lookup executed on the host CPU — the RTL is
untouched.

**Spec:** `docs/superpowers/specs/2026-08-11-lut-activation-design.md`

**Branch:** `brevitas-lut-activation`.

**Status (2026-08-11):** Tasks 1–6 are **done**. `Bundle 0, Error: 0` /
`Bundle 1, Error: 0` on a SiLU XOR model through a real RTL simulation — the C
`quant_lut` and the Python `sim.py` LUT path agree bit for bit. 116 tests passing
(baseline 41). Task 7 (PYNQ) and Task 8 (record it) remain.

**Simulation on `geonosis` uses xsim, not Docker.** This machine has no docker but
does have Vivado 2023.2, and both `Hardware` classes already carry an
`if SIM == 'xsim'` branch, so the whole `deepsocflow/sim/` Docker/Verilator
workaround is unnecessary here:

```python
verify_inference(None, hw, SIM='xsim', SIM_PATH='/home/software/Xilinx/Vivado/2023.2/bin/')
```

with `PATH=/home/software/Xilinx/Vivado/2023.2/bin:$PATH`. A full compile+run is
about 6 seconds. CLAUDE.md's 2026-08-11 FPGA entry flagged this as a real option;
it is now confirmed working.

## Global Constraints

- **Never `git commit` or `git push`.** The user handles both. This overrides any
  skill guidance that says otherwise.
- **Always set `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`** before any Python that imports
  torch on `geonosis`, or 10 unrelated tests fail with `GLIBCXX_3.4.31 not found`:
  ```bash
  export LD_LIBRARY_PATH=/home/jjarasur/.conda/envs/tony_cgra4ml/lib:$LD_LIBRARY_PATH
  ```
- **Run from the worktree root, never `cd run && python ...`** —
  `site-packages/deepsocflow.pth` points at a different checkout.
- Hardware config for XOR, unchanged from `main.py`: `processing_elements=(8,24)`,
  `bits_input=8`, `bits_weights=8`, `bits_bias=16`, `bits_sum=32`,
  **`axi_width=128`** (the default 64 silently corrupts output).
- **`ca_lut_idx = -1` must leave every existing bundle byte-identical.** The qkeras
  path and every relu/identity bundle keep running `quant_lrelu`. Verify by diffing
  exported artifacts, not by assuming.
- Scope is the core activation (`ca_*`) only. Do not touch the residual-add
  (`aa_*`) or pool (`pa_*`) slots — the brevitas adapter supports neither, so any
  code there would be untested speculation. Same reasoning CLAUDE.md already
  records for `adapter.py`'s residual/mid-network-softmax defects.

---

## Phase 1 — model layer — **DONE**

Already implemented and measured on this branch. Listed so the ledger is complete.

- [x] `lut.py` — `ActLut`, raw two's-complement indexing, power-of-two guards
- [x] `sim.py` — LUT path in `forward()`, `lut_grid` override, grid precedence
- [x] `ptq.py` — `act_input_bits`, `act_in_*` export
- [x] `lut_poc.py` — five reproducible experiments
- [x] 102 tests passing (baseline 41)

Re-verify before starting Phase 2:

```bash
python -m pytest deepsocflow/test/py/ -q          # expect 102 passed, 4 skipped
python -m deepsocflow.py.brevitas.lut_poc         # expect 0.00% on every 1b row
```

---

### Task 2: `check_hardware` validates the LUT against the hardware config

Do this first — it is the cheapest task and it is a gate. If `act_bits` or
`act_in_bits` do not fit the hardware's word widths, everything downstream produces
wrong `.bin` blobs silently, exactly as the `K_BITS`/`B_BITS` gap did before it was
closed on 2026-08-11.

**Files:** `deepsocflow/py/brevitas/export.py`, `deepsocflow/test/py/test_brevitas_export_inference.py`

- [x] **Step 1:** In `check_hardware`, for every bundle carrying a LUT, assert:
  - `lut.out_bits <= hw.X_BITS` — the table's output is stored as a packed
    activation word
  - `lut.in_bits <= 16` — sanity bound on table size; a wider index means the
    model was built with an `act_input_bits` nobody intends to ship
  - `lut.index_shift(acc_frac) >= 0` — already raised inside `ActLut`, but assert
    it here too so the failure names the bundle
  Follow the existing assertion-message style: name the bundle and both values.

- [x] **Step 2:** Two tests mirroring the existing `test_hardware_*_raises` pair —
  one where `act_input_bits` is wider than `X_BITS` allows, one where the model is
  fine. Assert on the message naming the bundle.

- [x] **Step 3:** `python -m pytest deepsocflow/test/py/ -q`

---

### Task 3: `adapter.py` carries the table to the legacy exporter

**Files:** `deepsocflow/py/brevitas/adapter.py`, `deepsocflow/test/py/test_brevitas_adapter.py`

The one subtlety: `_Act.shift_bits` currently means "shift onto the output grid"
(`adapter.py:154` computes `plog_slope + acc_frac - cfg['act_frac']`). On a LUT
bundle it must mean "shift onto the **index** grid" — `acc_frac - act_in_frac`.
Same field, different target. Getting this wrong produces a table indexed at the
wrong scale, which will still run and still look plausible.

- [x] **Step 1:** Extend `act_params` so curved activations return LUT parameters
  instead of raising `NotImplementedError`. Keep raising for anything genuinely
  unsupported. Preserve the existing power-of-two assertion for `leaky_relu`
  including the `negative_slope > 0` guard.

- [x] **Step 2:** Give `_Act` two new attributes, `lut` (the `ActLut`, or `None`)
  and `lut_bits`. Set `shift_bits = acc_frac - act_in_frac` when a LUT is present,
  leaving the existing formula otherwise. Document *why* the meaning changes, in
  the style of the existing `bias_val_shift` comment.

- [x] **Step 3:** Tests: a curved bundle produces an `_Act` with a table and the
  index-grid shift; a relu bundle still produces `lut is None` and the original
  shift; `act_params('leaky_relu')` still rejects the default slope with a legible
  message.

- [x] **Step 4:** `python -m pytest deepsocflow/test/py/ -q`

---

### Task 4: `rtl_export.py` emits the tables and the new `Bundle_t` fields

**Files:** `deepsocflow/py/brevitas/rtl_export.py`, `deepsocflow/test/py/test_brevitas_lut_export.py` (new)

**This is the task where the previous adapter work went wrong four times.** CLAUDE.md
is explicit: attributes that legacy fills in inside `call_int` — which this adapter
never runs — are invisible until a test drives the real `_export_bundles` path and
asserts on the emitted `config_fw.h` text. Do not test the adapter's attributes and
call it done.

- [x] **Step 1:** Collect the distinct tables across bundles in the same per-bundle
  loop that already writes `config_fw.h`/`config.json`. Deduplicate identical
  tables — two bundles with the same activation and the same grid should share one
  entry. Assign each bundle its `ca_lut_idx`, or `-1`.

- [x] **Step 2:** Emit into `config_fw.h`:
  ```c
  #define N_LUTS      <n>
  #define LUT_ENTRIES <max entries over all tables>
  static const i8 LUTS[N_LUTS][LUT_ENTRIES] = { ... };
  ```
  Narrower tables are zero-padded to `LUT_ENTRIES`. Emit `.ca_lut_idx` and
  `.ca_lut_bits` in the `Bundle_t` initializer alongside the existing `.ca_*`
  fields. When there are no tables, emit `N_LUTS 0` and every `ca_lut_idx` as `-1`
  — the file must stay valid C.

- [x] **Step 3:** Mirror the same data into `config.json` under `"luts"`, plus the
  two new per-bundle fields, built from the same local variables in the same loop
  so the two outputs cannot diverge.

- [x] **Step 4:** Tests that run the **real** `_export_bundles` and assert on the
  emitted text:
  - a SiLU model emits `N_LUTS 1` (both bundles share one table) and both bundles
    carry `ca_lut_idx=0`
  - the table's first entries match `ActLut.table` exactly
  - a relu-only model emits `N_LUTS 0` and every `.ca_lut_idx=-1`
  - `config.json`'s `"luts"` equals what `config_fw.h` contains

- [x] **Step 5: regression gate.** Export the **existing LeakyReLU XOR model** and
  byte-diff every artifact against a snapshot taken before this task. Only
  `config_fw.h` and `config.json` may differ, and only by the added fields. If any
  `.bin` or `*_exp.txt` differs, stop — something is being computed differently for
  models that should be untouched.

- [x] **Step 6:** `python -m pytest deepsocflow/test/py/ -q`

---

### Task 5: `runtime.h` executes the table

**Files:** `deepsocflow/c/runtime.h`

- [x] **Step 1:** Add the two fields to `Bundle_t` (`runtime.h:31-34`), next to the
  other `ca_*` fields.

- [x] **Step 2:** Add `quant_lut` beside `quant_lrelu` (`runtime.h:153`), matching
  its style — `static inline i32`, no allocation, ARM-friendly:
  ```c
  static inline i32 quant_lut(i32 x, i8 shift, i8 in_bits, const i8 *restrict lut){
    x = shift_round(x, shift);
    x = clip(x, -(1<<(in_bits-1)), (1<<(in_bits-1))-1);
    return lut[x & ((1<<in_bits)-1)];   // raw two's-complement index, table is pre-permuted
  }
  ```
  The clip is load-bearing: masking alone wraps, and an accumulator past the
  table's range must saturate to the end entry, not fold to the opposite sign.

- [x] **Step 3:** Dispatch at the core-activation site only (`runtime.h:383`):
  ```c
  out_val = pb->ca_lut_idx >= 0
    ? quant_lut(out_val, pb->ca_shift, pb->ca_lut_bits, LUTS[pb->ca_lut_idx])
    : quant_lrelu(out_val, pb->ca_nzero, pb->ca_shift, pb->ca_pl_scale);
  ```
  Leave `:390` (residual add) and `:475` (pool) alone — out of scope per the
  global constraints.

- [x] **Step 4:** Confirm it still compiles when `N_LUTS` is 0.

---

### Task 6: RTL simulation

**Files:** none — this is a verification gate.

- [ ] **Step 1 — NOT RUN on `geonosis`.** The qkeras baselines need TensorFlow,
  which has no Python 3.13 wheel (see CLAUDE.md's 2026-08-11 TF-decoupling entry),
  so they cannot run on this machine. Covered indirectly instead by Task 4 Step 5's
  byte-diff (brevitas path unchanged) and by compiling `runtime.h`'s guard with
  `N_LUTS` undefined (legacy path compiles the lookup out entirely). Run these on a
  machine with TF before merging:
  ```bash
  python deepsocflow/sim/docker_sim.py run/xor_qkeras.py
  python deepsocflow/sim/docker_sim.py run/example.py
  ```

- [x] **Step 2:** Point `main.py` at a SiLU model built with `act_input_bits=8`
  and run:
  ```bash
  python deepsocflow/sim/docker_sim.py deepsocflow/py/brevitas/main.py
  ```
  Expect `Bundle 0, Error: 0` and `Bundle 1, Error: 0` — **exactly zero**, not
  small. Anything non-zero on a non-softmax bundle means the C and Python tables
  disagree; bundle 2's softmax check cannot fail and proves nothing (see the spec).

- [x] **Step 3:** If a bundle is non-zero, the likely causes in order: `ca_shift`
  computed against `act_frac` instead of `act_in_frac` (Task 3 Step 2); the clip in
  `quant_lut` dropped or applied after the mask; table padding misread because
  `ca_lut_bits` is wrong.

---

### Task 7: PYNQ driver and real hardware

`pynq_driver.py` lives outside this repo, at `run/work_pynq/pynq_deploy/` on
`geonosis` (gitignored).

- [ ] **Step 1:** Mirror Task 5's `quant_lut` and dispatch in `pynq_driver.py`,
  reading `"luts"` and the two per-bundle fields from `config.json`.

- [ ] **Step 2:** Extend `test_pynq_driver_xor.py` — the existing harness that runs
  the driver's CPU-side logic against `BUNDLES[ib].ye_exp_p[ip][it]` without real
  hardware — to cover a LUT bundle. This is where a driver bug is cheap to find;
  the last one (`_tile_write`'s stale padding value) was found exactly here.

- [ ] **Step 3:** Run the notebook on the board with the **existing** bitstream.
  No re-synthesis: nothing in `deepsocflow/rtl/` changed. If a rebuild seems
  necessary, something in Tasks 3–5 leaked into the fabric config and should be
  found rather than worked around.

---

### Task 8: record it

- [ ] **Step 1:** CLAUDE.md — replace the "no way to execute non-piecewise-linear
  activations" Known Issue with a Progress Log entry. It must correct the claim
  that the gap exists "at the C/RTL deployment layer": the RTL was never involved.

- [ ] **Step 2:** Record the measured numbers (1a vs 1b per activation, the
  bit-exactness law, sigmoid's wrong prediction under 1a) so the next person does
  not re-derive them, and the two open questions: what `act_input_bits` should be
  for a real workload, and that XOR was too easy to answer it.

- [ ] **Step 3:** Add the `LD_LIBRARY_PATH` environment trap to
  `deepsocflow/sim/README.md`.

---

## Open questions to resolve during the work

1. **`act_input_bits` default.** 8 matches `X_BITS`, but bit-exactness holds down
   to 4 bits (16 B tables). Nothing yet says which is right for a real model —
   XOR's decision margin barely moves at any width.
2. **Table sharing across bundles.** Task 4 deduplicates identical tables. On a
   larger network with per-layer `act_frac` values this may dedupe to almost
   nothing; if so, `LUT_ENTRIES` padding starts to waste real space and a flat
   array with per-bundle offsets becomes the better layout.
