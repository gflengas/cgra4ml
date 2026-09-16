# Brevitas backend: end-to-end to RTL simulation

**Date:** 2026-08-11
**Designed on:** `brevitas-golden-ref`
**Implement on:** a new branch off `brevitas-golden-ref` (per the user; this
design work stays where it is)
**Status:** design approved, not yet implemented

## Goal

Make the brevitas backend drive a passing RTL simulation of the XOR model.

Today the brevitas pipeline stops at a golden reference nothing downstream reads:
`export.py::export_inference` writes flat text files (`y_exp.txt`,
`{ib}_y_nhwc_exp.txt`) but not the engine-layout files, `.bin` blobs, or
`config_fw.h` that the RTL testbench actually consumes. This spec closes that
gap.

**Definition of done:** `verify_inference` runs a verilator simulation of the
XOR model built by the brevitas backend and every per-bundle assertion passes.

## Non-goals

Explicitly out of scope, confirmed with the user:

- All six CLAUDE.md Known Issues (non-piecewise-linear activations, skip
  connections, conv/pool hyperparameters in the JSON, rounding mode, LeakyReLU
  power-of-two validation, the `xor.py` seed bug). They are deprioritized until
  the pipeline is genuinely end-to-end.
- conv / pooling / residual-add / flatten bundles. XOR has none. The adapter
  raises `NotImplementedError` for them rather than failing silently.
- Dropping the TensorFlow dependency. CLAUDE.md already records that
  `deepsocflow/__init__.py` pulls in TF/qkeras for any `deepsocflow.*` import,
  so reusing legacy code adds no new coupling.
- Replacing the existing Phase 1 `export_inference` (flat text). It stays
  untouched so the current 20 tests keep passing.

## Approach

Three approaches were considered for producing the RTL-facing files:

- **A. Adapter** — make brevitas bundles expose the attribute surface legacy
  `XBundle` provides, register them into the legacy `BUNDLES` global, and reuse
  the legacy export path unchanged.
- **B. Brevitas-side export** — call legacy `dataflow.py` primitives
  (`get_runtime_params`, `reorder_*_q2e_conv`, `pack_words_into_bytes`) from a
  new brevitas-owned `export_inference`, writing our own `config_fw.h` and
  buffer allocator.
- **C. Full port** — transcribe everything, no TF.

**Chosen: A.** The highest-risk artifact is `config_fw.h` — roughly 40 fields
per bundle of subtle shape-derived arithmetic (`xmodel.py:154-290`). B forces us
to re-derive it, which is exactly the kind of code that diverges silently and
only surfaces as an inexplicable RTL failure. A gets `config_fw.h`, output
buffer allocation, `.bin` packing, and every engine-layout file for free and
correct by construction, concentrating all new risk in one adapter that the
legacy baseline (below) lets us validate by diffing. C was rejected: most work,
most risk, and TF cannot be dropped anyway.

If the TF dependency is ever removed, A can be refactored into B later — with a
passing RTL simulation as the safety net that does not exist today.

## Architecture

brevitas owns the *numbers* (quantization, integer arithmetic, golden values).
Legacy owns the *format* (engine layout, `config_fw.h`, buffer allocation). The
adapter is the single seam between them.

```
xor.py (train float)
  └→ ptq.py::quantized_model  ──→ xor_graph.json          [exists]
        └→ sim.py::FixedPointModel (int forward)           [exists]
              └→ adapter.py                                [new — the only new risk]
                    └→ legacy BUNDLES
                          ├→ legacy export_inference  → vectors/*.txt, *.bin, config_fw.h
                          └→ legacy verify_inference  → RTL sim + diff   [reused unchanged]
```

`verify_inference` (`xmodel.py:335-398`) operates purely on `BUNDLES` attributes
(`b.ye_exp_p`, `b.oe_sum_exp`, `b.oe_exp_nhwc`, `b.xe`, `b.r`, `b.softmax`) —
precisely what the adapter populates. It needs no changes.

## Phases

Each phase has an exit criterion provable on its own, so a failure localizes to
one cause.

| Phase | Work | Exit criterion |
|---|---|---|
| 0 | Install verilator. Run the existing `run/example.py` legacy flow unmodified. | RTL sim passes. Proves the toolchain works before any of our code exists. |
| 1 | Build a qkeras XOR-equivalent (`XDense` 2→3→3→2, independent weights) and run the full legacy flow. | RTL sim passes. Yields a reference set of every file the RTL consumes, for a model *the same shape as* XOR. |
| 2 | Write `adapter.py` + the brevitas driver; refactor legacy `export_inference` to split its keras preamble from the bundle loop. | Files produced match the Phase 1 reference in name, count, and size; `config_fw.h` matches on every shape-derived field (see Verification for which fields may legitimately differ). |
| 3 | Run `verify_inference` on the brevitas bundles. | **RTL sim passes — this is the finish line.** |

Phases 0 and 1 touch no production code; they purchase a reference. Phase 1 also
becomes the regression test for the Phase 2 refactor of `xmodel.py`, which is why
it must precede it.

Phase 1 uses independent weights (not the trained `xor.pt`). Matching weights
exactly would require reconciling qkeras power-of-two quantization against
brevitas fixed-point quantization, a separate problem whose payoff — exact value
diffs rather than structural diffs — is not worth it: structure is what we need
to validate.

## Feasibility (verified, not assumed)

Ran `get_runtime_params` + `create_headers` against XOR's real shapes with
`Hardware(processing_elements=(8,24), bits_input=8, bits_weights=8,
bits_bias=16, bits_sum=32)`:

- All three bundles produce valid runtime params with no assertion failures.
- `CP=1, IT=1, XL=1` for every bundle — the simplest possible path through the
  hardware, ideal for a first bring-up.
- `bits_input=8, bits_weights=8, bits_bias=16` is a supported combination
  (`hardware.py:62-63`); `run/` contains a `SYS_BITS(x=8, k=8, b=16)` example.
- Everything the RTL sim needs is in the repo: `deepsocflow/rtl/`,
  `deepsocflow/c/sim.c`, `deepsocflow/firebridge/`. `hw.export()` generates
  `sources.txt`. No git submodules.

The legacy dense-to-conv reshape puts **batch in the H slot**: `x` becomes
`(1, XN, 1, CI)`, not `(XN, 1, 1, CI)` (`xbundle.py:126`). Getting this backwards
would silently produce wrong runtime params.

## The adapter

New file: `deepsocflow/py/brevitas/adapter.py`. Wraps each `FixedPointModel`
bundle in an object exposing legacy `XBundle`'s attribute surface, and registers
it into the legacy `BUNDLES` global.

Every value the legacy path needs is derivable from data brevitas already has:

| Legacy attribute | Source | Note |
|---|---|---|
| `core.w.itensor` `(KH,KW,CI,CO)` | `bundle['weight']` `(out,in)` | **Transpose required** — torch is `(out,in)`, keras is `(in,out)` — then reshape `(1,1,CI,CO)` |
| `core.x.itensor` `(XN,XH,XW,CI)` | `trace[name]['x']` `(batch,in)` | reshape `(1,batch,1,CI)` — batch in the H slot |
| `core.y.itensor` | `trace[name]['y']` | `y` is the bias-free conv-sum, matching legacy's definition |
| `core.b.itensor` | `bundle['bias']` | Must be `None` when absent, not an empty array — `xbundle.py:135,167` tests truthiness |
| `core.act.non_zero` | per activation type, see below | |
| `core.act.plog_slope` | per activation type, see below | |
| `core.act.shift_bits` | `acc_frac - act_frac` | Identical to `sim.py:173` |
| `core.act.out.{bits,frac}` | `act_bits`, `act_frac` | |
| `core.type/strides/padding` | `'dense'`, `(1,1)`, `'same'` | constants |
| `pool` / `add` / `flatten` | `None` / `None` / `False` | XOR has none |
| `softmax`, `pre_softmax`, `out` | already on `FixedPointModel` | |
| `next_ibs` / `prev_ib` / `ib` | from `bundle_order` + `input` | topology is a plain chain |

Activation parameters differ per type (XOR uses both `relu` and `identity`), per
`xlayers.py:20-24`:

| Activation | `non_zero` | `plog_slope` |
|---|---|---|
| `relu` (slope 0) | 0 | 0 |
| `identity` (type `None`, slope forced to 1) | 1 | 0 |
| `leaky_relu` (slope `2^-k`) | 1 | k |

**Design point:** the adapter pre-populates `.itensor` on each tensor and does
*not* re-run `call_int` — brevitas already computed the integer values. Legacy
`XBundle.export()` then contributes only the reorder step, producing `we`, `xe`,
`ye_exp`, `ye_exp_p`, `oe_*`, and `be`.

For the XTensor shim, reuse legacy `XTensor(..., from_int=True)` directly rather
than writing a substitute, so `ftensor` is derived by the same formula the legacy
assertions check against.

## Files touched

| File | Change |
|---|---|
| `deepsocflow/py/brevitas/adapter.py` | new — the adapter described above |
| `deepsocflow/py/brevitas/export.py` | add `export_rtl(model, hw, x_float, batch_size=4)`; leave existing `export_inference` untouched |
| `deepsocflow/py/xmodel.py` | refactor `export_inference` into a keras preamble plus a `_export_bundles(hw)` operating on already-populated `BUNDLES`, so both backends share one code path |
| `deepsocflow/py/brevitas/main.py` | call the RTL export path |
| `deepsocflow/test/py/test_brevitas_adapter.py` | new — adapter unit tests, no simulator needed |
| `CLAUDE.md` | on completion, fold "Current Priority" into a Progress Log entry and document how to reproduce the RTL run |

## Verification

Three layers, ordered so failures localize:

1. **Legacy regression.** After refactoring `xmodel.py`, the Phase 1 qkeras
   baseline must still pass RTL simulation. A failure here means the refactor is
   wrong, not the adapter.
2. **File-level diff, before touching RTL.** Compare brevitas output against the
   Phase 1 reference: `config_fw.h` field by field, and the vector files by name,
   count, and size. Because the two models carry different weights, they
   calibrate to different fracs, so the fields that may legitimately differ are
   exactly the calibration-dependent ones — the per-activation `ca_shift`
   (and `ca_nzero`/`ca_pl_scale` if the activation types were not matched).
   Every shape-derived field (`w_bpt`, `x_bpt`, `CP`, `IT`, `XN`/`XH`/`XW`,
   buffer indices) must match exactly; a mismatch there is an adapter bug.
3. **RTL simulation.** Reused `verify_inference` checks five points per bundle —
   `y_raw` (per pass and iteration), `y_sum`, `y_nhwc`, `y_tiled`, `y_packed` —
   asserting `error == 0`, except the softmax output which uses `atol=0.5`.

Automated tests cover the adapter only (weight transpose direction, dense
reshape placing batch in the H slot, `shift_bits`/`non_zero`/`plog_slope` per
activation type, `next_ibs`/`prev_ib` topology, bias-`None` handling). RTL
simulation stays a manual gate — it is slow and requires verilator, so it does
not belong in CI.

## Input data

All four XOR rows are exported (`XH=4`, `XL=1` — verified valid), rather than the
legacy `batch_size=1` convention. This makes the RTL compute the full XOR truth
table, so a pass proves the pipeline works rather than merely that one row
survives. Phase 1 and Phase 2 use the same convention so their outputs stay
comparable.

## Risks

| Risk | Mitigation |
|---|---|
| verilator version incompatible with the RTL — the largest risk and outside our control | Phase 0 surfaces it before any of our code is written. A failure there is an environment problem, and a signal to regroup rather than push on. |
| Refactoring `xmodel.py` breaks the qkeras backend | Phase 1 baseline is re-run as a regression test |
| `config_fw.h` is written to the current directory, not `DATA_DIR` (`xmodel.py:154`) | Pin the working directory explicitly in the driver and document it, rather than depending on where the command is run from |
| `if self.core.b` relies on XTensor truthiness | Adapter passes `None` for absent bias; covered by a unit test. XOR always has biases, so this is latent rather than immediate. |
| Adapter maps a value wrongly and silently | Layer 2 (file diff) catches it before RTL |

## Open items

None blocking. Phase 0's outcome determines whether the rest proceeds as
written.
