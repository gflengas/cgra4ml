# Value-LUT Activations — Design

**Goal:** make the curved activations (SiLU, Tanh, Sigmoid, GELU, SELU) deployable
end to end — brevitas → `sim.py` → `config_fw.h` → `runtime.h` → RTL → PYNQ — with
the same bit-exact agreement between layers that `relu`/`identity` already have.

**Status:** the model/Python half is implemented and measured on branch
`brevitas-lut-activation` (see "What is already done"). This spec covers the whole
design so the remaining firmware/export half can be built against a written
contract rather than by re-deriving it.

---

## The finding this design rests on

**Activation is computed on the host CPU, not in RTL.** The RTL under
`deepsocflow/rtl/` is a MAC array plus DMA: it multiplies, accumulates, and writes
raw `i32` to OCM. Everything after that — bias add, activation, pooling, softmax —
is C in `deepsocflow/c/runtime.h`, running on the ARM (ZCU102) or via the Python
`pynq_driver.py` on the PYNQ side.

Verified three ways:

- no activation logic exists under `deepsocflow/rtl/` (`grep -rn lrelu` finds nothing)
- `quant_lrelu` is called only from `runtime.h:383`, `:390`, `:475`
- `create_headers` (`deepsocflow/py/dataflow.py`) packs only tiling geometry
  (`KW`, `XW`, `XL`, `CM`, `XN`, weight RAM addresses) into the header the RTL
  receives — no activation parameters reach the fabric at all

**Consequence: adding an activation requires no RTL edit and no bitstream rebuild.**
The `design_1.bit` already validated on real hardware stays valid. This corrects
the current CLAUDE.md Known Issue, which describes the gap as existing "at the
C/RTL deployment layer" — the C half is right, the RTL half is not.

The project's "no multiplier, shift only" rule is a constraint on the *RTL
datapath*. A table lookup honours it anyway: index by shift, then load.

---

## Approach: value LUT, variant 1b

The activation becomes a precomputed table indexed by the accumulator after a
shift:

```
idx = clip(shift_round(acc, acc_frac - lut.in_frac), lut.in_bits)
out = lut[idx & (2**in_bits - 1)]          # raw two's-complement index
```

Three steps, no arithmetic beyond the shift the pipeline already performs. Prior
art: hls4ml's `UnaryLUT` (`hls4ml/converters/keras_v3/hgq2/unary_lut.py`), whose
raw-bit indexing trick (table pre-permuted, positive half first) is reused here so
no `+2**(bits-1)` bias add is needed.

### Why 1b and not 1a

The two variants differ only in **what brevitas computes**, not in the table
layout or the firmware — a 1a and a 1b table on the same grid are byte-identical.

| | brevitas evaluates the activation on | consequence |
|---|---|---|
| **1a** | the full-precision accumulator (`acc_frac` bits) | the table must index at `acc_frac` to match — measured 64–128 KB per activation |
| **1b** | the accumulator quantized to `act_in_bits` first | a `2**act_in_bits` table matches exactly, by construction |

Measured on trained XOR models, comparing against brevitas's own activation output
(captured by forward hook on `core.act`, not a reconstructed reference):

| activation | 1a @ 256 B | 1b @ 4–10 bits |
|---|---|---|
| SiLU | 29.7 % (worst 8 LSB) | **0 %** at every width |
| Tanh | 76.6 % (worst 31 LSB) | **0 %** |
| GELU | 12.5 % (worst 2 LSB) | **0 %** |
| Sigmoid | 100.0 % (worst 71 LSB) — **prediction wrong** | **0 %** |
| SELU | 21.9 % (worst 1 LSB) | **0 %** |

Sigmoid under 1a is the decisive case: it does not merely lose precision, it
changes the model's answer. 1a is not a cheaper-but-acceptable option.

Bit-exactness under 1b holds at 4, 5, 6, 7, 8 and 10 input bits — it comes from the
table sharing brevitas's grid, not from that grid being fine. Table size is
`2**act_input_bits` bytes per activation (16 B at 4 bits, 256 B at 8).

### Why bit-exactness is the requirement, not "close enough"

One activation ends up implemented five times — brevitas, `sim.py`, `runtime.h`,
the RTL testbench's expected values, and `pynq_driver.py`. When all five agree
exactly, any disagreement is a bug and is immediately localizable. Under a
tolerance, a real defect hides inside it. The project already has a cautionary
example: `xmodel.py:384` checks softmax with `atol=0.5` against probabilities in
`[0,1]` — an assertion that cannot fail. The `Error: 0` on bundles 0 and 1 is what
actually proves RTL matches Python.

### Invariants this does not break

Measured, not assumed, on both 1a and 1b models:

- `act_frac[N] == input_frac[N+1]` — the "single quantization point per bundle"
  property from the 2026-08-10 work. **Holds.** That invariant is about the seam
  *between* bundles; `input_quant` sits *inside* one, between accumulator and
  activation, where the pipeline already shifts.
- `acc_frac == bias_frac` — asserted at `sim.py:169`. **Holds.**

So 1b changes the shift *amount* (`acc_frac - act_in_frac` instead of
`acc_frac - act_frac`), not the shape of the datapath. Runtime cost versus 1a is
zero; versus `quant_lrelu` it is one table load.

### Rejected alternatives

- **Multi-threshold (FINN / QONNX `MultiThreshold`)** — exact for monotonic
  functions, and QONNX-native, which is attractive given this backend's target
  format. But SiLU and GELU are not monotonic, and it costs ~1 KB plus ~8
  comparisons per value against the LUT's 256 B and one load.
- **Integer polynomial (I-BERT)** — needs only ~12 B and matches this project's
  INT8×INT8→INT32 shape exactly, but is approximate by construction (GELU max
  error 1.8e-2), which forfeits bit-exactness.
- **Multiplierless PWL (GRAU / ISPA / OML-PLAC)** — the option CLAUDE.md already
  researched. Never bit-exact regardless of segment count. Its measured size
  advantage assumed a 10-bit-indexed LUT competitor; against a 256 B 1b table it
  has none.

---

## Where the table lives

Baked into `config_fw.h` as `static const i8`, emitted by `rtl_export.py` in the
same per-bundle loop that already writes `config_fw.h` and `config.json` — so the
two cannot drift apart, the same reasoning that governs the existing `config.json`
export.

Not appended to `wbx.bin`: that needs offset plumbing and DMA, and the tables are
small static data that never has to reach the fabric.

---

## Contract between layers

**Graph JSON** (`ptq.py::export_graph_json`) — new optional keys on a bundle whose
activation is curved and whose model was built with `act_input_bits`:

```
act_in_bits, act_in_frac, act_in_scale, act_in_signed
```

Absent for 1a models and for relu/leaky_relu/identity. `sim.py` treats their
absence as "index at the output grid" (1a).

**`Bundle_t`** (`runtime.h`) — two new fields:

```
i8 ca_lut_idx;    // index into LUTS, or -1 for the quant_lrelu path
i8 ca_lut_bits;   // index width; clip bound before the lookup
```

`ca_shift` is reused, carrying `acc_frac - act_in_frac` on LUT bundles instead of
`acc_frac - act_frac`. `ca_nzero`/`ca_pl_scale` are unused on those bundles.

**`config_fw.h`** — one table array:

```c
#define N_LUTS  2
#define LUT_ENTRIES 256        /* max over all tables; narrower ones are padded */
static const i8 LUTS[N_LUTS][LUT_ENTRIES] = { ... };
```

**`config.json`** — the same content mirrored under a `"luts"` key, for
`pynq_driver.py`.

---

## Scope

**In:** the core activation (`ca_*`) on dense bundles, for the five curved
activations, on the brevitas path only.

**Out:**

- the residual-add (`aa_*`) and pool (`pa_*`) activation slots — the brevitas
  adapter has no residual or pooling support at all, so there is nothing to test
  against
- the legacy qkeras path (`xlayers.py::XActivation`) — unchanged
- choosing `act_input_bits` for real workloads. XOR is too easy to show the cost:
  decision margin barely moves even at 4 bits. The width must be re-measured on a
  model whose decisions are not near-saturated.

---

## What is already done

On branch `brevitas-lut-activation`, 102 tests passing (baseline 41):

- `deepsocflow/py/brevitas/lut.py` — `ActLut` table builder, raw two's-complement
  indexing, power-of-two guards (`index_shift` rejects a left shift; fracs must be
  integers; leaky_relu slope must be a negative power of two), plus
  `exact_activation`/`mismatch` for measurement
- `deepsocflow/py/brevitas/sim.py` — LUT path in `forward()`, `lut_grid` override,
  grid precedence (explicit override → exported `act_in_*` → 1a default)
- `deepsocflow/py/brevitas/ptq.py` — `act_input_bits` (opt-in, `None` keeps
  current behaviour), applied only to curved activations, and `act_in_*` export
- `deepsocflow/py/brevitas/lut_poc.py` — five reproducible experiments
- `deepsocflow/test/py/test_brevitas_lut.py`, `test_brevitas_lut_1b.py`

## Environment note

On `geonosis`, `import torch` fails with `GLIBCXX_3.4.31 not found` unless
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` is set — the system libstdc++ predates what the
conda torch build needs. Pre-existing and unrelated to this work, but it silently
fails 10 tests without it.
