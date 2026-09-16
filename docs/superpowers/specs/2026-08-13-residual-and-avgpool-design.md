# Residual add and average pooling on the brevitas backend — design

## Context

Conv/stride/maxpool/flatten landed 2026-08-12/13 and run bit-exact through RTL simulation. The two capabilities still missing before the brevitas backend covers what `run/example.py` and `run/resnet50.py` exercise are **residual (skip) add** and **average pooling**. Both were deferred deliberately, and both already have partial implementations scattered across the tree — this document surveys what exists, records three defects found while surveying, and designs the remaining work.

The headline recommendation is counter-intuitive and worth stating up front: **do residual first, average pooling second.** Residual needs no C change; average pooling needs a fix inside `runtime.h`, which is shared with the qkeras backend and cannot be regression-tested on this machine (no TensorFlow under Python 3.13).

---

## What already exists — do not rebuild

| Piece | Where | State |
|---|---|---|
| `QuantResidualAdd` | `xlayer/quantOperation.py` | `forward` works (`act(x + x_add)`), tracks `source_ib`. `call_int` raises. |
| Skip wiring in the bundle | `xbundle/xbundle.py:36-41` | `XBundle.call(x, x_add)` sets `add.source_ib`, appends to the source's `next_add_ibs`, applies the add. Complete. |
| `XTensor.add_val_shift` | `brevitas/xtensor.py:61` | Already ported to numpy. Computes the two alignment shifts and the widened result. |
| Add-buffer allocation | `rtl_export.py:181-208` | Allocates/frees `add_out_buffer_idx` per bundle, mirroring the output-buffer logic. Complete. |
| Residual add | `c/runtime.h:447-453` | Complete: reads `add_buffers[add_in_buffer_idx]`, adds, applies `aa_*`. |
| `QuantAvgPool2d` | `xlayer/quantPooling.py` | Exists, but is brevitas's `TruncAvgPool2d` — **wrong semantics**, see below. |
| Avg pool divide | `c/runtime.h:534-538` | `div_round(result, count)` with an edge-aware `count`. |
| Legacy reference | `py/xlayers.py::XAdd`, `XPool.call_int` | The bit-exact behaviour both sides must reproduce. |

So the C side of residual is **done**, and the Python side is mostly plumbing.

---

## Defects found while surveying

### D1 — `runtime.h`'s avg-pool activation is applied to the wrong variable (real bug)

```c
if (pb->pool == POOL_AVG) {
  i32 count  = (ph_end-ph_beg)*(pw_end-pw_beg);
  result  = div_round(result, count);
  out_val = quant_lrelu(out_val, pb->pa_nzero, pb->pa_shift, pb->pa_pl_scale);  // <-- out_val, not result
}
tile_write(result, ...);                                                        // <-- result is what is written
```

`out_val` at this point is the **pre-pool** value of the current pixel. The activation is computed on it, assigned back to it, and then discarded — `result` is what `tile_write` stores. So **`pa_*` has no effect on any output today**, and the pooled result is never clipped to `X_BITS`.

Fixing it (`result = quant_lrelu(result, ...)`) is a one-line change, but it **changes numbers for the existing qkeras path**: `run/example.py`'s bundle 1 has an avg pool, and adding the clip converts what is currently a silent `i8` truncation in `write_x` into saturation. That is almost certainly the intended behaviour, but it is a behaviour change to a shared file that **cannot be regression-tested here**.

### D2 — `pa_*` is never applied to max pooling at all

The `quant_lrelu` call above is the only use of `pa_*` in `runtime.h`, and it sits inside `if (pb->pool == POOL_AVG)`. A non-identity activation on a **max** pool is therefore silently ignored. The current max-pool support already pins the activation to identity (`ptq.py::_quantize_pool`), which is correct — but nothing stops a future caller from passing something else and quietly getting no activation. **Add an assert.**

### D3 — `next_add_ibs` is a `set` in the adapter, but the exporter indexes it

`adapter.py`'s `BrevitasBundle.__init__` sets `self.next_add_ibs = set()`, while `rtl_export.py:205` frees the buffer with `buf['out'][-1] == b.ib`. Sets are neither ordered nor indexable → `TypeError`. Already recorded in CLAUDE.md as unreachable; **residual makes it reachable and it is the first thing that will break.** It must become a `list`, and the ordering must be ascending bundle index, because `[-1]` is relied on to mean "the last consumer".

---

## Design — residual add

### The hard constraint: no frac alignment exists in hardware

`Bundle_t` carries `b_val_shift`/`b_bias_shift` for the bias and `aa_nzero`/`aa_shift`/`aa_pl_scale` for the add activation, but **no `add_val_shift`/`add_a_shift`**. The C does a raw

```c
out_val += mp->add_buffers[pb->add_in_buffer_idx][iy_nhwc];
```

with no shift on either operand. Legacy's `XAdd.call_int` computes alignment shifts, but nothing ever transports them to the firmware.

**Therefore both addends must already sit on the same fractional grid.** This is a constraint on the *model*, not something the exporter can paper over.

Two ways to satisfy it:
1. **Share the activation quantizer** between the skip source's activation and the consuming bundle's core activation, so brevitas calibrates them to one scale. Clean, and the natural brevitas idiom.
2. **Assert and let the user fix it.** `check_hardware` refuses when `act_frac[source] != act_frac[consumer]`, naming both.

Do **both**: option 1 as the documented way to build such a model, option 2 as the guard that stops a silently-misaligned one from reaching the RTL. The assert is the load-bearing half — a frac mismatch here produces plausible numbers that are wrong by a power of two.

### Topology: how the skip gets declared

`quantized_model` walks `net.named_children()` in order and has no notion of branches. Options considered:

- **`torch.fx` tracing** — the "right" answer, but `ptq.py` already documents that brevitas's quant proxies break dynamo tracing (`DataDependentOutputException` on `aten.allclose`), so this is a research task, not a plumbing one.
- **A marker module in the float net** — requires the user to restructure their model around our convention.
- **An explicit spec argument** — `quantized_model(net, residuals={'block2': 'block0'})`, mapping the consuming layer's attribute name to the source layer's. ← **recommended**

The explicit spec is the smallest change, is trivially testable, and keeps the float model a plain `nn.Module`. It can be replaced by fx tracing later without changing anything downstream, because everything after `quantized_model` consumes the JSON.

### JSON schema

Add `"skip_from": "<layer name>"` (or absent/`null`). CLAUDE.md's Known Issues already names this exact field, so the naming is settled. It sits alongside `"input"`, which stays the single main-path predecessor.

### Changes by file

**`ptq.py`**
- `quantized_model(..., residuals=None)`; build `XBundle(core=..., add_act=...)` for consumers named in the spec.
- `export_graph_json`: emit `"skip_from"`, plus the add activation's `aa_*`-equivalent fields (`add_activation`, `add_act_bits`, `add_act_frac`, `add_act_signed`).
- Assert the source bundle precedes the consumer and that their spatial shapes match.

**`sim.py`**
- `_build_topology`: store `skip_from` and the add-activation config.
- `forward`: after the core activation, `out = out + self.outputs[skip_from]`, then apply the add activation (same shift+clip path as the core one).
- Assert `act_frac` equality between the two addends at build time, with both names in the message.

**`adapter.py`**
- `next_add_ibs`/`next_ibs` → `list` (D3), appended in ascending `ib`.
- `_Add` stand-in: `source_ib`, `out`, and an `_Act`. It does **not** need `add_val_shift`/`add_a_shift` — nothing transports them — but set them to `0` explicitly with a comment, so a reader does not assume they were forgotten.
- `BrevitasBundle.add` populated from the new config.

**`export.py::check_hardware`**
- The frac-equality assert above.
- The post-add value must still fit `X_BITS` (`add_val_shift` widens by 1 bit by construction).

### Why residual is the safer of the two

It touches only brevitas-side Python. `runtime.h`, `dataflow.py` and the buffer allocator already implement it, and `run/example.py` exercises that C path today — so a mistake shows up as a bit mismatch in our own simulation, not as a silent change to the qkeras baseline.

---

## Design — average pooling

### The core problem: brevitas's avg pool is not the hardware's avg pool

The hardware computes, per window:

```c
result = div_round(sum_of_window, count)     // count = ACTUAL window size, edge-clipped
```

`div_round` is a rounding integer divide with a specific tie-break:

```c
#define div_round(a, b) (((a)+((b)/2) - (~((b)|(a)/(b)) &1))/(b))
```

`QuantAvgPool2d` in `xlayer/quantPooling.py` is brevitas's `TruncAvgPool2d`, which **sums and then truncates to a bit width** — a shift, not a divide-and-round. The two agree only when `count` is a power of two and the rounding happens to match. **Do not use `TruncAvgPool2d` for this.**

This is the same lesson the LUT work landed on: make the model reproduce the hardware exactly, rather than hoping a library's approximation is close enough. A model whose accuracy number does not describe what the hardware computes is worse than a slower one that does.

### Recommendation

1. **Support `padding='valid'` only, at first.** With no edge clipping `count` is constant (`PKH*PKW`), which removes the hardest part of matching `runtime.h` (its `count` shrinks at the borders, derived from `ph_beg`/`ph_end` sweeps). 'same' avg pooling can follow once the constant-count case is proven.
2. **Write a `QuantAvgPool2dDivRound`** in `xlayer/quantPooling.py` implementing sum → `div_round` in the quantized domain, replacing `TruncAvgPool2d` for this path. `sim.py` gets the matching integer implementation.
3. **Pin `div_round` against the C itself**, not against a Python re-reading of it — the LUT work did exactly this with a standalone C unit test for `quant_lut`, and it is the only way to be sure of the tie-break. A small C harness compiling `runtime.h`'s macro and dumping a table of `(a, b) -> div_round(a,b)` that the Python test asserts against.
4. **Fix D1** (`result = quant_lrelu(result, ...)`) and **flag the qkeras behaviour change loudly** in the commit and CLAUDE.md. Do not land this without either running `run/example.py` somewhere with TensorFlow, or getting explicit sign-off that the change is accepted unverified.
5. **Add the D2 assert** so a non-identity activation on max pooling fails instead of being ignored.

### Bit growth

Legacy widens the accumulator for avg pooling: `bits = x.bits + ceil(log2(PKH*PKW))` (`xlayers.py:342`), asserted against `hw.INT_BITS`. `check_hardware` needs the same check — a 3x3 avg pool over 8-bit activations needs 12 bits of headroom before the divide.

---

## Recommended order

1. **D3** (`next_add_ibs` → list) — one line, unblocks everything residual.
2. **Residual add** — Python only, C already works.
3. **D2 assert** — one line, prevents a silent trap.
4. **`div_round` C-vs-Python pin** — cheap, and de-risks step 5.
5. **Average pooling ('valid' only)** + **D1** — last, because it is the only step that changes shared C behaviour.

## Verification

Per step, matching how conv was brought up: a new stage in `conv.py` (`StageF` residual, `StageG` avg pool), driven by `conv_main.py --stage`, each ending in a **real RTL run**, not a unit test. The 2026-08-11 adapter work and the 2026-08-12 conv work both established that this class of defect only appears once `_export_bundles` is actually driven.

- Every bundle must report `Error: 0`; keep the final bundle free of softmax so the last comparison is a genuine integer one.
- `sim.py` vs brevitas must be **exactly** 0 differing values, as for conv.
- `pytest deepsocflow/test/py/ -q` stays green (142 at time of writing).
- Before landing step 5, `run/example.py` (7 bundles, conv/pool/residual) must pass **somewhere** — it is the only avg-pool regression that exists, and it cannot run on this machine.

## Open questions

- **Can the two residual addends be forced onto a shared scale by brevitas cleanly?** Sharing one `QuantReLU` instance between two bundles may confuse the single-quantization-point-per-bundle invariant the 2026-08-10 work established. Worth a spike before committing to option 1.
- **Does `run/example.py` currently depend on the D1 bug?** Its pool activation is `type=None` (identity), so the fix adds a clip where there is none today. Whether any of its values actually reach the clip is unknown without running it.
