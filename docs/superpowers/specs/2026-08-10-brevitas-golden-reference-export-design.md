# Golden-reference (`*_exp.txt`) export for the brevitas backend

## Goal

Given a calibrated `ptq.quantized_model` and its exported graph JSON, produce the
same golden-reference artefacts in `hw.DATA_DIR` that the legacy qkeras backend's
`deepsocflow/py/xmodel.py::export_inference` produces — same filenames, same
formats, same flatten order — so a future RTL step can consume them unchanged.
This is what makes the qkeras → brevitas transition smooth: RTL/testbench tooling
built against the legacy file conventions keeps working without modification.

## Non-goals

- No verilator/icarus/xsim run, no `verify_inference` equivalent, no `_sim.txt`
  diffing against RTL. Stays Python-only.
- No C/RTL testbench, no edits to `deepsocflow/c/runtime.h`.
- No edits to legacy qkeras files (`xmodel.py`, `xbundle.py`, `xlayers.py`,
  `utils.py`, `dataflow.py`, `hardware.py`).
- No non-piecewise-linear activations (SiLU/Tanh/GELU/SELU) — `sim.py` keeps
  raising `ValueError` for these, as documented in CLAUDE.md Known Issues.
- No conv/pooling/residual models this session. Dense-only (`type == "linear"`),
  matching the XOR example.
- No `config_fw.h` generation this session (needs buffer allocation and
  `ca_shift`/`ca_nzero`/etc. that don't exist in the brevitas path yet — staged
  as future work).

## Bugs found while designing this (fix first — everything downstream depends on them)

Comparing `sim.py`'s `FixedPointModel` against brevitas's own (fake-quantized)
forward pass bundle-by-bundle showed they only agree on `argmax`, not on the
actual int values:

| bundle | brevitas (ground truth) | `sim.py` today |
|---|---|---|
| 0 | `[[0,66,0],[0,0,0],[0,134,104],[0,69,0]]` | `[[0,66,0],[0,0,0],[0,135,105],[0,69,0]]` |
| 1 | `[[140,111,22],[0,0,107],[18,8,159],[144,115,20]]` | `[[255,223,0],[0,0,107],[36,16,214],[255,233,0]]` |
| 2 | `[[47,-93],[-64,58],[-84,69],[50,-98]]` | `[[127,-128],[-118,104],[-128,127],[127,-128]]` |

1. **Missing inter-bundle requantization.** `bundle0.act_frac=6` but
   `bundle1.input_frac=5` in the JSON — `ptq.py` gives every `QuantLinear` its own
   `input_quant`, creating a second quantization point after each activation
   (qkeras only has one). `forward()` feeds bundle output straight into the next
   bundle without re-shifting, so the accumulator ends up off by 2x.
2. **`quantize_input` doesn't clip.** `X=1.0` at `input_frac=7` gives `128`, but
   signed int8 tops out at `127` — brevitas clips, `sim.py` didn't.
3. **Unsigned ReLU output overflows signed 8-bit.** `ACT_MAP` uses
   `Uint8ActPerTensorFixedPoint` for ReLU, so values reach `159` — outside signed
   int8. The legacy backend explicitly avoids this (`xlayers.py:31-33`: ReLU is
   reduced to `bits-1` "because we have everything signed"). Packing this into a
   fixed-width signed buffer would silently wrap `159 → -97`.

Fixing (1) and (2) reproduces brevitas bit-exactly on all three bundles and the
pre-softmax floats. (3) is closed by Task 5 (single quantization point, ReLU
narrowed to `bits-1`) — decided below to do now, not defer.

## Decisions (locked in)

- **`y_exp.txt` batch size = 1**, matching the legacy dense convention
  (`run/param_test.py`'s `export_inference(..., batch_size=1)`). `export_inference`
  still takes a `batch_size` parameter so the full 4-row XOR truth table can be
  exported when wanted, but the default and the one used in `main.py`/tests is 1.
- **Task 5 (single quantization point per bundle) is done now**, not staged.
  `ptq.py::_quantize_layer` stops giving every `QuantLinear` its own
  `input_quant` (only the first bundle keeps one); preceding activations return
  `QuantTensor` so `Int32Bias` can still resolve `input_scale`; unsigned
  activations are narrowed to `bits-1` (mirrors `xlayers.py:32`). This makes
  bug (1)'s requant step a no-op and closes bug (3) structurally rather than
  papering over it with a runtime range assert.
  - **Escape hatch:** if brevitas fights `QuantTensor` propagation through
    `QuantReLU → QuantLinear` without `input_quant`, stop, keep the explicit
    inter-bundle requant from the bug fix above (it's still correct, just not
    structurally matching qkeras's single-quantization-point model), and record
    the gap as a new CLAUDE.md Known Issue instead of half-landing it.

## Architecture

| File | Status | Role | Legacy mirror |
|---|---|---|---|
| `deepsocflow/py/brevitas/sim.py` | extend | bit-exact int executor; fixes the two bugs above, records per-bundle intermediates + softmax | `xbundle.py::XBundle.call_int` + `xlayers.py`'s `call_int` methods |
| `deepsocflow/py/brevitas/export.py` | new | `export_inference(model, hw, x_float, ...)` — writes all `*_exp.txt`/`.bin` files into `hw.DATA_DIR` | `xmodel.py::export_inference` |
| `deepsocflow/py/brevitas/dataflow.py` | new, Phase 2 only | thin adapter supplying `get_runtime_params`/`reorder_*_q2e_conv`/`pack_words_into_bytes` to `export.py` via a dense-shaped `core` shim | `deepsocflow/py/dataflow.py` |
| `deepsocflow/py/brevitas/ptq.py` | extend | add `signed` per tensor to the JSON; single-quantization-point mode (Task 5) | `xlayers.py` quantizer choices |
| `deepsocflow/py/brevitas/main.py` | extend | driver: build → load weights → forward → `export_inference` | `run/param_test.py` |

### `export.py` public surface

```
export_inference(model, hw, x_float, data_dir=None, clean=True, batch_size=1) -> dict
    model:      FixedPointModel (topology built, load_int_weights() already called)
    hw:         deepsocflow.py.brevitas.hardware.Hardware
    x_float:    np.ndarray / torch.Tensor
    batch_size: rows of x_float to export (default 1, matches legacy dense convention)
    returns:    {'files': [...], 'y_exp': ndarray, 'softmax_frac': int, 'softmax_max_i': int}

check_hardware(model, hw)            # bit-width + range asserts, raises AssertionError
softmax_from_int(logits_int, frac)   # mirrors xbundle.py:98-110 (2**17 factor), per-row normalized
```

Private formatting helpers so the file format lives in exactly one place:
`_savetxt_int`, `_savetxt_float`, `_to_nhwc`.

### Filenames (Phase 1 — layout-independent, this session's scope)

| File | fmt | Content | Legacy line |
|---|---|---|---|
| `y_exp.txt` | `%f` (last bundle has softmax) | final model output, batch_size=1, flattened | `xmodel.py:326-327` |
| `{ib}_y_nhwc_exp.txt` | `%d` | bundle `ib` output as `(1,XN,1,CO)`, C-order flatten | `xmodel.py:312` |

`export_inference` cleans `DATA_DIR` first (mirrors `xmodel.py:71-73`), gated by
`clean=True`, and prints the first/last 20 `y_exp` values like `xmodel.py:328-330`.

### Filenames (Phase 2 — engine layout, only if time allows this session)

| File | fmt | Content | Legacy line |
|---|---|---|---|
| `{ib}_xe.txt` | `%d` | `concat([xe[ip].flatten() for ip])` | `xmodel.py:313` |
| `{ib}_{ip}_x.txt` | `%d` | `xe[ip].flatten()` | `xmodel.py:318` |
| `{ib}_{ip}_{it}_w.txt` | `%d` | `we[ip][it].flatten()` | `xmodel.py:323` |
| `{ib}_{ip}_{it}_y_exp.txt` | `%d` | per-pass raw conv-sum (no bias), engine layout | `xmodel.py:324` |
| `x.bin`, `wb.bin`, `wbx.bin`, `x_all.bin`, `{ib}_x_sim.bin` | binary | packed words | `xmodel.py:290-304` |

**Reuse, don't port:** Phase 2 imports the legacy `deepsocflow/py/dataflow.py`
behind a brevitas-side adapter rather than transcribing the reorder functions.
They encode non-obvious hardware invariants (axis flips, column-interchange
tricks, `CONFIG_BEATS` padding) that a re-transcription could get subtly wrong
in a way only an RTL diff would catch — and RTL diffing is explicitly out of
scope this session. Importing adds no new coupling: `deepsocflow/__init__.py`
already unconditionally pulls in TF/qkeras for any `deepsocflow.py.brevitas.*`
import today.

### `shift_round` — single source of truth

For the brevitas backend, `deepsocflow/py/brevitas/sim.py::shift_round` is
canonical; `export.py` imports it from there. A parity test pins it against
`deepsocflow.py.utils.shift_round` over a sweep of `(n, s)` so the two Python
copies (plus `c/runtime.h`'s macro) can't silently diverge. Note: they differ in
`s<=0` handling (legacy: `>> s` with `half_b=0`; `sim.py`: `<< -s`) — the test
pins the overlap `s >= 1` and documents the difference rather than forcing one
to match the other's edge case.

## Data flow

```
xor.py: XOR() ──train()/load()──► float nn.Module (model/xor.pt)
                                        │
                    ptq.quantized_model(net, layer_bits=...)   [single quant point
                                        │                       per bundle - Task 5]
                            .quantization(X)                    [brevitas calibration_mode]
                                        │
                    ├── .export(X, xor.onnx)            → QONNX (unchanged, informational)
                    └── .export_graph_json(X, xor_graph.json)   [+ signed field]
                                        │
        FixedPointModel(json)  ._build_topology()   [shapes/topology/fracs, no values]
              .load_int_weights(json)               [int weight/bias arrays]
              .quantize_input(x_float)               [now clips to input_bits]
              .forward(x_int)                        [requant now a no-op post-Task-8;
                                                        records trace + softmax]
                                        │
                    export.export_inference(model, hw, x_float, batch_size=1)
                                        │
        hw.DATA_DIR/  y_exp.txt, {ib}_y_nhwc_exp.txt
```

### Per-bundle trace capture in `FixedPointModel.forward`

`forward` already builds `outputs = {}` internally. Promote it and add per-bundle:

- `self.trace[name] = {'x': requantized input, 'y': matmul-only (no bias),
  'acc': y + bias, 'out': post-activation}`
- `y` is the bias-free conv-sum — legacy's `self.core.y`, the basis for
  `{ib}_{ip}_{it}_y_exp.txt` in Phase 2.
- `out` is `oe_exp_nhwc` / `{ib}_y_nhwc_exp.txt`, and for the last bundle also
  feeds `y_exp.txt`.

Also add, mirroring `xbundle.py:98-110`: `self.pre_softmax` (int),
`self.softmax_frac` (last bundle's `act_frac`), `self.softmax_max_i`
(`int(max_float * 2**17)`), `self.softmax_out` (float32, **normalized per row**
via `axis=-1, keepdims=True` — legacy's `np.sum(exp, axis=1)[0]` divides every
row by row 0's sum, which is only correct for `batch_size==1`; since this
design's default is `batch_size=1` this deviation is inert for `y_exp.txt`
itself, but the fix keeps `FixedPointModel` correct for any batch size).

## Implementation tasks (ordered, each independently testable)

1. **Fix `quantize_input` clipping** (`sim.py:93-97`) — clip to the first
   bundle's `input_bits` after `np.rint`. *Test:* `X=1.0` at `frac=7, bits=8` →
   `127`, not `128`.
2. **Fix inter-bundle requantization** in `forward()` — track `prev_frac`;
   before each bundle's matmul, if `prev_frac != bundle['input_frac']`,
   `shift_round` and clip. Note in the docstring this becomes a no-op once Task
   8 lands (kept anyway for robustness / older JSONs). *Test:* per-bundle int
   levels equal the brevitas table above, exactly.
3. **Capture per-bundle trace + softmax** in `sim.py` — `self.trace`,
   `self.outputs`, `self.pre_softmax`, `self.softmax_frac`,
   `self.softmax_max_i`, `self.softmax_out`. Handle bias-less layers (zeros) and
   raise a clear `RuntimeError` (not a cryptic `TypeError`) if `forward()` is
   called before `load_int_weights()`.
4. **Add `signed` to the graph JSON** (`ptq.py::export_graph_json`) — emit
   `input_signed`, `weight.signed`, `bias.signed`, `act_signed`, sourced from
   the quantizer proxy or the injector class name (`Uint*` → unsigned). Update
   `sim.py` to read `act_signed`, keeping `UNSIGNED_ACTIVATIONS` only as a
   fallback for older JSONs. Closes the "no signed/unsigned flag" Known Issue.
5. **Single quantization point per bundle (Task 5)** — `ptq.py::_quantize_layer`:
   `input_quant` only on the first bundle; preceding activations
   `return_quant_tensor=True`; unsigned activations narrowed to `bits-1`
   (mirrors `xlayers.py:32`). Re-calibrate, re-export JSON, re-run Tasks 1-3's
   tests. If this fights brevitas's `QuantTensor` propagation, invoke the escape
   hatch above and document it instead of forcing it.
6. **New `deepsocflow/py/brevitas/export.py`, Phase 1** — `check_hardware`,
   `_savetxt_int`/`_savetxt_float`, `_to_nhwc`, `softmax_from_int`,
   `export_inference` writing `{ib}_y_nhwc_exp.txt` per bundle and `y_exp.txt`
   for the model (batch_size=1). Clean `DATA_DIR` first.
7. **`check_hardware` asserts** — mirroring `xmodel.py:55-57` /
   `xbundle.py:148-161`: bit-width matches per tensor, `ACC_WIDTH <= Y_BITS`,
   and (post-Task-8) every activation value fits signed `X_BITS`.
8. **Wire `main.py`** — construct an XOR-appropriate `Hardware`
   (`bits_sum=32` — required, since `ACC_WIDTH = 8+8+clog2(512) = 25` and the
   default `bits_sum=16` would trip the assert), then
   `FixedPointModel → load_int_weights → export_inference(batch_size=1)`.
9. **(Phase 2, optional this session)** — `deepsocflow/py/brevitas/dataflow.py`
   adapter + engine-layout files. Weight transpose note: brevitas stores
   `(out_features, in_features)`, legacy's `reorder_w_q2e_conv` wants
   `(KH,KW,CI,CO)` = `w.T.reshape(1,1,CI,CO)`.
10. **Update `CLAUDE.md`** — Progress Log entry; close the signed/unsigned
    Known Issue; if Task 5's escape hatch was used, add a new Known Issue for
    the deferred single-quantization-point structural difference.

## Test plan

New tests **must** be named `test_brevitas_*.py` — `.gitignore` only un-ignores
`deepsocflow/test/py/test_brevitas_*.py` and `test_suggest_weight_map.py`; any
other name is silently untracked.

### `deepsocflow/test/py/test_brevitas_sim.py` (unit, fast, no training needed)

Fixture: a hand-written 2-bundle graph JSON dict written to `tmp_path`.

- `test_shift_round_matches_legacy` — sweep vs `deepsocflow.py.utils.shift_round`
  for `s >= 1` (`pytest.importorskip('tensorflow')`).
- `test_shift_round_rounds_half_to_even`
- `test_quantize_input_clips_to_input_bits`
- `test_requantize_is_noop_when_fracs_match` — guards Task 5's end state.
- `test_forward_records_trace_per_bundle` — trace keys match bundle order;
  `acc == y + bias`.
- `test_forward_before_load_int_weights_raises` — `RuntimeError`, not `TypeError`.
- `test_unsupported_activation_raises` / `test_unsupported_layer_type_raises`
- `test_bias_less_layer_defaults_to_zeros`

### `deepsocflow/test/py/test_brevitas_export_inference.py` (end-to-end)

Load the checked-in `deepsocflow/py/brevitas/model/xor.pt` rather than training
in the test (training is slow and only float-32-cross-platform-deterministic
under `torch.manual_seed(0)`, not something to depend on in CI) — `pytest.skip`
with a clear message if it's absent. **Regenerate the graph JSON into `tmp_path`
every run** — never read the repo copy, or the test could pass against a stale
export. `DATA_DIR` is always `tmp_path / 'vectors'`.

- `test_fixed_point_model_matches_brevitas_bit_exactly` — the core assertion:
  compare `FixedPointModel.trace[name]['out']` against brevitas's own per-bundle
  int values with `np.array_equal`, not just `argmax`. Also compares
  pre-softmax floats.
- `test_predictions_match_brevitas` — `argmax` parity, all four XOR rows,
  `== [0,1,1,0]`.
- `test_y_exp_txt_matches_model_output` — `np.loadtxt(y_exp.txt)` vs
  `model.softmax_out.flatten()`, `atol=1e-6`.
- `test_y_nhwc_exp_txt_matches_bundle_outputs`
- `test_y_exp_is_float_format_when_softmax` / `test_y_nhwc_exp_is_int_format`
- `test_flatten_order_is_c_order`
- `test_export_cleans_data_dir`
- `test_hardware_bitwidth_mismatch_raises`
- `test_activation_out_of_signed_range_raises` — post-Task-5 this must pass with
  real XOR values (pre-Task-5 it would have failed, which is why Task 5 is done
  now rather than deferred).
- `test_missing_weight_values_raises`

### Phase 2 tests (only if Task 9 lands)

`test_brevitas_export_engine_layout.py`: file cross-product matches `r.CP × r.IT`,
`w` file shape assert (mirrors `xmodel.py:322`), and a roundtrip test using the
legacy inverse reorder (`reorder_y_e2q_conv`) as a free oracle — catches adapter
misuse without needing RTL.

### Running

```
cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_*.py -v
```

## Risks / open questions carried forward

1. Task 5 may not be a one-liner if brevitas's `QuantTensor` propagation through
   `QuantReLU → QuantLinear` without `input_quant` doesn't work cleanly — escape
   hatch documented above.
2. `RAM_WEIGHTS_DEPTH` drives `ACC_WIDTH` through `r.CM=512` (not the actual
   `CI=2`) — legacy behaviour, mirrored deliberately; forces `bits_sum >= 25`
   for XOR. Not a bug, just worth knowing before it looks like one.
3. `deepsocflow/py/brevitas/model/` is currently untracked in git; confirm
   `xor.pt` should be committed so `test_brevitas_export_inference.py`'s
   skip-if-absent fixture has something to find in CI.
