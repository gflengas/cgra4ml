# XBundle port to Brevitas backend (call() scope)

## Context

`deepsocflow/py/xbundle.py` (Keras/QKeras) defines `XBundle`, a layer that
bundles a core (conv/dense), optional pool, optional residual add, optional
flatten, and optional softmax into one hardware-mappable unit. It exposes
three methods: `call` (float/QAT forward), `call_int` (fixed-point
simulation against `hw` config, asserted to match `call`), and `export`
(reorders integer tensors into hardware engine layout).

`deepsocflow/py/brevitas/xbundle/xbundle.py` is currently an empty stub.
`deepsocflow/py/brevitas/xlayer/` has thin wrappers around Brevitas layers
(`QuantConv2d`, `QuantLinear`, `QuantReLU`, `QuantSigmoid`, `QuantTanh`,
pooling variants) but no `XTensor`, `BUNDLES`, or hardware/dataflow port.
The project is targeting PTQ (post-training quantization) with Brevitas,
not QAT — this doesn't change the shape of `call()`, since Brevitas layers
forward the same way regardless of how they were calibrated/trained.

## Goal

Port `XBundle.call()` to a working PyTorch/Brevitas equivalent. Add
`call_int` and `export` as stubs with the original signatures, so the
class shape matches the Keras original and future work has a clear slot.

## Design

### `XBundle(nn.Module)` — `deepsocflow/py/brevitas/xbundle/xbundle.py`

Constructor mirrors the original: `core, pool=None, add_act=None,
flatten=False, softmax=False`.

- `core` / `pool`: expected to already carry a `.act` attribute (same
  convention as the Keras original, where the model-builder attaches
  activation to the core/pool layer). Attaching `.act` to Brevitas layers
  is out of scope here — this task only consumes the convention.
- `add_act` → builds `self.add = QuantResidualAdd(act=add_act)` (new,
  see below) if provided, else `self.add = None`.
- `flatten` → `nn.Flatten()` if True, else `None`.
- `softmax` → `nn.Softmax(dim=-1)` if True, else `None`. Kept as plain
  float softmax (not a Brevitas quant layer) to match the original, where
  the final softmax runs on dequantized values, not as part of the
  hardware-quantized path.
- Bundle-graph bookkeeping fields, identical to the original: `ib`,
  `prev_ib`, `next_ibs`, `next_add_ibs`.

### `call(self, x, x_add=None)`

Mirrors the original structure:
1. Register self in `BUNDLES` (`self.ib = len(BUNDLES); BUNDLES.append(self)`).
2. If `x` carries an `.ib` attribute (set by a previous bundle), record
   `self.prev_ib` and append `self.ib` to that bundle's `next_ibs`.
   (Verified: plain `torch.Tensor` instances allow arbitrary attribute
   assignment, e.g. `t.ib = 5`, so this works without a custom tensor type.)
3. `x = self.core(x); x = self.core.act(x)`.
4. If `x_add is not None`: require `self.add is not None` (else raise, same
   as original), record `self.add.source_ib = x_add.ib`, append to that
   bundle's `next_add_ibs`, then `x = self.add(x, x_add)`.
5. If `self.pool`: `x = self.pool(x); x = self.pool.act(x)`.
6. If `self.flatten`: `x = self.flatten(x)`.
7. If `self.softmax`: `x = self.softmax(x)`.
8. Tag `x.ib = self.ib` and return `x`.

`forward = call` is set so the module also works via normal PyTorch
call syntax (`bundle(x)`), matching PyTorch convention while keeping the
`call` name for consistency with the rest of the codebase (`xlayers.py`
core layers use `call`/`call_int` too).

### `call_int(self, x, hw)` and `export(self, hw, is_last)`

Stubs with the original signatures, each raising `NotImplementedError`
with a short message naming what's missing (an `XTensor` port and a
`hardware.py`/`dataflow.py` port for the brevitas backend). A one-line
comment on each states its eventual responsibility (fixed-point
simulation + parity check against `call()`; hardware-engine-layout
tensor export), so the next port step has a clear slot to fill.

### Supporting additions (blocking `call()`, otherwise out of scope)

- **`deepsocflow/py/brevitas/utils.py`** (new): `BUNDLES = []` — mirrors
  `deepsocflow/py/utils.py`, a module-level list every `XBundle` instance
  registers itself into.
- **`QuantResidualAdd`** in `deepsocflow/py/brevitas/xlayer/quantOperation.py`
  (currently just a comment placeholder for "ResidualAdd"): a minimal
  `nn.Module` wrapping an activation module, mirroring `XAdd.call` — add
  two tensors elementwise, then apply the activation. `call_int` is not
  implemented (raises `NotImplementedError`) since it depends on
  `XTensor.add_val_shift`, which doesn't exist yet.

## Out of scope

- `call_int` / `export` real implementations (need `XTensor`,
  `hardware.py`, `dataflow.py` ports).
- Attaching `.act` onto `core`/`pool` layers (model-builder concern).
- `QuantResidualAdd.call_int`.

## Testing

Manual smoke test: construct an `XBundle` around a `QuantConv2d` (+
`QuantReLU` as `.act`) and run `call()` on a random tensor, confirm shape
and that `BUNDLES` records the bundle. No existing automated test suite
covers the brevitas backend yet, so this stays a manual check.
