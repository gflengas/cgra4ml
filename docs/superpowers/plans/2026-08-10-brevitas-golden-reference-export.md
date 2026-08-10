# Brevitas Golden-Reference Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the brevitas backend produce the same golden-reference `*_exp.txt` files the legacy qkeras backend produces (`deepsocflow/py/xmodel.py::export_inference`), so a future RTL step can consume them unchanged, while fixing the two real bugs that currently make `sim.py`'s integer forward pass diverge from brevitas's own output.

**Architecture:** Fix `FixedPointModel` (`deepsocflow/py/brevitas/sim.py`) so it's bit-exact against brevitas, teach `ptq.py` to emit a `signed` flag per tensor and to use a single quantization point per bundle (matching qkeras's structure instead of re-quantizing after every activation), then add a new `export.py` that walks a `FixedPointModel`'s per-bundle trace and writes `y_exp.txt` / `{ib}_y_nhwc_exp.txt` in the exact format/layout the legacy backend uses.

**Tech Stack:** Python, PyTorch + brevitas (quantization/calibration only), numpy (all integer execution), pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-brevitas-golden-reference-export-design.md`

## Global Constraints

- Only work on the `brevitas-qonnx-backend` branch (already checked out).
- **Do not run `git commit` or `git push` at any point in this plan** — the user
  handles all commits/pushes themselves (per `CLAUDE.md` Working Agreement).
  Where the writing-plans skill's default task structure would normally end in
  a commit step, this plan ends each task at "run tests, confirm green"
  instead — leave the working tree as-is for the user to review and commit.
- New test files must be named `deepsocflow/test/py/test_brevitas_*.py` exactly
  — `.gitignore` only un-ignores that pattern (and `test_suggest_weight_map.py`);
  any other name is silently untracked.
- Stays Python-only: no verilator/RTL run, no edits to `deepsocflow/c/runtime.h`
  or any legacy qkeras file (`xmodel.py`, `xbundle.py`, `xlayers.py`,
  `utils.py`, `dataflow.py`, `hardware.py` under `deepsocflow/py/` — not the
  `deepsocflow/py/brevitas/` versions, which are fair game).
- Dense-only (`type == "linear"`) — no conv/pooling/residual support added.
- No non-piecewise-linear activations — `sim.py` keeps raising `ValueError` for
  `silu`/`tanh`/etc.
- Engine-layout files (`{ib}_{ip}_{it}_y_exp.txt`, `.bin` blobs, the
  `dataflow.py` adapter) are explicitly **out of scope for this plan** — see
  the spec's Phase 2. If picked up later, it gets its own plan.

---

## Task 1: Test scaffolding + `shift_round` parity tests

Sets up the fixture helpers every later test in this file reuses, and locks
down that `sim.py`'s `shift_round` already agrees with the legacy
`deepsocflow.py.utils.shift_round` — a regression guard, not new behavior.

**Files:**
- Create: `deepsocflow/test/py/test_brevitas_sim.py`

**Interfaces:**
- Consumes: `deepsocflow.py.brevitas.sim.shift_round(n, s)` (exists today,
  unchanged), `deepsocflow.py.utils.shift_round(n, s)` (legacy, unchanged).
- Produces: `_bundle(...)` and `_write_graph(...)` fixture helpers, reused by
  every later task's tests in this file.

- [ ] **Step 1: Write the fixture helpers and the two `shift_round` tests**

```python
import json

import numpy as np
import pytest

from deepsocflow.py.brevitas import sim


def _bundle(input_frac, input_bits, weight_values, weight_frac, weight_bits,
            activation, act_bits, act_frac, bias_values=None, bias_frac=None,
            bias_bits=None, softmax=False, input_name=None, type_="linear"):
    """Builds one entry of a graph JSON's "layers" dict, matching the schema
    quantized_model.export_graph_json() produces (deepsocflow/py/brevitas/ptq.py).
    Only includes the fields sim.py actually reads."""
    in_features = len(weight_values[0])
    out_features = len(weight_values)
    layer = {
        "type": type_,
        "input": input_name,
        "input_bits": input_bits,
        "input_frac": input_frac,
        "in_features": in_features,
        "out_features": out_features,
        "weight": {"bits": weight_bits, "frac": weight_frac, "values": weight_values},
        "activation": activation,
        "act_bits": act_bits,
        "act_frac": act_frac,
        "softmax": softmax,
    }
    if bias_values is not None:
        layer["bias"] = {"bits": bias_bits, "frac": bias_frac, "values": bias_values}
    return layer


def _write_graph(tmp_path, layers, name="graph.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"layers": layers}))
    return str(path)


def test_shift_round_matches_legacy():
    pytest.importorskip("tensorflow")
    from deepsocflow.py.utils import shift_round as legacy_shift_round

    rng = np.random.default_rng(0)
    n = rng.integers(-100000, 100000, size=2000, dtype=np.int64)
    for s in range(1, 13):
        ours = sim.shift_round(n, s)
        legacy = np.asarray(legacy_shift_round(n, s), dtype=np.int64)
        assert np.array_equal(ours, legacy), f"mismatch at s={s}"


def test_shift_round_rounds_half_to_even():
    n = np.array([-10, -6, -2, 2, 6, 10], dtype=np.int64)
    # dividing by 4 (s=2): exact-half results round to the nearest EVEN value
    assert sim.shift_round(n, 2).tolist() == [-2, -2, 0, 0, 2, 2]
```

- [ ] **Step 2: Run the tests**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py -v`
Expected: both tests PASS (they exercise existing, already-correct code — this
task is a regression guard, not new behavior, so there's no red phase here).

---

## Task 2: Fix `quantize_input` clipping

`sim.py`'s `quantize_input` currently rounds a float input to its int level
but never clips it to what `input_bits` can represent, so `X=1.0` at
`input_frac=7` (signed int8) produces `128` — one past the signed 8-bit max of
`127` — while brevitas's own input quantizer clips. This task fixes it and
adds `input_bits` to the topology dict so later tasks can reuse it.

**Files:**
- Modify: `deepsocflow/py/brevitas/sim.py`
- Test: `deepsocflow/test/py/test_brevitas_sim.py`

**Interfaces:**
- Consumes: `_bundle`/`_write_graph` from Task 1.
- Produces: `FixedPointModel.bundles[name]['input_bits']` (new key, read by
  Task 3's requant fix).

- [ ] **Step 1: Write the failing test**

Append to `deepsocflow/test/py/test_brevitas_sim.py`:

```python
from deepsocflow.py.brevitas.sim import FixedPointModel


def test_quantize_input_clips_to_input_bits(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)

    x_int = model.quantize_input([[1.0, 0.0]])

    # 1.0 * 2**7 = 128, but signed int8 tops out at 127 - must clip, not wrap.
    assert x_int.tolist() == [[127, 0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py::test_quantize_input_clips_to_input_bits -v`
Expected: FAIL — `assert [[128, 0]] == [[127, 0]]` (or a `KeyError: 'input_bits'`
if `_build_topology` doesn't store it yet).

- [ ] **Step 3: Fix `sim.py`**

In `deepsocflow/py/brevitas/sim.py`, in `_build_topology`, add `input_bits` to
the stored dict:

```python
            self.bundles[name] = dict(
                input=cfg.get('input'),
                in_features=cfg['in_features'],
                out_features=cfg['out_features'],
                input_bits=cfg['input_bits'],
                input_frac=cfg['input_frac'],
                weight_frac=cfg['weight']['frac'],
                bias_frac=cfg['bias']['frac'],
                activation=cfg['activation'],
                act_bits=cfg['act_bits'],
                act_frac=cfg['act_frac'],
                softmax=cfg['softmax'],
                weight=None,  # populated by load_int_weights()
                bias=None,
            )
```

Replace `quantize_input`:

```python
    def quantize_input(self, x_float):
        """Quantizes a real-valued input using the first bundle's input scale,
        clipping to what its input_bits can represent (signed) - matches
        brevitas's own input quantizer, which clips out-of-range values
        instead of wrapping."""
        first = self.bundles[self.bundle_order[0]]
        x_int = np.rint(np.asarray(x_float, dtype=np.float64) * 2 ** first['input_frac'])
        bits = first['input_bits']
        x_int = np.clip(x_int, -2 ** (bits - 1), 2 ** (bits - 1) - 1)
        return x_int.astype(np.int64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py -v`
Expected: all tests PASS.

---

## Task 3: Fix inter-bundle requantization

`ptq.py` currently gives every `QuantLinear` its own `input_quant`, so each
bundle's `input_frac` can differ from the previous bundle's `act_frac` (e.g.
real `xor_graph.json`: `bundle0.act_frac=6` but `bundle1.input_frac=5`).
`forward()` feeds one bundle's output straight into the next without
re-shifting, so the accumulator ends up scaled wrong. This task requantizes
between bundles whenever the fracs don't match; it becomes a no-op once Task 6
makes every bundle share one quantization point, but stays in place for
robustness (and for older JSONs).

**Files:**
- Modify: `deepsocflow/py/brevitas/sim.py`
- Test: `deepsocflow/test/py/test_brevitas_sim.py`

**Interfaces:**
- Consumes: `bundle['input_bits']` from Task 2.
- Produces: `forward()`'s behavior is now correct across a frac mismatch;
  no new public interface.

- [ ] **Step 1: Write the failing test**

Append to `deepsocflow/test/py/test_brevitas_sim.py`:

```python
def test_requantizes_between_bundles_when_input_frac_differs_from_prev_act_frac(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6),
        "bundle1": _bundle(input_frac=5, input_bits=8,
                            weight_values=[[32]], weight_frac=5, weight_bits=8,
                            bias_values=[0], bias_frac=10, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=5,
                            input_name="bundle0"),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    # bundle0: x=[0.5,0.5] (int [64,64] @ frac=7), weight=[1.0,1.0] (int [64,64] @ frac=6)
    #   -> acc = 64*64 + 64*64 = 8192 @ frac=13 -> shift_round(8192, 7) = 64 @ frac=6 (=1.0)
    # bundle1 expects its input at frac=5, but bundle0's output is at frac=6 - must
    # requantize: shift_round(64, 6-5=1) = 32 @ frac=5 (still =1.0) before the matmul.
    out = model.forward(np.array([[64, 64]], dtype=np.int64))

    assert out.tolist() == [[32]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py::test_requantizes_between_bundles_when_input_frac_differs_from_prev_act_frac -v`
Expected: FAIL — `assert [[64]] == [[32]]` (bundle1 currently treats bundle0's
raw int output as if it were already at its own `input_frac`).

- [ ] **Step 3: Fix `forward()` in `sim.py`**

Replace `forward`:

```python
    def forward(self, x_int):
        outputs = {}
        prev_frac = {}  # bundle name -> the frac its output is actually stored at
        x_int = np.asarray(x_int, dtype=np.int64)

        for name in self.bundle_order:
            bundle = self.bundles[name]

            if bundle['input'] is None:
                inp = x_int
                inp_frac = self.bundles[self.bundle_order[0]]['input_frac']
            else:
                inp = outputs[bundle['input']]
                inp_frac = prev_frac[bundle['input']]

            # Requantize if the producing bundle's output frac doesn't match what
            # this bundle's input_quant expects - ptq.py currently gives every
            # QuantLinear its own input_quant (a second quantization point after
            # each activation), so these can genuinely differ. Becomes a no-op
            # once every bundle shares a single quantization point (see ptq.py's
            # own_input_quant flag).
            if inp_frac != bundle['input_frac']:
                inp = shift_round(inp, inp_frac - bundle['input_frac'])
                inp = np.clip(inp, -2 ** (bundle['input_bits'] - 1), 2 ** (bundle['input_bits'] - 1) - 1)

            acc_frac = bundle['input_frac'] + bundle['weight_frac']
            assert acc_frac == bundle['bias_frac'], (
                f"bundle '{name}': accumulator frac {acc_frac} != bias frac {bundle['bias_frac']}")

            acc = inp @ bundle['weight'].T + bundle['bias']  # int64, frac = acc_frac

            if bundle['activation'] == 'relu':
                acc = np.clip(acc, 0, None)

            out = shift_round(acc, acc_frac - bundle['act_frac'])

            if bundle['activation'] in UNSIGNED_ACTIVATIONS:
                out = np.clip(out, 0, 2 ** bundle['act_bits'] - 1)
            else:
                out = np.clip(out, -2 ** (bundle['act_bits'] - 1), 2 ** (bundle['act_bits'] - 1) - 1)

            outputs[name] = out
            prev_frac[name] = bundle['act_frac']

        return outputs[self.bundle_order[-1]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py -v`
Expected: all tests PASS, including the existing `xor.py`-based manual check —
also re-run `python -m deepsocflow.py.brevitas.main` and confirm predictions
are still `[0, 1, 1, 0]`.

---

## Task 4: Capture per-bundle trace, softmax output, and error handling

Adds `self.trace` (per-bundle `x`/`y`/`acc`/`out`), softmax handling
(`self.pre_softmax`/`softmax_frac`/`softmax_max_i`/`softmax_out`, mirroring
`xbundle.py:98-110`'s `2**17`-factor fixed-point softmax), a clear error when
`forward()` is called before weights are loaded, and defaults bias-less layers
to zero (today's code assumes every layer's JSON entry has a `"bias"` key,
which `export_graph_json` only emits `if quant_bias is not None` — untested
because XOR always has biases).

**Files:**
- Modify: `deepsocflow/py/brevitas/sim.py`
- Test: `deepsocflow/test/py/test_brevitas_sim.py`

**Interfaces:**
- Consumes: Tasks 2-3's `forward()`.
- Produces: `FixedPointModel.trace` (dict: bundle name -> `{'x','y','acc','out'}`,
  each an `np.ndarray[int64]`), `.pre_softmax` (`np.ndarray[int64]` or `None`),
  `.softmax_frac` (`int` or `None`), `.softmax_max_i` (`np.ndarray[int64]` or
  `None`), `.softmax_out` (`np.ndarray[float32]` or `None`) — all read by
  Task 7's `export.py`.

- [ ] **Step 1: Write the failing tests**

Append to `deepsocflow/test/py/test_brevitas_sim.py`:

```python
def test_forward_records_trace_per_bundle(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6),
        "bundle1": _bundle(input_frac=5, input_bits=8,
                            weight_values=[[32]], weight_frac=5, weight_bits=8,
                            bias_values=[0], bias_frac=10, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=5,
                            input_name="bundle0"),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    model.forward(np.array([[64, 64]], dtype=np.int64))

    assert set(model.trace.keys()) == {"bundle0", "bundle1"}
    for name in model.trace:
        t = model.trace[name]
        assert np.array_equal(t["acc"], t["y"] + model.bundles[name]["bias"])


def test_forward_before_load_int_weights_raises(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)  # load_int_weights() deliberately NOT called

    with pytest.raises(RuntimeError, match="load_int_weights"):
        model.forward(np.array([[64, 64]], dtype=np.int64))


def test_bias_less_layer_defaults_to_zeros(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            activation="identity", act_bits=8, act_frac=6),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    out = model.forward(np.array([[64, 64]], dtype=np.int64))

    # y = 64*64 + 64*64 = 8192 @ frac=13, bias=0 -> acc=8192
    # shift_round(8192, 13-6=7) = 64 @ frac=6 (=1.0)
    assert out.tolist() == [[64]]
    assert model.trace["bundle0"]["acc"].tolist() == [[8192]]


def test_softmax_output_matches_manual_softmax(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 0], [0, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0, 0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6, softmax=True),
    }
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    logits_int = model.forward(np.array([[64, 0]], dtype=np.int64))

    # weight is the identity matrix (scaled by 2**6), so bundle0's logits are
    # exactly the (rescaled) input: x=[64,0] @ frac=7 (=[0.5,0.0]) times identity
    # -> acc=[4096,0] @ frac=13 -> shift_round(.,7) -> int [32,0] @ act_frac=6
    assert logits_int.tolist() == [[32, 0]]
    assert model.pre_softmax.tolist() == logits_int.tolist()
    assert model.softmax_frac == 6

    logits_float = np.array([32, 0]) / 2 ** 6  # == [0.5, 0.0]
    expected = np.exp(logits_float) / np.exp(logits_float).sum()
    assert model.softmax_out[0].tolist() == pytest.approx(expected.tolist(), abs=1e-6)


def test_unsupported_activation_raises(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="silu", act_bits=8, act_frac=6),
    }
    json_path = _write_graph(tmp_path, layers)
    with pytest.raises(ValueError, match="silu"):
        FixedPointModel(json_path)


def test_unsupported_layer_type_raises(tmp_path):
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[64, 64]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="identity", act_bits=8, act_frac=6, type_="conv"),
    }
    json_path = _write_graph(tmp_path, layers)
    with pytest.raises(ValueError, match="conv"):
        FixedPointModel(json_path)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py -v`
Expected: `test_forward_records_trace_per_bundle` FAILS with `AttributeError:
'FixedPointModel' object has no attribute 'trace'`;
`test_forward_before_load_int_weights_raises` FAILS (raises `TypeError`
from `None @ ...`, not `RuntimeError`); `test_bias_less_layer_defaults_to_zeros`
FAILS with `KeyError: 'bias'`; `test_softmax_output_matches_manual_softmax`
FAILS with `AttributeError: 'FixedPointModel' object has no attribute
'pre_softmax'`. `test_unsupported_activation_raises` and
`test_unsupported_layer_type_raises` already PASS (existing behavior, just
newly covered).

- [ ] **Step 3: Implement in `sim.py`**

In `_build_topology`, handle a missing `"bias"` key (default its frac to the
accumulator frac, so the `acc_frac == bias_frac` assert in `forward` stays
valid with an all-zero bias):

```python
            bias_cfg = cfg.get('bias')
            bias_frac = bias_cfg['frac'] if bias_cfg is not None else cfg['input_frac'] + cfg['weight']['frac']

            self.bundles[name] = dict(
                input=cfg.get('input'),
                in_features=cfg['in_features'],
                out_features=cfg['out_features'],
                input_bits=cfg['input_bits'],
                input_frac=cfg['input_frac'],
                weight_frac=cfg['weight']['frac'],
                bias_frac=bias_frac,
                activation=cfg['activation'],
                act_bits=cfg['act_bits'],
                act_frac=cfg['act_frac'],
                softmax=cfg['softmax'],
                weight=None,  # populated by load_int_weights()
                bias=None,
            )
```

In `load_int_weights`, default a missing bias to zeros:

```python
    def load_int_weights(self, json_path):
        with open(json_path) as f:
            spec = json.load(f)['layers']

        for name in self.bundle_order:
            cfg = spec[name]
            bundle = self.bundles[name]
            bundle['weight'] = np.array(cfg['weight']['values'], dtype=np.int64)
            bias_cfg = cfg.get('bias')
            if bias_cfg is not None:
                bundle['bias'] = np.array(bias_cfg['values'], dtype=np.int64)
            else:
                bundle['bias'] = np.zeros(bundle['out_features'], dtype=np.int64)
```

Replace `forward` (adds the weights-loaded check, trace capture, and softmax):

```python
    def forward(self, x_int):
        if any(b['weight'] is None for b in self.bundles.values()):
            raise RuntimeError(
                "FixedPointModel.forward() called before load_int_weights() - "
                "weight/bias arrays are still None")

        self.outputs = {}
        self.trace = {}
        prev_frac = {}
        x_int = np.asarray(x_int, dtype=np.int64)

        for name in self.bundle_order:
            bundle = self.bundles[name]

            if bundle['input'] is None:
                inp = x_int
                inp_frac = self.bundles[self.bundle_order[0]]['input_frac']
            else:
                inp = self.outputs[bundle['input']]
                inp_frac = prev_frac[bundle['input']]

            if inp_frac != bundle['input_frac']:
                inp = shift_round(inp, inp_frac - bundle['input_frac'])
                inp = np.clip(inp, -2 ** (bundle['input_bits'] - 1), 2 ** (bundle['input_bits'] - 1) - 1)

            acc_frac = bundle['input_frac'] + bundle['weight_frac']
            assert acc_frac == bundle['bias_frac'], (
                f"bundle '{name}': accumulator frac {acc_frac} != bias frac {bundle['bias_frac']}")

            y = inp @ bundle['weight'].T          # bias-free conv-sum (matmul only)
            acc = y + bundle['bias']              # int64, frac = acc_frac

            acc_for_shift = np.clip(acc, 0, None) if bundle['activation'] == 'relu' else acc
            out = shift_round(acc_for_shift, acc_frac - bundle['act_frac'])

            if bundle['activation'] in UNSIGNED_ACTIVATIONS:
                out = np.clip(out, 0, 2 ** bundle['act_bits'] - 1)
            else:
                out = np.clip(out, -2 ** (bundle['act_bits'] - 1), 2 ** (bundle['act_bits'] - 1) - 1)

            self.trace[name] = {'x': inp, 'y': y, 'acc': acc, 'out': out}
            self.outputs[name] = out
            prev_frac[name] = bundle['act_frac']

        last_name = self.bundle_order[-1]
        last_bundle = self.bundles[last_name]
        logits_int = self.outputs[last_name]

        if last_bundle['softmax']:
            self.pre_softmax = logits_int
            self.softmax_frac = last_bundle['act_frac']
            logits_float = logits_int.astype(np.float64) / 2 ** self.softmax_frac
            factor = 2 ** 17  # fixed-point scale used by the legacy hardware softmax
            row_max = logits_float.max(axis=-1, keepdims=True)
            self.softmax_max_i = (row_max * factor).astype(np.int64)
            exp = np.exp(logits_float - row_max)
            self.softmax_out = (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)
        else:
            self.pre_softmax = None
            self.softmax_frac = None
            self.softmax_max_i = None
            self.softmax_out = None

        return logits_int
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py -v`
Expected: all tests PASS. Also re-run `python -m deepsocflow.py.brevitas.main`
to confirm the real XOR pipeline still predicts `[0, 1, 1, 0]`.

---

## Task 5: Add `signed` to the graph JSON

Adds `input_signed`/`weight.signed`/`bias.signed`/`act_signed` to
`export_graph_json`'s output, closing the "no signed/unsigned flag per tensor"
Known Issue, and updates `sim.py` to prefer `act_signed` when present (keeping
the name-based `UNSIGNED_ACTIVATIONS` fallback for JSONs generated before this
change).

**Files:**
- Modify: `deepsocflow/py/brevitas/ptq.py`
- Modify: `deepsocflow/py/brevitas/sim.py`
- Test: `deepsocflow/test/py/test_brevitas_sim.py`

**Interfaces:**
- Consumes: `quantized_model.export_graph_json` (existing).
- Produces: JSON keys `input_signed` (bool), `weight.signed` (bool),
  `bias.signed` (bool, only when a bias exists), `act_signed` (bool, only when
  the activation's quant is enabled) — consumed by `sim.py` and, later,
  `export.py`'s `check_hardware`.

- [ ] **Step 1: Write the failing test**

Append to `deepsocflow/test/py/test_brevitas_sim.py`:

```python
def test_prefers_act_signed_field_over_name_based_fallback(tmp_path):
    # A relu bundle whose JSON explicitly marks act_signed=True (e.g. hand-edited,
    # or from a future export where ReLU's output was clipped signed) must be
    # treated as signed even though 'relu' is in UNSIGNED_ACTIVATIONS by name.
    layers = {
        "bundle0": _bundle(input_frac=7, input_bits=8,
                            weight_values=[[127, 127]], weight_frac=6, weight_bits=8,
                            bias_values=[0], bias_frac=13, bias_bits=16,
                            activation="relu", act_bits=8, act_frac=6),
    }
    layers["bundle0"]["act_signed"] = True
    json_path = _write_graph(tmp_path, layers)
    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)

    # x=[127,127] @ frac=7, weight=[127,127] @ frac=6 -> acc=127*127+127*127=32258
    # @ frac=13 -> relu(32258)=32258 (already >=0) -> shift_round(32258, 7) = 252
    # pre-clip. Signed int8 clips this to 127; unsigned uint8 would leave it at
    # 252 - this is the discriminating case between the two range behaviors.
    out = model.forward(np.array([[127, 127]], dtype=np.int64))
    assert out.tolist() == [[127]]  # signed range [-128,127], not unsigned [0,255]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py::test_prefers_act_signed_field_over_name_based_fallback -v`
Expected: FAIL — today `sim.py` only looks at `UNSIGNED_ACTIVATIONS` by
activation name, ignoring any `act_signed` field, so `relu` clips to `[0, 255]`
regardless of what the JSON says.

- [ ] **Step 3: Add `signed` fields in `ptq.py`**

In `export_graph_json`, after computing `quant_input`:

```python
                quant_input = core.input_quant(x)
                quant_weight = core.quant_weight(quant_input)
                quant_bias = core.bias_quant(core.bias, quant_input, quant_weight) if core.bias is not None else None
```

add, right after (still inside the `with torch.no_grad():` loop body):

```python
                input_signed = bool(quant_input.signed)
                weight_signed = bool(quant_weight.signed)
                bias_signed = bool(quant_bias.signed) if quant_bias is not None else None
```

In the `layer = {...}` dict literal, add `"input_signed": input_signed,` next
to `"input_bits"`; in the `"weight": {...}` dict, add `"signed": weight_signed,`;
in the `if quant_bias is not None:` block's `layer["bias"] = {...}`, add
`"signed": bias_signed,`. After the existing:

```python
                if core.act.act_quant.is_quant_enabled:
                    act_scale = core.act.act_quant.scale().item()
                    layer["act_bits"] = int(core.act.act_quant.bit_width().item())
                    layer["act_frac"] = _frac_bits(act_scale)
                    layer["act_scale"] = act_scale
                    layer["act_zero_point"] = core.act.act_quant.zero_point().item()
```

add one more line inside that `if`:

```python
                    layer["act_signed"] = bool(core.act.act_quant.is_signed)
```

- [ ] **Step 4: Read `act_signed` in `sim.py`**

In `_build_topology`, store the field (falling back to the name-based set when
an older JSON doesn't have it):

```python
            act_signed = cfg.get('act_signed')
            if act_signed is None:
                act_signed = cfg['activation'] not in UNSIGNED_ACTIVATIONS

            self.bundles[name] = dict(
                input=cfg.get('input'),
                in_features=cfg['in_features'],
                out_features=cfg['out_features'],
                input_bits=cfg['input_bits'],
                input_frac=cfg['input_frac'],
                weight_frac=cfg['weight']['frac'],
                bias_frac=bias_frac,
                activation=cfg['activation'],
                act_bits=cfg['act_bits'],
                act_frac=cfg['act_frac'],
                act_signed=act_signed,
                softmax=cfg['softmax'],
                weight=None,
                bias=None,
            )
```

In `forward`, replace the unsigned-check with the stored flag:

```python
            if bundle['act_signed']:
                out = np.clip(out, -2 ** (bundle['act_bits'] - 1), 2 ** (bundle['act_bits'] - 1) - 1)
            else:
                out = np.clip(out, 0, 2 ** bundle['act_bits'] - 1)
```

(This replaces the earlier `if bundle['activation'] in UNSIGNED_ACTIVATIONS:`
branch from Task 4's `forward`.) Keep the `UNSIGNED_ACTIVATIONS` set/comment in
`sim.py` — it's now the documented fallback for JSONs without `act_signed`,
referenced by `_build_topology` above.

- [ ] **Step 5: Regenerate the real graph JSON and run all tests**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m deepsocflow.py.brevitas.xor`
Expected: prints `predictions: [0, 1, 1, 0]` and `qm predictions: [0, 1, 1, 0]`,
regenerates `deepsocflow/py/brevitas/model/xor_graph.json` with the new
`*_signed`/`act_signed` fields.

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py -v`
Expected: all tests PASS.

---

## Task 6: Single quantization point per bundle

Currently every `QuantLinear` gets its own `input_quant`, so a second
quantization point exists right after every activation (qkeras only ever has
one). This task makes only the first bundle keep an `input_quant`; every
following bundle consumes the previous bundle's activation output directly as
a `QuantTensor` (`return_quant_tensor=True`), which also lets `Int32Bias`
resolve its scale without a redundant re-quantization. It also narrows
unsigned (ReLU/Sigmoid) activations to `bits-1`, mirroring
`xlayers.py:31-33`'s "QKeras treats relu as unsigned, we have everything
signed, so we reduce bitwidth" — otherwise an unsigned 8-bit ReLU output can
reach 255, which silently wraps to a negative number if ever packed into a
signed 8-bit word.

**Escape hatch:** if brevitas's `QuantTensor` propagation through
`QuantReLU → QuantLinear` (with `input_quant=None`) doesn't work as verified
below, stop, revert this task's `ptq.py` changes, and add a new Known Issue to
`CLAUDE.md` instead of forcing it — Task 3's explicit requant already handles
the frac-mismatch case correctly regardless. (This was verified working
end-to-end against the real XOR model while writing this plan — see the
worked example in Step 3 below — so the escape hatch is not expected to
trigger, but is documented in case brevitas's behavior differs in a future
version.)

**Files:**
- Modify: `deepsocflow/py/brevitas/ptq.py`
- Test: `deepsocflow/test/py/test_brevitas_sim.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks.
- Produces: `quant_layer.has_own_input_quant` (bool, stashed on each built
  layer, same pattern as the existing `configured_bias_bits`) — read by
  `export_graph_json`. After this task, every bundle after the first has
  `input_frac == previous bundle's act_frac` in the exported JSON, by
  construction.

- [ ] **Step 1: Write the failing test**

Append to `deepsocflow/test/py/test_brevitas_sim.py`:

```python
def test_json_has_single_quantization_point_per_bundle():
    """Regenerates the real XOR graph and checks every bundle after the first
    has input_frac == the previous bundle's act_frac - i.e. Task 3's requant
    step is now a structural no-op, not a band-aid."""
    import json
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "deepsocflow.py.brevitas.xor"], check=True,
                    cwd="/Users/charaphat/CERN/cgra4ml", capture_output=True)

    with open("/Users/charaphat/CERN/cgra4ml/deepsocflow/py/brevitas/model/xor_graph.json") as f:
        layers = json.load(f)["layers"]

    names = list(layers.keys())
    for i in range(1, len(names)):
        prev, cur = layers[names[i - 1]], layers[names[i]]
        assert cur["input_frac"] == prev["act_frac"], (
            f"{names[i]}.input_frac ({cur['input_frac']}) != "
            f"{names[i-1]}.act_frac ({prev['act_frac']})")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py::test_json_has_single_quantization_point_per_bundle -v`
Expected: FAIL — today `bundle0.act_frac=6` but `bundle1.input_frac=5` in the
regenerated JSON (the exact mismatch documented in the design spec).

- [ ] **Step 3: Implement in `ptq.py`**

Replace `_quantize_layer`'s signature and body:

```python
def _quantize_layer(layer, weight_bits=8, bias_bits=32, own_input_quant=True):
    # Compute layers: plain torch type -> our xlayer quant equivalent.
    LAYER_MAP = {
        nn.Linear: QuantLinear,
        nn.Conv1d: QuantConv1d,
        nn.Conv2d: QuantConv2d,
        nn.Conv3d: QuantConv3d,
    }
    quant_cls = LAYER_MAP.get(type(layer))
    if quant_cls is None:
        return None

    has_bias = layer.bias is not None
    bias_quant = Int32Bias if has_bias else None

    # own_input_quant=False means this layer consumes the previous bundle's
    # activation output directly as an already-quantized QuantTensor (that
    # activation was built with return_quant_tensor=True) instead of
    # re-quantizing it - a single quantization point per bundle, matching
    # qkeras's structure, instead of one after every activation AND one
    # before every layer.
    input_quant = Int8ActPerTensorFixedPoint if own_input_quant else None

    if isinstance(layer, nn.Linear):
        quant_layer = quant_cls(
            layer.in_features, layer.out_features, bias=has_bias,
            weight_quant=Int8WeightPerTensorFixedPoint, weight_bit_width=weight_bits,
            input_quant=input_quant,
            bias_quant=bias_quant, bias_bit_width=bias_bits)
    else:
        kwargs = {name: getattr(layer, name) for name in _CONV_ATTRS}
        quant_layer = quant_cls(
            bias=has_bias,
            weight_quant=Int8WeightPerTensorFixedPoint, weight_bit_width=weight_bits,
            input_quant=input_quant,
            bias_quant=bias_quant, bias_bit_width=bias_bits, **kwargs)
    quant_layer.load_state_dict(layer.state_dict(), strict=False)

    quant_layer.configured_bias_bits = bias_bits if has_bias else None
    quant_layer.has_own_input_quant = own_input_quant
    return quant_layer
```

Replace `_quantize_activation`:

```python
def _quantize_activation(act):
    # Activations: plain torch type -> (our xlayer quant equivalent, power-of-two scale
    # act_quant matching its sign - Uint8 for ReLU/Sigmoid outputs, which are >= 0,
    # Int8 for everything else).
    ACT_MAP = {
        nn.ReLU: (QuantReLU, Uint8ActPerTensorFixedPoint),
        nn.Sigmoid: (QuantSigmoid, Uint8ActPerTensorFixedPoint),
        nn.Tanh: (QuantTanh, Int8ActPerTensorFixedPoint),
        nn.LeakyReLU: (QuantLeakyReLU, Int8ActPerTensorFixedPoint),
        nn.SiLU: (QuantSiLU, Int8ActPerTensorFixedPoint),
        nn.SELU: (QuantSELU, Int8ActPerTensorFixedPoint),
        nn.GELU: (QuantGELU, Int8ActPerTensorFixedPoint),
        nn.Identity: (QuantIdentity, Int8ActPerTensorFixedPoint),
    }
    entry = ACT_MAP.get(type(act))
    if entry is None:
        return None
    quant_cls, act_quant = entry

    # return_quant_tensor=True: every activation's output must carry its own
    # scale/bit-width so the next bundle's layer (built with
    # own_input_quant=False) can consume it directly instead of re-quantizing.
    kwargs = {"act_quant": act_quant, "return_quant_tensor": True}

    # Unsigned activations (ReLU/Sigmoid) are narrowed to bits-1 so their output
    # still fits the signed datapath every other tensor in this project uses
    # (mirrors xlayers.py:31-33: "QKeras treats relu as unsigned, we have
    # everything signed, so we reduce bitwidth" - an unsigned 8-bit value can
    # reach 255, which silently wraps when packed into a signed 8-bit word).
    if act_quant is Uint8ActPerTensorFixedPoint:
        kwargs["bit_width"] = 7

    return quant_cls(**kwargs)
```

In `quantized_model.__init__`, change the `_quantize_layer` call to pass
`own_input_quant`:

```python
            core = _quantize_layer(
                layer,
                weight_bits=overrides.get('weight_bits', weight_bits),
                bias_bits=overrides.get('bias_bits', bias_bits),
                own_input_quant=(len(self.bundles) == 0))
```

(`len(self.bundles) == 0` is true exactly while building the first bundle,
since bundles are appended to `self.bundles` at the end of each loop
iteration.)

In `export_graph_json`, replace the unconditional `quant_input = core.input_quant(x)`
line with:

```python
                quant_input = core.input_quant(x) if core.has_own_input_quant else x
```

Also fix the `"input_bits"` line a few lines below, in the same function. It
currently reads `"input_bits": int(core.input_quant.bit_width().item()),` -
`core.input_quant.bit_width()` returns `None` (not a tensor) whenever
`input_quant=None` was passed at construction (verified: calling `.item()` on
that raises `AttributeError: 'NoneType' object has no attribute 'item'`), so
this line would crash for every bundle after the first as soon as
`own_input_quant=False` is introduced. Use `quant_input.bit_width` instead -
it's always a real tensor on the resolved `QuantTensor`, whether `quant_input`
was freshly computed by `core.input_quant(x)` or passed through directly:

```python
                "input_bits": int(quant_input.bit_width.item()),
```

**Important, discovered while validating this task:** after this change,
`bundle0.act_bits` (ReLU, narrowed to 7) differs from `bundle2.act_bits`
(Identity, still 8) in the real XOR graph - and since bundle1/bundle2's
`input_bits` now equals the *previous* bundle's `act_bits` (single
quantization point), the real regenerated JSON ends up with `input_bits`/
`act_bits` of `7` on some bundles and `8` on others, not a uniform `8`
everywhere. This is expected and correct (7-bit ReLU values fit trivially
inside an 8-bit signed word - they just don't use its full range) - it means
Task 7's `check_hardware` must check "fits within `hw.X_BITS`"
(`bundle['act_bits'] <= hw.X_BITS`), not "equals `hw.X_BITS`" exactly. Task 7
is written that way already; this note just explains why.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_sim.py -v`
Expected: all tests PASS, including
`test_json_has_single_quantization_point_per_bundle`.

Also manually confirm bit-exactness end to end:

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m deepsocflow.py.brevitas.main`
Expected: `predictions: [0, 1, 1, 0]` — and since Task 3's requant is now a
no-op (fracs match by construction), this is the FixedPointModel's output
matching brevitas's own fake-quantized forward bit-for-bit, not just on
argmax.

---

## Task 7: `export.py` — golden-reference file export (Phase 1)

New module: `check_hardware` (bit-width/range asserts), `_savetxt_int`/
`_savetxt_float`/`_to_nhwc` (formatting helpers), and `export_inference`
(writes `y_exp.txt` and `{ib}_y_nhwc_exp.txt` into `hw.DATA_DIR`, cleaning it
first) — the brevitas-side equivalent of `xmodel.py::export_inference`'s
layout-independent files.

**Files:**
- Create: `deepsocflow/py/brevitas/export.py`
- Test: `deepsocflow/test/py/test_brevitas_export_inference.py`

**Interfaces:**
- Consumes: `FixedPointModel` (Tasks 2-6, needs `.trace`, `.softmax_out`,
  `.bundles`, `.bundle_order` all populated by a completed `forward()` call),
  `Hardware` (`deepsocflow/py/brevitas/hardware.py`, unchanged, needs
  `.X_BITS`, `.K_BITS`, `.B_BITS`, `.Y_BITS`, `.DATA_DIR`).
- Produces: `export_inference(model, hw, x_float, data_dir=None, clean=True,
  batch_size=1) -> dict` with keys `'files'` (list of str paths written),
  `'y_exp'` (`np.ndarray`), `'softmax_frac'` (`int`), `'softmax_max_i'`
  (`np.ndarray`); `check_hardware(model, hw)` (raises `AssertionError`, returns
  `None` on success).

- [ ] **Step 1: Write the failing tests**

Create `deepsocflow/test/py/test_brevitas_export_inference.py`:

```python
import json
import os

import numpy as np
import pytest
import torch

from deepsocflow.py.brevitas.export import check_hardware, export_inference
from deepsocflow.py.brevitas.hardware import Hardware
from deepsocflow.py.brevitas.sim import FixedPointModel
from deepsocflow.py.brevitas.xor import MODEL_DIR, MODEL_PATH, X, Y

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason=f"'{MODEL_PATH}' not found - run `python -m deepsocflow.py.brevitas.xor` to generate it")


def _build_model(tmp_path):
    """Rebuilds quantized_model from the checked-in trained weights, regenerates
    the graph JSON into tmp_path (never reads the repo copy - it's a derived
    artefact and could be stale), and returns a forward()-ready FixedPointModel."""
    from deepsocflow.py.brevitas.ptq import quantized_model
    from deepsocflow.py.brevitas.xor import XOR, load

    net = XOR()
    net = load(net, path=MODEL_PATH)
    qm = quantized_model(net, layer_bits={
        'hidden_1': {'weight_bits': 8, 'bias_bits': 16},
        'hidden_2': {'weight_bits': 8, 'bias_bits': 16},
        'out': {'weight_bits': 8, 'bias_bits': 16},
    })
    qm.quantization(X)

    json_path = str(tmp_path / "xor_graph.json")
    qm.export_graph_json(X, json_path)

    model = FixedPointModel(json_path)
    model.load_int_weights(json_path)
    return model, qm


def _hardware(tmp_path):
    return Hardware(
        processing_elements=(8, 24),
        bits_input=8, bits_weights=8, bits_bias=16, bits_sum=32,
        data_dir=str(tmp_path / "vectors"))


def test_export_inference_writes_y_exp_and_per_bundle_files(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)

    result = export_inference(model, hw, X, batch_size=1)

    assert os.path.exists(os.path.join(hw.DATA_DIR, "y_exp.txt"))
    for ib in range(len(model.bundle_order)):
        assert os.path.exists(os.path.join(hw.DATA_DIR, f"{ib}_y_nhwc_exp.txt"))
    assert set(result["files"]) == {
        os.path.join(hw.DATA_DIR, "y_exp.txt"),
        *(os.path.join(hw.DATA_DIR, f"{ib}_y_nhwc_exp.txt") for ib in range(len(model.bundle_order))),
    }


def test_y_exp_txt_matches_model_softmax_output(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    export_inference(model, hw, X, batch_size=1)

    y_exp = np.loadtxt(os.path.join(hw.DATA_DIR, "y_exp.txt"))
    model.forward(model.quantize_input(X[:1]))
    assert y_exp.tolist() == pytest.approx(model.softmax_out[0].tolist(), abs=1e-6)


def test_y_exp_is_float_format_when_softmax(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    export_inference(model, hw, X, batch_size=1)

    with open(os.path.join(hw.DATA_DIR, "y_exp.txt")) as f:
        lines = [line.strip() for line in f if line.strip()]
    assert all("." in line for line in lines)


def test_y_nhwc_exp_is_int_format(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    export_inference(model, hw, X, batch_size=1)

    with open(os.path.join(hw.DATA_DIR, "0_y_nhwc_exp.txt")) as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        int(line)  # raises ValueError if not a plain int (e.g. "1.0" or "1e5")


def test_y_nhwc_exp_matches_bundle_trace_output(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    export_inference(model, hw, X, batch_size=1)

    model.forward(model.quantize_input(X[:1]))
    for ib, name in enumerate(model.bundle_order):
        vals = np.loadtxt(os.path.join(hw.DATA_DIR, f"{ib}_y_nhwc_exp.txt"), dtype=np.int64)
        expected = model.trace[name]["out"].flatten()
        assert vals.tolist() == expected.tolist()


def test_export_cleans_data_dir(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = _hardware(tmp_path)
    os.makedirs(hw.DATA_DIR, exist_ok=True)
    junk_path = os.path.join(hw.DATA_DIR, "leftover_junk.txt")
    with open(junk_path, 'w') as f:
        f.write("stale")

    export_inference(model, hw, X, batch_size=1)

    assert not os.path.exists(junk_path)


def test_hardware_bitwidth_mismatch_raises(tmp_path):
    model, _ = _build_model(tmp_path)
    hw = Hardware(processing_elements=(8, 24), bits_input=4, bits_weights=8,
                   bits_bias=16, bits_sum=32, data_dir=str(tmp_path / "vectors"))

    with pytest.raises(AssertionError, match="X_BITS"):
        check_hardware(model, hw)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_export_inference.py -v`
Expected: all FAIL with `ModuleNotFoundError: No module named
'deepsocflow.py.brevitas.export'` (or skip entirely if `xor.pt` is missing —
run `python -m deepsocflow.py.brevitas.xor` first if so).

- [ ] **Step 3: Implement `export.py`**

Create `deepsocflow/py/brevitas/export.py`:

```python
import os

import numpy as np


def clog2(x):
    return int(np.ceil(np.log2(x)))


def check_hardware(model, hw):
    """Asserts every bundle's tensors FIT the hardware's configured bit-widths
    (<=, not ==): a bundle can legitimately declare fewer bits than hw.X_BITS
    (e.g. a ReLU activation narrowed to bits-1 so its non-negative output still
    fits the signed datapath - see ptq.py's _quantize_activation) without that
    being a hardware mismatch. Mirrors deepsocflow/py/xmodel.py:55-57 and
    xbundle.py:148-161. Must be called after model.forward() has populated
    model.trace, since the activation-range check inspects real computed
    values, not just declared bits."""
    for name in model.bundle_order:
        bundle = model.bundles[name]

        assert bundle['input_bits'] <= hw.X_BITS, (
            f"bundle '{name}': input_bits={bundle['input_bits']} > hw.X_BITS={hw.X_BITS}")
        assert bundle['act_bits'] <= hw.X_BITS, (
            f"bundle '{name}': act_bits={bundle['act_bits']} > hw.X_BITS={hw.X_BITS}")

        # ACC_WIDTH bound - Phase 1 uses the bundle's real in_features (CI) as the
        # channel count, unlike the legacy backend's RAM_WEIGHTS_DEPTH-derived r.CM
        # (which pads to the hardware's max channel depth, not the model's actual
        # shape) - that padding only matters once engine-layout export (Phase 2,
        # not in this plan) is wired in.
        acc_width = hw.K_BITS + hw.X_BITS + clog2(bundle['in_features'])
        assert acc_width <= hw.Y_BITS, (
            f"bundle '{name}': ACC_WIDTH={acc_width} > hw.Y_BITS={hw.Y_BITS}")

        if name in getattr(model, 'trace', {}):
            out = model.trace[name]['out']
            if bundle['act_signed']:
                lo, hi = -2 ** (hw.X_BITS - 1), 2 ** (hw.X_BITS - 1) - 1
            else:
                lo, hi = 0, 2 ** hw.X_BITS - 1
            assert out.min() >= lo and out.max() <= hi, (
                f"bundle '{name}': activation values [{out.min()},{out.max()}] "
                f"outside signed={bundle['act_signed']} hw.X_BITS={hw.X_BITS} range [{lo},{hi}]")


def _savetxt_int(path, arr):
    np.savetxt(path, np.asarray(arr).flatten(), fmt='%d')


def _savetxt_float(path, arr):
    np.savetxt(path, np.asarray(arr).flatten(), fmt='%f')


def _to_nhwc(out_2d):
    """(XN, CO) -> (1, XN, 1, CO), matching the dense-as-1x1-conv reshape at
    deepsocflow/py/xbundle.py:125-128."""
    xn, co = out_2d.shape
    return out_2d.reshape(1, xn, 1, co)


def export_inference(model, hw, x_float, data_dir=None, clean=True, batch_size=1):
    """Runs model.forward() on x_float[:batch_size] and writes the
    layout-independent golden-reference files legacy's xmodel.py::export_inference
    produces (y_exp.txt, {ib}_y_nhwc_exp.txt) into hw.DATA_DIR - same filenames,
    same np.savetxt formats, same flatten order, so a future RTL step can
    consume them unchanged. Engine-layout files ({ib}_{ip}_{it}_*, .bin blobs)
    are not produced here - see the design spec's Phase 2."""
    data_dir = data_dir or hw.DATA_DIR

    if clean:
        os.makedirs(data_dir, exist_ok=True)
        for entry in os.scandir(data_dir):
            os.remove(entry.path)
    else:
        os.makedirs(data_dir, exist_ok=True)

    x_batch = np.asarray(x_float)[:batch_size]
    x_int = model.quantize_input(x_batch)
    model.forward(x_int)

    check_hardware(model, hw)

    files = []

    last_name = model.bundle_order[-1]
    last_bundle = model.bundles[last_name]
    if last_bundle['softmax']:
        y_exp = model.softmax_out
        y_exp_path = os.path.join(data_dir, "y_exp.txt")
        _savetxt_float(y_exp_path, y_exp)
    else:
        y_exp = model.outputs[last_name]
        y_exp_path = os.path.join(data_dir, "y_exp.txt")
        _savetxt_int(y_exp_path, y_exp)
    files.append(y_exp_path)

    flat = y_exp.flatten()
    n = len(flat)
    for i in range(n):
        if i < 20 or n - i <= 20:
            print(f"y_exp {i}: {flat[i]}")

    for ib, name in enumerate(model.bundle_order):
        out = model.trace[name]['out']
        nhwc = _to_nhwc(out)
        path = os.path.join(data_dir, f"{ib}_y_nhwc_exp.txt")
        _savetxt_int(path, nhwc)
        files.append(path)

    print(f"Weights, inputs, outputs saved to {data_dir}/")

    return {
        'files': files,
        'y_exp': y_exp,
        'softmax_frac': model.softmax_frac,
        'softmax_max_i': model.softmax_max_i,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m pytest deepsocflow/test/py/test_brevitas_export_inference.py -v`
Expected: all tests PASS. Note: after Task 6, the real XOR graph has
`act_bits=7` on the ReLU bundles and `act_bits=8` on the final identity
bundle (Task 6 Step 3's narrowing) — with `check_hardware`'s `<=` check
(rather than `==`) this is expected and fine, since `_hardware()`'s
`bits_input=8` comfortably covers both. `test_hardware_bitwidth_mismatch_raises`
still triggers correctly because its `bits_input=4` is too narrow even for the
smallest (7-bit) bundle.

---

## Task 8: Wire `main.py`

Drives the full pipeline: build `FixedPointModel` from the graph JSON, load
int weights, construct an XOR-sized `Hardware`, and call `export_inference`.

**Files:**
- Modify: `deepsocflow/py/brevitas/main.py`

**Interfaces:**
- Consumes: `FixedPointModel` (Tasks 2-6), `export.export_inference` (Task 7),
  `Hardware` (`deepsocflow/py/brevitas/hardware.py`, unchanged).
- Produces: nothing new consumed elsewhere - this is the runnable entry point.

- [ ] **Step 1: Replace `main.py`**

```python
import os

from deepsocflow.py.brevitas.export import export_inference
from deepsocflow.py.brevitas.hardware import Hardware
from deepsocflow.py.brevitas.sim import FixedPointModel
from deepsocflow.py.brevitas.xor import X, Y

if __name__ == '__main__':
    json_path = os.path.join(os.path.dirname(__file__), 'model', 'xor_graph.json')

    # 1. build the model's structure from the already-quantized graph JSON
    #    (shapes, activations, topology - weight/bias arrays not populated yet)
    model = FixedPointModel(json_path)
    print(f"Built model '{json_path}' with {len(model.bundle_order)} bundles")

    # 2. only now pull the quantized int weight/bias values in, from the same JSON
    model.load_int_weights(json_path)
    print("Loaded int weights")

    x_int = model.quantize_input(X)
    logits_int = model.forward(x_int)  # pure int64 arithmetic from here on
    preds = logits_int.argmax(axis=-1)  # softmax is monotonic - argmax unaffected

    print()
    print("x_int:", x_int.tolist())
    print("logits_int:", logits_int.tolist())
    print("targets:    ", Y.tolist())
    print("predictions:", preds.tolist())

    print()
    print("Model graph:")
    model.print_graph()

    # 3. export the golden-reference files a future RTL step will diff against
    #    (batch_size=1 - matches the legacy dense convention, run/param_test.py)
    hw = Hardware(
        processing_elements=(8, 24),
        bits_input=8, bits_weights=8, bits_bias=16, bits_sum=32,
        data_dir=os.path.join(os.path.dirname(__file__), 'vectors'))

    print()
    result = export_inference(model, hw, X, batch_size=1)
    print(f"Exported golden-reference files: {result['files']}")
```

- [ ] **Step 2: Run it end to end**

Run: `cd /Users/charaphat/CERN/cgra4ml && python -m deepsocflow.py.brevitas.main`
Expected: prints the model graph, `predictions: [0, 1, 1, 0]`, and
`Exported golden-reference files: [...]` listing `y_exp.txt` and
`0_y_nhwc_exp.txt`/`1_y_nhwc_exp.txt`/`2_y_nhwc_exp.txt` under
`deepsocflow/py/brevitas/vectors/`. Confirm those files exist:

Run: `ls -la /Users/charaphat/CERN/cgra4ml/deepsocflow/py/brevitas/vectors/`
Expected: `y_exp.txt`, `0_y_nhwc_exp.txt`, `1_y_nhwc_exp.txt`,
`2_y_nhwc_exp.txt` all present.

---

## Task 9: Update `CLAUDE.md`

Records what changed, closes the signed/unsigned Known Issue, and documents
any new limitation found while running Task 7's tests (e.g. the
mixed-activation-bit-width note from Task 7 Step 4, if it triggered).

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Known Issues section**

In the `## quantized_model.export_graph_json()...` Known Issues entry, find
the bullet:

```
- **No `signed`/`unsigned` flag per tensor.** `ACT_MAP` in `ptq.py` uses `Uint8ActPerTensorFixedPoint` for ReLU/Sigmoid (unsigned - no sign bit) but `Int8ActPerTensorFixedPoint` for everything else (signed). The int-bits formula differs (`bits - frac` for unsigned vs `bits - frac - 1` for signed), but the JSON doesn't record which applies to a given tensor - a simulator would have to guess from the activation name.
```

Replace it with:

```
- ~~**No `signed`/`unsigned` flag per tensor.**~~ **Closed 2026-08-10** - `export_graph_json` now emits `input_signed`/`weight.signed`/`bias.signed`/`act_signed` per tensor (`ptq.py`), sourced from brevitas's own `QuantTensor.signed`/`act_quant.is_signed`. `sim.py` reads `act_signed` directly, falling back to the old name-based `UNSIGNED_ACTIVATIONS` set only for JSONs exported before this change.
```

- [ ] **Step 2: Add a Progress Log entry**

Append a new dated section after the existing `## 2026-08-10 - ...` entry (use
today's actual date if different):

```markdown
## 2026-08-10 - brevitas golden-reference (`*_exp.txt`) export, bit-exactness fixes, single quantization point

- Found and fixed two real bugs in `sim.py`'s `FixedPointModel`, discovered by comparing its per-bundle int output against brevitas's own forward pass bundle-by-bundle (not just `argmax`, which was hiding the divergence): (1) `forward()` fed one bundle's output straight into the next without requantizing when their fracs differed (every `QuantLinear` had its own `input_quant`, creating a second quantization point after each activation that qkeras never has) - fixed by requantizing (`shift_round` + clip) whenever a producing bundle's `act_frac` doesn't match the consuming bundle's `input_frac`; (2) `quantize_input` didn't clip the rounded input to `input_bits`, so `X=1.0` at `frac=7` produced `128` (one past signed int8's `127`) instead of clipping like brevitas's own input quantizer does.
- `ptq.py`: switched to a **single quantization point per bundle** - only the first bundle's `QuantLinear` gets an `input_quant`; every later bundle consumes the previous bundle's activation output directly as an already-quantized `QuantTensor` (`return_quant_tensor=True` on every activation), letting `Int32Bias` resolve its scale from that tensor directly. Also narrowed unsigned (ReLU/Sigmoid) activations to `bits-1` (mirrors the legacy `xlayers.py:31-33`'s reasoning: an unsigned 8-bit value can reach 255, which wraps if ever packed into this project's otherwise-all-signed 8-bit words). Verified empirically against the real XOR model: predictions stay `[0,1,1,0]`, and Task 3's inter-bundle requant fix becomes a structural no-op (`input_frac` now always equals the previous bundle's `act_frac`) rather than a band-aid.
- `ptq.py::export_graph_json`: added `input_signed`/`weight.signed`/`bias.signed`/`act_signed` fields, closing the "no signed/unsigned flag per tensor" Known Issue.
- `sim.py`: added per-bundle `self.trace` (`x`/`y`/`acc`/`out` per bundle, `y` being the bias-free conv-sum), softmax handling (`self.pre_softmax`/`softmax_frac`/`softmax_max_i`/`softmax_out`, mirroring the legacy `xbundle.py`'s `2**17`-factor fixed-point softmax, but normalized per-row rather than by row 0's sum - the legacy formula is only correct for `batch_size==1`), a `RuntimeError` (not a cryptic `TypeError`) when `forward()` is called before `load_int_weights()`, and defaulting bias-less layers to zero (the JSON export only emits a `"bias"` key `if quant_bias is not None` - previously unhandled since XOR always has biases).
- New `deepsocflow/py/brevitas/export.py`: `export_inference(model, hw, x_float, batch_size=1)` writes `y_exp.txt` (`%f` if the last bundle has softmax, else `%d`) and `{ib}_y_nhwc_exp.txt` per bundle into `hw.DATA_DIR` - same filenames/formats/flatten order as the legacy `deepsocflow/py/xmodel.py::export_inference`, so a future RTL step can consume them unchanged. `check_hardware(model, hw)` asserts bit-widths and activation ranges match the target `Hardware` config. Engine-layout files (`{ib}_{ip}_{it}_*`, `.bin` blobs) are deliberately deferred - see the design spec's Phase 2 (importing the legacy `dataflow.py`'s reorder functions behind a brevitas-side adapter, rather than transcribing them, since they encode non-obvious hardware invariants that only an RTL diff - out of scope this session - would catch if ported wrong).
- `main.py`: now builds `FixedPointModel` from the graph JSON, loads int weights, runs the int forward pass, and calls `export_inference` with an XOR-sized `Hardware` (`bits_input=8, bits_weights=8, bits_bias=16, bits_sum=32` - `bits_sum` needs headroom for `ACC_WIDTH = K_BITS + X_BITS + clog2(in_features)`).
- New tests: `deepsocflow/test/py/test_brevitas_sim.py` (unit, no training needed - shift_round parity vs legacy, input clipping, inter-bundle requant, per-bundle trace, softmax, error handling, signed-flag precedence, single-quantization-point structural check) and `deepsocflow/test/py/test_brevitas_export_inference.py` (end-to-end against the checked-in trained `xor.pt`, regenerating the graph JSON into `tmp_path` every run rather than reading the repo copy).
- Design doc: `docs/superpowers/specs/2026-08-10-brevitas-golden-reference-export-design.md`. Plan: `docs/superpowers/plans/2026-08-10-brevitas-golden-reference-export.md`.
```

If Task 7 Step 4's mixed-activation-bit-width issue triggered (some
bundles' `act_bits` is 7 post-Task-6 narrowing while others stay 8, so a
single `hw.X_BITS` can't match all of them), add one more bullet to this
entry describing it as a new Known Issue, and add a corresponding new
`## ...` entry under `# Known Issues` — write the exact bullet text based on
what the actual assertion failure said, since it depends on which bundle's
mismatch surfaced first.

- [ ] **Step 2: Confirm the file reads correctly**

Run: `cd /Users/charaphat/CERN/cgra4ml && head -40 CLAUDE.md`
Expected: the "Closed 2026-08-10" line renders under Known Issues, and the
file is still valid Markdown (no broken headers/lists).

---

## Not in this plan (see spec's Phase 2)

- Engine-layout files (`{ib}_{ip}_{it}_y_exp.txt`, `{ib}_xe.txt`,
  `{ib}_{ip}_x.txt`, `{ib}_{ip}_{it}_w.txt`, the `.bin` blobs) and the
  `deepsocflow/py/brevitas/dataflow.py` adapter that would produce them.
- Any RTL/verilator run or `_sim.txt` diffing (`verify_inference` equivalent).
- `config_fw.h` generation (needs buffer allocation and `ca_shift`/`ca_nzero`
  that don't exist in the brevitas path).
- Conv/pooling/residual model support.
