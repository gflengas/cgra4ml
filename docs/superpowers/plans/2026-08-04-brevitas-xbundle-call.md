# Brevitas XBundle.call() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `XBundle.call()` from the Keras/QKeras original (`deepsocflow/py/xbundle.py`) to a working PyTorch/Brevitas equivalent in `deepsocflow/py/brevitas/xbundle/xbundle.py`, with `call_int`/`export` present as stubs so the class shape matches the original.

**Architecture:** `XBundle` becomes a `torch.nn.Module` that orchestrates an existing `core` (+ `.act`), optional `pool` (+ `.act`), optional `QuantResidualAdd` (new), optional `nn.Flatten`, optional `nn.Softmax`. A module-level `BUNDLES` list (new, mirrors the Keras original's `utils.BUNDLES`) tracks bundle-to-bundle graph edges (`ib`/`prev_ib`/`next_ibs`/`next_add_ibs`) via attributes attached directly to output tensors (verified: plain `torch.Tensor` supports arbitrary attribute assignment).

**Tech Stack:** Python 3.11, PyTorch 2.12, Brevitas 0.12.1, pytest 7.4.0 (already a project dependency).

## Global Constraints

- Only work on the `brevitas-qonnx-backend` branch (per project CLAUDE.md).
- **Never run `git commit` or `git push`.** Every task below ends with a review checkpoint instead of a commit step — stage nothing, commit nothing. The user commits themselves.
- PTQ only, not QAT — this doesn't affect any code in this plan, but don't add training-loop/calibration code; that's out of scope.
- `call_int` and `export` are stubs in this plan (raise `NotImplementedError`) — do not attempt real fixed-point/hardware-layout implementations; they depend on an `XTensor` and `hardware.py`/`dataflow.py` port that doesn't exist yet for this backend.
- Follow the existing codebase convention: orchestration methods are named `call`/`call_int` (not `forward`), matching `deepsocflow/py/xlayers.py` and the Keras `XBundle`. `XBundle` additionally sets `forward = call` so it also works via plain `nn.Module.__call__`.

Spec: `docs/superpowers/specs/2026-08-04-brevitas-xbundle-design.md`

---

## File Structure

- Create: `deepsocflow/py/brevitas/__init__.py` — empty, makes `deepsocflow.py.brevitas` an importable package (sibling `xlayer/__init__.py` already exists; `brevitas/` and `xbundle/` are missing theirs).
- Create: `deepsocflow/py/brevitas/xbundle/__init__.py` — empty, same reason for `deepsocflow.py.brevitas.xbundle`.
- Create: `deepsocflow/py/brevitas/utils.py` — `BUNDLES = []`, mirrors `deepsocflow/py/utils.py`.
- Modify: `deepsocflow/py/brevitas/xlayer/quantOperation.py` — add `QuantResidualAdd`, replacing the `ResidualAdd` comment placeholder.
- Modify: `deepsocflow/py/brevitas/xbundle/xbundle.py` — replace the empty stub with the real `XBundle` class.
- Create: `deepsocflow/test/py/test_brevitas_utils.py`
- Create: `deepsocflow/test/py/test_brevitas_quant_operation.py`
- Create: `deepsocflow/test/py/test_brevitas_xbundle.py`

(`deepsocflow/test/py/` currently only has notebooks; these are the first `.py` pytest files there. `pytest` is already listed in `pyproject.toml`, so no new dependency is needed.)

---

### Task 1: `BUNDLES` global + package scaffolding

**Files:**
- Create: `deepsocflow/py/brevitas/__init__.py`
- Create: `deepsocflow/py/brevitas/xbundle/__init__.py`
- Create: `deepsocflow/py/brevitas/utils.py`
- Test: `deepsocflow/test/py/test_brevitas_utils.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `deepsocflow.py.brevitas.utils.BUNDLES` — a plain `list`, appended to by `XBundle` instances (Task 3) and read by `QuantResidualAdd`/`XBundle` to look up prior bundles by index.

- [ ] **Step 1: Write the failing test**

Create `deepsocflow/test/py/test_brevitas_utils.py`:

```python
from deepsocflow.py.brevitas.utils import BUNDLES


def test_bundles_is_a_list():
    assert isinstance(BUNDLES, list)


def test_bundles_starts_empty_after_clear():
    BUNDLES.clear()
    assert BUNDLES == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_utils.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'deepsocflow.py.brevitas.utils'` (and possibly `No module named 'deepsocflow.py.brevitas'` first, since that package doesn't exist yet either).

- [ ] **Step 3: Create the package files**

Create `deepsocflow/py/brevitas/__init__.py` (empty file).

Create `deepsocflow/py/brevitas/xbundle/__init__.py` (empty file).

Create `deepsocflow/py/brevitas/utils.py`:

```python
BUNDLES = []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_utils.py -v`
Expected: 2 passed

- [ ] **Step 5: Review checkpoint**

Do not commit (see Global Constraints). Leave the new files unstaged for the user to review and commit themselves.

---

### Task 2: `QuantResidualAdd`

**Files:**
- Modify: `deepsocflow/py/brevitas/xlayer/quantOperation.py`
- Test: `deepsocflow/test/py/test_brevitas_quant_operation.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `deepsocflow.py.brevitas.xlayer.quantOperation.QuantResidualAdd(act, sys_bits=None)`, an `nn.Module` with:
  - `.act` — the activation module passed in.
  - `.sys_bits` — passed through, stored as-is.
  - `.source_ib` — `None` initially; set externally by `XBundle.call()` in Task 3.
  - `forward(self, x, x_add)` → `self.act(x + x_add)` (also reachable as `self(x, x_add)`).
  - `call_int(self, x, hw)` → raises `NotImplementedError`.

- [ ] **Step 1: Write the failing test**

Create `deepsocflow/test/py/test_brevitas_quant_operation.py`:

```python
import pytest
import torch

from deepsocflow.py.brevitas.xlayer.quantActivation import QuantReLU
from deepsocflow.py.brevitas.xlayer.quantOperation import QuantResidualAdd


def test_residual_add_sums_then_applies_activation():
    add = QuantResidualAdd(act=QuantReLU())
    x = torch.tensor([[-1.0, 2.0]])
    x_add = torch.tensor([[3.0, -5.0]])

    out = add(x, x_add)

    assert torch.allclose(out, torch.relu(x + x_add), atol=1e-2)


def test_residual_add_requires_an_activation():
    with pytest.raises(ValueError):
        QuantResidualAdd(act=None)


def test_residual_add_source_ib_defaults_to_none():
    add = QuantResidualAdd(act=QuantReLU())
    assert add.source_ib is None


def test_residual_add_call_int_not_implemented():
    add = QuantResidualAdd(act=QuantReLU())
    with pytest.raises(NotImplementedError):
        add.call_int(x=None, hw=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_quant_operation.py -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'QuantResidualAdd' from 'deepsocflow.py.brevitas.xlayer.quantOperation'`

- [ ] **Step 3: Implement `QuantResidualAdd`**

Replace the contents of `deepsocflow/py/brevitas/xlayer/quantOperation.py`:

```python
import torch.nn as nn


class QuantResidualAdd(nn.Module):
    def __init__(self, act, sys_bits=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if act is None:
            raise ValueError("Activation function must be provided. Set type to none if no activation is needed")
        self.act = act
        self.sys_bits = sys_bits
        self.source_ib = None  # bundle index (ib) of the residual/skip-connection source

    def forward(self, x, x_add):
        return self.act(x + x_add)

    def call_int(self, x, hw):
        raise NotImplementedError(
            "QuantResidualAdd.call_int requires an XTensor port (add_val_shift) for the brevitas backend"
        )


'''
- Transpose
- Concatenate
'''
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_quant_operation.py -v`
Expected: 4 passed

- [ ] **Step 5: Review checkpoint**

Do not commit. Leave changes for the user to review and commit themselves.

---

### Task 3: `XBundle.call()`

**Files:**
- Modify: `deepsocflow/py/brevitas/xbundle/xbundle.py`
- Test: `deepsocflow/test/py/test_brevitas_xbundle.py`

**Interfaces:**
- Consumes:
  - `deepsocflow.py.brevitas.utils.BUNDLES` (Task 1) — module-level list.
  - `deepsocflow.py.brevitas.xlayer.quantOperation.QuantResidualAdd(act, sys_bits=None)` (Task 2).
- Produces: `deepsocflow.py.brevitas.xbundle.xbundle.XBundle(core, pool=None, add_act=None, flatten=False, softmax=False)`, an `nn.Module` with:
  - `.core`, `.pool` — passed through as-is (caller is responsible for `core.act` / `pool.act` already being set — out of scope here).
  - `.add` — `QuantResidualAdd(act=add_act)` if `add_act` given, else `None`.
  - `.flatten` — `nn.Flatten()` if `flatten=True`, else `None`.
  - `.softmax` — `nn.Softmax(dim=-1)` if `softmax=True`, else `None`.
  - `.ib`, `.prev_ib` — `int` or `None`.
  - `.next_ibs`, `.next_add_ibs` — `list[int]`.
  - `call(self, x, x_add=None)` → `torch.Tensor`, also reachable as `self(x)` / `self(x, x_add)` via `forward = call`.

- [ ] **Step 1: Write the failing test**

Create `deepsocflow/test/py/test_brevitas_xbundle.py`:

```python
import pytest
import torch

from deepsocflow.py.brevitas.utils import BUNDLES
from deepsocflow.py.brevitas.xbundle.xbundle import XBundle
from deepsocflow.py.brevitas.xlayer.quantActivation import QuantReLU
from deepsocflow.py.brevitas.xlayer.quantLayer import QuantConv2d


def _make_core(in_ch=3, out_ch=8, kernel_size=3, padding=1):
    core = QuantConv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
    core.act = QuantReLU()
    return core


def test_call_runs_core_and_act_and_registers_in_bundles():
    BUNDLES.clear()
    bundle = XBundle(core=_make_core())
    x = torch.randn(2, 3, 16, 16)

    out = bundle.call(x)

    assert out.shape == (2, 8, 16, 16)
    assert bundle.ib == 0
    assert BUNDLES == [bundle]
    assert out.ib == 0


def test_forward_alias_matches_call():
    BUNDLES.clear()
    bundle = XBundle(core=_make_core())
    x = torch.randn(2, 3, 16, 16)

    out = bundle(x)

    assert out.shape == (2, 8, 16, 16)
    assert bundle.ib == 0


def test_call_tracks_prev_ib_and_next_ibs_across_bundles():
    BUNDLES.clear()
    first = XBundle(core=_make_core())
    second = XBundle(core=_make_core(in_ch=8, out_ch=8))
    x = torch.randn(2, 3, 16, 16)

    mid = first.call(x)
    out = second.call(mid)

    assert second.prev_ib == first.ib
    assert first.next_ibs == [second.ib]
    assert out.ib == second.ib


def test_call_applies_residual_add_and_records_add_edges():
    BUNDLES.clear()
    src = XBundle(core=_make_core())
    dst = XBundle(core=_make_core(in_ch=8, out_ch=8), add_act=QuantReLU())
    x = torch.randn(2, 3, 16, 16)

    src_out = src.call(x)
    out = dst.call(src_out, x_add=src_out)

    assert dst.add.source_ib == src.ib
    assert src.next_add_ibs == [dst.ib]
    assert out.shape == src_out.shape


def test_call_raises_assertion_if_x_add_given_without_add_configured():
    BUNDLES.clear()
    src = XBundle(core=_make_core())
    other = XBundle(core=_make_core())
    x = torch.randn(2, 3, 16, 16)
    src_out = src.call(x)

    with pytest.raises(AssertionError):
        other.call(x, x_add=src_out)


def test_call_raises_value_error_if_add_configured_without_x_add():
    BUNDLES.clear()
    bundle = XBundle(core=_make_core(), add_act=QuantReLU())
    x = torch.randn(2, 3, 16, 16)

    with pytest.raises(ValueError):
        bundle.call(x)


def test_call_applies_flatten_and_softmax():
    BUNDLES.clear()
    core = QuantConv2d(3, 4, kernel_size=16, bias=True)
    core.act = QuantReLU()
    bundle = XBundle(core=core, flatten=True, softmax=True)
    x = torch.randn(2, 3, 16, 16)

    out = bundle.call(x)

    assert out.shape == (2, 4)
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_xbundle.py -v`
Expected: FAIL — `TypeError: XBundle.__init__() got an unexpected keyword argument 'core'` (current stub is `def __init__(self): pass`).

- [ ] **Step 3: Implement `XBundle`**

Replace the contents of `deepsocflow/py/brevitas/xbundle/xbundle.py`:

```python
import torch.nn as nn

from deepsocflow.py.brevitas.utils import BUNDLES
from deepsocflow.py.brevitas.xlayer.quantOperation import QuantResidualAdd


class XBundle(nn.Module):

    def __init__(self, core, pool=None, add_act=None, flatten=False, softmax=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core = core
        self.pool = pool

        self.add = QuantResidualAdd(act=add_act) if add_act else None
        self.flatten = nn.Flatten() if flatten else None
        self.softmax = nn.Softmax(dim=-1) if softmax else None

        self.ib = None            # this bundle's index in the global BUNDLES list
        self.prev_ib = None       # ib of the bundle whose output feeds this bundle's main input
        self.next_ibs = []        # ibs of bundles that consume this bundle's main output
        self.next_add_ibs = []    # ibs of bundles that consume this bundle's output via a residual/skip add

    def call(self, x, x_add=None):  # x_add: residual/skip-connection tensor to add, if any

        self.ib = len(BUNDLES)
        BUNDLES.append(self)

        if hasattr(x, "ib"):
            self.prev_ib = x.ib
            BUNDLES[self.prev_ib].next_ibs += [self.ib]

        x = self.core(x)
        x = self.core.act(x)

        if x_add is not None:
            assert self.add is not None, "Activation function must be provided for add layer"
            self.add.source_ib = x_add.ib
            BUNDLES[x_add.ib].next_add_ibs += [self.ib]
            x = self.add(x, x_add)
        elif self.add is not None:
            raise ValueError("A Bundle initialized with add_act, should have the add tensor passed")

        if self.pool:
            x = self.pool(x)
            x = self.pool.act(x)

        if self.flatten:
            x = self.flatten(x)

        if self.softmax:
            x = self.softmax(x)

        x.ib = self.ib
        return x

    forward = call

    def call_int(self, x, hw):  # x: XTensor input (only used for the first/ib==0 bundle), hw: Hardware config
        raise NotImplementedError(
            "XBundle.call_int requires an XTensor + hardware.py port for the brevitas backend; "
            "it will run the fixed-point simulation and assert parity against call()"
        )

    def export(self, hw, is_last):  # hw: Hardware config, is_last: True if this is the final bundle in the network
        raise NotImplementedError(
            "XBundle.export requires the hardware-engine-layout reorder helpers "
            "(reorder_*_q2e_conv) ported to the brevitas backend"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_xbundle.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the full new test suite together**

Run: `python -m pytest deepsocflow/test/py/test_brevitas_utils.py deepsocflow/test/py/test_brevitas_quant_operation.py deepsocflow/test/py/test_brevitas_xbundle.py -v`
Expected: 13 passed

- [ ] **Step 6: Review checkpoint**

Do not commit. Leave all changes for the user to review and commit themselves.

---

## Self-Review Notes

- **Spec coverage:** `XBundle` construction ✓ (Task 3), `call()` ✓ (Task 3), `call_int`/`export` stubs ✓ (Task 3), `BUNDLES` util ✓ (Task 1), `QuantResidualAdd` ✓ (Task 2). Out-of-scope items from the spec (`call_int`/`export` real implementations, attaching `.act` onto core/pool, `QuantResidualAdd.call_int`) are explicitly left unimplemented/stubbed, matching the spec.
- **Placeholder scan:** no TBD/TODO; every step has runnable code, verified by hand against a throwaway script during planning (Brevitas 0.12.1 / torch 2.12.1) — all listed test scenarios pass as written.
- **Type consistency:** `QuantResidualAdd(act, sys_bits=None)` used consistently in Task 2 and Task 3; `XBundle.call(self, x, x_add=None)` signature and `forward = call` alias consistent throughout; `BUNDLES` imported the same way (`from deepsocflow.py.brevitas.utils import BUNDLES`) in Tasks 2 test file (not needed there) and Task 3.
