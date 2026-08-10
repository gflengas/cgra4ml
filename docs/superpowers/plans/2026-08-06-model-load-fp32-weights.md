# Model.load_fp32_weights() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `deepsocflow/py/brevitas/utils.py`'s `Model` a safe way to load an external fp32 `state_dict` (e.g. torchvision pretrained weights) into its brevitas `Quant*` layers, via an explicit human-verified name map — plus a standalone tool that drafts that map instead of requiring it to be hand-typed from scratch.

**Architecture:** `Model.load_fp32_weights(state_dict, name_map)` validates every `{our_name: source_key}` entry in `name_map` against `dict(self.named_parameters()) | dict(self.named_buffers())` (source key exists, our name exists, shapes match), collects every failure into one `ValueError` if any exist, and only copies tensors if validation fully passes. A separate script, `deepsocflow/py/brevitas/tools/suggest_weight_map.py`, exposes `suggest_weight_map(model, twin)`, which pairs `model`'s and `twin`'s leaf `Conv*d`/`BatchNorm*d`/`Linear` submodules positionally (this works because brevitas's `QuantConv2d`/`QuantLinear` genuinely subclass `torch.nn.Conv2d`/`torch.nn.Linear` — verified via `issubclass()` against the installed brevitas version) and emits a candidate mapping dict for human review before it's saved as a real, checked-in weight map.

**Tech Stack:** Python 3.11.5, PyTorch + Brevitas (already used throughout `deepsocflow/py/brevitas/`), pytest 7.4.0 (already a project dependency).

## Global Constraints

- Only work on the `brevitas-qonnx-backend` branch (per project `CLAUDE.md`).
- **Never run `git commit` or `git push`.** Every task below ends with a review checkpoint instead of a commit step — stage nothing, commit nothing. The user commits themselves.
- `load_fp32_weights` must collect **all** validation problems and raise **one** `ValueError` listing every one of them — never fail-fast on the first bad entry.
- `load_fp32_weights` must cover both `self.named_parameters()` **and** `self.named_buffers()` — BatchNorm's `running_mean`/`running_var` are buffers, not parameters, and must be loadable too.
- `name_map` coverage is allowed to be partial — `load_fp32_weights` must not require every parameter/buffer in `Model` to appear in the map.
- `suggest_weight_map` is a standalone function in its own module, not a method on `Model` — it's a one-time authoring aid, not part of `Model`'s runtime API.
- `torchvision` is not a declared dependency in `pyproject.toml`. Any import of it must be local (inside a function body / `if __name__ == "__main__":` block), never a top-level module import, so the rest of the module stays importable without it.

Spec: `docs/superpowers/specs/2026-08-06-model-load-fp32-weights-design.md`

---

## File Structure

- Modify: `deepsocflow/py/brevitas/utils.py` — add `import torch` and a `load_fp32_weights` method on `Model`.
- Create: `deepsocflow/test/py/test_model_load_fp32_weights.py` — tests for `load_fp32_weights`.
  **Note:** `deepsocflow/test/py/test_brevitas_utils.py` already exists but currently fails to even
  import (`ImportError: cannot import name 'BUNDLES' from 'deepsocflow.py.brevitas.utils'` — `BUNDLES`/
  `reset_bundles` are referenced by that file, `test_brevitas_xbundle.py`, `xbundle.py`, and `resnet18.py`,
  but are missing from the current uncommitted `utils.py`). This is a pre-existing problem, unrelated to
  `load_fp32_weights`, and out of scope for this plan — do **not** fix it, and do not add anything to
  `test_brevitas_utils.py`, since any test appended there would fail to collect for this unrelated reason.
  Use the new file above instead.
- Create: `deepsocflow/py/brevitas/tools/__init__.py` — empty, makes `deepsocflow.py.brevitas.tools` an importable package (matches the `xlayer/__init__.py`, `xbundle/__init__.py` convention already in this codebase).
- Create: `deepsocflow/py/brevitas/tools/suggest_weight_map.py` — `suggest_weight_map(model, twin)` plus a `if __name__ == "__main__":` block wired to the real ResNet18 case.
- Create: `deepsocflow/test/py/test_suggest_weight_map.py`

---

### Task 1: `Model.load_fp32_weights()`

**Files:**
- Modify: `deepsocflow/py/brevitas/utils.py`
- Test: `deepsocflow/test/py/test_model_load_fp32_weights.py`

**Interfaces:**
- Consumes: nothing new — uses `Model`'s existing `self.layers` (an `nn.ModuleDict`, already set up by `Model.__init__`) via the standard `nn.Module.named_parameters()`/`named_buffers()`.
- Produces: `Model.load_fp32_weights(self, state_dict: dict, name_map: dict) -> None`. Task 2's `suggest_weight_map()` produces dicts in the exact shape this method expects for `name_map` (`{our_name: source_key}`), but there is no code-level dependency between the two tasks.

**Important:** `deepsocflow/test/py/test_brevitas_utils.py` already exists but currently fails to
import at all (`ImportError: cannot import name 'BUNDLES' from 'deepsocflow.py.brevitas.utils'`).
This is a pre-existing, unrelated problem — do not fix it and do not add anything to that file.
Put the new tests below in a brand new file, `deepsocflow/test/py/test_model_load_fp32_weights.py`,
so they're unaffected by that unrelated breakage.

- [ ] **Step 1: Write the failing tests**

Create `deepsocflow/test/py/test_model_load_fp32_weights.py`:

```python
import pytest
import torch
import torch.nn as nn

from deepsocflow.py.brevitas.utils import Model


def test_load_fp32_weights_copies_params_and_buffers():
    model = Model()
    model.layers["stem"] = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=1, bias=False),
        nn.BatchNorm2d(2),
    )
    source = {
        "conv.weight": torch.full((2, 1, 1, 1), 5.0),
        "bn.weight": torch.tensor([1.0, 2.0]),
        "bn.bias": torch.tensor([0.1, 0.2]),
        "bn.running_mean": torch.tensor([0.5, 0.6]),
        "bn.running_var": torch.tensor([1.5, 1.6]),
    }
    name_map = {
        "layers.stem.0.weight": "conv.weight",
        "layers.stem.1.weight": "bn.weight",
        "layers.stem.1.bias": "bn.bias",
        "layers.stem.1.running_mean": "bn.running_mean",
        "layers.stem.1.running_var": "bn.running_var",
    }

    model.load_fp32_weights(source, name_map)

    conv, bn = model.layers["stem"][0], model.layers["stem"][1]
    assert torch.equal(conv.weight, source["conv.weight"])
    assert torch.equal(bn.weight, source["bn.weight"])
    assert torch.equal(bn.bias, source["bn.bias"])
    assert torch.equal(bn.running_mean, source["bn.running_mean"])
    assert torch.equal(bn.running_var, source["bn.running_var"])


def test_load_fp32_weights_allows_partial_mapping():
    model = Model()
    model.layers["stem"] = nn.Linear(2, 3, bias=True)
    original_bias = model.layers["stem"].bias.clone()
    source = {"weight": torch.randn(3, 2)}
    name_map = {"layers.stem.weight": "weight"}  # bias intentionally left unmapped

    model.load_fp32_weights(source, name_map)

    assert torch.equal(model.layers["stem"].weight, source["weight"])
    assert torch.equal(model.layers["stem"].bias, original_bias)


def test_load_fp32_weights_raises_on_missing_source_key():
    model = Model()
    model.layers["stem"] = nn.Linear(2, 3)
    source = {"weight": torch.randn(3, 2)}
    name_map = {"layers.stem.weight": "weight", "layers.stem.bias": "missing_bias"}

    with pytest.raises(ValueError, match="missing_bias"):
        model.load_fp32_weights(source, name_map)


def test_load_fp32_weights_raises_on_unknown_target_name():
    model = Model()
    model.layers["stem"] = nn.Linear(2, 3)
    source = {"weight": torch.randn(3, 2)}
    name_map = {"layers.stem.nonexistent": "weight"}

    with pytest.raises(ValueError, match="nonexistent"):
        model.load_fp32_weights(source, name_map)


def test_load_fp32_weights_raises_on_shape_mismatch():
    model = Model()
    model.layers["stem"] = nn.Linear(2, 3)
    source = {"weight": torch.randn(4, 2)}
    name_map = {"layers.stem.weight": "weight"}

    with pytest.raises(ValueError, match="shape"):
        model.load_fp32_weights(source, name_map)


def test_load_fp32_weights_reports_all_errors_at_once():
    model = Model()
    model.layers["stem"] = nn.Linear(2, 3)
    source = {"weight": torch.randn(4, 2)}
    name_map = {
        "layers.stem.weight": "weight",       # shape mismatch
        "layers.stem.bias": "missing_bias",   # missing source key
        "layers.stem.nope": "weight",         # unknown target name
    }

    with pytest.raises(ValueError) as exc_info:
        model.load_fp32_weights(source, name_map)

    message = str(exc_info.value)
    assert "shape" in message
    assert "missing_bias" in message
    assert "nope" in message


def test_load_fp32_weights_copies_nothing_if_validation_fails():
    model = Model()
    model.layers["stem"] = nn.Linear(2, 3)
    original_weight = model.layers["stem"].weight.clone()
    source = {"weight": torch.randn(4, 2), "bias": torch.randn(3)}  # weight shape wrong
    name_map = {"layers.stem.weight": "weight", "layers.stem.bias": "bias"}

    with pytest.raises(ValueError):
        model.load_fp32_weights(source, name_map)

    assert torch.equal(model.layers["stem"].weight, original_weight)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest deepsocflow/test/py/test_model_load_fp32_weights.py -v`
Expected: all 7 tests FAIL with `AttributeError: 'Model' object has no attribute 'load_fp32_weights'`.

- [ ] **Step 3: Implement `load_fp32_weights`**

In `deepsocflow/py/brevitas/utils.py`, add `import torch` to the top-level imports (currently only `import json` and `import torch.nn as nn` are present):

```python
import json

import torch
import torch.nn as nn
```

Then add this method to the `Model` class, after `print_graph`:

```python
    def load_fp32_weights(self, state_dict, name_map):
        targets = dict(self.named_parameters())
        targets.update(dict(self.named_buffers()))

        errors = []
        for our_name, source_key in name_map.items():
            if source_key not in state_dict:
                errors.append(f"missing source key: '{source_key}' (mapped from '{our_name}')")
                continue
            if our_name not in targets:
                errors.append(f"unknown parameter/buffer: '{our_name}'")
                continue
            source_tensor = state_dict[source_key]
            target_tensor = targets[our_name]
            if source_tensor.shape != target_tensor.shape:
                errors.append(
                    f"shape mismatch: '{our_name}' is {tuple(target_tensor.shape)}, "
                    f"'{source_key}' is {tuple(source_tensor.shape)}")

        if errors:
            raise ValueError(f"load_fp32_weights found {len(errors)} problem(s):\n" + "\n".join(errors))

        with torch.no_grad():
            for our_name, source_key in name_map.items():
                targets[our_name].copy_(state_dict[source_key])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest deepsocflow/test/py/test_model_load_fp32_weights.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Stop for review**

Do not commit. Report the diff and test output back for review (per `CLAUDE.md`, the user handles all commits).

---

### Task 2: `suggest_weight_map()` mapping generator

**Files:**
- Create: `deepsocflow/py/brevitas/tools/__init__.py`
- Create: `deepsocflow/py/brevitas/tools/suggest_weight_map.py`
- Test: `deepsocflow/test/py/test_suggest_weight_map.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `suggest_weight_map(model: nn.Module, twin: nn.Module) -> dict` — a `{our_name: source_key}` dict in the exact shape `Model.load_fp32_weights` (Task 1) expects as `name_map`, meant to be reviewed/edited by a human before being saved and used that way.

- [ ] **Step 1: Write the failing tests**

Create `deepsocflow/test/py/test_suggest_weight_map.py`:

```python
import pytest
import torch.nn as nn

from deepsocflow.py.brevitas.tools.suggest_weight_map import suggest_weight_map
from deepsocflow.py.brevitas.xlayer.quantLayer import QuantConv2d, QuantLinear


class _QuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_conv = QuantConv2d(1, 2, kernel_size=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(2)
        self.head = QuantLinear(2, 3)


class _Fp32Twin(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 3)


def test_suggest_weight_map_pairs_by_position_and_type():
    name_map = suggest_weight_map(_QuantModel(), _Fp32Twin())

    assert name_map["stem_conv.weight"] == "conv1.weight"
    assert name_map["stem_bn.weight"] == "bn1.weight"
    assert name_map["stem_bn.bias"] == "bn1.bias"
    assert name_map["stem_bn.running_mean"] == "bn1.running_mean"
    assert name_map["stem_bn.running_var"] == "bn1.running_var"
    assert name_map["head.weight"] == "fc.weight"
    assert name_map["head.bias"] == "fc.bias"


def test_suggest_weight_map_raises_on_type_mismatch():
    class _BadTwin(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1, 2)  # quant model has a conv in this position, not linear
            self.bn1 = nn.BatchNorm2d(2)
            self.fc2 = nn.Linear(2, 3)

    with pytest.raises(ValueError, match="type mismatch"):
        suggest_weight_map(_QuantModel(), _BadTwin())


def test_suggest_weight_map_raises_on_module_count_mismatch():
    class _ShortTwin(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 2, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(2)
            # no Linear layer here - one matchable module short of _QuantModel

    with pytest.raises(ValueError, match="matchable modules"):
        suggest_weight_map(_QuantModel(), _ShortTwin())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest deepsocflow/test/py/test_suggest_weight_map.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'deepsocflow.py.brevitas.tools'`.

- [ ] **Step 3: Create the package file**

Create `deepsocflow/py/brevitas/tools/__init__.py` (empty file).

- [ ] **Step 4: Implement `suggest_weight_map`**

Create `deepsocflow/py/brevitas/tools/suggest_weight_map.py`:

```python
import torch.nn as nn

# QuantConv1d/2d/3d and QuantLinear genuinely subclass torch.nn.Conv1d/2d/3d and
# torch.nn.Linear respectively (verified against the installed brevitas version via
# issubclass()), so isinstance() checks below work uniformly across a plain fp32
# twin model and a brevitas-quantized Model - no brevitas-specific type needed here.
_MATCHABLE_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Linear)


def _local_state(module):
    state = dict(module.named_parameters(recurse=False))
    state.update(module.named_buffers(recurse=False))
    return state


def suggest_weight_map(model, twin):
    model_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, _MATCHABLE_TYPES)]
    twin_modules = [(name, m) for name, m in twin.named_modules() if isinstance(m, _MATCHABLE_TYPES)]

    if len(model_modules) != len(twin_modules):
        raise ValueError(
            f"model has {len(model_modules)} matchable modules but twin has {len(twin_modules)}; "
            "cannot pair positionally")

    name_map = {}
    for i, ((model_name, model_mod), (twin_name, twin_mod)) in enumerate(zip(model_modules, twin_modules)):
        if not isinstance(model_mod, type(twin_mod)):
            raise ValueError(
                f"type mismatch at position {i}: model module '{model_name}' is "
                f"{type(model_mod).__name__}, twin module '{twin_name}' is {type(twin_mod).__name__}")

        twin_state = _local_state(twin_mod)
        for local_name in _local_state(model_mod):
            if local_name not in twin_state:
                continue
            name_map[f"{model_name}.{local_name}"] = f"{twin_name}.{local_name}"

    return name_map


if __name__ == "__main__":
    import json
    import os

    import torchvision

    from deepsocflow.py.brevitas.utils import Model

    json_path = os.path.join(os.path.dirname(__file__), "..", "models", "resnet18", "resnet18.json")
    model = Model(json_path=json_path)
    twin = torchvision.models.resnet18()

    candidate = suggest_weight_map(model, twin)

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "resnet18", "resnet18_torchvision_weight_map.candidate.json")
    with open(out_path, "w") as f:
        json.dump(candidate, f, indent=2)

    print(f"Wrote {len(candidate)} candidate mappings to {out_path}")
    print("This is a draft - review it before using it as a real weight map.")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest deepsocflow/test/py/test_suggest_weight_map.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 6 (manual, optional): Sanity-check against real ResNet18**

Run: `python -m deepsocflow.py.brevitas.tools.suggest_weight_map`

This requires `torchvision` installed (already present in the dev environment per `python3 -c "import torchvision"` at plan-writing time, but not a declared `pyproject.toml` dependency — if it's missing, skip this step). Expected: prints `Wrote <N> candidate mappings to .../resnet18_torchvision_weight_map.candidate.json` with no error, and the written JSON's keys/values look like plausible ResNet18 pairings (e.g. `"layers.layer1.conv.weight": "conv1.weight"`, `"layers.layer1.bn.weight": "bn1.weight"`, ...) on manual inspection. This file is a scratch artifact for manual review, not committed.

- [ ] **Step 7: Stop for review**

Do not commit. Report the diff and test output back for review (per `CLAUDE.md`, the user handles all commits).
