# `Model.load_fp32_weights()` — loading pretrained fp32 weights into a brevitas `Model`

## Context

`deepsocflow/py/brevitas/utils.py`'s `Model` builds a brevitas-native
quantized architecture directly from a JSON spec (`Model._build_from_json`),
using `Quant*` layers (`QuantConv2dBN`, `QuantLinear`, `QuantReLU`, ...) from
first construction — there is no plain-fp32 intermediate stage.

The `Known Issues` note in `CLAUDE.md` documents that loading real
torchvision ResNet18 ImageNet weights into `Model(json_path=".../resnet18.json")`
produces near-random output (cosine similarity to the fp32 reference ~0.004),
and that running PTQ calibration afterward makes it categorically worse
(degenerate one-hot output). Two separate problems are tangled together in
that symptom:

1. **Getting fp32 pretrained weights into the `Quant*` layers at all** is
   currently unsupported — there's no method for it, and `Model`'s parameter
   names (e.g. `layers.layer1.conv.weight`) don't match a source like
   torchvision's (e.g. `conv1.weight`). Any naive positional or shape-only
   auto-matching risks assigning a tensor to the wrong parameter silently
   (two conv layers with the same channel count and kernel size produce
   identical shapes but are semantically different).
2. **Calibration itself** (`brevitas.graph.calibrate.calibration_mode`)
   appears not to work as expected on this model.

This design addresses only (1). Fixing (2) is scoped to a separate,
follow-up design.

## Goal

Add a method to `Model` that loads weights from an external fp32
`state_dict` into its `Quant*` layers, using an explicit, human-verified
name mapping — safe against the silent-mismatch failure mode above — plus a
standalone tool that helps generate a first draft of that mapping so it
doesn't have to be hand-typed from scratch.

## Design

### `Model.load_fp32_weights(self, state_dict, name_map)`

```python
def load_fp32_weights(self, state_dict: dict, name_map: dict) -> None:
```

- `state_dict`: a raw fp32 source state dict, e.g.
  `torchvision.models.resnet18(weights=...).state_dict()`. Not assumed to
  share any naming convention with `Model`.
- `name_map`: `{our_name: source_key}` — `our_name` keys into
  `dict(self.named_parameters()) | dict(self.named_buffers())` (buffers are
  included so BatchNorm's `running_mean`/`running_var` are covered, not just
  learnable weight/bias — omitting them would leave BatchNorm statistics at
  their randomly-initialized defaults after "loading pretrained weights,"
  which is itself enough to produce badly wrong inference). `source_key`
  indexes into `state_dict`.
- This mapping is produced once per (model, source) pair, reviewed by a
  human, and checked into the repo next to the model definition, e.g.
  `deepsocflow/py/brevitas/models/resnet18/resnet18_torchvision_weight_map.json`.

Behavior:

1. Build `targets = dict(self.named_parameters()) | dict(self.named_buffers())`.
2. Validate every entry in `name_map`:
   - `source_key` must exist in `state_dict`.
   - `our_name` must exist in `targets`.
   - `state_dict[source_key].shape` must equal `targets[our_name].shape`.
3. Collect **all** validation failures (don't stop at the first) and, if
   any exist, raise a single `ValueError` listing every failing entry. A
   real mapping for a model like ResNet18 has on the order of 60 entries;
   surfacing every problem in one pass avoids a fix-one/rerun/fix-next loop.
4. Only if validation passes completely: copy every tensor,
   `targets[our_name].copy_(state_dict[source_key])`, under
   `torch.no_grad()`.
5. `name_map` does not need to cover every parameter/buffer in `Model` —
   coverage is partial by design (e.g. a newly-added classifier head can be
   left at its random initialization while a pretrained backbone is loaded).

### Weight-map generator — `deepsocflow/py/brevitas/tools/suggest_weight_map.py`

A standalone script, not a method on `Model` — it's a one-time authoring
aid, not something `Model` needs at runtime, and keeping it separate avoids
coupling `Model`'s public API to any particular source model's structure.

- Input: a `Model` instance, and an fp32 "twin" `nn.Module` whose op
  sequence corresponds to `Model`'s (e.g. `torchvision.models.resnet18()`
  itself, for the ResNet18 case).
- Algorithm: walk both models' submodules in declaration order; pair them
  positionally by op-type — `nn.Conv2d` ↔ `QuantConv2dBN.conv`,
  `nn.BatchNorm2d` ↔ `QuantConv2dBN.bn`, `nn.Linear` ↔ `QuantLinear` — and
  confirm shape agreement at each pair. Mismatched op-type sequences abort
  with an error identifying where the two models' structures diverge,
  rather than guessing.
- Output: a candidate `{our_name: source_key}` JSON file.
- This output is a **draft**, not consumed directly by `load_fp32_weights`.
  A human reviews/edits it, then the verified result is what actually gets
  checked into the repo and passed as `name_map`.

## Out of scope

- Fixing `brevitas.graph.calibrate.calibration_mode` / PTQ or QAT
  correctness (`Model.calibrate()` / `Model.qat()`) — separate design.
- Automatically applying the generator's candidate mapping without human
  review.
- Support for source state dicts with no reasonable fp32 twin model to
  generate a candidate mapping from (in that case, `name_map` is written by
  hand directly).

## Testing

- Unit tests for `load_fp32_weights` using small synthetic modules/state
  dicts (not real pretrained weights): correct mapping copies values
  correctly; a `name_map` with a bad shape, missing source key, or unknown
  target name raises `ValueError` listing all such problems and copies
  nothing.
- Manual check for the generator script against the real ResNet18 case:
  run it with `Model(json_path=".../resnet18.json")` and
  `torchvision.models.resnet18()`, confirm the candidate mapping's op
  pairing looks correct on inspection.
