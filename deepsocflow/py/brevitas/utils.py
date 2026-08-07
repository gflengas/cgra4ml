import json

import torch
import torch.nn as nn

from deepsocflow.py.brevitas.xlayer.quantActivation import QuantIdentity, QuantReLU
from deepsocflow.py.brevitas.xlayer.quantLayer import QuantConv2d, QuantConv2dBN, QuantLinear
from deepsocflow.py.brevitas.xlayer.quantOperation import QuantResidualAdd
from deepsocflow.py.brevitas.xlayer.quantPooling import QuantAdaptiveAvgPool2d, QuantMaxPool2d

# Every XBundle instance registers itself here (mirrors deepsocflow/py/utils.py's
# BUNDLES) so bundle-to-bundle graph edges (ib/prev_ib/next_ibs/next_add_ibs) can be
# tracked across a forward pass.
BUNDLES = []


def reset_bundles():
    BUNDLES.clear()


# Layer types whose builder requires a QuantTensor input (not a plain Tensor) - see
# quantPooling.py's QuantAdaptiveAvgPool2d/QuantAvgPool2d comments for why. When such a
# layer's producer is built, its activation is forced to return_quant_tensor=True so the
# tensor arriving here carries the scale/bit-width metadata truncation needs.
TYPES_REQUIRING_QUANT_TENSOR_INPUT = {'adaptiveavgpool2d'}


def _build_activation(name, return_quant_tensor=False):
    if name is None:
        return QuantIdentity(return_quant_tensor=return_quant_tensor)
    name = name.lower()
    if name == 'relu':
        return QuantReLU(return_quant_tensor=return_quant_tensor)
    raise ValueError(f"Unsupported activation type: '{name}'")


def _build_conv2d(cfg, return_quant_tensor=False):
    act = _build_activation(cfg.get('activation'), return_quant_tensor)

    if cfg.get('batch_norm', True):
        return QuantConv2dBN(
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            kernel_size=cfg['kernel_size'],
            stride=cfg.get('stride', 1),
            padding=cfg.get('padding', 0),
            act=act)

    conv = QuantConv2d(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        kernel_size=cfg['kernel_size'],
        stride=cfg.get('stride', 1),
        padding=cfg.get('padding', 0),
        bias=cfg.get('bias', True))
    conv.act = act
    return conv


def _build_maxpool2d(cfg, return_quant_tensor=False):
    return QuantMaxPool2d(
        kernel_size=cfg['kernel_size'],
        stride=cfg.get('stride'),
        padding=cfg.get('padding', 0))


def _build_adaptiveavgpool2d(cfg, return_quant_tensor=False):
    return QuantAdaptiveAvgPool2d(output_size=tuple(cfg['output_size']))


def _build_flatten(cfg, return_quant_tensor=False):
    return nn.Flatten()


def _build_linear(cfg, return_quant_tensor=False):
    return QuantLinear(
        in_features=cfg['in_features'],
        out_features=cfg['out_features'],
        bias=cfg.get('bias', True))


def _build_relu(cfg, return_quant_tensor=False):
    return QuantReLU(return_quant_tensor=return_quant_tensor)


def _build_softmax(cfg, return_quant_tensor=False):
    return nn.Softmax(dim=-1)


def _build_residual_add(cfg, return_quant_tensor=False):
    return QuantResidualAdd(act=_build_activation(cfg.get('activation'), return_quant_tensor))


class Model(nn.Module):

    # Registry of supported layer types (checked in lowercase) -> builder function.
    # Each builder takes a layer's config dict (and whether its output must carry
    # QuantTensor metadata) and returns an nn.Module.
    LAYER_BUILDERS = {
        'conv2d': _build_conv2d,
        'maxpool2d': _build_maxpool2d,
        'adaptiveavgpool2d': _build_adaptiveavgpool2d,
        'flatten': _build_flatten,
        'linear': _build_linear,
        'relu': _build_relu,
        'softmax': _build_softmax,
        'residual_add': _build_residual_add,
    }

    def __init__(self, json_path=None, xlayer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_configs = {}
        self.layer_order = []
        self.layers = nn.ModuleDict()

        if json_path is not None:
            self._build_from_json(json_path)
        elif xlayer is not None:
            self._build_from_xlayer(xlayer)

    def _build_from_json(self, json_path):
        with open(json_path) as f:
            spec = json.load(f)

        self.layer_configs = spec['layers']
        self.layer_order = list(self.layer_configs.keys())

        # Find every layer name that directly feeds a layer type requiring QuantTensor
        # input, so its activation can be built with return_quant_tensor=True.
        quant_tensor_producers = set()
        for i, name in enumerate(self.layer_order):
            cfg = self.layer_configs[name]
            if cfg['type'].lower() in TYPES_REQUIRING_QUANT_TENSOR_INPUT:
                default_input = self.layer_order[i - 1] if i > 0 else None
                quant_tensor_producers.add(cfg.get('input', default_input))

        for name, cfg in self.layer_configs.items():
            layer_type = cfg['type'].lower()
            if layer_type not in self.LAYER_BUILDERS:
                raise ValueError(
                    f"Unknown layer type '{layer_type}' for layer '{name}' in {json_path}. "
                    f"Supported types: {sorted(self.LAYER_BUILDERS)}")
            self.layers[name] = self.LAYER_BUILDERS[layer_type](
                cfg, return_quant_tensor=name in quant_tensor_producers)

    def _build_from_xlayer(self, xlayer):
        raise NotImplementedError("Model._build_from_xlayer is not implemented yet")

    def _default_input(self, i):
        return self.layer_order[i - 1] if i > 0 else None

    def forward(self, x):
        outputs = {}
        self.output_shapes = {}

        for i, name in enumerate(self.layer_order):
            cfg = self.layer_configs[name]
            layer = self.layers[name]
            layer_type = cfg['type'].lower()

            if layer_type == 'residual_add':
                out = layer(outputs[cfg['input']], outputs[cfg['skip_from']])
            else:
                input_name = cfg.get('input', self._default_input(i))
                inp = outputs[input_name] if input_name is not None else x
                out = layer(inp)

            outputs[name] = out
            self.output_shapes[name] = tuple(out.shape)

        return outputs[self.layer_order[-1]]

    def print_graph(self):
        # Prints the built model as a table: layer name, type, source layer(s), and
        # output shape (populated the last time forward() ran - run a forward pass
        # first, e.g. with a dummy input, if output shapes should be shown).
        rows = []
        for i, name in enumerate(self.layer_order):
            cfg = self.layer_configs[name]
            layer_type = cfg['type']

            if layer_type.lower() == 'residual_add':
                source = f"{cfg['input']} + {cfg['skip_from']}"
            else:
                source = cfg.get('input', self._default_input(i)) or 'x (model input)'

            shape = self.output_shapes.get(name, '?') if hasattr(self, 'output_shapes') else '?'
            rows.append((name, layer_type, source, str(shape)))

        widths = [max(len(r[c]) for r in rows + [('layer', 'type', 'source', 'output_shape')]) for c in range(4)]
        header = ('layer', 'type', 'source', 'output_shape')
        for row in [header] + rows:
            print('  '.join(cell.ljust(w) for cell, w in zip(row, widths)))

    # Loads an fp32 state_dict into this Model's own params/buffers via an explicit
    # name_map. name_map direction: {our_name: source_key} - keys index into this
    # Model's own named_parameters()/named_buffers(), values index into the
    # caller-supplied state_dict. Coverage is allowed to be partial - not every
    # parameter/buffer needs an entry. All validation problems (missing source keys,
    # unknown target names, shape mismatches) are collected and raised together as one
    # ValueError; nothing is copied unless every entry validates. .copy_() follows
    # standard PyTorch dtype-coercion semantics - if a source tensor's dtype doesn't
    # match the target's, values are silently cast (e.g. a float source into an int64
    # buffer truncates). This method doesn't validate dtype, only shape.
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


# QuantConv1d/2d/3d and QuantLinear genuinely subclass torch.nn.Conv1d/2d/3d and
# torch.nn.Linear respectively (verified against the installed brevitas version via
# issubclass()), so isinstance() checks below work uniformly on a Model's own layers -
# no brevitas-specific type needed here.
_MATCHABLE_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Linear)


def _local_state(module):
    state = dict(module.named_parameters(recurse=False))
    state.update(module.named_buffers(recurse=False))
    return state


def _matchable_entries(model):
    entries = []
    for module_name, module in model.named_modules():
        if not isinstance(module, _MATCHABLE_TYPES):
            continue
        for local_name, tensor in _local_state(module).items():
            entries.append((f"{module_name}.{local_name}", tensor))
    return entries


# Pairs a Model's own Conv*d/BatchNorm*d/Linear parameters/buffers against a raw fp32
# source state_dict (e.g. torch.load("resnet18.pth"), or
# torchvision.models.resnet18().state_dict()) by declaration order - no twin nn.Module
# needs to be instantiated, any checkpoint's state_dict works as long as its parameter
# order matches model's. Output is a draft for a human to review before saving/using as
# a real name_map for Model.load_fp32_weights - never meant to be consumed directly
# without review. Raises one ValueError collecting every shape mismatch found, rather
# than stopping at the first.
def suggest_weight_map(model, source_state_dict):
    model_entries = _matchable_entries(model)
    source_entries = list(source_state_dict.items())

    if len(model_entries) != len(source_entries):
        raise ValueError(
            f"model has {len(model_entries)} matchable parameter(s)/buffer(s) but source "
            f"state_dict has {len(source_entries)}; cannot pair positionally")

    errors = []
    name_map = {}
    for i, ((our_name, our_tensor), (source_key, source_tensor)) in enumerate(zip(model_entries, source_entries)):
        if our_tensor.shape != source_tensor.shape:
            errors.append(
                f"shape mismatch at position {i}: '{our_name}' is {tuple(our_tensor.shape)}, "
                f"'{source_key}' is {tuple(source_tensor.shape)}")
            continue
        name_map[our_name] = source_key

    if errors:
        raise ValueError(f"suggest_weight_map found {len(errors)} problem(s):\n" + "\n".join(errors))

    return name_map
