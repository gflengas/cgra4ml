import json

import numpy as np

# Activations whose act_quant is unsigned (Uint8ActPerTensorFixedPoint in ptq.py's
# ACT_MAP) - their quantized output range is [0, 2**bits-1], not the signed
# [-2**(bits-1), 2**(bits-1)-1] used everywhere else. export_graph_json doesn't record
# this per tensor yet (see CLAUDE.md Known Issues: "No signed/unsigned flag per
# tensor"), so this mirrors ptq.py's ACT_MAP by activation name until that's fixed.
UNSIGNED_ACTIVATIONS = {'relu', 'sigmoid'}

# Activations this simulator can execute in pure integer arithmetic today - anything
# not piecewise-linear through the origin (silu/tanh/gelu/selu/...) has no
# integer-exact implementation yet (see CLAUDE.md Known Issues).
SUPPORTED_ACTIVATIONS = {'relu', 'identity'}


def shift_round(n, s):
    """Round-to-nearest-even right shift on an int64 array. Mirrors
    deepsocflow/py/utils.py::shift_round and deepsocflow/c/runtime.h's shift_round
    macro exactly - same formula, just on numpy arrays instead of a scalar."""
    n = np.asarray(n, dtype=np.int64)
    if s <= 0:
        return n << (-s)
    half = np.int64(1) << (s - 1)
    return (n + half - (~(n >> s) & 1)) >> s


class FixedPointModel:
    # Reads an already-quantized `quantized_model.export_graph_json()` graph (e.g.
    # deepsocflow/py/brevitas/model/xor_graph.json) and executes it in pure int64
    # arithmetic - matmul + bias-add + shift_round + relu/identity, mirroring
    # deepsocflow/c/runtime.h's quant_lrelu exactly. This is a bit-exact software
    # model of what the C firmware/RTL actually computes - unlike brevitas itself
    # (fake quantization: quantize+dequantize round-trip computed in float32), this
    # never touches a float after construction.
    #
    # Usage:
    #   model = FixedPointModel(json_path)   # 1. build the graph topology
    #   model.load_int_weights(json_path)    # 2. pull the quantized int weights in
    #   x_int = model.quantize_input(x_float)
    #   logits_int = model.forward(x_int)    # pre-softmax; softmax is monotonic,
    #                                         # so argmax(logits_int) == argmax(softmax(logits))

    def __init__(self, json_path):
        self.bundle_order = []
        self.bundles = {}
        self._build_topology(json_path)

    def _build_topology(self, json_path):
        with open(json_path) as f:
            spec = json.load(f)['layers']

        self.bundle_order = list(spec.keys())
        for name in self.bundle_order:
            cfg = spec[name]
            if cfg['type'] != 'linear':
                raise ValueError(
                    f"FixedPointModel only supports type='linear' bundles right now "
                    f"(bundle '{name}' has type='{cfg['type']}')")
            if cfg['activation'] not in SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"FixedPointModel can't execute activation '{cfg['activation']}' "
                    f"(bundle '{name}') in integer arithmetic yet - only "
                    f"{sorted(SUPPORTED_ACTIVATIONS)} are piecewise-linear through the "
                    f"origin. See CLAUDE.md Known Issues.")

            self.bundles[name] = dict(
                input=cfg.get('input'),
                in_features=cfg['in_features'],
                out_features=cfg['out_features'],
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

    def load_int_weights(self, json_path):
        with open(json_path) as f:
            spec = json.load(f)['layers']

        for name in self.bundle_order:
            cfg = spec[name]
            bundle = self.bundles[name]
            bundle['weight'] = np.array(cfg['weight']['values'], dtype=np.int64)
            bundle['bias'] = np.array(cfg['bias']['values'], dtype=np.int64)

    def quantize_input(self, x_float):
        """Quantizes a real-valued input using the first bundle's input scale."""
        first = self.bundles[self.bundle_order[0]]
        x_int = np.rint(np.asarray(x_float, dtype=np.float64) * 2 ** first['input_frac'])
        return x_int.astype(np.int64)

    def forward(self, x_int):
        outputs = {}
        x_int = np.asarray(x_int, dtype=np.int64)

        for name in self.bundle_order:
            bundle = self.bundles[name]
            inp = outputs[bundle['input']] if bundle['input'] is not None else x_int

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

        return outputs[self.bundle_order[-1]]

    def print_graph(self):
        rows = []
        for name in self.bundle_order:
            b = self.bundles[name]
            source = b['input'] or 'x (model input)'
            rows.append((name, f"{b['in_features']}->{b['out_features']}", source,
                         b['activation'], f"{b['act_bits']}b/frac{b['act_frac']}"))

        header = ('bundle', 'shape', 'source', 'activation', 'output')
        widths = [max(len(r[c]) for r in rows + [header]) for c in range(5)]
        for row in [header] + rows:
            print('  '.join(cell.ljust(w) for cell, w in zip(row, widths)))
